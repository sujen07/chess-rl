"""UCI front end for the trained RL policy + fast_mcts search.

Lets any UCI host (lichess-bot, cutechess-cli, Arena, En Croissant, ...) play
the model. Search runs in a background thread so `stop` works, the tree is
reused between moves, and leaves are evaluated in small batches using
virtual loss to keep the GPU busy.

    python uci_engine.py [--model-path chess_rl_model_v5.pt]
"""
import argparse
import math
import sys
import threading
import time
from pathlib import Path

RL_DIR = Path(__file__).resolve().parent
if str(RL_DIR) not in sys.path:
    sys.path.insert(0, str(RL_DIR))

import chess  # noqa: E402
import torch  # noqa: E402

from fast_mcts import (  # noqa: E402
    LocalEvaluator,
    Node,
    advance,
    backpropagate,
    evaluate_leaves,
)
from chess_environment import ACTION_VERSION, ENCODING_VERSION, POLICY_SIZE  # noqa: E402
from model import MODEL_VERSION, PolicyNetwork  # noqa: E402

ENGINE_NAME = "MinimaxAI RL v5"
ENGINE_AUTHOR = "sujen07"

# name -> (type, default, min, max); "Move Overhead" is what lichess-bot sets.
OPTIONS = {
    "Move Overhead": ("spin", 100, 0, 5000),
    "Nodes": ("spin", 0, 0, 1_000_000),          # per-move sim cap, 0 = time only
    "DefaultSims": ("spin", 800, 1, 1_000_000),  # used for a bare `go`
    "Batch": ("spin", 8, 1, 64),
    "CPuct": ("string", "1.41", None, None),
    "FPU": ("string", "0.25", None, None),
    "Greedy": ("check", False, None, None),      # play the policy head's top move, no search
}


def load_policy(model_path, device):
    # Same checks as play.load_policy; importing play would start pygame.
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if (
        ckpt.get("model_version") != MODEL_VERSION
        or ckpt.get("encoding_version") != ENCODING_VERSION
        or ckpt.get("action_version") != ACTION_VERSION
        or ckpt.get("policy_size") != POLICY_SIZE
    ):
        raise ValueError(f"{model_path} is incompatible with this engine")
    policy = PolicyNetwork(**ckpt["model_config"])
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.to(device)
    policy.eval()
    return policy


def log(msg):
    print(f"info string {msg}", flush=True)


def q_to_cp(q):
    """Map a [-1, 1] value to centipawns (Leela's curve)."""
    q = max(-0.999, min(0.999, q))
    return int(round(111.714640912 * math.tan(1.5620688421 * q)))


def _undo_virtual_loss(path):
    for node, edge in path:
        node.visits -= 1
        node.child_n[edge] -= 1
        node.child_w[edge] -= 1.0


def gather_batch(root, size, c_puct, fpu_reduction):
    """Collect up to `size` distinct leaves using virtual loss.

    Terminal leaves are backed up immediately. Returns the unexpanded leaves
    (virtual loss already removed) ready for evaluate_leaves.
    """
    leaves, paths, seen = [], [], set()
    for _ in range(size):
        node, path = root, []
        while True:
            if node.terminal is not None:
                _undo_virtual_loss(path)
                backpropagate(node, node.terminal)
                break
            if not node.expanded:
                if id(node) in seen:  # collision: tree too narrow to batch more
                    _undo_virtual_loss(path)
                    for p in paths:
                        _undo_virtual_loss(p)
                    return leaves
                seen.add(id(node))
                leaves.append(node)
                paths.append(path)
                break
            edge = node.select_edge(c_puct, fpu_reduction)
            # Pretend this edge was just visited and lost for the mover.
            node.visits += 1
            node.child_n[edge] += 1
            node.child_w[edge] += 1.0
            path.append((node, edge))
            node = node.child(edge)
    for p in paths:
        _undo_virtual_loss(p)
    return leaves


def principal_variation(root, max_len=12):
    pv, node = [], root
    while node is not None and node.terminal is None and node.expanded and len(pv) < max_len:
        if node.child_n.max() <= 0:
            break
        edge = int(node.child_n.argmax())
        pv.append(node.moves[edge])
        node = node.children.get(edge)
    return pv


class Engine:
    def __init__(self, model_path):
        self.model_path = model_path
        self.options = {name: spec[1] for name, spec in OPTIONS.items()}
        self.evaluator = None
        self.board = chess.Board()
        self.root = None
        self.root_key = None  # (start fen, tuple of uci moves) that root represents
        self.position_key = (chess.STARTING_FEN, ())
        self.thread = None
        self.stop_event = threading.Event()
        self.out_lock = threading.Lock()

    # ---- I/O -------------------------------------------------------------
    def send(self, line):
        with self.out_lock:
            print(line, flush=True)

    def ensure_loaded(self):
        if self.evaluator is not None:
            return
        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        policy = load_policy(self.model_path, device)
        self.evaluator = LocalEvaluator(policy, device)
        # Warm up CUDA kernels so the first move doesn't eat the clock.
        root = Node.root(chess.Board())
        evaluate_leaves(self.evaluator, [root])
        log(f"loaded {Path(self.model_path).name} on {device}")

    # ---- commands --------------------------------------------------------
    def cmd_uci(self):
        self.send(f"id name {ENGINE_NAME}")
        self.send(f"id author {ENGINE_AUTHOR}")
        for name, (kind, default, lo, hi) in OPTIONS.items():
            if kind == "spin":
                self.send(f"option name {name} type spin default {default} min {lo} max {hi}")
            elif kind == "check":
                self.send(f"option name {name} type check default {str(default).lower()}")
            else:
                self.send(f"option name {name} type string default {default}")
        self.send("uciok")

    def cmd_setoption(self, args):
        text = " ".join(args)
        if not text.startswith("name "):
            return
        text = text[5:]
        name, _, value = text.partition(" value ")
        name = name.strip()
        for known, (kind, *_rest) in OPTIONS.items():
            if known.lower() == name.lower():
                if kind == "spin":
                    self.options[known] = int(value)
                elif kind == "check":
                    self.options[known] = value.strip().lower() == "true"
                else:
                    self.options[known] = value.strip()
                return
        log(f"ignoring unknown option {name}")

    def cmd_position(self, args):
        if not args:
            return
        if args[0] == "startpos":
            fen, rest = chess.STARTING_FEN, args[1:]
        elif args[0] == "fen":
            fen, rest = " ".join(args[1:7]), args[7:]
        else:
            return
        moves = rest[1:] if rest and rest[0] == "moves" else []
        board = chess.Board(fen)
        for uci in moves:
            board.push_uci(uci)
        self.board = board
        self.position_key = (fen, tuple(moves))

    def reuse_or_new_root(self):
        """Walk the previous tree forward to the current position if possible."""
        fen, moves = self.position_key
        root = None
        if self.root is not None and self.root_key is not None:
            old_fen, old_moves = self.root_key
            if old_fen == fen and moves[: len(old_moves)] == old_moves:
                root = self.root
                for uci in moves[len(old_moves):]:
                    root = advance(root, chess.Move.from_uci(uci))
                    if root is None:
                        break
        if root is None:
            root = Node.root(self.board)
        self.root, self.root_key = root, self.position_key
        return root

    def time_budget(self, params):
        """Seconds to spend on this move, or None for no time limit."""
        overhead = self.options["Move Overhead"] / 1000.0
        if "movetime" in params:
            return max(0.01, params["movetime"] / 1000.0 - overhead)
        side = "w" if self.board.turn == chess.WHITE else "b"
        if f"{side}time" not in params:
            return None
        remaining = params[f"{side}time"] / 1000.0
        inc = params.get(f"{side}inc", 0) / 1000.0
        moves_to_go = params.get("movestogo", 0) or max(15, 40 - self.board.fullmove_number // 2)
        budget = remaining / moves_to_go + 0.75 * inc
        budget = min(budget, 0.25 * remaining)  # never sink the clock on one move
        return max(0.01, budget - overhead)

    def cmd_go(self, args):
        params, infinite = {}, False
        keys = {"wtime", "btime", "winc", "binc", "movestogo", "movetime", "nodes", "depth"}
        i = 0
        while i < len(args):
            if args[i] == "infinite":
                infinite = True
            elif args[i] in keys and i + 1 < len(args):
                params[args[i]] = int(args[i + 1])
                i += 1
            i += 1

        budget = None if infinite else self.time_budget(params)
        sims = params.get("nodes") or self.options["Nodes"]
        if not infinite and budget is None and not sims:
            sims = self.options["DefaultSims"]

        self.cmd_stop()
        self.stop_event.clear()
        self.thread = threading.Thread(
            target=self.search, args=(budget, sims, infinite), daemon=True
        )
        self.thread.start()

    def cmd_stop(self):
        if self.thread is not None:
            self.stop_event.set()
            self.thread.join()
            self.thread = None

    # ---- search ----------------------------------------------------------
    def search(self, budget, sims, infinite):
        try:
            self._search(budget, sims, infinite)
        except Exception as exc:  # never leave the host waiting on bestmove
            log(f"search error: {exc!r}")
            legal = list(self.board.legal_moves)
            self.send(f"bestmove {legal[0].uci() if legal else '0000'}")

    def _search(self, budget, sims, infinite):
        self.ensure_loaded()
        start = time.monotonic()
        deadline = None if budget is None else start + budget
        legal = list(self.board.legal_moves)
        if not legal:
            self.send("bestmove 0000")
            return

        root = self.reuse_or_new_root()
        if not root.expanded:
            evaluate_leaves(self.evaluator, [root])
        if self.options["Greedy"]:
            edge = int(root.raw_priors.argmax())
            self.send(f"info depth 1 score cp {q_to_cp(root.value_mean())} nodes 1 pv {root.moves[edge].uci()}")
            self.send(f"bestmove {root.moves[edge].uci()}")
            return
        start_visits = root.visits
        target = start_visits + sims if sims else None
        c_puct = float(self.options["CPuct"])
        fpu = float(self.options["FPU"])
        batch = self.options["Batch"]
        last_info = start

        while not self.stop_event.is_set():
            now = time.monotonic()
            if len(legal) == 1 and not infinite:
                break
            if deadline is not None and now >= deadline:
                break
            if target is not None and root.visits >= target:
                break
            if not infinite and self._decided(root, now, start, start_visits, deadline, target):
                break
            leaves = gather_batch(root, batch, c_puct, fpu)
            evaluate_leaves(self.evaluator, leaves)
            if now - last_info >= 1.0:
                self.report(root, start, start_visits)
                last_info = now

        if infinite:  # UCI: don't send bestmove until told to stop
            self.stop_event.wait()
        self.report(root, start, start_visits)
        edge = int(root.child_n.argmax()) if root.child_n.max() > 0 else int(root.priors.argmax())
        self.send(f"bestmove {root.moves[edge].uci()}")

    @staticmethod
    def _decided(root, now, start, start_visits, deadline, target):
        """True once the runner-up can no longer catch the leader."""
        if len(root.child_n) < 2:
            return True
        elapsed = now - start
        done = root.visits - start_visits
        if done < 64 or elapsed <= 0:
            return False
        left = float("inf")
        if deadline is not None:
            left = min(left, (deadline - now) * done / elapsed)
        if target is not None:
            left = min(left, target - root.visits)
        if left == float("inf"):
            return False
        top2 = sorted(root.child_n)[-2:]
        return top2[1] - top2[0] > left

    def report(self, root, start, start_visits):
        elapsed = max(time.monotonic() - start, 1e-6)
        nodes = root.visits
        pv = principal_variation(root)
        if root.child_n.max() > 0:
            edge = int(root.child_n.argmax())
            q = -root.child_w[edge] / root.child_n[edge]
        else:
            q = root.value_mean()
        nps = int((root.visits - start_visits) / elapsed)
        self.send(
            f"info depth {max(1, len(pv))} score cp {q_to_cp(q)} nodes {nodes} "
            f"nps {nps} time {int(elapsed * 1000)} pv {' '.join(m.uci() for m in pv)}"
        )

    # ---- main loop -------------------------------------------------------
    def loop(self):
        for raw in sys.stdin:
            parts = raw.strip().split()
            if not parts:
                continue
            cmd, args = parts[0], parts[1:]
            if cmd == "uci":
                self.cmd_uci()
            elif cmd == "isready":
                self.ensure_loaded()
                self.send("readyok")
            elif cmd == "setoption":
                self.cmd_setoption(args)
            elif cmd == "ucinewgame":
                self.cmd_stop()
                self.root = self.root_key = None
            elif cmd == "position":
                self.cmd_position(args)
            elif cmd == "go":
                self.cmd_go(args)
            elif cmd == "stop":
                self.cmd_stop()
            elif cmd == "ponderhit":
                pass
            elif cmd == "quit":
                break
        self.cmd_stop()


def main():
    parser = argparse.ArgumentParser(description="UCI engine for the RL chess policy.")
    parser.add_argument("--model-path", default=str(RL_DIR / "chess_rl_model_v5.pt"))
    args = parser.parse_args()
    Engine(args.model_path).loop()


if __name__ == "__main__":
    main()
