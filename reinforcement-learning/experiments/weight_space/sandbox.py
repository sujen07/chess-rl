"""Weight-space programming sandbox, modelled on the Exp 6/8 harness in
"Sparks of Weight-Space Programming" (Takehi et al., MIT CSAIL, Nov 2026).

Rules for the editor (Claude):
  * Reading weights is free and unlimited (`load_weights`).
  * Running the network is metered: every `probe()` call is one forward pass
    over at most MAX_BATCH positions and is written to ledger.jsonl.
    The run stops at BUDGET passes.
  * No gradients, optimizers, training data loaders, MCTS, games or arena runs.
    The only way to execute the network is `probe()`.
  * Edits go through `save_weights()`, which logs what changed vs the original.

The held-out evaluation (eval.sh: MCTS arena edited vs original) is run by the
user after the edit is frozen; the editor never sees it while editing.
"""

import json
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
RL_DIR = HERE.parents[1]
sys.path.insert(0, str(RL_DIR))

from chess_environment import board_to_tensor, legal_move_mask, move_to_index  # noqa: E402
from model import PolicyNetwork  # noqa: E402

ORIGINAL = HERE / "original.pt"
EDITED = HERE / "edited.pt"
LEDGER = HERE / "ledger.jsonl"
EDITS = HERE / "edits.jsonl"
BUDGET = 50
MAX_BATCH = 512


def _checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def load_weights(which="edited"):
    """State dict as float32 numpy arrays (free)."""
    ckpt = _checkpoint(EDITED if which == "edited" else ORIGINAL)
    return {k: v.float().numpy().copy() for k, v in ckpt["policy_state_dict"].items()}


def save_weights(state, note):
    """Write the edited checkpoint and log the diff against the original."""
    ckpt = _checkpoint(ORIGINAL)
    original = ckpt["policy_state_dict"]
    new_state, changed, total, per_tensor = {}, 0, 0, {}
    for key, ref in original.items():
        arr = torch.from_numpy(np.asarray(state[key], dtype=np.float32)).to(ref.dtype)
        if arr.shape != ref.shape:
            raise ValueError(f"{key}: shape {tuple(arr.shape)} != {tuple(ref.shape)}")
        if not torch.isfinite(arr).all():
            raise ValueError(f"{key}: non-finite values")
        n = int((arr != ref).sum())
        if n:
            per_tensor[key] = n
        changed += n
        total += ref.numel()
        new_state[key] = arr
    ckpt["policy_state_dict"] = new_state
    torch.save(ckpt, EDITED)
    entry = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "note": note,
        "weights_changed": changed,
        "fraction": changed / total,
        "per_tensor": per_tensor,
    }
    with EDITS.open("a") as f:
        f.write(json.dumps(entry) + "\n")
    print(f"saved: {changed} weights changed ({100 * changed / total:.4f}%)")
    return entry


def passes_used():
    if not LEDGER.exists():
        return 0
    return sum(1 for _ in LEDGER.open())


def board_from(spec):
    """FEN string, chess.Board (history kept), or (fen, [uci moves])."""
    if isinstance(spec, chess.Board):
        return spec
    if isinstance(spec, tuple):
        board = chess.Board(spec[0])
        for uci in spec[1]:
            board.push_uci(uci)
        return board
    return chess.Board(spec)


_REPLAY = None


def practice_set():
    """Self-play replay buffer from the v7 run (data, free to read): packed
    planes plus 600/80-sim MCTS visit targets and value targets. Use it as
    practice problems; it may not be used for gradient training."""
    global _REPLAY
    if _REPLAY is None:
        _REPLAY = dict(np.load(RL_DIR / "runs" / "v7_se" / "replay_buffer.npz"))
    return _REPLAY


def decode_rows(idxs):
    """Input planes [n, 103, 8, 8] for replay rows (no network involved)."""
    from sl_dataset import decode_planes
    r = practice_set()
    return decode_planes(torch.from_numpy(r["pieces"][idxs]), torch.from_numpy(r["aux"][idxs]))


def _record(used, which, n, activations, purpose):
    with LEDGER.open("a") as f:
        f.write(json.dumps({
            "pass": used + 1, "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "which": which, "batch": n, "activations": list(activations),
            "purpose": purpose,
        }) + "\n")
    print(f"[probe {used + 1}/{BUDGET}] {which} x{n} {purpose}")


def _net(which):
    ckpt = _checkpoint(EDITED if which == "edited" else ORIGINAL)
    net = PolicyNetwork(**ckpt["model_config"])
    net.load_state_dict(ckpt["policy_state_dict"])
    return net.eval()


@torch.no_grad()
def probe_replay(idxs, which="edited", activations=(), purpose=""):
    """One metered forward pass over replay rows.

    Returns full logits [n, 4672], values [n], and requested activations.
    """
    used = passes_used()
    if used >= BUDGET:
        raise RuntimeError(f"forward-pass budget exhausted ({BUDGET})")
    idxs = np.asarray(idxs)
    if not 1 <= len(idxs) <= MAX_BATCH:
        raise ValueError(f"batch must be 1..{MAX_BATCH}")
    net = _net(which)
    captured, hooks = {}, []
    modules = dict(net.named_modules())
    for name in activations:
        hooks.append(modules[name].register_forward_hook(
            lambda _m, _i, out, name=name: captured.__setitem__(name, out.detach().numpy().copy())
        ))
    logits, values = net(decode_rows(idxs))
    for h in hooks:
        h.remove()
    _record(used, which, len(idxs), activations, purpose)
    return logits.numpy(), values.numpy(), captured


@torch.no_grad()
def probe(positions, which="edited", activations=(), top_k=5, purpose=""):
    """One metered forward pass.

    Returns a list of dicts (value, top-k legal moves with probabilities,
    the full legal-move distribution) plus {module_name: activation array}
    for any requested module names (e.g. "blocks.9", "value_fc.1").
    """
    used = passes_used()
    if used >= BUDGET:
        raise RuntimeError(f"forward-pass budget exhausted ({BUDGET})")
    boards = [board_from(p) for p in positions]
    if not 1 <= len(boards) <= MAX_BATCH:
        raise ValueError(f"batch must be 1..{MAX_BATCH}")

    net = _net(which)
    captured, hooks = {}, []
    modules = dict(net.named_modules())
    for name in activations:
        hooks.append(modules[name].register_forward_hook(
            lambda _m, _i, out, name=name: captured.__setitem__(name, out.detach().numpy().copy())
        ))
    x = torch.cat([board_to_tensor(b) for b in boards])
    logits, values = net(x)
    for h in hooks:
        h.remove()

    results = []
    for board, row, value in zip(boards, logits, values):
        mask = legal_move_mask(board)
        probs = torch.softmax(row.masked_fill(mask == 0, float("-inf")), dim=0)
        dist = sorted(
            ((m.uci(), float(probs[move_to_index(m, board)])) for m in board.legal_moves),
            key=lambda t: -t[1],
        )
        results.append({"fen": board.fen(), "value": float(value), "top": dist[:top_k], "dist": dist})

    _record(used, which, len(boards), activations, purpose)
    return results, captured
