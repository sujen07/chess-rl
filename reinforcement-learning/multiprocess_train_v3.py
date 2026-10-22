"""Multiprocess self-play RL, v3: fast search and sample-efficient targets.

Changes from v2:
  * fast_mcts: bitboard encoding, lazy children, tree reuse between moves.
  * Each worker keeps `games_per_worker` games in flight and refills a slot
    as soon as its game ends, so GPU batches stay large.
  * Playout cap randomization (KataGo): a `full_search_prob` fraction of
    moves get `full_simulations` with root noise and become training
    samples; the rest get `fast_simulations` and are not recorded.
  * Resignation with a no-resign playthrough fraction that measures the
    false-positive rate.
  * Value target = (1 - q_weight) * game result + q_weight * root Q.
  * Each SGD batch mixes in `sl_mix_fraction` supervised Lichess rows to
    anchor the policy against drift.
  * Packed replay ring buffer decoded on the GPU; training is driven by a
    sample-reuse ratio instead of a fixed number of steps per round.
"""

import argparse
import json
import multiprocessing as mp
import queue
import threading
import time
import traceback
from collections import Counter, deque
from pathlib import Path

import chess
import numpy as np
import torch

from chess_environment import POLICY_SIZE, outcome_reward_for
from fast_mcts import (
    Node,
    add_dirichlet_noise,
    advance,
    evaluate_leaves,
    evaluate_packed,
    gather_leaf,
    is_game_over,
    policy_target,
    select_edge,
    training_repetition_exclusions,
)
from model import PolicyNetwork
from multiprocess_train_v2 import load_midgame_fens
from packed_replay import PackedReplay, soft_policy_loss
from train import load_checkpoint, save_checkpoint
from training_config import DEFAULT_CONFIG_PATH, load_training_config
from training_runtime import (
    amp_context,
    apply_lr_schedule,
    make_grad_scaler,
    seed_everything,
)

RL_DIR = Path(__file__).resolve().parent


# --------------------------------------------------------------------------
# Self-play worker
# --------------------------------------------------------------------------


class RemoteEvaluator:
    """Worker-side proxy: ships packed leaves to the coordinator's GPU."""

    def __init__(self, worker_id, request_queue, response_queue):
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue
        self.positions = 0

    def __call__(self, pieces, aux, legal_idx, legal_off):
        self.positions += len(pieces)
        self.request_queue.put((self.worker_id, pieces, aux, legal_idx, legal_off))
        response = self.response_queue.get()
        if isinstance(response, BaseException):
            raise response
        return response


class SelfPlayGame:
    def __init__(self, board, allow_resign):
        self.board = board
        self.allow_resign = allow_resign
        self.root = None
        self.plies = 0
        self.samples = []  # (pieces, aux, legal, pi_idx, pi_prob, q, mover)
        self.low_streak = {chess.WHITE: 0, chess.BLACK: 0}
        self.would_resign = None
        self.full = False
        self.target = 0
        self.noise_pending = False
        self.full_moves = 0
        self.search_visits = 0
        self.result = None
        self.termination = None


def self_play_loop(evaluator, settings, claim_game, emit_game, stop_event, rng):
    """Play games with a fixed number of slots until claim_game() runs dry."""
    s = settings
    midgame_fens = s["midgame_fens"]

    def new_game():
        if not claim_game():
            return None
        if midgame_fens and rng.random() < s["midgame_prob"]:
            board = chess.Board(midgame_fens[rng.integers(len(midgame_fens))])
        else:
            board = chess.Board()
        game = SelfPlayGame(board, allow_resign=rng.random() >= s["resign_playthrough"])
        prepare_search(game)
        return game

    def prepare_search(game):
        if game.root is None:
            game.root = Node.root(game.board)
        game.full = rng.random() < s["full_search_prob"]
        sims = s["full_simulations"] if game.full else s["fast_simulations"]
        game.target = sims + 1
        game.search_visits -= game.root.visits
        game.noise_pending = game.full and s["dirichlet_eps"] > 0
        apply_noise(game)

    def apply_noise(game):
        if game.noise_pending and game.root.expanded:
            add_dirichlet_noise(
                game.root, s["dirichlet_alpha"], s["dirichlet_eps"], rng
            )
            game.noise_pending = False

    def finish(game, result, termination):
        game.result, game.termination = result, termination
        game.root = None
        emit_game(game)

    def play_move(game):
        """Act on a finished search; True when the game is over."""
        root = game.root
        board = game.board
        mover = board.turn
        q = root.value_mean()
        game.search_visits += root.visits

        if s["resign_threshold"] > -1 and game.plies >= s["resign_min_ply"]:
            game.low_streak[mover] = (
                game.low_streak[mover] + 1 if q < s["resign_threshold"] else 0
            )
            if game.low_streak[mover] >= s["resign_consecutive"]:
                if game.allow_resign:
                    finish(game, "0-1" if mover == chess.WHITE else "1-0", "RESIGN")
                    return True
                if game.would_resign is None:
                    game.would_resign = mover

        excluded = (
            training_repetition_exclusions(root, s["repetition_value_threshold"])
            if s["avoid_training_repetitions"] else set()
        )
        if game.full:
            pi_idx, pi_prob = policy_target(root, excluded)
            game.samples.append(
                (root.pieces, root.aux, root.indices.astype(np.int16),
                 pi_idx, pi_prob, q, mover)
            )
            game.full_moves += 1

        temperature = 1.0 if game.plies < s["temp_threshold"] else 0.0
        move = root.moves[select_edge(root, temperature, excluded, rng)]
        board.push(move)
        game.plies += 1
        game.root = advance(root, move) if s["tree_reuse"] else None

        if is_game_over(board):
            finish(
                game,
                board.result(claim_draw=True),
                board.outcome(claim_draw=True).termination.name,
            )
            return True
        if game.plies >= s["max_moves"]:
            finish(game, "1/2-1/2", "MOVE_CAP")
            return True
        prepare_search(game)
        return False

    slots = [new_game() for _ in range(s["games_per_worker"])]
    while any(slots):
        if stop_event.is_set():
            return
        leaves = []
        for game in slots:
            # A reused subtree can already meet the target; just play it.
            if game is None or game.root.visits >= game.target:
                continue
            leaf = gather_leaf(
                game.root, s["c_puct"], s["fpu_reduction"],
                0.0 if game.full else None,
            )
            if leaf is not None:
                leaves.append(leaf)
        evaluate_leaves(evaluator, leaves)
        for i, game in enumerate(slots):
            if game is None:
                continue
            apply_noise(game)
            if game.root.visits >= game.target and play_move(game):
                slots[i] = new_game()


def pack_game(game, q_weight):
    """Training arrays + stats for one finished game (numpy only, picklable)."""
    capped = game.termination == "MOVE_CAP"
    values, weights = [], []
    for *_, q, mover in game.samples:
        z = outcome_reward_for(game.result, mover)
        if capped:
            # No ruled result: trust the search value, or skip if it is unused.
            values.append(q)
            weights.append(1.0 if q_weight > 0 else 0.0)
        else:
            values.append((1.0 - q_weight) * z + q_weight * q)
            weights.append(1.0)
    n = len(game.samples)
    false_resign = None
    if game.would_resign is not None:
        loser = {"1-0": chess.BLACK, "0-1": chess.WHITE}.get(game.result)
        false_resign = loser != game.would_resign
    return {
        "pieces": np.stack([x[0] for x in game.samples]) if n else np.zeros((0, 8, 32), np.uint8),
        "aux": np.stack([x[1] for x in game.samples]) if n else np.zeros((0, 4), np.uint8),
        "legal": [x[2] for x in game.samples],
        "pi_idx": [x[3] for x in game.samples],
        "pi_prob": [x[4] for x in game.samples],
        "value": np.array(values, dtype=np.float32),
        "weight": np.array(weights, dtype=np.float32),
        "result": game.result,
        "termination": game.termination,
        "plies": game.plies,
        "full_moves": game.full_moves,
        "search_visits": game.search_visits,
        "playthrough": not game.allow_resign,
        "false_resign": false_resign,
    }


def self_play_worker(
    worker_id, request_queue, response_queue, results_queue, error_queue,
    stop_event, games_claimed, claim_lock, episodes, settings, seed,
):
    torch.set_num_threads(1)
    rng = np.random.default_rng(seed + worker_id)
    settings = dict(settings)
    settings["midgame_fens"] = (
        load_midgame_fens(settings["midgame_fens"]) if settings["midgame_fens"] else []
    )
    evaluator = RemoteEvaluator(worker_id, request_queue, response_queue)

    def claim_game():
        with claim_lock:
            if games_claimed.value >= episodes:
                return False
            games_claimed.value += 1
            return True

    def emit_game(game):
        results_queue.put(pack_game(game, settings["q_weight"]))

    try:
        self_play_loop(evaluator, settings, claim_game, emit_game, stop_event, rng)
    except (KeyboardInterrupt, BrokenPipeError, EOFError):
        stop_event.set()
    except Exception:
        details = traceback.format_exc()
        error_queue.put(f"Self-play worker {worker_id} failed:\n{details}")
        print(details)
        stop_event.set()


# --------------------------------------------------------------------------
# Coordinator
# --------------------------------------------------------------------------


def inference_loop(
    policy, device, amp_enabled, model_lock, request_queue, response_queues,
    num_workers, stop_event, thread_stop, errors, stats, max_wait,
):
    """Merge pending worker requests into one GPU batch."""
    while not thread_stop.is_set():
        try:
            requests = [request_queue.get(timeout=0.1)]
        except queue.Empty:
            continue
        except (EOFError, OSError):
            break
        deadline = time.monotonic() + max_wait
        # Workers block on their reply, so at most one request each is pending.
        while len(requests) < num_workers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                requests.append(request_queue.get(timeout=remaining))
            except queue.Empty:
                break
            except (EOFError, OSError):
                break

        try:
            sizes = [len(r[1]) for r in requests]
            legal_sizes = [len(r[3]) for r in requests]
            legal_off = [np.zeros(1, dtype=np.int64)]
            base = 0
            for r in requests:
                legal_off.append(r[4][1:] + base)
                base += int(r[4][-1])
            with model_lock:
                legal_logits, values = evaluate_packed(
                    policy, device, amp_enabled,
                    np.concatenate([r[1] for r in requests]),
                    np.concatenate([r[2] for r in requests]),
                    np.concatenate([r[3] for r in requests]),
                    np.concatenate(legal_off),
                )
            stats["batches"] += 1
            stats["positions"] += sum(sizes)
            logit_parts = np.split(legal_logits, np.cumsum(legal_sizes)[:-1])
            value_parts = np.split(values, np.cumsum(sizes)[:-1])
            for r, logits_part, values_part in zip(requests, logit_parts, value_parts):
                response_queues[r[0]].put((logits_part, values_part))
        except Exception as exc:
            errors.append(exc)
            stop_event.set()
            for r in requests:
                response_queues[r[0]].put(RuntimeError(f"Coordinator inference failed: {exc}"))
            break


def train_step(policy, optimizer, scaler, replay, sl_data, rl_rows, sl_rows,
               value_coef, device, amp_enabled, rng):
    x, legal_mask, target, value, weight = replay.batch(replay.sample(rl_rows, rng), device)
    n_rl = len(value)
    if sl_data is not None and sl_rows > 0:
        sx, s_mask, s_move, s_value = sl_data.batch(
            rng.integers(0, len(sl_data), size=sl_rows), device
        )
        s_target = torch.zeros(sl_rows, POLICY_SIZE, device=device)
        s_target[torch.arange(sl_rows, device=device), s_move] = 1.0
        x = torch.cat([x, sx])
        legal_mask = torch.cat([legal_mask, s_mask])
        target = torch.cat([target, s_target])
        value = torch.cat([value, s_value])
        weight = torch.cat([weight, torch.ones(sl_rows, device=device)])

    policy.train()
    optimizer.zero_grad(set_to_none=True)
    with amp_context(device, amp_enabled):
        logits, predicted = policy(x)
    policy_rows = soft_policy_loss(logits, legal_mask, target)
    policy_loss = policy_rows.mean()
    sq = (predicted.float().reshape(-1) - value).square() * weight
    value_loss = sq.sum() / weight.sum().clamp_min(1.0)
    total = policy_loss + value_coef * value_loss
    scaler.scale(total).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    policy.eval()
    # Entropy of the RL targets: the floor the RL policy loss can reach.
    rl_target = target[:n_rl]
    target_entropy = -(rl_target * rl_target.clamp_min(1e-12).log()).sum(1).mean()
    return {
        "total": total.item(),
        "policy_rl": policy_rows[:n_rl].mean().item(),
        "policy_sl": policy_rows[n_rl:].mean().item() if len(policy_rows) > n_rl else float("nan"),
        "target_entropy": target_entropy.item(),
        "value_rl": (sq[:n_rl].sum() / weight[:n_rl].sum().clamp_min(1.0)).item(),
        "grad_norm": float(grad_norm),
    }


def save_state(ckpt_dir, policy, optimizer, scaler, games_done, results_window,
               replay_path, resolved_config, model_out, numbered, keep):
    paths = [ckpt_dir / "latest.pt"]
    if numbered:
        paths.insert(0, ckpt_dir / f"ckpt_{games_done:07d}.pt")
    for path in paths:
        save_checkpoint(
            path, policy, optimizer, games_done, results_window,
            scaler=scaler, replay_path=replay_path, resolved_config=resolved_config,
        )
    for stale in sorted(ckpt_dir.glob("ckpt_*.pt"))[:-keep]:
        stale.unlink()
    if model_out:
        Path(model_out).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {**policy.checkpoint_metadata(), "policy_state_dict": policy.state_dict()},
            model_out,
        )


def train(args, policy, optimizer, scaler, device, start_games, results_window):
    amp_enabled = not args.disable_amp and device.type == "cuda"
    policy.to(device).eval()
    rng = np.random.default_rng(args.seed + start_games)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    replay_path = Path(args.replay_path or ckpt_dir / "replay_buffer.npz")
    (ckpt_dir / "resolved_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )

    replay = PackedReplay(args.buffer_capacity)
    if start_games and replay.load(replay_path):
        print(f"Restored replay buffer with {len(replay):,} samples")

    sl_data = None
    sl_rows = int(round(args.train_batch_size * args.sl_mix_fraction))
    rl_rows = args.train_batch_size - sl_rows
    if sl_rows > 0:
        from sl_dataset import PackedDataset

        print(f"Loading SL mix data from {args.sl_data_dir} ...")
        sl_data = PackedDataset.load_dir(args.sl_data_dir, args.sl_mix_positions)
        print(f"SL mix: {len(sl_data):,} positions, {sl_rows}/{args.train_batch_size} rows per batch")

    settings = {
        key: getattr(args, key)
        for key in (
            "games_per_worker", "full_simulations", "fast_simulations",
            "full_search_prob", "c_puct", "fpu_reduction", "temp_threshold",
            "dirichlet_alpha", "dirichlet_eps", "max_moves", "tree_reuse",
            "resign_threshold", "resign_consecutive", "resign_min_ply",
            "resign_playthrough", "avoid_training_repetitions",
            "repetition_value_threshold", "q_weight", "midgame_fens",
            "midgame_prob",
        )
    }

    request_queue = mp.Queue()
    results_queue = mp.Queue()
    error_queue = mp.Queue()
    response_queues = [mp.Queue() for _ in range(args.num_workers)]
    stop_event = mp.Event()
    games_claimed = mp.Value("q", start_games)
    claim_lock = mp.Lock()
    model_lock = threading.Lock()
    thread_stop = threading.Event()
    inference_errors = []
    stats = {"batches": 0, "positions": 0}
    workers = []

    results_window = deque(results_window or [], maxlen=200)
    terminations = deque(maxlen=200)
    plies_window = deque(maxlen=200)
    resign_checks = deque(maxlen=500)  # playthrough games that would have resigned
    games_done = start_games
    samples_new = 0
    train_credit = 0.0
    steps = 0
    metrics = None
    run_start = last_log = time.time()
    last_ckpt_games = last_replay_games = start_games
    visits_total = 0
    plies_total = 0

    try:
        for worker_id in range(args.num_workers):
            worker = mp.Process(
                target=self_play_worker,
                args=(
                    worker_id, request_queue, response_queues[worker_id],
                    results_queue, error_queue, stop_event, games_claimed,
                    claim_lock, args.episodes, settings,
                    args.seed + 10_000 + start_games,
                ),
                name=f"self-play-{worker_id}",
            )
            worker.start()
            workers.append(worker)
        inference_thread = threading.Thread(
            target=inference_loop,
            args=(
                policy, device, amp_enabled, model_lock, request_queue,
                response_queues, args.num_workers, stop_event, thread_stop,
                inference_errors, stats, args.inference_max_wait_ms / 1000.0,
            ),
            name="policy-inference",
            daemon=True,
        )
        inference_thread.start()

        while games_done < args.episodes:
            try:
                game = results_queue.get(timeout=0.25)
            except queue.Empty:
                if inference_errors:
                    raise RuntimeError("Policy inference failed") from inference_errors[0]
                try:
                    raise RuntimeError(error_queue.get_nowait())
                except queue.Empty:
                    pass
                if not any(w.is_alive() for w in workers):
                    raise RuntimeError("All self-play workers exited early")
                continue

            replay.add(game)
            n_new = len(game["value"])
            samples_new += n_new
            games_done += 1
            results_window.append(game["result"])
            terminations.append(game["termination"])
            plies_window.append(game["plies"])
            visits_total += game["search_visits"]
            plies_total += game["plies"]
            if game["false_resign"] is not None:
                resign_checks.append(game["false_resign"])

            lr = apply_lr_schedule(
                optimizer, args.lr, args.min_lr, args.warmup_games, args.episodes, games_done
            )
            if len(replay) >= args.min_buffer_size:
                train_credit += n_new * args.sample_reuse
                # Bounded per game so inference never starves behind training.
                for _ in range(args.max_steps_per_game):
                    if train_credit < rl_rows:
                        break
                    with model_lock:
                        metrics = train_step(
                            policy, optimizer, scaler, replay, sl_data, rl_rows,
                            sl_rows, args.value_coef, device, amp_enabled, rng,
                        )
                    train_credit -= rl_rows
                    steps += 1

            now = time.time()
            if now - last_log >= args.log_every_seconds:
                last_log = now
                elapsed_h = max(now - run_start, 1e-9) / 3600
                n = max(len(results_window), 1)
                w, l = results_window.count("1-0"), results_window.count("0-1")
                d = n - w - l
                term_text = ", ".join(
                    f"{k}={v}" for k, v in Counter(terminations).most_common()
                )
                loss_text = (
                    f"p_rl={metrics['policy_rl']:.3f} (target H={metrics['target_entropy']:.3f}) "
                    f"p_sl={metrics['policy_sl']:.3f} v_rl={metrics['value_rl']:.3f} "
                    f"grad={metrics['grad_norm']:.2f}"
                    if metrics else "n/a (filling buffer)"
                )
                fp_text = (
                    f"{100 * sum(resign_checks) / len(resign_checks):.0f}% of {len(resign_checks)}"
                    if resign_checks else "n/a"
                )
                print(
                    f"[{time.strftime('%H:%M:%S')} | {elapsed_h * 60:.0f}m] "
                    f"games {games_done}/{args.episodes} "
                    f"({(games_done - start_games) / elapsed_h:.0f}/h) | "
                    f"samples {samples_new / elapsed_h:.0f}/h buffer {len(replay):,} | "
                    f"steps {steps} | lr {lr:.2e} | {loss_text} | "
                    f"visits/move {visits_total / max(plies_total, 1):.0f} | "
                    f"evals/s {stats['positions'] / (elapsed_h * 3600):.0f} "
                    f"infer_batch {stats['positions'] / max(stats['batches'], 1):.0f} | "
                    f"avg_plies {np.mean(plies_window):.0f} | "
                    f"W{w}/D{d}/L{l} | resign false+ {fp_text} | {term_text}",
                    flush=True,
                )

            if games_done - last_ckpt_games >= args.checkpoint_every:
                last_ckpt_games = games_done
                with model_lock:
                    save_state(
                        ckpt_dir, policy, optimizer, scaler, games_done,
                        results_window, replay_path, vars(args), args.model_out,
                        numbered=True, keep=args.checkpoint_keep,
                    )
                print(f"  -> checkpoint at {games_done} games", flush=True)
            if games_done - last_replay_games >= args.replay_save_every:
                last_replay_games = games_done
                replay.save(replay_path)

    except KeyboardInterrupt:
        print("Stopping training...")
    finally:
        stop_event.set()
        deadline = time.monotonic() + 10.0
        for worker in workers:
            worker.join(timeout=max(0.0, deadline - time.monotonic()))
        for worker in workers:
            if worker.is_alive():
                worker.terminate()
            worker.join(timeout=1.0)
        thread_stop.set()
        replay.save(replay_path)
        with model_lock:
            save_state(
                ckpt_dir, policy, optimizer, scaler, games_done, results_window,
                replay_path, vars(args), args.model_out, numbered=False,
                keep=args.checkpoint_keep,
            )
    return policy


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))
    config_args, _ = config_parser.parse_known_args()

    p = argparse.ArgumentParser(
        description="Fast multiprocess self-play RL (v3).", parents=[config_parser]
    )
    add = p.add_argument
    # Model / optimizer.
    add("--channels", type=int, default=128)
    add("--num-blocks", type=int, default=10)
    add("--value-head-channels", type=int, default=32)
    add("--value-hidden", type=int, default=256)
    add("--lr", type=float, default=1e-4)
    add("--min-lr", type=float, default=2e-5)
    add("--warmup-games", type=int, default=1000)
    add("--weight-decay", type=float, default=1e-4)
    # Self-play.
    add("--episodes", type=int, default=100_000)
    add("--num-workers", type=int, default=mp.cpu_count())
    add("--games-per-worker", type=int, default=48)
    add("--full-simulations", type=int, default=400)
    add("--fast-simulations", type=int, default=80)
    add("--full-search-prob", type=float, default=0.25)
    add("--c-puct", type=float, default=1.41)
    add("--fpu-reduction", type=float, default=0.25)
    add("--temp-threshold", type=int, default=30)
    add("--dirichlet-alpha", type=float, default=0.3)
    add("--dirichlet-eps", type=float, default=0.25)
    add("--max-moves", type=int, default=240)
    add("--tree-reuse", action=argparse.BooleanOptionalAction, default=True)
    add("--resign-threshold", type=float, default=-0.9, help="<= -1 disables")
    add("--resign-consecutive", type=int, default=3)
    add("--resign-min-ply", type=int, default=20)
    add("--resign-playthrough", type=float, default=0.1)
    add("--avoid-training-repetitions", action=argparse.BooleanOptionalAction, default=True)
    add("--repetition-value-threshold", type=float, default=-0.25)
    add("--midgame-fens", type=str, default=None)
    add("--midgame-prob", type=float, default=0.0)
    add("--inference-max-wait-ms", type=float, default=2.0)
    # Training.
    add("--q-weight", type=float, default=0.5)
    add("--value-coef", type=float, default=1.0)
    add("--train-batch-size", type=int, default=1024)
    add("--sample-reuse", type=float, default=4.0,
        help="expected SGD uses of each RL sample")
    add("--max-steps-per-game", type=int, default=4)
    add("--buffer-capacity", type=int, default=300_000)
    add("--min-buffer-size", type=int, default=20_000)
    add("--sl-mix-fraction", type=float, default=0.3)
    add("--sl-data-dir", type=str, default=str(RL_DIR / "data" / "lichess_sl"))
    add("--sl-mix-positions", type=int, default=3_000_000)
    # Bookkeeping.
    add("--checkpoint-dir", type=str, default=str(RL_DIR / "checkpoints_v5"))
    add("--checkpoint-every", type=int, default=500, help="games")
    add("--checkpoint-keep", type=int, default=20)
    add("--replay-save-every", type=int, default=2000, help="games")
    add("--replay-path", type=str, default=None)
    add("--resume", type=str, default=None)
    add("--init-model", type=str, default=None)
    add("--model-out", type=str, default=str(RL_DIR / "chess_rl_model_v5.pt"))
    add("--log-every-seconds", type=float, default=60.0)
    add("--seed", type=int, default=0)
    add("--disable-amp", action="store_true")

    try:
        defaults = load_training_config(config_args.config, "selfplay_v3", include_common=False)
    except (FileNotFoundError, OSError, ValueError) as exc:
        p.error(str(exc))
    valid = {a.dest for a in p._actions}
    unknown = sorted(set(defaults) - valid)
    if unknown:
        p.error(f"unknown selfplay_v3 config keys: {', '.join(unknown)}")
    p.set_defaults(**defaults)
    args = p.parse_args()

    if args.num_workers < 1 or args.games_per_worker < 1:
        p.error("--num-workers and --games-per-worker must be at least 1")
    if min(args.full_simulations, args.fast_simulations) < 1:
        p.error("simulation counts must be at least 1")
    for name in ("full_search_prob", "resign_playthrough", "sl_mix_fraction",
                 "q_weight", "midgame_prob", "dirichlet_eps"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            p.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.sl_mix_fraction >= 1.0:
        p.error("--sl-mix-fraction must leave room for RL rows")
    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PolicyNetwork(
        channels=args.channels, num_blocks=args.num_blocks,
        value_head_channels=args.value_head_channels, value_hidden=args.value_hidden,
    )
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = make_grad_scaler(not args.disable_amp and device.type == "cuda")

    start_games, results_window = 0, None
    if args.resume and Path(args.resume).is_file():
        meta = load_checkpoint(args.resume, policy, optimizer, device=device, scaler=scaler)
        start_games = meta["games_done"]
        results_window = meta["results_window"]
        if meta.get("replay_path"):
            args.replay_path = meta["replay_path"]
        print(f"Resumed from {args.resume} at {start_games} games")
    elif args.init_model:
        load_checkpoint(args.init_model, policy, device=device)
        print(f"Warm start from {args.init_model}")
    else:
        print("Warning: training from random weights")

    train(args, policy, optimizer, scaler, device, start_games, results_window)
    print(f"Final model weights saved to {args.model_out}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
