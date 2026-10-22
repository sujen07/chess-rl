import argparse
import multiprocessing as mp
import queue
import threading
import time
import traceback
import json
from collections import Counter, deque
from pathlib import Path

import chess
import torch

from chess_environment import PositionRecord, outcome_reward_for
from mcts import (
    mcts_search_batch,
    policy_target,
    select_move,
    training_repetition_exclusions,
)
from model import PolicyNetwork
from replay_buffer import ReplayBuffer
from train import load_checkpoint, save_checkpoint
from training_config import DEFAULT_CONFIG_PATH, load_training_config
from training_runtime import (
    amp_context,
    apply_lr_schedule,
    load_replay_buffer,
    make_grad_scaler,
    save_replay_buffer,
    seed_everything,
    train_batch,
)


RL_DIR = Path(__file__).resolve().parent


class RemoteModel:
    """CPU-worker proxy for the coordinator-owned policy network."""

    def __init__(self, worker_id, request_queue, response_queue):
        self.worker_id = worker_id
        self.request_queue = request_queue
        self.response_queue = response_queue

    def __call__(self, x):
        self.request_queue.put((self.worker_id, x.detach().cpu()))
        response = self.response_queue.get()
        if isinstance(response, BaseException):
            raise response
        return response


def _self_play_round(
    remote_model,
    stop_event,
    games_per_batch,
    num_simulations,
    c_puct,
    temp_threshold,
    dirichlet_alpha,
    dirichlet_eps,
    max_moves,
    avoid_training_repetitions,
    repetition_value_threshold,
):
    """Play one root-parallel round, returning compact, flattened samples."""
    boards = [chess.Board() for _ in range(games_per_batch)]
    histories = [[] for _ in range(games_per_batch)]
    plies = [0] * games_per_batch
    done = [False] * games_per_batch
    results = [None] * games_per_batch
    terminations = [None] * games_per_batch

    while not all(done):
        if stop_event.is_set():
            return None

        active = [i for i in range(games_per_batch) if not done[i]]
        roots = mcts_search_batch(
            remote_model,
            [boards[i] for i in active],
            num_simulations=num_simulations,
            c=c_puct,
            device=None,
            add_dirichlet_noise=True,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps,
        )

        for root, i in zip(roots, active):
            temperature = 1.0 if plies[i] < temp_threshold else 0.0
            excluded_moves = (
                training_repetition_exclusions(
                    root, value_threshold=repetition_value_threshold
                )
                if avoid_training_repetitions else set()
            )
            pi = policy_target(
                root, temperature=1.0, excluded_moves=excluded_moves
            )
            move = select_move(
                root, temperature=temperature, excluded_moves=excluded_moves
            )

            histories[i].append(
                (PositionRecord.from_board(boards[i]), pi, boards[i].turn)
            )
            boards[i].push(move)
            plies[i] += 1

            game_over = boards[i].is_game_over(claim_draw=True)
            if game_over or plies[i] >= max_moves:
                done[i] = True
                if game_over:
                    results[i] = boards[i].result(claim_draw=True)
                    terminations[i] = boards[i].outcome(
                        claim_draw=True
                    ).termination.name
                else:
                    results[i] = "1/2-1/2"
                    terminations[i] = "MOVE_CAP"

    positions = []
    policies = []
    outcomes = []
    value_weights = []
    for i, history in enumerate(histories):
        result = results[i] or "1/2-1/2"
        value_weight = 0.0 if terminations[i] == "MOVE_CAP" else 1.0
        for position, pi, mover in history:
            positions.append(position)
            policies.append(pi)
            outcomes.append(outcome_reward_for(result, mover))
            value_weights.append(value_weight)

    return (
        positions,
        policies,
        outcomes,
        value_weights,
        results,
        terminations,
        plies,
    )


def self_play_worker(
    worker_id,
    request_queue,
    response_queue,
    results_queue,
    stop_event,
    games_per_batch,
    num_simulations,
    c_puct,
    temp_threshold,
    dirichlet_alpha,
    dirichlet_eps,
    max_moves,
    avoid_training_repetitions,
    repetition_value_threshold,
    seed,
    games_claimed,
    claim_lock,
    episodes,
    error_queue,
):
    """Run CPU-only self-play rounds until the coordinator asks us to stop."""
    torch.set_num_threads(1)
    seed_everything(seed + worker_id)
    remote_model = RemoteModel(worker_id, request_queue, response_queue)

    try:
        while not stop_event.is_set():
            with claim_lock:
                if games_claimed.value >= episodes:
                    break
                round_games = min(
                    games_per_batch, episodes - games_claimed.value
                )
                games_claimed.value += round_games
            result = _self_play_round(
                remote_model=remote_model,
                stop_event=stop_event,
                games_per_batch=round_games,
                num_simulations=num_simulations,
                c_puct=c_puct,
                temp_threshold=temp_threshold,
                dirichlet_alpha=dirichlet_alpha,
                dirichlet_eps=dirichlet_eps,
                max_moves=max_moves,
                avoid_training_repetitions=avoid_training_repetitions,
                repetition_value_threshold=repetition_value_threshold,
            )
            if result is None:
                break
            results_queue.put(result)
    except (KeyboardInterrupt, BrokenPipeError, EOFError):
        error_queue.put(f"Self-play worker {worker_id} was interrupted")
        stop_event.set()
    except Exception:
        details = traceback.format_exc()
        error_queue.put(f"Self-play worker {worker_id} failed:\n{details}")
        print(details)
        stop_event.set()


def _inference_loop(
    policy,
    device,
    model_lock,
    request_queue,
    response_queues,
    stop_event,
    thread_stop_event,
    errors,
    amp_enabled,
    inference_stats,
):
    """Batch pending worker requests and evaluate them on the coordinator."""
    failure = None

    while not thread_stop_event.is_set():
        try:
            first_request = request_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        except (EOFError, OSError):
            break

        requests = [first_request]
        time.sleep(0.001)
        while True:
            try:
                requests.append(request_queue.get_nowait())
            except queue.Empty:
                break
            except (EOFError, OSError):
                break

        if failure is not None:
            for worker_id, _ in requests:
                response_queues[worker_id].put(RuntimeError(failure))
            continue

        try:
            sizes = [x.shape[0] for _, x in requests]
            x_batch = torch.cat([x for _, x in requests], dim=0)
            with model_lock:
                with torch.no_grad():
                    with amp_context(device, amp_enabled):
                        logits, values = policy(x_batch.to(device))
                    logits = logits.detach().cpu()
                    values = values.detach().cpu()
            inference_stats["batches"] += 1
            inference_stats["positions"] += x_batch.shape[0]

            logits_parts = logits.split(sizes, dim=0)
            value_parts = values.split(sizes, dim=0)
            for (worker_id, _), worker_logits, worker_values in zip(
                requests, logits_parts, value_parts
            ):
                response_queues[worker_id].put((worker_logits, worker_values))
        except Exception as exc:
            failure = f"Coordinator inference failed: {exc}"
            errors.append(exc)
            stop_event.set()
            for worker_id, _ in requests:
                response_queues[worker_id].put(RuntimeError(failure))


def _save_training_state(
    model_lock,
    checkpoint_dir,
    policy,
    optimizer,
    games_done,
    results_window,
    scaler,
    replay_path,
    resolved_config,
    model_out=None,
):
    with model_lock:
        save_checkpoint(
            Path(checkpoint_dir) / "latest.pt",
            policy,
            optimizer,
            games_done,
            results_window,
            scaler=scaler,
            replay_path=replay_path,
            resolved_config=resolved_config,
        )
        if model_out is not None:
            model_path = Path(model_out)
            model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    **policy.checkpoint_metadata(),
                    "policy_state_dict": policy.state_dict(),
                },
                model_path,
            )


def train_multiprocess(
    policy,
    optimizer,
    episodes,
    device,
    num_workers,
    games_per_batch=4,
    num_simulations=100,
    c_puct=1.41,
    temp_threshold=15,
    dirichlet_alpha=0.3,
    dirichlet_eps=0.25,
    value_coef=1.0,
    epochs_per_batch=2,
    max_moves=200,
    avoid_training_repetitions=True,
    repetition_value_threshold=-0.25,
    checkpoint_dir="checkpoints",
    checkpoint_every=5,
    start_games=0,
    results_window=None,
    buffer_capacity=50_000,
    train_batch_size=2048,
    min_buffer_size=1024,
    model_out=None,
    base_lr=5e-4,
    min_lr=5e-5,
    warmup_games=2_000,
    amp_enabled=True,
    scaler=None,
    replay_path=None,
    replay_save_every=50,
    checkpoint_keep=10,
    resolved_config=None,
    seed=0,
):
    """Coordinate remote GPU inference, replay storage, and policy training."""
    policy.to(device)
    scaler = scaler or make_grad_scaler(amp_enabled)
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)
    results_window = deque(results_window or [], maxlen=100)
    termination_window = deque(maxlen=100)
    games_done = start_games
    round_idx = 0
    run_start = time.time()
    model_lock = threading.Lock()

    request_queue = mp.Queue()
    results_queue = mp.Queue()
    error_queue = mp.Queue()
    response_queues = [mp.Queue() for _ in range(num_workers)]
    stop_event = mp.Event()
    games_claimed = mp.Value("q", start_games)
    claim_lock = mp.Lock()
    thread_stop_event = threading.Event()
    inference_errors = []
    inference_stats = {"batches": 0, "positions": 0}
    workers = []
    inference_thread = None
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    replay_path = Path(replay_path or ckpt_dir / "replay_buffer.pt")
    if start_games and load_replay_buffer(replay_path, replay_buffer):
        print(f"Restored replay buffer with {len(replay_buffer)} samples")
    if resolved_config:
        (ckpt_dir / "resolved_config.json").write_text(
            json.dumps(resolved_config, indent=2, default=str), encoding="utf-8"
        )

    try:
        if games_done < episodes:
            for worker_id in range(num_workers):
                worker = mp.Process(
                    target=self_play_worker,
                    args=(
                        worker_id,
                        request_queue,
                        response_queues[worker_id],
                        results_queue,
                        stop_event,
                        games_per_batch,
                        num_simulations,
                        c_puct,
                        temp_threshold,
                        dirichlet_alpha,
                        dirichlet_eps,
                        max_moves,
                        avoid_training_repetitions,
                        repetition_value_threshold,
                        seed + 10_000 + start_games,
                        games_claimed,
                        claim_lock,
                        episodes,
                        error_queue,
                    ),
                    name=f"self-play-{worker_id}",
                )
                worker.start()
                workers.append(worker)

            inference_thread = threading.Thread(
                target=_inference_loop,
                args=(
                    policy,
                    device,
                    model_lock,
                    request_queue,
                    response_queues,
                    stop_event,
                    thread_stop_event,
                    inference_errors,
                    amp_enabled,
                    inference_stats,
                ),
                name="policy-inference",
                daemon=True,
            )
            inference_thread.start()

        while games_done < episodes:
            round_start = time.time()
            try:
                (
                    positions,
                    policies,
                    outcomes,
                    value_weights,
                    game_results,
                    terminations,
                    game_lengths,
                ) = results_queue.get(timeout=0.25)
            except queue.Empty:
                if inference_errors:
                    raise RuntimeError("Policy inference thread failed") from inference_errors[0]
                try:
                    worker_error = error_queue.get_nowait()
                except queue.Empty:
                    worker_error = None
                if worker_error is not None:
                    raise RuntimeError(worker_error)
                if workers and not any(worker.is_alive() for worker in workers):
                    raise RuntimeError("All self-play workers exited before training completed")
                continue

            if positions:
                replay_buffer.add(positions, policies, outcomes, value_weights)

            games_done += len(game_results)
            round_idx += 1
            results_window.extend(game_results)
            termination_window.extend(terminations)

            lr = apply_lr_schedule(
                optimizer, base_lr, min_lr, warmup_games, episodes, games_done
            )
            loss_metrics = None
            if len(replay_buffer) >= min_buffer_size:
                with model_lock:
                    for _ in range(epochs_per_batch):
                        loss_metrics = train_batch(
                            policy,
                            optimizer,
                            scaler,
                            *replay_buffer.sample(train_batch_size),
                            value_coef=value_coef,
                            device=device,
                            amp_enabled=amp_enabled,
                        )

            wins = results_window.count("1-0")
            losses = results_window.count("0-1")
            draws = results_window.count("1/2-1/2")
            n = max(len(results_window), 1)
            termination_counts = Counter(termination_window)
            termination_text = ", ".join(
                f"{name}={count}"
                for name, count in termination_counts.most_common()
            ) or "n/a"
            loss_text = (
                f"{loss_metrics['total']:.4f} "
                f"(p={loss_metrics['policy']:.4f}, v={loss_metrics['value']:.4f})"
                if loss_metrics is not None
                else "n/a (filling buffer)"
            )
            grad_text = (
                f"{loss_metrics['grad_norm']:.2f}"
                if loss_metrics is not None else "n/a"
            )
            elapsed = max(time.time() - run_start, 1e-9)
            games_per_hour = (games_done - start_games) / (elapsed / 3600)
            mean_inference_batch = (
                inference_stats["positions"] / max(inference_stats["batches"], 1)
            )
            try:
                queue_depth = request_queue.qsize()
            except (NotImplementedError, OSError):
                queue_depth = -1
            print(
                f"[{time.strftime('%H:%M:%S')} | "
                f"round {time.time() - round_start:.1f}s | "
                f"total {(time.time() - run_start) / 60:.1f}m] "
                f"Round {round_idx} | games {games_done}/{episodes} | "
                f"loss={loss_text} | "
                f"lr={lr:.2e} | grad={grad_text} | "
                f"buffer={len(replay_buffer)}/{buffer_capacity} | "
                f"{games_per_hour:.0f} games/h | "
                f"avg_plies={sum(game_lengths) / len(game_lengths):.1f} | "
                f"infer_batch={mean_inference_batch:.1f} queue={queue_depth} | "
                f"[last {n}: W{wins}/D{draws}/L{losses} "
                f"white_win%={100 * wins / n:.0f}] | "
                f"terminations: {termination_text}"
            )

            if checkpoint_every and round_idx % checkpoint_every == 0:
                with model_lock:
                    for checkpoint_path in (
                        ckpt_dir / f"ckpt_{games_done:07d}.pt",
                        ckpt_dir / "latest.pt",
                    ):
                        save_checkpoint(
                            checkpoint_path,
                            policy,
                            optimizer,
                            games_done,
                            results_window,
                            scaler=scaler,
                            replay_path=replay_path,
                            resolved_config=resolved_config,
                        )
                    numbered = sorted(ckpt_dir.glob("ckpt_*.pt"))
                    for stale in numbered[:-checkpoint_keep]:
                        stale.unlink()
                print(f"  ↳ checkpoint saved at {games_done} games")
            if replay_save_every and round_idx % replay_save_every == 0:
                save_replay_buffer(replay_path, replay_buffer)

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
        for worker in workers:
            worker.join(timeout=1.0)

        thread_stop_event.set()
        if inference_thread is not None:
            inference_thread.join(timeout=5.0)

        save_replay_buffer(replay_path, replay_buffer)
        _save_training_state(
            model_lock=model_lock,
            checkpoint_dir=checkpoint_dir,
            policy=policy,
            optimizer=optimizer,
            games_done=games_done,
            results_window=results_window,
            scaler=scaler,
            replay_path=replay_path,
            resolved_config=resolved_config,
            model_out=model_out,
        )

        request_queue.close()
        results_queue.close()
        error_queue.close()
        for response_queue in response_queues:
            response_queue.close()

    return policy


def parse_args():
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML training configuration file",
    )
    config_args, _ = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Train the RL chess policy with multiprocess self-play.",
        parents=[config_parser],
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument(
        "--games-per-batch",
        type=int,
        default=4,
        help="concurrent self-play games per worker round",
    )
    parser.add_argument("--num-workers", type=int, default=max(1, mp.cpu_count() - 2))
    parser.add_argument("--num-simulations", type=int, default=100)
    parser.add_argument("--c-puct", type=float, default=1.41)
    parser.add_argument("--temp-threshold", type=int, default=15)
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3)
    parser.add_argument("--dirichlet-eps", type=float, default=0.25)
    parser.add_argument("--epochs-per-batch", type=int, default=2)
    parser.add_argument("--value-coef", type=float, default=1.0)
    parser.add_argument("--buffer-capacity", type=int, default=50_000)
    parser.add_argument("--train-batch-size", type=int, default=2048)
    parser.add_argument("--min-buffer-size", type=int, default=1024)
    parser.add_argument("--max-moves", type=int, default=200)
    parser.add_argument(
        "--avoid-training-repetitions",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--repetition-value-threshold", type=float, default=-0.25)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--min-lr", type=float, default=5e-5)
    parser.add_argument("--warmup-games", type=int, default=2000)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--value-head-channels", type=int, default=32)
    parser.add_argument("--value-hidden", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=str(RL_DIR / "checkpoints"),
    )
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--checkpoint-keep", type=int, default=10)
    parser.add_argument("--replay-save-every", type=int, default=50)
    parser.add_argument("--replay-path", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--model-out",
        type=str,
        default=str(RL_DIR / "chess_rl_model.pt"),
    )

    try:
        config_defaults = load_training_config(config_args.config, "multiprocess")
    except (FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))
    valid_keys = {action.dest for action in parser._actions}
    unknown_keys = sorted(set(config_defaults) - valid_keys)
    if unknown_keys:
        parser.error(f"unknown multiprocess config keys: {', '.join(unknown_keys)}")
    parser.set_defaults(**config_defaults)
    args = parser.parse_args()

    if args.episodes < 0:
        parser.error("--episodes must be non-negative")
    if args.games_per_batch < 1:
        parser.error("--games-per-batch must be at least 1")
    if args.num_workers < 1:
        parser.error("--num-workers must be at least 1")
    if args.num_simulations < 1:
        parser.error("--num-simulations must be at least 1")
    if args.epochs_per_batch < 0:
        parser.error("--epochs-per-batch must be non-negative")
    if args.max_moves < 1:
        parser.error("--max-moves must be at least 1")
    if not 0.0 <= args.dirichlet_eps <= 1.0:
        parser.error("--dirichlet-eps must be between 0 and 1")
    if args.dirichlet_alpha <= 0:
        parser.error("--dirichlet-alpha must be positive")

    return args


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    policy = PolicyNetwork(
        channels=args.channels,
        num_blocks=args.num_blocks,
        value_head_channels=args.value_head_channels,
        value_hidden=args.value_hidden,
    )
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = make_grad_scaler(not args.disable_amp and device.type == "cuda")

    start_games = 0
    results_window = None
    if args.resume:
        meta = load_checkpoint(
            args.resume, policy, optimizer, device=device, scaler=scaler
        )
        start_games = meta["games_done"]
        results_window = meta["results_window"]
        if meta.get("replay_path"):
            args.replay_path = meta["replay_path"]
        print(f"Resumed from {args.resume} at {start_games} games")

    train_multiprocess(
        policy=policy,
        optimizer=optimizer,
        episodes=args.episodes,
        device=device,
        num_workers=args.num_workers,
        games_per_batch=args.games_per_batch,
        num_simulations=args.num_simulations,
        c_puct=args.c_puct,
        temp_threshold=args.temp_threshold,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_eps=args.dirichlet_eps,
        value_coef=args.value_coef,
        epochs_per_batch=args.epochs_per_batch,
        max_moves=args.max_moves,
        avoid_training_repetitions=args.avoid_training_repetitions,
        repetition_value_threshold=args.repetition_value_threshold,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        start_games=start_games,
        results_window=results_window,
        buffer_capacity=args.buffer_capacity,
        train_batch_size=args.train_batch_size,
        min_buffer_size=args.min_buffer_size,
        model_out=args.model_out,
        base_lr=args.lr,
        min_lr=args.min_lr,
        warmup_games=args.warmup_games,
        amp_enabled=not args.disable_amp,
        scaler=scaler,
        replay_path=args.replay_path,
        replay_save_every=args.replay_save_every,
        checkpoint_keep=args.checkpoint_keep,
        resolved_config=vars(args),
        seed=args.seed,
    )
    print(f"Final model weights saved to {args.model_out}")


if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()
