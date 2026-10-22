import sys
import time
import json
import random
from collections import Counter, deque
from pathlib import Path

import chess
import numpy as np
import torch
from model import MODEL_VERSION, PolicyNetwork
from chess_environment import *
from mcts import (
    mcts_search_batch,
    policy_target,
    select_move,
    training_repetition_exclusions,
)
from replay_buffer import ReplayBuffer
from training_config import DEFAULT_CONFIG_PATH, load_training_config
from training_runtime import (
    apply_lr_schedule,
    config_fingerprint,
    load_replay_buffer,
    make_grad_scaler,
    save_replay_buffer,
    seed_everything,
    train_batch,
)

CHESS_DIR = Path(__file__).resolve().parent.parent
RL_DIR = Path(__file__).resolve().parent

# The pygame UI is only needed when visualizing. Import it lazily so headless
# training (e.g. on a server with no display) never depends on pygame.
chess_ui = None


def _ensure_ui():
    """Import chess/main.py (pygame UI) on demand. Returns the module."""
    global chess_ui
    if chess_ui is None:
        if str(CHESS_DIR) not in sys.path:
            sys.path.insert(0, str(CHESS_DIR))
        import main as _chess_ui  # noqa: WPS433 (intentional lazy import)

        chess_ui = _chess_ui
    return chess_ui


def _pump_events():
    """Process pygame events; return False if the user closed the window."""
    ui = _ensure_ui()
    for event in ui.pygame.event.get():
        if event.type == ui.pygame.QUIT:
            return False
    return True


def _draw_position(board, last_move=None, caption=None):
    ui = _ensure_ui()
    if caption:
        ui.pygame.display.set_caption(caption)
    ui.draw_board_and_pieces(
        ui.screen,
        board,
        ui.pieces,
        selected_piece=None,
        blink_timer=1,
        last_ai_move=last_move,
        possible_moves=[],
        player=chess.WHITE,
    )
    ui.pygame.display.flip()


def save_checkpoint(
    path,
    policy,
    optimizer,
    games_done,
    results_window,
    scaler=None,
    replay_path=None,
    resolved_config=None,
):
    """Persist full training state so a run can be resumed later."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            **policy.checkpoint_metadata(),
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
            "games_done": games_done,
            "results_window": list(results_window),
            "replay_path": str(replay_path) if replay_path is not None else None,
            "resolved_config": resolved_config,
            "config_fingerprint": (
                config_fingerprint(resolved_config) if resolved_config else None
            ),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": (
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
            ),
        },
        path,
    )


def load_checkpoint(path, policy, optimizer=None, device=None, scaler=None):
    """Restore training state saved by save_checkpoint. Returns a metadata dict."""
    device = device or torch.device("cpu")
    # Always unpickle on CPU so ByteTensor RNG states are not remapped to CUDA.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    expected_metadata = policy.checkpoint_metadata()
    if ckpt.get("model_version") != MODEL_VERSION:
        raise ValueError(
            "Checkpoint is not compatible with Chess RL v2; start fresh or use a v2 checkpoint"
        )
    if ckpt.get("model_config") != policy.config:
        raise ValueError(
            f"Checkpoint model config {ckpt.get('model_config')} does not match "
            f"requested config {policy.config}"
        )
    for key in ("encoding_version", "action_version", "policy_size"):
        if ckpt.get(key) != expected_metadata[key]:
            raise ValueError(
                f"Checkpoint {key}={ckpt.get(key)} does not match "
                f"expected {expected_metadata[key]}"
            )
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.to(device)
    if optimizer is not None and ckpt.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        # Ensure any optimizer state tensors (exp_avg, etc.) live on `device`,
        # otherwise a resumed step mixes CPU/MPS tensors and crashes.
        for state in optimizer.state.values():
            for key, val in state.items():
                if isinstance(val, torch.Tensor):
                    state[key] = val.to(device)
    if scaler is not None and ckpt.get("scaler_state_dict") is not None:
        scaler.load_state_dict(ckpt["scaler_state_dict"])
    if ckpt.get("python_rng_state") is not None:
        random.setstate(ckpt["python_rng_state"])
    if ckpt.get("numpy_rng_state") is not None:
        np.random.set_state(ckpt["numpy_rng_state"])
    if ckpt.get("torch_rng_state") is not None:
        cpu_state = ckpt["torch_rng_state"]
        if isinstance(cpu_state, torch.Tensor):
            cpu_state = cpu_state.detach().cpu().contiguous().to(torch.uint8)
        torch.set_rng_state(cpu_state)
    if torch.cuda.is_available() and ckpt.get("cuda_rng_state") is not None:
        # map_location can strip ByteTensor typing; set_rng_state_all requires it.
        try:
            cuda_states = []
            for state in ckpt["cuda_rng_state"]:
                if isinstance(state, torch.Tensor):
                    cuda_states.append(
                        state.detach().cpu().contiguous().to(torch.uint8)
                    )
                else:
                    cuda_states.append(
                        torch.as_tensor(state, dtype=torch.uint8).contiguous()
                    )
            torch.cuda.set_rng_state_all(cuda_states)
        except (RuntimeError, TypeError, ValueError) as exc:
            print(f"Warning: could not restore CUDA RNG state ({exc}); continuing")
    return {
        "games_done": ckpt.get("games_done", 0),
        "results_window": ckpt.get("results_window", []),
        "replay_path": ckpt.get("replay_path"),
        "resolved_config": ckpt.get("resolved_config"),
    }


def self_play_games_batch(policy, device, num_games, num_simulations, c_puct, temp_threshold,
                           dirichlet_alpha, dirichlet_eps, max_moves=200,
                           avoid_training_repetitions=True,
                           repetition_value_threshold=-0.25,
                           on_move=None, vis_index=None):
    """Play `num_games` self-play games concurrently (root parallelization).

    Every ply, all still-running games advance one MCTS search together, and
    mcts_search_batch shares their leaf evaluations across a single forward
    pass per simulation round instead of one pass per game — the lever that
    actually matters on a GPU, since a small network's single-position forward
    pass is dominated by launch overhead rather than compute.

    Returns compact v2 samples, value weights, results, terminations, and
    lengths. The sample arrays are flattened across every game; `results` and
    `terminations` are one entry per game (len == num_games). `terminations`
    names why each game ended (e.g. "CHECKMATE", "THREEFOLD_REPETITION",
    "MOVE_CAP" for hitting the ply cap without a ruled game-over, or
    "INTERRUPTED" if the visualization window was closed mid-game).
    on_move(board, move), if given, is called after `vis_index`'s game makes
    a move and should return False to stop every game immediately.
    """
    boards = [chess.Board() for _ in range(num_games)]
    histories = [[] for _ in range(num_games)]  # (position, pi, mover)
    plies = [0] * num_games
    done = [False] * num_games
    results = [None] * num_games
    terminations = [None] * num_games

    while not all(done):
        active = [i for i in range(num_games) if not done[i]]
        roots = mcts_search_batch(
            policy, [boards[i] for i in active],
            num_simulations=num_simulations, c=c_puct, device=device,
            add_dirichlet_noise=True,
            dirichlet_alpha=dirichlet_alpha, dirichlet_eps=dirichlet_eps,
        )

        for root, i in zip(roots, active):
            temperature = 1.0 if plies[i] < temp_threshold else 0.0
            excluded_moves = (
                training_repetition_exclusions(
                    root, value_threshold=repetition_value_threshold
                )
                if avoid_training_repetitions else set()
            )
            # Store the temperature=1 (raw visit-count) distribution as the
            # training target regardless of temperature used to pick the move.
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
                    terminations[i] = boards[i].outcome(claim_draw=True).termination.name
                else:
                    results[i] = "1/2-1/2"
                    terminations[i] = "MOVE_CAP"

            if i == vis_index and on_move is not None and not on_move(boards[i], move):
                done = [True] * num_games
                break

    positions_flat, pis_flat, outcomes_flat, value_weights_flat = [], [], [], []
    for i in range(num_games):
        # Move-cap or user-interrupted games are treated as drawn (no signal to learn).
        result = results[i] if results[i] is not None else "1/2-1/2"
        terminations[i] = terminations[i] or "INTERRUPTED"
        value_weight = 0.0 if terminations[i] in ("MOVE_CAP", "INTERRUPTED") else 1.0
        for position, pi, mover in histories[i]:
            positions_flat.append(position)
            pis_flat.append(pi)
            outcomes_flat.append(outcome_reward_for(result, mover))
            value_weights_flat.append(value_weight)

    return (
        positions_flat,
        pis_flat,
        outcomes_flat,
        value_weights_flat,
        results,
        terminations,
        plies,
    )


def train(
    policy,
    optimizer,
    episodes,
    device,
    games_per_batch=4,
    num_simulations=100,
    c_puct=1.41,
    temp_threshold=15,
    dirichlet_alpha=0.3,
    dirichlet_eps=0.25,
    value_coef=1.0,
    epochs_per_batch=2,
    max_moves=200,
    visualize=False,
    move_delay=0.15,
    checkpoint_dir="checkpoints",
    checkpoint_every=5,
    start_games=0,
    results_window=None,
    buffer_capacity=50_000,
    train_batch_size=2048,
    min_buffer_size=1024,
    avoid_training_repetitions=True,
    repetition_value_threshold=-0.25,
    base_lr=5e-4,
    min_lr=5e-5,
    warmup_games=2_000,
    amp_enabled=True,
    scaler=None,
    replay_path=None,
    replay_save_every=50,
    checkpoint_keep=10,
    resolved_config=None,
):
    """Run versioned AlphaZero-style self-play and optimization."""
    policy.to(device)
    running = True
    scaler = scaler or make_grad_scaler(amp_enabled)

    results_window = deque(results_window or [], maxlen=100)
    draw_reason_window = deque(maxlen=100)

    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    replay_path = Path(replay_path or ckpt_dir / "replay_buffer.pt")
    games_done = start_games
    batch_idx = 0
    replay_buffer = ReplayBuffer(capacity=buffer_capacity)
    if start_games and load_replay_buffer(replay_path, replay_buffer):
        print(f"Restored replay buffer with {len(replay_buffer)} samples")
    if resolved_config:
        (ckpt_dir / "resolved_config.json").write_text(
            json.dumps(resolved_config, indent=2, default=str), encoding="utf-8"
        )
    run_start = time.time()

    while games_done < episodes and running:
        batch_start = time.time()
        batch_size = min(games_per_batch, episodes - games_done)

        show = visualize and batch_size > 0
        if show:
            _draw_position(
                chess.Board(),
                caption=(
                    f"RL Chess — batch {batch_idx + 1} ({batch_size} games), "
                    f"{games_done}/{episodes} games done"
                ),
            )

        def on_move(board, move):
            nonlocal running
            running = _pump_events()
            if running:
                _draw_position(board, last_move=move)
                time.sleep(move_delay)
            return running

        (
            all_positions,
            all_pis,
            all_outcomes,
            all_value_weights,
            batch_results,
            batch_terminations,
            batch_lengths,
        ) = self_play_games_batch(
            policy, device,
            num_games=batch_size,
            num_simulations=num_simulations,
            c_puct=c_puct,
            temp_threshold=temp_threshold,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_eps=dirichlet_eps,
            max_moves=max_moves,
            avoid_training_repetitions=avoid_training_repetitions,
            repetition_value_threshold=repetition_value_threshold,
            on_move=on_move if show else None,
            vis_index=0 if show else None,
        )

        if not running:
            break

        games_done += batch_size
        batch_idx += 1

        if all_positions:
            replay_buffer.add(
                all_positions, all_pis, all_outcomes, all_value_weights
            )
            loss_metrics = None
            lr = apply_lr_schedule(
                optimizer, base_lr, min_lr, warmup_games, episodes, games_done
            )
            if len(replay_buffer) >= min_buffer_size:
                for _ in range(epochs_per_batch):
                    sample = replay_buffer.sample(train_batch_size)
                    loss_metrics = train_batch(
                        policy,
                        optimizer,
                        scaler,
                        *sample,
                        value_coef=value_coef,
                        device=device,
                        amp_enabled=amp_enabled,
                    )

            results_window.extend(batch_results)
            wins = results_window.count("1-0")
            losses = results_window.count("0-1")
            draws = results_window.count("1/2-1/2")
            n = max(len(results_window), 1)

            draw_reason_window.extend(
                termination for result, termination in zip(batch_results, batch_terminations)
                if result == "1/2-1/2"
            )
            draw_reason_counts = Counter(draw_reason_window)
            draw_reason_str = ", ".join(
                f"{reason}={count}" for reason, count in draw_reason_counts.most_common()
            ) or "n/a"

            loss_str = (
                f"{loss_metrics['total']:.4f} "
                f"(p={loss_metrics['policy']:.4f}, v={loss_metrics['value']:.4f})"
                if loss_metrics is not None else "n/a (filling buffer)"
            )
            grad_str = (
                f"{loss_metrics['grad_norm']:.2f}"
                if loss_metrics is not None else "n/a"
            )
            batch_elapsed = time.time() - batch_start
            run_elapsed = time.time() - run_start
            games_per_hour = (games_done - start_games) / max(run_elapsed / 3600, 1e-9)
            print(
                f"[{time.strftime('%H:%M:%S')} | batch {batch_elapsed:.1f}s | "
                f"total {run_elapsed / 60:.1f}m] "
                f"Batch {batch_idx} | games {games_done}/{episodes} | "
                f"loss={loss_str} | "
                f"lr={lr:.2e} | grad={grad_str} | "
                f"buffer={len(replay_buffer)}/{buffer_capacity} | "
                f"{games_per_hour:.0f} games/h | avg_plies={sum(batch_lengths) / len(batch_lengths):.1f} | "
                f"[last {n}: W{wins}/D{draws}/L{losses} "
                f"white_win%={100 * wins / n:.0f}] | "
                f"draws[last {len(draw_reason_window)}]: {draw_reason_str}"
            )

        # Periodic checkpointing.
        if checkpoint_every and batch_idx % checkpoint_every == 0:
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
        if replay_save_every and batch_idx % replay_save_every == 0:
            save_replay_buffer(replay_path, replay_buffer)

    save_replay_buffer(replay_path, replay_buffer)
    save_checkpoint(
        ckpt_dir / "latest.pt",
        policy,
        optimizer,
        games_done,
        results_window,
        scaler=scaler,
        replay_path=replay_path,
        resolved_config=resolved_config,
    )

    if visualize and chess_ui is not None:
        chess_ui.pygame.quit()

    return policy


def visualize_training(policy, device, games=3, move_delay=0.4):
    """Play self-play games with the policy and render them using chess/main.py UI."""
    policy.eval()
    pygame = _ensure_ui().pygame
    pygame.display.set_caption("RL Chess — Training Visualization")

    running = True
    for game_idx in range(games):
        if not running:
            break

        board = chess.Board()
        last_move = None
        _draw_position(board)

        while not board.is_game_over(claim_draw=True) and running:
            running = _pump_events()
            if not running:
                break

            x = board_to_tensor(board).to(device)
            logits, _ = policy(x)
            move = tensor_to_move(logits, board)

            board.push(move)
            last_move = move
            _draw_position(board, last_move=last_move)
            time.sleep(move_delay)

        if running:
            print(f"Game {game_idx + 1}/{games} result: {board.result(claim_draw=True)}")
            end_pause = time.time() + 1.5
            while time.time() < end_pause and running:
                running = _pump_events()
                pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    import argparse

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_PATH),
        help="YAML training configuration file",
    )
    config_args, _ = config_parser.parse_known_args()

    parser = argparse.ArgumentParser(
        description="Train the RL chess policy.",
        parents=[config_parser],
    )
    parser.add_argument("--episodes", type=int, default=1000,
                        help="total number of self-play games")
    parser.add_argument("--games-per-batch", type=int, default=4,
                        help="self-play games run concurrently (batched MCTS) before each update")
    parser.add_argument("--num-simulations", type=int, default=100,
                        help="MCTS simulations per move")
    parser.add_argument("--c-puct", type=float, default=1.41,
                        help="PUCT exploration constant")
    parser.add_argument("--temp-threshold", type=int, default=15,
                        help="plies before move selection switches from sampling to greedy")
    parser.add_argument("--dirichlet-alpha", type=float, default=0.3,
                        help="Dirichlet root-noise concentration")
    parser.add_argument("--dirichlet-eps", type=float, default=0.25,
                        help="fraction of root prior replaced by Dirichlet noise")
    parser.add_argument("--epochs-per-batch", type=int, default=2,
                        help="gradient-update passes per self-play batch, each over a "
                             "fresh minibatch sampled from the replay buffer")
    parser.add_argument("--value-coef", type=float, default=1.0,
                        help="weight on the value-head MSE loss term")
    parser.add_argument("--buffer-capacity", type=int, default=50_000,
                        help="max samples kept in the replay buffer")
    parser.add_argument("--train-batch-size", type=int, default=2048,
                        help="samples drawn from the replay buffer per gradient update")
    parser.add_argument("--min-buffer-size", type=int, default=1024,
                        help="skip training until the buffer holds at least this many samples")
    parser.add_argument("--max-moves", type=int, default=200,
                        help="ply cap per self-play game (treated as a draw if reached)")
    parser.add_argument(
        "--avoid-training-repetitions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="exclude immediate third repetitions from non-losing self-play targets",
    )
    parser.add_argument("--repetition-value-threshold", type=float, default=-0.25)
    parser.add_argument("--lr", type=float, default=0.001,
                        help="AdamW peak learning rate")
    parser.add_argument("--min-lr", type=float, default=5e-5)
    parser.add_argument("--warmup-games", type=int, default=2000)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--value-head-channels", type=int, default=32)
    parser.add_argument("--value-hidden", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--no-visualize", action="store_true",
                        help="train headless (no pygame window)")
    parser.add_argument("--move-delay", type=float, default=0.1,
                        help="seconds between rendered moves of the shown game")
    parser.add_argument("--checkpoint-dir", type=str,
                        default=str(RL_DIR / "checkpoints"))
    parser.add_argument("--checkpoint-every", type=int, default=5,
                        help="save a checkpoint every N batches")
    parser.add_argument("--checkpoint-keep", type=int, default=10)
    parser.add_argument("--replay-save-every", type=int, default=50)
    parser.add_argument("--replay-path", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None,
                        help="path to a checkpoint to resume from")
    parser.add_argument("--model-out", type=str,
                        default=str(RL_DIR / "chess_rl_model.pt"),
                        help="where to save the final trained weights")

    try:
        config_defaults = load_training_config(config_args.config, "single_process")
    except (FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))
    valid_keys = {action.dest for action in parser._actions}
    unknown_keys = sorted(set(config_defaults) - valid_keys)
    if unknown_keys:
        parser.error(f"unknown single-process config keys: {', '.join(unknown_keys)}")
    parser.set_defaults(**config_defaults)
    args = parser.parse_args()
    if args.episodes < 0:
        parser.error("--episodes must be non-negative")
    if args.games_per_batch < 1:
        parser.error("--games-per-batch must be at least 1")
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

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
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

    policy = train(
        policy,
        optimizer,
        episodes=args.episodes,
        device=device,
        games_per_batch=args.games_per_batch,
        num_simulations=args.num_simulations,
        c_puct=args.c_puct,
        temp_threshold=args.temp_threshold,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_eps=args.dirichlet_eps,
        epochs_per_batch=args.epochs_per_batch,
        value_coef=args.value_coef,
        max_moves=args.max_moves,
        avoid_training_repetitions=args.avoid_training_repetitions,
        repetition_value_threshold=args.repetition_value_threshold,
        buffer_capacity=args.buffer_capacity,
        train_batch_size=args.train_batch_size,
        min_buffer_size=args.min_buffer_size,
        visualize=not args.no_visualize,
        move_delay=args.move_delay,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_every=args.checkpoint_every,
        checkpoint_keep=args.checkpoint_keep,
        start_games=start_games,
        results_window=results_window,
        base_lr=args.lr,
        min_lr=args.min_lr,
        warmup_games=args.warmup_games,
        amp_enabled=not args.disable_amp,
        scaler=scaler,
        replay_path=args.replay_path,
        replay_save_every=args.replay_save_every,
        resolved_config=vars(args),
    )

    model_path = Path(args.model_out)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {**policy.checkpoint_metadata(), "policy_state_dict": policy.state_dict()},
        model_path,
    )
    print(f"Final model weights saved to {args.model_out}")
