#!/usr/bin/env python3
"""Supervised warm-start from packed Lichess shards (Chess RL v2).

Reads shard_*.npz files from prepare_lichess_dataset.py, trains the policy head
on the moves strong players chose and the value head on eval-blended outcomes,
and writes a weights file that multiprocess_train_v2.py loads via --init-model.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from model import PolicyNetwork
from sl_dataset import PackedDataset, packed_loss
from train import load_checkpoint, save_checkpoint
from training_config import DEFAULT_CONFIG_PATH, load_training_config
from training_runtime import (
    amp_context,
    apply_lr_schedule,
    make_grad_scaler,
    seed_everything,
)

RL_DIR = Path(__file__).resolve().parent
LOSS_KEYS = ("total", "policy", "value")


@torch.no_grad()
def evaluate(policy, dataset, batch_size, value_coef, device, amp_enabled):
    """Mean losses and top-1 move accuracy over the whole dataset."""
    if len(dataset) == 0:
        return {key: float("nan") for key in (*LOSS_KEYS, "accuracy")}
    policy.eval()
    sums = dict.fromkeys((*LOSS_KEYS, "accuracy"), 0.0)
    for start in range(0, len(dataset), batch_size):
        idxs = np.arange(start, min(start + batch_size, len(dataset)))
        x, legal_mask, move, value = dataset.batch(idxs, device)
        with amp_context(device, amp_enabled):
            logits, values = policy(x)
            total, policy_loss, value_loss = packed_loss(
                logits, values, legal_mask, move, value, value_coef
            )
        predicted = logits.float().masked_fill(~legal_mask, float("-inf")).argmax(1)
        n = len(idxs)
        sums["total"] += total.item() * n
        sums["policy"] += policy_loss.item() * n
        sums["value"] += value_loss.item() * n
        sums["accuracy"] += (predicted == move).float().sum().item()
    return {key: total / len(dataset) for key, total in sums.items()}


def save_loss_plot(history, plot_path: Path):
    plot_path = Path(plot_path)
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = [row["epoch"] for row in history]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4), constrained_layout=True)
    for ax, key in zip(axes, (*LOSS_KEYS, "accuracy")):
        if key != "accuracy":
            ax.plot(epochs, [row[f"train_{key}"] for row in history],
                    label="train", marker="o", markersize=3)
        ax.plot(epochs, [row[f"val_{key}"] for row in history],
                label="val", marker="o", markersize=3)
        ax.set_title("Val move accuracy" if key == "accuracy" else f"{key.title()} loss")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle("Supervised warm start")
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"Loss plot saved to {plot_path}")


def train_supervised(
    policy,
    optimizer,
    train_data,
    val_data,
    device,
    num_epochs,
    train_batch_size,
    value_coef,
    base_lr,
    min_lr,
    warmup_steps,
    amp_enabled,
    scaler,
    checkpoint_dir,
    checkpoint_keep,
    log_every=500,
    start_epoch=0,
    resolved_config=None,
    plot_path=None,
    model_out=None,
):
    policy.to(device)
    ckpt_dir = Path(checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    if resolved_config:
        (ckpt_dir / "resolved_config.json").write_text(
            json.dumps(resolved_config, indent=2, default=str), encoding="utf-8"
        )

    steps_per_epoch = len(train_data) // train_batch_size
    if steps_per_epoch < 1:
        raise ValueError("Fewer training positions than one batch")
    total_steps = steps_per_epoch * num_epochs
    step = start_epoch * steps_per_epoch
    print(
        f"{steps_per_epoch:,} steps/epoch x {num_epochs} epochs "
        f"(batch {train_batch_size})"
    )

    history_path = ckpt_dir / "loss_history.json"
    history = (
        json.loads(history_path.read_text(encoding="utf-8"))
        if start_epoch and history_path.is_file()
        else []
    )
    for epoch in range(start_epoch + 1, num_epochs + 1):
        policy.train()
        order = np.random.permutation(len(train_data))
        sums = dict.fromkeys(LOSS_KEYS, 0.0)
        window = dict.fromkeys(LOSS_KEYS, 0.0)
        window_start = time.monotonic()

        for batch_idx in range(steps_per_epoch):
            lr = apply_lr_schedule(
                optimizer, base_lr, min_lr, warmup_steps, total_steps, step
            )
            idxs = order[batch_idx * train_batch_size:(batch_idx + 1) * train_batch_size]
            x, legal_mask, move, value = train_data.batch(idxs, device)

            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, amp_enabled):
                losses = packed_loss(
                    *policy(x), legal_mask, move, value, value_coef
                )
            scaler.scale(losses[0]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            step += 1

            for key, loss in zip(LOSS_KEYS, losses):
                value_ = loss.item()
                sums[key] += value_
                window[key] += value_
            if (batch_idx + 1) % log_every == 0:
                elapsed = time.monotonic() - window_start
                print(
                    f"  epoch {epoch} step {batch_idx + 1:,}/{steps_per_epoch:,} | "
                    f"loss={window['total'] / log_every:.4f} "
                    f"(p={window['policy'] / log_every:.4f}, "
                    f"v={window['value'] / log_every:.4f}) | "
                    f"lr={lr:.2e} | "
                    f"{log_every * train_batch_size / elapsed:,.0f} pos/s",
                    flush=True,
                )
                window = dict.fromkeys(LOSS_KEYS, 0.0)
                window_start = time.monotonic()

        train_avg = {key: total / steps_per_epoch for key, total in sums.items()}
        val_avg = evaluate(
            policy, val_data, train_batch_size, value_coef, device, amp_enabled
        )
        history.append({
            "epoch": epoch,
            "lr": lr,
            **{f"train_{key}": train_avg[key] for key in LOSS_KEYS},
            **{f"val_{key}": val_avg[key] for key in (*LOSS_KEYS, "accuracy")},
        })
        print(
            f"Epoch {epoch}/{num_epochs} | "
            f"train={train_avg['total']:.4f} "
            f"(p={train_avg['policy']:.4f}, v={train_avg['value']:.4f}) | "
            f"val={val_avg['total']:.4f} "
            f"(p={val_avg['policy']:.4f}, v={val_avg['value']:.4f}, "
            f"acc={val_avg['accuracy']:.1%})",
            flush=True,
        )

        for checkpoint_path in (
            ckpt_dir / f"ckpt_epoch_{epoch:04d}.pt",
            ckpt_dir / "latest.pt",
        ):
            save_checkpoint(
                checkpoint_path,
                policy,
                optimizer,
                games_done=epoch,
                results_window=[],
                scaler=scaler,
                resolved_config=resolved_config,
            )
        for stale in sorted(ckpt_dir.glob("ckpt_epoch_*.pt"))[:-checkpoint_keep]:
            stale.unlink()
        history_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
        if plot_path is not None:
            save_loss_plot(history, Path(plot_path))

    if model_out is not None:
        model_path = Path(model_out)
        model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {**policy.checkpoint_metadata(), "policy_state_dict": policy.state_dict()},
            model_path,
        )
        print(f"Final model weights saved to {model_out}")

    return policy, history


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
        description="Supervised warm-start from packed Lichess shards.",
        parents=[config_parser],
    )
    parser.add_argument("--data-dir", type=str, default=str(RL_DIR / "data" / "lichess_sl"))
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="Load at most this many positions (default: every shard)",
    )
    parser.add_argument("--num-epochs", type=int, default=4)
    parser.add_argument(
        "--val-positions",
        type=int,
        default=100_000,
        help="Positions held out from the end of the data for validation",
    )
    parser.add_argument("--train-batch-size", type=int, default=1024)
    parser.add_argument("--value-coef", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--min-lr", type=float, default=1e-5)
    parser.add_argument("--warmup-steps", type=int, default=2000)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--channels", type=int, default=96)
    parser.add_argument("--num-blocks", type=int, default=8)
    parser.add_argument("--value-head-channels", type=int, default=32)
    parser.add_argument("--value-hidden", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--disable-amp", action="store_true")
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument(
        "--checkpoint-dir", type=str, default=str(RL_DIR / "checkpoints_sl")
    )
    parser.add_argument("--checkpoint-keep", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument(
        "--model-out", type=str, default=str(RL_DIR / "chess_sl_model.pt")
    )
    parser.add_argument(
        "--plot-path",
        type=str,
        default=str(RL_DIR / "checkpoints_sl" / "loss_curve.png"),
    )
    # Keys present in shared YAML sections but unused by supervised training.
    for unused in ("--warmup-games", "--checkpoint-every", "--buffer-capacity",
                   "--min-buffer-size", "--replay-save-every"):
        parser.add_argument(unused, type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--replay-path", type=str, default=None, help=argparse.SUPPRESS)

    try:
        config_defaults = load_training_config(
            config_args.config, "supervised", include_common=False
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))

    valid_keys = {action.dest for action in parser._actions}
    unknown_keys = sorted(set(config_defaults) - valid_keys)
    if unknown_keys:
        parser.error(f"unknown supervised config keys: {', '.join(unknown_keys)}")
    parser.set_defaults(**config_defaults)
    args = parser.parse_args()

    if args.num_epochs < 1:
        parser.error("--num-epochs must be at least 1")
    if args.val_positions < 0:
        parser.error("--val-positions must be non-negative")
    if args.train_batch_size < 1:
        parser.error("--train-batch-size must be at least 1")
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
    print(f"Device: {device}")

    policy = PolicyNetwork(
        channels=args.channels,
        num_blocks=args.num_blocks,
        value_head_channels=args.value_head_channels,
        value_hidden=args.value_hidden,
    )
    optimizer = torch.optim.AdamW(
        policy.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    amp_enabled = not args.disable_amp
    scaler = make_grad_scaler(amp_enabled and device.type == "cuda")

    start_epoch = 0
    if args.resume:
        meta = load_checkpoint(
            args.resume, policy, optimizer, device=device, scaler=scaler
        )
        start_epoch = meta["games_done"]
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    print(f"Loading shards from {args.data_dir}")
    data = PackedDataset.load_dir(args.data_dir, max_positions=args.max_positions)
    val_positions = min(args.val_positions, len(data) // 10)
    train_data, val_data = data.split_tail(val_positions)
    del data
    print(f"Train={len(train_data):,}  Val={len(val_data):,}")

    train_supervised(
        policy,
        optimizer,
        train_data,
        val_data,
        device=device,
        num_epochs=args.num_epochs,
        train_batch_size=args.train_batch_size,
        value_coef=args.value_coef,
        base_lr=args.lr,
        min_lr=args.min_lr,
        warmup_steps=args.warmup_steps,
        amp_enabled=amp_enabled,
        scaler=scaler,
        checkpoint_dir=args.checkpoint_dir,
        checkpoint_keep=args.checkpoint_keep,
        log_every=args.log_every,
        start_epoch=start_epoch,
        resolved_config=vars(args),
        plot_path=args.plot_path,
        model_out=args.model_out,
    )


if __name__ == "__main__":
    main()
