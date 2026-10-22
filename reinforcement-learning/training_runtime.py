import hashlib
import json
import math
import os
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def amp_context(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast(device_type="cuda", dtype=torch.float16)
    return nullcontext()


def make_grad_scaler(enabled):
    return torch.amp.GradScaler(
        "cuda",
        init_scale=8192.0,
        enabled=enabled and torch.cuda.is_available(),
    )


def scheduled_lr(base_lr, min_lr, warmup_games, total_games, games_done):
    if warmup_games > 0 and games_done < warmup_games:
        warmup_fraction = max(games_done, 1) / warmup_games
        return base_lr * (0.1 + 0.9 * warmup_fraction)
    progress = (games_done - warmup_games) / max(total_games - warmup_games, 1)
    progress = min(max(progress, 0.0), 1.0)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))


def apply_lr_schedule(
    optimizer, base_lr, min_lr, warmup_games, total_games, games_done
):
    lr = scheduled_lr(base_lr, min_lr, warmup_games, total_games, games_done)
    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def train_batch(
    policy,
    optimizer,
    scaler,
    positions,
    policies,
    outcomes,
    value_weights,
    value_coef,
    device,
    amp_enabled,
):
    optimizer.zero_grad(set_to_none=True)
    outcomes_tensor = torch.tensor(outcomes, dtype=torch.float32)
    weights_tensor = torch.tensor(value_weights, dtype=torch.float32)
    with amp_context(device, amp_enabled):
        total_loss, policy_loss, value_loss = policy.alphazero_loss(
            positions,
            policies,
            outcomes_tensor,
            value_weights=weights_tensor,
            value_coef=value_coef,
        )
    scaler.scale(total_loss).backward()
    scaler.unscale_(optimizer)
    grad_norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1.0)
    scaler.step(optimizer)
    scaler.update()
    return {
        "total": total_loss.detach().item(),
        "policy": policy_loss.detach().item(),
        "value": value_loss.detach().item(),
        "grad_norm": float(grad_norm),
    }


@torch.no_grad()
def eval_batch(
    policy,
    positions,
    policies,
    outcomes,
    value_weights,
    value_coef,
    device,
    amp_enabled,
):
    outcomes_tensor = torch.tensor(outcomes, dtype=torch.float32)
    weights_tensor = torch.tensor(value_weights, dtype=torch.float32)
    with amp_context(device, amp_enabled):
        total_loss, policy_loss, value_loss = policy.alphazero_loss(
            positions,
            policies,
            outcomes_tensor,
            value_weights=weights_tensor,
            value_coef=value_coef,
        )
    return {
        "total": total_loss.detach().item(),
        "policy": policy_loss.detach().item(),
        "value": value_loss.detach().item(),
    }


def save_replay_buffer(path, replay_buffer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic replace avoids leaving a truncated .pt if the process is killed mid-save.
    tmp_path = path.with_name(f"{path.name}.tmp")
    torch.save(replay_buffer.state_dict(), tmp_path)
    os.replace(tmp_path, path)


def load_replay_buffer(path, replay_buffer):
    path = Path(path)
    if not path.is_file():
        return False
    try:
        replay_buffer.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=False)
        )
    except (RuntimeError, EOFError, OSError, ValueError) as exc:
        print(
            f"Warning: could not restore replay buffer from {path} ({exc}); "
            "starting with an empty buffer"
        )
        return False
    return True


def config_fingerprint(config):
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]
