import hashlib
import json
import math
import random
from contextlib import nullcontext

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


def config_fingerprint(config):
    payload = json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:12]
