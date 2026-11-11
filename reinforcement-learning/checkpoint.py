"""Save and restore full training state (weights, optimizer, AMP scaler, RNG)."""

import random
from pathlib import Path

import numpy as np
import torch

from model import MODEL_VERSION, full_model_config
from training_runtime import config_fingerprint


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
    if full_model_config(ckpt.get("model_config") or {}) != policy.config:
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


