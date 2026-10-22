"""Ring-buffer replay of packed self-play samples (sl_dataset row format).

Each sample is ~300 bytes of planes plus its legal-move and visit-count
arrays, and batches decode on the GPU, so training no longer re-parses FENs
on the coordinator while self-play inference waits on the model lock.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from chess_environment import ACTION_VERSION, ENCODING_VERSION, POLICY_SIZE
from sl_dataset import decode_planes

REPLAY_FORMAT_VERSION = 1
_RAGGED = ("legal", "pi_idx", "pi_prob")


class PackedReplay:
    def __init__(self, capacity):
        self.capacity = capacity
        self.pieces = np.zeros((capacity, 8, 32), dtype=np.uint8)
        self.aux = np.zeros((capacity, 4), dtype=np.uint8)
        self.value = np.zeros(capacity, dtype=np.float32)
        self.weight = np.zeros(capacity, dtype=np.float32)
        self.legal = [None] * capacity
        self.pi_idx = [None] * capacity
        self.pi_prob = [None] * capacity
        self.size = 0
        self.next = 0
        self.total_added = 0

    def __len__(self):
        return self.size

    def add(self, samples):
        """`samples`: dict of pieces/aux/value/weight arrays + ragged lists."""
        n = len(samples["value"])
        if n == 0:
            return
        slots = (self.next + np.arange(n)) % self.capacity
        self.pieces[slots] = samples["pieces"]
        self.aux[slots] = samples["aux"]
        self.value[slots] = samples["value"]
        self.weight[slots] = samples["weight"]
        for name in _RAGGED:
            store = getattr(self, name)
            for slot, row in zip(slots, samples[name]):
                store[slot] = row
        self.next = int((self.next + n) % self.capacity)
        self.size = min(self.size + n, self.capacity)
        self.total_added += n

    def sample(self, n, rng):
        return rng.integers(0, self.size, size=n)

    def batch(self, idxs, device):
        """(x, legal_mask, policy_target, value, weight) tensors on `device`."""
        b = len(idxs)
        legal = [self.legal[i] for i in idxs]
        pi_idx = [self.pi_idx[i] for i in idxs]
        legal_mask = _scatter(
            torch.zeros(b, POLICY_SIZE, dtype=torch.bool, device=device), legal, True
        )
        target = torch.zeros(b, POLICY_SIZE, device=device)
        rows = np.repeat(np.arange(b), [len(r) for r in pi_idx])
        target[
            torch.from_numpy(rows).to(device),
            torch.from_numpy(np.concatenate(pi_idx).astype(np.int64)).to(device),
        ] = torch.from_numpy(
            np.concatenate([self.pi_prob[i] for i in idxs]).astype(np.float32)
        ).to(device)
        x = decode_planes(
            torch.from_numpy(self.pieces[idxs]).to(device),
            torch.from_numpy(self.aux[idxs]).to(device),
        )
        value = torch.from_numpy(self.value[idxs]).to(device)
        weight = torch.from_numpy(self.weight[idxs]).to(device)
        return x, legal_mask, target, value, weight

    def _ordered(self):
        """Slot order from oldest to newest."""
        if self.size < self.capacity:
            return np.arange(self.size)
        return (self.next + np.arange(self.capacity)) % self.capacity

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        order = self._ordered()
        arrays = {
            "format_version": REPLAY_FORMAT_VERSION,
            "encoding_version": ENCODING_VERSION,
            "action_version": ACTION_VERSION,
            "total_added": self.total_added,
            "pieces": self.pieces[order],
            "aux": self.aux[order],
            "value": self.value[order],
            "weight": self.weight[order],
        }
        for name in _RAGGED:
            rows = [getattr(self, name)[i] for i in order]
            arrays[f"{name}_off"] = np.concatenate(
                [[0], np.cumsum([len(r) for r in rows])]
            ).astype(np.int64)
            arrays[f"{name}_flat"] = (
                np.concatenate(rows) if rows else np.zeros(0, dtype=np.float32)
            )
        # Atomic replace avoids a truncated file if the run is killed mid-save.
        tmp = path.with_name(path.name + ".tmp.npz")
        np.savez(tmp, **arrays)
        os.replace(tmp, path)

    def load(self, path):
        path = Path(path)
        if not path.is_file():
            return False
        with np.load(path) as data:
            if (
                int(data["format_version"]) != REPLAY_FORMAT_VERSION
                or int(data["encoding_version"]) != ENCODING_VERSION
                or int(data["action_version"]) != ACTION_VERSION
            ):
                print(f"Warning: {path} has an incompatible replay format; ignoring it")
                return False
            samples = {
                key: data[key][-self.capacity:]
                for key in ("pieces", "aux", "value", "weight")
            }
            n = len(samples["value"])
            for name in _RAGGED:
                off, flat = data[f"{name}_off"], data[f"{name}_flat"]
                rows = [flat[off[i]:off[i + 1]] for i in range(len(off) - 1)]
                samples[name] = rows[len(rows) - n:]
            total_added = int(data["total_added"])
        self.add(samples)
        self.total_added = total_added
        return True


def _scatter(mask, rows_of_indices, value):
    b = len(rows_of_indices)
    rows = np.repeat(np.arange(b), [len(r) for r in rows_of_indices])
    flat = np.concatenate(rows_of_indices).astype(np.int64)
    mask[torch.from_numpy(rows).to(mask.device), torch.from_numpy(flat).to(mask.device)] = value
    return mask


def soft_policy_loss(logits, legal_mask, target):
    """Per-row cross-entropy of a (soft) target over legal moves only."""
    log_probs = F.log_softmax(
        logits.float().masked_fill(~legal_mask, float("-inf")), dim=1
    )
    return -(target * log_probs.masked_fill(~legal_mask, 0.0)).sum(dim=1)
