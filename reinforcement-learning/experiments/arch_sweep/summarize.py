#!/usr/bin/env python3
"""Collect the arch sweep runs into a Markdown table (results.md).

Per run: params, final-epoch validation metrics, training throughput, and
batch-1 / batch-64 inference latency (what MCTS pays per leaf evaluation).
"""

import json
import re
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from chess_environment import INPUT_CHANNELS  # noqa: E402
from model import PolicyNetwork  # noqa: E402

RUNS = ("baseline", "baseline_seed1", "se", "transformer", "transformer_bias", "hybrid")


@torch.no_grad()
def latency_ms(policy, batch, device, iters=200):
    x = torch.randn(batch, INPUT_CHANNELS, 8, 8, device=device)
    for _ in range(20):
        policy(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        policy(x)
    if device.type == "cuda":
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters * 1e3


def describe(cfg):
    if cfg["arch"] == "transformer":
        text = f"transformer d{cfg['channels']} x{cfg['attention_layers']}L"
        return text + (" +attn bias" if cfg["attention_bias"] else "")
    text = f"resnet {cfg['channels']}ch x{cfg['num_blocks']}"
    if cfg["se_ratio"]:
        text += " +SE"
    if cfg["attention_layers"]:
        text += f" +{cfg['attention_layers']} attn"
    return text


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for name in RUNS:
        run_dir = HERE / name
        if not (run_dir / "model.pt").is_file():
            continue
        ckpt = torch.load(run_dir / "model.pt", map_location=device)
        policy = PolicyNetwork(**ckpt["model_config"]).to(device).eval()
        policy.load_state_dict(ckpt["policy_state_dict"])
        final = json.loads((run_dir / "loss_history.json").read_text())[-1]
        speeds = [
            float(s.replace(",", ""))
            for s in re.findall(r"([\d,]+) pos/s", (run_dir / "train.log").read_text())
        ]
        rows.append({
            "run": name,
            "model": describe(ckpt["model_config"]),
            "params": sum(p.numel() for p in policy.parameters()),
            "acc": final["val_accuracy"],
            "policy": final["val_policy"],
            "value": final["val_value"],
            "train_pos_s": sum(speeds) / len(speeds) if speeds else float("nan"),
            "b1_ms": latency_ms(policy, 1, device),
            "b64_ms": latency_ms(policy, 64, device),
        })

    base = next((r for r in rows if r["run"] == "baseline"), None)
    lines = [
        "| Run | Model | Params | Val acc | Δ acc | Val policy loss | Val value loss "
        "| Train pos/s | Infer b=1 (ms) | Infer b=64 (ms) |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        delta = f"{(r['acc'] - base['acc']) * 100:+.2f} pp" if base else ""
        lines.append(
            f"| {r['run']} | {r['model']} | {r['params'] / 1e6:.2f}M | {r['acc']:.2%} "
            f"| {delta} | {r['policy']:.4f} | {r['value']:.4f} "
            f"| {r['train_pos_s']:,.0f} | {r['b1_ms']:.2f} | {r['b64_ms']:.2f} |"
        )
    table = "\n".join(lines)
    (HERE / "results.md").write_text(table + "\n", encoding="utf-8")
    print(table)


if __name__ == "__main__":
    main()
