#!/usr/bin/env bash
# Held-out evaluation: edited vs original v7 SE, MCTS arena (both colours per opening).
# Usage: experiments/weight_space/eval.sh [pairs] [sims]
set -euo pipefail
cd "$(dirname "$0")/../.."
PAIRS="${1:-100}"
SIMS="${2:-200}"
PY=../.venv/bin/python
D=experiments/weight_space
echo "== MCTS arena: edited vs original, ${PAIRS} pairs, ${SIMS} sims =="
$PY arena.py "$D/edited.pt" "$D/original.pt" --pairs "$PAIRS" --sims "$SIMS" --seed 7777
echo "== Greedy policy head: edited vs original, ${PAIRS} pairs =="
$PY arena.py "$D/edited.pt" "$D/original.pt" --pairs "$PAIRS" --sims 0 --seed 7777
