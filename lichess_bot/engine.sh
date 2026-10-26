#!/usr/bin/env bash
# Launches the RL model as a UCI engine with the project's venv.
# Set MODEL_PATH to play a different checkpoint.
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
RL_DIR="$REPO/reinforcement-learning"
exec "$REPO/.venv/bin/python" "$RL_DIR/uci_engine.py" \
    --model-path "${MODEL_PATH:-$RL_DIR/chess_rl_model_v7_se.pt}"
