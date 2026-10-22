#!/usr/bin/env bash
# Runs the model on Lichess through lichess-bot.
#   LICHESS_BOT_TOKEN=lip_xxx ./run_bot.sh            # play
#   LICHESS_BOT_TOKEN=lip_xxx ./run_bot.sh -u         # one-time: upgrade account to BOT
# Extra arguments are passed to lichess-bot (e.g. -v for verbose logs).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
PY="$REPO/.venv/bin/python"

if [[ -z "${LICHESS_BOT_TOKEN:-}" && -f "$HERE/.token" ]]; then
    LICHESS_BOT_TOKEN="$(tr -d '[:space:]' < "$HERE/.token")"
    export LICHESS_BOT_TOKEN
fi
if [[ -z "${LICHESS_BOT_TOKEN:-}" ]]; then
    echo "Set LICHESS_BOT_TOKEN or put the token in $HERE/.token" >&2
    exit 1
fi

if [[ ! -d "$HERE/lichess-bot" ]]; then
    git clone --depth 1 https://github.com/lichess-bot-devs/lichess-bot.git "$HERE/lichess-bot"
fi
if command -v uv >/dev/null; then
    uv pip install -q --python "$PY" -r "$HERE/lichess-bot/requirements.txt"
else
    "$PY" -m pip install -q -r "$HERE/lichess-bot/requirements.txt"
fi

cd "$HERE/lichess-bot"
exec "$PY" lichess-bot.py --config "$HERE/config.yml" "$@"
