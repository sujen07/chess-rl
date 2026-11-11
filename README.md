# chess-rl

An AlphaZero-style chess engine: a residual policy/value network warm-started with
supervised learning on Lichess games, then improved by self-play reinforcement learning
with MCTS. It plays on Lichess as a bot.

- `reinforcement-learning/` – model, search, training, evaluation, UCI engine
- `lichess_bot/` – runs the model on Lichess via lichess-bot (see its README)
- `gui/` – pygame board UI used by `reinforcement-learning/play.py`

## Setup

```bash
uv venv && uv pip install -r requirements.txt
```

## Layout (`reinforcement-learning/`)

| Path | What |
|---|---|
| `chess_environment.py`, `model.py` | board encoding (history planes), move indexing, ResNet / SE / attention networks |
| `fast_mcts.py` | batched PUCT search with bitboard encoding, tree reuse, Gumbel completed-Q targets |
| `prepare_lichess_dataset.py`, `sl_dataset.py` | Lichess PGN dump → packed training shards |
| `train_supervised.py` | supervised warm start |
| `selfplay.py`, `packed_replay.py` | multiprocess self-play RL trainer and GPU-decoded replay buffer |
| `arena.py`, `eval_stockfish.py` | head-to-head Elo with confidence intervals; games vs UCI_Elo-limited Stockfish |
| `uci_engine.py`, `play.py` | UCI engine (used by the Lichess bot); play against it in a window |
| `configs/` | one YAML per run; `configs/history/` holds the exact settings of every past run |
| `experiments/` | self-contained studies with their own write-ups (architecture sweep, weight-space editing) |
| `results/` | small committed evidence (loss curves, Stockfish game records) |
| `runs/`, `models/`, `data/` | checkpoints, logs, final weights and datasets (gitignored) |

The v1–v4 trainers (`train.py`, `multiprocess_train*.py`, pure-Python MCTS) are kept at
the git tag `legacy-pipelines`.

## Reproduce

All commands run from `reinforcement-learning/`.

```bash
# 1. Data: download a Lichess monthly dump and pack it into shards under data/lichess_sl
python prepare_lichess_dataset.py --help

# 2. Supervised warm start (writes models/chess_sl_model_se.pt)
python train_supervised.py --config configs/supervised_se.yaml

# 3. Self-play RL (resumes automatically from runs/<run>/latest.pt)
python selfplay.py --config configs/selfplay_v8.yaml

# 4. Evaluate
python arena.py models/chess_rl_model_v8_se.pt models/chess_rl_model_v7_se.pt --pairs 100
python eval_stockfish.py models/chess_rl_model_v8_se.pt --stockfish /path/to/stockfish

# Tests
python -m unittest test_core test_fast_mcts
```

Any config value can be overridden on the command line, for example `--episodes 1000`.

## Runs

| Run | Config | Change |
|---|---|---|
| v4 | `configs/history/v4.json` | first multiprocess self-play from the SL net |
| v5 | `configs/history/v5.json` | fast MCTS, playout cap randomization, resignation, SL mix |
| v6 | `configs/history/v6.json` | v5 extended at a lower LR (a 1.5e-4 peak regressed −122 Elo) |
| v7 | `configs/selfplay_v7.yaml` | fresh run from the SE warm start, 600-sim targets |
| v8 | `configs/selfplay_v8.yaml` | completed-Q policy targets, higher LR, lighter SL anchor |
