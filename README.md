# chess-rl

A chess engine trained with supervised learning on Lichess games followed by
self-play reinforcement learning with MCTS, plus a bot that plays it on Lichess.

- `reinforcement-learning/` – model, MCTS, training (supervised + self-play), evaluation, UCI engine
- `lichess_bot/` – runs the model on Lichess via lichess-bot (see its README)
- `gui/` – pygame board UI used by `reinforcement-learning/play.py` (copied from minimax-ai-player)

## Setup

```bash
uv venv && uv pip install -r requirements.txt
```
