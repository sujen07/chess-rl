# Run history

The exact resolved settings (`resolved_config.json`) each past run was launched
with, copied from its checkpoint directory. Paths are shown relative to
`reinforcement-learning/` as they were at the time (`checkpoints_*`, `chess_*_model*.pt`);
those outputs now live in `runs/<run>/` and `models/`.

| File | Trainer | Notes |
|---|---|---|
| `sl.json` | `train_supervised.py` | 128x10 ResNet warm start |
| `sl_se.json` | `train_supervised.py` | same + SE (`se_ratio 4`) |
| `v4.json` | `multiprocess_train_v2.py` (tag `legacy-pipelines`) | first self-play run |
| `v5.json` | `selfplay.py` (then `multiprocess_train_v3.py`) | fast MCTS, playout cap, SL mix |
| `v6.json` | `selfplay.py` | v5 extended from 100k games at 5e-5 / 1e-5 |
| `v6_lr15e-5.json` | `selfplay.py` | v5 extended at a 1.5e-4 peak: regressed to −122 Elo vs v5 |
| `v7_se.json` | `selfplay.py` | fresh run from the SE warm start (`configs/selfplay_v7.yaml`) |
