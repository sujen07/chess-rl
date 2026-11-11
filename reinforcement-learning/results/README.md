# Results

Small, committed evidence from runs whose full outputs live in `runs/` (gitignored).

- `supervised/` – loss curves of the two supervised warm starts (`configs/supervised*.yaml`).
- `stockfish/` – games against UCI_Elo-limited Stockfish (`eval_stockfish.py`).
  The `_1s` files used 1 s/move for Stockfish. Each level has 10–20 games, so treat
  per-level scores as rough.
- Experiments with their own write-ups: `experiments/arch_sweep/results.md`,
  `experiments/weight_space/README.md`.
