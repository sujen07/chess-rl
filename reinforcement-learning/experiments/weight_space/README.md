# Weight-space programming on the v7 SE model

An experiment in the style of *Sparks of Weight-Space Programming* (Takehi, Teoh,
Zhang, Khattab; MIT CSAIL, Nov 2026). An LLM (Claude Opus 5.5) reads the network's
weights, finds a fault, and hand-writes an edit. There is no gradient descent and no
training loop.

## Sandbox rules (`sandbox.py`)

| | |
|---|---|
| Model | `original.pt` = `models/chess_rl_model_v7_se.pt` (sha256 `8fc4acc9…`), read-only |
| Reading weights | free (`load_weights`) |
| Running the net | metered: `probe()` / `probe_replay()`, ≤512 positions per pass, **budget 50 passes**, logged to `ledger.jsonl` |
| Data | v7 replay buffer as "practice problems" (positions + 600-sim search targets). Used for diagnosis only, never for gradient training |
| Not allowed | gradients, optimizers, MCTS, games, arena runs, Stockfish |
| Edits | via `save_weights()`, logged with diffs to `edits.jsonl` |
| Held-out eval | `eval.sh` / `conversion_eval.py`, run by the user after the edit was frozen |

The rules were enforced by convention and logging, not by OS isolation.
Analysis was done with offline numpy on activations captured during metered passes,
for example applying a linear readout to captured `value_conv` outputs. Every edit was
then checked with real metered passes.

**Used: 28 of 50 passes.** The edit was frozen with 22 passes left.

## What I found

1. **Calibration in its own distribution is near-perfect.** On 3,072 recent self-play
   positions the best policy temperature is 1.05 (cross-entropy gain 0.1%) and the
   value slope is 1.04. Every tactic category matches 600-sim search within about 5%:
   checks, captures of free and defended pieces, moves that hang a piece, castling,
   promotions and mate-in-1. Coarse re-tuning would be cosmetic.
2. **The stalemate losses vs Stockfish were not lost wins.** In both games that ended in
   stalemate, the net valued the alternatives at ≈0. The positions (KRP v KR with the
   defending king in front, and a bishop fortress) look like theoretical draws.
3. **The real fault: no progress signal in won endgames.** Self-play resigns at −0.9, so
   trivially won endings are rarely played out. In KQK and KRK the value head is flat
   with respect to how close the lone king is to the edge (correlation ≈0). With the
   defending side to move it reads ≈−0.75 instead of −1. Search has nothing pulling it
   toward mate.
4. **Spare capacity.** 203 of the 256 value-head hidden units (`value_fc.0`) never fire,
   on practice positions or on endgames. Several `value_conv` channels encode king
   location: 1, 6, 7, 12, 16 and 24 track the enemy king; 0, 8 and 14 track our king.

## The edit (v4, final): a "mop-up" circuit in 6 dead value units

Changes are in `value_fc.0` rows 2, 3, 4, 7, 8, 9 plus their biases and output weights:
12,300 weights, **0.33%** of the model. The policy head and all live units are untouched.

For each side (A = strong side to move, B = lone king to move):

- **Gate.** A low-dimensional linear detector over pooled `value_conv` maps: per channel,
  the mean, edge-ring mean and back-rank mean. It was found by whitened difference of
  means (closed form) between won major-piece endgames and normal practice positions,
  plus *hard negatives*: KPK, KNK, KBK, KNNK, KRKR, KQKQ, wrong-bishop + rook pawn,
  KNP-v-KR and pawn endings.
- **Edge readout.** A hand-written centre-to-edge map applied to the king-location
  channels.
- **Circuit.** Paired ReLUs give `gate × edge`:
  `ReLU(g + e) − ReLU(g) = e` while the gate is open, and 0 when it is closed.
  A third unit makes a clamped gate, `ReLU(g) − ReLU(g − 1)`, which carries a constant
  offset so a centre king keeps the original value. Only kings near the edge change.
  The swing is about ±0.7 in pre-tanh units.

Iterations, all logged in `edits.jsonl`:

- **v1** produced a dangerous false positive: K vs K + h-pawn (a dead draw) moved from
  −0.06 to −0.92.
- **v2** refit the gates with hard negatives.
- **v3** overshot: the offset rode on an unbounded gate and saturated tanh.
- **v4** uses a clamped gate.

## Verification inside the sandbox (fresh positions, metered passes)

- **Fresh won endgames.** Value now tracks the lone king's edge distance, with
  correlation 0.3–0.75 in KQK, KRK, KQPK, KRPK, KRRK and KQRK; before it was ≈0.
  KRRK and KQRK were never used to build the circuit.
  - KQK, strong side to move: edge distance 0/1/2 now gives 0.976 / 0.970 / 0.888.
    The original was flat at ≈0.91.
- **Fresh practice split (1,024 positions, never used for building).**
  - Policy logits are bit-identical.
  - Value MSE went from 0.14732 to 0.14738.
  - 3% of positions changed (max |Δv| 0.30); they are bare-king endings, and most moved
    toward the search target.
- **Not fixed.** KBNK and KBBK; the gate does not cover minor-piece mates.

## Prediction (written before any eval)

- **Head-to-head arena (`eval.sh`):** a small effect, about +0 to +20 Elo. The circuit is
  silent until a bare-king ending appears. 100 pairs gives a 95% CI of about ±40 Elo,
  so expect "not significant".
- **Conversion (`conversion_eval.py`):** this is where it should show. Expect a higher
  mate rate and fewer plies to mate in KRK and KQK. KRPK and KQPK should improve less,
  since the pawn usually promotes anyway.
- **Risk:** a missed false positive in some drawn ending could cost a half-point.

## How to evaluate

```bash
cd reinforcement-learning
experiments/weight_space/eval.sh 100 200          # MCTS arena + greedy, edited vs original
../.venv/bin/python experiments/weight_space/conversion_eval.py experiments/weight_space/original.pt
../.venv/bin/python experiments/weight_space/conversion_eval.py experiments/weight_space/edited.pt
```

## Results (held-out, run after freezing)

Raw output is in `results/`.

**Conversion (`conversion_eval.py`, 200 starts, 200 sims, model plays both sides)**

| | KQK | KRK | KQPK | KRPK | KRRK | Total mated |
|---|---|---|---|---|---|---|
| original: avg plies to mate | 18.1 | 33.9 | 12.8 | 19.7 | 17.1 | 200/200 |
| edited: avg plies to mate | 18.3 | 33.9 | 13.4 | 21.4 | 17.6 | 200/200 |

**Arena, edited vs original (100 pairs)**

- MCTS with 200 sims: score 0.500, +0 Elo, 95% CI [−5, +5].
- Greedy policy: 0.500. This is identical by construction, because the edit leaves the
  policy untouched and greedy play never reads the value head.

**Verdict: no improvement, and the premise was wrong.**

- The original never fails to convert. With 200 sims, MCTS finds these mates by search
  alone; a flat value head is not an obstacle at this search depth. My diagnosis looked
  at the raw value head, but the behaviour that matters is value plus search, and I had
  no way to measure that inside the sandbox.
- The edit is slightly *slower* to mate in pawn endings: KRPK took 21.4 plies instead of
  19.7. Likely cause: the edge bonus competes with the pawn-promotion plan.
- The arena is a dead heat with a very tight CI. The circuit rarely fires, so most games
  are presumably identical to original-vs-original, and colour-swapped pairs split 1–1.
- Takeaway, consistent with the paper: a hand edit can only help by removing a real
  obstacle. Here the network was already at its training optimum, and the one
  weight-level gap I found is covered by search. A better sandbox would let the editor
  spend budget on small search-in-the-loop checks, such as a handful of MCTS conversion
  games, not just raw forward passes.
