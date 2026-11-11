"""Targeted eval: how reliably does a model convert trivially won endgames?

Random KQK / KRK / KQPK / KRPK / KRRK starts (strong side to move, no
immediate capture of the strong piece). The model plays both sides with MCTS,
so the only question is whether the strong side delivers mate. A non-mate
(move cap, repetition, 50-move rule, stalemate) is a failed conversion.

Usage: python experiments/weight_space/conversion_eval.py MODEL.pt [--n 40] [--sims 200]
Run it on both original.pt and edited.pt with the same --seed.
"""

import argparse
import random
import statistics
import sys
from collections import Counter
from pathlib import Path

import chess
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from arena import load_policy, play_games_mcts  # noqa: E402
from fast_mcts import LocalEvaluator  # noqa: E402

SETS = {
    "KQK": [chess.QUEEN],
    "KRK": [chess.ROOK],
    "KQPK": [chess.QUEEN, chess.PAWN],
    "KRPK": [chess.ROOK, chess.PAWN],
    "KRRK": [chess.ROOK, chess.ROOK],
}


def start(rng, pieces, strong):
    while True:
        b = chess.Board(None)
        sq = rng.sample(range(64), 2 + len(pieces))
        b.set_piece_at(sq[0], chess.Piece(chess.KING, strong))
        b.set_piece_at(sq[1], chess.Piece(chess.KING, not strong))
        if any(p == chess.PAWN and chess.square_rank(s) in (0, 7) for s, p in zip(sq[2:], pieces)):
            continue
        for s, p in zip(sq[2:], pieces):
            b.set_piece_at(s, chess.Piece(p, strong))
        b.turn = strong
        if not b.is_valid() or b.is_game_over():
            continue
        # The lone king must not be able to capture anything on its first move.
        b.push(chess.Move.null())
        hanging = any(b.is_capture(m) for m in b.legal_moves)
        b.pop()
        if not hanging:
            return b


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model")
    p.add_argument("--n", type=int, default=40, help="starts per material set")
    p.add_argument("--sims", type=int, default=200)
    p.add_argument("--max-moves", type=int, default=120, help="plies before giving up")
    p.add_argument("--seed", type=int, default=99)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluators = {"m": LocalEvaluator(load_policy(args.model, device), device)}
    rng = random.Random(args.seed)
    boards, labels = [], []
    for name, pieces in SETS.items():
        for k in range(args.n):
            strong = chess.WHITE if k % 2 == 0 else chess.BLACK
            boards.append(start(rng, pieces, strong))
            labels.append((name, strong))
    outcomes = play_games_mcts(
        ["m"] * len(boards), ["m"] * len(boards), [b.copy() for b in boards],
        evaluators, args.sims, args.max_moves, 1.41, 0.25,
    )
    print(f"{args.model}  sims={args.sims}  max_plies={args.max_moves}")
    for name in SETS:
        rows = [(o, s) for o, (n, s) in zip(outcomes, labels) if n == name]
        mates = [o[2] for o, s in rows
                 if o[1] == "CHECKMATE" and o[0] == ("1-0" if s == chess.WHITE else "0-1")]
        fails = Counter(o[1] for o, s in rows if o[1] != "CHECKMATE")
        avg = f"{statistics.mean(mates):.1f}" if mates else "-"
        print(f"  {name:5s} mated {len(mates):3d}/{len(rows)}  avg plies to mate {avg:>6s}  failures {dict(fails)}")
    total = sum(1 for o, (n, s) in zip(outcomes, labels)
                if o[1] == "CHECKMATE" and o[0] == ("1-0" if s == chess.WHITE else "0-1"))
    print(f"  TOTAL conversion {total}/{len(boards)}")


if __name__ == "__main__":
    main()
