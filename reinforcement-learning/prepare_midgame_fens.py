#!/usr/bin/env python3
"""Build a midgame FEN start-position pool from Lichess PGNs.

Reuses the same game filters as prepare_lichess_dataset.py, but only keeps a
few midgame FENs per game for seeding self-play.

Example:
  python prepare_midgame_fens.py \\
    --pgn data/lichess_raw/lichess_db_standard_rated_2025-01.pgn.zst \\
    --min-elo 2000 \\
    --min-ply 25 \\
    --max-ply 50 \\
    --per-game 2 \\
    --max-fens 50000 \\
    --out data/midgame_fens.pt
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import chess
import chess.pgn
import torch

from prepare_lichess_dataset import (
    download_month,
    game_passes_filters,
    open_pgn_stream,
    resolve_buckets,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--pgn",
        type=Path,
        help="Local .pgn or .pgn.zst dump",
    )
    source.add_argument(
        "--month",
        help="YYYY-MM dump to download into --download-dir",
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=Path("data/lichess_raw"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/midgame_fens.pt"),
        help="Output .pt file with a list of FEN strings (default: data/midgame_fens.pt)",
    )
    parser.add_argument("--min-elo", type=int, default=2000)
    parser.add_argument(
        "--require-both-rated",
        action="store_true",
        help="Require both players to meet --min-elo",
    )
    parser.add_argument(
        "--time-controls",
        default="blitz,rapid,classical",
        help="Comma-separated buckets: bullet,blitz,rapid,classical (or 'all')",
    )
    parser.add_argument(
        "--min-ply",
        type=int,
        default=25,
        help="Earliest half-move index to sample (0-based, before the move)",
    )
    parser.add_argument(
        "--max-ply",
        type=int,
        default=50,
        help="Latest half-move index to sample (inclusive)",
    )
    parser.add_argument(
        "--min-pieces",
        type=int,
        default=12,
        help="Skip positions with fewer pieces than this",
    )
    parser.add_argument(
        "--max-pieces",
        type=int,
        default=24,
        help="Skip positions with more pieces than this",
    )
    parser.add_argument(
        "--per-game",
        type=int,
        default=2,
        help="Max midgame FENs to keep from each accepted game",
    )
    parser.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="Stop after this many accepted games",
    )
    parser.add_argument(
        "--max-fens",
        type=int,
        default=50_000,
        help="Stop after collecting this many FENs",
    )
    parser.add_argument(
        "--max-plies",
        type=int,
        default=500,
        help="Skip games longer than this many half-moves",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=1000)
    return parser.parse_args()


def midgame_fens_from_game(game, args, rng: random.Random) -> list[str]:
    """Replay one game and return up to --per-game midgame FENs."""
    board = game.board()
    candidates: list[str] = []

    for ply, move_node in enumerate(game.mainline()):
        move = move_node.move
        if move not in board.legal_moves:
            return []

        if args.min_ply <= ply <= args.max_ply and not board.is_game_over(claim_draw=True):
            num_pieces = len(board.piece_map())
            if args.min_pieces <= num_pieces <= args.max_pieces and any(board.legal_moves):
                candidates.append(board.fen())

        board.push(move)
        if ply + 1 > args.max_plies:
            break

    if not candidates:
        return []
    if len(candidates) <= args.per_game:
        return candidates
    return rng.sample(candidates, args.per_game)


def main():
    args = parse_args()
    if args.min_ply < 0 or args.max_ply < args.min_ply:
        raise SystemExit("--min-ply/--max-ply must satisfy 0 <= min_ply <= max_ply")
    if args.per_game < 1:
        raise SystemExit("--per-game must be at least 1")
    if args.min_pieces < 2 or args.max_pieces < args.min_pieces:
        raise SystemExit("invalid --min-pieces/--max-pieces")

    allowed_buckets = resolve_buckets(args.time_controls)
    rng = random.Random(args.seed)

    if args.month:
        pgn_path = download_month(args.month, args.download_dir)
    else:
        pgn_path = args.pgn
        if not pgn_path.exists():
            raise SystemExit(f"PGN not found: {pgn_path}")

    stream, raw_handle = open_pgn_stream(pgn_path)
    fens: list[str] = []
    seen: set[str] = set()
    scanned = accepted = skipped = 0

    try:
        while True:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            scanned += 1

            if scanned % args.progress_every == 0:
                print(
                    f"scanned={scanned:,} accepted={accepted:,} "
                    f"fens={len(fens):,} skipped={skipped:,}",
                    file=sys.stderr,
                )

            if not game_passes_filters(game, args, allowed_buckets):
                skipped += 1
                continue

            ply_count = sum(1 for _ in game.mainline())
            if ply_count == 0 or ply_count > args.max_plies:
                skipped += 1
                continue
            # Need enough length to reach the midgame window.
            if ply_count <= args.min_ply:
                skipped += 1
                continue

            picked = midgame_fens_from_game(game, args, rng)
            if not picked:
                skipped += 1
                continue

            accepted += 1
            for fen in picked:
                if fen in seen:
                    continue
                seen.add(fen)
                fens.append(fen)
                if args.max_fens is not None and len(fens) >= args.max_fens:
                    break

            if args.max_fens is not None and len(fens) >= args.max_fens:
                break
            if args.max_games is not None and accepted >= args.max_games:
                break
    finally:
        stream.close()
        if raw_handle is not None:
            raw_handle.close()

    if not fens:
        raise SystemExit("No midgame FENs collected; relax filters or check the PGN.")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "fens": fens,
        "source_pgn": str(pgn_path),
        "min_elo": args.min_elo,
        "require_both_rated": args.require_both_rated,
        "time_controls": args.time_controls,
        "min_ply": args.min_ply,
        "max_ply": args.max_ply,
        "min_pieces": args.min_pieces,
        "max_pieces": args.max_pieces,
        "per_game": args.per_game,
        "games_scanned": scanned,
        "games_accepted": accepted,
        "games_skipped": skipped,
        "num_fens": len(fens),
        "seed": args.seed,
    }
    torch.save(payload, args.out)

    print()
    print(f"Done. accepted_games={accepted:,} fens={len(fens):,}")
    print(f"Wrote {args.out}")
    print(
        "Next: pass this file to multiprocess_train_v2.py via "
        "--midgame-fens and --midgame-prob."
    )


if __name__ == "__main__":
    main()
