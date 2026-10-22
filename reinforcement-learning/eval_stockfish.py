"""Estimate a checkpoint's Elo by playing strength-limited Stockfish.

Stockfish's UCI_Elo is calibrated against CCRL 40/4-style ratings, so the
estimate is on that scale (not Lichess/Chess.com). The model plays batched
MCTS with tree reuse; each game gets its own single-threaded Stockfish.
"""

import argparse
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import chess
import chess.engine
import chess.pgn
import numpy as np
import torch

from arena import elo_from_score, load_policy, score_for_candidate
from fast_mcts import LocalEvaluator, Node, advance, is_game_over, run_searches, visit_distribution

OPENINGS = [
    "e2e4 e7e5 g1f3 b8c6 f1b5",          # Ruy Lopez
    "e2e4 c7c5 g1f3 d7d6 d2d4",          # Sicilian
    "e2e4 e7e6 d2d4 d7d5",               # French
    "e2e4 c7c6 d2d4 d7d5",               # Caro-Kann
    "d2d4 d7d5 c2c4 e7e6 b1c3",          # QGD
    "d2d4 g8f6 c2c4 g7g6 b1c3 f8g7",     # King's Indian
    "d2d4 g8f6 c2c4 e7e6 b1c3 f8b4",     # Nimzo-Indian
    "c2c4 e7e5 b1c3 g8f6",               # English
    "g1f3 d7d5 g2g3 g8f6 f1g2",          # Reti
    "e2e4 e7e5 g1f3 b8c6 f1c4 f8c5",     # Italian
]


def start_board(line):
    board = chess.Board()
    for uci in line.split():
        board.push_uci(uci)
    return board


def open_engine(path, elo):
    engine = chess.engine.SimpleEngine.popen_uci(path)
    engine.configure({"Threads": 1, "Hash": 16, "UCI_LimitStrength": True, "UCI_Elo": elo})
    return engine


def play_level(evaluator, sf_path, elo, games, sims, movetime, max_plies, c_puct, fpu):
    boards = [start_board(OPENINGS[(g // 2) % len(OPENINGS)]) for g in range(games)]
    model_white = [g % 2 == 0 for g in range(games)]
    starts = [b.copy() for b in boards]
    trees = [None] * games
    outcomes = [None] * games
    engines = [open_engine(sf_path, elo) for _ in range(games)]

    def push(i, move):
        boards[i].push(move)
        trees[i] = advance(trees[i], move)
        if is_game_over(boards[i]):
            outcomes[i] = boards[i].result(claim_draw=True)
        elif boards[i].ply() - starts[i].ply() >= max_plies:
            outcomes[i] = "1/2-1/2"

    def sf_move(i):
        return engines[i].play(boards[i], chess.engine.Limit(time=movetime)).move

    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            while True:
                active = [i for i in range(games) if outcomes[i] is None]
                if not active:
                    break
                mine = [i for i in active if (boards[i].turn == chess.WHITE) == model_white[i]]
                theirs = [i for i in active if i not in mine]
                sf_futures = {i: pool.submit(sf_move, i) for i in theirs}
                if mine:
                    for i in mine:
                        if trees[i] is None:
                            trees[i] = Node.root(boards[i])
                    roots = [trees[i] for i in mine]
                    run_searches(evaluator, roots, [sims + 1] * len(roots),
                                 c_puct=c_puct, fpu_reduction=fpu)
                    for i, root in zip(mine, roots):
                        edges, probs = visit_distribution(root, temperature=0)
                        push(i, root.moves[edges[int(np.argmax(probs))]])
                for i, fut in sf_futures.items():
                    push(i, fut.result())
    finally:
        for e in engines:
            e.quit()

    games_out = []
    for i in range(games):
        game = chess.pgn.Game.from_board(boards[i])
        game.headers["White"] = "model" if model_white[i] else f"Stockfish{elo}"
        game.headers["Black"] = f"Stockfish{elo}" if model_white[i] else "model"
        game.headers["Result"] = outcomes[i]
        games_out.append(game)
    scores = [score_for_candidate(outcomes[i], model_white[i]) for i in range(games)]
    return scores, games_out


def mle_elo(results):
    """Maximum-likelihood rating given (opponent_elo, score) pairs."""
    lo, hi = 0.0, 4000.0
    for _ in range(100):
        r = (lo + hi) / 2
        grad = sum(s - 1 / (1 + 10 ** ((o - r) / 400)) for o, s in results)
        lo, hi = (r, hi) if grad > 0 else (lo, r)
    return (lo + hi) / 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model")
    p.add_argument("--stockfish", required=True)
    p.add_argument("--elos", type=int, nargs="+", default=[1400, 1700, 2000, 2300])
    p.add_argument("--games", type=int, default=20, help="games per level (even)")
    p.add_argument("--sims", type=int, default=400)
    p.add_argument("--movetime", type=float, default=0.2)
    p.add_argument("--max-plies", type=int, default=300)
    p.add_argument("--c-puct", type=float, default=1.41)
    p.add_argument("--fpu-reduction", type=float, default=0.25)
    p.add_argument("--pgn", default="stockfish_eval.pgn")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluator = LocalEvaluator(load_policy(args.model, device), device)
    all_results = []
    with open(args.pgn, "w") as pgn:
        for elo in args.elos:
            scores, games = play_level(evaluator, args.stockfish, elo, args.games, args.sims,
                                       args.movetime, args.max_plies, args.c_puct,
                                       args.fpu_reduction)
            for g in games:
                print(g, file=pgn, end="\n\n")
            pgn.flush()
            all_results += [(elo, s) for s in scores]
            score = sum(scores) / len(scores)
            w, d = scores.count(1.0), scores.count(0.5)
            print(f"SF {elo}: +{w} ={d} -{len(scores) - w - d}  score={score:.3f}  "
                  f"perf={elo + elo_from_score(score):.0f}", flush=True)

    est = mle_elo(all_results)
    info = sum(
        (math.log(10) / 400) ** 2 * (q := 1 / (1 + 10 ** ((o - est) / 400))) * (1 - q)
        for o, _ in all_results
    )
    print(f"Estimated Elo (CCRL-ish scale): {est:.0f} ± {1.96 / math.sqrt(info):.0f} (95%)")


if __name__ == "__main__":
    main()
