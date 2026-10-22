"""Reproducible checkpoint arena: batched MCTS play (default) or greedy policy.

Greedy play only measures the policy head; RL mostly improves the value head
and play under search, so compare checkpoints with --sims > 0.
"""

import argparse
import math
import random
import shutil
import statistics
from collections import Counter
from pathlib import Path

import chess
import numpy as np
import torch

from chess_environment import (
    ACTION_VERSION,
    ENCODING_VERSION,
    POLICY_SIZE,
    board_to_tensor,
    tensor_to_move,
)
from fast_mcts import (
    LocalEvaluator,
    Node,
    advance,
    evaluate_leaves,
    gather_leaf,
    is_game_over,
    visit_distribution,
)
from model import MODEL_VERSION, PolicyNetwork


def load_policy(path, device):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("model_version") != MODEL_VERSION:
        raise ValueError(f"{path} is not a Chess RL v2 checkpoint")
    if (
        checkpoint.get("encoding_version") != ENCODING_VERSION
        or checkpoint.get("action_version") != ACTION_VERSION
        or checkpoint.get("policy_size") != POLICY_SIZE
    ):
        raise ValueError(f"{path} uses an incompatible representation")
    policy = PolicyNetwork(**checkpoint["model_config"])
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.to(device).eval()
    return policy


@torch.no_grad()
def greedy_move(policy, board, device):
    logits, _ = policy(board_to_tensor(board).to(device))
    return tensor_to_move(logits[0], board)


def opening_board(seed, opening_plies):
    rng = random.Random(seed)
    board = chess.Board()
    for _ in range(opening_plies):
        moves = list(board.legal_moves)
        if not moves:
            break
        board.push(rng.choice(moves))
    return board


def play_game(white, black, board, device, max_moves):
    plies = 0
    while not board.is_game_over(claim_draw=True) and plies < max_moves:
        policy = white if board.turn == chess.WHITE else black
        board.push(greedy_move(policy, board, device))
        plies += 1
    if board.is_game_over(claim_draw=True):
        outcome = board.outcome(claim_draw=True)
        return board.result(claim_draw=True), outcome.termination.name, plies
    return "1/2-1/2", "MOVE_CAP", plies


def play_games_mcts(white_of, black_of, boards, evaluators, sims, max_moves,
                    c_puct, fpu_reduction):
    """Play every board to the end concurrently with batched MCTS.

    white_of[i] / black_of[i] name the evaluator for each side of game i.
    Each side keeps its own tree, advanced by both players' moves.
    """
    n = len(boards)
    trees = [{chess.WHITE: None, chess.BLACK: None} for _ in range(n)]
    plies = [0] * n
    outcomes = [None] * n

    def owner(i):
        return white_of[i] if boards[i].turn == chess.WHITE else black_of[i]

    while True:
        active = [i for i in range(n) if outcomes[i] is None]
        if not active:
            return outcomes
        leaves = {name: [] for name in evaluators}
        for i in active:
            side = boards[i].turn
            if trees[i][side] is None:
                trees[i][side] = Node.root(boards[i])
            root = trees[i][side]
            if root.visits <= sims:
                leaf = gather_leaf(root, c_puct, fpu_reduction)
                if leaf is not None:
                    leaves[owner(i)].append(leaf)
        for name, evaluator in evaluators.items():
            evaluate_leaves(evaluator, leaves[name])

        for i in active:
            board = boards[i]
            root = trees[i][board.turn]
            if root.visits <= sims:
                continue
            edges, probs = visit_distribution(root, temperature=0)
            move = root.moves[edges[int(np.argmax(probs))]]
            board.push(move)
            plies[i] += 1
            for side in (chess.WHITE, chess.BLACK):
                trees[i][side] = advance(trees[i][side], move)
            if is_game_over(board):
                outcome = board.outcome(claim_draw=True)
                outcomes[i] = (board.result(claim_draw=True), outcome.termination.name, plies[i])
            elif plies[i] >= max_moves:
                outcomes[i] = ("1/2-1/2", "MOVE_CAP", plies[i])


def score_for_candidate(result, candidate_is_white):
    if result == "1/2-1/2":
        return 0.5
    candidate_won = (result == "1-0") == candidate_is_white
    return 1.0 if candidate_won else 0.0


def elo_from_score(score):
    score = min(max(score, 1e-6), 1 - 1e-6)
    return 400 * math.log10(score / (1 - score))


def main():
    parser = argparse.ArgumentParser(description="Chess RL v2 checkpoint arena")
    parser.add_argument("candidate")
    parser.add_argument("baseline")
    parser.add_argument("--pairs", type=int, default=50)
    parser.add_argument("--opening-plies", type=int, default=6)
    parser.add_argument("--max-moves", type=int, default=240)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--promote-winner", type=str, default=None)
    parser.add_argument(
        "--sims", type=int, default=100,
        help="MCTS simulations per move; 0 plays the greedy policy head",
    )
    parser.add_argument("--c-puct", type=float, default=1.41)
    parser.add_argument("--fpu-reduction", type=float, default=0.25)
    args = parser.parse_args()
    if args.pairs < 1:
        parser.error("--pairs must be at least 1")
    if args.max_moves < 1:
        parser.error("--max-moves must be at least 1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate = load_policy(args.candidate, device)
    baseline = load_policy(args.baseline, device)
    scores = []
    pair_scores = []
    terminations = Counter()
    lengths = []

    games = [
        (pair, candidate_is_white)
        for pair in range(args.pairs)
        for candidate_is_white in (True, False)
    ]
    starts = [opening_board(args.seed + pair, args.opening_plies) for pair in range(args.pairs)]
    if args.sims > 0:
        evaluators = {
            "candidate": LocalEvaluator(candidate, device),
            "baseline": LocalEvaluator(baseline, device),
        }
        outcomes = play_games_mcts(
            ["candidate" if c else "baseline" for _, c in games],
            ["baseline" if c else "candidate" for _, c in games],
            [starts[pair].copy() for pair, _ in games],
            evaluators, args.sims, args.max_moves, args.c_puct, args.fpu_reduction,
        )
    else:
        outcomes = []
        for pair, candidate_is_white in games:
            white = candidate if candidate_is_white else baseline
            black = baseline if candidate_is_white else candidate
            outcomes.append(
                play_game(white, black, starts[pair].copy(), device, args.max_moves)
            )

    for pair in range(args.pairs):
        current_pair_scores = []
        for candidate_is_white in (True, False):
            result, termination, plies = outcomes[2 * pair + (0 if candidate_is_white else 1)]
            game_score = score_for_candidate(result, candidate_is_white)
            scores.append(game_score)
            current_pair_scores.append(game_score)
            terminations[termination] += 1
            lengths.append(plies)
        pair_scores.append(sum(current_pair_scores) / len(current_pair_scores))

    score = sum(scores) / len(scores)
    standard_error = (
        statistics.stdev(pair_scores) / math.sqrt(len(pair_scores))
        if len(pair_scores) > 1 else 0.5
    )
    low_score = max(1e-6, score - 1.96 * standard_error)
    high_score = min(1 - 1e-6, score + 1.96 * standard_error)
    print(
        f"candidate score={score:.3f} ({sum(scores):.1f}/{len(scores)}) | "
        f"elo={elo_from_score(score):+.0f} "
        f"95% CI [{elo_from_score(low_score):+.0f}, {elo_from_score(high_score):+.0f}] | "
        f"avg_plies={sum(lengths) / len(lengths):.1f}"
    )
    print("terminations:", ", ".join(f"{k}={v}" for k, v in terminations.items()))

    if args.promote_winner and len(pair_scores) >= 20 and low_score > 0.5:
        destination = Path(args.promote_winner)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.candidate, destination)
        print(f"Promoted candidate to {destination}")


if __name__ == "__main__":
    main()
