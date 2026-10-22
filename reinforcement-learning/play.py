"""Play interactively against a trained RL policy (MCTS by default; --sims 0 for greedy)."""
import sys
import time
from pathlib import Path

import chess
import numpy as np
import torch

from chess_environment import ACTION_VERSION, ENCODING_VERSION, POLICY_SIZE
from model import MODEL_VERSION, PolicyNetwork
from chess_environment import board_to_tensor, tensor_to_move
from fast_mcts import LocalEvaluator, Node, evaluate_leaves, gather_leaf, visit_distribution

RL_DIR = Path(__file__).resolve().parent
CHESS_DIR = RL_DIR.parent / "gui"

if str(CHESS_DIR) not in sys.path:
    sys.path.insert(0, str(CHESS_DIR))

import main as chess_ui  # noqa: E402 (needs sys.path set up first)
import pygame  # noqa: E402


def load_policy(model_path, device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    if ckpt.get("model_version") != MODEL_VERSION:
        raise ValueError("This player requires a Chess RL v2 model")
    if (
        ckpt.get("encoding_version") != ENCODING_VERSION
        or ckpt.get("action_version") != ACTION_VERSION
        or ckpt.get("policy_size") != POLICY_SIZE
    ):
        raise ValueError("Model representation is incompatible with this player")
    policy = PolicyNetwork(**ckpt["model_config"])
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.to(device)
    policy.eval()
    return policy


@torch.no_grad()
def agent_move(policy, board, device):
    """Greedy move straight from the policy head — no search."""
    x = board_to_tensor(board).to(device)
    logits, _ = policy(x)
    return tensor_to_move(logits, board)


def mcts_move(evaluator, board, sims, c_puct=1.41, fpu_reduction=0.25):
    """Pick the most-visited move after `sims` MCTS simulations."""
    root = Node.root(board)
    while root.visits <= sims:
        leaf = gather_leaf(root, c_puct, fpu_reduction)
        if leaf is not None:
            evaluate_leaves(evaluator, [leaf])
    edges, probs = visit_distribution(root, temperature=0)
    return root.moves[edges[int(np.argmax(probs))]]


def select_color_screen(screen):
    """Let the user pick a side to play as before the game starts."""
    title_font = pygame.font.Font(None, 48)
    button_font = pygame.font.Font(None, 36)
    white_button = pygame.Rect(150, 300, 130, 50)
    black_button = pygame.Rect(320, 300, 130, 50)

    while True:
        screen.fill(pygame.Color("white"))

        title = title_font.render("Play vs RL Agent", True, pygame.Color("black"))
        screen.blit(title, (150, 200))

        pygame.draw.rect(screen, pygame.Color("grey"), white_button)
        pygame.draw.rect(screen, pygame.Color("grey"), black_button)
        white_text = button_font.render("White", True, pygame.Color("black"))
        black_text = button_font.render("Black", True, pygame.Color("black"))
        screen.blit(white_text, (white_button.x + 25, white_button.y + 10))
        screen.blit(black_text, (black_button.x + 25, black_button.y + 10))

        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if white_button.collidepoint(x, y):
                    return chess.WHITE
                if black_button.collidepoint(x, y):
                    return chess.BLACK


def run_game(policy, device, player=chess.WHITE, sims=800):
    evaluator = LocalEvaluator(policy, device)
    screen = chess_ui.screen
    pieces = chess_ui.pieces
    last_ai_move = None
    board = chess.Board()

    selected_piece = None
    possible_moves = []
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN and player == board.turn:
                x, y = event.pos
                if player == chess.WHITE:
                    col = x // 75
                    row = 7 - (y // 75)
                else:
                    col = 7 - (x // 75)
                    row = y // 75
                square = chess.square(col, row)

                if selected_piece is None:
                    piece = board.piece_at(square)
                    if piece and piece.color == board.turn:
                        selected_piece = square
                        possible_moves = [move for move in board.legal_moves if move.from_square == selected_piece]
                else:
                    if board.piece_at(selected_piece).piece_type == chess.PAWN and chess.square_rank(square) in [0, 7]:
                        move = chess.Move(selected_piece, square, promotion=chess.QUEEN)
                    else:
                        move = chess.Move(selected_piece, square)
                    if move in possible_moves:
                        board.push(move)
                    selected_piece = None
                    possible_moves = []

        if running and player != board.turn and not board.is_game_over(claim_draw=True):
            start = time.time()
            if sims > 0:
                move = mcts_move(evaluator, board, sims)
            else:
                move = agent_move(policy, board, device)
            board.push(move)
            last_ai_move = move
            print("Agent move time:", time.time() - start)

        if board.is_game_over(claim_draw=True):
            print(f"Game over ({board.outcome(claim_draw=True).termination.name}): "
                  f"{board.result(claim_draw=True)}")
            running = False

        chess_ui.draw_board_and_pieces(
            screen, board, pieces,
            selected_piece=selected_piece,
            blink_timer=1,
            last_ai_move=last_ai_move,
            possible_moves=possible_moves,
            player=player,
        )
        pygame.display.flip()
        time.sleep(0.3)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Play chess against the trained RL policy.")
    parser.add_argument("--model-path", type=str,
                         default=str(RL_DIR / "chess_rl_model_v5.pt"),
                         help="checkpoint (.pt) or raw state_dict to load")
    parser.add_argument("--sims", type=int, default=800,
                        help="MCTS simulations per agent move; 0 plays the greedy policy head")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    policy = load_policy(args.model_path, device)
    print(f"Loaded policy from {args.model_path} on {device}")

    screen = chess_ui.screen
    pygame.display.set_caption("Chess — vs RL Agent")
    while True:
        player = select_color_screen(screen)
        run_game(policy, device, player=player, sims=args.sims)


if __name__ == "__main__":
    main()
