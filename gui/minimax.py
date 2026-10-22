import chess
import random
import time

def terminal(board):
    if board.is_checkmate() or board.is_stalemate():
        return True
    return False


CHECKMATE_SCORE = 1000  # A large value for checkmate

def evaluate_board(board):
    # Check for game-ending conditions
    if board.is_checkmate():
        # If it's checkmate, check who is the winner
        if board.turn == chess.WHITE:
            return -CHECKMATE_SCORE  # White is checkmated, bad for White
        else:
            return CHECKMATE_SCORE   # Black is checkmated, good for White
    elif board.is_stalemate() or board.is_insufficient_material():
        # Draw conditions
        return 0
    num_pieces = len(board.piece_map())
    endgame = num_pieces < 14
    # Basic material count evaluation for non-terminal states
    score = 0
    for (piece, value) in [(chess.KNIGHT, 3), (chess.BISHOP, 3),
                           (chess.ROOK, 5), (chess.QUEEN, 9), (chess.PAWN, 1)]:
        score += len(board.pieces(piece, chess.WHITE)) * value
        score -= len(board.pieces(piece, chess.BLACK)) * value

        king_positions = {
        chess.WHITE: board.king(chess.WHITE),
        chess.BLACK: board.king(chess.BLACK)
    }
    
    for color, king_pos in king_positions.items():
        king_score = 0
        if endgame:
            # Higher score for king in corner or edge during endgame
            if king_pos in [0, 7, 56, 63]:  # Corners
                king_score = 3
            elif king_pos in [1, 6, 8, 15, 48, 55, 57, 62]:  # Edge but not corner
                king_score = 2
            else:
                king_score = 1
        else:
            # In middle game, central control might be more valuable
            # [Add logic for middle game king positioning if desired]
            pass

        if color == chess.WHITE:
            score += king_score
        else:
            score -= king_score

    # Normalize the score to be between -1 and 1
    # Note: This normalization is optional and depends on your specific use case
    normalized_score = score / CHECKMATE_SCORE
    return max(min(normalized_score, 1), -1)

def evaluate_move(board, move):
    """
    Evaluate the given move.
    This function can be as complex as needed, considering various chess strategies.
    """
    # Example: Prioritize captures
    if board.is_capture(move):
        return 10
    if move.promotion:
        return 7
    #board.push(move)
    #if board.is_checkmate():
    #    board.pop()
    #    return 15
    #if board.is_check():
    #    board.pop()
    #    return 5
    #else:
    #    board.pop()
    return 0

def sort_moves(board):
    legal_moves = list(board.legal_moves)
    random.shuffle(legal_moves)
    scored_moves = [(move, evaluate_move(board, move)) for move in legal_moves]
    scored_moves.sort(key=lambda x: x[1], reverse=True)  # Sort by score in descending order
    sorted_moves = [move for move, score in scored_moves]
    return sorted_moves
    

def minimax(board, depth):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board=board):
        return None
    global count_optim
    count_optim = 0
    

    def optimize(board, depth=3, alpha=float('-inf'), beta=float('inf')):
        global count_optim
        count_optim += 1
        if terminal(board=board) or depth <= 0:
            start = time.time()
            score = evaluate_board(board=board), None
            end = time.time()
            return score

        
        #start = time.time()
        legal_moves = sort_moves(board)
        #end = time.time()
        #legal_moves = list(board.legal_moves)

        if board.turn == chess.WHITE:  # Maximizing player
            value = float('-inf')
            optimal_action = None

            for action in legal_moves:
                board.push(action)
                score, _ = optimize(board, depth=depth -1, alpha=alpha, beta=beta)
                board.pop()
                if score > value:
                    value = score
                    optimal_action = action
                alpha = max(alpha, value)
                if alpha >= beta:
                    break
        else:  # Minimizing player
            value = float('inf')
            optimal_action = None
            for action in legal_moves:
                board.push(action)
                score, _ = optimize(board, depth=depth -1, alpha=alpha, beta=beta)
                board.pop()
                if score < value:
                    value = score
                    optimal_action = action
                beta = min(beta, value)
                if alpha >= beta:
                    break
        return value, optimal_action
    
    
    move = optimize(board, depth=depth)[1]
    print(count_optim)
    return move
