import chess
import torch
from dataclasses import dataclass

# Versioned AlphaZero-style representation.
ENCODING_VERSION = 2
ACTION_VERSION = 2
HISTORY_LENGTH = 8
MOVE_PLANES = 73
POLICY_SIZE = 64 * MOVE_PLANES
INPUT_CHANNELS = HISTORY_LENGTH * 12 + 7
CHECKMATE_SCORE = 40  # A large value for checkmate

_QUEEN_DIRECTIONS = (
    (0, 1), (1, 1), (1, 0), (1, -1),
    (0, -1), (-1, -1), (-1, 0), (-1, 1),
)
_KNIGHT_DELTAS = (
    (1, 2), (2, 1), (2, -1), (1, -2),
    (-1, -2), (-2, -1), (-2, 1), (-1, 2),
)
_UNDERPROMOTIONS = (chess.KNIGHT, chess.BISHOP, chess.ROOK)


@dataclass(frozen=True)
class PositionRecord:
    """Compact replay representation; FENs are current-first."""

    history_fens: tuple
    repetition_count: int

    @property
    def current_fen(self):
        return self.history_fens[0]

    def board(self):
        return chess.Board(self.current_fen)

    @classmethod
    def from_board(cls, board):
        cursor = board.copy()
        history_fens = []
        for _ in range(HISTORY_LENGTH):
            history_fens.append(cursor.fen())
            if not cursor.move_stack:
                break
            cursor.pop()

        repetition_count = 3 if board.is_repetition(3) else (
            2 if board.is_repetition(2) else 1
        )
        return cls(tuple(history_fens), repetition_count)


@dataclass(frozen=True)
class SparsePolicy:
    indices: torch.Tensor
    probabilities: torch.Tensor


def to_policy_square(square, turn):
    """
    Map a real-board square into the side-to-move policy frame.
    Black to move: vertical flip so the mover's pieces sit on the low ranks.
    """
    return chess.square_mirror(square) if turn == chess.BLACK else square


def _position_record(position):
    return position if isinstance(position, PositionRecord) else PositionRecord.from_board(position)


def board_to_tensor(position):
    """
    Encode eight position frames plus rule state from the current mover's view.

    The seven auxiliary planes are mover/opponent K/Q castling rights, en
    passant target, normalized halfmove clock, and current repetition count.
    """
    record = _position_record(position)
    board = record.board()
    tensor = torch.zeros(1, INPUT_CHANNELS, 8, 8)
    turn = board.turn

    for frame, fen in enumerate(record.history_fens[:HISTORY_LENGTH]):
        historical_board = chess.Board(fen)
        for square, piece in historical_board.piece_map().items():
            channel = frame * 12 + piece.piece_type - 1
            if piece.color != turn:
                channel += 6
            policy_sq = to_policy_square(square, turn)
            tensor[
                0,
                channel,
                chess.square_file(policy_sq),
                chess.square_rank(policy_sq),
            ] = 1

    aux = HISTORY_LENGTH * 12
    opponent = not turn
    castling = (
        board.has_kingside_castling_rights(turn),
        board.has_queenside_castling_rights(turn),
        board.has_kingside_castling_rights(opponent),
        board.has_queenside_castling_rights(opponent),
    )
    for offset, available in enumerate(castling):
        if available:
            tensor[0, aux + offset].fill_(1)

    if board.ep_square is not None:
        ep_square = to_policy_square(board.ep_square, turn)
        tensor[
            0,
            aux + 4,
            chess.square_file(ep_square),
            chess.square_rank(ep_square),
        ] = 1
    tensor[0, aux + 5].fill_(min(board.halfmove_clock, 100) / 100.0)
    tensor[0, aux + 6].fill_(min(record.repetition_count - 1, 2) / 2.0)
    return tensor


def outcome_reward(result):
    """Terminal reward from White's perspective (+1 win, -1 loss, 0 draw).

    Kept on the same [-1, 1] scale as evaluate_board / the shaping rewards so the
    value head isn't crushed by a scale mismatch between terminal and per-step signals.
    """
    if result == "1-0":
        return 1.0
    if result == "0-1":
        return -1.0
    return 0.0


def outcome_reward_for(result, turn):
    """outcome_reward converted to `turn`'s perspective (+1 turn wins, -1 turn loses)."""
    reward = outcome_reward(result)
    return reward if turn == chess.WHITE else -reward


def terminal_value(board):
    """Game value from the perspective of the side to move at this (terminal) board.

    A terminal board has no legal moves, so this is always <= 0: -1 if the side
    to move is checkmated, 0 for any other terminal state (stalemate, draw claims).
    """
    return -1.0 if board.is_checkmate() else 0.0


def tensor_to_board(tensor):
    raise NotImplementedError(
        "The oriented history tensor is intentionally not reversible; "
        "retain PositionRecord for board reconstruction"
    )


def step(board, move):
    board.push(move)
    return board


def move_to_index(move, board):
    """Map a legal move to an oriented AlphaZero 73-plane action."""
    turn = board.turn
    from_sq = to_policy_square(move.from_square, turn)
    to_sq = to_policy_square(move.to_square, turn)
    from_file = chess.square_file(from_sq)
    from_rank = chess.square_rank(from_sq)
    delta = (
        chess.square_file(to_sq) - from_file,
        chess.square_rank(to_sq) - from_rank,
    )

    if move.promotion in _UNDERPROMOTIONS:
        if delta[1] != 1 or delta[0] not in (-1, 0, 1):
            raise ValueError(f"Invalid underpromotion move: {move}")
        piece_offset = _UNDERPROMOTIONS.index(move.promotion)
        move_plane = 64 + piece_offset * 3 + delta[0] + 1
    elif delta in _KNIGHT_DELTAS:
        move_plane = 56 + _KNIGHT_DELTAS.index(delta)
    else:
        distance = max(abs(delta[0]), abs(delta[1]))
        if distance < 1:
            raise ValueError(f"Invalid zero-length move: {move}")
        direction = (delta[0] // distance, delta[1] // distance)
        if direction not in _QUEEN_DIRECTIONS:
            raise ValueError(f"Move is not representable: {move}")
        move_plane = _QUEEN_DIRECTIONS.index(direction) * 7 + distance - 1

    return move_plane * 64 + from_sq


def legal_move_mask(board, device=None, dtype=torch.float32):
    """Binary mask over the policy space: 1 = legal, 0 = illegal."""
    mask = torch.zeros(POLICY_SIZE, device=device, dtype=dtype)
    for move in board.legal_moves:
        mask[move_to_index(move, board)] = 1
    return mask


def masked_policy_logits(tensor, board):
    """Mask illegal move logits to -inf over the 8x8x73 policy space."""
    logits = tensor.reshape(-1)
    if logits.numel() < POLICY_SIZE:
        raise ValueError(f"Expected at least {POLICY_SIZE} logits, got {logits.numel()}")
    logits = logits[:POLICY_SIZE]

    mask = legal_move_mask(board, device=logits.device, dtype=logits.dtype)
    return logits.masked_fill(mask == 0, float("-inf"))


def index_to_move(index, board):
    """Decode an action by matching its unique index among legal moves."""
    for move in board.legal_moves:
        if move_to_index(move, board) == int(index):
            return move
    raise ValueError("No legal move found for index; board may have no legal moves")


def tensor_to_move(tensor, board):
    """Greedy legal move from policy logits (for evaluation / visualization)."""
    masked_logits = masked_policy_logits(tensor, board)
    return index_to_move(masked_logits.argmax().item(), board)


def legal_moves(board):
    return list(board.legal_moves)


def terminal(board):
    return board.is_game_over(claim_draw=True)


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
