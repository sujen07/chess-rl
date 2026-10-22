"""Packed supervised-learning dataset for the warm start.

The replay format (PositionRecord + SparsePolicy objects) costs ~2 KB per
position and re-parses eight FENs every time a sample is encoded, which caps
supervised training at a few hundred thousand positions. This module stores
each position as ~350 bytes of numpy arrays and rebuilds the exact
board_to_tensor() input on the GPU, so tens of millions of positions fit in
RAM and the training loop is GPU-bound instead of python-chess-bound.

Shard layout (shard_*.npz, one row per position):
  pieces     uint8 [N, 8, 32]  8 history frames x 64 squares, 4-bit piece codes
                               in the mover-oriented frame (0 empty, 1-6 mover
                               P..K, 7-12 opponent P..K), two squares per byte
  aux        uint8 [N, 4]      castling bits, en-passant slot (255 = none),
                               halfmove clock (capped at 100), repetition count
  move       int16 [N]         played move in the 73-plane policy space
  value      float32 [N]       value target from the mover's perspective
  legal_off  int64 [N + 1]     offsets into legal_idx
  legal_idx  int16 [L]         legal-move policy indices, concatenated
"""

from __future__ import annotations

import math
from pathlib import Path

import chess
import numpy as np
import torch
import torch.nn.functional as F

from chess_environment import (
    ACTION_VERSION,
    ENCODING_VERSION,
    HISTORY_LENGTH,
    INPUT_CHANNELS,
    POLICY_SIZE,
    move_to_index,
    outcome_reward_for,
)

PACKED_FORMAT_VERSION = 1
PACKED_KEYS = ("pieces", "aux", "move", "value", "legal_off", "legal_idx")
NO_EP = 255
# Lichess' centipawn -> win-probability model (lichess.org accuracy docs).
LICHESS_WIN_COEF = 0.00368208

# Storage slot p = file * 8 + rank of the policy-frame square, so reshaping a
# frame to [8, 8] lines up with board_to_tensor's tensor[..., file, rank].
_SLOT_TO_SQUARE_WHITE = np.array(
    [(p % 8) * 8 + p // 8 for p in range(64)], dtype=np.int64
)
_SLOT_TO_SQUARE_BLACK = _SLOT_TO_SQUARE_WHITE ^ 56  # vertical mirror
# Swap mover/opponent codes when Black is to move.
_SWAP_COLORS = np.array([0] + list(range(7, 13)) + list(range(1, 7)), dtype=np.uint8)


def _absolute_codes(board):
    """[64] piece codes indexed by real square: 1-6 White P..K, 7-12 Black."""
    codes = np.zeros(64, dtype=np.uint8)
    for piece_type in chess.PIECE_TYPES:
        for square in board.pieces(piece_type, chess.WHITE):
            codes[square] = piece_type
        for square in board.pieces(piece_type, chess.BLACK):
            codes[square] = 6 + piece_type
    return codes


def _slot(square, turn):
    policy_sq = chess.square_mirror(square) if turn == chess.BLACK else square
    return chess.square_file(policy_sq) * 8 + chess.square_rank(policy_sq)


def encode_position(board, history_codes):
    """Pack `board` (with `history_codes`, current-first) like board_to_tensor."""
    pieces, aux = encode_planes(board, history_codes)
    legal = np.array(
        [move_to_index(move, board) for move in board.legal_moves], dtype=np.int16
    )
    return pieces, aux, legal


def encode_planes(board, history_codes):
    """(pieces, aux) half of encode_position(), without the legal-move list."""
    turn = board.turn
    frames = np.zeros((HISTORY_LENGTH, 64), dtype=np.uint8)
    history = np.stack(history_codes[:HISTORY_LENGTH])
    if turn == chess.WHITE:
        frames[: len(history)] = history[:, _SLOT_TO_SQUARE_WHITE]
    else:
        frames[: len(history)] = _SWAP_COLORS[history[:, _SLOT_TO_SQUARE_BLACK]]
    pieces = frames[:, 0::2] | (frames[:, 1::2] << 4)

    opponent = not turn
    castling = (
        int(board.has_kingside_castling_rights(turn))
        | int(board.has_queenside_castling_rights(turn)) << 1
        | int(board.has_kingside_castling_rights(opponent)) << 2
        | int(board.has_queenside_castling_rights(opponent)) << 3
    )
    # board_to_tensor rebuilds the board from a FEN, which only records a
    # legally capturable en-passant square.
    ep = (
        _slot(board.ep_square, turn)
        if board.ep_square is not None and board.has_legal_en_passant()
        else NO_EP
    )
    # A third occurrence implies a second, so skip the costlier check when
    # the position has not repeated at all.
    repetition = (
        (3 if board.is_repetition(3) else 2) if board.is_repetition(2) else 1
    )
    aux = np.array(
        [castling, ep, min(board.halfmove_clock, 100), repetition], dtype=np.uint8
    )
    return pieces, aux


def eval_to_value(pov_score, turn):
    """Lichess %eval (PovScore) -> expected score in [-1, 1] for `turn`."""
    score = pov_score.pov(turn)
    if score.is_mate():
        return 1.0 if score.mate() > 0 else -1.0
    win_prob = 1.0 / (1.0 + math.exp(-LICHESS_WIN_COEF * score.score()))
    return 2.0 * win_prob - 1.0


def encode_game(game, eval_weight=0.0, max_plies=500):
    """Replay a parsed PGN game into packed arrays, or None if unusable.

    Value target = eval_weight * engine_eval + (1 - eval_weight) * game_result
    for positions that have a Lichess %eval annotation, game_result otherwise.
    """
    result = game.headers.get("Result")
    board = game.board()
    history = [_absolute_codes(board)]  # oldest-first
    pieces, aux, moves, values, legal = [], [], [], [], []
    prev_eval = None  # %eval on the previous node = eval of the current board

    for ply, node in enumerate(game.mainline()):
        if ply >= max_plies:
            return None
        move = node.move
        if not board.is_legal(move):
            return None

        packed, aux_row, legal_row = encode_position(board, history[::-1])
        z = outcome_reward_for(result, board.turn)
        if prev_eval is not None and eval_weight > 0:
            target = eval_weight * eval_to_value(prev_eval, board.turn) + (
                1.0 - eval_weight
            ) * z
        else:
            target = z

        pieces.append(packed)
        aux.append(aux_row)
        moves.append(move_to_index(move, board))
        values.append(target)
        legal.append(legal_row)

        board.push(move)
        history.append(_absolute_codes(board))
        if len(history) > HISTORY_LENGTH:
            history.pop(0)
        prev_eval = node.eval()

    if not pieces:
        return None
    lengths = np.array([len(row) for row in legal], dtype=np.int64)
    return {
        "pieces": np.stack(pieces),
        "aux": np.stack(aux),
        "move": np.array(moves, dtype=np.int16),
        "value": np.array(values, dtype=np.float32),
        "legal_off": np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64),
        "legal_idx": np.concatenate(legal),
    }


def concat_packed(chunks):
    """Concatenate encode_game() outputs, rebasing legal-move offsets."""
    legal_off = [np.zeros(1, dtype=np.int64)]
    base = 0
    for chunk in chunks:
        legal_off.append(chunk["legal_off"][1:] + base)
        base += int(chunk["legal_off"][-1])
    packed = {
        key: np.concatenate([c[key] for c in chunks])
        for key in PACKED_KEYS
        if key != "legal_off"
    }
    packed["legal_off"] = np.concatenate(legal_off)
    return packed


def slice_packed(packed, start, stop):
    off = packed["legal_off"]
    return {
        "pieces": packed["pieces"][start:stop],
        "aux": packed["aux"][start:stop],
        "move": packed["move"][start:stop],
        "value": packed["value"][start:stop],
        "legal_off": off[start : stop + 1] - off[start],
        "legal_idx": packed["legal_idx"][off[start] : off[stop]],
    }


def save_packed_shard(path, packed):
    np.savez(
        path,
        format_version=PACKED_FORMAT_VERSION,
        encoding_version=ENCODING_VERSION,
        action_version=ACTION_VERSION,
        **packed,
    )


class PackedDataset:
    """In-memory packed positions with GPU-side batch decoding."""

    def __init__(self, packed):
        self.packed = packed

    def __len__(self):
        return len(self.packed["move"])

    @classmethod
    def load_dir(cls, data_dir, max_positions=None):
        paths = sorted(Path(data_dir).glob("shard_*.npz"))
        if not paths:
            raise FileNotFoundError(
                f"No shard_*.npz files under {data_dir}; "
                "build them with prepare_lichess_dataset.py"
            )
        chunks, total = [], 0
        for path in paths:
            with np.load(path) as shard:
                if (
                    int(shard["format_version"]) != PACKED_FORMAT_VERSION
                    or int(shard["encoding_version"]) != ENCODING_VERSION
                    or int(shard["action_version"]) != ACTION_VERSION
                ):
                    raise ValueError(f"{path} was written for a different encoding")
                chunk = {key: shard[key] for key in PACKED_KEYS}
            if max_positions is not None and total + len(chunk["move"]) > max_positions:
                chunk = slice_packed(chunk, 0, max_positions - total)
            chunks.append(chunk)
            total += len(chunk["move"])
            print(f"  loaded {path.name} ({total:,} positions)")
            if max_positions is not None and total >= max_positions:
                break
        return cls(concat_packed(chunks))

    def split_tail(self, n):
        """(head, tail): hold out the last n positions (whole games, mostly)."""
        total = len(self)
        cut = total - n
        return (
            PackedDataset(slice_packed(self.packed, 0, cut)),
            PackedDataset(slice_packed(self.packed, cut, total)),
        )

    def batch(self, idxs, device):
        """(x, legal_mask, move, value) tensors on `device` for rows `idxs`."""
        p = self.packed
        idxs = np.sort(idxs)
        starts = p["legal_off"][idxs]
        lengths = p["legal_off"][idxs + 1] - starts
        flat = np.concatenate(
            [p["legal_idx"][s : s + n] for s, n in zip(starts, lengths)]
        ).astype(np.int64)
        rows = np.repeat(np.arange(len(idxs)), lengths)

        legal_mask = torch.zeros(len(idxs), POLICY_SIZE, dtype=torch.bool, device=device)
        legal_mask[torch.from_numpy(rows).to(device), torch.from_numpy(flat).to(device)] = True
        x = decode_planes(
            torch.from_numpy(p["pieces"][idxs]).to(device),
            torch.from_numpy(p["aux"][idxs]).to(device),
        )
        move = torch.from_numpy(p["move"][idxs].astype(np.int64)).to(device)
        value = torch.from_numpy(p["value"][idxs]).to(device)
        return x, legal_mask, move, value


def decode_planes(pieces, aux):
    """Inverse of encode_position(): [B, INPUT_CHANNELS, 8, 8] float tensor."""
    b = pieces.shape[0]
    codes = torch.stack((pieces & 15, pieces >> 4), dim=-1).reshape(
        b, HISTORY_LENGTH, 64
    ).long()
    planes = F.one_hot(codes, 13)[..., 1:]  # [B, frame, slot, piece]
    planes = planes.permute(0, 1, 3, 2).reshape(b, HISTORY_LENGTH * 12, 8, 8).float()

    aux = aux.long()
    extra = torch.zeros(b, INPUT_CHANNELS - HISTORY_LENGTH * 12, 64, device=pieces.device)
    for bit in range(4):
        extra[:, bit] = ((aux[:, 0] >> bit) & 1).float().unsqueeze(1)
    has_ep = aux[:, 1] != NO_EP
    extra[has_ep, 4, aux[has_ep, 1]] = 1.0
    # float64 so CUDA's reciprocal-multiply matches board_to_tensor bit-for-bit.
    # MPS has no float64; float32 there is off by at most 1 ulp.
    halfmove = aux[:, 2].float() if pieces.device.type == "mps" else aux[:, 2].double()
    extra[:, 5] = (halfmove / 100.0).float().unsqueeze(1)
    extra[:, 6] = ((aux[:, 3] - 1).clamp(max=2).float() / 2.0).unsqueeze(1)
    return torch.cat([planes, extra.reshape(b, -1, 8, 8)], dim=1)


def packed_loss(logits, values, legal_mask, move, value, value_coef=1.0):
    """alphazero_loss() for one-hot policy targets on pre-encoded tensors."""
    logits = logits.float().masked_fill(~legal_mask, float("-inf"))
    policy_loss = F.cross_entropy(logits, move)
    value_loss = F.mse_loss(values.float(), value)
    return policy_loss + value_coef * value_loss, policy_loss, value_loss
