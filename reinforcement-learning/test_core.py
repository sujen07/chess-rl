"""Board encoding, packed SL data and checkpoint compatibility."""

import io
import random
import tempfile
import unittest
from pathlib import Path

import chess
import chess.pgn
import numpy as np
import torch

from chess_environment import (
    INPUT_CHANNELS,
    PositionRecord,
    board_to_tensor,
    index_to_move,
    move_to_index,
)
from checkpoint import load_checkpoint
from model import PolicyNetwork, checkpoint_model_config
from sl_dataset import PackedDataset, concat_packed, encode_game


def repeated_board():
    board = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
        board.push_uci(uci)
    return board


class CoreTests(unittest.TestCase):
    def test_history_and_rule_planes(self):
        board = repeated_board()
        record = PositionRecord.from_board(board)
        tensor = board_to_tensor(record)
        self.assertEqual(tensor.shape, (1, INPUT_CHANNELS, 8, 8))
        self.assertEqual(record.repetition_count, 2)
        self.assertTrue(torch.all(tensor[0, -1] == 0.5))

        no_castling = chess.Board()
        no_castling.castling_rights = chess.BB_EMPTY
        self.assertFalse(
            torch.equal(board_to_tensor(chess.Board()), board_to_tensor(no_castling))
        )

    def test_action_round_trip_including_promotions(self):
        boards = [
            chess.Board(),
            chess.Board("4k3/P7/8/8/8/8/7p/4K3 w - - 0 1"),
            chess.Board("4k3/8/8/8/8/8/p7/4K3 b - - 0 1"),
        ]
        black_to_move = chess.Board()
        black_to_move.push_uci("e2e4")
        boards.append(black_to_move)
        for board in boards:
            indices = []
            for move in board.legal_moves:
                index = move_to_index(move, board)
                self.assertEqual(index_to_move(index, board), move)
                indices.append(index)
            self.assertEqual(len(indices), len(set(indices)))

    def test_packed_encoding_matches_board_to_tensor(self):
        rng = random.Random(7)
        games, boards = [], []
        for _ in range(8):
            board, game = chess.Board(), chess.pgn.Game()
            node = game
            for _ in range(rng.randint(5, 120)):
                if board.is_game_over():
                    break
                move = rng.choice(list(board.legal_moves))
                boards.append(board.copy())
                board.push(move)
                node = node.add_variation(move)
            game.headers["Result"] = "1-0"
            games.append(game)
        game = chess.pgn.Game()  # repetitions and an en-passant capture
        node, board = game, chess.Board()
        for uci in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2 + ["e2e4", "a7a6", "e4e5", "d7d5", "e5d6"]:
            boards.append(board.copy())
            board.push_uci(uci)
            node = node.add_variation(chess.Move.from_uci(uci))
        game.headers["Result"] = "1/2-1/2"
        games.append(game)

        data = PackedDataset(concat_packed([encode_game(g) for g in games]))
        x, legal_mask, move, _ = data.batch(np.arange(len(data)), "cpu")
        expected = torch.cat([board_to_tensor(b) for b in boards])
        self.assertTrue(torch.equal(x, expected))
        self.assertTrue(bool(legal_mask[torch.arange(len(boards)), move].all()))
        self.assertEqual(
            legal_mask.sum(1).tolist(), [b.legal_moves.count() for b in boards]
        )

    def test_packed_value_blends_engine_eval(self):
        game = chess.pgn.read_game(io.StringIO(
            '[Result "0-1"]\n\n1. e4 { [%eval 0.0] } e5 { [%eval #2] } 2. Qh5 0-1\n'
        ))
        values = encode_game(game, eval_weight=0.5)["value"]
        # Ply 0 has no eval yet; ply 1 is a level eval with Black winning the
        # game; at ply 2 the engine's mate for White cancels White's loss.
        np.testing.assert_allclose(values, [-1.0, 0.5, 0.0])

    def test_legacy_checkpoint_is_rejected(self):
        policy = PolicyNetwork(
            channels=16, num_blocks=1, value_head_channels=4, value_hidden=16
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save({"policy_state_dict": policy.state_dict()}, path)
            with self.assertRaisesRegex(ValueError, "not compatible"):
                load_checkpoint(path, policy)

    def test_checkpoint_without_newer_model_options_loads(self):
        policy = PolicyNetwork(
            channels=16, num_blocks=1, value_head_channels=4, value_hidden=16
        )
        legacy_config = {
            "channels": 16, "num_blocks": 1, "value_head_channels": 4, "value_hidden": 16,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old_config.pt"
            torch.save(
                {
                    **policy.checkpoint_metadata(),
                    "model_config": legacy_config,
                    "policy_state_dict": policy.state_dict(),
                },
                path,
            )
            self.assertEqual(checkpoint_model_config(path), policy.config)
            load_checkpoint(path, policy)



if __name__ == "__main__":
    unittest.main()
