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
    POLICY_SIZE,
    PositionRecord,
    SparsePolicy,
    board_to_tensor,
    index_to_move,
    move_to_index,
)
from mcts import (
    Node,
    mcts_search_batch,
    policy_target,
    training_repetition_exclusions,
)
from model import PolicyNetwork
from replay_buffer import ReplayBuffer
from sl_dataset import PackedDataset, concat_packed, encode_game
from train import load_checkpoint
from training_runtime import load_replay_buffer, save_replay_buffer


def repeated_board():
    board = chess.Board()
    for uci in ("g1f3", "g8f6", "f3g1", "f6g8"):
        board.push_uci(uci)
    return board


class ZeroModel:
    def __call__(self, x):
        return torch.zeros(x.shape[0], POLICY_SIZE), torch.zeros(x.shape[0])


class ChessRLV2Tests(unittest.TestCase):
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

    def test_repetition_history_survives_mcts_children(self):
        board = repeated_board()
        root = mcts_search_batch(ZeroModel(), [board], num_simulations=1)[0]
        self.assertTrue(root.children)
        child = root.children[chess.Move.from_uci("g1f3")]
        self.assertGreaterEqual(len(child.board.move_stack), len(board.move_stack))
        self.assertTrue(child.board.is_repetition(2))

    def test_mcts_children_keep_encoder_history_after_capture(self):
        board = chess.Board()
        for uci in ("e2e4", "d7d5", "g1f3", "g8f6"):
            board.push_uci(uci)
        capture = chess.Move.from_uci("e4d5")
        root = mcts_search_batch(ZeroModel(), [board], num_simulations=1)[0]
        child_board = root.children[capture].board
        expected = board.copy()
        expected.push(capture)
        self.assertEqual(child_board.halfmove_clock, 0)
        self.assertTrue(
            torch.equal(board_to_tensor(child_board), board_to_tensor(expected))
        )

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

    def test_training_target_excludes_immediate_third_repetition(self):
        board = chess.Board()
        for uci in (
            "g1f3", "g8f6", "f3g1", "f6g8",
            "g1f3", "g8f6", "f3g1",
        ):
            board.push_uci(uci)
        repeating_move = chess.Move.from_uci("f6g8")
        root = Node(board)
        root.visits = 10
        for move in board.legal_moves:
            child_board = board.copy()
            child_board.push(move)
            child = Node(child_board, parent=root, action=move, prior=0.1)
            child.visits = 1
            root.children[move] = child
        root.children[repeating_move].visits = 20

        excluded = training_repetition_exclusions(root)
        self.assertIn(repeating_move, excluded)
        target = policy_target(root, excluded_moves=excluded)
        self.assertNotIn(
            move_to_index(repeating_move, board), target.indices.tolist()
        )

        root.value_sum = -10
        self.assertEqual(training_repetition_exclusions(root), set())

    def test_training_filter_blocks_opponent_claim_on_next_move(self):
        board = chess.Board()
        for uci in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"):
            board.push_uci(uci)
        enabling_move = chess.Move.from_uci("f3g1")
        root = Node(board)
        root.visits = 10
        for move in board.legal_moves:
            child_board = board.copy()
            child_board.push(move)
            child = Node(child_board, parent=root, action=move, prior=0.1)
            child.visits = 1
            root.children[move] = child

        self.assertFalse(root.children[enabling_move].board.is_repetition(3))
        self.assertTrue(
            root.children[enabling_move].board.can_claim_threefold_repetition()
        )
        self.assertIn(enabling_move, training_repetition_exclusions(root))

    def test_sparse_loss_and_masked_value(self):
        board = chess.Board()
        move = next(iter(board.legal_moves))
        target = SparsePolicy(
            torch.tensor([move_to_index(move, board)]),
            torch.tensor([1.0]),
        )
        policy = PolicyNetwork(
            channels=16, num_blocks=1, value_head_channels=4, value_hidden=16
        )
        total, policy_loss, value_loss = policy.alphazero_loss(
            [PositionRecord.from_board(board)],
            [target],
            torch.tensor([1.0]),
            value_weights=torch.tensor([0.0]),
        )
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(torch.isfinite(policy_loss))
        self.assertEqual(value_loss.item(), 0.0)

    def test_legacy_checkpoint_is_rejected(self):
        policy = PolicyNetwork(
            channels=16, num_blocks=1, value_head_channels=4, value_hidden=16
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            torch.save({"policy_state_dict": policy.state_dict()}, path)
            with self.assertRaisesRegex(ValueError, "not compatible"):
                load_checkpoint(path, policy)

    def test_replay_round_trip(self):
        board = chess.Board()
        move = next(iter(board.legal_moves))
        target = SparsePolicy(
            torch.tensor([move_to_index(move, board)]), torch.tensor([1.0])
        )
        source = ReplayBuffer(capacity=10)
        source.add([PositionRecord.from_board(board)], [target], [0.0], [1.0])
        restored = ReplayBuffer(capacity=10)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.pt"
            save_replay_buffer(path, source)
            self.assertTrue(load_replay_buffer(path, restored))
        self.assertEqual(len(restored), 1)
        self.assertEqual(restored.sample(1)[0][0].current_fen, board.fen())


if __name__ == "__main__":
    unittest.main()
