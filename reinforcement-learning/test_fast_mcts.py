import random
import tempfile
import threading
import unittest
from pathlib import Path

import chess
import numpy as np
import torch

from chess_environment import POLICY_SIZE, board_to_tensor, move_to_index
from fast_mcts import (
    LocalEvaluator,
    Node,
    absolute_codes,
    advance,
    is_game_over,
    mcts_search_batch,
    policy_target,
    training_repetition_exclusions,
)
from model import PolicyNetwork
from multiprocess_train_v3 import pack_game, self_play_loop, train_step
from packed_replay import PackedReplay, soft_policy_loss
from sl_dataset import _absolute_codes, decode_planes


def zero_evaluator(pieces, aux, legal_idx, legal_off):
    return np.zeros(len(legal_idx), np.float32), np.zeros(len(pieces), np.float32)


def random_boards(seed, games=12, max_plies=150):
    rng = random.Random(seed)
    boards = []
    for _ in range(games):
        board = chess.Board()
        for _ in range(rng.randint(1, max_plies)):
            if board.is_game_over():
                break
            board.push(rng.choice(list(board.legal_moves)))
            boards.append(board.copy())
    # Repetitions, en passant and a capture that resets the halfmove clock.
    board = chess.Board()
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2 + ["e2e4", "a7a6", "e4e5", "d7d5"]:
        board.push_uci(uci)
        boards.append(board.copy())
    return boards


def all_nodes(root):
    stack = [root]
    while stack:
        node = stack.pop()
        yield node
        stack.extend(node.children.values())


def tiny_policy():
    return PolicyNetwork(channels=16, num_blocks=1, value_head_channels=4, value_hidden=16)


class FastMctsTests(unittest.TestCase):
    def test_absolute_codes_match_reference(self):
        for board in random_boards(1):
            np.testing.assert_array_equal(absolute_codes(board), _absolute_codes(board))

    def test_terminal_check_matches_python_chess(self):
        boards = random_boards(2, games=30, max_plies=300)
        board = chess.Board()
        for uci in ["g1f3", "g8f6", "f3g1", "f6g8"] * 3:
            board.push_uci(uci)
            boards.append(board.copy())
        boards.append(chess.Board("7k/5Q2/6K1/8/8/8/8/8 b - - 0 1"))  # stalemate
        for board in boards:
            self.assertEqual(is_game_over(board), board.is_game_over(claim_draw=True))

    def test_tree_encodings_match_board_to_tensor(self):
        for board in random_boards(3, games=6, max_plies=60):
            if is_game_over(board):
                continue
            root = mcts_search_batch(zero_evaluator, [board], num_simulations=30)[0]
            nodes = [n for n in all_nodes(root) if n.terminal is None]
            x = decode_planes(
                torch.from_numpy(np.stack([n.pieces for n in nodes])),
                torch.from_numpy(np.stack([n.aux for n in nodes])),
            )
            expected = torch.cat([board_to_tensor(n.board) for n in nodes])
            self.assertTrue(torch.equal(x, expected))
            for n in nodes[:5]:
                self.assertEqual(
                    n.indices.tolist(), [move_to_index(m, n.board) for m in n.moves]
                )

    def test_root_encoding_uses_game_history(self):
        board = random_boards(4, games=1, max_plies=40)[-1]
        root = Node.root(board)
        x = decode_planes(
            torch.from_numpy(root.pieces[None]), torch.from_numpy(root.aux[None])
        )
        self.assertTrue(torch.equal(x, board_to_tensor(board)))

    def test_finds_mate_in_one(self):
        board = chess.Board("6k1/5ppp/8/8/8/8/5PPP/R5K1 w - - 0 1")
        root = mcts_search_batch(zero_evaluator, [board], num_simulations=300)[0]
        best = root.moves[int(np.argmax(root.child_n))]
        self.assertEqual(best, chess.Move.from_uci("a1a8"))
        self.assertGreater(root.value_mean(), 0.5)

    def test_tree_reuse_keeps_stats_and_history(self):
        board = chess.Board()
        root = mcts_search_batch(zero_evaluator, [board], num_simulations=200)[0]
        edge = int(np.argmax(root.child_n))
        move = root.moves[edge]
        visits = int(root.child_n[edge])
        child = advance(root, move)
        self.assertIsNotNone(child)
        self.assertIsNone(child.parent)
        self.assertEqual(child.visits, visits)
        board.push(move)
        x = decode_planes(
            torch.from_numpy(child.pieces[None]), torch.from_numpy(child.aux[None])
        )
        self.assertTrue(torch.equal(x, board_to_tensor(board)))

    def test_repetition_exclusion_blocks_opponent_claim(self):
        board = chess.Board()
        for uci in ("g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"):
            board.push_uci(uci)
        enabling = chess.Move.from_uci("f3g1")
        root = mcts_search_batch(zero_evaluator, [board], num_simulations=200)[0]
        edge = root.moves.index(enabling)
        if root.child_n[edge] == 0:
            root.child(edge)
            root.child_n[edge] = 1
        excluded = training_repetition_exclusions(root, value_threshold=-1.0)
        self.assertIn(edge, excluded)
        idx, _ = policy_target(root, excluded)
        self.assertNotIn(move_to_index(enabling, board), idx.tolist())
        # A clearly losing side may still steer toward the draw.
        root.value_sum = -0.9 * root.visits
        self.assertEqual(training_repetition_exclusions(root, -0.25), set())

    def test_local_evaluator_matches_full_forward(self):
        policy = tiny_policy().eval()
        board = random_boards(5, games=1, max_plies=30)[-1]
        root = Node.root(board)
        legal_logits, values = LocalEvaluator(policy, "cpu")(
            root.pieces[None], root.aux[None], root.indices,
            np.array([0, len(root.indices)]),
        )
        with torch.no_grad():
            logits, value = policy(board_to_tensor(board))
        np.testing.assert_allclose(legal_logits, logits[0, root.indices].numpy(), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(values, value.numpy(), rtol=1e-5, atol=1e-5)


class SelfPlayPipelineTests(unittest.TestCase):
    def settings(self, **overrides):
        settings = dict(
            games_per_worker=3, full_simulations=24, fast_simulations=8,
            full_search_prob=0.5, c_puct=1.41, fpu_reduction=0.25,
            temp_threshold=10, dirichlet_alpha=0.3, dirichlet_eps=0.25,
            max_moves=60, tree_reuse=True, resign_threshold=-0.9,
            resign_consecutive=2, resign_min_ply=4, resign_playthrough=0.5,
            avoid_training_repetitions=True, repetition_value_threshold=-0.25,
            q_weight=0.5, midgame_fens=[], midgame_prob=0.0,
        )
        settings.update(overrides)
        return settings

    def play(self, games, **overrides):
        settings = self.settings(**overrides)
        remaining = [games]
        finished = []

        def claim():
            if remaining[0] == 0:
                return False
            remaining[0] -= 1
            return True

        self_play_loop(
            LocalEvaluator(tiny_policy().eval(), "cpu"), settings, claim,
            lambda g: finished.append(pack_game(g, settings["q_weight"])),
            threading.Event(), np.random.default_rng(0),
        )
        return finished

    def test_self_play_produces_consistent_samples(self):
        games = self.play(5)
        self.assertEqual(len(games), 5)
        for game in games:
            n = len(game["value"])
            self.assertEqual(n, game["full_moves"])
            self.assertEqual(game["pieces"].shape, (n, 8, 32))
            self.assertTrue(np.all(np.abs(game["value"]) <= 1.0))
            for legal, idx, prob in zip(game["legal"], game["pi_idx"], game["pi_prob"]):
                self.assertTrue(set(idx.tolist()) <= set(legal.tolist()))
                self.assertAlmostEqual(float(prob.sum()), 1.0, places=5)
            self.assertIn(game["result"], ("1-0", "0-1", "1/2-1/2"))
            self.assertGreater(game["search_visits"], 0)

    def test_resignation_ends_game(self):
        values = {1: 1, 2: 3, 3: 3, 4: 5, 5: 9}

        def material_evaluator(pieces, aux, legal_idx, legal_off):
            # Current frame codes: 1-6 mover P..K, 7-12 opponent P..K.
            frame = np.stack((pieces[:, 0] & 15, pieces[:, 0] >> 4), axis=-1).reshape(len(pieces), -1)
            balance = sum(
                v * ((frame == code).sum(1) - (frame == code + 6).sum(1))
                for code, v in values.items()
            )
            return (
                np.zeros(len(legal_idx), np.float32),
                np.clip(balance / 10.0, -1, 1).astype(np.float32),
            )

        settings = self.settings(
            resign_playthrough=0.0, games_per_worker=1, resign_min_ply=0,
            midgame_fens=["4k3/8/8/8/8/2q5/3q4/7K w - - 0 1"], midgame_prob=1.0,
        )
        finished = []
        remaining = [1]

        def claim():
            remaining[0] -= 1
            return remaining[0] >= 0

        self_play_loop(
            material_evaluator, settings, claim,
            lambda g: finished.append(pack_game(g, 0.5)),
            threading.Event(), np.random.default_rng(0),
        )
        self.assertEqual(finished[0]["termination"], "RESIGN")
        self.assertEqual(finished[0]["result"], "0-1")

    def test_replay_round_trip_and_train_step(self):
        games = self.play(3)
        replay = PackedReplay(capacity=50)
        for game in games:
            replay.add(game)
        self.assertGreater(len(replay), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.npz"
            replay.save(path)
            restored = PackedReplay(capacity=50)
            self.assertTrue(restored.load(path))
        self.assertEqual(len(restored), len(replay))
        order = replay._ordered()
        np.testing.assert_array_equal(restored.pieces[: len(order)], replay.pieces[order])
        self.assertEqual(
            [r.tolist() for r in restored.pi_idx[: len(order)]],
            [replay.pi_idx[i].tolist() for i in order],
        )

        x, mask, target, value, weight = replay.batch(np.arange(len(replay)), "cpu")
        self.assertTrue(bool((target[~mask] == 0).all()))
        torch.testing.assert_close(target.sum(1), torch.ones(len(replay)))

        policy = tiny_policy()
        optimizer = torch.optim.AdamW(policy.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        metrics = train_step(
            policy, optimizer, scaler, replay, None, 16, 0, 1.0,
            torch.device("cpu"), False, np.random.default_rng(0),
        )
        self.assertTrue(np.isfinite(metrics["total"]))

    def test_soft_policy_loss_ignores_illegal_moves(self):
        logits = torch.randn(2, POLICY_SIZE)
        mask = torch.zeros(2, POLICY_SIZE, dtype=torch.bool)
        mask[:, :3] = True
        target = torch.zeros(2, POLICY_SIZE)
        target[:, 0] = 1.0
        loss = soft_policy_loss(logits, mask, target)
        expected = -torch.log_softmax(logits[:, :3], dim=1)[:, 0]
        torch.testing.assert_close(loss, expected)


if __name__ == "__main__":
    unittest.main()
