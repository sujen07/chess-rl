"""Array-based MCTS for fast self-play and evaluation.

Compared to mcts.py this search:
  * encodes positions straight from bitboards into the packed sl_dataset
    format (no FEN round trips), carrying the 8-frame history down the tree;
  * creates a child board only when the child is first visited, and keeps
    per-edge visit/value statistics in numpy arrays;
  * lets callers keep a subtree between moves (tree reuse);
  * talks to the network through an "evaluator" that receives packed
    positions plus their legal-move indices and returns only the legal
    logits, so worker <-> GPU traffic stays a few KB per batch.

Evaluator contract:
    evaluator(pieces uint8 [B, 8, 32], aux uint8 [B, 4],
              legal_idx int64 [L], legal_off int64 [B + 1])
        -> (legal_logits float32 [L], values float32 [B])
"""

from __future__ import annotations

import math

import chess
import numpy as np
import torch

from chess_environment import HISTORY_LENGTH, move_to_index
from sl_dataset import decode_planes, encode_planes

_PIECE_CODES = np.arange(1, 13, dtype=np.uint8)[:, None]
_COLORS_AND_TYPES = [
    (piece_type, color)
    for color in (chess.WHITE, chess.BLACK)
    for piece_type in chess.PIECE_TYPES
]
_HISTORY_STACK = HISTORY_LENGTH - 1


def absolute_codes(board):
    """[64] piece codes by real square (1-6 White P..K, 7-12 Black), via bitboards."""
    masks = np.array(
        [board.pieces_mask(pt, color) for pt, color in _COLORS_AND_TYPES],
        dtype="<u8",
    )
    bits = np.unpackbits(masks.view(np.uint8), bitorder="little").reshape(12, 64)
    return (bits * _PIECE_CODES).sum(axis=0, dtype=np.uint8)


_INDEX_CACHE = {}


def move_indices(board, moves):
    """move_to_index for many moves; it only depends on the move and side to move."""
    turn = board.turn
    out = np.empty(len(moves), dtype=np.int64)
    for i, move in enumerate(moves):
        key = (move.from_square, move.to_square, move.promotion, turn)
        index = _INDEX_CACHE.get(key)
        if index is None:
            index = _INDEX_CACHE[key] = move_to_index(move, board)
        out[i] = index
    return out


def terminal_status(board, legal_moves):
    """None if play continues, else the value for the side to move.

    Same verdict as board.is_game_over(claim_draw=True), but skips the
    expensive repetition / fifty-move claim checks while the halfmove clock
    makes them impossible (a claimable threefold needs >= 8 reversible plies
    after the next move).
    """
    if not legal_moves:
        return -1.0 if board.is_check() else 0.0
    if board.is_insufficient_material():
        return 0.0
    if board.halfmove_clock >= 7 and board.can_claim_draw():
        return 0.0
    return None


def is_game_over(board):
    """Fast equivalent of board.is_game_over(claim_draw=True)."""
    return terminal_status(board, list(board.legal_moves)) is not None


class Node:
    __slots__ = (
        "board", "parent", "edge", "history", "moves", "indices",
        "pieces", "aux", "priors", "raw_priors", "child_n", "child_w",
        "children", "visits", "value_sum", "terminal",
    )

    def __init__(self, board, history, parent=None, edge=-1):
        self.board = board
        self.parent = parent
        self.edge = edge
        self.history = history
        self.moves = list(board.legal_moves)
        self.terminal = terminal_status(board, self.moves)
        self.visits = 0
        self.value_sum = 0.0
        self.priors = None
        self.raw_priors = None
        self.children = {}
        if self.terminal is None:
            self.indices = move_indices(board, self.moves)
            self.pieces, self.aux = encode_planes(board, history)
            n = len(self.moves)
            self.child_n = np.zeros(n, dtype=np.float64)
            self.child_w = np.zeros(n, dtype=np.float64)

    @classmethod
    def root(cls, board):
        """A fresh root for a game board; reads history from its move stack."""
        cursor = board.copy()
        history = [absolute_codes(cursor)]
        while cursor.move_stack and len(history) < HISTORY_LENGTH:
            cursor.pop()
            history.append(absolute_codes(cursor))
        stack = max(board.halfmove_clock, _HISTORY_STACK)
        return cls(board.copy(stack=stack), tuple(history))

    @property
    def expanded(self):
        return self.priors is not None

    def value_mean(self):
        return self.value_sum / self.visits if self.visits else 0.0

    def child(self, edge):
        node = self.children.get(edge)
        if node is None:
            # Repetition only depends on reversible history and the encoder
            # reads HISTORY_LENGTH frames; keep whichever suffix is longer.
            board = self.board.copy(
                stack=max(self.board.halfmove_clock, _HISTORY_STACK)
            )
            board.push(self.moves[edge])
            history = (absolute_codes(board),) + self.history[:_HISTORY_STACK]
            node = Node(board, history, parent=self, edge=edge)
            self.children[edge] = node
        return node

    def select_edge(self, c_puct, fpu_reduction):
        n = self.child_n
        # FPU: unvisited moves look slightly worse than this node's own value.
        fpu = self.value_mean() - fpu_reduction
        q = np.where(n > 0, -self.child_w / np.maximum(n, 1.0), fpu)
        u = c_puct * math.sqrt(self.visits) * self.priors / (1.0 + n)
        return int(np.argmax(q + u))


def backpropagate(node, value):
    """`value` is from the perspective of the side to move at `node`."""
    while True:
        node.visits += 1
        node.value_sum += value
        parent = node.parent
        if parent is None:
            return
        parent.child_n[node.edge] += 1
        parent.child_w[node.edge] += value
        value = -value
        node = parent


def gather_leaf(root, c_puct=1.41, fpu_reduction=0.0, root_fpu_reduction=None):
    """Walk to a leaf; return it if it needs a network eval, else None.

    Terminal leaves are backed up immediately and count as a simulation.
    `root_fpu_reduction` (default: fpu_reduction) lets a noised root keep
    exploring unvisited moves.
    """
    node = root
    fpu = fpu_reduction if root_fpu_reduction is None else root_fpu_reduction
    while True:
        if node.terminal is not None:
            backpropagate(node, node.terminal)
            return None
        if not node.expanded:
            return node
        node = node.child(node.select_edge(c_puct, fpu))
        fpu = fpu_reduction


def evaluate_leaves(evaluator, leaves):
    """One batched network call for `leaves`; sets priors and backs up values."""
    if not leaves:
        return
    lengths = np.array([len(leaf.indices) for leaf in leaves], dtype=np.int64)
    legal_off = np.concatenate([[0], np.cumsum(lengths)])
    legal_logits, values = evaluator(
        np.stack([leaf.pieces for leaf in leaves]),
        np.stack([leaf.aux for leaf in leaves]),
        np.concatenate([leaf.indices for leaf in leaves]),
        legal_off,
    )
    for i, leaf in enumerate(leaves):
        logits = legal_logits[legal_off[i]:legal_off[i + 1]].astype(np.float64)
        priors = np.exp(logits - logits.max())
        priors /= priors.sum()
        leaf.priors = leaf.raw_priors = priors
        backpropagate(leaf, float(values[i]))


def add_dirichlet_noise(root, alpha, eps, rng=np.random):
    noise = rng.dirichlet([alpha] * len(root.raw_priors))
    root.priors = (1 - eps) * root.raw_priors + eps * noise


def advance(root, move):
    """Reuse the subtree under `move` as the next root, or None if unexplored."""
    if root is None:
        return None
    try:
        edge = root.moves.index(move)
    except ValueError:
        return None
    node = root.children.get(edge)
    if node is None or node.terminal is not None or not node.expanded:
        return None
    node.parent = None
    node.edge = -1
    node.priors = node.raw_priors  # drop any noise this node saw as a child
    root.children = {}  # let the rest of the old tree be collected
    return node


def run_searches(
    evaluator,
    roots,
    visit_targets,
    c_puct=1.41,
    fpu_reduction=0.0,
    root_fpu_reduction=None,
    noise=None,
    dirichlet_alpha=0.3,
    dirichlet_eps=0.25,
    rng=np.random,
):
    """Search every root until root.visits >= its target, batching leaves.

    `noise[i]` adds Dirichlet noise to root i (after it is expanded).
    """
    noise = noise or [False] * len(roots)
    if any(root.terminal is not None for root in roots):
        raise ValueError("cannot search a finished game")
    fresh = [root for root in roots if not root.expanded]
    evaluate_leaves(evaluator, fresh)
    for root, wants_noise in zip(roots, noise):
        if wants_noise:
            add_dirichlet_noise(root, dirichlet_alpha, dirichlet_eps, rng)

    while True:
        leaves = []
        active = False
        for root, target in zip(roots, visit_targets):
            if root.visits >= target:
                continue
            active = True
            leaf = gather_leaf(root, c_puct, fpu_reduction, root_fpu_reduction)
            if leaf is not None:
                leaves.append(leaf)
        if not active:
            return roots
        evaluate_leaves(evaluator, leaves)


def mcts_search_batch(evaluator, boards, num_simulations=100, c_puct=1.41, **kwargs):
    """Fresh search per board; returns the roots (simulations after the root eval)."""
    roots = [Node.root(board) for board in boards]
    return run_searches(
        evaluator, roots, [num_simulations + 1] * len(roots), c_puct=c_puct, **kwargs
    )


def visit_distribution(root, temperature=1.0, excluded_edges=()):
    """(edges, probabilities) over root moves from visit counts."""
    edges = [e for e in range(len(root.moves)) if e not in excluded_edges]
    if not edges:
        edges = list(range(len(root.moves)))
    visits = root.child_n[edges]
    if temperature == 0:
        probs = np.zeros_like(visits)
        probs[np.argmax(visits)] = 1.0
    else:
        scaled = visits ** (1.0 / temperature)
        total = scaled.sum()
        probs = scaled / total if total > 0 else np.full_like(scaled, 1 / len(scaled))
    return edges, probs


def policy_target(root, excluded_edges=()):
    """(policy indices int16, probabilities float32) from temperature-1 visits."""
    edges, probs = visit_distribution(root, 1.0, excluded_edges)
    return root.indices[edges].astype(np.int16), probs.astype(np.float32)


def select_edge(root, temperature=1.0, excluded_edges=(), rng=np.random):
    edges, probs = visit_distribution(root, temperature, excluded_edges)
    return edges[rng.choice(len(edges), p=probs)]


def training_repetition_exclusions(root, value_threshold=-0.25):
    """Root edges to suppress in self-play targets (see mcts.py)."""
    # A move can only complete or enable a threefold claim after enough
    # reversible plies; skip the push-and-check work before that.
    if root.board.halfmove_clock < 6 or root.value_mean() < value_threshold:
        return set()
    excluded = set()
    for edge in np.flatnonzero(root.child_n > 0):
        board = root.child(int(edge)).board
        if board.is_repetition(3) or board.can_claim_threefold_repetition():
            excluded.add(int(edge))
    return excluded if len(excluded) < len(root.moves) else set()


class LocalEvaluator:
    """In-process evaluator around a PolicyNetwork (arena, tests, play)."""

    def __init__(self, policy, device, amp_enabled=True):
        self.policy = policy
        self.device = torch.device(device)
        self.amp_enabled = amp_enabled and self.device.type == "cuda"

    @torch.no_grad()
    def __call__(self, pieces, aux, legal_idx, legal_off):
        return evaluate_packed(
            self.policy, self.device, self.amp_enabled, pieces, aux, legal_idx, legal_off
        )


@torch.no_grad()
def evaluate_packed(policy, device, amp_enabled, pieces, aux, legal_idx, legal_off):
    """Shared GPU path: decode, forward, and gather only the legal logits."""
    x = decode_planes(
        torch.from_numpy(pieces).to(device), torch.from_numpy(aux).to(device)
    )
    with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
        logits, values = policy(x)
    rows = np.repeat(np.arange(len(pieces)), np.diff(legal_off))
    legal_logits = logits[
        torch.from_numpy(rows).to(device), torch.from_numpy(legal_idx).to(device)
    ]
    return (
        legal_logits.float().cpu().numpy(),
        values.float().reshape(-1).cpu().numpy(),
    )
