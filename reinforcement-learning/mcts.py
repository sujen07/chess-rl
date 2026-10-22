import numpy as np
from chess_environment import *

class Node:
    def __init__(self, board, parent=None, action=None, prior=0):
        self.board = board
        self.parent = parent
        self.action = action
        self.children = {} # {action: child node}
        self.visits = 0
        self.value_sum = 0
        self.prior = prior

    def is_leaf(self):
        return len(self.children) == 0

    def is_root(self):
        return self.parent is None

    def value_mean(self):
        return self.value_sum / self.visits if self.visits > 0 else 0

    def select_child(self, c=1.41):
        # -child.value_mean(): a child's value is from its own mover's perspective,
        # which is this node's opponent, so it has to be negated before comparing.
        return max(self.children.values(), key=lambda child: -child.value_mean() + c * child.prior * np.sqrt(self.visits) / (1 + child.visits))

@torch.no_grad()
def expand_batch(model, nodes, device=None):
    """One forward pass shared by every node in `nodes`.

    Mutates each node's .children in place (one per legal move) and returns
    the leaf value for each node, in the same order. This is the batching
    lever: on a GPU/MPS backend a small network's single-position forward
    pass is dominated by launch overhead, not compute, so evaluating N leaves
    at once costs barely more than evaluating one.
    """
    x = torch.cat([board_to_tensor(node.board) for node in nodes])
    if device is not None:
        x = x.to(device)
    logits, values = model(x)

    for i, node in enumerate(nodes):
        masked_logits = masked_policy_logits(logits[i], node.board)
        probs = torch.softmax(masked_logits, dim=0)
        for move in node.board.legal_moves:
            prior = probs[move_to_index(move, node.board)].item()
            # Repetition only depends on reversible history, and the encoder
            # reads HISTORY_LENGTH frames. Preserve whichever suffix is longer
            # so tree search sees the same draw rules and inputs as play.
            next_board = node.board.copy(
                stack=max(node.board.halfmove_clock, HISTORY_LENGTH - 1)
            )
            next_board.push(move)
            node.children[move] = Node(next_board, parent=node, action=move, prior=prior)

    return values.tolist()


def expand(model, node, device=None):
    return expand_batch(model, [node], device=device)[0]


def mcts_search_batch(model, boards, num_simulations=100, c=1.41, device=None,
                       add_dirichlet_noise=False, dirichlet_alpha=0.3, dirichlet_eps=0.25):
    """Run `num_simulations` of MCTS on len(boards) independent trees at once,
    batching every round's leaf evaluations into a single forward pass.

    This is root parallelization: statistics are NOT shared across trees, only
    the (expensive, GPU-bound) network calls are — each round, every tree
    contributes at most one leaf to a single batched expand_batch() call.
    """
    roots = [Node(b.copy()) for b in boards]
    expand_batch(model, roots, device=device)

    if add_dirichlet_noise:
        for root in roots:
            if root.children:
                noise = np.random.dirichlet([dirichlet_alpha] * len(root.children))
                for child, n in zip(root.children.values(), noise):
                    child.prior = (1 - dirichlet_eps) * child.prior + dirichlet_eps * n

    for _ in range(num_simulations):
        leaves = []
        for root in roots:
            node = root
            while not node.is_leaf():
                node = node.select_child(c)
            leaves.append(node)

        to_expand = []
        for node in leaves:
            if terminal(node.board):
                backpropagate(node, terminal_value(node.board))
            else:
                to_expand.append(node)

        if to_expand:
            values = expand_batch(model, to_expand, device=device)
            for node, value in zip(to_expand, values):
                backpropagate(node, value)

    return roots


def mcts_search(model, board, num_simulations=100, c=1.41, device=None, add_dirichlet_noise=False,
                dirichlet_alpha=0.3, dirichlet_eps=0.25):
    """Single-tree MCTS; a thin convenience wrapper around mcts_search_batch."""
    return mcts_search_batch(
        model, [board], num_simulations=num_simulations, c=c, device=device,
        add_dirichlet_noise=add_dirichlet_noise,
        dirichlet_alpha=dirichlet_alpha, dirichlet_eps=dirichlet_eps,
    )[0]


def backpropagate(node, value):
    while node is not None:
        node.visits += 1
        node.value_sum += value
        value = -value  # flip for the opponent's perspective
        node = node.parent


def visit_distribution(root, temperature=1.0, excluded_moves=None):
    """root's legal moves and their probabilities, derived from visit counts."""
    excluded_moves = excluded_moves or set()
    moves = [move for move in root.children if move not in excluded_moves]
    if not moves:
        moves = list(root.children.keys())
    visits = np.array([root.children[m].visits for m in moves], dtype=np.float64)

    if temperature == 0:
        probs = np.zeros_like(visits)
        probs[np.argmax(visits)] = 1.0
    else:
        scaled = visits ** (1.0 / temperature)
        total = scaled.sum()
        probs = scaled / total if total > 0 else np.full_like(scaled, 1 / len(scaled))

    return moves, probs


def policy_target(root, temperature=1.0, excluded_moves=None):
    """Sparse visit-count target over the versioned policy action space."""
    moves, probs = visit_distribution(root, temperature, excluded_moves)
    by_index = {}
    for move, p in zip(moves, probs):
        index = move_to_index(move, root.board)
        by_index[index] = by_index.get(index, 0.0) + float(p)
    return SparsePolicy(
        indices=torch.tensor(list(by_index), dtype=torch.int64),
        probabilities=torch.tensor(list(by_index.values()), dtype=torch.float32),
    )


def select_move(root, temperature=1.0, excluded_moves=None):
    moves, probs = visit_distribution(root, temperature, excluded_moves)
    idx = np.random.choice(len(moves), p=probs)
    return moves[idx]


def training_repetition_exclusions(root, value_threshold=-0.25):
    """Moves to suppress in self-play targets, never in deployed inference.

    A draw remains available when MCTS considers the mover clearly worse, and
    the filter is disabled if every legal continuation repeats.
    """
    if root.value_mean() < value_threshold:
        return set()
    repetition_claims = {
        move
        for move, child in root.children.items()
        # claim_draw=True ends self-play as soon as the opponent can claim on
        # their next move, so suppress both an existing third occurrence and
        # a move that hands the opponent that immediate claim.
        if child.board.is_repetition(3)
        or child.board.can_claim_threefold_repetition()
    }
    return (
        repetition_claims
        if len(repetition_claims) < len(root.children)
        else set()
    )
