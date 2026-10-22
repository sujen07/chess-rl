import random
from collections import deque

from chess_environment import ACTION_VERSION, ENCODING_VERSION


class ReplayBuffer:
    """Fixed-size FIFO buffer of compact v2 self-play samples.

    Samples accumulate across self-play batches so gradient updates draw from
    a mix of recent and slightly-older games instead of only the batch that
    was just played, which smooths out the correlation between consecutive
    positions from the same game.
    """

    def __init__(self, capacity=50_000):
        self.capacity = capacity
        self.boards = deque(maxlen=capacity)
        self.pis = deque(maxlen=capacity)
        self.outcomes = deque(maxlen=capacity)
        self.value_weights = deque(maxlen=capacity)

    def __len__(self):
        return len(self.boards)

    def add(self, boards, pis, outcomes, value_weights=None):
        if value_weights is None:
            value_weights = [1.0] * len(boards)
        self.boards.extend(boards)
        self.pis.extend(pis)
        self.outcomes.extend(outcomes)
        self.value_weights.extend(value_weights)

    def sample(self, batch_size):
        """Sample up to `batch_size` items uniformly at random, without replacement."""
        n = len(self.boards)
        batch_size = min(batch_size, n)
        idxs = random.sample(range(n), batch_size)
        boards = [self.boards[i] for i in idxs]
        pis = [self.pis[i] for i in idxs]
        outcomes = [self.outcomes[i] for i in idxs]
        value_weights = [self.value_weights[i] for i in idxs]
        return boards, pis, outcomes, value_weights

    def state_dict(self):
        return {
            "encoding_version": ENCODING_VERSION,
            "action_version": ACTION_VERSION,
            "capacity": self.capacity,
            "boards": list(self.boards),
            "pis": list(self.pis),
            "outcomes": list(self.outcomes),
            "value_weights": list(self.value_weights),
        }

    def load_state_dict(self, state):
        if (
            state.get("encoding_version") != ENCODING_VERSION
            or state.get("action_version") != ACTION_VERSION
        ):
            raise ValueError("Replay buffer is not compatible with Chess RL v2")
        for values, target in (
            (state.get("boards", []), self.boards),
            (state.get("pis", []), self.pis),
            (state.get("outcomes", []), self.outcomes),
            (state.get("value_weights", []), self.value_weights),
        ):
            target.extend(values[-self.capacity:])
