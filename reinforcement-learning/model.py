import torch
import torch.nn as nn
import torch.nn.functional as F

from chess_environment import (
    ACTION_VERSION,
    ENCODING_VERSION,
    INPUT_CHANNELS,
    MOVE_PLANES,
    POLICY_SIZE,
    board_to_tensor,
    legal_move_mask,
)


MODEL_VERSION = 2
DEFAULT_MODEL_CONFIG = {
    "channels": 96,
    "num_blocks": 8,
    "value_head_channels": 32,
    "value_hidden": 256,
}


def _group_norm(channels, max_groups=16):
    """GroupNorm with a group count that divides `channels`.

    GroupNorm (unlike BatchNorm) is independent of batch size, so it stays valid
    when self-play rollouts shrink the batch down to a single active game.
    """
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class ResidualBlock(nn.Module):
    """Standard pre-activation-free residual block: (conv-norm-relu) x2 + skip."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = _group_norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = _group_norm(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.relu(x + residual)


class PolicyNetwork(nn.Module):
    """Residual CNN with separate policy and value heads (AlphaZero-style trunk).

    channels / num_blocks control capacity. Defaults are a deeper, wider network
    than the original 3-conv stack while keeping the same (logits, value) API.
    """

    def __init__(
        self,
        channels=96,
        num_blocks=8,
        value_head_channels=32,
        value_hidden=256,
    ):
        super().__init__()
        self.config = {
            "channels": channels,
            "num_blocks": num_blocks,
            "value_head_channels": value_head_channels,
            "value_hidden": value_hidden,
        }

        # Input stem: lift history and rule-state planes into the trunk width.
        self.stem = nn.Sequential(
            nn.Conv2d(INPUT_CHANNELS, channels, kernel_size=3, padding=1, bias=False),
            _group_norm(channels),
            nn.ReLU(inplace=True),
        )

        # Residual trunk.
        self.blocks = nn.Sequential(*[ResidualBlock(channels) for _ in range(num_blocks)])

        # Spatial action head: 73 move types at each origin square.
        self.policy_conv = nn.Conv2d(channels, MOVE_PLANES, kernel_size=1)

        # Value head: 1x1 conv -> MLP -> scalar in [-1, 1] (matches reward scale).
        self.value_conv = nn.Sequential(
            nn.Conv2d(channels, value_head_channels, kernel_size=1, bias=False),
            _group_norm(value_head_channels),
            nn.ReLU(inplace=True),
        )
        self.value_fc = nn.Sequential(
            nn.Linear(value_head_channels * 8 * 8, value_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(value_hidden, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)

        p = self.policy_conv(x)
        logits = p.permute(0, 1, 3, 2).contiguous().flatten(1)

        v = self.value_conv(x)
        value = torch.tanh(self.value_fc(v.flatten(1))).squeeze(-1)
        return logits, value

    def alphazero_loss(
        self,
        positions,
        mcts_policies,
        outcomes,
        value_weights=None,
        value_coef=1.0,
    ):
        device = next(self.parameters()).device

        x = torch.cat([board_to_tensor(position) for position in positions]).to(device)
        logits_batch, values = self.forward(x)

        policy_losses = []
        for i, (position, target) in enumerate(zip(positions, mcts_policies)):
            board = position.board() if hasattr(position, "board") else position
            mask = legal_move_mask(board, device=logits_batch.device, dtype=logits_batch.dtype)
            masked_logits = logits_batch[i].masked_fill(mask == 0, float("-inf"))
            log_probs_i = F.log_softmax(masked_logits, dim=0)
            indices = target.indices.to(device=device)
            probabilities = target.probabilities.to(
                device=device, dtype=log_probs_i.dtype
            )
            policy_losses.append(-(probabilities * log_probs_i[indices]).sum())

        policy_loss = torch.stack(policy_losses).mean()
        values = values.reshape(-1)
        outcomes = outcomes.to(device=device, dtype=values.dtype)
        if value_weights is None:
            value_weights = torch.ones_like(outcomes)
        else:
            value_weights = value_weights.to(device=device, dtype=values.dtype)
        squared_errors = (values - outcomes).square() * value_weights
        weight_sum = value_weights.sum()
        value_loss = (
            squared_errors.sum() / weight_sum.clamp_min(1.0)
            if weight_sum.item() > 0
            else values.sum() * 0.0
        )
        total_loss = policy_loss + value_coef * value_loss
        return total_loss, policy_loss, value_loss

    def checkpoint_metadata(self):
        return {
            "model_version": MODEL_VERSION,
            "encoding_version": ENCODING_VERSION,
            "action_version": ACTION_VERSION,
            "model_config": dict(self.config),
            "policy_size": POLICY_SIZE,
        }

        