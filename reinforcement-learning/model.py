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
    "arch": "resnet",
    "se_ratio": 0,
    "attention_layers": 0,
    "attention_heads": 4,
    "attention_bias": False,
    "ffn_mult": 2,
}
ARCHITECTURES = ("resnet", "transformer")


def _group_norm(channels, max_groups=16):
    """GroupNorm with a group count that divides `channels`.

    GroupNorm (unlike BatchNorm) is independent of batch size, so it stays valid
    when self-play rollouts shrink the batch down to a single active game.
    """
    groups = min(max_groups, channels)
    while channels % groups != 0:
        groups -= 1
    return nn.GroupNorm(groups, channels)


class SqueezeExcitation(nn.Module):
    """Lc0-style SE: global pool -> MLP -> per-channel gate and bias.

    Gives every 3x3 conv block a cheap view of the whole board (material,
    king safety, game phase) that local convolutions only see after many layers.
    """

    def __init__(self, channels, ratio):
        super().__init__()
        hidden = max(1, channels // ratio)
        self.fc1 = nn.Linear(channels, hidden)
        self.fc2 = nn.Linear(hidden, 2 * channels)

    def forward(self, x):
        pooled = x.mean(dim=(2, 3))
        gate, bias = self.fc2(F.relu(self.fc1(pooled))).chunk(2, dim=1)
        return x * torch.sigmoid(gate)[:, :, None, None] + bias[:, :, None, None]


class ResidualBlock(nn.Module):
    """Standard pre-activation-free residual block: (conv-norm-relu) x2 + skip.

    se_ratio > 0 adds squeeze-and-excitation before the skip connection.
    """

    def __init__(self, channels, se_ratio=0):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm1 = _group_norm(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.norm2 = _group_norm(channels)
        self.se = SqueezeExcitation(channels, se_ratio) if se_ratio else None

    def forward(self, x):
        residual = x
        x = F.relu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        if self.se is not None:
            x = self.se(x)
        return F.relu(x + residual)


class AttentionBlock(nn.Module):
    """Pre-norm transformer layer over the 64 squares as tokens.

    attention_bias adds a learned per-head 64x64 logit bias, a cheap stand-in
    for Lc0's "smolgen": it lets heads specialise in board geometry (diagonals,
    files, knight hops) instead of relearning it from positional embeddings.
    """

    def __init__(self, dim, heads, ffn_mult=2, attention_bias=False):
        super().__init__()
        if dim % heads:
            raise ValueError(f"dim {dim} is not divisible by heads {heads}")
        self.heads = heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_mult * dim),
            nn.GELU(),
            nn.Linear(ffn_mult * dim, dim),
        )
        self.bias = nn.Parameter(torch.zeros(heads, 64, 64)) if attention_bias else None

    def forward(self, tokens):
        b, n, d = tokens.shape
        q, k, v = (
            self.qkv(self.norm1(tokens))
            .reshape(b, n, 3, self.heads, d // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        mask = None if self.bias is None else self.bias.to(q.dtype)
        attn = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        tokens = tokens + self.proj(attn.transpose(1, 2).reshape(b, n, d))
        return tokens + self.ffn(self.norm2(tokens))


class PolicyNetwork(nn.Module):
    """Residual CNN with separate policy and value heads (AlphaZero-style trunk).

    channels / num_blocks control capacity. Defaults are a deeper, wider network
    than the original 3-conv stack while keeping the same (logits, value) API.

    arch="resnet": conv stem + residual blocks (optionally SE), followed by
    `attention_layers` transformer layers (a conv/attention hybrid when > 0).
    arch="transformer": per-square linear embedding + learned position
    embedding, then `attention_layers` transformer layers of width `channels`.
    Both trunks end in a [B, channels, 8, 8] map shared by the same heads.
    """

    def __init__(
        self,
        channels=96,
        num_blocks=8,
        value_head_channels=32,
        value_hidden=256,
        arch="resnet",
        se_ratio=0,
        attention_layers=0,
        attention_heads=4,
        attention_bias=False,
        ffn_mult=2,
    ):
        super().__init__()
        if arch not in ARCHITECTURES:
            raise ValueError(f"unknown arch {arch!r}; expected one of {ARCHITECTURES}")
        if arch == "transformer" and attention_layers < 1:
            raise ValueError("arch='transformer' needs attention_layers >= 1")
        self.config = {
            "channels": channels,
            "num_blocks": num_blocks,
            "value_head_channels": value_head_channels,
            "value_hidden": value_hidden,
            "arch": arch,
            "se_ratio": se_ratio,
            "attention_layers": attention_layers,
            "attention_heads": attention_heads,
            "attention_bias": attention_bias,
            "ffn_mult": ffn_mult,
        }

        if arch == "transformer":
            # Token stem: each square's 103 input features become one token.
            self.stem = nn.Linear(INPUT_CHANNELS, channels)
            self.position = nn.Parameter(torch.randn(1, 64, channels) * 0.02)
            self.blocks = nn.Sequential()
        else:
            # Input stem: lift history and rule-state planes into the trunk width.
            self.stem = nn.Sequential(
                nn.Conv2d(INPUT_CHANNELS, channels, kernel_size=3, padding=1, bias=False),
                _group_norm(channels),
                nn.ReLU(inplace=True),
            )
            # Residual trunk.
            self.blocks = nn.Sequential(
                *[ResidualBlock(channels, se_ratio) for _ in range(num_blocks)]
            )

        self.attention = nn.Sequential(*[
            AttentionBlock(channels, attention_heads, ffn_mult, attention_bias)
            for _ in range(attention_layers)
        ])
        self.attention_norm = nn.LayerNorm(channels) if attention_layers else None

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
        if self.config["arch"] == "transformer":
            tokens = self.stem(x.flatten(2).transpose(1, 2)) + self.position
        else:
            x = self.blocks(self.stem(x))
            tokens = x.flatten(2).transpose(1, 2) if self.attention_norm else None
        if self.attention_norm is not None:
            tokens = self.attention_norm(self.attention(tokens))
            x = tokens.transpose(1, 2).reshape(tokens.shape[0], -1, 8, 8)

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

        