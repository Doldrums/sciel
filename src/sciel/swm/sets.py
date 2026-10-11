"""Context encoders of the world model: a Set-Transformer over the island population and a causal lineage transformer."""

from __future__ import annotations

import torch
from torch import nn

HEADS = 4
INDUCING = 8


class MAB(nn.Module):
    def __init__(self, d: int):
        super().__init__()
        self.attention = nn.MultiheadAttention(d, HEADS, batch_first=True)
        self.norm1, self.norm2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.feed = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, q, k, mask=None):
        h = self.norm1(q + self.attention(q, k, k, key_padding_mask=mask, need_weights=False)[0])
        return self.norm2(h + self.feed(h))


class SetEncoder(nn.Module):
    """ISAB then PMA; a learned null member keeps an empty population well defined."""

    def __init__(self, d: int):
        super().__init__()
        self.null = nn.Parameter(torch.zeros(1, 1, d))
        self.inducing = nn.Parameter(torch.randn(1, INDUCING, d) * 0.1)
        self.seed = nn.Parameter(torch.randn(1, 1, d) * 0.1)
        self.induce, self.back, self.pool = MAB(d), MAB(d), MAB(d)

    def forward(self, members, mask):
        b = members.shape[0]
        x = torch.cat([self.null.expand(b, -1, -1), members], 1)
        mask = torch.cat([mask.new_zeros(b, 1), mask], 1)
        h = self.induce(self.inducing.expand(b, -1, -1), x, mask)
        x = self.back(x, h)
        return self.pool(self.seed.expand(b, -1, -1), x, mask)[:, 0]


class LineageEncoder(nn.Module):
    """Two causal layers over (ancestor latent, action, reward) tokens: the smallest depth that can form induction heads."""

    def __init__(self, d: int, actions: int, length: int):
        super().__init__()
        self.action = nn.Embedding(actions + 1, d)
        self.reward = nn.Linear(1, d)
        self.position = nn.Parameter(torch.zeros(1, length + 1, d))
        self.start = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, HEADS, 2 * d, dropout=0.1, batch_first=True, norm_first=True)
        self.layers = nn.TransformerEncoder(layer, 2, enable_nested_tensor=False)

    def forward(self, z, actions, rewards, mask):
        b, length = actions.shape
        tokens = z + self.action(actions) + self.reward(rewards[..., None])
        x = torch.cat([self.start.expand(b, -1, -1), tokens], 1) + self.position[:, :length + 1]
        mask = torch.cat([mask.new_zeros(b, 1), mask], 1)
        causal = torch.triu(torch.ones(length + 1, length + 1, dtype=torch.bool), 1)
        h = self.layers(x, mask=causal, src_key_padding_mask=mask)
        last = (~mask).sum(1) - 1
        return h[torch.arange(b), last]
