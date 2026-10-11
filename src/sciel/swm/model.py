"""Latent search world model: JEPA-style transition predictor with an ensemble of outcome heads."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

LATENT = 32
HEADS = 5
HIDDEN = 64
VAR_WEIGHT = 1.0
EMA = 0.95

torch.set_num_threads(1)


def _mlp(i: int, o: int) -> nn.Module:
    return nn.Sequential(nn.Linear(i, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, o))


class Network(nn.Module):
    def __init__(self, features: int, actions: int):
        super().__init__()
        self.encoder = _mlp(features, LATENT)
        self.target = _mlp(features, LATENT)
        self.target.load_state_dict(self.encoder.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.action = nn.Embedding(actions, LATENT)
        self.predictor = _mlp(2 * LATENT, LATENT)
        self.value = nn.ModuleList(_mlp(2 * LATENT, 1) for _ in range(HEADS))
        self.valid = _mlp(2 * LATENT, 1)

    def joint(self, x, a):
        return torch.cat([self.encoder(x), self.action(a)], dim=-1)

    def heads(self, x, a):
        h = self.joint(x, a)
        return torch.cat([head(h) for head in self.value], dim=-1), self.valid(h).squeeze(-1)

    def latent_heads(self, z, a):
        h = torch.cat([z, self.action(a)], dim=-1)
        return torch.sigmoid(torch.cat([head(h) for head in self.value], dim=-1)), h

    @torch.no_grad()
    def update_target(self):
        for t, s in zip(self.target.parameters(), self.encoder.parameters()):
            t.mul_(EMA).add_((1 - EMA) * s)


def _variance(z):
    std = torch.sqrt(z.var(dim=0) + 1e-4)
    return torch.relu(1.0 - std).mean()


def fit(data: dict, actions: int, seed: int, epochs: int = 150, lr: float = 3e-3) -> Network:
    torch.manual_seed(seed)
    x, a = torch.tensor(data["x"]), torch.tensor(data["a"])
    y, w = torch.tensor(data["y"]), torch.tensor(data["w"])
    valid, boot = torch.tensor(data["valid"]), torch.tensor(data["boot"])
    child, has_child = torch.tensor(data["child"]), torch.tensor(data["has_child"])
    net = Network(x.shape[1], actions)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    for _ in range(epochs):
        opt.zero_grad()
        values, validity = net.heads(x, a)
        value_loss = (bce(values, y[:, None].expand_as(values)) * boot * w[:, None]).sum() / (boot * w[:, None]).sum()
        valid_loss = bce(validity, valid).mean()
        loss = value_loss + valid_loss
        if has_child.any():
            h = net.joint(x[has_child], a[has_child])
            predicted = net.predictor(h)
            with torch.no_grad():
                goal = net.target(child[has_child])
            loss = loss + ((predicted - goal) ** 2).mean() + VAR_WEIGHT * _variance(net.encoder(x))
        loss.backward()
        opt.step()
        net.update_target()
    return net.eval()


def _optimistic(p, bonus: float):
    return p.mean(-1) + bonus * p.std(-1)


@torch.no_grad()
def lookahead(net: Network, x: np.ndarray, actions: np.ndarray, gamma: float, bonus: float) -> np.ndarray:
    """Optimistic two-step return of each first action, imagined through the latent predictor."""
    z = net.encoder(torch.tensor(x[None])).expand(len(actions), -1)
    first = torch.tensor(actions)
    p1, h1 = net.latent_heads(z, first)
    z2 = net.predictor(h1)
    second = torch.stack([net.latent_heads(z2, first.new_full((len(actions),), int(a)))[0] for a in actions], 1)
    return (_optimistic(p1, bonus) + gamma * _optimistic(second, bonus).max(1).values).numpy()


@torch.no_grad()
def predict(net: Network, x: np.ndarray, a: np.ndarray) -> dict:
    values, validity = net.heads(torch.tensor(x), torch.tensor(a))
    p = torch.sigmoid(values)
    return {"value": p.mean(-1).numpy(), "epistemic": p.std(-1).numpy(),
            "valid": torch.sigmoid(validity).numpy()}
