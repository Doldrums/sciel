"""Latent search world model.

State = program encoder latent + Set-Transformer summary of the island population + causal lineage summary.
A mixture density predicts the child's latent against an EMA target encoder (JEPA style); an ensemble
of heads predicts the lineage event and a head predicts validity.
"""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import torch
from torch import nn

from .sets import LineageEncoder, SetEncoder

LATENT = 32
HEADS = 5
HIDDEN = 64
COMPONENTS = 4
LINEAGE = 8
POPULATION = 16
VAR_WEIGHT = 1.0
EMA = 0.95

torch.set_num_threads(1)


def _mlp(i: int, o: int) -> nn.Module:
    return nn.Sequential(nn.Linear(i, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, o))


class Network(nn.Module):
    def __init__(self, features: int, actions: int):
        super().__init__()
        self.actions = actions
        self.encoder = _mlp(features, LATENT)
        self.target = _mlp(features, LATENT)
        self.target.load_state_dict(self.encoder.state_dict())
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.action = nn.Embedding(actions, LATENT)
        self.population = SetEncoder(LATENT)
        self.lineage = LineageEncoder(LATENT, actions, LINEAGE)
        self.mixture = _mlp(4 * LATENT, COMPONENTS * (1 + 2 * LATENT))
        self.value = nn.ModuleList(_mlp(4 * LATENT, 1) for _ in range(HEADS))
        self.valid = _mlp(4 * LATENT, 1)

    def context(self, batch: dict):
        z = self.encoder(batch["x"])
        pop = self.population(self.encoder(batch["pop"]), batch["pop_mask"])
        lin = self.lineage(self.encoder(batch["lin_x"]), batch["lin_a"], batch["lin_r"], batch["lin_mask"])
        return z, torch.cat([pop, lin], -1)

    def joint(self, z, context, a):
        return torch.cat([z, context, self.action(a)], -1)

    def heads(self, h):
        return torch.cat([head(h) for head in self.value], -1), self.valid(h).squeeze(-1)

    def density(self, h):
        out = self.mixture(h)
        logits = out[..., :COMPONENTS]
        mu, log_std = out[..., COMPONENTS:].reshape(*out.shape[:-1], COMPONENTS, 2 * LATENT).split(LATENT, -1)
        return torch.log_softmax(logits, -1), mu, log_std.clamp(-5, 2)

    @torch.no_grad()
    def update_target(self):
        for t, s in zip(self.target.parameters(), self.encoder.parameters()):
            t.mul_(EMA).add_((1 - EMA) * s)


def _variance(z):
    return torch.relu(1.0 - torch.sqrt(z.var(dim=0) + 1e-4)).mean()


def _nll(log_pi, mu, log_std, goal):
    d = (goal[:, None] - mu) / log_std.exp()
    log_prob = -0.5 * (d ** 2).sum(-1) - log_std.sum(-1) - 0.5 * LATENT * np.log(2 * np.pi)
    return -torch.logsumexp(log_pi + log_prob, -1)


def tensors(data: dict) -> dict:
    out = {k: torch.tensor(v) for k, v in data.items()}
    for key in ("pop_mask", "lin_mask", "has_child"):
        if key in out:
            out[key] = out[key].bool()
    for key in ("a", "lin_a"):
        out[key] = out[key].long()
    return out


def _loss(net, batch, bce):
    z, context = net.context(batch)
    h = net.joint(z, context, batch["a"])
    values, validity = net.heads(h)
    weight = batch["boot"] * batch["w"][:, None]
    loss = (bce(values, batch["y"][:, None].expand_as(values)) * weight).sum() / weight.sum().clamp(min=1e-6)
    loss = loss + bce(validity, batch["valid"]).mean() + VAR_WEIGHT * _variance(z)
    has_child = batch["has_child"]
    if has_child.any():
        with torch.no_grad():
            goal = net.target(batch["child"][has_child])
        loss = loss + _nll(*net.density(h[has_child]), goal).mean() / LATENT
    return loss


@torch.no_grad()
def _held_out(net, batch, bce) -> float:
    net.eval()
    z, context = net.context(batch)
    values, _ = net.heads(net.joint(z, context, batch["a"]))
    return float(bce(values.mean(-1), batch["y"]).mean())


def fit(train_data: dict, valid_data: dict, actions: int, seed: int, steps: int = 300, size: int = 256,
        lr: float = 3e-3, check: int = 20) -> tuple:
    """Minibatch training with early stopping on held-out value log loss; returns the network and that loss."""
    torch.manual_seed(seed)
    train, valid = tensors(train_data), tensors(valid_data)
    n = len(train["x"])
    net = Network(train["x"].shape[1], actions)
    opt = torch.optim.Adam([p for p in net.parameters() if p.requires_grad], lr=lr, weight_decay=1e-3)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    generator = torch.Generator().manual_seed(seed)
    best, best_state = float("inf"), deepcopy(net.state_dict())
    for step in range(1, steps + 1):
        net.train()
        batch = {k: v[torch.randint(n, (size,), generator=generator)] for k, v in train.items()} if n > size else train
        opt.zero_grad()
        _loss(net, batch, bce).backward()
        opt.step()
        net.update_target()
        if step % check == 0:
            score = _held_out(net, valid, bce)
            if score < best:
                best, best_state = score, deepcopy(net.state_dict())
    net.load_state_dict(best_state)
    return net.eval(), best


def _optimistic(p, bonus: float):
    return p.mean(-1) + bonus * p.std(-1)


@torch.no_grad()
def lookahead(net: Network, batch: dict, actions: np.ndarray, gamma: float, bonus: float) -> np.ndarray:
    """Optimistic two-step return of each first action, the second step taken in expectation over the mixture."""
    n, k = len(actions), COMPONENTS
    z, context = net.context(tensors(batch))
    z, context = z.expand(n, -1), context.expand(n, -1)
    first = torch.tensor(actions)
    h1 = net.joint(z, context, first)
    p1 = torch.sigmoid(net.heads(h1)[0])
    log_pi, mu, _ = net.density(h1)
    z2 = mu.reshape(n * k, LATENT)
    c2 = context.repeat_interleave(k, 0)
    second = torch.stack([_optimistic(torch.sigmoid(net.heads(net.joint(z2, c2, first.new_full((n * k,), int(a))))[0]),
                                      bonus) for a in actions], -1).max(-1).values.reshape(n, k)
    return (_optimistic(p1, bonus) + gamma * (log_pi.exp() * second).sum(-1)).numpy()


@torch.no_grad()
def predict(net: Network, batch: dict) -> dict:
    t = tensors(batch)
    z, context = net.context(t)
    values, validity = net.heads(net.joint(z, context, t["a"]))
    p = torch.sigmoid(values)
    return {"value": p.mean(-1).numpy(), "epistemic": p.std(-1).numpy(),
            "valid": torch.sigmoid(validity).numpy()}
