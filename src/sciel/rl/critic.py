"""Critic of step returns: Q(x, a) per covered decision, an ensemble of small networks.

x is the parent program (score relative to the population best, generation, latent of the world model when it is
ready), the context and the decisions already taken in this step; a is the action, a pool item being encoded by its
standing when it was offered (none, trial, tested, culture) so the critic transfers to items it never saw and never
sees a verdict reached after the choice. Two models are fitted, one per island parity; each scores the other's rows.
"""

from __future__ import annotations

import base64
import io
import math
import zlib

import numpy as np
import torch
from torch import nn

from ..swm.model import LATENT

HEADS = 5
HIDDEN = 64
STEPS = 200
BATCH = 256
CHOSEN = 16
POOLS = ("strategy", "finding")


def token(engine, decision: str, action: str, standing: dict | None = None) -> str:
    """Pool items by their standing when offered (now, when no record of it is given), other actions by name."""
    if decision not in POOLS or action == "none":
        return f"{decision}:{action}"
    if standing:
        return f"{decision}:{standing.get(action, 'trial')}"
    pool = engine.strategies.pool if decision == "strategy" else engine.findings.pool
    item = next((i for i in pool if i.id == action), None)
    return f"{decision}:{item.status if item else 'trial'}"


def state(engine, pid: str, context: str, world=None) -> np.ndarray:
    world = world or engine.swm
    program = engine.db._all.get(pid)
    best = engine.db.pop_best
    relative = program.score / abs(best) if program is not None and best not in (None, 0, float("-inf")) else 0.0
    fields = dict(part.split("=") for part in context.split("|"))
    latent = world.latent(pid) if world.ready else None
    head = [relative, math.log1p(program.gen if program else 0), float(fields.get("phase", 0)),
            float(fields.get("third", 0)) / 2]
    return np.concatenate([head, latent if latent is not None else np.zeros(LATENT)]).astype(np.float32)


def chosen(engine, records: list) -> np.ndarray:
    out = np.zeros(CHOSEN, dtype=np.float32)
    for record in records:
        out[zlib.crc32(token(engine, record["decision"], record["choice"], record.get("standing")).encode())
            % CHOSEN] += 1.0
    return out


class Critic(nn.Module):
    def __init__(self, features: int, actions: int):
        super().__init__()
        self.action = nn.Embedding(actions, 8)
        self.heads = nn.ModuleList(nn.Sequential(nn.Linear(features + 8, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, 1))
                                   for _ in range(HEADS))

    def forward(self, x, a):
        h = torch.cat([x, self.action(a)], -1)
        return torch.cat([head(h) for head in self.heads], -1)


def fit(x: np.ndarray, a: np.ndarray, y: np.ndarray, actions: int, seed: int) -> Critic:
    torch.manual_seed(seed)
    net = Critic(x.shape[1], actions)
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    xt, at, yt = torch.tensor(x), torch.tensor(a).long(), torch.tensor(y, dtype=torch.float32)
    boot = torch.tensor(np.random.default_rng(seed).poisson(1.0, size=(len(y), HEADS)), dtype=torch.float32)
    generator = torch.Generator().manual_seed(seed)
    for _ in range(STEPS):
        idx = torch.randint(len(y), (min(BATCH, len(y)),), generator=generator)
        loss = ((net(xt[idx], at[idx]) - yt[idx, None]) ** 2 * boot[idx]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return net.eval()


@torch.no_grad()
def predict(net: Critic, x: np.ndarray, a: np.ndarray) -> np.ndarray:
    """Ensemble predictions, one column per head."""
    return net(torch.tensor(x), torch.tensor(a).long()).numpy()


def dump(net: Critic) -> str:
    buffer = io.BytesIO()
    torch.save(net.state_dict(), buffer)
    return base64.b64encode(buffer.getvalue()).decode()


def load(data: str, features: int, actions: int) -> Critic:
    net = Critic(features, actions)
    net.load_state_dict(torch.load(io.BytesIO(base64.b64decode(data))))
    return net.eval()
