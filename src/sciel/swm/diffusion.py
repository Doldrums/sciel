"""Conditional diffusion over behavior descriptors, guided toward strong solutions with classifier-free guidance."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

STEPS = 50
TRAIN_STEPS = 400
HIDDEN = 128
DROP = 0.2
GUIDANCE = 2.0


def _schedule():
    betas = torch.linspace(1e-4, 0.05, STEPS)
    alphas = 1 - betas
    return betas, alphas, torch.cumprod(alphas, 0)


class Denoiser(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim + 3, HIDDEN), nn.SiLU(), nn.Linear(HIDDEN, HIDDEN), nn.SiLU(),
                                 nn.Linear(HIDDEN, dim))

    def forward(self, x, t, condition, present):
        phase = t.float()[:, None] / STEPS * math.pi
        return self.net(torch.cat([x, torch.sin(phase), condition[:, None] * present[:, None], present[:, None]], -1))


def sample(behaviors: np.ndarray, ranks: np.ndarray, count: int, seed: int) -> np.ndarray:
    """Train on (behavior, quality rank) pairs, then sample behaviors conditioned on the top rank."""
    torch.manual_seed(seed)
    x0 = torch.tensor(behaviors * 2 - 1, dtype=torch.float32)
    rank = torch.tensor(ranks, dtype=torch.float32)
    _, alphas, bars = _schedule()
    model = Denoiser(x0.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    for _ in range(TRAIN_STEPS):
        t = torch.randint(STEPS, (len(x0),))
        noise = torch.randn_like(x0)
        xt = bars[t].sqrt()[:, None] * x0 + (1 - bars[t]).sqrt()[:, None] * noise
        present = (torch.rand(len(x0)) > DROP).float()
        loss = ((model(xt, t, rank, present) - noise) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        x = torch.randn(count, x0.shape[1])
        top, on, off = torch.ones(count), torch.ones(count), torch.zeros(count)
        for step in reversed(range(STEPS)):
            t = torch.full((count,), step)
            eps = (1 + GUIDANCE) * model(x, t, top, on) - GUIDANCE * model(x, t, top, off)
            x = (x - (1 - alphas[step]) / (1 - bars[step]).sqrt() * eps) / alphas[step].sqrt()
            if step:
                x = x + (1 - alphas[step]).sqrt() * torch.randn_like(x)
    return ((x.numpy() + 1) / 2).clip(0, 1)
