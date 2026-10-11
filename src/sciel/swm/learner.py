"""Population actor-critic: one learner over all islands, V-trace corrected by the logged propensities."""

from __future__ import annotations

import base64
import io

import numpy as np
import torch
from torch import nn

GAMMA = 0.9
RHO_BAR = 1.0
ENTROPY = 0.01
HIDDEN = 64
MIN_ROWS = 40
MIX = 0.8


def _norm(world, pid: str, context: str) -> np.ndarray:
    return ((world._x(pid, context) - world.mean) / world.std).astype(np.float32)


class ActorCritic(nn.Module):
    def __init__(self, features: int, actions: int):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(features, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN), nn.GELU())
        self.policy = nn.Linear(HIDDEN, actions)
        self.value = nn.Linear(HIDDEN, 1)

    def forward(self, x):
        h = self.trunk(x)
        return self.policy(h), self.value(h).squeeze(-1)


class PopulationLearner:
    def __init__(self, decision: str = "operator"):
        self.decision = decision
        self.actions: list[str] = []
        self.net = None

    @property
    def ready(self) -> bool:
        return self.net is not None

    def fit(self, ledger, world, seed: int, epochs: int = 150, lr: float = 3e-3) -> bool:
        if not world.ready:
            return False
        rows = [r for r in world._rows(ledger) if r.decision == self.decision]
        if len(rows) < MIN_ROWS:
            return False
        torch.manual_seed(seed)
        self.actions = sorted({a for r in rows for a in r.distribution})
        index = {a: i for i, a in enumerate(self.actions)}
        x = torch.tensor(np.stack([_norm(world, r.state, r.context) for r in rows]))
        nxt = [world._program(r.child) is not None if r.child else False for r in rows]
        x_next = torch.tensor(np.stack([_norm(world, r.child, r.context) if n else np.zeros(x.shape[1])
                                        for r, n in zip(rows, nxt)]).astype(np.float32))
        alive = torch.tensor(nxt, dtype=torch.float32)
        a = torch.tensor([index[r.choice] for r in rows])
        mu = torch.tensor([r.propensity for r in rows], dtype=torch.float32)
        reward = torch.tensor([r.reward for r in rows], dtype=torch.float32)
        mask = torch.tensor([[a_ in r.distribution for a_ in self.actions] for r in rows])
        net = ActorCritic(x.shape[1], len(self.actions))
        opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
        for _ in range(epochs):
            opt.zero_grad()
            logits, v = net(x)
            logits = logits.masked_fill(~mask, -1e9)
            logp = torch.log_softmax(logits, -1)
            with torch.no_grad():
                _, v_next = net(x_next)
                rho = torch.clamp(logp.gather(1, a[:, None]).squeeze(1).exp() / mu, max=RHO_BAR)
                target = v + rho * (reward + GAMMA * alive * v_next - v)
                advantage = reward + GAMMA * alive * v_next - v
            policy_loss = -(rho * logp.gather(1, a[:, None]).squeeze(1) * advantage).mean()
            entropy = -(logp.exp() * logp).masked_fill(~mask, 0).sum(-1).mean()
            loss = policy_loss + 0.5 * ((v - target) ** 2).mean() - ENTROPY * entropy
            loss.backward()
            opt.step()
        self.net = net.eval()
        return True

    @torch.no_grad()
    def policy(self, world, pid: str, context: str, distribution: dict) -> dict:
        x = world._x(pid, context)
        if not self.ready or x is None or any(a not in self.actions for a in distribution):
            return dict(distribution)
        logits, _ = self.net(torch.tensor(_norm(world, pid, context)[None]))
        index = {a: i for i, a in enumerate(self.actions)}
        chosen = torch.softmax(logits[0, [index[a] for a in distribution]], -1)
        return {a: MIX * float(p) + (1 - MIX) * distribution[a] for a, p in zip(distribution, chosen)}

    def state(self) -> dict:
        weights = None
        if self.net is not None:
            buffer = io.BytesIO()
            torch.save(self.net.state_dict(), buffer)
            weights = base64.b64encode(buffer.getvalue()).decode()
        return {"decision": self.decision, "actions": self.actions, "weights": weights}

    def load_state(self, data: dict) -> None:
        self.decision, self.actions = data["decision"], list(data["actions"])
        self.net = None
        if data["weights"] is not None:
            state = torch.load(io.BytesIO(base64.b64decode(data["weights"])))
            net = ActorCritic(state["trunk.0.weight"].shape[1], len(self.actions))
            net.load_state_dict(state)
            self.net = net.eval()
