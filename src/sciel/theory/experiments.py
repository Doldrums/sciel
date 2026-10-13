"""Discriminating experiments: a randomized block that decides the most valuable undecided claim."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from ..causal.ledger import HORIZON
from .claims import ALPHA, FAMILY

MIN_EFFECT = 0.02
SHRINK_ROWS = 50
OVERHEAD = 1.3
MAX_REPEATS = 2
MAX_BLOCK = 24
SHARE = 0.5


@dataclass
class Experiment:
    decision: str
    context: str
    action: str
    hypothesis: str
    start: int
    length: int
    required: int
    expected: float
    status: str = "running"
    outcome: str | None = None


def _required(claim, family: int) -> int:
    """Rows for a mixture-of-bets e-process to reach the e-BH bar, at an effect shrunk toward zero."""
    variance = claim.se ** 2 * max(claim.n, 1)
    effect = abs(claim.effect) * claim.n / (claim.n + SHRINK_ROWS)
    return math.ceil(OVERHEAD * 2 * variance * math.log(family / ALPHA) / max(effect, 1e-9) ** 2)


class Experiments:
    def __init__(self):
        self.active: Experiment | None = None
        self.history: list[Experiment] = []

    def propose(self, theory, iteration: int, remaining: int, context: str) -> Experiment | None:
        if self.active is not None:
            return None
        best = None
        family = max(len(theory.claims), 1)
        for claim in theory.claims:
            repeats = sum((e.decision, e.action) == (claim.decision, claim.action) for e in self.history)
            if (claim.decided or abs(claim.effect) < MIN_EFFECT or claim.decision not in FAMILY
                    or repeats >= MAX_REPEATS):
                continue
            extra = _required(claim, family) - claim.n
            if extra <= 0 or extra > remaining:
                continue
            score = abs(claim.effect) / extra
            if best is None or score > best[0]:
                best = (score, claim, extra)
        if best is None:
            return None
        _, claim, extra = best
        self.active = Experiment(
            decision=claim.decision, context=claim.context, action=claim.action,
            hypothesis=(f"{claim.decision} '{claim.action}' changes the child's event rate by "
                        f"{claim.effect:+.2f}"),
            start=iteration, length=min(MAX_BLOCK, extra), required=extra, expected=claim.effect)
        return self.active

    def allocate(self, decision: str, context: str, distribution: dict) -> dict:
        exp = self.active
        if (exp is None or exp.status != "running" or exp.decision != decision
                or exp.action not in distribution):
            return dict(distribution)
        rest = sum(p for a, p in distribution.items() if a != exp.action)
        if rest <= 0:
            return dict(distribution)
        return {a: SHARE if a == exp.action else (1 - SHARE) * p / rest for a, p in distribution.items()}

    def tick(self, iteration: int, theory, ledger) -> Experiment | None:
        exp = self.active
        if exp is None or iteration < exp.start + exp.length:
            return None
        if iteration < exp.start + exp.length + HORIZON:
            exp.status = "awaiting"
            return None
        theory.update(ledger)
        claim = next((c for c in theory.claims if (c.decision, c.context, c.action)
                      == (exp.decision, exp.context, exp.action)), None)
        exp.status = "done"
        if claim is None or not claim.decided:
            exp.outcome = "undecided"
        else:
            exp.outcome = "supported" if claim.effect * exp.expected > 0 else "refuted"
        self.history.append(exp)
        self.active = None
        return exp

    def state(self) -> dict:
        return {"active": asdict(self.active) if self.active else None,
                "history": [asdict(e) for e in self.history]}

    def load_state(self, data: dict) -> None:
        self.active = Experiment(**data["active"]) if data["active"] else None
        self.history = [Experiment(**e) for e in data["history"]]
