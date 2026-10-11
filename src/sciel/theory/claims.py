"""Claims "in this context, this action changes the lineage-event rate" estimated from the decision ledger."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import NormalDist

from ..causal.estimators import action_effect
from ..causal.ledger import HORIZON

DECISIONS = ("operator", "mode", "strategy", "research_open", "method", "planner")
MIN_ROWS = 12
MIN_TAKEN = 8
SHIFT = 4.0
MIX = 0.3


@dataclass
class Claim:
    decision: str
    context: str
    action: str
    effect: float
    low: float
    high: float
    n: int
    taken: int
    se: float = 0.0

    @property
    def decided(self) -> bool:
        return self.taken >= MIN_TAKEN and (self.low > 0.0 or self.high < 0.0)

    def card(self) -> str:
        verb = "raises" if self.effect > 0 else "lowers"
        return (f"{self.decision} '{self.action}' {verb} the chance of an island improvement within "
                f"{HORIZON} steps by {abs(self.effect):.2f} (family-wise 95% CI {self.low:+.2f}..{self.high:+.2f}, "
                f"{self.taken} trials)")


class Theory:
    def __init__(self):
        self.claims: list[Claim] = []
        self.outcome_model = None

    def update(self, ledger) -> list[Claim]:
        tested = []
        model = self.outcome_model
        q = model.q if model is not None and model.ready else None
        for decision in DECISIONS:
            rows = [r for r in ledger.labeled(decision) if len(r.distribution) > 1]
            for context in sorted({r.context for r in rows}):
                local = [r for r in rows if r.context == context]
                if len(local) < MIN_ROWS:
                    continue
                for action in sorted({a for r in local for a in r.distribution}):
                    e = action_effect(local, action, q)
                    if e["taken"]:
                        tested.append((decision, context, action, e))
        z = NormalDist().inv_cdf(1 - 0.025 / max(len(tested), 1))
        self.claims = [Claim(d, c, a, e["effect"], e["effect"] - z * e["se"], e["effect"] + z * e["se"],
                             e["n"], e["taken"], e["se"]) for d, c, a, e in tested]
        return [c for c in self.claims if c.decided]

    def tilt(self, decision: str, context: str, distribution: dict) -> dict:
        effects = {c.action: c.effect for c in self.claims
                   if c.decision == decision and c.context == context and c.decided}
        if not effects:
            return dict(distribution)
        shifted = {a: p * math.exp(SHIFT * effects.get(a, 0.0)) for a, p in distribution.items()}
        total = sum(shifted.values())
        return {a: (1 - MIX) * p + MIX * shifted[a] / total for a, p in distribution.items()}

    def refuted(self, decision: str, action: str) -> bool:
        decided = [c.effect for c in self.claims if c.decision == decision and c.action == action and c.decided]
        return bool(decided) and max(decided) < 0

    def cards(self, context: str, k: int = 3) -> list[str]:
        decided = sorted((c for c in self.claims if c.context == context and c.decided),
                         key=lambda c: -abs(c.effect))
        return [c.card() for c in decided[:k]]

    def state(self) -> dict:
        return {"claims": [asdict(c) for c in self.claims]}

    def load_state(self, data: dict) -> None:
        self.claims = [Claim(**c) for c in data["claims"]]
