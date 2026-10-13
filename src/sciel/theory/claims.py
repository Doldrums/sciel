"""Claims "in this context, this action changes the child's event rate against the other actions", tested anytime-validly.

Each claim is a betting e-process on DR contrasts (causal/sequential.py); rejections are controlled online with e-LOND
in the order claims first appear, and the reported interval is a confidence sequence at the false-coverage level.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from statistics import NormalDist

from ..causal.sequential import comparator, contrasts, lond_level, sequence
from .schools import DECISIONS as SCHOOL_DECISIONS, island_context

DECISIONS = ("operator", "mode", "strategy", "research_open", "method", "planner", "spawn_target", "procedure",
             "eviction_planner", "inspiration_source", "finding", "idea_target", "origin", "cell_stage", "cell_scope", "sibling")
ALPHA = 0.05
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
    e: float = 1.0
    rejected: bool = False

    @property
    def decided(self) -> bool:
        return self.rejected and self.taken >= MIN_TAKEN and (self.low > 0.0 or self.high < 0.0)

    def card(self) -> str:
        verb = "raises" if self.effect > 0 else "lowers"
        return (f"{self.decision} '{self.action}' {verb} the chance that a child improves its island by "
                f"{abs(self.effect):.2f} against the alternatives (anytime 95% interval {self.low:+.2f}..{self.high:+.2f}, "
                f"{self.taken} trials)")


def evpi(claim) -> float:
    """Expected value of perfect information about a claim with estimate N(effect, se^2)."""
    if claim.se <= 0:
        return 0.0
    z = abs(claim.effect) / claim.se
    return claim.se * NormalDist().pdf(z) - abs(claim.effect) * NormalDist().cdf(-z)


def open_question(claims: list, context: str):
    undecided = [c for c in claims if c.context == context and not c.decided]
    return max(undecided, key=evpi, default=None)


def question_text(claim) -> str:
    return (f"Does {claim.decision} '{claim.action}' change the chance that a child improves its island "
            f"(current estimate {claim.effect:+.2f}, interval {claim.low:+.2f}..{claim.high:+.2f})?")


def _testable(row, ledger) -> bool:
    return len(row.distribution) > 1 and ledger.settled(row)


class Theory:
    def __init__(self):
        self.claims: list[Claim] = []
        self.order: list[list] = []
        self.rejected: set = set()
        self.shift, self.mix = SHIFT, MIX

    def update(self, ledger) -> list[Claim]:
        cells = defaultdict(list)
        for row in sorted(ledger.rows, key=lambda r: r.iteration):
            if row.decision in DECISIONS and _testable(row, ledger):
                cells[(row.decision, row.context)].append(row)
                if row.decision in SCHOOL_DECISIONS:
                    cells[(row.decision, island_context(row.island))].append(row)
        known = {tuple(k) for k in self.order}
        for (decision, context), rows in cells.items():
            for action in dict.fromkeys(a for r in rows for a in r.distribution):
                if (decision, context, action) not in known:
                    self.order.append([decision, context, action])
                    known.add((decision, context, action))
        position = {"|".join(k): i for i, k in enumerate(self.order, 1)}
        tested = []
        for index, (decision, context, action) in enumerate(self.order, 1):
            rows = cells.get((decision, context), [])
            if len(rows) < MIN_ROWS:
                continue
            fixed, reference = comparator(rows, action)
            if not fixed:
                continue
            psi, bound = contrasts(rows, action, reference)
            if not len(psi):
                continue
            key = f"{decision}|{context}|{action}"
            peak = sequence(psi, bound, ALPHA)["e_peak"]
            earlier = sum(position[k] < index for k in self.rejected)
            if key not in self.rejected and peak >= 1.0 / lond_level(ALPHA, index, earlier):
                self.rejected.add(key)
            tested.append((decision, context, action, key, psi, bound, sum(r.choice == action for r in rows)))
        level = ALPHA * max(len(self.rejected), 1) / max(len(self.order), 1)
        self.claims = []
        for decision, context, action, key, psi, bound, taken in tested:
            s = sequence(psi, bound, level)
            self.claims.append(Claim(decision, context, action, s["mean"], s["low"], s["high"], s["n"], taken,
                                     s["se"], s["e_peak"], key in self.rejected))
        return [c for c in self.claims if c.decided]

    def tilt(self, decision: str, context: str, distribution: dict) -> dict:
        effects = {c.action: c.effect for c in self.claims
                   if c.decision == decision and c.context == context and c.decided}
        if not effects:
            return dict(distribution)
        shifted = {a: p * math.exp(self.shift * effects.get(a, 0.0)) for a, p in distribution.items()}
        total = sum(shifted.values())
        return {a: (1 - self.mix) * p + self.mix * shifted[a] / total for a, p in distribution.items()}

    def refuted(self, decision: str, action: str) -> bool:
        decided = [c.effect for c in self.claims if c.decision == decision and c.action == action and c.decided]
        return bool(decided) and max(decided) < 0

    def cards(self, context: str, k: int = 3) -> list[str]:
        decided = sorted((c for c in self.claims if c.context == context and c.decided),
                         key=lambda c: -abs(c.effect))
        return [c.card() for c in decided[:k]]

    def state(self) -> dict:
        return {"claims": [asdict(c) for c in self.claims], "order": self.order, "rejected": sorted(self.rejected)}

    def load_state(self, data: dict) -> None:
        self.claims = [Claim(**c) for c in data["claims"]]
        self.order = [list(k) for k in data["order"]]
        self.rejected = set(data["rejected"])
