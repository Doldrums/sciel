"""Claims "this action changes the child's event rate against the other actions", tested anytime-validly.

The family is fixed by decision: every decision whose actions do not come and go, plus method and procedure against
a reference action, pooled over contexts. Each claim is a mixture-of-bets e-process on DR contrasts
(causal/sequential.py); e-BH on the current values controls the false discovery rate under any dependence, and the
reported interval is the matching e-BY confidence set. Knowledge items are judged by sibling pairs instead (trials.py).
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import asdict, dataclass
from statistics import NormalDist

import numpy as np

from ..causal.sequential import comparator, contrasts, continued, e_bh, mixture, summary

DECISIONS = ("operator", "mode", "strategy", "research_open", "method", "planner", "spawn_target", "procedure",
             "eviction_planner", "inspiration_source", "finding", "idea_target", "origin", "cell_stage", "cell_scope", "sibling")
FAMILY = tuple(d for d in DECISIONS if d not in ("strategy", "finding"))
CONTEXT = "all"
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
    undecided = [c for c in claims if not c.decided]
    return max(undecided, key=evpi, default=None)


def question_text(claim) -> str:
    return (f"Does {claim.decision} '{claim.action}' change the chance that a child improves its island "
            f"(current estimate {claim.effect:+.2f}, interval {claim.low:+.2f}..{claim.high:+.2f})?")


def _testable(row, ledger) -> bool:
    return len(row.distribution) > 1 and ledger.settled(row)


class Theory:
    def __init__(self):
        self.claims: list[Claim] = []
        self.shift, self.mix = SHIFT, MIX
        self.prior: dict[str, list] = {}
        self.wealth: dict[str, list] = {}

    def update(self, ledger) -> list[Claim]:
        cells = defaultdict(list)
        for row in sorted(ledger.rows, key=lambda r: r.iteration):
            if row.decision in FAMILY and _testable(row, ledger):
                cells[row.decision].append(row)
        tested = {}
        for decision, rows in cells.items():
            if len(rows) < MIN_ROWS:
                continue
            for action in dict.fromkeys(a for r in rows for a in r.distribution):
                fixed, reference = comparator(rows, action)
                if not fixed:
                    continue
                psi, bound = contrasts(rows, action, reference)
                if len(psi):
                    s = mixture(psi, bound, ALPHA)
                    prior = self.prior.get(f"{decision}|{action}")
                    if prior is not None:
                        s = {**continued(s["log_wealth"], np.array(prior), psi, ALPHA)}
                    self.wealth[f"{decision}|{action}"] = s["log_wealth"].tolist()
                    tested[(decision, action)] = (psi, s, sum(r.choice == action for r in rows))
        rejected = e_bh({k: v[1]["e"] for k, v in tested.items()}, ALPHA)
        level = ALPHA * max(len(rejected), 1) / max(len(tested), 1)
        self.claims = []
        for (decision, action), (psi, s, taken) in tested.items():
            c = summary(psi, s["log_wealth"], level)
            effect = c["mean"] if f"{decision}|{action}" not in self.prior else (c["low"] + c["high"]) / 2
            self.claims.append(Claim(decision, CONTEXT, action, effect, c["low"], c["high"], c["n"], taken,
                                     c["se"], s["e"], (decision, action) in rejected and not c["empty"]))
        for key, prior in self.prior.items():
            self.wealth.setdefault(key, prior)
        return [c for c in self.claims if c.decided]

    def tilt(self, decision: str, context: str, distribution: dict) -> dict:
        effects = {c.action: c.effect for c in self.claims if c.decision == decision and c.decided}
        if not effects:
            return dict(distribution)
        shifted = {a: p * math.exp(self.shift * effects.get(a, 0.0)) for a, p in distribution.items()}
        total = sum(shifted.values())
        return {a: (1 - self.mix) * p + self.mix * shifted[a] / total for a, p in distribution.items()}

    def refuted(self, decision: str, action: str) -> bool:
        decided = [c.effect for c in self.claims if c.decision == decision and c.action == action and c.decided]
        return bool(decided) and max(decided) < 0

    def cards(self, context: str, k: int = 3) -> list[str]:
        decided = sorted((c for c in self.claims if c.decided), key=lambda c: -abs(c.effect))
        return [c.card() for c in decided[:k]]

    def state(self) -> dict:
        return {"claims": [asdict(c) for c in self.claims]}

    def load_state(self, data: dict) -> None:
        self.claims = [Claim(**c) for c in data["claims"]]
