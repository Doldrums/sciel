"""Claims "in this context, this action changes the child's event rate against the other actions", tested anytime-validly.

Each claim is a betting e-process on DR contrasts (causal/sequential.py); rejections are controlled online with e-LOND
in the order claims first appear, and the reported interval is a confidence sequence at the false-coverage level.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass

from ..causal.sequential import contrasts, lond_level, sequence

DECISIONS = ("operator", "mode", "strategy", "research_open", "method", "planner", "spawn_target", "procedure",
             "eviction_planner", "inspiration_source")
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


def _closed(row) -> bool:
    return len(row.distribution) > 1 and (row.scope == "lineage" or row.outcome is not None)


class Theory:
    def __init__(self):
        self.claims: list[Claim] = []
        self.order: list[list] = []
        self.rejected: set = set()
        self.shift, self.mix = SHIFT, MIX

    def update(self, ledger) -> list[Claim]:
        cells = defaultdict(list)
        for row in ledger.rows:
            if row.decision in DECISIONS and _closed(row):
                cells[(row.decision, row.context)].append(row)
        known = {tuple(k) for k in self.order}
        for (decision, context), rows in cells.items():
            for action in dict.fromkeys(a for r in rows for a in r.distribution):
                if (decision, context, action) not in known:
                    self.order.append([decision, context, action])
                    known.add((decision, context, action))
        tested = []
        for index, (decision, context, action) in enumerate(self.order, 1):
            rows = cells.get((decision, context), [])
            if len(rows) < MIN_ROWS:
                continue
            psi, bound = contrasts(rows, action)
            if not len(psi):
                continue
            key = f"{decision}|{context}|{action}"
            peak = sequence(psi, bound, ALPHA)["e_peak"]
            if key not in self.rejected and peak >= 1.0 / lond_level(ALPHA, index, len(self.rejected)):
                self.rejected.add(key)
            tested.append((decision, context, action, key, psi, bound, sum(r.choice == action for r in rows)))
        level = ALPHA * max(len(self.rejected), 1) / max(len(tested), 1)
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
