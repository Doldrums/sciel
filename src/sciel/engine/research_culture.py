"""Research cells under the decision policy: opening and method suggestion are randomized arms credited by the theory.

A method is credited for its own measured effect and, at a discount, for the effect of the methods built on it;
a method proven harmful withdraws every method that derives from or imports it.
"""

from __future__ import annotations

import math

SUGGEST_SHARE = 0.5
DECAY = 0.5
CREDIT_SHIFT = 4.0


class ResearchCulture:
    def __init__(self, engine):
        self.engine = engine
        self.context = None
        self.state = None
        self._links: dict[str, set] = {}

    def _adjust(self, decision: str, distribution: dict) -> dict:
        engine = self.engine
        return engine.experiments.allocate(decision, self.context,
                                           engine.theory.tilt(decision, self.context, distribution))

    def open(self, rule: bool) -> bool:
        rule = rule or self.engine.swm.uncertain(self.state, self.context, "operator", self.engine.op_bandit.arms)
        base = self.engine.decisions.mix({"skip": float(not rule), "open": float(rule)})
        return self.engine.decisions.sample("research_open", self._adjust("research_open", base)) == "open"

    def _sources(self, tool_id: str) -> set:
        if tool_id not in self._links:
            manifest = self.engine.research_tools.library.get(tool_id)["manifest"]
            self._links[tool_id] = set(manifest["parents"]) | set(manifest["dependencies"])
        return self._links[tool_id]

    def ancestors(self, tool_id: str) -> dict:
        depth, frontier = {}, [tool_id]
        for level in range(1, 64):
            frontier = [s for t in frontier for s in self._sources(t) if s not in depth and s != tool_id]
            if not frontier:
                break
            depth.update({s: level for s in frontier})
        return depth

    def effect(self, tool_id: str) -> float:
        claims = [c for c in self.engine.theory.claims if c.decision == "method" and c.action == tool_id]
        taken = sum(c.taken for c in claims)
        return sum(c.effect * c.taken for c in claims) / taken if taken else 0.0

    def credit(self, published: list) -> dict:
        credit = {t: self.effect(t) for t in published}
        for tool in published:
            for ancestor, level in self.ancestors(tool).items():
                if ancestor in credit:
                    credit[ancestor] += DECAY ** level * self.effect(tool)
        return credit

    def retired(self, tool_id: str) -> bool:
        return any(self.engine.theory.refuted("method", t) for t in {tool_id, *self.ancestors(tool_id)})

    def suggest(self, shown: list) -> str | None:
        if not shown:
            return None
        credit = self.credit(shown)
        weights = {t: math.exp(CREDIT_SHIFT * credit[t]) for t in shown}
        total = sum(weights.values())
        distribution = {"none": 1 - SUGGEST_SHARE} | {t: SUGGEST_SHARE * w / total for t, w in weights.items()}
        choice = self.engine.decisions.sample("method", self._adjust("method", distribution))
        return None if choice == "none" else choice
