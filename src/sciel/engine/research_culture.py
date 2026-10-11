"""Research cells under the decision policy: opening and method suggestion are randomized arms credited by the theory."""

from __future__ import annotations

SUGGEST_SHARE = 0.5


class ResearchCulture:
    def __init__(self, engine):
        self.engine = engine
        self.context = None
        self.state = None

    def _adjust(self, decision: str, distribution: dict) -> dict:
        engine = self.engine
        return engine.experiments.allocate(decision, self.context,
                                           engine.theory.tilt(decision, self.context, distribution))

    def open(self, rule: bool) -> bool:
        rule = rule or self.engine.swm.uncertain(self.state, self.context, "operator", self.engine.op_bandit.arms)
        base = self.engine.decisions.mix({"skip": float(not rule), "open": float(rule)})
        return self.engine.decisions.sample("research_open", self._adjust("research_open", base)) == "open"

    def retired(self, tool_id: str) -> bool:
        return self.engine.theory.refuted("method", tool_id)

    def suggest(self, shown: list) -> str | None:
        if not shown:
            return None
        distribution = {"none": 1 - SUGGEST_SHARE} | {i: SUGGEST_SHARE / len(shown) for i in shown}
        choice = self.engine.decisions.sample("method", self._adjust("method", distribution))
        return None if choice == "none" else choice
