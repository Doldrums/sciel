"""Engine decisions drawn from explicit distributions and journaled with their propensities."""

from __future__ import annotations

import math

SCHEMA = "sciel-decision/1"
REQUIRED = ("island", "mode", "parent", "operator", "model", "refine", "strategy")


def _record(name: str, distribution: dict, choice) -> dict:
    if not distribution or choice not in distribution:
        raise ValueError(f"{name} decision has no distribution over its choice")
    values = list(distribution.values())
    if any(not math.isfinite(p) or p < 0 for p in values) or abs(sum(values) - 1.0) > 1e-6:
        raise ValueError(f"{name} decision distribution is not a probability vector")
    if distribution[choice] <= 0:
        raise ValueError(f"{name} decision chose an option with zero probability")
    return {"schema": SCHEMA, "decision": name, "choice": choice,
            "propensity": distribution[choice],
            "distribution": {str(k): float(v) for k, v in distribution.items()}}


class DecisionPolicy:
    def __init__(self, epsilon: float, rng):
        if type(epsilon) not in (int, float) or not 0.0 < epsilon <= 0.5:
            raise ValueError("decision_epsilon must be in (0, 0.5]")
        self.epsilon = float(epsilon)
        self.rng = rng
        self.pending: list[dict] = []

    def mix(self, distribution: dict) -> dict:
        n = len(distribution)
        return {k: (1.0 - self.epsilon) * p + self.epsilon / n for k, p in distribution.items()}

    def draw(self, name: str, options: list, rule_choice):
        """A deterministic rule becomes the (1-epsilon) mode of a recorded distribution."""
        base = {o: float(o == rule_choice) for o in options}
        if len(options) < 2:
            return rule_choice, _record(name, base, rule_choice)
        distribution = self.mix(base)
        choice = self.rng.choices(options, weights=[distribution[o] for o in options], k=1)[0]
        return choice, _record(name, distribution, choice)

    def choose(self, name: str, options: list, rule_choice):
        choice, record = self.draw(name, options, rule_choice)
        self.pending.append(record)
        return choice

    def sample(self, name: str, distribution: dict):
        options = list(distribution)
        choice = self.rng.choices(options, weights=[distribution[o] for o in options], k=1)[0]
        self.pending.append(_record(name, distribution, choice))
        return choice

    def flag(self, name: str, rule: bool) -> bool:
        return self.choose(name, [False, True], bool(rule))

    def observe(self, name: str, distribution: dict, choice) -> None:
        """A choice already sampled by a stochastic component from this distribution."""
        self.pending.append(_record(name, distribution, choice))

    def drain(self) -> list[dict]:
        records, self.pending = self.pending, []
        missing = set(REQUIRED) - {r["decision"] for r in records}
        if missing:
            raise ValueError("decisions without a logged distribution: " + ", ".join(sorted(missing)))
        return records
