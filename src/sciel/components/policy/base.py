"""Single decision-point contract (P6, I-14): floor + propensity in the base."""

from __future__ import annotations

from typing import Protocol


class DecisionPolicy(Protocol):
    def decide(self, ctx, rng): ...
    def observe(self, ev) -> None: ...
    def state(self) -> dict: ...
    def load_state(self, d: dict) -> None: ...


def floor_clip(probs: dict, floor: float) -> dict:
    if floor <= 0.0:
        raise ValueError("floor must be > 0 (I-5: argmax forbidden)")
    keys = list(probs)
    k = len(keys)
    if k == 0:
        raise ValueError("empty probability vector")
    if floor * k >= 1.0:
        return {key: 1.0 / k for key in keys}
    clipped = {key: max(0.0, float(probs[key])) for key in keys}
    total = sum(clipped.values())
    if total <= 0.0:
        return {key: 1.0 / k for key in keys}
    norm = {key: clipped[key] / total for key in keys}
    return {key: (1.0 - floor * k) * norm[key] + floor for key in keys}


def boltzmann(stat: dict, temperature: float) -> dict:
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")
    keys = list(stat)
    m = max(stat.values())
    exps = {k: _exp((stat[k] - m) / temperature) for k in keys}
    total = sum(exps.values())
    return {k: v / total for k, v in exps.items()}


def sample(probs: dict, rng) -> str:
    keys = list(probs)
    weights = [probs[k] for k in keys]
    return rng.choices(keys, weights=weights, k=1)[0]


def _exp(x: float) -> float:
    import math
    if x < -700.0:
        return 0.0
    return math.exp(x)
