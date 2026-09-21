"""Selection-mode tilts for the pi controller.

A Tilt is the injection point for a selection mode: novelty_chain blends the tilt's
target distribution into the kernel's rho as rho = (1 - beta) * rho + beta * tilt(view).
The tilt carries its own beta (mixing strength); target(view) returns a non-negative
distribution over view.ids summing to 1 over its support.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class Tilt:
    beta: float
    target: Callable

    def __post_init__(self) -> None:
        if not 0.0 <= self.beta <= 1.0:
            raise ValueError("tilt beta must be in [0, 1]")

    def __call__(self, view) -> dict:
        return self.target(view)


def elite_target(view, k: int) -> dict:
    ranked = sorted(view.ids, key=lambda i: view.scores[i], reverse=True)
    elite = ranked[:max(1, min(k, len(ranked)))]
    w = 1.0 / len(elite)
    return {i: w for i in elite}


def uniform_target(view) -> dict:
    n = len(view.ids)
    if n == 0:
        return {}
    w = 1.0 / n
    return {i: w for i in view.ids}
