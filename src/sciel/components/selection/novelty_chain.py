"""Parent selection C1: novelty_chain in temperature form, phase-1 index policy."""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..policy.base import floor_clip


@dataclass
class IslandView:
    ids: list
    scores: dict
    inspiration_ids: dict


@dataclass
class Selected:
    parent_id: str
    inspiration_ids: list
    propensity: float
    debug: dict


class NoveltyChainKernel:
    supports_tilt = True

    def __init__(self, floor: float = 1e-3, tau_floor: float = 1e-3):
        if floor <= 0.0:
            raise ValueError("floor must be > 0 (I-5)")
        self.floor = floor
        self.tau_floor = tau_floor
        self.visits: dict = {}
        self.reward_sum: dict = {}

    def _tau(self, scores: list) -> float:
        if len(scores) < 2:
            return max(self.tau_floor, 1.0)
        mean = sum(scores) / len(scores)
        var = sum((s - mean) ** 2 for s in scores) / len(scores)
        return max(self.tau_floor, math.sqrt(var))

    def select(self, view: IslandView, tilt, rng, *, before_visit=None) -> Selected:
        ids = view.ids
        scores = [view.scores[i] for i in ids]
        tau = self._tau(scores)
        m = max(scores)
        weights = {}
        for i in ids:
            v = self.visits.get(i, 0)
            weights[i] = math.exp((view.scores[i] - m) / tau) / (1 + v)
        total = sum(weights.values())
        rho = {i: weights[i] / total for i in ids}
        if tilt is not None:
            t = tilt(view)
            rho = {i: (1 - tilt.beta) * rho[i] + tilt.beta * t.get(i, 0.0) for i in ids}
        rho = floor_clip(rho, self.floor)
        pid = rng.choices(ids, weights=[rho[i] for i in ids], k=1)[0]
        if before_visit is not None:
            # Experimental diagnostics see the exact sampled distribution and
            # pre-decision visits. Copies prevent a logger from changing policy.
            before_visit(pid, dict(rho), {"tau": tau})
        self.visits[pid] = self.visits.get(pid, 0) + 1
        return Selected(parent_id=pid, inspiration_ids=view.inspiration_ids.get(pid, []),
                        propensity=rho[pid],
                        debug={"tau": tau, "n": len(ids), "rho_min": min(rho.values()),
                               "rho_sum": sum(rho.values())})

    def observe(self, parent_id: str, ev) -> None:
        self.reward_sum[parent_id] = self.reward_sum.get(parent_id, 0.0) + ev.r_norm

    def state(self) -> dict:
        return {"visits": dict(self.visits), "reward_sum": dict(self.reward_sum),
                "floor": self.floor, "tau_floor": self.tau_floor}

    def load_state(self, d: dict) -> None:
        self.visits = dict(d["visits"])
        self.reward_sum = dict(d["reward_sum"])
        self.floor = d["floor"]
        self.tau_floor = d["tau_floor"]
