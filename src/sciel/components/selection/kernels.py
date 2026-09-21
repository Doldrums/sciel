"""Alternative parent-selection kernels.

stock_70_20_10 mixes elite, remaining and uniform score ranks. UCB1 uses first-play
urgency; ucb_sw uses an event window. Both apply a softmax probability floor so
selection propensities remain measurable."""

from __future__ import annotations

import math
from collections import deque

from ..policy.base import boltzmann, floor_clip
from .novelty_chain import IslandView, Selected


class _KernelBase:
    supports_tilt = False

    def __init__(self, floor: float):
        if floor <= 0.0:
            raise ValueError("floor must be > 0 (I-5)")
        self.floor = floor
        self.visits: dict = {}

    def _finalize(self, rho: dict, view: IslandView, rng, debug: dict) -> Selected:
        rho = floor_clip(rho, self.floor)
        pid = rng.choices(view.ids, weights=[rho[i] for i in view.ids], k=1)[0]
        self.visits[pid] = self.visits.get(pid, 0) + 1
        debug = {**debug, "rho_min": min(rho.values()), "rho_sum": sum(rho.values())}
        return Selected(parent_id=pid, inspiration_ids=view.inspiration_ids.get(pid, []),
                        propensity=rho[pid], debug=debug)


class StockSelectionKernel(_KernelBase):
    def __init__(self, floor: float = 1e-3, elite_ratio: float = 0.2,
                 weights: tuple = (0.7, 0.2, 0.1)):
        super().__init__(floor)
        self.elite_ratio = elite_ratio
        self.w_elite, self.w_rest, self.w_unif = weights

    def select(self, view: IslandView, tilt, rng) -> Selected:
        ids = view.ids
        n = len(ids)
        ranked = sorted(ids, key=lambda i: view.scores[i], reverse=True)
        n_elite = max(1, math.ceil(self.elite_ratio * n))
        elite = set(ranked[:n_elite])
        rest = [i for i in ids if i not in elite]
        rho = {i: 0.0 for i in ids}
        for i in elite:
            rho[i] += self.w_elite / len(elite)
        if rest:
            for i in rest:
                rho[i] += self.w_rest / len(rest)
        else:
            for i in elite:
                rho[i] += self.w_rest / len(elite)
        for i in ids:
            rho[i] += self.w_unif / n
        return self._finalize(rho, view, rng, {"kernel": "stock", "n_elite": n_elite})

    def observe(self, parent_id: str, ev) -> None:
        pass

    def state(self) -> dict:
        return {"kind": "stock", "floor": self.floor, "elite_ratio": self.elite_ratio,
                "weights": [self.w_elite, self.w_rest, self.w_unif],
                "visits": dict(self.visits)}

    def load_state(self, d: dict) -> None:
        self.floor = d["floor"]
        self.elite_ratio = d["elite_ratio"]
        self.w_elite, self.w_rest, self.w_unif = d["weights"]
        self.visits = dict(d["visits"])


class UCB1Kernel(_KernelBase):
    def __init__(self, floor: float = 1e-3, c: float = 1.41421356,
                 fpu: float = 1.0, cap: int = 20, temperature: float = 0.25):
        super().__init__(floor)
        self.c = c
        self.fpu = fpu
        self.cap = cap
        self.temperature = temperature
        self.reward_sum: dict = {}

    def _candidates(self, view: IslandView) -> list:
        ranked = sorted(view.ids, key=lambda i: view.scores[i], reverse=True)
        return ranked[:min(self.cap, len(ranked))]

    def _index(self, i: str, total: int) -> float:
        n = self.visits.get(i, 0)
        if n == 0:
            # UCB1's defining behaviour is to play every arm once before comparing.
            # A fixed fpu of 1.0 sat BELOW the index of a once-played arm with the
            # worst possible outcome (0 + 1.414*sqrt(ln 2) = 1.177), so a fresh
            # program was penalised about 2x against a proven-bad one. The urgency
            # has to dominate whatever the played arms can currently reach.
            return max(self.fpu, self._max_played_index(total) + 1e-9)
        mean = self.reward_sum.get(i, 0.0) / n
        return mean + self.c * math.sqrt(math.log(total + 1) / n)

    def _max_played_index(self, total: int) -> float:
        best = 0.0
        for i, n in self.visits.items():
            if n <= 0:
                continue
            mean = self.reward_sum.get(i, 0.0) / n
            best = max(best, mean + self.c * math.sqrt(math.log(total + 1) / n))
        return best

    def select(self, view: IslandView, tilt, rng) -> Selected:
        cands = self._candidates(view)
        total = sum(self.visits.get(i, 0) for i in cands)
        index = {i: self._index(i, total) for i in cands}
        base = {i: 0.0 for i in view.ids}
        soft = boltzmann(index, self.temperature)
        base.update(soft)
        return self._finalize(base, view, rng, {"kernel": "ucb1", "cands": len(cands)})

    def observe(self, parent_id: str, ev) -> None:
        self.reward_sum[parent_id] = self.reward_sum.get(parent_id, 0.0) + ev.r_norm

    def state(self) -> dict:
        return {"kind": "ucb1", "floor": self.floor, "c": self.c, "fpu": self.fpu,
                "cap": self.cap, "temperature": self.temperature,
                "visits": dict(self.visits), "reward_sum": dict(self.reward_sum)}

    def load_state(self, d: dict) -> None:
        self.floor, self.c, self.fpu = d["floor"], d["c"], d["fpu"]
        self.cap, self.temperature = d["cap"], d["temperature"]
        self.visits = dict(d["visits"])
        self.reward_sum = dict(d["reward_sum"])


class UCBSlidingWindowKernel(_KernelBase):
    def __init__(self, floor: float = 1e-3, c: float = 1.41421356,
                 fpu: float = 1.0, cap: int = 20, temperature: float = 0.25,
                 window: int = 50):
        super().__init__(floor)
        self.c = c
        self.fpu = fpu
        self.cap = cap
        self.temperature = temperature
        self.window = window
        self.events: dict = {}

    def _candidates(self, view: IslandView) -> list:
        ranked = sorted(view.ids, key=lambda i: view.scores[i], reverse=True)
        return ranked[:min(self.cap, len(ranked))]

    def _index(self, i: str, total: int) -> float:
        """Windowed mean reward plus a confidence radius over the PULL count.

        `observe` returns early unless the outcome was an island event, so the buffer
        holds records, not pulls. Using its length as n inverted the exploration
        bonus: an arm pulled 200 times with one record scored 2.29 against 1.54 for an
        arm pulled 3 times with three records, i.e. ~20x the selection weight at
        temperature 0.25 for the arm that had already been tried 200 times.
        `self.visits` has the real pull counts and was maintained but never read."""
        buf = self.events.get(i)
        pulls = self.visits.get(i, 0)
        if pulls <= 0:
            return self.fpu
        mean = (sum(buf) / len(buf)) if buf else 0.0
        return mean + self.c * math.sqrt(math.log(total + 1) / pulls)

    def select(self, view: IslandView, tilt, rng) -> Selected:
        cands = self._candidates(view)
        total = sum(self.visits.get(i, 0) for i in cands)
        index = {i: self._index(i, total) for i in cands}
        base = {i: 0.0 for i in view.ids}
        base.update(boltzmann(index, self.temperature))
        return self._finalize(base, view, rng, {"kernel": "ucb_sw", "cands": len(cands)})

    def observe(self, parent_id: str, ev) -> None:
        if not ev.is_island_event:
            return
        if parent_id not in self.events:
            self.events[parent_id] = deque(maxlen=self.window)
        self.events[parent_id].append(ev.r_norm)

    def state(self) -> dict:
        return {"kind": "ucb_sw", "floor": self.floor, "c": self.c, "fpu": self.fpu,
                "cap": self.cap, "temperature": self.temperature, "window": self.window,
                "visits": dict(self.visits),
                "events": {k: list(v) for k, v in self.events.items()}}

    def load_state(self, d: dict) -> None:
        self.floor, self.c, self.fpu = d["floor"], d["c"], d["fpu"]
        self.cap, self.temperature, self.window = d["cap"], d["temperature"], d["window"]
        self.visits = dict(d["visits"])
        self.events = {k: deque(v, maxlen=self.window) for k, v in d["events"].items()}


def build_kernel(policy: str, cfg: dict):
    floor = cfg.get("selection_floor", 1e-3)
    if policy == "stock_70_20_10":
        return StockSelectionKernel(floor)
    if policy == "ucb1":
        return UCB1Kernel(floor, cap=cfg.get("selection_cap", 20),
                          temperature=cfg.get("selection_temperature", 0.25))
    if policy == "ucb_sw":
        return UCBSlidingWindowKernel(floor, cap=cfg.get("selection_cap", 20),
                                      temperature=cfg.get("selection_temperature", 0.25),
                                      window=cfg.get("selection_window", 50))
    raise NotImplementedError(f"selection_policy '{policy}' unknown")
