"""Operator bandit with event-window maxima and a Boltzmann probability floor."""

from __future__ import annotations

from collections import deque
from copy import deepcopy

from .prompt.sources import source_digest

from ..components.policy.base import boltzmann, floor_clip

ARMS = ["diff", "full_rewrite", "error_repair"]


def usable_inspirations(parent_id, parent_code, rows, limit=3):
    """Freeze distinct donor sources; similarity or score never disqualifies one.

    An inspiration is available input, not evidence that a child used its code.
    Preserve optional legacy metadata without inventing a program identity.
    """
    if type(limit) is not int or limit < 0:
        raise ValueError('invalid crossover donor limit')
    if limit == 0:
        return []
    seen_sources = {source_digest(parent_code)}
    seen_ids = {parent_id}
    result = []
    for row in rows:
        if type(row) is not dict or type(row.get('code')) is not str or not row['code'].strip():
            continue
        ident = row.get('program_id')
        digest = source_digest(row['code'])
        if row.get('source_sha256') is not None and row['source_sha256'] != digest:
            raise ValueError('crossover donor source/provenance hash mismatch')
        if digest in seen_sources or (ident is not None and ident in seen_ids):
            continue
        result.append({**deepcopy(row), 'source_sha256': digest})
        seen_sources.add(digest)
        if ident is not None:
            seen_ids.add(ident)
        if len(result) >= limit:
            break
    return result


class OpBandit:
    def __init__(self, temperature: float = 0.5, floor: float = 0.05,
                 window: int = 50, arms: list | None = None,
                 intensify_boost: float = 0.5):
        if floor <= 0.0:
            raise ValueError("floor must be > 0 (I-5)")
        self.temperature = temperature
        self.floor = floor
        self.window = window
        self.intensify_boost = intensify_boost
        self.arms = list(arms) if arms is not None else list(ARMS)
        self.events = {a: deque(maxlen=window) for a in self.arms}

    def _available(self, fresh_failure: bool, cross_available: bool = True) -> list:
        return [a for a in self.arms
                if (a != 'error_repair' or fresh_failure)
                and (a != 'cross' or cross_available)]

    def _window_max(self, arm: str) -> float:
        buf = self.events[arm]
        return max(buf) if buf else 0.0

    def pick(self, fresh_failure: bool, rng, intensify: bool = False,
             *, cross_available: bool = True) -> tuple:
        arms = self._available(fresh_failure, cross_available)
        if not arms:
            raise ValueError('no eligible mutation operators')
        raw = {a: self._window_max(a) for a in arms}
        stat = raw
        if intensify and "diff" in raw:
            # exploit hold: a strong incumbent has stalled. Bias the local-refinement
            # arm so budget deepens the construction instead of high-variance rewrites
            # -- window-max reward structurally favors rewriters (one lucky jump), which
            # starves diff (small, consistent gains) exactly when refinement is needed.
            stat = dict(raw)
            stat["diff"] = max(raw.values()) + self.intensify_boost
        p = floor_clip(boltzmann(stat, self.temperature), self.floor)
        op = rng.choices(arms, weights=[p[a] for a in arms], k=1)[0]
        # report the true window-max reward, never the synthetic intensify boost
        return op, p[op], {a: raw[a] for a in arms}

    def observe(self, op_used: str, ev) -> None:
        if op_used in self.events and ev.is_island_event:
            self.events[op_used].append(ev.r_norm)

    def state(self) -> dict:
        return {"temperature": self.temperature, "floor": self.floor,
                "window": self.window, "intensify_boost": self.intensify_boost,
                "events": {a: list(self.events[a]) for a in self.arms}}

    def load_state(self, d: dict) -> None:
        self.temperature = d["temperature"]
        self.floor = d["floor"]
        self.window = d["window"]
        self.intensify_boost = d.get("intensify_boost", self.intensify_boost)
        self.events = {a: deque(d["events"].get(a, []), maxlen=self.window)
                       for a in self.arms}
