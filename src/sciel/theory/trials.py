"""Knowledge items judged by their sibling pairs, and the generator's forecasts judged against the base rate.

Promotion and withdrawal of an item are selection rules with no error claim. The inferential report is e-BH at a
fixed report time over every item ever opened, each with its value stopped at its own verdict.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

from ..causal.sequential import continued, e_bh, mixture

ALPHA = 0.05
PROMOTE = 20.0
MAX_PAIRS = 80


@dataclass
class Trial:
    item: str
    kind: str
    status: str = "open"
    n: int = 0
    mean: float = 0.0
    e: float = 1.0
    low: float = -1.0
    high: float = 1.0
    applied_mean: float | None = None
    prior: list = field(default_factory=list)


class Trials:
    def __init__(self):
        self.trials: dict[str, Trial] = {}
        self.forecasts: list[list] = []

    def update(self, pairs: list[dict]) -> list[Trial]:
        """Recompute every open trial from its pairs in launch order; return trials whose verdict changed."""
        by_item: dict[str, list[dict]] = {}
        for pair in pairs:
            by_item.setdefault(pair["item"], []).append(pair)
        changed = []
        for item, rows in by_item.items():
            trial = self.trials.setdefault(item, Trial(item=item, kind=rows[0]["kind"]))
            if trial.status != "open":
                continue
            d = np.array([r["d"] for r in rows])
            prior = np.array(trial.prior) if trial.prior else None
            s = continued(mixture(d, 1.0, ALPHA)["log_wealth"], prior, d, ALPHA)
            applied = [r["d"] for r in rows if r["applied"]]
            trial.n, trial.mean, trial.e, trial.low, trial.high = len(rows), s["mean"], s["e"], s["low"], s["high"]
            trial.applied_mean = float(np.mean(applied)) if applied else None
            if trial.e >= PROMOTE:
                trial.status = "culture" if trial.mean > 0 else "retired"
            elif trial.n >= MAX_PAIRS:
                trial.status = "not_established"
            if trial.status != "open":
                changed.append(trial)
        return changed

    def wealth(self, pairs: list[dict]) -> dict[str, list]:
        """Each item's betting wealth over its own pairs and any inherited wealth, for the next generation."""
        out = {}
        for item, trial in self.trials.items():
            d = np.array([p["d"] for p in pairs if p["item"] == item])
            prior = np.array(trial.prior) if trial.prior else None
            out[item] = continued(mixture(d, 1.0, ALPHA)["log_wealth"], prior, d, ALPHA)["log_wealth"].tolist()
        return out

    def inherit(self, item: str, kind: str, prior: list) -> None:
        self.trials[item] = Trial(item=item, kind=kind, prior=prior)

    def status(self, item: str) -> str | None:
        trial = self.trials.get(item)
        return trial.status if trial else None

    def report(self) -> dict:
        """e-BH over every item ever opened, open ones at their current value, closed ones stopped at their verdict."""
        evalues = {item: t.e for item, t in self.trials.items()}
        return {"items": len(evalues), "rejected": sorted(e_bh(evalues, ALPHA)) if evalues else []}

    def forecast(self, q: float, event: bool) -> None:
        self.forecasts.append([float(q), float(event)])

    def forecast_skill(self) -> dict:
        """Brier of the generator's forecasts against the running base rate known before each forecast."""
        if not self.forecasts:
            return {"n": 0, "e": 1.0, "mean": 0.0}
        q, y = np.array(self.forecasts).T
        base = (np.concatenate([[0.0], np.cumsum(y)[:-1]]) + 0.5) / (np.arange(len(y)) + 1.0)
        gain = (base - y) ** 2 - (q - y) ** 2
        s = mixture(gain, 1.0, ALPHA)
        return {"n": s["n"], "e": s["e"], "mean": s["mean"]}

    def state(self) -> dict:
        return {"trials": [asdict(t) for t in self.trials.values()], "forecasts": self.forecasts}

    def load_state(self, data: dict) -> None:
        self.trials = {t["item"]: Trial(**t) for t in data["trials"]}
        self.forecasts = [list(f) for f in data["forecasts"]]
