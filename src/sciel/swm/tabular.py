"""TabPFN forecast of whether an operator step raises its lineage best, from what is known when the operator is chosen:
the parent's rank and gap in its population, how often it was already expanded, the run's phase, and the operator.

A transformer pretrained on synthetic tables forecasts from a few dozen rows without training and takes the steps of
earlier runs as context, so it stands between the base rate of a cold start and the world model's network, which
needs a run's worth of steps. Optional: without the tabpfn package the world model gates its network against base
rate and cell means alone.
"""

from __future__ import annotations

import math

import numpy as np

DECISION = "operator"
FEATURES = 5
MAX_CONTEXT = 1500
ESTIMATORS = 4


def available() -> bool:
    try:
        import tabpfn  # noqa: F401
    except ImportError:
        return False
    return True


def describe(db, parent: str, population: list, expansions: int, context: list) -> list:
    """Rank of the parent in its population, its relative gap to the population best, log of earlier expansions,
    and the run's phase and third."""
    score = db._all[parent].score
    pool = [db._all[p].score for p in population if p in db._all and math.isfinite(db._all[p].score)]
    if not math.isfinite(score):
        return [0.0, 1.0, math.log1p(expansions), *context]
    pool = pool or [score]
    best = max(pool)
    return [float(np.mean(np.array(pool) <= score)), (best - score) / (abs(best) + 1e-12), math.log1p(expansions),
            *context]


class Tabular:
    """Rows are [features..., action, outcome]; the context keeps the latest MAX_CONTEXT, inherited rows first."""

    def __init__(self, rows: list, seed: int, cache: bool = False):
        rows = rows[-MAX_CONTEXT:]
        self.actions = sorted({r[FEATURES] for r in rows})
        y = np.array([r[FEATURES + 1] for r in rows], dtype=int)
        self.rate = float(y.mean()) if len(y) else 0.0
        self.model = None
        if 0 < y.sum() < len(y):
            from tabpfn import TabPFNClassifier
            # A cached context makes the few-row queries of an acting step cheap; out-of-fold scoring queries once.
            self.model = TabPFNClassifier(n_estimators=ESTIMATORS, ignore_pretraining_limits=True, random_state=seed,
                                          fit_mode="fit_with_cache" if cache else "fit_preprocessors")
            self.model.fit(self._x(rows), y)

    def _x(self, rows) -> np.ndarray:
        return np.array([list(r[:FEATURES]) + [float(r[FEATURES] == a) for a in self.actions] for r in rows],
                        dtype=np.float32)

    def predict(self, rows: list) -> np.ndarray:
        if self.model is None or not rows:
            return np.full(len(rows), self.rate)
        return self.model.predict_proba(self._x(rows))[:, 1]
