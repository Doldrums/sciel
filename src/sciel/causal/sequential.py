"""Anytime-valid inference for claims: DR contrasts with predictable outcome models, betting e-processes and e-LOND.

The contrast of an action is its value minus the uniform mixture of the other actions of the same decision, so the
target does not drift when the theory tilts the logging policy. Each row enters in decision order with an outcome that
is closed when the row is written (the child's own event), and its outcome model uses only earlier rows.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

GRID = np.linspace(-1.0, 1.0, 81)
CAP = 0.5
PRIOR = 2.0


def contrasts(rows: list, action: str) -> tuple[np.ndarray, np.ndarray]:
    """Doubly robust pseudo-outcomes of `action` against the other actions, with their known bounds."""
    sums, counts = defaultdict(float), defaultdict(int)
    total, seen = 0.0, 0
    psi, bound = [], []
    for row in rows:
        others = [b for b in row.distribution if b != action]
        if action not in row.distribution or not others:
            continue
        mean = (total + PRIOR * 0.5) / (seen + PRIOR)

        def q(b):
            return (sums[b] + PRIOR * mean) / (counts[b] + PRIOR)

        y = row.reward if row.scope == "lineage" else row.outcome
        value = q(action) + (row.choice == action) * (y - q(action)) / row.distribution[action]
        rest = sum(q(b) + (row.choice == b) * (y - q(b)) / row.distribution[b] for b in others) / len(others)
        psi.append(value - rest)
        bound.append(1.0 + 1.0 / min(row.distribution.values()))
        sums[row.choice] += y
        counts[row.choice] += 1
        total += y
        seen += 1
    return np.asarray(psi, dtype=float), np.asarray(bound, dtype=float)


def _log_capital(x: np.ndarray, bound: np.ndarray, sign: float) -> np.ndarray:
    """Log wealth of betting sign*(x - m) with a predictable plug-in bet, for every grid value m at once."""
    t = np.arange(1, len(x) + 1)[:, None]
    past_mean = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x, 0)[:-1] / t[:-1]])
    past_sq = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(x ** 2, 0)[:-1] / t[:-1]])
    var = np.maximum(past_sq - past_mean ** 2, 0.0) + 1.0 / t
    limit = CAP / (bound[:, None] + np.abs(GRID)[None])
    lam = np.clip(sign * past_mean / (var + past_mean ** 2), 0.0, limit)
    return np.cumsum(np.log1p(sign * lam * x), 0)


def sequence(psi: np.ndarray, bound: np.ndarray, alpha: float) -> dict:
    """Hedged capital over the grid: a running-intersection confidence sequence and the e-process at zero."""
    if len(psi) == 0:
        return {"n": 0, "mean": 0.0, "se": 0.0, "low": -1.0, "high": 1.0, "e_peak": 1.0}
    x = psi[:, None] - GRID[None]
    up, down = _log_capital(x, bound, 1.0), _log_capital(x, bound, -1.0)
    capital = np.logaddexp(up, down) - math.log(2.0)
    rejected = (capital >= math.log(1.0 / alpha)).any(0)
    kept = GRID[~rejected]
    zero = int(np.argmin(np.abs(GRID)))
    se = float(psi.std(ddof=1) / math.sqrt(len(psi))) if len(psi) > 1 else 1.0
    return {"n": len(psi), "mean": float(psi.mean()), "se": se,
            "low": float(kept.min()) if len(kept) else float(psi.mean()),
            "high": float(kept.max()) if len(kept) else float(psi.mean()),
            "e_peak": float(np.exp(capital[:, zero].max()))}


def lond_level(alpha: float, index: int, rejections: int) -> float:
    """e-LOND level of the index-th hypothesis (1-based) given the rejections among earlier hypotheses."""
    gamma = 6.0 / (math.pi ** 2 * index ** 2)
    return alpha * gamma * (rejections + 1)
