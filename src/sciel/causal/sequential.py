"""Anytime-valid inference for claims: DR contrasts with predictable outcome models, betting e-processes and e-LOND.

The contrast of an action is its value minus the uniform mixture of the other actions of the same decision when every
row offers the same alternatives, and minus one reference action offered in every row when the alternatives change, so
the target does not drift when the theory tilts the logging policy or arms enter and leave. Rows enter in decision order
once their outcomes are settled, so the order never depends on an outcome, and each row's outcome model uses only
earlier rows.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

GRID = np.linspace(-1.0, 1.0, 81)
CAP = 0.5
PRIOR = 2.0


def comparator(rows: list, action: str) -> tuple[bool, str | None]:
    """Whether `action` has a fixed comparison target in these rows: the shared alternatives or a reference action."""
    others = [frozenset(r.distribution) - {action} for r in rows if action in r.distribution]
    others = [o for o in others if o]
    if not others or len(set(others)) == 1:
        return True, None
    common = frozenset.intersection(*others)
    reference = next((a for r in rows if action in r.distribution for a in r.distribution if a in common), None)
    return reference is not None, reference


def contrasts(rows: list, action: str, reference: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Doubly robust pseudo-outcomes of `action` against the other actions or a reference action, with known bounds."""
    sums, counts = defaultdict(float), defaultdict(int)
    total, seen = 0.0, 0
    psi, bound = [], []
    for row in rows:
        others = [reference] if reference is not None else [b for b in row.distribution if b != action]
        if action not in row.distribution or not others or any(b not in row.distribution for b in others):
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


LAMBDAS = np.array([v for v in np.linspace(-0.9, 0.9, 19) if abs(v) > 1e-9])


def mixture(psi: np.ndarray, bound, alpha: float) -> dict:
    """Equal-weight mixture of fixed bets on every grid value m: the current e-value at zero and the current
    confidence set, both valid at any stopping time; the log wealth per (m, bet) lets a later run continue it."""
    if len(psi) == 0:
        return {"n": 0, "mean": 0.0, "se": 0.0, "low": -1.0, "high": 1.0, "e": 1.0, "empty": False,
                "log_wealth": np.zeros((len(GRID), len(LAMBDAS)))}
    bound = np.broadcast_to(np.asarray(bound, dtype=float), psi.shape)
    x = psi[:, None, None] - GRID[None, :, None]
    lam = LAMBDAS[None, None, :] / (bound[:, None, None] + np.abs(GRID)[None, :, None])
    log_wealth = np.log1p(lam * x).sum(0)
    return {**_summary(psi, log_wealth, alpha), "log_wealth": log_wealth}


def _summary(psi: np.ndarray, log_wealth: np.ndarray, alpha: float) -> dict:
    capital = np.logaddexp.reduce(log_wealth, axis=1) - math.log(log_wealth.shape[1])
    kept = GRID[capital < math.log(1.0 / alpha)]
    zero = int(np.argmin(np.abs(GRID)))
    step = GRID[1] - GRID[0]
    se = float(psi.std(ddof=1) / math.sqrt(len(psi))) if len(psi) > 1 else 1.0
    return {"n": len(psi), "mean": float(psi.mean()), "se": se,
            "low": float(kept.min() - step) if len(kept) else 0.0,
            "high": float(kept.max() + step) if len(kept) else 0.0,
            "e": float(np.exp(capital[zero])), "empty": not len(kept)}


def continued(log_wealth: np.ndarray, prior: np.ndarray | None, psi: np.ndarray, alpha: float) -> dict:
    """The same betting martingale carried on from an earlier run's final wealth."""
    total = log_wealth if prior is None else log_wealth + prior
    return {**_summary(psi if len(psi) else np.zeros(1), total, alpha), "log_wealth": total}


def e_bh(evalues: dict, alpha: float, weights: dict | None = None) -> set:
    """e-BH: reject the largest k with weighted e-value at least K/(alpha*k); valid under any dependence."""
    keys = list(evalues)
    k_total = len(keys)
    weights = weights or {k: 1.0 for k in keys}
    ranked = sorted(keys, key=lambda k: -weights[k] * evalues[k])
    rejected = 0
    for k in range(1, k_total + 1):
        if weights[ranked[k - 1]] * evalues[ranked[k - 1]] >= k_total / (alpha * k):
            rejected = k
    return set(ranked[:rejected])


def lond_level(alpha: float, index: int, rejections: int) -> float:
    """e-LOND level of the index-th hypothesis (1-based) given the rejections among earlier hypotheses."""
    gamma = 6.0 / (math.pi ** 2 * index ** 2)
    return alpha * gamma * (rejections + 1)
