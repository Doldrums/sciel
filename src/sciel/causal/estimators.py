"""Off-policy value of one decision type from logged propensities: self-normalized IPW and doubly robust."""

from __future__ import annotations

import math
from collections import defaultdict

Z = 1.96
PRIOR = 2.0


def _outcome_model(rows) -> dict:
    by_action, by_cell = defaultdict(list), defaultdict(list)
    for r in rows:
        by_action[r.choice].append(r.outcome)
        by_cell[(r.context, r.choice)].append(r.outcome)
    mean = sum(r.outcome for r in rows) / len(rows)
    action = {a: (sum(v) + PRIOR * mean) / (len(v) + PRIOR) for a, v in by_action.items()}
    return {cell: (sum(v) + PRIOR * action[cell[1]]) / (len(v) + PRIOR) for cell, v in by_cell.items()} | {
        ("*", a): m for a, m in action.items()} | {("*", "*"): mean}


def _q(model, context, action) -> float:
    return model.get((context, action), model.get(("*", action), model[("*", "*")]))


def _clustered(values, clusters) -> tuple:
    n = len(values)
    mean = sum(values) / n
    totals = defaultdict(float)
    for v, c in zip(values, clusters):
        totals[c] += v - mean
    clustered = sum(t * t for t in totals.values()) / (n * n)
    independent = sum((v - mean) ** 2 for v in values) / (n * n)
    return mean, math.sqrt(max(clustered, independent))


def evaluate(rows, target) -> dict:
    """Value of `target(row) -> distribution` against the logging policy that produced the rows."""
    if not rows:
        return {"n": 0}
    model = _outcome_model(rows)
    weights = [target(r).get(r.choice, 0.0) / r.propensity for r in rows]
    total = sum(weights)
    snipw = sum(w * r.outcome for w, r in zip(weights, rows)) / total if total > 0 else float("nan")
    dr_terms = []
    for w, r in zip(weights, rows):
        policy = target(r)
        direct = sum(p * _q(model, r.context, a) for a, p in policy.items())
        dr_terms.append(direct + w * (r.outcome - _q(model, r.context, r.choice)))
    dr, se = _clustered(dr_terms, [r.island for r in rows])
    ess = total * total / sum(w * w for w in weights) if total > 0 else 0.0
    return {"n": len(rows), "ess": ess, "snipw": snipw, "dr": dr, "low": dr - Z * se, "high": dr + Z * se}


def action_effect(rows, action: str, q=None) -> dict:
    """Effect of always taking `action` versus the logging policy, paired per row; `q` is an optional per-row outcome model."""
    if not rows:
        return {"n": 0}
    model = _outcome_model(rows)
    diffs = []
    for r in rows:
        learned = q(r, action) if q is not None else None
        base = _q(model, r.context, action) if learned is None else learned
        dr_action = base + ((r.outcome - base) / r.propensity if r.choice == action else 0.0)
        diffs.append(dr_action - r.outcome)
    effect, se = _clustered(diffs, [r.island for r in rows])
    taken = sum(r.choice == action for r in rows)
    return {"n": len(rows), "taken": taken, "effect": effect, "se": se,
            "low": effect - Z * se, "high": effect + Z * se}
