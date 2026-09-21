"""Shared per-program feature builder for the population Set-Transformer -- the single
source of truth used by BOTH offline distillation training (scripts/train_pop_transformer)
and live in-run inference (parent-selection tilt), so features can't drift.

All features are available BOTH offline (from a run journal's lineage) AND live (from the
in-memory Program set), which rules out code-length/grid features (not journalled) and
per-program time (ill-defined for a whole-island set):
    [score, gen, rank_frac, score_minus_setmax, score_minus_setmean]
Set-relative features (rank, gaps to the set best/mean) make each program aware of where
it stands in the population -- the "meta-cognition" signal.
"""

from __future__ import annotations

import numpy as np

N_POP_FEATURES = 5


def pop_features(items) -> np.ndarray:
    """items: list of dicts with 'score' and 'gen'. Returns (N, N_POP_FEATURES)."""
    scores = np.array([float(it["score"]) for it in items], dtype=float)
    gens = np.array([float(it.get("gen", 0)) for it in items], dtype=float)
    smax = scores.max() if len(scores) else 0.0
    smean = scores.mean() if len(scores) else 0.0
    order = (-scores).argsort()                        # rank_frac: 0=best, 1=worst
    rank = np.empty(len(scores)); rank[order] = np.arange(len(scores))
    rank_frac = rank / max(1, len(scores) - 1)
    return np.stack([scores, gens, rank_frac, scores - smax, scores - smean], axis=1)
