"""Bounded LP radius proposal at fixed centers, never a packing certificate.

The native time limit is cooperative, not a hard preemption guarantee. Callers
must share their original search deadline and retain an outer process watchdog.
This module does not use Python or NumPy randomness and does not run joint SLSQP.
"""
import math as _lp_math
import time as _lp_time
import warnings as _lp_warnings

import numpy as np
from scipy.optimize import linprog as _lp_linprog, OptimizeWarning as _LPOptimizeWarning

LP_CONTRACT = "sciel-circle-lp-initializer/1"
LP_MAX_SECONDS = 0.05
_LP_N = 26
_LP_TOLERANCE = 1e-6
_LP_I, _LP_J = np.triu_indices(_LP_N, k=1)
_LP_A = np.zeros((325, _LP_N))
_LP_A[np.arange(325), _LP_I] = 1.0
_LP_A[np.arange(325), _LP_J] = 1.0
_LP_A.flags.writeable = False


class LPInitializationError(RuntimeError):
    """The fixed, bounded, zero-feasible LP contradicted its runtime contract."""


def new_lp_stats():
    return {"lp_calls": 0, "lp_accepted": 0, "lp_fallbacks": 0,
            "lp_deadline_skips": 0, "lp_errors": 0, "lp_time_seconds": 0.0,
            "lp_fallback_reasons": {}}


def _lp_feasible(centers, radii):
    """Independent scalar checks of original walls/pairs; never repair a vector."""
    if radii.shape != (_LP_N,) or not np.isfinite(radii).all() or np.any(radii < 0):
        return False
    for i, ((x, y), radius) in enumerate(zip(centers, radii)):
        if min(x, y, 1 - x, 1 - y) - radius < -_LP_TOLERANCE:
            return False
        for j in range(i):
            if _lp_math.hypot(x - centers[j, 0], y - centers[j, 1]) - radius - radii[j] < -_LP_TOLERANCE:
                return False
    return True


def initialize_radii(params, *, deadline, stats=None, clock=None, linprog_fn=None):
    """Return a copied 78-vector; only accepted LP radii may differ.

    A normal time-limit return or invalid proposal retains the exact original
    guess. Infeasible/unbounded status is impossible for clipped centers and
    therefore raises, as do other unexpected runtime/programming defects.
    A feasible already-computed vector can return after the deadline: the caller
    must consider it and then exit via its existing cooperative deadline check.
    """
    clock = _lp_time.monotonic if clock is None else clock
    linprog_fn = _lp_linprog if linprog_fn is None else linprog_fn
    stats = new_lp_stats() if stats is None else stats
    for key, value in new_lp_stats().items():
        stats.setdefault(key, value)
    started = clock()
    try:
        if type(deadline) not in (float, int) or not _lp_math.isfinite(deadline):
            raise ValueError("LP deadline must be finite")
        original = np.asarray(params)
        if np.iscomplexobj(original):
            raise ValueError("LP initializer parameters must be real")
        original = np.asarray(original, dtype=float).copy()
        if original.shape != (78,) or not np.isfinite(original).all():
            raise ValueError("LP initializer requires 78 finite parameters")
        centers = original[:52].reshape(_LP_N, 2)
        if np.any(centers < 0) or np.any(centers > 1):
            raise ValueError("LP initializer requires centers in the unit square")
        if clock() >= deadline:
            stats["lp_deadline_skips"] += 1
            return original
        wall_upper = np.min(np.column_stack((centers, 1 - centers)), axis=1)
        delta = centers[_LP_I] - centers[_LP_J]
        distances = np.hypot(delta[:, 0], delta[:, 1])
        remaining = deadline - clock()
        if remaining <= 0:
            stats["lp_deadline_skips"] += 1
            return original
        options = {"time_limit": min(LP_MAX_SECONDS, remaining), "threads": 1,
                   "parallel": False, "random_seed": 0}
        stats["lp_calls"] += 1
        # SciPy 1.15.3 forwards these three verified native HiGHS 1.8.0 options.
        # Suppress only that exact, expected forwarding warning; never errors or
        # arbitrary option warnings. The frozen runtime preflight verifies them.
        with _lp_warnings.catch_warnings():
            _lp_warnings.simplefilter("error")
            _lp_warnings.filterwarnings("ignore", category=_LPOptimizeWarning,
                message=r"^Unrecognized options detected: \{'threads': 1, 'parallel': False, 'random_seed': 0\}\. These will be passed to HiGHS verbatim\.$")
            result = linprog_fn(-np.ones(_LP_N), A_ub=_LP_A.copy(), b_ub=distances,
                bounds=np.column_stack((np.zeros(_LP_N), wall_upper)), method="highs", options=options)
        if result.status in (2, 3):
            raise LPInitializationError("zero-feasible bounded LP returned infeasible or unbounded")
        if result.status == 1 and "time limit" in str(result.message).lower():
            reason = "time_limit"
        elif result.status != 0 or not result.success:
            raise LPInitializationError("unexpected LP termination status")
        else:
            try:
                vector = np.asarray(result.x)
                if np.iscomplexobj(vector):
                    vector = None
                else:
                    vector = np.asarray(vector, dtype=float).copy()
            except (ValueError, TypeError, OverflowError):
                vector = None
            if vector is None or vector.shape != (_LP_N,) or not np.isfinite(vector).all():
                reason = "invalid_vector"
            elif not _lp_feasible(centers, vector):
                reason = "infeasible_vector"
            elif (_lp_feasible(centers, original[52:]) and
                  _lp_math.fsum(map(float, vector)) < _lp_math.fsum(map(float, original[52:]))):
                reason = "lower_than_feasible_guess"
            else:
                proposal = original.copy()
                proposal[52:] = vector
                stats["lp_accepted"] += 1
                return proposal
        stats["lp_fallbacks"] += 1
        reasons = stats["lp_fallback_reasons"]
        reasons[reason] = reasons.get(reason, 0) + 1
        return original
    except Exception:
        stats["lp_errors"] += 1
        raise
    finally:
        stats["lp_time_seconds"] += max(0.0, clock() - started)
