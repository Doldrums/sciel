"""Bounded single-circle restart proposal; no optimizer, RNG or certificate.

The caller supplies the existing 78-value uniform draw and its original shared
deadline. An accepted initializer is only tolerance-feasible, not an incumbent
improvement or a guarantee that the subsequent joint SLSQP will help.
"""
import math as _relocation_math
import time as _relocation_time

import numpy as np

RELOCATION_CONTRACT = "sciel-circle-relocation/1"
RELOCATION_PERIOD = 4
RELOCATION_POINTS = 16
_RELOCATION_N = 26
_RELOCATION_TOL = 1e-6


def new_relocation_stats():
    return {"relocation_attempts": 0, "relocation_accepted": 0,
            "relocation_fallbacks": 0, "relocation_deadline_skips": 0,
            "relocation_errors": 0, "relocation_time_seconds": 0.0,
            "relocation_fallback_reasons": {}}


def _relocation_vector(value, name):
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError(name + " must be real")
    array = np.asarray(array, dtype=float).copy()
    if array.shape != (78,) or not np.isfinite(array).all():
        raise ValueError(name + " must have 78 finite parameters")
    return array


def _relocation_feasible(params):
    """Check original geometry independently, without clipping or shrinking."""
    centers, radii = params[:52].reshape(26, 2), params[52:]
    if (not np.isfinite(params).all() or np.any(radii < 0)
            or np.any(centers < -_RELOCATION_TOL)
            or np.any(centers > 1 + _RELOCATION_TOL)
            or np.any(radii > .5 + _RELOCATION_TOL)):
        return False
    for i, ((x, y), radius) in enumerate(zip(centers, radii)):
        if min(x, y, 1 - x, 1 - y) - radius < -_RELOCATION_TOL:
            return False
        for k in range(i):
            if (_relocation_math.hypot(x - centers[k, 0], y - centers[k, 1])
                    - radius - radii[k] < -_RELOCATION_TOL):
                return False
    return True


def relocate_circle(incumbent, noise, default_guess, restart_index, *, deadline,
                    stats=None, clock=None):
    """Copy one restart: every fourth index moves circle (index//4)%26.

    The first 32 components of the supplied [-.06,.06] draw specify 16 locations
    over the whole square. Remove only the selected circle when measuring free
    radius, and use the first maximal nonnegative clearance. Retain the other
    25 circles exactly. No available location or an invalid computed proposal
    returns the exact ordinary clipped guess. Even a lower-sum proposal may be
    used: incumbent retention and improvement checks belong to the caller.

    Fixed work has no nested solver. Expiry before work skips it; a completed
    feasible proposal can return after expiry so the caller can consider it and
    immediately check the shared deadline before dispatching joint SLSQP.
    """
    clock = _relocation_time.monotonic if clock is None else clock
    stats = new_relocation_stats() if stats is None else stats
    for key, value in new_relocation_stats().items():
        stats.setdefault(key, value)
    started = clock()
    try:
        if (type(restart_index) is not int or restart_index < 0
                or type(deadline) not in (int, float)
                or not _relocation_math.isfinite(deadline)):
            raise ValueError("relocation index/deadline invalid")
        original = _relocation_vector(default_guess, "default guess")
        base = _relocation_vector(incumbent, "incumbent")
        draw = _relocation_vector(noise, "noise")
        if np.any(draw < -.06) or np.any(draw > .06):
            raise ValueError("relocation requires the original [-.06,.06] draw")
        if not _relocation_feasible(base):
            raise ValueError("relocation incumbent must be tolerance-feasible")
        if restart_index % RELOCATION_PERIOD:
            return original
        if clock() >= deadline:
            stats["relocation_deadline_skips"] += 1
            return original
        stats["relocation_attempts"] += 1
        chosen = (restart_index // RELOCATION_PERIOD) % _RELOCATION_N
        centers, radii = base[:52].reshape(26, 2), base[52:]
        points = ((draw[:32] + .06) / .12).reshape(RELOCATION_POINTS, 2)
        best_index, best_radius = None, -1.0
        for index, (x, y) in enumerate(points):
            clearance = min(x, y, 1 - x, 1 - y)
            for k in range(_RELOCATION_N):
                if k != chosen:
                    clearance = min(clearance, _relocation_math.hypot(
                        x - centers[k, 0], y - centers[k, 1]) - radii[k])
            if clearance >= 0 and (best_index is None or clearance > best_radius):
                best_index, best_radius = index, float(clearance)
        reason = "no_nonnegative_clearance"
        if best_index is not None:
            proposal = base.copy()
            proposal[2 * chosen:2 * chosen + 2] = points[best_index]
            proposal[52 + chosen] = best_radius
            if _relocation_feasible(proposal):
                stats["relocation_accepted"] += 1
                return proposal
            reason = "infeasible_proposal"
        stats["relocation_fallbacks"] += 1
        reasons = stats["relocation_fallback_reasons"]
        reasons[reason] = reasons.get(reason, 0) + 1
        return original
    except Exception:
        stats["relocation_errors"] += 1
        raise
    finally:
        stats["relocation_time_seconds"] += max(0.0, clock() - started)
