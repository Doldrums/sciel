"""Hand-authored H30 template: deadline-aware SLSQP for 26 circles.

This standalone file needs only NumPy and SciPy. It preserves the archived
winner's initial grid, global NumPy restart draws, 25-restart limit, maxiter and
ftol; it is a new implementation, not an edited historical search result.

The deadline is cooperative: checks in Python objective/constraint/Jacobian and
callback functions cannot preempt a stalled native solver. Keep an outer process
timeout. An initial feasible packing survives even when no solve completes.

Incumbents must be finite, have nonnegative radii, and satisfy the canonical
Circle evaluator's 1e-6 wall/pair tolerance. No invalid state is projected,
shrunk or otherwise repaired. Tolerance-feasible is NOT strict certification.
The constraint functions themselves retain the original zero-tolerance
inequalities; the tolerance applies only to incumbent admission.
"""

from __future__ import annotations

import math
import time

import numpy as np
from scipy.optimize import minimize


N_CIRCLES = 26
N_PARAMS = 3 * N_CIRCLES
FEASIBILITY_TOL = 1e-6
_PAIR_I, _PAIR_J = np.triu_indices(N_CIRCLES, k=1)
_N_WALLS = 4 * N_CIRCLES

# Updated in place after every call, including an unexpected solver exception.
# This is diagnostic state for a single synchronous invocation, not thread-local.
LAST_STATS: dict = {}


class Deadline(RuntimeError):
    """Cooperative deadline raised only by this template's checks."""


def _parameters(params):
    values = np.asarray(params)
    if np.iscomplexobj(values):
        raise ValueError("packing parameters must be real")
    values = np.asarray(values, dtype=float)
    if values.shape != (N_PARAMS,) or not np.isfinite(values).all():
        raise ValueError(f"packing parameters must be finite with shape ({N_PARAMS},)")
    return values


def initial_parameters():
    """The archived recipe: 5x5 centres, circle 26 at (0,0), wall-clipped .01 radii."""
    centers = np.zeros((N_CIRCLES, 2), dtype=float)
    for row in range(5):
        for col in range(5):
            centers[5 * row + col] = ((col + 0.5) / 5, (row + 0.5) / 5)
    radii = np.full(N_CIRCLES, 0.01)
    radii = np.minimum(radii, np.min(np.column_stack((centers, 1 - centers)), axis=1))
    return np.concatenate((centers.ravel(), radii))


def objective(params):
    """Minimise minus the sum of radii."""
    values = _parameters(params)
    return -float(np.sum(values[2 * N_CIRCLES:]))


def objective_jac(params):
    _parameters(params)
    gradient = np.zeros(N_PARAMS, dtype=float)
    gradient[2 * N_CIRCLES:] = -1.0
    return gradient


def packing_constraints(params):
    """Original distance inequalities, in wall-per-circle then i<j pair order."""
    values = _parameters(params)
    centers = values[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
    radii = values[2 * N_CIRCLES:]
    x, y = centers.T
    walls = np.column_stack((x - radii, 1 - x - radii,
                             y - radii, 1 - y - radii)).ravel()
    differences = centers[_PAIR_I] - centers[_PAIR_J]
    distances = np.sqrt(np.sum(differences ** 2, axis=1))
    pairs = distances - radii[_PAIR_I] - radii[_PAIR_J]
    return np.concatenate((walls, pairs))


def packing_jacobian(params):
    """Analytic Jacobian; use the zero norm subgradient at coincident centres.

    Euclidean distance is not differentiable at coincidence. Zero is a specified
    subgradient there; it does not promise that SLSQP can separate coincident
    positive-radius circles. Radius derivatives remain -1 for both circles.
    """
    values = _parameters(params)
    centers = values[:2 * N_CIRCLES].reshape(N_CIRCLES, 2)
    jac = np.zeros((_N_WALLS + len(_PAIR_I), N_PARAMS), dtype=float)
    circles = np.arange(N_CIRCLES)
    jac[4 * circles, 2 * circles] = 1.0
    jac[4 * circles + 1, 2 * circles] = -1.0
    jac[4 * circles + 2, 2 * circles + 1] = 1.0
    jac[4 * circles + 3, 2 * circles + 1] = -1.0
    for offset in range(4):
        jac[4 * circles + offset, 2 * N_CIRCLES + circles] = -1.0
    differences = centers[_PAIR_I] - centers[_PAIR_J]
    distances = np.sqrt(np.sum(differences ** 2, axis=1))
    directions = np.divide(differences, distances[:, None],
                           out=np.zeros_like(differences),
                           where=distances[:, None] > 0.0)
    rows = _N_WALLS + np.arange(len(_PAIR_I))
    for dim in range(2):
        jac[rows, 2 * _PAIR_I + dim] = directions[:, dim]
        jac[rows, 2 * _PAIR_J + dim] = -directions[:, dim]
    jac[rows, 2 * N_CIRCLES + _PAIR_I] = -1.0
    jac[rows, 2 * N_CIRCLES + _PAIR_J] = -1.0
    return jac


def is_feasible(params):
    """Match canonical tolerance admission without altering the proposed state."""
    try:
        values = _parameters(params)
    except (ValueError, TypeError, OverflowError):
        return False
    if np.any(values[2 * N_CIRCLES:] < 0.0):
        return False
    # Positive radii plus walls imply bounded centres. Check those bounds before
    # distances so arbitrarily large but finite rejected inputs cannot overflow.
    centers = values[:2 * N_CIRCLES]
    if np.any(centers < -FEASIBILITY_TOL) or np.any(centers > 1 + FEASIBILITY_TOL):
        return False
    if np.any(values[2 * N_CIRCLES:] > 0.5 + FEASIBILITY_TOL):
        return False
    return bool(np.all(packing_constraints(values) >= -FEASIBILITY_TOL))


class Incumbent:
    """Copy and retain only strictly better tolerance-feasible observations.

    Counts describe observations, not distinct parameter vectors. Equal-valued
    observations never replace the incumbent or consume an improvement count.
    """

    def __init__(self, initial):
        if not is_feasible(initial):
            raise ValueError("initial incumbent must be tolerance-feasible")
        self._params = _parameters(initial).copy()
        self._sum = float(np.sum(self._params[2 * N_CIRCLES:]))
        self.candidate_checks = 1
        self.feasible_states = 1
        self.improvements = 0

    def consider(self, params):
        self.candidate_checks += 1
        if not is_feasible(params):
            return False
        values = _parameters(params)
        self.feasible_states += 1
        total = float(np.sum(values[2 * N_CIRCLES:]))
        if total <= self._sum:
            return False
        self._params = values.copy()
        self._sum = total
        self.improvements += 1
        return True

    def result(self):
        return (self._params[:2 * N_CIRCLES].reshape(N_CIRCLES, 2).copy(),
                self._params[2 * N_CIRCLES:].copy(), self._sum)


def solve_packing(*, budget_seconds=27.0, max_restarts=25, maxiter=15000):
    """Run bounded restarts with a cooperative deadline and a valid fallback.

    NumPy's global RNG intentionally matches the archived restart recipe. This
    function never seeds it. Caller-controlled seeding is a separate experimental
    contract. A returned unsuccessful SLSQP result may still improve the incumbent
    if independently tolerance-feasible; its success flag is not a certificate.
    """
    if isinstance(budget_seconds, bool) or not isinstance(budget_seconds, (int, float)) \
            or not math.isfinite(budget_seconds) or budget_seconds < 0:
        raise ValueError("budget_seconds must be a finite nonnegative number")
    if type(max_restarts) is not int or max_restarts < 0:
        raise ValueError("max_restarts must be a nonnegative integer")
    if type(maxiter) is not int or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    started_at = time.monotonic()
    initial = initial_parameters()
    incumbent = Incumbent(initial)
    stats = {"started_starts": 0, "completed_starts": 0, "successful_starts": 0,
             "callback_calls": 0, "deadline_reached": False,
             "stop_reason": "restart_limit", "budget_seconds": float(budget_seconds),
             "feasibility_tolerance": FEASIBILITY_TOL,
             "incumbent_feasibility": "tolerance_feasible"}
    LAST_STATS.clear()

    def check_deadline(params=None):
        if time.monotonic() - started_at >= budget_seconds:
            # The latest feasible state survives even if the clock expires before
            # SLSQP reaches its next iteration callback or returns a result.
            if params is not None:
                incumbent.consider(params)
            raise Deadline("cooperative packing deadline")

    def timed(function):
        def call(params):
            check_deadline(params)
            return function(params)
        return call

    def callback(params):
        stats["callback_calls"] += 1
        incumbent.consider(params)
        check_deadline()

    bounds = [(None, None)] * (2 * N_CIRCLES) + [(0.0, None)] * N_CIRCLES
    constraints = {"type": "ineq", "fun": timed(packing_constraints),
                   "jac": timed(packing_jacobian)}
    try:
        for _ in range(max_restarts):
            check_deadline()
            # Same global draw, shape and clamps as the archived winner.
            guess = initial + np.random.uniform(-0.06, 0.06, initial.shape)
            guess[2 * N_CIRCLES:] = np.maximum(guess[2 * N_CIRCLES:], 0.001)
            guess[:2 * N_CIRCLES] = np.clip(guess[:2 * N_CIRCLES], 0.001, 0.999)
            stats["started_starts"] += 1
            result = minimize(timed(objective), guess, method="SLSQP",
                              jac=timed(objective_jac), bounds=bounds,
                              constraints=constraints, callback=callback,
                              options={"maxiter": maxiter, "ftol": 1e-7, "disp": False})
            stats["completed_starts"] += 1
            stats["successful_starts"] += int(bool(result.success))
            incumbent.consider(result.x)
            check_deadline()
    except Deadline:
        stats["deadline_reached"] = True
        stats["stop_reason"] = "deadline"
    except Exception:
        # Unexpected defects remain failures, rather than being disguised as a
        # completed search. The finally block still publishes diagnostic counts.
        stats["stop_reason"] = "error"
        raise
    finally:
        stats.update(candidate_checks=incumbent.candidate_checks,
                     feasible_states=incumbent.feasible_states,
                     improvements=incumbent.improvements,
                     best_sum=incumbent.result()[2],
                     elapsed_seconds=max(0.0, time.monotonic() - started_at))
        LAST_STATS.update(stats)
    return incumbent.result()


def run_packing():
    """Canonical task entry point; the external 30-second smoke cap still applies."""
    return solve_packing()
