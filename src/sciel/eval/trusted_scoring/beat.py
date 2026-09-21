"""Pure Beat-the-Average metric copied from the pinned legacy evaluator.

Inputs are inert arrays produced by the bounded numeric decoder. This module
never loads a candidate, reads a path, consumes a candidate score, or authenticates
an execution. Source helper bodies intentionally retain binary64 reduction order.
"""
import heapq
import math

import numpy as np


BENCHMARK = 0.4006



MAX_SUPPORT = 2000



TOL = 1e-9



def _success_probability(support, weights):
    # Binary64 support points are exact dyadic rationals. Compare their integer
    # multiples, not rounded/possibly infinite pair sums and thresholds. A
    # common positive factor cancels from X1 + X2 + X3 < 2*X4.
    ratios = [float(x).as_integer_ratio() for x in support]
    denominator = max(d for _, d in ratios)
    points = [n * (denominator // d) for n, d in ratios]
    divisor = math.gcd(*points) or 1
    points = [point // divisor for point in points]
    w = _normalized_weights(weights)

    # Keep the vectorized path when both pair sums and 2*x-y fit exactly.
    # Widely separated exponents use O(n) big-integer storage below, never an
    # n^2 object array (which could be huge even under MAX_SUPPORT).
    if max(points) > np.iinfo(np.int64).max // 2:
        return _wide_success_probability(points, w)
    s = np.asarray(points, dtype=np.int64)

    P2 = (s[:, None] + s[None, :]).ravel()
    W2 = (w[:, None] * w[None, :]).ravel()
    order = np.argsort(P2, kind="stable")
    P2s = P2[order]
    W2s = W2[order]
    prefix = np.concatenate([[0.0], np.cumsum(W2s)])

    total = 0.0
    for k in range(len(s)):
        thresholds = 2 * s - s[k]
        idx = np.searchsorted(P2s, thresholds, side="left")
        mass = prefix[idx]
        total += w[k] * float(np.sum(w * mass))
    return total



def _wide_success_probability(points, weights):
    """Merge exact ordered sums/thresholds with bounded O(n) heap storage."""
    ordered = sorted(zip(points, map(float, weights)))
    s, w = zip(*ordered)
    count = len(s)
    # i<=j halves the pair stream; off-diagonal mass counts both orders.
    pairs = [(s[i] + s[i], i, i) for i in range(count)]
    thresholds = [(2 * s[0] - s[k], k, 0) for k in range(count)]
    heapq.heapify(pairs)
    heapq.heapify(thresholds)
    mass = total = mass_error = total_error = 0.0
    while thresholds:
        threshold, k, m = heapq.heappop(thresholds)
        while pairs and pairs[0][0] < threshold:
            _, i, j = heapq.heappop(pairs)
            increment = w[i] * w[j] * (1 if i == j else 2) - mass_error
            updated = mass + increment
            mass_error = (updated - mass) - increment
            mass = updated
            if j + 1 < count:
                heapq.heappush(pairs, (s[i] + s[j + 1], i, j + 1))
        increment = w[k] * w[m] * mass - total_error
        updated = total + increment
        total_error = (updated - total) - increment
        total = updated
        if m + 1 < count:
            heapq.heappush(thresholds, (2 * s[m + 1] - s[k], k, m + 1))
    return total



def _normalized_weights(weights):
    w = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(w).all() or np.any(w < 0):
        raise ValueError("Weights must be finite and non-negative.")
    largest = float(w.max()) if w.size else 0.0
    if largest <= 0:
        raise ValueError("Total weight must be positive.")
    # Normalize a nonnegative measure without overflow or cancellation. An
    # absolute negative tolerance is unsafe when the total mass is very small.
    scaled = w / largest
    return scaled / scaled.sum()



_BEHAVIOR_ZERO = {"b_dispersion": 0.0, "b_boundary_frac": 0.0, "b_contact": 0.0,
                  "b_offset": 0.0, "b_concentration": 0.0, "b_structure": 0.0}



def _squash(x):
    """Unbounded ratio -> [0,1). Every behavior slot shares one scale on purpose: fed
    to a linear trunk, a slot that ranges to 4 while the rest range to 1 dominates the
    others for no reason but its units."""
    x = float(x)
    return x / (1.0 + x) if x > 0 and np.isfinite(x) else 0.0



def _finite(d):
    return {k: (float(v) if np.isfinite(v) else 0.0) for k, v in d.items()}



def _behavior(support, weights):
    """The measure's own shape: how the mass is spread over the support."""
    try:
        sup = np.asarray(support, dtype=np.float64).ravel()
        w = np.asarray(weights, dtype=np.float64).ravel()
        tot = float(w.sum())
        if sup.size == 0 or tot <= 0 or not np.isfinite(sup).all() \
                or not np.isfinite(w).all():
            return dict(_BEHAVIOR_ZERO)
        p = w / tot
        mean_p = float(p.mean())
        span = float(sup.max() - sup.min())
        live = p > 1e-9
        ent = float(-(p[live] * np.log(p[live])).sum())
        max_ent = float(np.log(sup.size)) if sup.size > 1 else 0.0
        k = max(1, sup.size // 4)
        return _finite({
            "b_dispersion": _squash(float(p.std() / mean_p) if mean_p > 0 else 0.0),
            "b_boundary_frac": float((sup <= TOL).mean()),
            "b_contact": float(ent / max_ent) if max_ent > 0 else 0.0,
            "b_offset": float(np.clip(float((p * sup).sum() - sup.min()) / span, 0.0, 1.0))
                        if span > 0 else 0.0,
            "b_concentration": float(np.sort(p)[-k:].sum()),
            "b_structure": float(live.mean()),
        })
    except (ValueError, FloatingPointError, ZeroDivisionError):
        return dict(_BEHAVIOR_ZERO)



def score(support, weights):
    support = np.asarray(support, dtype=np.float64).ravel()
    weights = np.asarray(weights, dtype=np.float64).ravel()

    if support.shape != weights.shape:
        raise ValueError("support and weights must have the same shape.")
    if support.size < 1:
        raise ValueError("Measure must have at least one support point.")
    # _success_probability builds n^2 pair arrays and loops over n, so cost grows
    # quadratically in memory with no ceiling. Every other evaluator in the suite
    # bounds its object by construction (fixed shape, or a value range); this one
    # accepted any length, and the task declares no cpu/mem rlimit, so only the wall
    # clock stood between a large support and an allocation failure.
    if support.size > MAX_SUPPORT:
        raise ValueError(
            f"support has {support.size} points; the limit is {MAX_SUPPORT} "
            "(the scorer is quadratic in the number of support points)")
    if not np.all(np.isfinite(support)) or not np.all(np.isfinite(weights)):
        raise ValueError("support/weights contain NaN or infinite values.")
    if np.any(support < 0):
        raise ValueError("Support points must lie in [0, inf).")
    weights = _normalized_weights(weights)

    success_prob = _success_probability(support, weights)
    if not np.isfinite(success_prob) or not -1e-12 <= success_prob <= 1.0 + 1e-12:
        raise ValueError("Success probability must lie in [0, 1].")
    success_prob = float(np.clip(success_prob, 0.0, 1.0))
    combined = success_prob / BENCHMARK

    return {
        "ok": True,
        "score": float(combined),
        "cases": {
            "success_prob": float(success_prob),
            "combined_score": float(combined),
            **_behavior(support, weights),
        },
    }
