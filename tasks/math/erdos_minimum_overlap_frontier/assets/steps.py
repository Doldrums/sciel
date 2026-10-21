"""The task's objective for a candidate to build on: from sciel.task import asset; steps = asset("steps").

For heights h in [0, 1] of n equal steps on [0, 2] with sum h = n/2, the bound is F(h) = (2/n) max_s c_s with
c = correlate(h, 1 - h, 'full') = convolve(h, (1 - h)[::-1]). F is invariant to repeating every step k times, so a
construction moves to a finer grid without changing its bound.
"""

from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from sciel.candidate_modules import load_candidate

MAX_STEPS = 65536
# clean puts heights on multiples of 2**-GRID, where binary64 holds them and their complements exactly.
GRID = 52
_evaluator = load_candidate(str(Path(__file__).resolve().parents[1] / "evaluator.py"), "evaluator")


def _overlap(h):
    r = (1 - h)[::-1]
    return np.convolve(h, r) if h.size <= 256 else fftconvolve(h, r)


def _correlate(w, x):
    return np.correlate(w, x, "valid") if x.size <= 256 else fftconvolve(w, x[::-1], "valid")


def bound(h) -> float:
    """F(h) in floating point, through an FFT for large n; accurate to about 1e-12 relative. It does not check the
    constraints."""
    h = np.asarray(h, dtype=float)
    return float(2 / h.size * _overlap(h).max())


def exact_bound(h) -> float:
    """F(h) exactly as the evaluator scores it, after its validity checks; slower than bound."""
    return float(_evaluator.bound(_evaluator._heights(h)))


def _gradient(h, w):
    """The gradient in h of sum_s w_s u_s, u_s = (2/n) c_s, through both h and 1 - h."""
    return 2 / h.size * (_correlate(w, (1 - h)[::-1]) - _correlate(w, h)[::-1])


def smooth_bound(h, beta: float) -> tuple:
    """A soft maximum of F over the shifts, (1/beta) log sum_s exp(beta u_s) with u_s = (2/n) c_s, and its gradient in
    h, which ignores the constraints (clean restores them). It is at most log(2n-1)/beta above F; raise beta as the
    search converges."""
    h = np.asarray(h, dtype=float)
    u = 2 / h.size * _overlap(h)
    w = np.exp(beta * (u - u.max()))
    total = w.sum()
    w /= total
    return float(u.max() + np.log(total) / beta), _gradient(h, w)


def subgradient(h, rtol: float = 1e-9) -> tuple:
    """F(h) and the average gradient of the shifts within rtol of the maximum, which are the shifts a step must
    lower."""
    h = np.asarray(h, dtype=float)
    u = 2 / h.size * _overlap(h)
    w = (u >= u.max() * (1 - rtol)).astype(float)
    w /= w.sum()
    return float(u.max()), _gradient(h, w)


def active(h, rtol: float = 1e-9) -> np.ndarray:
    """The shifts s, in 0..2n-2, where c_s is within rtol of its maximum."""
    c = _overlap(np.asarray(h, dtype=float))
    return np.flatnonzero(c >= c.max() * (1 - rtol))


def refine(h, k: int) -> np.ndarray:
    """Every step split into k equal steps: the same function, so the same bound, with k times the freedom."""
    return np.repeat(np.asarray(h, dtype=float), int(k))


def resample(h, n: int) -> np.ndarray:
    """The step function on n steps, each the mean of the old function over it; the bound changes slightly and the
    sum needs clean to be exact again."""
    h = np.asarray(h, dtype=float)
    edges = np.linspace(0, h.size, n + 1)
    cumulative = np.concatenate([[0.0], np.cumsum(h)])
    area = np.interp(edges, np.arange(h.size + 1), cumulative)
    return np.diff(area) * n / h.size


def valid(h) -> bool:
    """Whether the evaluator accepts h: in [0, 1], none within 2**-60 of 0 or 1 without equalling it, at most MAX_STEPS
    of them, and sum h exactly n/2 as rationals."""
    try:
        h = _evaluator._heights(h)
    except ValueError:
        return False
    x = _evaluator._integers(h.tolist() + [1.0])
    return 2 * sum(x[:-1]) == h.size * x[-1]


def clean(h) -> np.ndarray:
    """Heights the evaluator accepts, unchanged when they already are: otherwise clipped to [0, 1], shifted so they
    sum to n/2, rounded to multiples of 2**-52, and the remaining units of 2**-52 added to or taken from the heights
    strictly inside (0, 1) so the sum is exactly n/2."""
    h = np.clip(np.asarray(h, dtype=float), 0.0, 1.0)
    if h.size > MAX_STEPS:
        h = np.clip(resample(h, MAX_STEPS), 0.0, 1.0)
    if valid(h):
        return h
    n = h.size
    lo, hi = -1.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if np.clip(h + mid, 0.0, 1.0).sum() < n / 2:
            lo = mid
        else:
            hi = mid
    q = np.rint(np.ldexp(np.clip(h + 0.5 * (lo + hi), 0.0, 1.0), GRID)).astype(np.int64)
    top = 1 << GRID
    residual = n * (top // 2) - sum(q.tolist())
    while residual:
        sign = 1 if residual > 0 else -1
        room = np.flatnonzero((q > 0) & (q < top))
        if room.size == 0:
            room = np.flatnonzero(q < top) if sign > 0 else np.flatnonzero(q > 0)
        room = room[:abs(residual)]
        q[room] += sign
        residual -= sign * room.size
    return np.ldexp(q.astype(float), -GRID)
