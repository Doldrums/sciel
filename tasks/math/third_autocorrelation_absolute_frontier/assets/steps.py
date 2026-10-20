"""The task's objective for a candidate to build on: from sciel.task import asset; steps = asset("steps").

For heights h of n equal steps on [-1/4, 1/4], the bound is F(h) = 2n max_k |c_k| / (sum h)^2 with c = convolve(h, h),
the autoconvolution and not the autocorrelation. F is invariant to scaling h and to repeating every step k times, so a
construction moves to a finer grid without changing its bound.
"""

from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from sciel.candidate_modules import load_candidate

MAX_STEPS = 65536
MIN_RELATIVE = 2.0 ** -60
_evaluator = load_candidate(str(Path(__file__).resolve().parents[1] / "evaluator.py"), "evaluator")


def _conv(h):
    return np.convolve(h, h) if h.size <= 256 else fftconvolve(h, h)


def bound(h) -> float:
    """F(h) in floating point, through an FFT for large n; accurate to about 1e-12 relative."""
    h = np.asarray(h, dtype=float)
    return float(2 * h.size * np.abs(_conv(h)).max() / h.sum() ** 2)


def exact_bound(h) -> float:
    """F(h) exactly as the evaluator scores it, after its validity checks; slower than bound."""
    return float(_evaluator.bound(_evaluator._heights(h)))


def smooth_bound(h, beta: float) -> tuple:
    """A soft maximum of F over the lags, (1/beta) log sum_k exp(beta u_k) with u_k = 2n |c_k| / (sum h)^2, and its
    gradient in h. It is at most log(2n-1)/beta above F; raise beta as the search converges."""
    h = np.asarray(h, dtype=float)
    n, s = h.size, h.sum()
    c = _conv(h)
    u = 2 * n * np.abs(c) / s ** 2
    z = beta * (u - u.max())
    w = np.exp(z)
    total = w.sum()
    w /= total
    value = u.max() + np.log(total) / beta
    a = w * np.sign(c)
    corr = np.correlate(a, h, "valid") if n <= 256 else fftconvolve(a, h[::-1], "valid")
    return float(value), 4 * n / s ** 2 * corr - 2 / s * float(w @ u)


def subgradient(h, rtol: float = 1e-9) -> tuple:
    """F(h) and the average gradient of the lags within rtol of the maximum, which are the lags a step must lower."""
    h = np.asarray(h, dtype=float)
    n, s = h.size, h.sum()
    c = _conv(h)
    m = np.abs(c).max()
    a = np.where(np.abs(c) >= m * (1 - rtol), np.sign(c), 0.0)
    a /= np.abs(a).sum()
    corr = np.correlate(a, h, "valid") if n <= 256 else fftconvolve(a, h[::-1], "valid")
    value = 2 * n * m / s ** 2
    return float(value), 4 * n / s ** 2 * corr - 2 / s * value


def active(h, rtol: float = 1e-9) -> np.ndarray:
    """The lags k, in 0..2n-2, where |c_k| is within rtol of its maximum."""
    c = np.abs(_conv(np.asarray(h, dtype=float)))
    return np.flatnonzero(c >= c.max() * (1 - rtol))


def refine(h, k: int) -> np.ndarray:
    """Every step split into k equal steps: the same function, so the same bound, with k times the freedom."""
    return np.repeat(np.asarray(h, dtype=float), int(k))


def resample(h, n: int) -> np.ndarray:
    """The step function on n steps, each the mean of the old function over it; the bound changes slightly."""
    h = np.asarray(h, dtype=float)
    edges = np.linspace(0, h.size, n + 1)
    cumulative = np.concatenate([[0.0], np.cumsum(h)])
    area = np.interp(edges, np.arange(h.size + 1), cumulative)
    return np.diff(area) * n / h.size


def clean(h) -> np.ndarray:
    """Heights the evaluator accepts: scaled to max |h| = 1, the sign making sum h positive, heights below 2**-60 of
    the largest set to exactly 0, and at most MAX_STEPS of them."""
    h = np.asarray(h, dtype=float)
    if h.size > MAX_STEPS:
        h = resample(h, MAX_STEPS)
    h = h / np.abs(h).max()
    if h.sum() < 0:
        h = -h
    h[np.abs(h) < MIN_RELATIVE] = 0.0
    return h
