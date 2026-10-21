"""The task's objective for a candidate to build on: from sciel.task import asset; corr = asset("corr").

For non-negative weights w of the step function f = sum_i w_i 1[i/K, (i+1)/K), the bound is
F(K, w) = K min_{0<=k<=K} c_k / (sum w)^2 with c_k = sum_i w_i w_{i+k}, the autocorrelation at the shifts k/K in [0, 1]
(c_k = 0 once k reaches len(w)). F is invariant to scaling w and to splitting every cell in m (K becomes mK), so a
construction moves to a finer grid without changing its bound.
"""

from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

from sciel.candidate_modules import load_candidate

MAX_K = 65536
MAX_WEIGHTS = 81920
MIN_RELATIVE = 2.0 ** -60
_evaluator = load_candidate(str(Path(__file__).resolve().parents[1] / "evaluator.py"), "evaluator")


def load(output) -> tuple:
    """(K, weights) of a scored construction, from the output its evaluation recorded (a population entry's
    "output"), where the weights are kept exactly as base64 binary64."""
    return int(output["K"]), _evaluator.decode(output["weights_float64_b64"])


def correlations(K, w) -> np.ndarray:
    """c_k for k = 0..K in floating point, through an FFT for long w; accurate to about 1e-12 of c_0."""
    w = np.asarray(w, dtype=float)
    n = w.size
    full = np.correlate(w, w, "full") if n <= 512 else fftconvolve(w, w[::-1])
    c = np.zeros(K + 1)
    m = min(n, K + 1)
    c[:m] = full[n - 1:n - 1 + m]
    return c


def value(K, w) -> float:
    """F(K, w) in floating point."""
    w = np.asarray(w, dtype=float)
    return float(K * correlations(K, w).min() / w.sum() ** 2)


def exact_value(K, w) -> float:
    """F(K, w) exactly as the evaluator scores it, after its validity checks; slower than value."""
    return float(_evaluator.bound(*_evaluator._construction((K, w))))


def _lag_gradient(K, w, p):
    """sum_k p_k dc_k/dw, with dc_k/dw_j = w_{j+k} + w_{j-k}, for weights p on the lags 0..K."""
    n = w.size
    p = p[:n]
    if n <= 512:
        ahead = np.correlate(w, p, "full")[p.size - 1:p.size - 1 + n]
        behind = np.convolve(w, p)[:n]
    else:
        ahead = fftconvolve(w, p[::-1])[p.size - 1:p.size - 1 + n]
        behind = fftconvolve(w, p)[:n]
    return ahead + behind


def smooth_value(K, w, beta: float) -> tuple:
    """A soft minimum of F over the lags, -(1/beta) log sum_k exp(-beta u_k) with u_k = K c_k / (sum w)^2, and its
    gradient in w. It is at most log(K + 1)/beta below F; raise beta as the search converges."""
    w = np.asarray(w, dtype=float)
    s = w.sum()
    u = K * correlations(K, w) / s ** 2
    z = -beta * (u - u.min())
    p = np.exp(z)
    total = p.sum()
    p /= total
    soft = u.min() - np.log(total) / beta
    return float(soft), K / s ** 2 * _lag_gradient(K, w, p) - 2 / s * float(p @ u)


def supergradient(K, w, rtol: float = 1e-9) -> tuple:
    """F(K, w) and the average gradient of the lags within rtol of the minimum, the lags a step must raise."""
    w = np.asarray(w, dtype=float)
    s = w.sum()
    u = K * correlations(K, w) / s ** 2
    p = (u <= u.min() * (1 + rtol)).astype(float)
    p /= p.sum()
    return float(u.min()), K / s ** 2 * _lag_gradient(K, w, p) - 2 / s * float(u.min())


def active(K, w, rtol: float = 1e-9) -> np.ndarray:
    """The lags k, in 0..K, where c_k is within rtol of its minimum."""
    c = correlations(K, w)
    return np.flatnonzero(c <= c.min() * (1 + rtol))


def refine(K, w, m: int) -> tuple:
    """(m K, every cell split into m): the same function, so the same bound, with m times the freedom."""
    return int(K) * int(m), np.repeat(np.asarray(w, dtype=float), int(m))


def clean(K, w) -> tuple:
    """(K, weights the evaluator accepts): negative weights set to 0, scaled to max 1, weights below 2**-60 of the
    largest set to exactly 0, and at most min(2K, MAX_WEIGHTS) of them."""
    w = np.maximum(np.asarray(w, dtype=float)[:min(2 * int(K), MAX_WEIGHTS)], 0.0)
    w = w / w.max()
    w[w < MIN_RELATIVE] = 0.0
    return int(K), w
