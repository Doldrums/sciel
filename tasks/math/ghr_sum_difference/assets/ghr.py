"""The two-temperature bound for the Gyarmati-Hennecart-Ruzsa constant C_3a; a program uses it as:
from sciel.task import asset; ghr = asset("ghr").

A candidate is {"mask": [m, ...], "x": x, "y": y}: a finite set M of non-negative integers containing 0 and a
positive integer, and two tilts x, y in (0, 1), each a binary64 number or an [numerator, denominator] pair (both read
exactly). With S = M + M, D = M - M and kappa(d) = min{a + b : a, b in M, a - b = d},

    P_-(x) = sum_{d in D} x^kappa(d),   P_+(y) = sum_{s in S} y^s,   mu = x P_-'(x) / P_-(x),
    rho = log P_-(x) - mu log x,        sigma = log P_+(y) - mu log y,

and if sigma < rho then C_3a >= 2 - sigma / rho (Olafson, two-temperature cyclic quotient, teorth/optimizationproblems
PR 148). rho is the entropy of the Gibbs law x^kappa(d) / P_-(x) on D: the difference words of one type class. sigma
bounds the growth of all sums at the same coordinate budget mu; any y is valid, the best one solves
sum_s s y^s / P_+(y) = mu.
"""

from fractions import Fraction

import mpmath
import numpy as np
from scipy.optimize import minimize_scalar

MAX_ELEMENT = 2000
DIGITS = 50


def _tilt(t):
    if isinstance(t, Fraction):
        q = t
    elif isinstance(t, (list, tuple)):
        if len(t) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in t):
            raise ValueError("a tilt pair must be [numerator, denominator] integers")
        q = Fraction(int(t[0]), int(t[1]))
    else:
        if isinstance(t, bool) or not isinstance(t, (int, float, np.integer, np.floating)):
            raise ValueError("a tilt must be a number or an [numerator, denominator] pair")
        if not np.isfinite(float(t)):
            raise ValueError("a tilt must be finite")
        q = Fraction(float(t))
    if not 0 < q < 1:
        raise ValueError("tilts must lie strictly between 0 and 1")
    return q


def canonical(candidate):
    """The candidate with the mask sorted and checked and the tilts as exact Fractions; raises ValueError."""
    if not isinstance(candidate, dict) or set(candidate) - {"mask", "x", "y"} or len(candidate) != 3:
        raise ValueError('the candidate must be {"mask": [...], "x": x, "y": y}')
    raw = list(candidate["mask"])
    if any(isinstance(m, bool) or not isinstance(m, (int, np.integer)) for m in raw):
        raise ValueError("mask elements must be integers")
    mask = sorted({int(m) for m in raw})
    if len(mask) != len(raw):
        raise ValueError("mask elements must be distinct")
    if not mask or mask[0] != 0 or len(mask) < 2:
        raise ValueError("the mask must contain 0 and a positive integer")
    if mask[-1] > MAX_ELEMENT:
        raise ValueError(f"mask elements must be at most {MAX_ELEMENT}")
    return {"mask": mask, "x": _tilt(candidate["x"]), "y": _tilt(candidate["y"])}


def costs(mask):
    """(c, sums): c[k] is the number of differences d in M - M with kappa(d) = k; sums is the sorted array M + M."""
    m = np.asarray(sorted(set(int(v) for v in mask)), dtype=np.int64)
    d = (m[:, None] - m[None, :]).ravel()
    s = (m[:, None] + m[None, :]).ravel()
    order = np.lexsort((s, d))
    d, s = d[order], s[order]
    first = np.r_[True, d[1:] != d[:-1]]
    return np.bincount(s[first]), np.unique(s)


def kappa(mask):
    """{d: kappa(d)} over M - M."""
    m = np.asarray(sorted(set(int(v) for v in mask)), dtype=np.int64)
    out = {}
    for a in m:
        for b in m:
            d, c = int(a - b), int(a + b)
            if c < out.get(d, c + 1):
                out[d] = c
    return out


def rates(mask, x, y=None):
    """Float (bound, rho, sigma, mu, y) at tilt x; y = None takes the best sum tilt for that x."""
    c, sums = costs(mask)
    k = np.arange(c.size, dtype=np.float64)
    w = c * np.exp(k * np.log(x))
    p = w.sum()
    mu = (k * w).sum() / p
    rho = np.log(p) - mu * np.log(x)
    s = sums.astype(np.float64)

    def sig(t):
        return np.log(np.exp(s * np.log(t)).sum()) - mu * np.log(t)

    if y is None:
        y = minimize_scalar(sig, bounds=(1e-6, 1 - 1e-12), method="bounded", options={"xatol": 1e-15}).x
    sigma = sig(y)
    return 2 - sigma / rho, rho, sigma, mu, y


def best(mask, x0=None):
    """Float (bound, x, y): both tilts optimized for the mask (about 10 ms for 100 elements)."""
    lo, hi = (1e-4, 1 - 1e-9) if x0 is None else (max(1e-4, x0 - 0.05), min(1 - 1e-9, x0 + 0.05))
    r = minimize_scalar(lambda t: -rates(mask, t)[0], bounds=(lo, hi), method="bounded", options={"xatol": 1e-14})
    v, _, _, _, y = rates(mask, r.x)
    return float(v), float(r.x), float(y)


def value(candidate):
    """Float bound of a candidate (no certification)."""
    c = canonical(candidate)
    return float(rates(c["mask"], float(c["x"]), float(c["y"]))[0])


def make(mask, x, y):
    return {"mask": [int(m) for m in sorted(set(mask))], "x": float(x), "y": float(y)}


def load(output):
    """A candidate from the output object the evaluator scored (population view)."""
    return {"mask": [int(m) for m in output["mask"]], "x": output["x"], "y": output["y"]}


def certify(candidate):
    """(lower, rho, sigma): lower is an mpf at 50 digits below 2 - sigma / rho, rho and sigma are (lo, hi) mpf pairs. Every power
    sum has non-negative coefficients and is evaluated by interval Horner from the exact rational tilts; raises
    ValueError unless sigma < rho is proved."""
    c = canonical(candidate)
    counts, sums = costs(c["mask"])
    iv = mpmath.iv
    saved, iv.dps = iv.dps, DIGITS
    try:
        x = iv.mpf(c["x"].numerator) / c["x"].denominator
        y = iv.mpf(c["y"].numerator) / c["y"].denominator
        p, dp = iv.mpf(0), iv.mpf(0)
        for k in range(counts.size - 1, -1, -1):
            p = p * x + int(counts[k])
            dp = dp * x + k * int(counts[k])
        mu = dp / p
        ind = np.zeros(int(sums[-1]) + 1, dtype=np.int64)
        ind[sums] = 1
        q = iv.mpf(0)
        for k in range(ind.size - 1, -1, -1):
            q = q * y + int(ind[k])
        rho = iv.log(p) - mu * iv.log(x)
        sigma = iv.log(q) - mu * iv.log(y)
        if not (sigma.b < rho.a and rho.a > 0):
            raise ValueError("sigma < rho is not proved: no bound")
        # Endpoints are read out at the same precision, so they are not rounded again.
        with mpmath.workdps(DIGITS):
            lower = mpmath.mpf((2 - sigma / rho).a)
            rho, sigma = (mpmath.mpf(rho.a), mpmath.mpf(rho.b)), (mpmath.mpf(sigma.a), mpmath.mpf(sigma.b))
    finally:
        iv.dps = saved
    return lower, rho, sigma


def down(x) -> float:
    """An mpf rounded down to binary64."""
    f = float(x)
    return float(np.nextafter(f, -np.inf)) if mpmath.mpf(f) > x else f


def exact_value(candidate) -> float:
    """The certified lower bound as scored, rounded down to binary64; about 2 ms for 75 elements, 0.6 s for 2001."""
    return down(certify(candidate)[0])
