"""The task's objective for a candidate to build on: from sciel.task import asset; bergman = asset("bergman").

Polynomials are coefficient lists, constant term first. In A^2 with normalized area measure ||z^n||^2 = 1/(n + 1), so
||p||^2 = sum p_n^2 / (n + 1). A pair (f, g) proves Korenblum's constant kappa <= c when ||f|| > ||g||, |f| <= |g| on
|z| = c and |z| = 1, and g has no zeros on c <= |z| <= 1; the maximum principle for f/g fills in the annulus.
"""

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

from fractions import Fraction
import math
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from sciel.candidate_modules import load_candidate

_evaluator = load_candidate(str(Path(__file__).resolve().parents[1] / "evaluator.py"), "evaluator")
MAX_DEGREE = _evaluator.MAX_DEGREE
REFERENCE = _evaluator.REFERENCE
U = 2.0 ** -53


def exact(values) -> list:
    """Coefficients as exact Fractions (ints, binary64, Fractions or strings like '2/3')."""
    return [_evaluator.exact(v) for v in values]


def norm2(p) -> Fraction:
    """Exact ||p||^2."""
    return _evaluator.norm2(exact(p))


def gap(f, g) -> Fraction:
    """Exact ||f||^2 - ||g||^2; the bound needs it positive."""
    return norm2(f) - norm2(g)


def norm2_float(p) -> float:
    p = np.asarray(p, dtype=float)
    return float(np.sum(p * p / np.arange(1, p.size + 1)))


def load(output) -> tuple:
    """(c, f, g) of a scored pair from the output its evaluation recorded (a population entry's "output"): c as
    binary64, f and g as exact Fractions."""
    return float.fromhex(output["c"]), [Fraction(x) for x in output["f"]], [Fraction(x) for x in output["g"]]


def values(p, r, theta) -> np.ndarray:
    """p(r e^{i theta}) at the angles theta, in floating point."""
    p = np.asarray([float(x) for x in p]) * r ** np.arange(len(p))
    return np.polyval(p[::-1], np.exp(1j * np.asarray(theta, dtype=float)))


def modulus_bounds(p, r, lo, hi) -> tuple:
    """Rigorous bounds (low, high) of |p(r e^{i theta})| for theta in [lo, hi]: the value at the midpoint, its
    rounding error and the Lipschitz constant sum n |p_n| r^n in theta."""
    coef = np.asarray([float(x) for x in p]) * np.cumprod(np.r_[1.0, np.full(len(p) - 1, float(r))])
    k = np.arange(len(p), dtype=float)
    mid, half = (lo + hi) / 2, (hi - lo) / 2 * (1 + 1e-9)
    value = abs(complex(np.exp(1j * mid * k) @ coef))
    error = 2 * U * float(np.sum(np.abs(coef) * (math.pi * k + 2 * len(p) + 16))) * 1.01
    reach = float(np.sum(k * np.abs(coef))) * (1 + 4 * (len(p) + 4) * U) * half
    return max(0.0, value - error - reach), value + error + reach


def _grid(p, r, size):
    coef = np.zeros(size, dtype=complex)
    coef[:len(p)] = np.asarray([float(x) for x in p]) * r ** np.arange(len(p))
    return np.fft.ifft(coef) * size  # p(r e^{2 pi i j / size}) for j = 0..size-1


def slack(f, g, r, size=8192, refine=True) -> float:
    """min over theta of |g|^2 - |f|^2 on |z| = r, from an FFT grid refined around its lowest point; negative when
    |f| > |g| somewhere on the circle."""
    vf, vg = np.abs(_grid(f, r, size)) ** 2, np.abs(_grid(g, r, size)) ** 2
    j = int(np.argmin(vg - vf))
    if not refine:
        return float((vg - vf)[j])
    step = 2 * math.pi / size
    theta = 2 * math.pi * j / size

    def d(t):
        return float(np.abs(values(g, r, [t])[0]) ** 2 - np.abs(values(f, r, [t])[0]) ** 2)
    best = minimize_scalar(d, bounds=(theta - step, theta + step), method="bounded", options={"xatol": 1e-13})
    return min(float((vg - vf)[j]), d(theta), float(best.fun))


def ratio_max(f, g, r, size=8192) -> float:
    """max over theta of |f| / |g| on |z| = r, on an FFT grid."""
    vf, vg = np.abs(_grid(f, r, size)), np.abs(_grid(g, r, size))
    return float(np.max(vf / np.maximum(vg, 1e-300)))


def smallest_c(f, g, margin=1e-10, low=0.05, scan=400):
    """The smallest c with |f| <= |g| on the whole annulus c <= |z| < 1, in floating point, plus margin and rounded
    up to binary64; None when |f| > |g| somewhere on |z| = 1. Scans r down from 1, bisects the first crossing, and
    the margin leaves room for the certified check (about 1e-10 suffices at degree 400)."""
    if slack(f, g, 1.0) < -1e-12 * norm2_float(g):
        return None
    rs = 1.0 - (1.0 - low) * (np.arange(1, scan + 1) / scan) ** 2
    above = 1.0
    for r in rs:
        if slack(f, g, r, 4096, refine=False) < 0:
            lo, hi = r, above
            for _ in range(80):
                mid = (lo + hi) / 2
                if slack(f, g, mid) < 0:
                    lo = mid
                else:
                    hi = mid
                if hi - lo < 1e-15:
                    break
            c = min(hi + margin, math.nextafter(1.0, 0.0))
            return c
        above = r
    return low


def check(c, f, g) -> dict:
    """The evaluator's certificate for (c, f, g): norm gap, outer and inner margins, zeros of g inside |z| < c;
    raises ValueError with the failing condition."""
    c, f, g = _evaluator.construction((c, f, g))
    return _evaluator.certify(c, f, g)


def score(c) -> float:
    """combined_score of the bound c."""
    return REFERENCE / c
