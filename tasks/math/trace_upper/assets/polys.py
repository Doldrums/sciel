"""Totally positive polynomials for the certificate's set A: from sciel.task import asset; polys = asset("polys").

A monic Q = x^n + a_1 x^(n-1) + ... + a_n has all roots in [0, B] only if every derivative does, and Q^(n-k)/(n-k)! is
P_k = sum_{i<=k} a_i C(n-i, k-i) x^(k-i), whose critical points are the roots of P_(k-1). So a_k ranges over the
integers that make P_k alternate in sign at those points and keep its sign at 0 and B (Robinson's enumeration).
"""

from math import comb
from pathlib import Path

import numpy as np
import sympy

from sciel.candidate_modules import load_candidate

_trace = load_candidate(str(Path(__file__).resolve().parent / "trace.py"), "trace")
X = sympy.Symbol("x")


def _range(n, a, upper):
    """The integer interval of a_k (k = len(a)) for which P_k keeps all its roots real in [0, upper]."""
    k = len(a)
    R = np.array([a[i] * comb(n - i, k - i) for i in range(k)] + [0.0])
    crit = np.sort(np.roots(R[:-1] * np.arange(k, 0, -1)).real) if k > 1 else np.zeros(0)
    lo, hi = -np.inf, np.inf
    for y, sign in [(0.0, (-1) ** k)] + [(y, (-1) ** (k - 1 - i)) for i, y in enumerate(crit)] + [(upper, 1)]:
        v = -np.polyval(R, y)
        if sign > 0:
            lo = max(lo, v)
        else:
            hi = min(hi, v)
    return int(np.ceil(lo - 1e-9)), int(np.floor(hi + 1e-9))


def totally_positive(degree: int, upper: float = 6.0, max_ratio: float = 2.0) -> list:
    """Every irreducible monic integer polynomial of the given degree with all roots real in [0, upper] and trace at
    most max_ratio * degree, as coefficient lists with the constant term first (the certificate's format). Degree 6
    gives 242 polynomials in 3 s, degree 7 gives 1069 in about a minute."""
    n, out = int(degree), []

    def grow(a):
        if len(a) == n + 1:
            P = sympy.Poly(a, X, domain="ZZ")
            if P.is_irreducible and P.count_roots(0, upper) == n:
                out.append([int(c) for c in reversed(a)])
            return
        lo, hi = _range(n, a, upper)
        if len(a) == 1:
            lo = max(lo, -int(np.floor(max_ratio * n + 1e-9)))
        for c in range(lo, hi + 1):
            grow(a + [c])

    grow([1])
    return out


def candidates(cert, max_degree: int = 6, upper: float = 6.0, max_ratio: float = 2.0) -> list:
    """Totally positive polynomials of degree 1..max_degree not already in the certificate, with int log|Q| dmu / deg Q
    for its measure, lowest first. A valid certificate makes every value >= 0 (f >= 0 on C gives it through the
    resultants); the lowest are the nearly active constraints, the natural ones to add to A before re-solving lp().
    Degree 6 takes seconds, degree 7 about a minute."""
    have = {tuple(q) if q[-1] > 0 else tuple(-c for c in q) for q in cert[0]}
    pool = [q for n in range(1, max_degree + 1) for q in totally_positive(n, upper, max_ratio) if tuple(q) not in have]
    if not pool:
        return []
    values = _trace.energies(cert, pool) / np.array([len(q) - 1 for q in pool])
    return [(q, float(v)) for v, q in sorted(zip(values, pool), key=lambda t: t[0])]
