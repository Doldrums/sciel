"""Bohnenblust-Hille ratio of functions on the Boolean cube for C_26a; a program uses it as:
from sciel.task import asset; bh = asset("bh").

A candidate is {"n": n, "terms": [[mask, a], ...]}: f(x) = sum a x^S on {-1, 1}^n, 1 <= n <= MAX_N, with S the set
bits of mask (bit i = coordinate i), distinct masks and nonzero integer coefficients a, |a| <= MAX_COEF, at most
MAX_TERMS terms. With d the largest |S| (at least 1) and p = 2d / (d + 1),

    R(f) = (sum |a|^p)^(1 / p) / max_x |f(x)|,

and every f gives C_26a >= BH(d) >= R(f). The ratio is scale invariant, so integer coefficients lose nothing. Point
x is the bitmask m with x_i = -1 iff bit i of m is set, so x^S = (-1)^|m & mask| and f at all 2^n points is the Walsh
transform of the dense coefficient array. Numpy arrays below are dense coefficient arrays of length 2^n.
"""

import mpmath
import numpy as np

MAX_N = 22
MAX_TERMS = 1 << 16
MAX_COEF = 1 << 40
DIGITS = 50


def popcounts(n):
    """|S| for every index S < 2^n."""
    p = np.zeros(1, dtype=np.int64)
    for _ in range(n):
        p = np.concatenate([p, p + 1])
    return p


def wht(a):
    """Walsh-Hadamard transform in a's own arithmetic: f(m) = sum_S a(S) (-1)^|S & m|."""
    a = np.array(a, copy=True)
    h = 1
    while h < len(a):
        a = a.reshape(-1, 2, h)
        a = np.stack([a[:, 0] + a[:, 1], a[:, 0] - a[:, 1]], axis=1).reshape(-1)
        h *= 2
    return a


def degree(a):
    """The largest |S| with a(S) != 0, at least 1."""
    a = np.asarray(a)
    n = len(a).bit_length() - 1
    nz = np.flatnonzero(a)
    return max(1, int(popcounts(n)[nz].max())) if len(nz) else 1


def ratio(a, d=None):
    """R(f) in floats for a dense (float or integer) coefficient array, at degree d (default: the degree of f)."""
    a = np.asarray(a, dtype=np.float64)
    d = degree(a) if d is None else d
    p = 2 * d / (d + 1)
    sup = float(np.abs(wht(a)).max())
    return float((np.abs(a) ** p).sum() ** (1 / p)) / sup if sup > 0 else 0.0


def check(candidate):
    """(n, masks int64, coefficients int64); raises ValueError on malformed input."""
    if not isinstance(candidate, dict) or set(candidate) != {"n", "terms"}:
        raise ValueError('candidate must be {"n": n, "terms": [[mask, a], ...]}')
    n = candidate["n"]
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not 1 <= int(n) <= MAX_N:
        raise ValueError(f"n must be an integer in [1, {MAX_N}]")
    n = int(n)
    terms = candidate["terms"]
    if not isinstance(terms, (list, tuple)) or not 1 <= len(terms) <= MAX_TERMS:
        raise ValueError(f"terms must be a list of 1 to {MAX_TERMS} [mask, a] pairs")
    masks, coefs = [], []
    for t in terms:
        if not isinstance(t, (list, tuple)) or len(t) != 2:
            raise ValueError("each term must be [mask, a]")
        for v in t:
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
                raise ValueError("masks and coefficients must be integers")
        m, c = int(t[0]), int(t[1])
        if not 0 <= m < 1 << n:
            raise ValueError("mask out of range [0, 2^n)")
        if c == 0 or abs(c) > MAX_COEF:
            raise ValueError(f"coefficients must be nonzero with |a| <= {MAX_COEF}")
        masks.append(m)
        coefs.append(c)
    masks = np.array(masks, dtype=np.int64)
    if len(np.unique(masks)) != len(masks):
        raise ValueError("masks must be distinct")
    return n, masks, np.array(coefs, dtype=np.int64)


def array(candidate):
    """Dense int64 coefficient array of a candidate."""
    n, masks, coefs = check(candidate)
    a = np.zeros(1 << n, dtype=np.int64)
    a[masks] = coefs
    return a


def make(a, scale=1):
    """Candidate of a dense coefficient array, entries multiplied by scale and rounded to integers."""
    a = np.asarray(a)
    n = len(a).bit_length() - 1
    if len(a) != 1 << n or not 1 <= n <= MAX_N:
        raise ValueError("a must have length 2^n, 1 <= n <= MAX_N")
    r = np.rint(np.asarray(a, dtype=np.float64) * scale).astype(np.int64)
    nz = np.flatnonzero(r)
    return {"n": n, "terms": [[int(m), int(r[m])] for m in nz]}


def value(candidate):
    """The float ratio of a candidate dict (no certification)."""
    return ratio(array(candidate))


def load(output):
    """The candidate dict of a population output."""
    return {"n": int(output["n"]), "terms": [[int(m), int(c)] for m, c in output["terms"]]}


def certify(candidate):
    """(lower, d, sup): lower an mpf rounded down from interval powers at DIGITS digits over the distinct |a|, d the
    degree and sup = max |f| an exact integer."""
    a = array(candidate)
    d = degree(a)
    sup = int(np.abs(wht(a)).max())
    vals, counts = np.unique(np.abs(a[a != 0]), return_counts=True)
    iv = mpmath.iv
    saved, iv.dps = iv.dps, DIGITS
    try:
        p = iv.mpf(2 * d) / (d + 1)
        total = iv.mpf(0)
        for v, k in zip(vals.tolist(), counts.tolist()):
            total += int(k) * iv.exp(p * iv.log(iv.mpf(int(v))))
        r = iv.exp(iv.log(total) / p) / sup
        with mpmath.workdps(DIGITS):
            lower = mpmath.mpf(r.a)
    finally:
        iv.dps = saved
    return lower, d, sup
