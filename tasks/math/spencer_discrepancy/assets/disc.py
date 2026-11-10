"""Exact discrepancy of square matrices for Spencer's constant C_10c; a program uses it as:
from sciel.task import asset; disc = asset("disc").

A candidate is {"rows": [[a_11, ..., a_1n], ...], "den": D}: an n x n integer matrix with entries in [-D, D],
1 <= D <= MAX_DEN, 1 <= n <= MAX_N, standing for A = rows / D in [-1, 1]^{n x n}. Its discrepancy is

    disc(A) = min over x in {-1, 1}^n of max_i |(Ax)_i|,

and disc(A) / sqrt(n) is a lower bound on C_10c. x and -x give the same value, so x_1 = 1 is fixed and the other
2^(n-1) sign vectors are enumerated as all sums of a low half and a high half of the columns.
"""

import math

import numpy as np

MAX_N = 28
MAX_DEN = 2 ** 20
CHUNK = 1 << 22


def _signs(k):
    """(2^k, k) array of all sign vectors on k coordinates."""
    idx = np.arange(1 << k)[:, None]
    return 1 - 2 * ((idx >> np.arange(k)) & 1)


def disc_numerator(rows):
    """min over x of max_i |(Mx)_i| for an integer (or float) matrix M, exactly in its own arithmetic."""
    m = np.asarray(rows)
    n = m.shape[1]
    base = m[:, 0]
    free = list(range(1, n))
    lo, hi = free[: len(free) // 2 + len(free) % 2], free[len(free) // 2 + len(free) % 2:]
    low = base[:, None] + m[:, lo] @ _signs(len(lo)).T            # (rows, 2^|lo|)
    high = m[:, hi] @ _signs(len(hi)).T if hi else np.zeros((m.shape[0], 1), dtype=m.dtype)
    step = max(1, CHUNK // max(1, low.size))
    best = None
    for s in range(0, high.shape[1], step):
        h = high[:, s:s + step]
        worst = np.abs(low[:, None, :] + h[:, :, None]).max(axis=0)  # (chunk, 2^|lo|)
        v = worst.min()
        best = v if best is None else min(best, v)
    return best


def check(candidate):
    """(rows int64 array, den); raises ValueError on malformed input."""
    if not isinstance(candidate, dict) or set(candidate) != {"rows", "den"}:
        raise ValueError('candidate must be {"rows": [[...], ...], "den": D}')
    den = candidate["den"]
    if isinstance(den, bool) or not isinstance(den, (int, np.integer)) or not 1 <= int(den) <= MAX_DEN:
        raise ValueError(f"den must be an integer in [1, {MAX_DEN}]")
    rows = candidate["rows"]
    if not isinstance(rows, (list, tuple)) or not 1 <= len(rows) <= MAX_N:
        raise ValueError(f"rows must be a list of 1 to {MAX_N} rows")
    n = len(rows)
    for r in rows:
        if not isinstance(r, (list, tuple)) or len(r) != n:
            raise ValueError("the matrix must be square")
        for v in r:
            if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or abs(int(v)) > int(den):
                raise ValueError("entries must be integers in [-den, den]")
    return np.array([[int(v) for v in r] for r in rows], dtype=np.int64), int(den)


def exact(candidate):
    """(disc numerator, den, n): disc(A) = numerator / den."""
    rows, den = check(candidate)
    return int(disc_numerator(rows)), den, rows.shape[0]


def value(candidate):
    """disc(A) / sqrt(n) as a float."""
    num, den, n = exact(candidate)
    return num / den / math.sqrt(n)


def float_value(a):
    """disc(A) / sqrt(n) for a float matrix in [-1, 1] (no rounding to a denominator)."""
    a = np.asarray(a, dtype=np.float64)
    return float(disc_numerator(a)) / math.sqrt(a.shape[0])


def make(a, den=MAX_DEN):
    """Candidate of a float matrix in [-1, 1], entries rounded to multiples of 1 / den."""
    a = np.clip(np.asarray(a, dtype=np.float64), -1, 1)
    return {"rows": np.rint(a * den).astype(np.int64).tolist(), "den": int(den)}


def load(output):
    """The candidate dict of a population output."""
    return {"rows": [list(map(int, r)) for r in output["rows"]], "den": int(output["den"])}
