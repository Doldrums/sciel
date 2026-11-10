"""Spectral entropy over total influence for the Fourier Entropy-Influence constant C_71; a program uses it as:
from sciel.task import asset; fei = asset("fei").

A candidate is {"n": n, "table": hex}: a Boolean function f on {-1, 1}^n, 2 <= n <= MAX_N, whose truth table is the
integer int(hex, 16) with bit x set when f(x) = -1 (x read as the n-bit index of the point, bit i = coordinate i).
With c(S) = sum_x f(x) chi_S(x) the integer Walsh coefficients, f^(S)^2 = c(S)^2 / 4^n,

    H = sum_S f^(S)^2 log2(1 / f^(S)^2),   I = sum_S f^(S)^2 |S|,

and if f is balanced (c(empty) = 0) with I > 1, O'Donnell and Tan's amplification gives C_71 >= H / (I - 1)
(Hod 2017, Prop. 1.2). Numpy arrays below hold f as +-1 int64 of length 2^n.
"""

from fractions import Fraction

import mpmath
import numpy as np

MAX_N = 22
DIGITS = 50


def popcounts(n):
    """|S| for every index S < 2^n."""
    p = np.zeros(1, dtype=np.int64)
    for _ in range(n):
        p = np.concatenate([p, p + 1])
    return p


def wht(f):
    """Integer Walsh-Hadamard transform: c(S) = sum_x f(x) (-1)^|S & x|."""
    a = np.asarray(f, dtype=np.int64).copy()
    h = 1
    while h < len(a):
        a = a.reshape(-1, 2, h)
        a = np.stack([a[:, 0] + a[:, 1], a[:, 0] - a[:, 1]], axis=1).reshape(-1)
        h *= 2
    return a


def rates(f):
    """(ratio, H, I) in floats for a +-1 array f; ratio is -inf when f is unbalanced or I <= 1."""
    f = np.asarray(f, dtype=np.int64)
    n = len(f).bit_length() - 1
    c = wht(f)
    w = (c.astype(np.float64) / len(f)) ** 2
    infl = float((w * popcounts(n)).sum())
    nz = w > 0
    ent = float(-(w[nz] * np.log2(w[nz])).sum())
    if c[0] != 0 or infl <= 1:
        return float("-inf"), ent, infl
    return ent / (infl - 1), ent, infl


def value(candidate):
    """The float ratio of a candidate dict (no certification)."""
    return rates(array(candidate))[0]


def array(candidate):
    """+-1 int64 array of a candidate {"n", "table"}; raises ValueError on malformed input."""
    if not isinstance(candidate, dict) or set(candidate) != {"n", "table"}:
        raise ValueError('candidate must be {"n": n, "table": hex}')
    n = candidate["n"]
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or not 2 <= int(n) <= MAX_N:
        raise ValueError(f"n must be an integer in [2, {MAX_N}]")
    n = int(n)
    table = candidate["table"]
    if not isinstance(table, str):
        raise ValueError("table must be a hex string")
    table = table.strip().lower().removeprefix("0x")
    if not table or any(ch not in "0123456789abcdef" for ch in table):
        raise ValueError("table must be a hex string")
    value = int(table, 16)
    if value >> (1 << n):
        raise ValueError("table has bits beyond 2^n points")
    raw = value.to_bytes((1 << n) // 8 if n >= 3 else 1, "little")
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")[: 1 << n]
    return 1 - 2 * bits.astype(np.int64)


def make(f):
    """Candidate dict of a +-1 array of length 2^n."""
    f = np.asarray(f)
    n = len(f).bit_length() - 1
    if len(f) != 1 << n or not 2 <= n <= MAX_N or not np.all(np.abs(f) == 1):
        raise ValueError("f must be +-1 of length 2^n, 2 <= n <= MAX_N")
    bits = (f < 0).astype(np.uint8)
    raw = np.packbits(np.pad(bits, (0, (-len(bits)) % 8)), bitorder="little").tobytes()
    return {"n": n, "table": format(int.from_bytes(raw, "little"), "x")}


def load(output):
    """The candidate dict of a population output."""
    return {"n": int(output["n"]), "table": output["table"]}


def certify(candidate):
    """(lower, H, I) with lower an mpf rounded down: H by interval logs at DIGITS digits over the distinct |c(S)|,
    I as an exact Fraction. Raises ValueError unless f is balanced with I > 1."""
    f = array(candidate)
    n = len(f).bit_length() - 1
    c = wht(f)
    if c[0] != 0:
        raise ValueError("f is not balanced: c(empty) != 0")
    sq = c * c
    infl = Fraction(int((sq * popcounts(n)).sum()), 4 ** n)
    if infl <= 1:
        raise ValueError("total influence must exceed 1")
    vals, counts = np.unique(np.abs(c[c != 0]), return_counts=True)
    iv = mpmath.iv
    saved, iv.dps = iv.dps, DIGITS
    try:
        ent = iv.mpf(0)
        for v, k in zip(vals.tolist(), counts.tolist()):
            w = iv.mpf(v * v) / iv.mpf(4 ** n)
            ent += int(k) * w * (-iv.log(w) / iv.log(2))
        ratio = ent / (iv.mpf(infl.numerator) / infl.denominator - 1)
        # Endpoints are read out at the same precision, so they are not rounded again.
        with mpmath.workdps(DIGITS):
            lower = mpmath.mpf(ratio.a)
    finally:
        iv.dps = saved
    return lower, ent, infl
