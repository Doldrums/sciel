"""The task's objective for a candidate to build on: from sciel.task import asset; fei = asset("fei").

A function f: {-1, 1}^n -> {-1, 1} is a table of 2^n bits, entry x = 1 where f(x) = -1, with bit i of x set when
x_{i+1} = -1. Its spectrum is a_S = 2^n fhat(S) = sum_x f(x) chi_S(x), chi_S(x) = (-1)^|S & x|, so
fhat(S)^2 = a_S^2 / 4^n, H = 2n - 2 sum_S a_S^2 log2|a_S| / 4^n and Inf = sum_S a_S^2 |S| / 4^n. The bound is F = H / (Inf - 1) for balanced f
(a_0 = 0) with Inf > 1. Flipping entry x changes every coefficient: a_S -= 2 f(x) chi_S(x), so a move costs O(2^n);
a single flip unbalances f, and swapping a TRUE and a FALSE entry keeps it balanced.
"""

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

from pathlib import Path

import numpy as np

from sciel.candidate_modules import load_candidate

MAX_N = 20
_evaluator = load_candidate(str(Path(__file__).resolve().parents[1] / "evaluator.py"), "evaluator")
popcounts = _evaluator.popcounts


def load(output) -> tuple:
    """(n, bits) of a scored function, from the output its evaluation recorded (a population entry's "output")."""
    n = int(output["n"])
    return n, _evaluator.decode(n, output["true_hex"])


def spectrum(n, bits) -> np.ndarray:
    """The exact integers a_S, int64, indexed by the bitmask of S."""
    return _evaluator.spectrum(n, bits)


def influence(n, bits):
    """Inf as an exact Fraction."""
    return _evaluator.influence(n, spectrum(n, bits))


def entropy(n, bits) -> tuple:
    """H enclosed in outward-rounded [lower, upper] Fractions, as the evaluator computes it."""
    H = _evaluator.entropy(n, spectrum(n, bits))
    return _evaluator._endpoint(H._mpi_[0]), _evaluator._endpoint(H._mpi_[1])


def exact_value(n, bits) -> float:
    """The certified lower end of H / (Inf - 1), exactly as scored after the validity checks; raises when f is not
    balanced or Inf <= 1."""
    return float(_evaluator.certify(*_evaluator._construction((n, bits)))["bound"][0])


def stats(n, a) -> tuple:
    """(H, Inf) in floating point from a spectrum."""
    w = (a * a).astype(np.float64) / 4.0 ** n
    nz = a != 0
    H = float(2 * n - 2 * (w[nz] * np.log2(np.abs(a[nz]).astype(np.float64))).sum())
    return H, float((w * popcounts(n)).sum())


def value(n, a) -> float:
    """H / (Inf - 1) in floating point from a spectrum; -inf when f is not balanced or Inf <= 1."""
    H, inf = stats(n, a)
    return H / (inf - 1) if a[0] == 0 and inf > 1 else float("-inf")


def character(n, x) -> np.ndarray:
    """chi_S(x) = (-1)^|S & x| for every S, int64."""
    return 1 - 2 * (popcounts(n)[np.arange(1 << n) & int(x)] & 1)


def flip(n, a, bits, x) -> np.ndarray:
    """The spectrum after flipping entry x of bits (bits itself is unchanged)."""
    return a - 2 * (1 - 2 * int(bits[x])) * character(n, x)


def flip_delta(n, bits, x, a=None) -> tuple:
    """(change of H, change of Inf) in floating point when entry x flips."""
    a = spectrum(n, bits) if a is None else a
    H0, I0 = stats(n, a)
    H1, I1 = stats(n, flip(n, a, bits, x))
    return H1 - H0, I1 - I0


def swap(n, a, bits, x, y) -> np.ndarray:
    """The spectrum after flipping entries x and y, which must differ, so balance is kept."""
    if bits[x] == bits[y]:
        raise ValueError("swap needs one TRUE and one FALSE entry")
    sx = 1 - 2 * int(bits[x])
    return a - 2 * sx * (character(n, x) - character(n, y))


def compose(g, h) -> tuple:
    """(m k, bits) of g(h(x^1), ..., h(x^m)) for g on m and h on k variables, block j of h on variables jk..jk+k-1;
    the O'Donnell-Tan composition. For balanced h, H(g o h) = H(g) + Inf(g) H(h) and Inf(g o h) = Inf(g) Inf(h), so
    iterating a balanced g drives H / Inf to H(g) / (Inf(g) - 1)."""
    (m, gb), (k, hb) = g, h
    if m * k > MAX_N:
        raise ValueError(f"composition on {m * k} variables exceeds {MAX_N}")
    gb, hb = np.asarray(gb, dtype=np.int64), np.asarray(hb, dtype=np.int64)
    idx = np.arange(1 << (m * k), dtype=np.int64)
    z = np.zeros_like(idx)
    for j in range(m):
        z |= hb[(idx >> (j * k)) & ((1 << k) - 1)] << j
    return m * k, gb[z].astype(np.uint8)


def rehost(n, bits, i, cells) -> tuple:
    """(n + 1, bits) where a new variable x_{n+1} takes over the role of x_{i+1} on the given cells: cells is a
    boolean array over the 2^n inputs, read at x with bit i cleared, and on those inputs f reads x_{n+1} in place of
    x_{i+1}. This is Numaro's split of an auxiliary variable's four cells two and two, from n = 17 to 18."""
    if n + 1 > MAX_N:
        raise ValueError(f"rehosting needs n + 1 <= {MAX_N}")
    bits, cells = np.asarray(bits, dtype=np.uint8), np.asarray(cells, dtype=bool)
    b = 1 << i
    idx = np.arange(1 << (n + 1), dtype=np.int64)
    z = idx & ((1 << n) - 1)
    moved = cells[z & ~b]
    z = np.where(moved, (z & ~b) | (((idx >> n) & 1) << i), z)
    return n + 1, bits[z].copy()
