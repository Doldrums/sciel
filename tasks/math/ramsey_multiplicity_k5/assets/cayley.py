"""The task's objective for a candidate to build on: from sciel.task import asset; cayley = asset("cayley").

q: F_2^n -> [0, 1] (n <= 8, length 2^n, z encoded as an integer) is the red probability between parts x and y of a
step graphon on 2^n equal parts when x XOR y = z; q[0] is the probability inside a part. The objective is the
monochromatic K5 density F(q) = t(K5, W) + t(K5, 1 - W). F is unchanged by lift (a coordinate that q ignores) and by
any linear bijection of F_2^n applied to the index z.
"""

from fractions import Fraction
from pathlib import Path

import numpy as np

from sciel.candidate_modules import load_candidate

_evaluator = load_candidate(str(Path(__file__).resolve().parents[1] / "evaluator.py"), "evaluator")


def _g(q):
    """G(z) = trace((D_u A)^3), u = q * q(. XOR z), A[a, b] = q(a XOR b), for every z."""
    N = q.size
    idx = np.arange(N)
    xor = idx[:, None] ^ idx[None, :]
    A = q[xor]
    out = np.empty(N)
    for lo in range(0, N, 64):
        z = np.arange(lo, min(lo + 64, N))
        P = (q[None, :] * q[xor[z]])[:, :, None] * A[None]
        out[z] = np.einsum("kab,kba->k", P @ P, P)
    return out


def value_and_gradient(q) -> tuple:
    """F(q) in floating point and its gradient in q, about 0.2 s at n = 8. dF/dq(z) = 10 N^-4 (G_red(z) - G_blue(z))."""
    q = np.asarray(q, dtype=float)
    N = q.size
    red, blue = _g(q), _g(1 - q)
    return float((q @ red + (1 - q) @ blue) / N ** 4), 10 * (red - blue) / N ** 4


def value(q) -> float:
    """F(q) in floating point, accurate to about 1e-15 relative."""
    return value_and_gradient(q)[0]


def exact_value(kernel) -> Fraction:
    """F exactly as the evaluator scores it, for an array q of multiples of 2**-32 or a pair (numerators,
    denominator); about 10 s at n = 8."""
    return _evaluator.density(*_evaluator._kernel(kernel))


def snap(q, bits: int = 20) -> tuple:
    """q clipped to [0, 1] and rounded to multiples of 2**-bits, as (numerators, 2**bits), a kernel the evaluator
    accepts (bits <= 32)."""
    d = 1 << int(bits)
    return [int(x) for x in np.rint(np.clip(np.asarray(q, dtype=float), 0, 1) * d)], d


def as_float(kernel) -> np.ndarray:
    """q as binary64 values from a pair (numerators, denominator) or an array."""
    if isinstance(kernel, tuple) and len(kernel) == 2:
        return np.asarray(kernel[0], dtype=float) / kernel[1]
    return np.asarray(kernel, dtype=float)


def load(output) -> tuple:
    """The (numerators, denominator) of a scored kernel, from the output its evaluation recorded (a population
    entry's "output")."""
    return list(output["numerators"]), int(output["denominator"])


def lift(q) -> np.ndarray:
    """q on F_2^(n+1) ignoring the new top coordinate: the same graphon, so the same F, with twice the freedom."""
    q = np.asarray(q, dtype=float)
    return np.concatenate([q, q])


def relabel(q, matrix) -> np.ndarray:
    """q composed with the linear map of F_2^n whose columns are the integers in matrix (images of the unit
    vectors); for an invertible map F is unchanged."""
    q = np.asarray(q, dtype=float)
    n = q.size.bit_length() - 1
    image = np.zeros(q.size, dtype=np.int64)
    for i in range(n):
        image ^= ((np.arange(q.size) >> i) & 1) * int(matrix[i])
    return q[image]
