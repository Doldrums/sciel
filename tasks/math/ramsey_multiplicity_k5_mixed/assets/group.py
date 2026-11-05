"""The task's objective for a candidate to build on: from sciel.task import asset; group = asset("group").

On G = Z_{m_1} x ... x Z_{m_k} (order N <= 256, mixed-radix index, first modulus most significant) a symmetric
q: G -> [0, 1] is the red probability between parts x and y of a step graphon on N equal parts when x - y = z; q[0]
is the probability inside a part. The objective is the monochromatic K5 density F(q) = t(K5, W) + t(K5, 1 - W). F is
unchanged by lift (a factor that q ignores) and by any automorphism of G applied to the index z.
"""

import math
from pathlib import Path

import numpy as np

from sciel.candidate_modules import load_candidate

_evaluator = load_candidate(str(Path(__file__).resolve().parents[1] / "evaluator.py"), "evaluator")
MAX_ORDER = _evaluator.MAX_ORDER
_TABLES = {}


def tables(moduli) -> tuple:
    """(difference, negation) of the group: difference[a, b] the index of a - b, negation[z] the index of -z."""
    key = tuple(int(m) for m in moduli)
    if key not in _TABLES:
        _TABLES[key] = _evaluator.group(key)
    return _TABLES[key]


def _g(q, difference):
    """G(z) = trace((D_u A)^3), u = q * q(. - z), A[a, b] = q(a - b), for every z."""
    N = q.size
    A = q[difference]
    out = np.empty(N)
    for lo in range(0, N, 64):
        z = np.arange(lo, min(lo + 64, N))
        P = (q[None, :] * q[difference[:, z].T])[:, :, None] * A[None]
        out[z] = np.einsum("kab,kba->k", P @ P, P)
    return out


def value_and_gradient(moduli, q) -> tuple:
    """F(q) in floating point and its gradient along symmetric q, about 0.1 s at N = 192. The raw derivative is
    10 N^-4 (G_red(z) - G_blue(z)); it is averaged over z and -z so a step keeps q symmetric."""
    difference, negation = tables(moduli)
    q = np.asarray(q, dtype=float)
    N = q.size
    red, blue = _g(q, difference), _g(1 - q, difference)
    g = 10 * (red - blue) / N ** 4
    return float((q @ red + (1 - q) @ blue) / N ** 4), (g + g[negation]) / 2


def value(moduli, q) -> float:
    """F(q) in floating point, accurate to about 1e-15 relative."""
    return value_and_gradient(moduli, q)[0]


def symmetrize(moduli, q) -> np.ndarray:
    """(q(z) + q(-z)) / 2."""
    q = np.asarray(q, dtype=float)
    return (q + q[tables(moduli)[1]]) / 2


def snap(moduli, q, bits: int = 32) -> dict:
    """The kernel {"moduli", "q": (numerators, 2**bits)} of q symmetrized, clipped to [0, 1] and rounded to multiples
    of 2**-bits, which the evaluator accepts (bits <= 32)."""
    d = 1 << int(bits)
    x = np.rint(np.clip(symmetrize(moduli, q), 0, 1) * d)
    x = np.minimum(x, x[tables(moduli)[1]])
    return {"moduli": [int(m) for m in moduli], "q": ([int(v) for v in x], d)}


def exact_value(kernel):
    """F exactly as the evaluator scores it, a Fraction, for a kernel dict; about 5 s at N = 192."""
    return _evaluator.density(*_evaluator._kernel(kernel))


def as_float(kernel) -> tuple:
    """(moduli, q as binary64) of a kernel dict."""
    q = kernel["q"]
    if isinstance(q, tuple) and len(q) == 2:
        return list(kernel["moduli"]), np.asarray(q[0], dtype=float) / q[1]
    return list(kernel["moduli"]), np.asarray(q, dtype=float)


def load(output) -> dict:
    """The kernel dict of a scored kernel, from the output its evaluation recorded (a population entry's "output")."""
    return {"moduli": list(output["moduli"]), "q": (list(output["numerators"]), int(output["denominator"]))}


def lift(moduli, q, m) -> tuple:
    """(moduli + [m], q on G x Z_m ignoring the new factor): the same graphon, so the same F, with more freedom."""
    if math.prod(moduli) * m > MAX_ORDER:
        raise ValueError(f"the lifted order exceeds {MAX_ORDER}")
    return list(moduli) + [int(m)], np.repeat(np.asarray(q, dtype=float), m)


def orbits(moduli) -> np.ndarray:
    """For every z the smaller index of z and -z, the free coordinates of a symmetric q."""
    return np.minimum(np.arange(math.prod(moduli)), tables(moduli)[1])
