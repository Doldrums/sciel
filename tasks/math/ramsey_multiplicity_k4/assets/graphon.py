"""The task's objective for a candidate to build on: from sciel.task import asset; graphon = asset("graphon").

q has one red probability per class of pairs of the m = 3840 parts (classes are orbits of a fixed group G joined
with their transposes, numbered as the evaluator numbers them); W[i, j] = q[classes[i, j]] is the step graphon. The
objective is the monochromatic K4 density F(q) = t(K4, W) + t(K4, 1 - W).
"""

from pathlib import Path

import numpy as np

from sciel.candidate_modules import load_candidate

_evaluator = load_candidate(str(Path(__file__).resolve().parents[1] / "evaluator.py"), "evaluator")


def structure() -> tuple:
    """(classes, rows, cols, sizes): classes[i, j] is the class of the pair (i, j) (an m x m int32 array),
    (rows[c], cols[c]) a representative pair of class c and sizes[c] its number of ordered pairs."""
    return _evaluator.structure()


def build(q) -> np.ndarray:
    """W as an m x m array of floats."""
    return np.asarray(q, dtype=float)[structure()[0]]


def _s(W):
    """s_c = sum_{k,l} W_ik W_jk W_il W_jl W_kl at the representative (i, j) of every class c."""
    _, rows, cols, _ = structure()
    out = np.zeros(rows.size)
    for i in np.unique(rows):
        mine = np.nonzero(rows == i)[0]
        S = np.nonzero(W[i])[0]
        X = W[np.ix_(cols[mine], S)] * W[i, S]
        out[mine] = ((X @ W[np.ix_(S, S)]) * X).sum(axis=1)
    return out


def value_and_gradient(q) -> tuple:
    """F(q) in floating point and its gradient in q, dF/dq_c = 6 sizes_c (s_red - s_blue) / m^4; about 25 s on
    one core."""
    q = np.asarray(q, dtype=float)
    _, _, _, sizes = structure()
    m4 = float(structure()[0].shape[0]) ** 4
    W = build(q)
    red, blue = _s(W), _s(1 - W)
    return float(sizes @ (q * red + (1 - q) * blue) / m4), 6 * sizes * (red - blue) / m4


def value(q) -> float:
    """F(q) in floating point."""
    return value_and_gradient(q)[0]


def exact_value(kernel):
    """F as a Fraction, exactly as the evaluator scores it, for a pair (numerators, denominator) or an array of
    multiples of 2**-30; about 2 minutes on 12 cores, in one process per core."""
    return _evaluator.density(*_evaluator._kernel(kernel))


def snap(q, bits: int = 30) -> tuple:
    """q clipped to [0, 1] and rounded to multiples of 2**-bits, as (numerators, 2**bits), a kernel the evaluator
    accepts (bits <= 30)."""
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
