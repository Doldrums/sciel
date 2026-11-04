"""The task's objective for a candidate to build on: from sciel.task import asset; weights = asset("weights").

q: {0, ..., n} -> [0, 1] (n <= 10, length n + 1) is the red probability between parts x and y of a step graphon on
the 2^n parts F_2^n when |x XOR y| = k; q[0] is the probability inside a part. The objective is the monochromatic K6
density F(q) = t(K6, W) + t(K6, 1 - W) = 32^-n sum_classes m (prod_e q(w_e) + prod_e (1 - q(w_e))) over the classes
of orbits(n). The classes depend only on n and are cached, so after the first call at a given n (5 s at n = 10)
each value and gradient takes about 0.3 s.
"""

from fractions import Fraction
from pathlib import Path

import numpy as np

from sciel.candidate_modules import load_candidate

_evaluator = load_candidate(str(Path(__file__).resolve().parents[1] / "evaluator.py"), "evaluator")


def orbits(n: int) -> tuple:
    """(W, m): the 15 pair weights of a representative of each class of 5-tuples (x_0 = 0) and the class sizes,
    summing to 32^n."""
    return _evaluator.orbits(int(n))


def _side(q, W, p):
    """sum p prod_e q(w_e) and its gradient in q."""
    V = q[W]
    prefix = np.cumprod(np.concatenate([np.ones((V.shape[0], 1)), V[:, :-1]], axis=1), axis=1)
    suffix = np.cumprod(np.concatenate([np.ones((V.shape[0], 1)), V[:, :0:-1]], axis=1), axis=1)[:, ::-1]
    others = prefix * suffix
    value = float(p @ (prefix[:, -1] * V[:, -1]))
    grad = np.bincount(W.ravel(), weights=(others * p[:, None]).ravel(), minlength=q.size)
    return value, grad


def value_and_gradient(q) -> tuple:
    """F(q) in floating point and its gradient in q."""
    q = np.asarray(q, dtype=float)
    n = q.size - 1
    W, m = orbits(n)
    W = W.astype(np.int64)
    p = m / float(32 ** n)
    red, g_red = _side(q, W, p)
    blue, g_blue = _side(1 - q, W, p)
    return red + blue, g_red - g_blue


def value(q) -> float:
    """F(q) in floating point, accurate to about 1e-14 relative."""
    return value_and_gradient(q)[0]


def exact_value(kernel) -> Fraction:
    """F exactly as the evaluator scores it, for an array q of multiples of 2**-32 or a pair (numerators,
    denominator)."""
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
