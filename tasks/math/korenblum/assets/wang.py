"""Wang's construction and its multi-factor generalization: from sciel.task import asset; wang = asset("wang").

Wang (Integral Equations Operator Theory 61, 2008) bounds Korenblum's constant by the pairs f = (a + z^n) psi,
g = z (1 + a z^n) psi. On |z| = 1, |a + z^n| = |1 + a z^n| for real |a| < 1, so |f| = |g| there identically for every
psi, and |f / g| = |a + z^n| / (r |1 + a z^n|) on |z| = r is largest at z^n = r^n, where it is 1 at the root c(a)
of a + r^n = r (1 + a r^n). z^n + a fails to be dominated by z (1 + a z^n), that is ||f|| > ||g|| for some psi, exactly
when a^2 > (n - 2) / (2 (n - 1)); at the threshold the extremal weight is psi = (1 - b z^n)^-2 with b = 2a / (n - 2),
where ||f|| = ||g||. n = 10 is best: a = 2/3, b = 1/6 and c(2/3) = 0.6778994229535, Wang's kappa <= 0.6778994...;
every a > 2/3 gives a valid pair with c(a) slightly above it (dc/da is about 1.2).
"""

from fractions import Fraction
import math

import numpy as np
from scipy.linalg import eigh

from sciel.task import asset

bergman = asset("bergman")


def threshold(n=10) -> float:
    """The domination threshold a_n with a_n^2 = (n - 2) / (2 (n - 1)); 2/3 at n = 10."""
    return math.sqrt((n - 2) / (2 * (n - 1)))


def ratio_root(a, n=10) -> float:
    """c(a): the root in (0, 1) of a + r^n = r (1 + a r^n), where |f / g| reaches 1 on the inner circle."""
    a = float(a)

    def phi(r):
        return a + r ** n - r * (1 + a * r ** n)
    rs = np.linspace(1e-3, 1 - 1e-9, 200001)
    sign = np.nonzero(phi(rs) <= 0)[0]
    if not sign.size:
        return 1.0
    lo, hi = rs[sign[0] - 1], rs[sign[0]]
    for _ in range(200):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if phi(mid) > 0 else (lo, mid)
    return hi


RECORD = float(ratio_root(Fraction(2, 3), 10))


def multiply(p, q) -> list:
    """Exact product of two coefficient lists."""
    out = [Fraction(0)] * (len(p) + len(q) - 1)
    for i, x in enumerate(p):
        if x:
            for j, y in enumerate(q):
                out[i + j] += x * y
    return out


def factors(pairs) -> tuple:
    """Numerator prod (a_i + z^{n_i}) and denominator z prod (1 + a_i z^{n_i}) of the ratio f / g, exactly, for
    pairs [(a_i, n_i)] with real |a_i| < 1; both have modulus prod |a_i + z^{n_i}| on |z| = 1."""
    num, den = [Fraction(1)], [Fraction(0), Fraction(1)]
    for a, n in pairs:
        a = Fraction(a)
        num = multiply(num, [a] + [Fraction(0)] * (n - 1) + [Fraction(1)])
        den = multiply(den, [Fraction(1)] + [Fraction(0)] * (n - 1) + [a])
    return num, den


def spread(psi, step) -> list:
    """psi(z^step) as coefficients in z."""
    out = [Fraction(0)] * ((len(psi) - 1) * step + 1)
    for k, x in enumerate(psi):
        out[k * step] = Fraction(x)
    return out


def pair(a=2 / 3, n=10, b=None, K=30) -> tuple:
    """Wang's pair (f, g) exactly, with psi = sum_{k <= K} (k + 1) b^k z^{nk}, the truncated extremal weight rounded
    to binary64 (|f| = |g| on |z| = 1 holds for any psi); a is taken exactly, b defaults to 2a / (n - 2)."""
    a = Fraction(a)
    b = 2 * a / (n - 2) if b is None else Fraction(b)
    psi = spread([Fraction(float((k + 1) * b ** k)) for k in range(K + 1)], n)
    num, den = factors([(a, n)])
    return multiply(num, psi), multiply(den, psi)


def default(delta=1e-11, margin=1e-10) -> tuple:
    """(c, f, g): Wang's pair at a = 2/3 + delta in binary64, degree 311, with c = c(a) + margin. Its norm gap is
    about 1.2 delta and its combined_score 0.99999996."""
    a = 2 / 3 + delta
    f, g = pair(a, 10)
    return ratio_root(a, 10) + margin, f, g


def best_weight(num, den, K, step=1) -> tuple:
    """The weight psi(z^step) of degree at most K * step maximizing ||num psi|| / ||den psi||, a generalized
    eigenproblem on the Bergman Gram matrices in floating point: (ratio, psi coefficients in z^step). A ratio above 1
    makes (num psi, den psi) a pair with ||f|| > ||g||."""
    num, den = np.asarray([float(x) for x in num]), np.asarray([float(x) for x in den])
    size = max(num.size, den.size) + K * step

    def gram(p):
        cols = np.zeros((size, K + 1))
        for k in range(K + 1):
            cols[k * step:k * step + p.size, k] = p
        return cols.T @ (cols / np.arange(1, size + 1)[:, None])
    values, vectors = eigh(gram(num), gram(den))
    psi = vectors[:, -1] / vectors[0, -1] if vectors[0, -1] else vectors[:, -1]
    return math.sqrt(max(values[-1], 0.0)), psi
