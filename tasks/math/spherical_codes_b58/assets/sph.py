"""Spherical codes: N points on the unit sphere S^(d-1) whose minimal angle is as large as possible, i.e. whose largest
pair cosine is as small as possible. The cosine of two points, exactly as scored, the near-active pairs, a smooth
maximum with gradients, a cosine minimizer, restarts, and lattice and neighbour-N constructions.

A code is an (n, d) array of nonzero points; only their directions count (the cosine of x and y is
<x, y> / (|x| |y|)), so points need not be normalized.
"""

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

import base64
from fractions import Fraction
import itertools
import math
import time

import numpy as np
from scipy.optimize import linprog, minimize
from scipy.sparse import coo_matrix

from sciel.task import asset

try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
except ImportError:
    pass

_data = asset("sph_data")
_best = asset("sph_best")
# Instances are keyed "d_n", e.g. "10_74" for 74 points in R^10 (on S^9).
INSTANCES = tuple(_data.INSTANCES)
DIM = {k: int(k.split("_")[0]) for k in INSTANCES}
SIZE = {k: int(k.split("_")[1]) for k in INSTANCES}
# The table's cosine as displayed (12 digits, rounded up), and the bar: the exact squared cosine of best(key).
PUBLISHED = {k: Fraction(_data.COS[k]) for k in INSTANCES}
BEST_COS2 = {k: Fraction(_best.COS2[k]) for k in INSTANCES}
BEST = {k: math.sqrt(float(BEST_COS2[k])) for k in INSTANCES}


def decode(text: str, d: int) -> np.ndarray:
    """Points from base64 of little-endian binary64 rows of d coordinates, as (n, d)."""
    return np.frombuffer(base64.b64decode(text), dtype="<f8").reshape(-1, d).astype(np.float64)


def encode(points) -> str:
    return base64.b64encode(np.asarray(points, dtype="<f8").tobytes()).decode()


def published(key: str) -> np.ndarray:
    """The table's coordinates for the instance, as (n, d)."""
    return decode(_data.COORDS[key], DIM[key])


def best(key: str) -> np.ndarray:
    """The bar's points for the instance (the table's code after local search), unit rows, as (n, d)."""
    return decode(_best.COORDS[key], DIM[key])


def directions(points) -> np.ndarray:
    """The unit vectors of the (n, d) points in floating point. Raises ValueError on a non-finite or zero point."""
    x = np.asarray(points, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("points must be a finite (n, d) array")
    top = np.abs(x).max(axis=1, keepdims=True)
    if (top == 0).any():
        raise ValueError("a point is the zero vector")
    x = x / top
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def cosines(points) -> np.ndarray:
    """The (n, n) floating cosines between the points, -inf on the diagonal."""
    u = directions(points)
    g = u @ u.T
    np.fill_diagonal(g, -np.inf)
    return g


def max_cos_fast(points) -> float:
    """The largest pair cosine in floating point (within about 1e-15 of the exact one)."""
    return float(cosines(points).max())


def _integers(row):
    ratios = [v.as_integer_ratio() for v in row]
    q = max(b for _, b in ratios)
    return [a * (q // b) for a, b in ratios]


def cos2(points) -> Fraction:
    """The largest pair cosine squared with its sign, exactly: a binary64 point is an integer vector a over a power of
    two, the cosine of two points is <a, b> / (|a| |b|), so <a, b> |<a, b>| / (|a|^2 |b|^2) is a rational with the
    same order. Raises ValueError on a zero or non-finite point and when two points have the same direction."""
    p = np.asarray(points, dtype=np.float64)
    g = cosines(p)
    # A floating cosine of unit rows is within about 2 (d + 2) units of 2^-53 of the exact one, so every pair that can
    # have the largest cosine is within 1e-10 of the floating maximum; those are compared exactly.
    i, j = np.nonzero(np.triu(g >= g.max() - 1e-10, 1))
    ints, norms, top = {}, {}, None
    for a, b in zip(i.tolist(), j.tolist()):
        for k in (a, b):
            if k not in ints:
                ints[k] = _integers(p[k].tolist())
                norms[k] = sum(v * v for v in ints[k])
        ip = sum(x * y for x, y in zip(ints[a], ints[b]))
        value = Fraction(ip * abs(ip), norms[a] * norms[b])
        if top is None or value > top:
            top = value
    if top >= 1:
        raise ValueError("two points have the same direction")
    return top


def sqrt_signed(c2: Fraction, bits=160) -> Fraction:
    """sign(c2) sqrt(|c2|) rounded toward zero to a multiple of 2^-bits."""
    root = math.isqrt(abs(c2.numerator) * 4 ** bits // c2.denominator)
    return Fraction(root if c2 > 0 else -root, 2 ** bits)


def max_cos(points) -> float:
    """The largest pair cosine as scored: from the exact square, rounded to binary64."""
    return float(sqrt_signed(cos2(points)))


def contacts(points, tol=1e-10) -> np.ndarray:
    """The pairs (k, 2) of i < j whose cosine is within tol of the largest."""
    g = cosines(points)
    return np.argwhere(np.triu(g >= g.max() - tol, 1))


def degrees(points, tol=1e-10) -> np.ndarray:
    """The number of contacts of each point; points of degree below d are usually free to move (rattlers)."""
    pairs = contacts(points, tol)
    return np.bincount(pairs.ravel(), minlength=len(np.asarray(points)))


def soft_max_cos(x, beta, n, d):
    """The smooth maximum (1 / beta) log sum_{i != j} exp(beta c_ij) of the pair cosines of the flat (n * d) vector x,
    and its gradient in x. It tends to the largest cosine as beta grows; beta near 1 / (gap between the largest
    cosines) is sharp enough to rank nearby codes."""
    x = x.reshape(n, d)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    u = x / norm
    g = u @ u.T
    np.fill_diagonal(g, -np.inf)
    top = g.max()
    w = np.exp(beta * (g - top))
    total = w.sum()
    value = top + math.log(total) / beta
    w /= total
    grad_u = (w + w.T) @ u
    grad_x = (grad_u - (grad_u * u).sum(axis=1, keepdims=True) * u) / norm
    return value, grad_x.ravel()


def descend(points, betas=(10, 30, 100, 300, 1e3, 3e3, 1e4, 3e4, 1e5, 1e6, 1e7, 1e8), iters=1000,
            deadline=None) -> np.ndarray:
    """L-BFGS on soft_max_cos for each sharpness in betas in turn (start low from a random code, high from a good
    one; sharpness up to 1e8 brings a near-optimal code close to its optimum much faster than maximize, which then
    finishes). Returns the unit vectors with the smallest true largest cosine among the start and the stages."""
    u = directions(points)
    n, d = u.shape
    top, top_c = u, max_cos_fast(u)
    def stop(intermediate_result):
        if deadline is not None and time.time() > deadline:
            raise StopIteration
    for beta in betas:
        if deadline is not None and time.time() > deadline:
            break
        res = minimize(soft_max_cos, u.ravel(), args=(beta, n, d), jac=True, method="L-BFGS-B", callback=stop,
                       options={"maxiter": iters, "gtol": 1e-14, "ftol": 1e-16})
        u = directions(res.x.reshape(n, d))
        c = max_cos_fast(u)
        if c < top_c:
            top, top_c = u, c
    return top


def _lp_step(u, step, near, seconds=10.0):
    n, d = u.shape
    g = u @ u.T
    np.fill_diagonal(g, -np.inf)
    i, j = np.nonzero(np.triu(g >= g.max() - near, 1))
    k = len(i)
    # Row r: c_ij + <u_j - c_ij u_i, du_i> + <u_i - c_ij u_j, du_j> - z <= 0; minimize z.
    gi = u[j] - g[i, j][:, None] * u[i]
    gj = u[i] - g[i, j][:, None] * u[j]
    rows = np.repeat(np.arange(k), 2 * d + 1)
    cols = np.concatenate([(d * i)[:, None] + np.arange(d), (d * j)[:, None] + np.arange(d),
                           np.full((k, 1), n * d)], axis=1).ravel()
    vals = np.concatenate([gi, gj, -np.ones((k, 1))], axis=1).ravel()
    a = coo_matrix((vals, (rows, cols)), shape=(k, n * d + 1)).tocsr()
    c = np.zeros(n * d + 1)
    c[-1] = 1.0
    bounds = [(-step, step)] * (n * d) + [(None, None)]
    res = linprog(c, A_ub=a, b_ub=-g[i, j], bounds=bounds, method="highs-ipm",
                  options={"time_limit": seconds, "ipm_optimality_tolerance": 1e-12,
                           "primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10})
    if res.status != 0 or res.x is None:
        return u
    return directions(u + res.x[:-1].reshape(n, d))


def maximize(points, rounds=200, step=None, deadline=None) -> np.ndarray:
    """Sequential linear programming on the near-active pairs: each round moves every point by at most step per
    coordinate to lower the linearized largest cosine, keeps the move and widens the step when the true largest
    cosine falls, and otherwise quarters the step (also when an LP takes over 10 s); below 1e-15 it starts over from
    16 times the largest step that moved, and stops after a sweep that does not move. One round is one sparse LP with n d variables.
    Returns the best unit vectors found."""
    u = directions(points)
    top = max_cos_fast(u)
    step = 1e-3 if step is None else step
    widest, moved, largest = step, False, 0.0
    for _ in range(rounds):
        if deadline is not None and time.time() > deadline:
            break
        if step < 1e-15:
            # The LP is solved to about 1e-10, so small steps can fail before the optimum: start over from 16 times
            # the largest step that moved in the last sweep, while it still moved.
            if not moved:
                break
            step, moved, largest = min(16 * largest, widest), False, 0.0
        left = 10.0 if deadline is None else min(10.0, deadline - time.time())
        # A move of at most step per coordinate changes a cosine by at most 2 sqrt(d) step, so pairs farther than
        # twice that below the largest cosine cannot overtake it.
        q = _lp_step(u, step, near=4 * math.sqrt(u.shape[1]) * step + 1e-13, seconds=max(left, 0.1))
        c = max_cos_fast(q)
        if c < top - 1e-16:
            u, top, moved, largest = q, c, True, max(largest, step)
            step = min(2 * step, widest)
        else:
            step /= 4
    return u


def shake(points, sigma, rng, group=None) -> np.ndarray:
    """The unit points moved by Gaussian noise of scale sigma times the minimal chord: all of them, or only the
    indices in group."""
    u = directions(points)
    scale = sigma * math.sqrt(max(2 - 2 * max_cos_fast(u), 1e-12))
    idx = np.arange(len(u)) if group is None else np.asarray(group)
    u[idx] += rng.normal(scale=scale, size=(len(idx), u.shape[1]))
    return directions(u)


def improve(points, deadline, rng, sigma=(1e-4, 1e-2)) -> np.ndarray:
    """Basin hopping until deadline: descend and maximize, then repeatedly shake by a log-uniform sigma in the range, descend at
    high sharpness, maximize, and keep the result when its largest cosine is lower."""
    top = descend(points, betas=(1e5, 1e6, 1e7, 1e8), deadline=deadline)
    top = maximize(top, rounds=10**6, deadline=deadline)
    top_c = max_cos_fast(top)
    while time.time() < deadline:
        q = shake(top, 10 ** rng.uniform(*np.log10(sigma)), rng)
        q = descend(q, betas=(3e3, 3e4, 3e5, 3e6, 3e7, 1e8), deadline=deadline)
        q = maximize(q, rounds=10**6, deadline=deadline)
        c = max_cos_fast(q)
        if c < top_c:
            top, top_c = q, c
    return top


def random_code(d: int, n: int, rng, deadline=None) -> np.ndarray:
    """A fresh local optimum: n uniform random points on S^(d-1), descended from low to high sharpness, then
    maximized."""
    u = descend(rng.normal(size=(n, d)), deadline=deadline)
    return maximize(u, rounds=10**6, deadline=deadline)


def e8_roots() -> np.ndarray:
    """The 240 minimal vectors of E8 (norm 2): the 112 permutations of (+-1, +-1, 0^6) and the 128 vectors
    (+-1/2)^8 with an even number of minus signs. Cosines 0, +-1/2, -1."""
    out = [v for v in dn_roots(8)]
    for signs in itertools.product((-0.5, 0.5), repeat=8):
        if sum(s < 0 for s in signs) % 2 == 0:
            out.append(np.array(signs))
    return np.array(out)


def dn_roots(m: int) -> np.ndarray:
    """The 2 m (m - 1) minimal vectors of D_m: +-e_i +-e_j, i < j."""
    out = []
    for i, j in itertools.combinations(range(m), 2):
        for a, b in itertools.product((-1.0, 1.0), repeat=2):
            v = np.zeros(m)
            v[i], v[j] = a, b
            out.append(v)
    return np.array(out)


def antipodal(points) -> np.ndarray:
    """The points and their negatives."""
    u = directions(points)
    return np.vstack([u, -u])


def remove(points, k: int) -> np.ndarray:
    """k points removed greedily, each time one with the most contacts at the largest cosine (ties by largest
    second cosine)."""
    u = directions(points)
    for _ in range(k):
        g = cosines(u)
        top = g.max()
        load = (g >= top - 1e-10).sum(axis=1) + np.sort(g, axis=1)[:, -2]
        u = np.delete(u, int(np.argmax(load)), axis=0)
    return u


def hole(points, rng, tries=4000) -> np.ndarray:
    """An approximate deepest hole: the direction farthest in angle from every point, from the best of tries random
    directions refined by the smooth maximum of its cosines. Returns a unit vector."""
    u = directions(points)
    cand = directions(rng.normal(size=(tries, u.shape[1])))
    x = cand[np.argmin((cand @ u.T).max(axis=1))]
    for beta in (100, 1000, 10000):
        def f(y):
            norm = np.linalg.norm(y)
            c = u @ (y / norm)
            top = c.max()
            w = np.exp(beta * (c - top))
            value = top + math.log(w.sum()) / beta
            gu = (w / w.sum()) @ u
            return value, (gu - (gu @ y / norm) * y / norm) / norm
        x = minimize(f, x, jac=True, method="L-BFGS-B", options={"maxiter": 200}).x
    return x / np.linalg.norm(x)


def insert(points, k: int, rng) -> np.ndarray:
    """k points added one at a time at approximate deepest holes."""
    u = directions(points)
    for _ in range(k):
        u = np.vstack([u, hole(u, rng)])
    return u
