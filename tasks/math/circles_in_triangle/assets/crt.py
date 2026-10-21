"""Equal circles in the isosceles right triangle with legs 1: the radius of a set of centres, its contacts, rattlers,
a radius maximizer and hexagonal starts.

The triangle has its right angle at the origin and its sides on x = 0, y = 0 and x + y = 1. The radius of N centres
is the largest r with every centre at distance >= r from the three sides and every pair at distance >= 2r:
r = min(x, y, (1 - x - y) / sqrt 2, min pair distance / 2), over all centres.
"""

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

import base64
from fractions import Fraction
import math

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from sciel.task import asset

try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
except ImportError:
    pass

_data = asset("crt_data")
_best = asset("crt_best")
SIZES = tuple(range(150, 300))
RADIUS = {n: float(_data.RADIUS[n]) for n in SIZES}
RADIUS_EXACT = {n: Fraction(_data.RADIUS[n]) for n in SIZES}
DATE = dict(_data.DATE)
# The bar: the published radius, or for the N in crt_best the radius of the published packing improved further.
BEST_R2 = {n: Fraction(_best.R2[n]) if n in _best.R2 else RADIUS_EXACT[n] ** 2 for n in SIZES}
BEST = {n: math.sqrt(float(BEST_R2[n])) for n in SIZES}
_SQRT2 = math.sqrt(2.0)


def published(n: int) -> np.ndarray:
    """The published best centres for n circles as an (n, 2) array."""
    return np.array(_data.COORDS[n].split(), dtype=np.float64).reshape(n, 2)


def decode(text: str) -> np.ndarray:
    """Centres from a scored output's packings entry (base64 of little-endian binary64 x y pairs), as (n, 2)."""
    return np.frombuffer(base64.b64decode(text), dtype="<f8").reshape(-1, 2).astype(np.float64)


def best(n: int) -> np.ndarray:
    """The best known centres for n circles, the bar the task scores against, as an (n, 2) array."""
    return np.array(_best.COORDS.get(n, _data.COORDS[n]).split(), dtype=np.float64).reshape(n, 2)


def slacks(points) -> tuple:
    """Float distances: (side (n, 3) distances to x = 0, y = 0, x + y = 1; pair (n, n) centre distances, inf on the
    diagonal)."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    side = np.stack([p[:, 0], p[:, 1], (1.0 - p[:, 0] - p[:, 1]) / _SQRT2], axis=1)
    pair = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(pair, np.inf)
    return side, pair


def radius_fast(points) -> float:
    """The radius in floating point, negative when a centre is outside the triangle."""
    side, pair = slacks(points)
    return float(min(side.min(), pair.min() / 2))


def radius2(points) -> Fraction:
    """The squared radius exactly, from the binary64 centres as rationals, as scored. Raises ValueError when a centre
    is outside the closed triangle or two centres coincide."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(p).all():
        raise ValueError("centres must be finite")
    ratios = [v.as_integer_ratio() for v in p.ravel().tolist()]
    d = max(q for _, q in ratios)
    z = [a * (d // q) for a, q in ratios]
    xs, ys = z[0::2], z[1::2]
    if min(xs) < 0 or min(ys) < 0 or max(x + y for x, y in zip(xs, ys)) > d:
        raise ValueError("a centre is outside the triangle")
    best = min(min(4 * x * x, 4 * y * y, 2 * (d - x - y) ** 2) for x, y in zip(xs, ys))
    # Floating squared distances are within a few units in the last place of the exact ones, so every pair that can
    # be the closest is within a relative 1e-9 of the floating minimum; those are compared exactly.
    sq = ((p[:, None, :] - p[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(sq, np.inf)
    i, j = np.nonzero(np.triu(sq <= sq.min() * (1 + 1e-9) + 1e-300, 1))
    for a, b in zip(i.tolist(), j.tolist()):
        best = min(best, (xs[a] - xs[b]) ** 2 + (ys[a] - ys[b]) ** 2)
    if best == 0:
        raise ValueError("two centres coincide or a centre is on a side")
    return Fraction(best, 4 * d * d)


def radius(points) -> float:
    """The radius as scored: the square root of the exact squared radius, correctly rounded twice."""
    return math.sqrt(float(radius2(points)))


def contacts(points, tol=1e-9) -> tuple:
    """The contact graph: (pairs (k, 2) of i < j whose distance is within tol * r of 2r, walls (m, 2) of (i, side)
    with side 0 for x = 0, 1 for y = 0, 2 for x + y = 1, whose distance is within tol * r of r)."""
    side, pair = slacks(points)
    r = min(side.min(), pair.min() / 2)
    i, j = np.nonzero(np.triu(pair - 2 * r <= tol * r, 1))
    w = np.argwhere(side - r <= tol * r)
    return np.stack([i, j], axis=1), w


_WALL_DIRECTIONS = np.array([[-1.0, 0.0], [0.0, -1.0], [1 / _SQRT2, 1 / _SQRT2]])


def rattlers(points, tol=1e-9) -> list:
    """Indices of circles free to move: a circle is held only when the directions to its contacts with held circles
    and sides leave no angular gap of pi or more; others are removed until none changes."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    pairs, walls = contacts(p, tol)
    free = np.zeros(len(p), dtype=bool)
    while True:
        directions = [[] for _ in range(len(p))]
        for i, j in pairs:
            if not free[i] and not free[j]:
                u = p[j] - p[i]
                directions[i].append(u)
                directions[j].append(-u)
        for i, s in walls:
            directions[i].append(_WALL_DIRECTIONS[s])
        changed = False
        for i in np.nonzero(~free)[0]:
            if len(directions[i]) < 3:
                held = False
            else:
                a = np.sort(np.arctan2(*np.array(directions[i])[:, ::-1].T))
                held = np.diff(np.append(a, a[0] + 2 * np.pi)).max() < np.pi - 1e-9
            if not held:
                free[i] = changed = True
        if not changed:
            return np.nonzero(free)[0].tolist()


def _lp_step(p, step):
    """One linear program on the near-active constraints: the move of at most step per coordinate that most increases
    the linearized radius. Returns the moved centres."""
    n = len(p)
    side, pair = slacks(p)
    r = min(side.min(), pair.min() / 2)
    near = 4 * step + 1e-12
    rows, cols, vals, rhs = [], [], [], []
    k = 0
    for i, s in np.argwhere(side - r <= near):
        if s == 2:
            rows += [k, k, k]
            cols += [2 * i, 2 * i + 1, 2 * n]
            vals += [1 / _SQRT2, 1 / _SQRT2, 1.0]
        else:
            rows += [k, k]
            cols += [2 * i + s, 2 * n]
            vals += [-1.0, 1.0]
        rhs.append(side[i, s] - r)
        k += 1
    for i, j in np.argwhere(np.triu(pair - 2 * r <= 2 * near, 1)):
        u = (p[i] - p[j]) / pair[i, j]
        rows += [k] * 5
        cols += [2 * i, 2 * i + 1, 2 * j, 2 * j + 1, 2 * n]
        vals += [-u[0], -u[1], u[0], u[1], 2.0]
        rhs.append(pair[i, j] - 2 * r)
        k += 1
    a = coo_matrix((vals, (rows, cols)), shape=(k, 2 * n + 1)).tocsr()
    c = np.zeros(2 * n + 1)
    c[-1] = -1.0
    bounds = [(-step, step)] * (2 * n) + [(None, step)]
    res = linprog(c, A_ub=a, b_ub=np.array(rhs), bounds=bounds, method="highs")
    if res.status != 0:
        return p
    return p + res.x[:-1].reshape(n, 2)


def maximize(points, rounds=40, step=None, deadline=None) -> np.ndarray:
    """Sequential linear programming on the active contacts: each round moves the centres by at most step per
    coordinate to raise the linearized radius, keeps the move and widens the step when the floating radius rises,
    and otherwise halves the step. Returns the best centres found."""
    import time
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2).copy()
    best = radius_fast(p)
    step = 0.01 * abs(best) if step is None else step
    top = step
    for _ in range(rounds):
        if deadline is not None and time.time() > deadline or step < 1e-17:
            break
        q = _lp_step(p, step)
        r = radius_fast(q)
        if r > best:
            p, best = q, r
            step = min(2 * step, top)
        else:
            step /= 4
    return p


def hex_lattice(n: int, angle=0.0, shift=(0.0, 0.0)) -> np.ndarray:
    """n centres of a hexagonal lattice in the triangle: rows at angle (radians) from the x axis, anchored at the
    corner (r, r) plus shift (in units of r), with the largest spacing 2r that fits n of them. Returns (n, 2)."""
    def fit(r):
        a = 2 * r * np.array([math.cos(angle), math.sin(angle)])
        b = 2 * r * np.array([math.cos(angle + math.pi / 3), math.sin(angle + math.pi / 3)])
        m = int(2 / r) + 2
        i, j = np.meshgrid(np.arange(-m, m + 1), np.arange(-m, m + 1))
        q = r * (1 + np.asarray(shift)) + i.reshape(-1, 1) * a + j.reshape(-1, 1) * b
        eps = 1e-12
        inside = (q[:, 0] >= r - eps) & (q[:, 1] >= r - eps) & (q[:, 0] + q[:, 1] <= 1 - _SQRT2 * r + eps)
        q = q[inside]
        return q[np.lexsort((q[:, 0], q[:, 1]))]
    lo, hi = 0.0, 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        if len(fit(mid)) >= n:
            lo = mid
        else:
            hi = mid
    return np.clip(fit(lo)[:n], 0.0, 1.0)
