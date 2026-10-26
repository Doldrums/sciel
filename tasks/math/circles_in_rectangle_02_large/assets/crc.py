"""Equal circles in the rectangle of width 1 and height H (H = ASPECT, the task's own): the radius of a set of
centres, its contacts, rattlers, a radius maximizer and hexagonal starts.

The rectangle is centred at the origin with its sides on x = -1/2, x = 1/2, y = -H/2 and y = H/2. The radius of N
centres is the largest r with every centre at distance >= r from the four sides and every pair at distance >= 2r:
r = min(1/2 - |x|, H/2 - |y|, min pair distance / 2), over all centres.
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

_data = asset("crc_data")
_best = asset("crc_best")
ASPECT = Fraction(_data.ASPECT)
H = float(ASPECT)
SIZES = tuple(_data.SIZES)
RADIUS = {n: float(_data.RADIUS[n]) for n in SIZES}
RADIUS_EXACT = {n: Fraction(_data.RADIUS[n]) for n in SIZES}
# The bar: the radius of the published packing improved further by local search (crc_best).
BEST_R2 = {n: Fraction(_best.R2[n]) for n in SIZES}
BEST = {n: math.sqrt(float(BEST_R2[n])) for n in SIZES}
LOWER = np.array([-0.5, -H / 2])
UPPER = np.array([0.5, H / 2])


def decode(text: str) -> np.ndarray:
    """Centres from base64 of little-endian binary64 x y pairs (a scored output's packings entry), as (n, 2)."""
    return np.frombuffer(base64.b64decode(text), dtype="<f8").reshape(-1, 2).astype(np.float64)


def published(n: int) -> np.ndarray:
    """The published best centres for n circles, rounded to binary64, as an (n, 2) array."""
    return decode(_data.COORDS[n])


def best(n: int) -> np.ndarray:
    """The best known centres for n circles, the bar the task scores against, as an (n, 2) array."""
    return decode(_best.COORDS[n])


def slacks(points) -> tuple:
    """Float distances: (side (n, 4) distances to x = -1/2, x = 1/2, y = -H/2, y = H/2; pair (n, n) centre
    distances, inf on the diagonal)."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    side = np.stack([p[:, 0] + 0.5, 0.5 - p[:, 0], p[:, 1] + H / 2, H / 2 - p[:, 1]], axis=1)
    pair = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(pair, np.inf)
    return side, pair


def radius_fast(points) -> float:
    """The radius in floating point, negative when a centre is outside the rectangle."""
    side, pair = slacks(points)
    return float(min(side.min(), pair.min() / 2))


def radius2(points) -> Fraction:
    """The squared radius exactly, from the binary64 centres as rationals and H as the exact decimal ASPECT, as
    scored. Raises ValueError when a centre is outside the closed rectangle or two centres coincide."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(p).all():
        raise ValueError("centres must be finite")
    ratios = [v.as_integer_ratio() for v in p.ravel().tolist()]
    d = max(q for _, q in ratios)
    z = [a * (d // q) for a, q in ratios]
    xs, ys = z[0::2], z[1::2]
    a, b = ASPECT.numerator, ASPECT.denominator
    # Side distances times 2bd are integers: b(d +- 2x) and ad +- 2by.
    side = min(min(b * (d + 2 * x), b * (d - 2 * x), a * d + 2 * b * y, a * d - 2 * b * y) for x, y in zip(xs, ys))
    if side < 0:
        raise ValueError("a centre is outside the rectangle")
    best = side * side
    # Floating squared distances are within a few units in the last place of the exact ones, so every pair that can
    # be the closest is within a relative 1e-9 of the floating minimum; those are compared exactly.
    sq = ((p[:, None, :] - p[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(sq, np.inf)
    i, j = np.nonzero(np.triu(sq <= sq.min() * (1 + 1e-9) + 1e-300, 1))
    for u, v in zip(i.tolist(), j.tolist()):
        best = min(best, b * b * ((xs[u] - xs[v]) ** 2 + (ys[u] - ys[v]) ** 2))
    if best == 0:
        raise ValueError("two centres coincide or a centre is on a side")
    return Fraction(best, 4 * b * b * d * d)


def radius(points) -> float:
    """The radius as scored: the square root of the exact squared radius, correctly rounded twice."""
    return math.sqrt(float(radius2(points)))


def contacts(points, tol=1e-9) -> tuple:
    """The contact graph: (pairs (k, 2) of i < j whose distance is within tol * r of 2r, walls (m, 2) of (i, side)
    with side 0 for x = -1/2, 1 for x = 1/2, 2 for y = -H/2, 3 for y = H/2, whose distance is within tol * r of r)."""
    side, pair = slacks(points)
    r = min(side.min(), pair.min() / 2)
    i, j = np.nonzero(np.triu(pair - 2 * r <= tol * r, 1))
    w = np.argwhere(side - r <= tol * r)
    return np.stack([i, j], axis=1), w


_WALL_DIRECTIONS = np.array([[-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])


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
        rows += [k, k]
        cols += [2 * i + s // 2, 2 * n]
        vals += [-1.0 if s % 2 == 0 else 1.0, 1.0]
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
    # The moves are solved in units of step: unscaled, moves far below the solver's 1e-7 tolerances stall it.
    bounds = [(-1.0, 1.0)] * (2 * n) + [(None, 1.0)]
    # A time limit keeps a rare slow solve from running past the stage; it then counts as no move.
    res = linprog(c, A_ub=a, b_ub=np.array(rhs) / step, bounds=bounds, method="highs", options={"time_limit": 10.0})
    if res.status != 0:
        return p
    return p + step * res.x[:-1].reshape(n, 2)


def maximize(points, rounds=40, step=None, deadline=None) -> np.ndarray:
    """Sequential linear programming on the active contacts: each round moves the centres by at most step per
    coordinate to raise the linearized radius, keeps the move and widens the step when the floating radius rises,
    and otherwise divides the step by 4. Returns the best centres found."""
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
    """n centres of a hexagonal lattice in the rectangle: rows at angle (radians) from the x axis (0 for rows along
    the long side, pi / 2 along the short side), anchored at the corner (-1/2 + r, -H/2 + r) plus shift (in units of
    r), with the largest spacing 2r that fits n of them. Returns (n, 2)."""
    def fit(r):
        a = 2 * r * np.array([math.cos(angle), math.sin(angle)])
        b = 2 * r * np.array([math.cos(angle + math.pi / 3), math.sin(angle + math.pi / 3)])
        m = int(1.5 / r) + 2
        i, j = np.meshgrid(np.arange(-m, m + 1), np.arange(-m, m + 1))
        q = LOWER + r * (1 + np.asarray(shift)) + i.reshape(-1, 1) * a + j.reshape(-1, 1) * b
        eps = 1e-12
        inside = (np.abs(q[:, 0]) <= 0.5 - r + eps) & (np.abs(q[:, 1]) <= H / 2 - r + eps)
        q = q[inside]
        return q[np.lexsort((q[:, 0], q[:, 1]))]
    lo, hi = 0.0, H / 2
    for _ in range(60):
        mid = (lo + hi) / 2
        if len(fit(mid)) >= n:
            lo = mid
        else:
            hi = mid
    return np.clip(fit(lo)[:n], LOWER, UPPER)
