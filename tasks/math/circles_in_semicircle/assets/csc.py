"""Equal circles in the semicircle {y >= 0, x^2 + y^2 <= 1}: the radius of a set of centres, exactly and in floating
point, its contacts, rattlers, a radius maximizer and hexagonal starts.

The radius of N centres is the largest r with every circle inside the semicircle and no two overlapping: every centre
at distance >= r from the diameter y = 0 and >= r inside the arc (sqrt(x^2 + y^2) <= 1 - r), and every pair at
distance >= 2r: r = min(y, 1 - sqrt(x^2 + y^2), min pair distance / 2), over all centres.
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


def _sign(x):
    return (x > 0) - (x < 0)


def _pair(b, u, c, v):
    """The sign of b sqrt(u) + c sqrt(v) for rationals b, c and u, v >= 0, exactly."""
    s, t = _sign(b) * (u > 0), _sign(c) * (v > 0)
    if s == t or t == 0:
        return s
    if s == 0:
        return t
    d = b * b * u - c * c * v
    return s if d > 0 else t if d < 0 else 0


def _sign3(a, b, u, c, v):
    """The sign of a + b sqrt(u) + c sqrt(v), exactly: the sign of the roots, then, when a has the other sign, the
    sign of a^2 - (b sqrt(u) + c sqrt(v))^2 = a^2 - b^2 u - c^2 v - 2bc sqrt(uv)."""
    roots, sa = _pair(b, u, c, v), _sign(a)
    if roots == 0 or sa == 0 or sa == roots:
        return sa or roots
    d = _pair(a * a - b * b * u - c * c * v, 1, -2 * b * c, u * v)
    return sa if d > 0 else roots if d < 0 else 0


class Surd:
    """The exact number a + b sqrt(c) for rationals a, b and c >= 0: every radius is one, since it is a side distance
    y, half a pair distance sqrt(s) or an arc distance 1 - sqrt(q). Compared exactly; float() rounds it."""

    __slots__ = ("a", "b", "c")

    def __init__(self, a, b=0, c=0):
        self.a, self.b, self.c = Fraction(a), Fraction(b), Fraction(c)

    def _cmp(self, other):
        other = other if isinstance(other, Surd) else Surd(other)
        return _sign3(self.a - other.a, self.b, self.c, -other.b, other.c)

    def __lt__(self, other):
        return self._cmp(other) < 0

    def __le__(self, other):
        return self._cmp(other) <= 0

    def __gt__(self, other):
        return self._cmp(other) > 0

    def __ge__(self, other):
        return self._cmp(other) >= 0

    def __eq__(self, other):
        return self._cmp(other) == 0

    def __hash__(self):
        return hash((self.a, self.b, self.c))

    def __mul__(self, k):
        k = Fraction(k)
        return Surd(self.a * k, self.b * k, self.c)

    __rmul__ = __mul__

    def __float__(self):
        return float(self.a) + float(self.b) * math.sqrt(float(self.c))

    def __repr__(self):
        return f"Surd({str(self.a)!r}, {str(self.b)!r}, {str(self.c)!r})"


_data = asset("csc_data")
_best = asset("csc_best")
SIZES = tuple(_data.SIZES)
RADIUS = {n: float(_data.RADIUS[n]) for n in SIZES}
RADIUS_EXACT = {n: Fraction(_data.RADIUS[n]) for n in SIZES}
# The bar: the exact radius of the published centres as binary64 (csc_best), as a + b sqrt(c).
BEST_EXACT = {n: Surd(*_best.R[n]) for n in SIZES}
BEST = {n: float(BEST_EXACT[n]) for n in SIZES}


def decode(text: str) -> np.ndarray:
    """Centres from base64 of little-endian binary64 x y pairs (a scored output's packings entry), as (n, 2)."""
    return np.frombuffer(base64.b64decode(text), dtype="<f8").reshape(-1, 2).astype(np.float64)


def published(n: int) -> np.ndarray:
    """The published best centres for n circles, rounded to binary64, as an (n, 2) array."""
    return decode(_data.COORDS[n])


def best(n: int) -> np.ndarray:
    """The best known centres for n circles, the bar the task scores against, as an (n, 2) array."""
    return decode(_best.COORDS[n])


def clip(points) -> np.ndarray:
    """The centres put back into the closed semicircle: negative y to 0, centres past the arc onto it."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2).copy()
    p[:, 1] = np.maximum(p[:, 1], 0.0)
    norm = np.sqrt((p ** 2).sum(axis=1))
    out = norm > 1
    p[out] /= norm[out, None]
    return p


def slacks(points) -> tuple:
    """Float distances: (side (n, 2) distances to y = 0 and inside the arc, 1 - sqrt(x^2 + y^2); pair (n, n) centre
    distances, inf on the diagonal)."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    side = np.stack([p[:, 1], 1 - np.sqrt((p ** 2).sum(axis=1))], axis=1)
    pair = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(pair, np.inf)
    return side, pair


def radius_fast(points) -> float:
    """The radius in floating point, negative when a centre is outside the semicircle."""
    side, pair = slacks(points)
    return float(min(side.min(), pair.min() / 2))


def radius_exact(points) -> Surd:
    """The radius exactly, from the binary64 centres as rationals, as scored: a Surd a + b sqrt(c) that compares
    exactly. Raises ValueError when a centre is outside the closed semicircle or the radius is zero."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(p).all():
        raise ValueError("centres must be finite")
    ratios = [v.as_integer_ratio() for v in p.ravel().tolist()]
    d = max(q for _, q in ratios)
    z = [a * (d // q) for a, q in ratios]
    xs, ys = z[0::2], z[1::2]
    # The centres times d are integers, so y and x^2 + y^2 are exact; the arc distance is 1 - sqrt(max q) / d.
    side = min(ys)
    q = max(x * x + y * y for x, y in zip(xs, ys))
    if side < 0 or q > d * d:
        raise ValueError("a centre is outside the semicircle")
    best = min(Surd(Fraction(side, d)), Surd(1, -1, Fraction(q, d * d)))
    # Floating squared distances are within a few units in the last place of the exact ones, so every pair that can
    # be the closest is within a relative 1e-9 of the floating minimum; those are compared exactly.
    sq = ((p[:, None, :] - p[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(sq, np.inf)
    i, j = np.nonzero(np.triu(sq <= sq.min() * (1 + 1e-9) + 1e-300, 1))
    pair = min((xs[u] - xs[v]) ** 2 + (ys[u] - ys[v]) ** 2 for u, v in zip(i.tolist(), j.tolist()))
    best = min(best, Surd(0, 1, Fraction(pair, 4 * d * d)))
    if best <= 0:
        raise ValueError("two centres coincide or a centre is on the boundary")
    return best


def radius(points) -> float:
    """The radius as scored: the exact radius rounded to binary64."""
    return float(radius_exact(points))


def contacts(points, tol=1e-9) -> tuple:
    """The contact graph: (pairs (k, 2) of i < j whose distance is within tol * r of 2r, walls (m, 2) of (i, side)
    with side 0 for y = 0, 1 for the arc, whose distance is within tol * r of r)."""
    side, pair = slacks(points)
    r = min(side.min(), pair.min() / 2)
    i, j = np.nonzero(np.triu(pair - 2 * r <= tol * r, 1))
    w = np.argwhere(side - r <= tol * r)
    return np.stack([i, j], axis=1), w


def rattlers(points, tol=1e-9) -> list:
    """Indices of circles free to move: a circle is held only when the directions to its contacts with held circles
    and the boundary leave no angular gap of pi or more; others are removed until none changes."""
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
            directions[i].append(np.array([0.0, -1.0]) if s == 0 else p[i])
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
    norm = 1 - side[:, 1]
    rows, cols, vals, rhs = [], [], [], []
    k = 0
    for i, s in np.argwhere(side - r <= near):
        if s == 0:
            rows += [k, k]
            cols += [2 * i + 1, 2 * n]
            vals += [-1.0, 1.0]
        else:
            # The arc distance 1 - |c| falls by u . move to first order, u the unit vector to the centre.
            u = p[i] / norm[i]
            rows += [k] * 3
            cols += [2 * i, 2 * i + 1, 2 * n]
            vals += [u[0], u[1], 1.0]
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
    """n centres of a hexagonal lattice in the semicircle: rows at angle (radians) from the x axis, anchored at
    (0, r) on the diameter plus shift (in units of r), with the largest spacing 2r that fits n of them. Returns
    (n, 2)."""
    def fit(r):
        a = 2 * r * np.array([math.cos(angle), math.sin(angle)])
        b = 2 * r * np.array([math.cos(angle + math.pi / 3), math.sin(angle + math.pi / 3)])
        m = int(1.5 / r) + 2
        i, j = np.meshgrid(np.arange(-m, m + 1), np.arange(-m, m + 1))
        q = r * (np.array([0.0, 1.0]) + np.asarray(shift)) + i.reshape(-1, 1) * a + j.reshape(-1, 1) * b
        eps = 1e-12
        inside = (q[:, 1] >= r - eps) & (np.sqrt((q ** 2).sum(axis=1)) <= 1 - r + eps)
        q = q[inside]
        return q[np.lexsort((q[:, 0], q[:, 1]))]
    lo, hi = 0.0, 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        if len(fit(mid)) >= n:
            lo = mid
        else:
            hi = mid
    return clip(fit(lo)[:n])
