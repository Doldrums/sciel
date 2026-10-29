"""Equal circles in the regular pentagon with circumradius 1: the radius of a set of centres, exactly and in floating
point, its contacts, rattlers, a radius maximizer and hexagonal starts.

The pentagon has its centre at the origin and a flat bottom edge: its sides have outer unit normals n_k at angles
-pi/2 + 2 pi k / 5, k = 0..4, and lie at distance h = cos(pi/5) = (1 + sqrt 5) / 4 from the origin. The radius of N
centres is the largest r with every centre at distance >= r inside every side (h - n_k . c >= r) and every pair at
distance >= 2r: r = min(h - n_k . c, min pair distance / 2), over all centres and sides.
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


def _sign5(a, b):
    """The sign of a + b sqrt 5 for rationals a, b, exactly."""
    sa, sb = _sign(a), _sign(b)
    if sb == 0:
        return sa
    if sa == 0 or sa == sb:
        return sb
    return sa if a * a > 5 * b * b else sb


# Fractions within 1e-40 of sqrt 5 and of t = sqrt(10 + 2 sqrt 5), for rounding exact numbers to binary64.
_S5 = Fraction(math.isqrt(5 * 10 ** 80), 10 ** 40)
_T = Fraction(math.isqrt(int((10 + 2 * _S5) * 10 ** 80)), 10 ** 40)


class Exact:
    """The exact number a + b sqrt 5 + (c + d sqrt 5) t for rationals a, b, c, d, with t = sqrt(10 + 2 sqrt 5) =
    4 cos(pi/10). The sides' normals and distance lie in this field (sin(pi/10) = (sqrt 5 - 1) / 4, cos(3 pi/10) =
    (sqrt 5 - 1) t / 8), so every side distance of rational centres is one, and so is its square; a squared pair
    distance is rational. Compared exactly; float() rounds it."""

    __slots__ = ("a", "b", "c", "d")

    def __init__(self, a, b=0, c=0, d=0):
        self.a, self.b, self.c, self.d = Fraction(a), Fraction(b), Fraction(c), Fraction(d)

    def __add__(self, o):
        o = o if isinstance(o, Exact) else Exact(o)
        return Exact(self.a + o.a, self.b + o.b, self.c + o.c, self.d + o.d)

    __radd__ = __add__

    def __neg__(self):
        return Exact(-self.a, -self.b, -self.c, -self.d)

    def __sub__(self, o):
        return self + -(o if isinstance(o, Exact) else Exact(o))

    def __rsub__(self, o):
        return -self + o

    def __mul__(self, o):
        o = o if isinstance(o, Exact) else Exact(o)
        # (u + v t)(w + z t) = u w + v z t^2 + (u z + v w) t, with t^2 = 10 + 2 sqrt 5 and products in Q(sqrt 5).
        m = lambda a, b, c, d: (a * c + 5 * b * d, a * d + b * c)
        uw, vz = m(self.a, self.b, o.a, o.b), m(self.c, self.d, o.c, o.d)
        uz, vw = m(self.a, self.b, o.c, o.d), m(self.c, self.d, o.a, o.b)
        vzt = m(vz[0], vz[1], 10, 2)
        return Exact(uw[0] + vzt[0], uw[1] + vzt[1], uz[0] + vw[0], uz[1] + vw[1])

    __rmul__ = __mul__

    def sign(self):
        """-1, 0 or 1, exactly: the signs of u = a + b sqrt 5 and v = c + d sqrt 5, then, when they differ, the sign
        of u^2 - v^2 t^2."""
        su, sv = _sign5(self.a, self.b), _sign5(self.c, self.d)
        if sv == 0:
            return su
        if su == 0 or su == sv:
            return sv
        p, q = self.c * self.c + 5 * self.d * self.d, 2 * self.c * self.d
        w = _sign5(self.a * self.a + 5 * self.b * self.b - 10 * p - 10 * q, 2 * self.a * self.b - 2 * p - 10 * q)
        return su if w > 0 else sv

    def _cmp(self, other):
        return (self - other).sign()

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
        return hash((self.a, self.b, self.c, self.d))

    def __float__(self):
        return float(self.a + self.b * _S5 + (self.c + self.d * _S5) * _T)

    def __repr__(self):
        return f"Exact({str(self.a)!r}, {str(self.b)!r}, {str(self.c)!r}, {str(self.d)!r})"


# The sides: distance h from the origin and outer unit normals n_k = (cos, sin)(-pi/2 + 2 pi k / 5), exactly and
# in floating point.
H_EXACT = Exact(Fraction(1, 4), Fraction(1, 4))
NORMALS_EXACT = (
    (Exact(0), Exact(-1)),
    (Exact(0, 0, Fraction(1, 4)), Exact(Fraction(1, 4), Fraction(-1, 4))),
    (Exact(0, 0, Fraction(-1, 8), Fraction(1, 8)), Exact(Fraction(1, 4), Fraction(1, 4))),
    (Exact(0, 0, Fraction(1, 8), Fraction(-1, 8)), Exact(Fraction(1, 4), Fraction(1, 4))),
    (Exact(0, 0, Fraction(-1, 4)), Exact(Fraction(1, 4), Fraction(-1, 4))),
)
H = math.cos(math.pi / 5)
NORMALS = np.array([[math.cos(a), math.sin(a)] for a in -math.pi / 2 + 2 * math.pi * np.arange(5) / 5])

_data = asset("cpt_data")
_best = asset("cpt_best")
SIZES = tuple(_data.SIZES)
RADIUS = {n: float(_data.RADIUS[n]) for n in SIZES}
RADIUS_EXACT = {n: Fraction(_data.RADIUS[n]) for n in SIZES}
# The bar: the exact squared radius of the published centres as binary64 (cpt_best).
BEST_R2 = {n: Exact(*_best.R2[n]) for n in SIZES}
BEST = {n: math.sqrt(float(BEST_R2[n])) for n in SIZES}


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
    """The centres put back into the closed pentagon: each centre past a side is projected onto it, side by side,
    until none is past one (a centre past a corner lands on the corner)."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2).copy()
    for _ in range(100):
        excess = p @ NORMALS.T - H
        if excess.max() <= 0:
            break
        for k in range(5):
            p -= np.maximum(p @ NORMALS[k] - H, 0.0)[:, None] * NORMALS[k]
    return p


def slacks(points) -> tuple:
    """Float distances: (side (n, 5) distances inside the five sides, h - n_k . c; pair (n, n) centre distances, inf
    on the diagonal)."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    side = H - p @ NORMALS.T
    pair = np.sqrt(((p[:, None, :] - p[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(pair, np.inf)
    return side, pair


def radius_fast(points) -> float:
    """The radius in floating point, negative when a centre is outside the pentagon."""
    side, pair = slacks(points)
    return float(min(side.min(), pair.min() / 2))


def radius2(points) -> Exact:
    """The squared radius exactly, from the binary64 centres as rationals, as scored: an Exact that compares
    exactly. Raises ValueError when a centre is outside the closed pentagon or the radius is zero."""
    p = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(p).all():
        raise ValueError("centres must be finite")
    ratios = [v.as_integer_ratio() for v in p.ravel().tolist()]
    d = max(q for _, q in ratios)
    z = [a * (d // q) for a, q in ratios]
    xs, ys = z[0::2], z[1::2]
    # A floating side distance is within a few 1e-16 of the exact one, so the nearest sides are among those within
    # 1e-12 of the floating minimum; those are computed exactly in the field of Exact.
    side = H - p @ NORMALS.T
    near = np.argwhere(side <= side.min() + 1e-12)
    wall = min(H_EXACT - nx * Fraction(xs[i], d) - ny * Fraction(ys[i], d)
               for i, (nx, ny) in ((i, NORMALS_EXACT[k]) for i, k in near.tolist()))
    if wall.sign() < 0:
        raise ValueError("a centre is outside the pentagon")
    # Floating squared distances are within a few units in the last place of the exact ones, so every pair that can
    # be the closest is within a relative 1e-9 of the floating minimum; those are compared exactly.
    sq = ((p[:, None, :] - p[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(sq, np.inf)
    i, j = np.nonzero(np.triu(sq <= sq.min() * (1 + 1e-9) + 1e-300, 1))
    pair = min((xs[u] - xs[v]) ** 2 + (ys[u] - ys[v]) ** 2 for u, v in zip(i.tolist(), j.tolist()))
    if wall.sign() == 0 or pair == 0:
        raise ValueError("two centres coincide or a centre is on a side")
    return min(wall * wall, Exact(Fraction(pair, 4 * d * d)))


def radius(points) -> float:
    """The radius as scored: the square root of the exact squared radius rounded to binary64."""
    return math.sqrt(float(radius2(points)))


def contacts(points, tol=1e-9) -> tuple:
    """The contact graph: (pairs (k, 2) of i < j whose distance is within tol * r of 2r, walls (m, 2) of (i, side)
    with side k the side of normal n_k, whose distance is within tol * r of r)."""
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
            directions[i].append(NORMALS[s])
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
        # The distance h - n_s . c to side s falls by n_s . move.
        rows += [k] * 3
        cols += [2 * i, 2 * i + 1, 2 * n]
        vals += [NORMALS[s, 0], NORMALS[s, 1], 1.0]
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
    """n centres of a hexagonal lattice in the pentagon: rows at angle (radians) from the x axis, anchored at
    (0, r - h) on the bottom side plus shift (in units of r), with the largest spacing 2r that fits n of them. Returns
    (n, 2)."""
    def fit(r):
        a = 2 * r * np.array([math.cos(angle), math.sin(angle)])
        b = 2 * r * np.array([math.cos(angle + math.pi / 3), math.sin(angle + math.pi / 3)])
        m = int(1.5 / r) + 2
        i, j = np.meshgrid(np.arange(-m, m + 1), np.arange(-m, m + 1))
        q = np.array([0.0, r - H]) + r * np.asarray(shift) + i.reshape(-1, 1) * a + j.reshape(-1, 1) * b
        eps = 1e-12
        inside = (H - q @ NORMALS.T >= r - eps).all(axis=1)
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
