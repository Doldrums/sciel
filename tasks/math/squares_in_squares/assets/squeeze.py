"""Squeeze and repair packings of unit squares by sequential linear programming on the close pairs and the box.

Each step linearizes, around the current packing, the separation of every close pair along its best separating axis
and the four sides of the bounding box, and solves one LP over moves of every centre and angle within a trust region:
minimize L + weight * (sum of pair violations below the gap). A step is kept when the true side plus weight times
the true violations below half the gap falls; then the region grows, otherwise it shrinks. Started from an overlapping
packing the same LP pushes the squares apart, so one call both repairs and compacts. The half-width of a square along
an axis is (|cos a| + |sin a|) / 2 of the relative angle a, concave between kinks at a = 0 mod pi/2; within twice
the trust region of a kink it is linearized with both one-sided slopes, so the model never underestimates it.
"""

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

import time

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

from sciel.task import asset

sis = asset("sis")
_KINK = 1e-7


def _slopes(alpha, reach):
    """Derivative of (|cos a| + |sin a|) / 2, and whether a kink (a multiple of pi/2) is within reach of a, where
    the model takes both one-sided slopes."""
    c, s = np.cos(alpha), np.sin(alpha)
    kink = np.minimum(np.abs(c), np.abs(s)) < max(_KINK, reach)
    return 0.5 * (-np.sign(c) * s + np.sign(s) * c), kink


def _rows(p, i, j, gap, reach):
    """Linear model of the best axis gap of each pair: (columns (m, 6), coefficients (m, 6), right-hand sides)."""
    n = len(p)
    g4 = sis.axis_gaps(p, i, j)
    k = g4.argmax(1)
    owner = np.where(k < 2, i, j)
    other = np.where(k < 2, j, i)
    phi = p[owner, 2] + np.where(k % 2 == 1, np.pi / 2, 0.0)
    a = np.stack([np.cos(phi), np.sin(phi)], 1)
    perp = np.stack([-np.sin(phi), np.cos(phi)], 1)
    d = p[j, :2] - p[i, :2]
    proj = (a * d).sum(1)
    sigma = np.where(proj >= 0, 1.0, -1.0)
    value = sigma * proj - 0.5 - sis.half_width(p[other, 2] - phi)
    slope, kink = _slopes(p[other, 2] - phi, reach)
    cols, coefs, rhs = [], [], []
    for variant in (0, 1, 2):
        # variant 0: the smooth slope; at a kink, variants 1 and 2 carry the two one-sided slopes instead.
        use = ~kink if variant == 0 else kink
        if not use.any():
            continue
        h = slope[use] if variant == 0 else np.full(use.sum(), 0.5 if variant == 1 else -0.5)
        s, aa, pp, dd = sigma[use], a[use], perp[use], d[use]
        dg = np.stack([-s * aa[:, 0], -s * aa[:, 1], s * aa[:, 0], s * aa[:, 1],
                       s * (pp * dd).sum(1) + h, -h], 1)
        cols.append(np.stack([i[use], n + i[use], j[use], n + j[use], 2 * n + owner[use], 2 * n + other[use]], 1))
        coefs.append(-dg)
        rhs.append(value[use] - gap)
    if not cols:
        return np.zeros((0, 6), int), np.zeros((0, 6)), np.zeros(0)
    return np.vstack(cols), np.vstack(coefs), np.concatenate(rhs)


def _box(p, reach):
    """Rows x + w <= X0 + L, X0 <= x - w and the same in y, w the half extent (|cos t| + |sin t|) / 2."""
    n = len(p)
    idx = np.arange(n)
    w = sis.half_width(p[:, 2])
    slope, kink = _slopes(p[:, 2], reach)
    rows, cols, vals, rhs = [], [], [], []
    count = 0
    for variant in (0, 1, 2):
        use = ~kink if variant == 0 else kink
        if not use.any():
            continue
        h = slope[use] if variant == 0 else np.full(use.sum(), 0.5 if variant == 1 else -0.5)
        u, m = idx[use], use.sum()
        for axis, origin in ((0, 3 * n), (1, 3 * n + 1)):
            centre = p[use, axis]
            # high side: centre + d + w + h dt - origin - L <= 0
            r = count + np.arange(m)
            rows += [r, r, r, r]
            cols += [axis * n + u, 2 * n + u, np.full(m, origin), np.full(m, 3 * n + 2)]
            vals += [np.ones(m), h, -np.ones(m), -np.ones(m)]
            rhs.append(-(centre + w[use]))
            count += m
            # low side: origin - centre - d + w + h dt <= 0
            r = count + np.arange(m)
            rows += [r, r, r]
            cols += [axis * n + u, 2 * n + u, np.full(m, origin)]
            vals += [-np.ones(m), h, np.ones(m)]
            rhs.append(centre - w[use])
            count += m
    return np.concatenate(rows), np.concatenate(cols), np.concatenate(vals), np.concatenate(rhs), count


def violation(p, gap: float) -> float:
    """Sum over close pairs of the shortfall of their separation below half the gap."""
    _, _, g = sis.gaps(p)
    return float(np.maximum(0.0, 0.5 * gap - g).sum()) if len(g) else 0.0


def squeeze(p, rounds: int = 200, step: float = 0.05, gap: float = 1e-12, weight: float = 1e3,
            deadline: float | None = None, min_step: float = 1e-10, free=None) -> np.ndarray:
    """Sequential LP on the side and the pair separations; returns the packing of lowest side + weight * violation
    found; an overlapping start is pushed apart first, and once every pair is at least gap / 2 apart only steps that
keep them so are taken. step is the initial trust region for centres and angles; with free (indices) only those
squares move, which is much faster on large n."""
    p = np.array(p, dtype=np.float64)
    n = len(p)
    moving = np.ones(n, bool) if free is None else np.isin(np.arange(n), free)
    broken = violation(p, gap)
    merit = sis.side(p) + weight * broken
    for _ in range(rounds):
        if step < min_step or (deadline is not None and time.time() > deadline):
            break
        near = max(0.1, 3 * step)
        i, j, g = sis.gaps(p, *sis.pairs(p, near))
        close = (g < near) & (moving[i] | moving[j])
        # The pairs aim at gap plus a second-order margin, so a step the model accepts stays apart in truth.
        pc, pv, pr = _rows(p, i[close], j[close], gap + 2 * step * step, 2 * step)
        br, bc, bv, brhs, nb = _box(p, 2 * step)
        m = len(pr)
        nvar = 3 * n + 3 + m
        # pair rows: coefficients on the six moves, minus the row's own slack
        prow = np.repeat(np.arange(m), 6) + nb
        rows = np.concatenate([br, prow, nb + np.arange(m)])
        cols = np.concatenate([bc, pc.ravel(), 3 * n + 3 + np.arange(m)])
        vals = np.concatenate([bv, pv.ravel(), -np.ones(m)])
        a = coo_matrix((vals, (rows, cols)), shape=(nb + m, nvar)).tocsr()
        c = np.zeros(nvar)
        c[3 * n + 2] = 1.0
        c[3 * n + 3:] = weight
        bounds = np.zeros((nvar, 2))
        bounds[:3 * n] = (-step, step)
        bounds[:3 * n][~np.tile(moving, 3)] = 0.0
        bounds[3 * n:3 * n + 3] = (-np.inf, np.inf)
        bounds[3 * n + 3:] = (0.0, np.inf)
        res = linprog(c, A_ub=a, b_ub=np.concatenate([brhs, pr]), bounds=bounds, method="highs")
        if res.status != 0:
            step *= 0.5
            continue
        q = p.copy()
        q[:, 0] += res.x[:n]
        q[:, 1] += res.x[n:2 * n]
        q[:, 2] += res.x[2 * n:3 * n]
        apart = violation(q, gap)
        value = sis.side(q) + weight * apart
        # Once apart, a step must keep every pair apart; before, it must lower the merit.
        if value < merit - 1e-15 and (apart == 0 or broken > 0):
            p, merit, broken, step = q, value, apart, min(step * 1.5, 0.25)
        else:
            step *= 0.5
    return p


def group(p, centre: int, k: int) -> np.ndarray:
    """Indices of the k squares whose centres are nearest square centre's (itself included)."""
    p = np.asarray(p)
    return np.argsort(((p[:, :2] - p[centre, :2]) ** 2).sum(1))[:k]


def jiggle(p, idx, scale: float, rng, turn: float | None = None) -> np.ndarray:
    """Move squares idx by normal noise of the given scale and turn them by normal noise of turn radians."""
    p = np.array(p, dtype=np.float64)
    idx = np.asarray(idx)
    p[idx, :2] += rng.normal(scale=scale, size=(len(idx), 2))
    p[idx, 2] += rng.normal(scale=scale if turn is None else turn, size=len(idx))
    return p


def boundary(p, tol: float = 1e-6) -> np.ndarray:
    """Indices of the squares touching the bounding box within tol."""
    q = sis.corners(p)
    lo, hi = q.reshape(-1, 2).min(0), q.reshape(-1, 2).max(0)
    s = sis.side(p)
    hit = np.zeros(len(q), bool)
    for axis in (0, 1):
        if hi[axis] - lo[axis] >= s - tol:
            hit |= (q[:, :, axis].min(1) <= lo[axis] + tol) | (q[:, :, axis].max(1) >= hi[axis] - tol)
    return np.flatnonzero(hit)
