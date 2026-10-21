"""Exact tools for no-isosceles subsets of the 64 x 64 grid {0..63}^2.

A set is valid when no three of its points form a possibly flat isosceles triangle, that is, when from every
point the squared distances to all the others are distinct. BEST is AlphaEvolve's 112-point set (arXiv 2511.02864,
section 6.39, December 2025), the best known.
"""

import numpy as np

N = 64

BEST = [(0, 1), (0, 25), (0, 38), (0, 62), (1, 2), (1, 3), (1, 23), (1, 30), (1, 33), (1, 40), (1, 60), (1, 61),
        (2, 11), (2, 27), (2, 36), (2, 52), (5, 1), (5, 7), (5, 17), (5, 25), (5, 38), (5, 46), (5, 56), (5, 62),
        (6, 1), (6, 17), (6, 46), (6, 62), (7, 2), (7, 61), (8, 2), (8, 61), (10, 0), (10, 63), (11, 22), (11, 26),
        (11, 37), (11, 41), (14, 5), (14, 7), (14, 56), (14, 58), (16, 0), (16, 63), (17, 0), (17, 63), (18, 2),
        (18, 61), (24, 5), (24, 11), (24, 52), (24, 58), (25, 1), (25, 7), (25, 56), (25, 62), (38, 1), (38, 7),
        (38, 56), (38, 62), (39, 5), (39, 11), (39, 52), (39, 58), (45, 2), (45, 61), (46, 0), (46, 63), (47, 0),
        (47, 63), (49, 5), (49, 7), (49, 56), (49, 58), (52, 22), (52, 26), (52, 37), (52, 41), (53, 0), (53, 63),
        (55, 2), (55, 61), (56, 2), (56, 61), (57, 1), (57, 17), (57, 46), (57, 62), (58, 1), (58, 7), (58, 17),
        (58, 25), (58, 38), (58, 46), (58, 56), (58, 62), (59, 26), (59, 37), (61, 11), (61, 27), (61, 36),
        (61, 52), (62, 3), (62, 23), (62, 30), (62, 33), (62, 40), (62, 60), (63, 1), (63, 25), (63, 38), (63, 62)]

_GRID = np.array([(x, y) for x in range(N) for y in range(N)], dtype=np.int64)


def as_array(points) -> np.ndarray:
    return np.asarray(sorted({(int(x), int(y)) for x, y in points}), dtype=np.int64).reshape(-1, 2)


def squared(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Squared distances between every row of a and every row of b, exact in int64."""
    d = a[:, None, :] - b[None, :, :]
    return (d * d).sum(axis=2)


def conflicts(points) -> np.ndarray:
    """For each point, the number of isosceles triangles with their apex there (pairs at equal distance)."""
    p = as_array(points)
    d = squared(p, p)
    out = np.zeros(len(p), dtype=np.int64)
    for i in range(len(p)):
        _, counts = np.unique(np.delete(d[i], i), return_counts=True)
        out[i] = int((counts * (counts - 1) // 2).sum())
    return out


def is_valid(points) -> bool:
    p = as_array(points)
    return len(p) == len(points) and bool((p >= 0).all() and (p < N).all()) and int(conflicts(p).sum()) == 0


def addable(points) -> np.ndarray:
    """Grid points that keep a valid set valid when added alone: no new apex at the point, and at no old point
    does its distance repeat an existing one."""
    p = as_array(points)
    if len(p) == 0:
        return _GRID.copy()
    taken = np.zeros((N, N), dtype=bool)
    taken[p[:, 0], p[:, 1]] = True
    g = _GRID[~taken[_GRID[:, 0], _GRID[:, 1]]]
    dg = squared(g, p)
    s = np.sort(dg, axis=1)
    ok = (np.diff(s, axis=1) != 0).all(axis=1)
    dp = squared(p, p)
    for j in range(len(p)):
        ok &= ~np.isin(dg[:, j], dp[j])
    return g[ok]


def repair(points, rng=None) -> list:
    """A valid subset: drop the point in most isosceles triangles until none is left."""
    rng = np.random.default_rng(0) if rng is None else rng
    p = [tuple(map(int, q)) for q in as_array(points) if 0 <= q[0] < N and 0 <= q[1] < N]
    while True:
        c = conflicts(p)
        if len(c) == 0 or c.max() == 0:
            return p
        worst = np.flatnonzero(c == c.max())
        p.pop(int(rng.choice(worst)))


def grow(points, rng=None) -> list:
    """Add addable points in random order until none is left; the set stays valid."""
    rng = np.random.default_rng(0) if rng is None else rng
    p = [tuple(map(int, q)) for q in as_array(points)]
    while True:
        a = addable(p)
        if len(a) == 0:
            return p
        p.append(tuple(map(int, a[rng.integers(len(a))])))
