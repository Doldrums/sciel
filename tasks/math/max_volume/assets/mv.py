"""Maximal volume polyhedra: n points on the unit sphere S^2 whose convex hull has the largest volume. The volume
exactly as scored, a floating volume with its gradient, a projected ascent, shakes and basin hopping, symmetric
orbits and neighbour-n constructions.

A configuration is an (n, 3) array of nonzero points; only their directions count, so points need not be
normalized. The scored polytope is the hull of the directions rounded inward (directions): each is a binary64 vector
of length at most 1 exactly, so it lies in the closed unit ball. A hull volume is convex in each vertex, so no n
points in the ball beat the best n points on the sphere, and the rounding costs about 1e-16 of the volume.
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
from scipy.optimize import minimize
from scipy.spatial import ConvexHull, QhullError

from sciel.task import asset

try:
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
except ImportError:
    pass

_data = asset("mv_data")
_best = asset("mv_best")
# Instances are keyed by n as a string, "4" to "130".
INSTANCES = tuple(_data.INSTANCES)
SIZE = {k: int(k) for k in INSTANCES}
# The table's volume as displayed (12 decimals), and the bar: the exact volume of best(key) and its float.
PUBLISHED = {k: Fraction(_data.VOLUME[k]) for k in INSTANCES}
BEST_VOLUME = {k: Fraction(_best.VOLUME[k]) for k in INSTANCES}
BEST = {k: float(BEST_VOLUME[k]) for k in INSTANCES}


def decode(text: str) -> np.ndarray:
    """Points from base64 of little-endian binary64 rows of 3 coordinates, as (n, 3)."""
    return np.frombuffer(base64.b64decode(text), dtype="<f8").reshape(-1, 3).astype(np.float64)


def encode(points) -> str:
    return base64.b64encode(np.asarray(points, dtype="<f8").tobytes()).decode()


def published(key: str) -> np.ndarray:
    """The table's coordinates for the instance (12 decimals), as (n, 3)."""
    return decode(_data.COORDS[key])


def best(key: str) -> np.ndarray:
    """The bar's points for the instance (the table's configuration after local search), as (n, 3)."""
    return decode(_best.COORDS[key])


def _norm2(row) -> Fraction:
    return sum(Fraction(v) ** 2 for v in row)


def directions(points) -> np.ndarray:
    """The directions of the (n, 3) points as scored: unit vectors in floating point, each row shrunk by a factor
    1 - 2^-52 until its exact squared length is at most 1. Raises ValueError on a non-finite or zero point."""
    x = np.asarray(points, dtype=np.float64)
    if x.ndim != 2 or x.shape[1] != 3 or not np.isfinite(x).all():
        raise ValueError("points must be a finite (n, 3) array")
    top = np.abs(x).max(axis=1, keepdims=True)
    if (top == 0).any():
        raise ValueError("a point is the zero vector")
    u = x / top
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    for i in np.nonzero((u * u).sum(axis=1) > 1 - 1e-15)[0]:
        while _norm2(u[i].tolist()) > 1:
            u[i] *= 1 - 2.0 ** -52
    return u


def _integers(u):
    """The rows of u as integer vectors over one power of two 2^s: (list of int triples, s)."""
    ratios = [[v.as_integer_ratio() for v in row] for row in u.tolist()]
    q = max(b for row in ratios for _, b in row)
    return [tuple(a * (q // b) for a, b in row) for row in ratios], q.bit_length() - 1


def _det(a, b, c, d):
    """det(a - d, b - d, c - d): positive when d is on the inner side of the triangle a, b, c (counterclockwise seen
    from outside)."""
    ax, ay, az = a[0] - d[0], a[1] - d[1], a[2] - d[2]
    bx, by, bz = b[0] - d[0], b[1] - d[1], b[2] - d[2]
    cx, cy, cz = c[0] - d[0], c[1] - d[1], c[2] - d[2]
    return ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx)


def _qhull_faces(u):
    """Qhull's triangles of the hull of the unit rows u, each ordered counterclockwise seen from outside."""
    hull = ConvexHull(u)
    f = hull.simplices.copy()
    a, b, c = u[f[:, 0]], u[f[:, 1]], u[f[:, 2]]
    flip = np.einsum("ij,ij->i", np.cross(b - a, c - a), hull.equations[:, :3]) < 0
    f[flip, 1], f[flip, 2] = f[flip, 2], f[flip, 1].copy()
    return f


def _certified(u, ints, f):
    """Whether the triangles f are exactly a closed surface with every point on their inner side or on them."""
    edges = {}
    for a, b, c in f.tolist():
        for e in ((a, b), (b, c), (c, a)):
            if e in edges:
                return False
            edges[e] = True
    if any((b, a) not in edges for a, b in edges):
        return False
    A, B, C = u[f[:, 0]][:, None], u[f[:, 1]][:, None], u[f[:, 2]][:, None]
    ad, bd, cd = A - u[None], B - u[None], C - u[None]
    det = np.einsum("fnk,fnk->fn", ad, np.cross(bd, cd))
    aa, ba, ca = np.abs(ad), np.abs(bd), np.abs(cd)
    perm = (aa[..., 0] * (ba[..., 1] * ca[..., 2] + ba[..., 2] * ca[..., 1])
            + aa[..., 1] * (ba[..., 2] * ca[..., 0] + ba[..., 0] * ca[..., 2])
            + aa[..., 2] * (ba[..., 0] * ca[..., 1] + ba[..., 1] * ca[..., 0]))
    # Shewchuk's bound: a floating orientation of binary64 points is off by less than 7.8e-16 of the permanent.
    if (det < -2e-15 * perm).any():
        return False
    for i, j in zip(*np.nonzero(np.abs(det) <= 2e-15 * perm)):
        a, b, c = f[i]
        if j not in (a, b, c) and _det(ints[a], ints[b], ints[c], ints[j]) < 0:
            return False
    return True


def _exact_faces(ints):
    """The hull's triangles by exact beneath-beyond, ordered counterclockwise seen from outside. Raises ValueError
    when the points are coplanar."""
    p0 = ints[0]
    rest = range(1, len(ints))
    i1 = next((i for i in rest if ints[i] != p0), None)
    if i1 is None:
        raise ValueError("the hull is flat")
    p1 = ints[i1]
    def cross(a, b):
        return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
    sub = lambda a, b: (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    i2 = next((i for i in rest if cross(sub(p1, p0), sub(ints[i], p0)) != (0, 0, 0)), None)
    if i2 is None:
        raise ValueError("the hull is flat")
    i3 = next((i for i in rest if _det(p0, p1, ints[i2], ints[i]) != 0), None)
    if i3 is None:
        raise ValueError("the hull is flat")
    a, b, c, d = 0, i1, i2, i3
    if _det(ints[a], ints[b], ints[c], ints[d]) < 0:
        b, c = c, b
    faces = {(a, b, c), (a, d, b), (b, d, c), (c, d, a)}
    for i in range(len(ints)):
        if i in (a, b, c, d):
            continue
        p = ints[i]
        seen = [t for t in faces if _det(ints[t[0]], ints[t[1]], ints[t[2]], p) < 0]
        if not seen:
            continue
        edges = {e for t in seen for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
        faces.difference_update(seen)
        faces.update((x, y, i) for x, y in edges if (y, x) not in edges)
    return np.array(sorted(faces))


def volume(points) -> Fraction:
    """The volume of the convex hull of directions(points), exactly: the directions are integer vectors over a power
    of two, Qhull's triangles are checked exactly to bound every point (else the hull is rebuilt exactly), and the
    volume is the sum of a . (b x c) / 6 over them. Raises ValueError on a zero or non-finite point, two points with the
    same direction or a flat hull."""
    u = directions(points)
    if len(np.unique(u, axis=0)) < len(u):
        raise ValueError("two points have the same direction")
    ints, s = _integers(u)
    try:
        f = _qhull_faces(u)
    except (QhullError, ValueError):
        f = None
    if f is None or not _certified(u, ints, f):
        f = _exact_faces(ints)
    total = sum(_det(ints[a], ints[b], ints[c], (0, 0, 0)) for a, b, c in f.tolist())
    if total <= 0:
        raise ValueError("the hull is flat")
    return Fraction(total, 6 << (3 * s))


def faces(points) -> np.ndarray:
    """The hull's triangles (k, 3) as indices, counterclockwise seen from outside (Qhull on the unit directions)."""
    return _qhull_faces(directions(points))


def degrees(points) -> np.ndarray:
    """The number of hull edges at each point (0 for a point inside the hull)."""
    f = faces(points)
    n = len(np.asarray(points))
    e = np.unique(np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1), axis=0)
    return np.bincount(e.ravel(), minlength=n)


def volume_fast(points) -> float:
    """The hull volume of the unit directions in floating point (within about 1e-15 of the exact one)."""
    return float(volume_grad(np.asarray(points, dtype=np.float64))[0])


def volume_grad(x):
    """The hull volume of the directions of the (n, 3) points x and its gradient in x (n, 3): with the hull's
    triangles fixed, d V / d a = (b x c) / 6 summed over the triangles a, b, c at a, projected on the sphere."""
    x = np.asarray(x, dtype=np.float64).reshape(-1, 3)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    u = x / norm
    f = _qhull_faces(u)
    a, b, c = u[f[:, 0]], u[f[:, 1]], u[f[:, 2]]
    bc, ca, ab = np.cross(b, c), np.cross(c, a), np.cross(a, b)
    value = np.einsum("ij,ij->", a, bc) / 6
    g = np.zeros_like(u)
    np.add.at(g, f[:, 0], bc)
    np.add.at(g, f[:, 1], ca)
    np.add.at(g, f[:, 2], ab)
    g /= 6
    return value, (g - (g * u).sum(axis=1, keepdims=True) * u) / norm


def maximize(points, deadline=None, iters=3000) -> np.ndarray:
    """L-BFGS on the hull volume of the directions, restarted while it gains (the triangles change between steps, so
    each restart starts from a fresh curvature model). Returns the unit directions with the largest volume seen."""
    u = directions(points)
    n = len(u)
    top, top_v = u, volume_fast(u)
    def f(z):
        v, g = volume_grad(z.reshape(n, 3))
        return -v, -g.ravel()
    def stop(intermediate_result):
        if deadline is not None and time.time() > deadline:
            raise StopIteration
    for _ in range(50):
        if deadline is not None and time.time() > deadline:
            break
        res = minimize(f, top.ravel(), jac=True, method="L-BFGS-B", callback=stop,
                       options={"maxiter": iters, "maxcor": 30, "gtol": 1e-16, "ftol": 1e-17})
        q = directions(res.x.reshape(n, 3))
        v = volume_fast(q)
        if v <= top_v + 1e-16:
            break
        top, top_v = q, v
    return top


def shake(points, sigma, rng, group=None) -> np.ndarray:
    """The unit points moved by Gaussian noise of scale sigma times the typical edge length sqrt(8 pi / (sqrt(3) n)):
    all of them, or only the indices in group."""
    u = directions(points).copy()
    scale = sigma * math.sqrt(8 * math.pi / (math.sqrt(3) * len(u)))
    idx = np.arange(len(u)) if group is None else np.asarray(group)
    u[idx] += rng.normal(scale=scale, size=(len(idx), 3))
    return directions(u)


def improve(points, deadline, rng, sigma=(1e-3, 3e-1)) -> np.ndarray:
    """Basin hopping until deadline: maximize, then repeatedly shake all points or a point and its nearest
    neighbours by a log-uniform sigma in the range, maximize, and keep the result when its volume is larger."""
    top = maximize(points, deadline=deadline)
    top_v = volume_fast(top)
    while time.time() < deadline:
        s = 10 ** rng.uniform(*np.log10(sigma))
        if rng.random() < 0.5:
            q = shake(top, s, rng)
        else:
            centre = top[rng.integers(len(top))]
            q = shake(top, s, rng, np.argsort(-(top @ centre))[:int(rng.integers(1, 8))])
        q = maximize(q, deadline=deadline)
        v = volume_fast(q)
        if v > top_v:
            top, top_v = q, v
    return top


def spiral(n: int, offset=0.5) -> np.ndarray:
    """n points on the generalized spiral (Fibonacci lattice): heights 1 - 2 (i + offset) / n, golden-angle turns."""
    i = np.arange(n) + offset
    z = 1 - 2 * i / n
    r = np.sqrt(np.maximum(1 - z * z, 0))
    t = math.pi * (3 - math.sqrt(5)) * np.arange(n)
    return np.stack([r * np.cos(t), r * np.sin(t), z], axis=1)


def random_config(n: int, rng, deadline=None) -> np.ndarray:
    """A fresh local optimum: n uniform random points on S^2, maximized."""
    return maximize(rng.normal(size=(n, 3)), deadline=deadline)


def rotation_group(name: str) -> np.ndarray:
    """The rotation group "C2", "D2", "T" (12), "O" (24) or "I" (60) as (g, 3, 3) matrices; T, O and I have
    the tetrahedron, octahedron and icosahedron with a vertex on the z axis or the cube with faces on the axes."""
    def axis_angle(axis, angle):
        k = np.asarray(axis, dtype=float) / np.linalg.norm(axis)
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K
    gens = {"C2": [axis_angle((0, 0, 1), math.pi)],
            "D2": [axis_angle((0, 0, 1), math.pi), axis_angle((1, 0, 0), math.pi)],
            "T": [axis_angle((1, 1, 1), 2 * math.pi / 3), axis_angle((0, 0, 1), math.pi)],
            "O": [axis_angle((1, 1, 1), 2 * math.pi / 3), axis_angle((0, 0, 1), math.pi / 2)],
            "I": [axis_angle((0, 1, (1 + math.sqrt(5)) / 2), 2 * math.pi / 5),
                  axis_angle((1, 1, 1), 2 * math.pi / 3)]}[name]
    group = [np.eye(3)]
    while True:
        new = [g @ h for g in group for h in gens]
        added = False
        for m in new:
            if not any(np.abs(m - g).max() < 1e-9 for g in group):
                group.append(m)
                added = True
        if not added:
            return np.array(group)


def orbit(seeds, name: str) -> np.ndarray:
    """The distinct images of the seed directions under rotation_group(name)."""
    out = []
    for p in np.einsum("gij,sj->gsi", rotation_group(name), directions(np.atleast_2d(seeds))).reshape(-1, 3):
        if not any(np.abs(p - q).max() < 1e-9 for q in out):
            out.append(p)
    return np.array(out)


def icosahedron() -> np.ndarray:
    """The 12 vertices of the regular icosahedron, the optimum for n = 12."""
    t = (1 + math.sqrt(5)) / 2
    v = [(0, s1, s2 * t) for s1, s2 in itertools.product((-1, 1), repeat=2)]
    v = [p for q in v for p in (q, q[1:] + q[:1], q[2:] + q[:2])]
    return directions(np.array(v, dtype=float))


def remove(points, k: int) -> np.ndarray:
    """k points removed greedily, each time the one whose removal loses the least volume."""
    u = directions(points)
    for _ in range(k):
        loss = [volume_fast(np.delete(u, i, axis=0)) for i in range(len(u))]
        u = np.delete(u, int(np.argmax(loss)), axis=0)
    return u


def hole(points) -> np.ndarray:
    """The deepest cap outside the hull: the outer normal of the triangle whose plane is nearest the centre (adding
    a point there gains the most volume to first order). Returns a unit vector."""
    u = directions(points)
    f = _qhull_faces(u)
    a, b, c = u[f[:, 0]], u[f[:, 1]], u[f[:, 2]]
    nrm = np.cross(b - a, c - a)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True)
    i = int(np.argmin(np.einsum("ij,ij->i", nrm, a)))
    return nrm[i]


def insert(points, k: int) -> np.ndarray:
    """k points added one at a time at the deepest cap, each followed by a short maximize."""
    u = directions(points)
    for _ in range(k):
        u = maximize(np.vstack([u, hole(u)]), iters=300)
    return u
