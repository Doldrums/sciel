"""Coverings of the unit sphere by N equal caps: the covering radius of a point set and the published best.

The covering radius of N unit vectors is the largest angle from any point of the sphere to its nearest vector. When
the origin lies strictly inside their convex hull it is attained at a circumcentre of a hull face: each face's plane
n.x + d = 0 (unit outward normal n, d < 0) cuts the sphere in a circle of angular radius arccos(-d) around n.
"""

import numpy as np
from scipy.spatial import ConvexHull

from sciel.task import asset

_data = asset("caps_data")
SIZES = tuple(range(4, 131))
RADIUS_DEG = dict(_data.RADIUS_DEG)


def published(n: int) -> np.ndarray:
    """The published best configuration for n caps as an (n, 3) array of unit vectors."""
    return unit(np.array(_data.COORDS[n].split(), dtype=np.float64).reshape(n, 3))


def unit(points) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    return p / np.linalg.norm(p, axis=1, keepdims=True)


def faces(points) -> tuple:
    """Hull faces of the normalized points: (simplices (F, 3), angular radii in degrees (F,), centres (F, 3)).
    Raises ValueError when the points do not cover the sphere (the origin is not strictly inside the hull)."""
    p = unit(points)
    hull = ConvexHull(p)
    normals, offsets = hull.equations[:, :3], hull.equations[:, 3]
    if not (offsets < -1e-12).all():
        raise ValueError("the points lie in a closed hemisphere: they do not cover the sphere")
    return hull.simplices, np.degrees(np.arccos(np.clip(-offsets, -1.0, 1.0))), normals


def radius(points) -> float:
    """The covering radius in degrees."""
    return float(faces(points)[1].max())
