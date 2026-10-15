"""Pack 26 non-overlapping circles in the unit square so that the sum of their radii is as large as possible.

The best known packings reach a sum of about 2.635. This seed places a 5x5 grid and one tiny circle.
"""

import math

N = 26
TOLERANCE = 1e-9


def solve():
    """Return (centers, radii): 26 (x, y) pairs and 26 radii."""
    centers, radii = [], []
    for i in range(5):
        for j in range(5):
            centers.append((0.1 + 0.2 * i, 0.1 + 0.2 * j))
            radii.append(0.1)
    centers.append((0.2, 0.2))
    radii.append(0.0)
    return centers, radii


def _check(result):
    centers, radii = result
    centers = [(float(x), float(y)) for x, y in centers]
    radii = [float(r) for r in radii]
    if len(centers) != N or len(radii) != N:
        raise ValueError(f"need {N} centers and {N} radii")
    for (x, y), r in zip(centers, radii):
        if not all(math.isfinite(v) for v in (x, y, r)) or r < 0:
            raise ValueError("coordinates and radii must be finite, radii non-negative")
        if x - r < -TOLERANCE or x + r > 1 + TOLERANCE or y - r < -TOLERANCE or y + r > 1 + TOLERANCE:
            raise ValueError(f"circle at ({x:.4f}, {y:.4f}) leaves the square")
    for i in range(N):
        for j in range(i + 1, N):
            gap = math.dist(centers[i], centers[j]) - radii[i] - radii[j]
            if gap < -TOLERANCE:
                raise ValueError(f"circles {i} and {j} overlap by {-gap:.2e}")
    return centers, radii


def score(result):
    """Sum of radii of a valid packing."""
    _, radii = _check(result)
    return sum(radii)


def metrics(result):
    """Shape of a packing: how many circles touch the border, and how unequal the radii are."""
    centers, radii = _check(result)
    touching = sum(min(x - r, y - r, 1 - x - r, 1 - y - r) < 1e-6 for (x, y), r in zip(centers, radii))
    mean = sum(radii) / N
    return {"border_share": touching / N, "radius_spread": math.sqrt(sum((r - mean) ** 2 for r in radii) / N)}
