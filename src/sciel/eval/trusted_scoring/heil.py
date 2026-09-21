"""Pure Heilbronn metric, extracted unchanged from the pinned legacy evaluator."""
import itertools
import numpy as np


BENCHMARK = 0.036529889880030156


TOL = 1e-6


NUM_POINTS = 11


def _check_inside_triangle_wtol(points, tol=1e-6):
    for x, y in points:
        cond1 = y >= -tol
        cond2 = np.sqrt(3) * x <= np.sqrt(3) - y + tol
        cond3 = y <= np.sqrt(3) * x + tol
        if not (cond1 and cond2 and cond3):
            raise ValueError(
                f"Point ({x}, {y}) is outside the equilateral triangle (tolerance: {tol})."
            )


def _triangle_area(a, b, c):
    return np.abs(a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])) / 2


_TRI_A = np.array([0.0, 0.0])


_TRI_B = np.array([1.0, 0.0])


_TRI_C = np.array([0.5, np.sqrt(3) / 2])


_TRI_CENTROID = (_TRI_A + _TRI_B + _TRI_C) / 3.0


def _edge_distances(p):
    """Distance from a point to each of the three edges of the unit triangle."""
    s3 = np.sqrt(3)
    return (p[1],                                  # edge AB: y = 0
            (s3 - s3 * p[0] - p[1]) / 2.0,         # edge BC: s3 x + y = s3
            (s3 * p[0] - p[1]) / 2.0)


_BEHAVIOR_ZERO = {"b_dispersion": 0.0, "b_boundary_frac": 0.0, "b_contact": 0.0,
                  "b_offset": 0.0, "b_concentration": 0.0, "b_structure": 0.0}


def _behavior(points, areas):
    """Six bounded shape descriptors of the placement. Never raises: a degenerate
    configuration reports zeros rather than aborting an otherwise valid evaluation."""
    zero = dict(_BEHAVIOR_ZERO)
    try:
        pts = np.asarray(points, dtype=float)
        a = np.asarray(areas, dtype=float)
        if pts.shape[0] < 3 or a.size == 0 or not np.isfinite(pts).all() \
                or not np.isfinite(a).all():
            return zero
        mean_area = float(a.mean())
        dispersion = float(a.std() / mean_area) if mean_area > 0 else 0.0

        on_edge = sum(1 for p in pts if min(abs(d) for d in _edge_distances(p)) < 1e-3)

        d = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        nn = d.min(axis=1)
        # a fully spread placement has nearest neighbours near 1/sqrt(n)
        contact = float(np.clip(nn.mean() * np.sqrt(pts.shape[0]), 0.0, 1.0))

        offset = float(np.linalg.norm(pts.mean(axis=0) - _TRI_CENTROID) / 0.5)

        concentration = float((a < mean_area).mean())

        srt = np.sort(a)
        structure = float(srt[0] / srt[1]) if srt.size > 1 and srt[1] > 0 else 0.0

        out = {"b_dispersion": dispersion / (1.0 + dispersion), "b_boundary_frac": on_edge / len(pts),
               "b_contact": contact, "b_offset": min(offset, 1.0),
               "b_concentration": concentration, "b_structure": structure}
        return {k: (float(v) if np.isfinite(v) else 0.0) for k, v in out.items()}
    except (ValueError, FloatingPointError, ZeroDivisionError):
        return zero


def score(points):
    _check_inside_triangle_wtol(points, TOL)
    a = np.array([0, 0])
    b = np.array([1, 0])
    c = np.array([0.5, np.sqrt(3) / 2])
    areas = [_triangle_area(p1, p2, p3)
             for p1, p2, p3 in itertools.combinations(points, 3)]
    min_triangle_area = min(areas)
    min_area_normalized = float(min_triangle_area / _triangle_area(a, b, c))
    combined = float(min_area_normalized / BENCHMARK)
    cases = {
        "min_area": float(min_triangle_area),
        "min_area_normalized": min_area_normalized,
        "combined_score": combined,
        "validity": 1.0,
    }
    cases.update(_behavior(points, areas))
    return {"ok": True, "score": combined, "cases": cases}
