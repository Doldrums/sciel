"""Evaluator for heilbronn_triangle (n=11); smoke/full/confirm on the sciel contract."""

from sciel.candidate_modules import load_candidate
import itertools
import hashlib
import json
from pathlib import Path
import time
import traceback
import uuid

import numpy as np

BENCHMARK = 0.036529889880030156
TOL = 1e-6
NUM_POINTS = 11


def _load(program_path, module_name="program"):
    return load_candidate(program_path, module_name)


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


# --- behavior readout -------------------------------------------------------
# The score is one number: the smallest triangle area. These describe the SHAPE of
# the point configuration instead, so two placements with the same score can still
# land in different cells of the diversity map. None of them is a monotone function
# of the score. Keys are the engine's task-agnostic behavior slots.

_TRI_A = np.array([0.0, 0.0])
_TRI_B = np.array([1.0, 0.0])
_TRI_C = np.array([0.5, np.sqrt(3) / 2])
_TRI_CENTROID = (_TRI_A + _TRI_B + _TRI_C) / 3.0


def _edge_distances(p):
    """Distance from a point to each of the three edges of the unit triangle."""
    s3 = np.sqrt(3)
    return (p[1],                                  # edge AB: y = 0
            (s3 - s3 * p[0] - p[1]) / 2.0,         # edge BC: s3 x + y = s3
            (s3 * p[0] - p[1]) / 2.0)              # edge CA: y = s3 x


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


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _evaluate(program_path, stage):
    source_sha = _source_hash(program_path)
    evaluator_sha = _source_hash(__file__)
    invocation_id = uuid.uuid4().hex
    program = _load(program_path, "program_" + stage)
    if not hasattr(program, "heilbronn_triangle11"):
        raise AttributeError("missing heilbronn_triangle11")
    start_time = time.time()
    points = program.heilbronn_triangle11()
    eval_time = time.time() - start_time
    # Candidate dtypes must not choose the scorer's arithmetic precision.
    # In float16 even exactly collinear representable points can have area > 0.
    # Detach the exact float64 values used by both scoring and saved evidence.
    points = np.array(points, dtype=np.float64, copy=True)
    if points.shape != (NUM_POINTS, 2):
        raise ValueError(f"Invalid shapes: points = {points.shape}, expected {(NUM_POINTS, 2)}")
    if not np.isfinite(points).all():
        raise ValueError("Points must contain only finite coordinates.")
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
    witness = {
        "schema": "sciel-full-output/1",
        "source_sha256": source_sha, "evaluator_sha256": evaluator_sha,
        "invocation_id": invocation_id, "stage": stage, "confirmation": None,
        "input_sha256": None, "seed_sha256": None,
        "output": {"points": points.tolist()},
    }
    # JSON round-trips each scored binary64 coordinate without decimal rounding.
    # Keep artifacts as strings through the evaluator subprocess interface.
    artifacts = {"eval_time": f"{eval_time:.4f}", "full_output": _canonical(witness)}
    if source_sha != _source_hash(program_path) or evaluator_sha != _source_hash(__file__):
        raise ValueError("source or evaluator changed during invocation")
    return {
        "ok": True,
        "score": combined,
        "cases": cases,
        "artifacts": artifacts,
    }


def smoke(program_path):
    try:
        return _evaluate(program_path, "smoke")
    except Exception as e:
        return _fail(str(e))


def full(program_path):
    try:
        return _evaluate(program_path, "full")
    except Exception as e:
        return _fail(str(e))


def confirm(program_path, n=3):
    """Re-run the program n times and keep the WORST result.

    This used to be `return full(program_path)`: n was ignored, one extra
    draw was taken, and the engine still recorded the result as a confirmed
    multi-sample. Keeping the worst is the point of a confirmation -- a
    record should survive its own repetition."""
    n = max(1, int(n or 1))
    runs = [full(program_path) for _ in range(n)]
    failures = [r for r in runs if not r.get("ok")]
    # A single failed repetition disproves reliable execution. Do not certify
    # the surviving successes as if every requested trial had succeeded.
    selected = (next(i for i, result in enumerate(runs) if not result.get("ok"))
                if failures else min(range(n), key=lambda i: runs[i]["score"]))
    worst = dict(runs[selected])
    worst["cases"] = {**worst["cases"], "n_trials": float(n)}
    worst["artifacts"] = dict(worst.get("artifacts", {}))
    selection = {"trials": n, "selected_trial": selected,
                 "selection": "first_failure_else_lowest_score"}
    worst["artifacts"]["confirmation_selection"] = _canonical(selection)
    if "full_output" in worst["artifacts"]:
        witness = json.loads(worst["artifacts"]["full_output"])
        witness["confirmation"] = selection
        worst["artifacts"]["full_output"] = _canonical(witness)
    return worst


def _fail(msg):
    return {
        "ok": False,
        "score": 0.0,
        "cases": {"combined_score": 0.0, "min_area": 0.0,
                  "min_area_normalized": 0.0, "validity": 0.0,
                  **_BEHAVIOR_ZERO},
        "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]},
    }
