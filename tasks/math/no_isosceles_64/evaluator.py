"""Evaluator for no-isosceles subsets of the 64 x 64 grid; smoke/full/confirm on the sciel contract.

C(64) is the largest subset of {0..63}^2 with no three points forming a possibly flat isosceles triangle. The best
known is AlphaEvolve's 112 (arXiv 2511.02864, section 6.39). A set is checked exactly in integers: from every point
the squared distances to all the other points must be distinct, which also excludes a point at the midpoint of two
others. combined_score = |S| / 112, so a score above 1 is a new lower bound.
"""

from sciel.candidate_modules import load_candidate
import hashlib
import json
from pathlib import Path
import time
import traceback
import uuid

import numpy as np

REFERENCE = 112
N = 64


def _load(program_path, module_name="program"):
    return load_candidate(program_path, module_name)


def _points(values):
    """The candidate's points as distinct integer pairs inside the grid."""
    out = []
    for item in values:
        x, y = item
        if int(x) != x or int(y) != y:
            raise ValueError(f"point {item!r} is not a pair of integers")
        out.append((int(x), int(y)))
    if not out:
        raise ValueError("the set is empty")
    if len(set(out)) != len(out):
        raise ValueError("the set repeats a point")
    if not all(0 <= x < N and 0 <= y < N for x, y in out):
        raise ValueError(f"points must lie in {{0..{N - 1}}}^2")
    return sorted(out)


def check(points):
    """Raise at the first isosceles triangle, naming its apex and the two equally distant points."""
    p = np.asarray(points, dtype=np.int64)
    d = p[:, None, :] - p[None, :, :]
    d = (d * d).sum(axis=2)
    for i in range(len(p)):
        row = np.delete(d[i], i)
        others = np.delete(np.arange(len(p)), i)
        values, first, counts = np.unique(row, return_index=True, return_counts=True)
        if (counts > 1).any():
            k = int(np.flatnonzero(counts > 1)[0])
            pair = others[row == values[k]][:2]
            raise ValueError(f"isosceles: {points[pair[0]]} and {points[pair[1]]} are equally far from {points[i]}")


# --- behavior readout -------------------------------------------------------
# The score is one number. These describe the SHAPE of the set instead, so two sets of the same size can land in
# different cells of the diversity map.

_BEHAVIOR_ZERO = {"b_dispersion": 0.0, "b_boundary_frac": 0.0, "b_contact": 0.0,
                  "b_offset": 0.0, "b_concentration": 0.0, "b_structure": 0.0}


def _behavior(points):
    """Six bounded descriptors: radial spread, share on the border, share fixed by the half-turn, distance of the
    centroid from the centre, share in the four corner quarters' outer rings, and size. Never raises."""
    try:
        p = np.asarray(points, dtype=np.float64)
        c = (N - 1) / 2.0
        r = np.hypot(p[:, 0] - c, p[:, 1] - c) / np.hypot(c, c)
        border = ((p == 0) | (p == N - 1)).any(axis=1).mean()
        mapped = {(N - 1 - int(x), N - 1 - int(y)) for x, y in points}
        symmetric = len(mapped & set(points)) / len(points)
        offset = np.hypot(p[:, 0].mean() - c, p[:, 1].mean() - c) / np.hypot(c, c)
        outer = (np.minimum(p, N - 1 - p).min(axis=1) < N / 8).mean()
        out = {"b_dispersion": float(r.std() / (r.mean() + 1e-12)) / 2.0, "b_boundary_frac": float(border),
               "b_contact": float(symmetric), "b_offset": float(min(offset, 1.0)),
               "b_concentration": float(outer), "b_structure": min(len(points) / (2.0 * N), 1.0)}
        return {k: (min(max(float(v), 0.0), 1.0) if np.isfinite(v) else 0.0) for k, v in out.items()}
    except (ValueError, FloatingPointError, ZeroDivisionError):
        return dict(_BEHAVIOR_ZERO)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _evaluate(program_path, stage):
    source_sha = _source_hash(program_path)
    evaluator_sha = _source_hash(__file__)
    invocation_id = uuid.uuid4().hex
    program = _load(program_path, "program_" + stage)
    if not hasattr(program, "points"):
        raise AttributeError("missing points")
    start_time = time.time()
    pts = _points(program.points())
    eval_time = time.time() - start_time
    check(pts)
    combined = len(pts) / REFERENCE
    cases = {"size": float(len(pts)), "combined_score": combined, "validity": 1.0}
    cases.update(_behavior(pts))
    witness = {
        "schema": "sciel-full-output/1",
        "source_sha256": source_sha, "evaluator_sha256": evaluator_sha,
        "invocation_id": invocation_id, "stage": stage, "confirmation": None,
        "input_sha256": None, "seed_sha256": None,
        "output": {"points": [list(p) for p in pts], "size": len(pts)},
    }
    artifacts = {"eval_time": f"{eval_time:.4f}", "full_output": _canonical(witness)}
    if source_sha != _source_hash(program_path) or evaluator_sha != _source_hash(__file__):
        raise ValueError("source or evaluator changed during invocation")
    return {"ok": True, "score": combined, "cases": cases, "artifacts": artifacts}


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
    """Re-run the program n times and keep the worst result; a record should survive its own repetition."""
    n = max(1, int(n or 1))
    runs = [full(program_path) for _ in range(n)]
    failures = [r for r in runs if not r.get("ok")]
    selected = (next(i for i, result in enumerate(runs) if not result.get("ok"))
                if failures else min(range(n), key=lambda i: runs[i]["score"]))
    worst = dict(runs[selected])
    worst["cases"] = {**worst["cases"], "n_trials": float(n)}
    worst["artifacts"] = dict(worst.get("artifacts", {}))
    selection = {"trials": n, "selected_trial": selected, "selection": "first_failure_else_lowest_score"}
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
        "cases": {"combined_score": 0.0, "size": 0.0, "validity": 0.0, **_BEHAVIOR_ZERO},
        "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]},
    }
