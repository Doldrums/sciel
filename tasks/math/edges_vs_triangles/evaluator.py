"""Evaluator for edges_vs_triangles; smoke/full/confirm on the sciel contract."""

from sciel.candidate_modules import load_candidate
import time
import traceback

import numpy as np

BENCHMARK = 64
NUM_VERTICES = 16


def _load(program_path, module_name):
    return load_candidate(program_path, module_name)


# --- behavior readout -------------------------------------------------------
# The score here is one number, and the other reported field is a linear restatement
# of it. These describe the SHAPE of the constructed object instead, so the diversity
# map has something to separate equal-scoring constructions by. Bounded, finite, and
# never raising: a degenerate object reports zeros rather than failing an evaluation
# that otherwise succeeded.

_BEHAVIOR_ZERO = {"b_dispersion": 0.0, "b_boundary_frac": 0.0, "b_contact": 0.0,
                  "b_offset": 0.0, "b_concentration": 0.0, "b_structure": 0.0}


def _squash(x):
    """Unbounded ratio -> [0,1). Every behavior slot shares one scale on purpose: fed
    to a linear trunk, a slot that ranges to 4 while the rest range to 1 dominates the
    others for no reason but its units."""
    x = float(x)
    return x / (1.0 + x) if x > 0 and np.isfinite(x) else 0.0


def _finite(d):
    return {k: (float(v) if np.isfinite(v) else 0.0) for k, v in d.items()}


def _c(n, k):
    """n choose k, as a float; 0 when n < k."""
    if n < k:
        return 0.0
    out = 1.0
    for i in range(k):
        out *= (n - i) / (i + 1)
    return out


def _behavior(A):
    """A: the symmetric 0/1 adjacency matrix, diagonal already cleared."""
    try:
        A = np.asarray(A, dtype=np.float64)
        n = A.shape[0]
        if n < 2:
            return dict(_BEHAVIOR_ZERO)
        deg = A.sum(axis=1)
        mean_deg = float(deg.mean())
        total = float(deg.sum())
        k = max(1, n // 4)
        top = float(np.sort(deg)[-k:].sum())
        # Four-cycles, the structure a triangle-free graph is actually built out of.
        # tr(A^4) = 8*C4 + 2m + 4*sum C(d,2); dropping either correction term turns
        # this into a degree statistic that fires on stars and paths, which have none.
        c4 = float(np.trace(A @ A @ A @ A) - total - 2 * float((deg * (deg - 1)).sum())) / 8.0
        return _finite({
            "b_dispersion": _squash(float(deg.std() / mean_deg) if mean_deg > 0 else 0.0),
            "b_boundary_frac": float((deg == 0).mean()),
            "b_contact": float(np.clip(mean_deg / (n - 1), 0.0, 1.0)),
            "b_offset": float(np.clip((deg.max() - deg.min()) / (n - 1), 0.0, 1.0)),
            "b_concentration": float(top / total) if total > 0 else 0.0,
            "b_structure": float(np.clip(c4 / max(3.0 * _c(n, 4), 1.0), 0.0, 1.0)),
        })
    except (ValueError, FloatingPointError, ZeroDivisionError):
        return dict(_BEHAVIOR_ZERO)


def _evaluate(program_path, module_name):
    program = _load(program_path, module_name)
    if not hasattr(program, "graph_edges"):
        raise AttributeError("missing graph_edges")
    start_time = time.time()
    M = program.graph_edges()
    eval_time = time.time() - start_time

    if not isinstance(M, np.ndarray):
        M = np.array(M)
    if M.shape != (NUM_VERTICES, NUM_VERTICES):
        raise ValueError(
            f"Invalid shapes: matrix = {M.shape}, expected {(NUM_VERTICES, NUM_VERTICES)}"
        )

    A = (np.asarray(M) >= 0.5).astype(int)
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)

    num_edges = int(A.sum() // 2)
    triangles = np.trace(A @ A @ A) // 6

    if triangles > 0:
        return {
            "ok": False,
            "score": 0.0,
            "cases": {"num_edges": float(num_edges), "combined_score": 0.0,
                      **_behavior(A)},
            "artifacts": {"error": "graph contains triangles"},
        }

    combined = float(num_edges / BENCHMARK)
    return {
        "ok": True,
        "score": combined,
        "cases": {
            "num_edges": float(num_edges),
            "combined_score": combined,
            **_behavior(A),
        },
        "artifacts": {"eval_time": f"{eval_time:.4f}"},
    }


def smoke(program_path):
    try:
        return _evaluate(program_path, "program_smoke")
    except Exception as e:
        return _fail(str(e))


def full(program_path):
    try:
        return _evaluate(program_path, "program_full")
    except Exception as e:
        return _fail(str(e))


def confirm(program_path, n=1):
    """Re-run n times, keep the worst, and report how many trials ran.

    Ignoring n while the engine flagged the result as a confirmed
    multi-sample let a single fresh draw replace the full score."""
    n = max(1, int(n or 1))
    runs = [full(program_path) for _ in range(n)]
    failures = [r for r in runs if not r.get("ok")]
    # A single failed repetition disproves reliable execution. Do not certify
    # the surviving successes as if every requested trial had succeeded.
    worst = failures[0] if failures else min(runs, key=lambda r: r["score"])
    worst = dict(worst)
    worst["cases"] = {**worst["cases"], "n_trials": float(n)}
    return worst


def _fail(msg):
    return {
        "ok": False,
        "score": 0.0,
        "cases": {"combined_score": 0.0, "num_edges": 0.0, **_BEHAVIOR_ZERO},
        "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]},
    }
