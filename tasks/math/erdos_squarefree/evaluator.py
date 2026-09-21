"""Evaluator for erdos_squarefree; smoke/full/confirm on the sciel contract."""

from sciel.candidate_modules import load_candidate
import time
import traceback

import numpy as np

BENCHMARK = 0.04
N = 500


def _load(program_path, module_name):
    return load_candidate(program_path, module_name)


def _squarefree_sieve(limit):
    sf = np.ones(limit + 1, dtype=bool)
    sf[0] = False
    i = 2
    while i * i <= limit:
        sq = i * i
        sf[sq::sq] = False
        i += 1
    return sf


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


def _behavior(subset):
    """subset: the chosen integers in {1..N}. Shape of the choice, not its score."""
    try:
        v = np.sort(np.asarray(subset, dtype=np.float64))
        if v.size < 2 or not np.isfinite(v).all():
            return dict(_BEHAVIOR_ZERO)
        gaps = np.diff(v)
        mean_gap = float(gaps.mean())
        return _finite({
            "b_dispersion": _squash(float(gaps.std() / mean_gap) if mean_gap > 0 else 0.0),
            "b_boundary_frac": float((v > 0.9 * N).mean()),
            "b_contact": float(np.clip(mean_gap / N, 0.0, 1.0)),
            "b_offset": float(np.clip(abs(float(v.mean()) / N - 0.5) * 2.0, 0.0, 1.0)),
            "b_concentration": float((v <= N / 2.0).mean()),
            "b_structure": float((np.round(v) % 2 == 0).mean()),
        })
    except (ValueError, FloatingPointError, ZeroDivisionError):
        return dict(_BEHAVIOR_ZERO)


def _evaluate(program_path, module_name):
    program = _load(program_path, module_name)
    if not hasattr(program, "erdos_squarefree500"):
        raise AttributeError("missing erdos_squarefree500")
    start_time = time.time()
    subset = program.erdos_squarefree500()
    eval_time = time.time() - start_time

    if not isinstance(subset, np.ndarray):
        subset = np.array(subset)
    subset = np.asarray(subset).reshape(-1)

    if subset.size == 0:
        raise ValueError("Subset is empty.")
    if not np.all(np.isfinite(subset)):
        raise ValueError("Subset contains NaN or infinite values.")
    if not np.allclose(subset, np.round(subset)):
        raise ValueError("Subset must contain integer values.")
    subset = np.round(subset).astype(np.int64)
    if np.any(subset < 1) or np.any(subset > N):
        raise ValueError(f"Subset elements must lie in {{1, ..., {N}}}.")
    if np.unique(subset).size != subset.size:
        raise ValueError("Subset contains duplicate elements.")

    sf = _squarefree_sieve(N * N + 1)
    products = np.outer(subset, subset) + 1
    is_sf = sf[products]
    violations = int(np.sum(np.triu(is_sf, k=1)))

    density = subset.size / N
    subset_score = density - violations
    combined = subset_score / BENCHMARK
    if not np.isfinite(combined):
        combined = 0.0

    return {
        "ok": True,
        "score": float(combined),
        "cases": {
            "subset_score": float(subset_score),
            "subset_size": float(subset.size),
            "violations": float(violations),
            "combined_score": float(combined),
            **_behavior(subset),
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
        "cases": {
            **_BEHAVIOR_ZERO,
            "combined_score": 0.0,
            "subset_score": 0.0,
            "subset_size": 0.0,
            "violations": 0.0,
        },
        "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]},
    }
