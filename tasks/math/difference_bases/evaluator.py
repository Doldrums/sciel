"""Evaluator for difference_bases; smoke/full/confirm on the sciel contract."""

from sciel.candidate_modules import load_candidate
import itertools
import time
import traceback

import numpy as np

BENCHMARK = 8
MIN_VAL = 0
MAX_VAL = 60


def _load(program_path, module_name):
    return load_candidate(program_path, module_name)


def _is_sidon(values):
    diffs = [abs(int(a) - int(b)) for a, b in itertools.combinations(values, 2)]
    return len(diffs) == len(set(diffs))


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


def _behavior(values):
    """values: the sorted Sidon set inside [MIN_VAL, MAX_VAL]."""
    try:
        v = np.sort(np.asarray(values, dtype=np.float64))
        span = float(MAX_VAL - MIN_VAL)
        if v.size < 2 or span <= 0 or not np.isfinite(v).all():
            return dict(_BEHAVIOR_ZERO)
        gaps = np.diff(v)
        mean_gap = float(gaps.mean())
        near_end = float(((v - MIN_VAL < 0.1 * span) | (MAX_VAL - v < 0.1 * span)).mean())
        mid = (MIN_VAL + MAX_VAL) / 2.0
        return _finite({
            "b_dispersion": _squash(float(gaps.std() / mean_gap) if mean_gap > 0 else 0.0),
            "b_boundary_frac": near_end,
            # NOT mean_gap/span: that equals b_structure/(n-1) exactly, and n is the
            # score, so the slot would carry no information of its own. Gap regularity
            # does: an evenly spaced basis and a clustered one differ here at equal span.
            "b_contact": float(np.clip(float(gaps.min()) / mean_gap, 0.0, 1.0))
                         if mean_gap > 0 else 0.0,
            "b_offset": float(np.clip(abs(float(v.mean()) - mid) / (span / 2.0), 0.0, 1.0)),
            "b_concentration": float((v < mid).mean()),
            "b_structure": float(np.clip((float(v.max()) - float(v.min())) / span, 0.0, 1.0)),
        })
    except (ValueError, FloatingPointError, ZeroDivisionError):
        return dict(_BEHAVIOR_ZERO)


def _evaluate(program_path, module_name):
    program = _load(program_path, module_name)
    if not hasattr(program, "difference_basis"):
        raise AttributeError("missing difference_basis")
    start_time = time.time()
    basis = program.difference_basis()
    eval_time = time.time() - start_time

    if not isinstance(basis, np.ndarray):
        basis = np.array(basis)

    values = np.unique(basis.astype(np.int64))
    set_size = int(values.size)

    if np.any(values < MIN_VAL) or np.any(values > MAX_VAL) or not _is_sidon(values):
        return {
            "ok": False,
            "score": 0.0,
            "cases": {"set_size": float(set_size), "combined_score": 0.0,
                      **_behavior(values)},
            "artifacts": {"error": "not a Sidon set"},
        }

    combined = float(set_size / BENCHMARK)
    return {
        "ok": True,
        "score": combined,
        "cases": {
            "set_size": float(set_size),
            "combined_score": combined,
            **_behavior(values),
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
        "cases": {"combined_score": 0.0, "set_size": 0.0, **_BEHAVIOR_ZERO},
        "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]},
    }
