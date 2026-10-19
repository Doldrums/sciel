"""Evaluator for Erdős' minimum overlap problem; smoke/full/confirm on the sciel contract.

By Haugland (2016) the constant is the infimum, over step functions h on [0, 2] with values in [0, 1]
and integral 1, of max_k integral h(x) (1 - h(x + k)) dx, with h and 1 - h zero outside [0, 2]. For n
equal steps the integral is piecewise linear in k with knots on the grid, so the bound is
(2/n) max_s sum_i h_i (1 - h_i+s). It is computed exactly: binary64 heights are exact rationals over
one power of two, the correlation is taken in integers, the integral must be exactly 1 in those
rationals, and the bound is rounded once.
"""

from sciel.candidate_modules import load_candidate
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
import uuid

import numpy as np

REFERENCE = 0.380876
MAX_STEPS = 10000
# A height this close to 0 or 1 makes the exact integers grow without changing the bound; such
# heights must be exactly 0 or 1.
MIN_RELATIVE = 2.0 ** -60


def _load(program_path, module_name="program"):
    return load_candidate(program_path, module_name)


def _heights(values):
    """The candidate's heights as binary64, with the shape and range checks of the task."""
    h = np.array(values, dtype=np.float64, copy=True)
    if h.ndim != 1 or not 1 <= h.size <= MAX_STEPS:
        raise ValueError(f"heights must be a 1-D sequence of 1..{MAX_STEPS} numbers, got shape {h.shape}")
    if not np.isfinite(h).all():
        raise ValueError("heights must be finite")
    if (h < 0).any() or (h > 1).any():
        raise ValueError("heights must lie in [0, 1]")
    if ((h > 0) & (h < MIN_RELATIVE)).any() or ((h < 1) & (h > 1 - MIN_RELATIVE)).any():
        raise ValueError("a height is within 2**-60 of 0 or 1 without equalling it; set it to exactly 0 or 1")
    return h


def _integers(values):
    """Exact binary64 values as integers over one common power-of-two denominator."""
    ratios = [float(v).as_integer_ratio() for v in values]
    scale = max(d for _, d in ratios)
    return [n * (scale // d) for n, d in ratios]


def _convolve(a, b):
    """Exact convolution of non-negative integer lists by Kronecker substitution."""
    width = (max(a).bit_length() + max(b).bit_length() + min(len(a), len(b)).bit_length()) // 8 + 2

    def pack(xs):
        return int.from_bytes(b"".join(x.to_bytes(width, "little") for x in xs), "little")
    product = (pack(a) * pack(b)).to_bytes(width * (len(a) + len(b)), "little")
    return [int.from_bytes(product[i * width:(i + 1) * width], "little") for i in range(len(a) + len(b) - 1)]


def bound(heights) -> Fraction:
    h = _integers(list(heights) + [1.0])
    h, scale = h[:-1], h[-1]
    if 2 * sum(h) != len(h) * scale:
        raise ValueError("the integral of h must be exactly 1: heights must sum to exactly n/2")
    rest = [scale - x for x in h]
    return Fraction(2 * max(_convolve(h, rest[::-1])), len(h) * scale * scale)


# --- behavior readout -------------------------------------------------------
# The score is one number. These describe the SHAPE of the step function instead, so two
# functions with the same bound can land in different cells of the diversity map.

_BEHAVIOR_ZERO = {"b_dispersion": 0.0, "b_boundary_frac": 0.0, "b_contact": 0.0,
                  "b_offset": 0.0, "b_concentration": 0.0, "b_structure": 0.0}


def _behavior(h):
    """Six bounded descriptors: height spread, mass near the ends, zero steps, centre of mass,
    mass in the tallest tenth of steps, and resolution. Never raises."""
    try:
        a = np.abs(np.asarray(h, dtype=np.float64))
        mass = float(a.sum())
        if a.size == 0 or mass <= 0 or not np.isfinite(a).all():
            return dict(_BEHAVIOR_ZERO)
        n = a.size
        x = (np.arange(n) + 0.5) / n
        live = a[a > 0]
        spread = float(live.std() / live.mean())
        ends = float(a[(x < 0.2) | (x > 0.8)].sum() / mass)
        offset = abs(float((a * x).sum() / mass) - 0.5) / 0.5
        top = np.sort(a)[::-1][:max(1, n // 10)]
        out = {"b_dispersion": spread / (1.0 + spread), "b_boundary_frac": ends,
               "b_contact": float((a == 0).mean()), "b_offset": min(offset, 1.0),
               "b_concentration": float(top.sum() / mass),
               "b_structure": math.log(n) / math.log(MAX_STEPS)}
        return {k: (float(v) if np.isfinite(v) else 0.0) for k, v in out.items()}
    except (ValueError, FloatingPointError, ZeroDivisionError):
        return dict(_BEHAVIOR_ZERO)


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
    if not hasattr(program, "step_heights"):
        raise AttributeError("missing step_heights")
    start_time = time.time()
    h = _heights(program.step_heights())
    eval_time = time.time() - start_time
    value = bound(h)
    upper = float(value)
    combined = float(REFERENCE / value)
    cases = {"upper_bound": upper, "inverse_bound": float(1 / value), "steps": float(h.size),
             "combined_score": combined, "validity": 1.0}
    cases.update(_behavior(h))
    witness = {
        "schema": "sciel-full-output/1",
        "source_sha256": source_sha, "evaluator_sha256": evaluator_sha,
        "invocation_id": invocation_id, "stage": stage, "confirmation": None,
        "input_sha256": None, "seed_sha256": None,
        "output": {"heights": h.tolist(), "bound": [value.numerator, value.denominator]},
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
    """Re-run the program n times and keep the worst result; a record should survive its own
    repetition."""
    n = max(1, int(n or 1))
    runs = [full(program_path) for _ in range(n)]
    failures = [r for r in runs if not r.get("ok")]
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
        "cases": {"combined_score": 0.0, "upper_bound": 0.0, "inverse_bound": 0.0, "steps": 0.0,
                  "validity": 0.0, **_BEHAVIOR_ZERO},
        "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]},
    }
