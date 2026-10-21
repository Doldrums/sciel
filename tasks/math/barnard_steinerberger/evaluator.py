"""Evaluator for the Barnard-Steinerberger autocorrelation constant from below; smoke/full/confirm on the sciel
contract.

C is the least constant with min_{0<=t<=1} int f(x) f(x+t) dx <= C (int f)^2 for every non-negative f in L^1
(Tao's optimization constants, problem 1c). A non-negative step function f = sum_i w_i 1[i/K, (i+1)/K) proves
C >= K min_{0<=k<=K} sum_i w_i w_{i+k} / (sum_i w_i)^2: its autocorrelation is linear between consecutive multiples
of 1/K, so the minimum over t is at some k/K, where it equals (1/K) sum_i w_i w_{i+k} (Kravitz, Theorem 3.1). This is
the bound rozetyp's verify.py computes. It is computed exactly: the weights are binary64 (integers up to 2**53
convert exactly), read as rationals over one power of two, the correlations are taken in integers, and the ratio is
rounded once.
"""

from sciel.candidate_modules import load_candidate
import base64
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
import uuid

import numpy as np

# Lean-checked atom-plus-polynomial construction of hashiqiqixian (minimum-autocorrelation-certificate, September
# 2026); the best published step function is rozetyp's K = 6144 certificate, 0.4103287175.
REFERENCE = 0.4103779774321215
MAX_K = 65536
# The witness keeps the weights as base64 binary64, which must fit the 1 MiB artifact limit.
MAX_WEIGHTS = 81920
# A nonzero weight below this fraction of the largest makes the exact integers grow without changing the bound;
# such weights must be exactly zero.
MIN_RELATIVE = 2.0 ** -60


def _load(program_path, module_name="program"):
    return load_candidate(program_path, module_name)


def _construction(result):
    """K and the weights as binary64, read exactly: integers up to 2**53 convert without rounding."""
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise ValueError("step_weights() must return a pair (K, weights)")
    K, values = result
    if isinstance(K, (bool, np.bool_)) or not isinstance(K, (int, np.integer)) or not 1 <= int(K) <= MAX_K:
        raise ValueError(f"K must be an integer in 1..{MAX_K}")
    K = int(K)
    w = np.asarray(values)
    limit = min(2 * K, MAX_WEIGHTS)
    if w.ndim != 1 or not 1 <= w.size <= limit:
        raise ValueError(f"weights must be a 1-D sequence of 1..min(2K, {MAX_WEIGHTS}) = {limit} numbers, "
                         f"got shape {w.shape}")
    if w.dtype.kind in "iu":
        if any(abs(int(x)) > 2 ** 53 for x in w.tolist()):
            raise ValueError("integer weights must be at most 2**53")
    elif w.dtype.kind != "f":
        raise ValueError(f"weights must be integers or floats, got dtype {w.dtype}")
    w = w.astype(np.float64)
    if not np.isfinite(w).all():
        raise ValueError("weights must be finite")
    if (w < 0).any():
        raise ValueError("weights must be non-negative")
    top = float(w.max())
    if top <= 0:
        raise ValueError("weights must not all be zero")
    if ((w > 0) & (w < MIN_RELATIVE * top)).any():
        raise ValueError("a nonzero weight is below 2**-60 of the largest; set it to exactly 0")
    return K, w


def encode(w) -> str:
    """Weights as base64 of little-endian binary64, exact and about 11 bytes each."""
    return base64.b64encode(np.asarray(w, dtype="<f8").tobytes()).decode("ascii")


def decode(text) -> np.ndarray:
    return np.frombuffer(base64.b64decode(text), dtype="<f8").astype(np.float64)


def _integers(values):
    """Exact binary64 values as integers over one common power-of-two denominator."""
    ratios = [v.as_integer_ratio() for v in np.asarray(values, dtype=np.float64).tolist()]
    scale = max(d for _, d in ratios)
    return [n * (scale // d) for n, d in ratios]


def _convolve(a, b):
    """Exact convolution of non-negative integer lists by Kronecker substitution."""
    width = (max(a).bit_length() + max(b).bit_length() + min(len(a), len(b)).bit_length()) // 8 + 2

    def pack(xs):
        return int.from_bytes(b"".join(x.to_bytes(width, "little") for x in xs), "little")
    product = (pack(a) * pack(b)).to_bytes(width * (len(a) + len(b)), "little")
    return [int.from_bytes(product[i * width:(i + 1) * width], "little") for i in range(len(a) + len(b) - 1)]


def correlations(K, w):
    """For integers w, the integers sum_i w_i w_{i+k} for k = 0..K, zero once k reaches len(w)."""
    n = len(w)
    full = _convolve(w, w[::-1])
    return [full[n - 1 + k] if k < n else 0 for k in range(K + 1)]


def bound(K, weights) -> Fraction:
    w = _integers(weights)
    return Fraction(K * min(correlations(K, w)), sum(w) ** 2)


# --- behavior readout -------------------------------------------------------
# The score is one number. These describe the SHAPE of the step function instead, so two
# functions with the same bound can land in different cells of the diversity map.

_BEHAVIOR_ZERO = {"b_dispersion": 0.0, "b_boundary_frac": 0.0, "b_contact": 0.0,
                  "b_offset": 0.0, "b_concentration": 0.0, "b_structure": 0.0}


def _behavior(K, weights):
    """Six bounded descriptors: weight spread, mass near the ends, empty cells, centre of mass, mass in the
    heaviest cell (the near-atom of the best constructions), and resolution. Never raises."""
    try:
        a = np.asarray(weights, dtype=np.float64)
        a = a / a.max()
        mass = float(a.sum())
        if a.size == 0 or mass <= 0 or not np.isfinite(a).all():
            return dict(_BEHAVIOR_ZERO)
        n = a.size
        x = (np.arange(n) + 0.5) / n
        live = a[a > 0]
        spread = float(live.std() / live.mean())
        ends = float(a[(x < 0.1) | (x > 0.9)].sum() / mass)
        offset = abs(float((a * x).sum() / mass) - 0.5) / 0.5
        out = {"b_dispersion": spread / (1.0 + spread), "b_boundary_frac": ends,
               "b_contact": float((a == 0).mean()), "b_offset": min(offset, 1.0),
               "b_concentration": float(a.max() / mass),
               "b_structure": math.log(K) / math.log(MAX_K)}
        return {k: (float(v) if np.isfinite(v) else 0.0) for k, v in out.items()}
    except (ValueError, FloatingPointError, ZeroDivisionError, OverflowError):
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
    if not hasattr(program, "step_weights"):
        raise AttributeError("missing step_weights")
    start_time = time.time()
    K, w = _construction(program.step_weights())
    eval_time = time.time() - start_time
    value = bound(K, w)
    combined = float(value / Fraction(REFERENCE))
    cases = {"lower_bound": float(value), "K": float(K), "steps": float(len(w)),
             "combined_score": combined, "validity": 1.0}
    cases.update(_behavior(K, w))
    witness = {
        "schema": "sciel-full-output/1",
        "source_sha256": source_sha, "evaluator_sha256": evaluator_sha,
        "invocation_id": invocation_id, "stage": stage, "confirmation": None,
        "input_sha256": None, "seed_sha256": None,
        "output": {"K": K, "weights_float64_b64": encode(w), "bound": [value.numerator, value.denominator]},
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
        "cases": {"combined_score": 0.0, "lower_bound": 0.0, "K": 0.0, "steps": 0.0,
                  "validity": 0.0, **_BEHAVIOR_ZERO},
        "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]},
    }
