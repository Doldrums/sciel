"""Evaluator for the Fourier Entropy-Influence constant C_71 from below; smoke/full/confirm on the sciel contract.

C_71 is the least C with H(f^2) <= C Inf(f) for every Boolean f (Friedgut-Kalai). A candidate is the truth table of
one Boolean function f on n <= 22 variables. If f is balanced and its total influence I exceeds 1, the
O'Donnell-Tan amplification (composing f with itself) gives C_71 >= H / (I - 1) (assets/fei.py). It is scored as a
certified lower bound: the Walsh coefficients are exact integers, I is an exact fraction, H is enclosed by interval
logs at 50 digits over the distinct coefficient sizes, and the lower end of H / (I - 1) is rounded down.
"""

from sciel.candidate_modules import load_candidate
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
import uuid

import mpmath
import numpy as np

_fei = load_candidate(str(Path(__file__).resolve().parent / "assets" / "fei.py"), "fei")

# Numaro's n = 18 function ([Num2026], zenodo 21497769), enclosed by this evaluator as
# 6.52184571092304657565814397294857846836666...; truncated so that it scores at least 1.
with mpmath.workdps(50):
    REFERENCE = mpmath.mpf("6.521845710923046575658143972948578468366")


def _down(x):
    """A float no larger than the mpf x."""
    v = float(x)
    return v if mpmath.mpf(v) <= x else math.nextafter(v, -math.inf)


def _load(program_path, module_name="program"):
    return load_candidate(program_path, module_name)


# --- behavior readout -------------------------------------------------------
# The score is one number. These describe the SHAPE of the function instead, so two functions with the same bound can
# land in different cells of the diversity map.

_BEHAVIOR_ZERO = {"b_dispersion": 0.0, "b_boundary_frac": 0.0, "b_contact": 0.0,
                  "b_offset": 0.0, "b_concentration": 0.0, "b_structure": 0.0}


def _behavior(f, n):
    """Six bounded descriptors: n over the cap, share of points on the boundary (some neighbour has the other value),
    spectral weight on level 1, influence per variable, share of nonzero coefficients, and the share of coordinates
    along which f is monotone. Never raises."""
    try:
        c = _fei.wht(f)
        w = (c.astype(np.float64) / len(f)) ** 2
        level = _fei.popcounts(n)
        idx = np.arange(len(f))
        boundary = np.zeros(len(f), dtype=bool)
        monotone = 0
        for i in range(n):
            g = f[idx ^ (1 << i)]
            boundary |= g != f
            low = (idx >> i) & 1 == 0
            d = f[idx[low] | (1 << i)] - f[idx[low]]
            monotone += bool(np.all(d >= 0) or np.all(d <= 0))
        out = {"b_dispersion": n / _fei.MAX_N, "b_boundary_frac": float(boundary.mean()),
               "b_contact": float(w[level == 1].sum()), "b_offset": float((w * level).sum()) / n,
               "b_concentration": float(np.count_nonzero(c)) / len(f), "b_structure": monotone / n}
        return {k: (min(max(float(v), 0.0), 1.0) if math.isfinite(v) else 0.0) for k, v in out.items()}
    except (ValueError, ZeroDivisionError, IndexError, TypeError):
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
    if not hasattr(program, "candidate"):
        raise AttributeError("missing candidate")
    start_time = time.time()
    raw = program.candidate()
    eval_time = time.time() - start_time
    candidate = raw
    f = _fei.array(candidate)
    n = int(candidate["n"])
    lower, ent, infl = _fei.certify(candidate)
    with mpmath.workdps(50):
        combined = _down(lower / REFERENCE)
        digits = mpmath.nstr(lower, 40, strip_zeros=False)
    output = _fei.make(f)
    cases = {"lower_bound": _down(lower), "combined_score": combined, "validity": 1.0,
             "variables": float(n), "influence": float(infl)}
    cases.update(_behavior(f, n))
    witness = {
        "schema": "sciel-full-output/1",
        "source_sha256": source_sha, "evaluator_sha256": evaluator_sha,
        "invocation_id": invocation_id, "stage": stage, "confirmation": None,
        "input_sha256": None, "seed_sha256": None,
        "output": {**output, "lower_bound": digits},
    }
    artifacts = {"eval_time": f"{eval_time:.4f}", "lower_bound": digits,
                 "influence": f"{infl.numerator}/{infl.denominator}", "entropy": mpmath.nstr(ent.a, 30),
                 "full_output": _canonical(witness)}
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
        "cases": {"combined_score": 0.0, "lower_bound": 0.0, "validity": 0.0, "variables": 0.0, "influence": 0.0,
                  **_BEHAVIOR_ZERO},
        "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]},
    }
