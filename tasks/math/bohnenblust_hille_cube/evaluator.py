"""Evaluator for the Bohnenblust-Hille constant on the Boolean cube C_26a from below; smoke/full/confirm on the sciel
contract.

C_26a is the sup over d of the least C with ||f^||_{p_d} <= C ||f||_inf for every f on {-1, 1}^n of degree at most
d, p_d = 2d / (d + 1). A candidate is one f with integer Fourier coefficients on n <= 22 variables. Its sup norm is
exact (an integer Walsh transform over all 2^n points), the l_p norm is enclosed by interval powers at 50 digits over
the distinct coefficient sizes (assets/bh.py), and the lower end of R = ||f^||_p / ||f||_inf is the bound; the score
is its ratio to 2, the table's value (the limit of the address functions).
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

_bh = load_candidate(str(Path(__file__).resolve().parent / "assets" / "bh.py"), "bh")

# The table's lower bound ([ADGP2025]): the address functions reach 2^((d - 1) / d) -> 2.
REFERENCE = 2


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


def _behavior(a, n, d, sup):
    """Six bounded descriptors: n over the cap, share of points where |f| reaches its sup, share of points where f
    vanishes, the degree over n, share of nonzero coefficients, and distinct coefficient sizes per coefficient. Never
    raises."""
    try:
        f = np.abs(_bh.wht(a))
        nz = a[a != 0]
        out = {"b_dispersion": n / _bh.MAX_N, "b_boundary_frac": float((f == sup).mean()),
               "b_contact": float((f == 0).mean()), "b_offset": d / n,
               "b_concentration": len(nz) / len(a), "b_structure": len(np.unique(np.abs(nz))) / len(nz)}
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
    candidate = _bh.load(raw) if isinstance(raw, dict) and set(raw) == {"n", "terms"} else raw
    a = _bh.array(candidate)
    n = int(candidate["n"])
    lower, d, sup = _bh.certify(candidate)
    with mpmath.workdps(50):
        combined = _down(lower / REFERENCE)
        digits = mpmath.nstr(lower, 40, strip_zeros=False)
    output = _bh.make(a)
    cases = {"lower_bound": _down(lower), "combined_score": combined, "validity": 1.0,
             "variables": float(n), "degree": float(d)}
    cases.update(_behavior(a, n, d, sup))
    witness = {
        "schema": "sciel-full-output/1",
        "source_sha256": source_sha, "evaluator_sha256": evaluator_sha,
        "invocation_id": invocation_id, "stage": stage, "confirmation": None,
        "input_sha256": None, "seed_sha256": None,
        "output": {**output, "degree": d, "sup": sup, "lower_bound": digits},
    }
    artifacts = {"eval_time": f"{eval_time:.4f}", "lower_bound": digits, "degree": str(d), "sup": str(sup),
                 "n": str(n), "full_output": _canonical(witness)}
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
        "cases": {"combined_score": 0.0, "lower_bound": 0.0, "validity": 0.0, "variables": 0.0, "degree": 0.0,
                  **_BEHAVIOR_ZERO},
        "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]},
    }
