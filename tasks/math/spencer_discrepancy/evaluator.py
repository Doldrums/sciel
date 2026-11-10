"""Evaluator for Spencer's discrepancy constant C_10c from below; smoke/full/confirm on the sciel contract.

C_10c is the least K with disc(A) <= K sqrt(n) for every n and every A in [-1, 1]^{n x n}, where disc(A) is the
minimum over sign vectors x of ||Ax||_inf. A candidate is one n x n matrix with rational entries rows / den. Its
discrepancy is computed exactly in integers by enumerating all 2^(n-1) sign vectors with x_1 = 1 (assets/disc.py),
and disc / sqrt(n) is the lower bound; the score is its ratio to 5 / sqrt(8).
"""

from sciel.candidate_modules import load_candidate
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
import uuid

import numpy as np

_disc = load_candidate(str(Path(__file__).resolve().parent / "assets" / "disc.py"), "disc")

# Y. H.'s Hamming-code matrix ([H2026]): 5 / sqrt(8).
REFERENCE = 5 / math.sqrt(8)


def _load(program_path, module_name="program"):
    return load_candidate(program_path, module_name)


# --- behavior readout -------------------------------------------------------
# The score is one number. These describe the SHAPE of the matrix instead, so two matrices with the same bound can
# land in different cells of the diversity map.

_BEHAVIOR_ZERO = {"b_dispersion": 0.0, "b_boundary_frac": 0.0, "b_contact": 0.0,
                  "b_offset": 0.0, "b_concentration": 0.0, "b_structure": 0.0}


def _behavior(rows, den, num):
    """Six bounded descriptors: n over the cap, share of entries at +-1, share of zero entries, mean |entry|, the
    discrepancy over n (how close the matrix is to the trivial n), and the share of distinct rows up to sign. Never
    raises."""
    try:
        n = rows.shape[0]
        a = np.abs(rows)
        distinct = {tuple(r) if r[np.flatnonzero(r)[0]] > 0 else tuple(-r) for r in rows if np.any(r)}
        out = {"b_dispersion": n / _disc.MAX_N, "b_boundary_frac": float((a == den).mean()),
               "b_contact": float((a == 0).mean()), "b_offset": float(a.mean()) / den,
               "b_concentration": num / den / n, "b_structure": len(distinct) / n}
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
    rows, den = _disc.check(raw)
    n = rows.shape[0]
    num = int(_disc.disc_numerator(rows))
    lower = num / den / math.sqrt(n)
    combined = lower / REFERENCE
    output = {"rows": rows.tolist(), "den": den}
    cases = {"lower_bound": lower, "combined_score": combined, "validity": 1.0,
             "dimension": float(n), "discrepancy": num / den}
    cases.update(_behavior(rows, den, num))
    witness = {
        "schema": "sciel-full-output/1",
        "source_sha256": source_sha, "evaluator_sha256": evaluator_sha,
        "invocation_id": invocation_id, "stage": stage, "confirmation": None,
        "input_sha256": None, "seed_sha256": None,
        "output": {**output, "discrepancy": f"{num}/{den}"},
    }
    artifacts = {"eval_time": f"{eval_time:.4f}", "discrepancy": f"{num}/{den}", "n": str(n),
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
        "cases": {"combined_score": 0.0, "lower_bound": 0.0, "validity": 0.0, "dimension": 0.0, "discrepancy": 0.0,
                  **_BEHAVIOR_ZERO},
        "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]},
    }
