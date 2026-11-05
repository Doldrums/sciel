"""Evaluator for the Gyarmati-Hennecart-Ruzsa constant C_3a from below; smoke/full/confirm on the sciel contract.

C_3a is the largest exponent with arbitrarily large integer sets A, B, |A + B| << |A| and |A - B| >> |A + B|^C_3a. A
candidate is a digit mask M (finite, 0 in M) and two tilts x, y in (0, 1). Olafson's two-temperature cyclic quotient
(PR 148) turns it into C_3a >= 2 - sigma / rho, where rho is the entropy of the Gibbs law x^kappa(d) on M - M with
kappa(d) the cheapest witness a + b of d = a - b, and sigma = log P_+(y) - mu log y bounds the sums at the same
mean budget mu (assets/ghr.py). It is scored as a certified lower bound: the tilts are read as exact rationals, the
power sums are evaluated by interval Horner at 50 digits, and the lower end of the enclosed bound is rounded down.
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

_ghr = load_candidate(str(Path(__file__).resolve().parent / "assets" / "ghr.py"), "ghr")

# Olafson's <4, 5> mask cut at 80 (PR 148), enclosed by this evaluator as 1.21418118095053104745496921148485...;
# truncated so that it scores at least 1.
with mpmath.workdps(50):
    REFERENCE = mpmath.mpf("1.214181180950531047454969211484")


def _load(program_path, module_name="program"):
    return load_candidate(program_path, module_name)


# --- behavior readout -------------------------------------------------------
# The score is one number. These describe the SHAPE of the mask instead, so two masks with the same bound can land in
# different cells of the diversity map.

_BEHAVIOR_ZERO = {"b_dispersion": 0.0, "b_boundary_frac": 0.0, "b_contact": 0.0,
                  "b_offset": 0.0, "b_concentration": 0.0, "b_structure": 0.0}


def _behavior(candidate, rho, sigma):
    """Six bounded descriptors: largest element over the cap, share of gaps below it, share of differences that cost
    more than |d|, the difference tilt x, the entropy gap (rho - sigma) / rho, and the gcd-free count of generators
    (elements not sums of two smaller nonzero elements) over 16. Never raises."""
    try:
        mask = candidate["mask"]
        top = mask[-1]
        kap = _ghr.kappa(mask)
        present = set(mask)
        generators = [m for m in mask[1:] if not any(m - a in present for a in mask[1:] if a < m)]
        out = {"b_dispersion": top / _ghr.MAX_ELEMENT, "b_boundary_frac": 1 - len(mask) / (top + 1),
               "b_contact": sum(1 for d, c in kap.items() if c > abs(d)) / len(kap),
               "b_offset": float(candidate["x"]), "b_concentration": float((rho - sigma) / rho),
               "b_structure": min(len(generators) / 16, 1.0)}
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
    candidate = _ghr.canonical(raw)
    lower, rho, sigma = _ghr.certify(candidate)
    with mpmath.workdps(50):
        combined = _ghr.down(lower / REFERENCE)
        digits = mpmath.nstr(lower, 40, strip_zeros=False)
    output = {"mask": candidate["mask"],
              "x": [candidate["x"].numerator, candidate["x"].denominator],
              "y": [candidate["y"].numerator, candidate["y"].denominator]}
    cases = {"lower_bound": _ghr.down(lower), "combined_score": combined, "validity": 1.0,
             "mask_size": float(len(candidate["mask"])), "max_element": float(candidate["mask"][-1])}
    cases.update(_behavior(candidate, float(rho[0]), float(sigma[1])))
    witness = {
        "schema": "sciel-full-output/1",
        "source_sha256": source_sha, "evaluator_sha256": evaluator_sha,
        "invocation_id": invocation_id, "stage": stage, "confirmation": None,
        "input_sha256": None, "seed_sha256": None,
        "output": {**output, "lower_bound": digits},
    }
    artifacts = {"eval_time": f"{eval_time:.4f}", "lower_bound": digits,
                 "rho": mpmath.nstr(rho[0], 30), "sigma": mpmath.nstr(sigma[1], 30),
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
        "cases": {"combined_score": 0.0, "lower_bound": 0.0, "validity": 0.0, "mask_size": 0.0, "max_element": 0.0,
                  **_BEHAVIOR_ZERO},
        "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]},
    }
