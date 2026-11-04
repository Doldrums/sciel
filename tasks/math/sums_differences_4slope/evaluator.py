"""Evaluator for the four-slope sums-differences constant C_3c = SD({0,1,2,oo}; -1); smoke/full/confirm on the sciel
contract.

C_3c is the least C with H(X - Y) <= C max(H(X), H(Y), H(X + Y), H(X + 2Y)) for all discrete X, Y, so any pair gives
a lower bound. The candidate is a mixture of finite laws on Z^2 (assets/entropy.py): a share t_k of many independent
digits is drawn from law k, and the digits are spread in a large base so that every form's entropy adds over them.
Per digit the bound is sum_k t_k H_k(x - y) / max over the four denominator forms of sum_k t_k H_k(form). It is
scored as a certified lower bound: the image law of every form is exact in rationals, logarithms are enclosed in
mpmath interval arithmetic at 50 digits, and the lower end of the enclosed ratio is rounded down.
"""

from sciel.candidate_modules import load_candidate
import hashlib
import json
from pathlib import Path
import time
import traceback
import uuid

import mpmath
import numpy as np

_entropy = load_candidate(str(Path(__file__).resolve().parent / "assets" / "entropy.py"), "entropy")

FORMS = _entropy.FORMS_3C
# Jaswal's two-law certificate (PR 221), enclosed by this evaluator as 1.67489421130711622876434373706095716...;
# truncated so that it scores at least 1.
with mpmath.workdps(50):
    REFERENCE = mpmath.mpf("1.674894211307116228764343737060957")


def _load(program_path, module_name="program"):
    return load_candidate(program_path, module_name)


# --- behavior readout -------------------------------------------------------
# The score is one number. These describe the SHAPE of the construction instead, so two constructions with the same
# bound can land in different cells of the diversity map.

_BEHAVIOR_ZERO = {"b_dispersion": 0.0, "b_boundary_frac": 0.0, "b_contact": 0.0,
                  "b_offset": 0.0, "b_concentration": 0.0, "b_structure": 0.0}


def _behavior(candidate):
    """Six bounded descriptors: spread of the support (mean max-norm radius over 32), share of points below weight
    1e-6, slack of the second largest denominator form, share of the first law, largest weight, support size over
    the cap. Never raises."""
    try:
        laws, mix = _entropy.as_float(candidate)
        points = np.concatenate([p for p, _ in laws])
        weights = np.concatenate([w * t for (_, w), t in zip(laws, mix)])
        E = sum(t * _entropy.entropies_and_jacobian(p, w, FORMS)[0] for (p, w), t in zip(laws, mix))
        top = np.sort(E[1:])
        out = {"b_dispersion": min(float(np.abs(points - np.median(points, axis=0)).max(axis=1).mean()) / 32, 1.0),
               "b_boundary_frac": float((weights < 1e-6).mean()),
               "b_contact": min(float((top[-1] - top[-2]) / top[-1]) * 1e3, 1.0),
               "b_offset": float(mix[0]), "b_concentration": float(weights.max()),
               "b_structure": min(points.shape[0] / (2 * _entropy.MAX_POINTS), 1.0)}
        return {k: (float(v) if np.isfinite(v) else 0.0) for k, v in out.items()}
    except (ValueError, FloatingPointError, ZeroDivisionError, IndexError):
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
    if not hasattr(program, "law"):
        raise AttributeError("missing law")
    start_time = time.time()
    candidate = program.law()
    eval_time = time.time() - start_time
    candidate = _entropy.canonical(candidate)
    lower, entropies = _entropy.certify(candidate, FORMS)
    with mpmath.workdps(50):
        combined = _entropy.down(lower / REFERENCE)
        digits = mpmath.nstr(lower, 40, strip_zeros=False)
    cases = {"lower_bound": _entropy.down(lower), "combined_score": combined, "validity": 1.0,
             "points": float(sum(len(law["points"]) for law in candidate["laws"])),
             "laws": float(len(candidate["laws"]))}
    cases.update(_behavior(candidate))
    witness = {
        "schema": "sciel-full-output/1",
        "source_sha256": source_sha, "evaluator_sha256": evaluator_sha,
        "invocation_id": invocation_id, "stage": stage, "confirmation": None,
        "input_sha256": None, "seed_sha256": None,
        "output": {**candidate, "lower_bound": digits},
    }
    artifacts = {"eval_time": f"{eval_time:.4f}", "lower_bound": digits,
                 "entropies": _canonical([mpmath.nstr(h.a, 30) for h in entropies]),
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
        "cases": {"combined_score": 0.0, "lower_bound": 0.0, "validity": 0.0, "points": 0.0, "laws": 0.0,
                  **_BEHAVIOR_ZERO},
        "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]},
    }
