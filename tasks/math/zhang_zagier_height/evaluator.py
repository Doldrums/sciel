"""Evaluator for the essential minimum C_82 of the Zhang-Zagier height from above; smoke/full/confirm on the sciel
contract.

C_82 is the supremum of H such that only finitely many algebraic numbers a have h(a) + h(1 - a) <= H. Doche's
limit-point construction (JTNB 13, 2001) turns integer polynomials R_j (monic), S_i with multiplicities m_j and
exponents q_i into the bound

    C_82 <= B = (1/D) [ sum_j m_j log M(R_j o T) + int_0^1 max(0, log|A(T(u))| - log|Q(T(u))|) dt ],

T(z) = z(1 - z), Q = prod R_j^m_j, A = prod S_i^q_i, deg A < D = deg Q (assets/doche.py, Pinho's certificate format).
It is scored as a certified upper bound: every root of every R_j o T and S_i o T is enclosed in a disc that holds
exactly one root, numerator and denominator discs are disjoint, the Mahler part is summed from the enclosures and the
integral is bounded above on an adaptive dyadic tiling with outward margins (Patankar's checker scheme).
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

_doche = load_candidate(str(Path(__file__).resolve().parent / "assets" / "doche.py"), "doche")

# Patankar's [Pat26c] (PR 280), bounded by this evaluator as 0.2520591385264467 (his Arb checker: 0.25205913852600);
# rounded up so that it scores at least 1.
REFERENCE = mpmath.mpf("0.2520591385265")


def _load(program_path, module_name="program"):
    return load_candidate(program_path, module_name)


# --- behavior readout -------------------------------------------------------
# The score is one number. These describe the SHAPE of the construction instead, so two constructions with the same
# bound can land in different cells of the diversity map.

_BEHAVIOR_ZERO = {"b_dispersion": 0.0, "b_boundary_frac": 0.0, "b_contact": 0.0,
                  "b_offset": 0.0, "b_concentration": 0.0, "b_structure": 0.0}


def _behavior(den, num, mahler, integral):
    """Six bounded descriptors: mean denominator degree over the cap, share of the integral in the bound (x 1000),
    deg A / D, share of the largest multiplicity, largest numerator degree over 64, numerator count over the cap.
    Never raises."""
    try:
        D = sum((len(p) - 1) * m for p, m in den)
        out = {"b_dispersion": float(np.mean([len(p) - 1 for p, _ in den])) / _doche.MAX_DEGREE,
               "b_boundary_frac": min(1000 * integral / float(mahler + integral), 1.0),
               "b_contact": float(sum((len(p) - 1) * q for p, q in num) / D),
               "b_offset": max(m for _, m in den) / sum(m for _, m in den),
               "b_concentration": max((len(p) - 1 for p, _ in num), default=0) / 64,
               "b_structure": len(num) / _doche.MAX_NUMERATORS}
        return {k: (min(max(float(v), 0.0), 1.0) if math.isfinite(v) else 0.0) for k, v in out.items()}
    except (ValueError, ZeroDivisionError, IndexError, TypeError):
        return dict(_BEHAVIOR_ZERO)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _down(x):
    f = float(x)
    return float(np.nextafter(f, -np.inf)) if mpmath.mpf(f) > x else f


def _evaluate(program_path, stage):
    source_sha = _source_hash(program_path)
    evaluator_sha = _source_hash(__file__)
    invocation_id = uuid.uuid4().hex
    program = _load(program_path, "program_" + stage)
    if not hasattr(program, "certificate"):
        raise AttributeError("missing certificate")
    start_time = time.time()
    raw = program.certificate()
    eval_time = time.time() - start_time
    den, num = _doche.canonical(raw)
    candidate = {"denominators": [{"coeffs": p, "multiplicity": m} for p, m in den],
                 "numerators": [{"coeffs": p, "exponent": [q.numerator, q.denominator]} for p, q in num]}
    upper, mahler, integral, panels = _doche.certify(candidate)
    with mpmath.workdps(40):
        combined = _down(REFERENCE / upper)
        digits = mpmath.nstr(upper, 20, strip_zeros=False)
    D = sum((len(p) - 1) * m for p, m in den)
    cases = {"upper_bound": float(upper), "combined_score": combined, "validity": 1.0,
             "denominators": float(len(den)), "numerators": float(len(num)),
             "degree_ratio": float(sum((len(p) - 1) * q for p, q in num) / D)}
    cases.update(_behavior(den, num, mahler, integral))
    witness = {
        "schema": "sciel-full-output/1",
        "source_sha256": source_sha, "evaluator_sha256": evaluator_sha,
        "invocation_id": invocation_id, "stage": stage, "confirmation": None,
        "input_sha256": None, "seed_sha256": None,
        "output": {**candidate, "upper_bound": digits},
    }
    artifacts = {"eval_time": f"{eval_time:.4f}", "upper_bound": digits,
                 "mahler_part": mpmath.nstr(mahler, 20), "integral_part": f"{integral:.12e}",
                 "panels": str(panels), "full_output": _canonical(witness)}
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
        "cases": {"combined_score": 0.0, "upper_bound": 0.0, "validity": 0.0, "denominators": 0.0,
                  "numerators": 0.0, "degree_ratio": 0.0, **_BEHAVIOR_ZERO},
        "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]},
    }
