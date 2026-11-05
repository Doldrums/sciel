"""Evaluator for the tight alternating knot constant C_22b from above; smoke/full/confirm on the sciel contract.

C_22b is the largest b with L >= b C for every alternating knot (L ropelength at unit radius thickness, C crossing
number). A candidate is a four-strand superhelix: two coaxial helical axes of radius R and rise h per radian, each
carrying an outer strand of radius a(u) and an inner strand of radius i(u) on the opposite side, twisting N = P / Q
times per axis turn (Huh-Kim-Oh's superhelix with Li's unequal radii; R. Pinho, PR 324). Closed by PR 324's caps it is
T(2, 4 P n + 3), so C_22b <= (length per crossing) / thickness of the periodic body (assets/helix.py). It is scored
as a certified upper bound: the length by a per-cell second-order bound, the thickness as the least of a curvature
branch and bound, the bundle gap R - max r and a branch and bound on the squared self-distance, every bound from
derivative sup norms of the trigonometric polynomials, with absolute margins.
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

_helix = load_candidate(str(Path(__file__).resolve().parent / "assets" / "helix.py"), "helix")

# Pinho's PR 324 claim 7.090412 (his own body bound 7.0904111 after a 10001/10000 scale margin); this evaluator
# bounds the same construction by 7.0904084, so it scores just above 1.
REFERENCE = 7.090412


def _load(program_path, module_name="program"):
    return load_candidate(program_path, module_name)


# --- behavior readout -------------------------------------------------------
# The score is one number. These describe the SHAPE of the construction instead, so two constructions with the same
# bound can land in different cells of the diversity map.

_BEHAVIOR_ZERO = {"b_dispersion": 0.0, "b_boundary_frac": 0.0, "b_contact": 0.0,
                  "b_offset": 0.0, "b_concentration": 0.0, "b_structure": 0.0}


def _behavior(c, tau, binding):
    """Six bounded descriptors: N over 100, Q over its cap, inner over outer mean radius, R over h, modulation
    amplitude over mean radius (both strands), and which constraint binds (curvature 0, bundles 0.5, contact 1).
    Never raises."""
    try:
        mod = sum(sum(abs(x) for x in r[1:]) / r[0] for r in (c["outer"], c["inner"])) / 2
        out = {"b_dispersion": c["P"] / c["Q"] / 100, "b_boundary_frac": c["Q"] / _helix.MAX_Q,
               "b_contact": c["inner"][0] / c["outer"][0], "b_offset": c["R"] / c["h"],
               "b_concentration": mod, "b_structure": {"curvature": 0.0, "bundles": 0.5, "contact": 1.0}[binding]}
        return {k: (min(max(float(v), 0.0), 1.0) if math.isfinite(v) else 0.0) for k, v in out.items()}
    except (ValueError, ZeroDivisionError, IndexError, TypeError, KeyError):
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
    P, Q, R, h, outer, inner = _helix.canonical(raw)
    candidate = _helix.make(P, Q, R, h, outer, inner)
    upper, length, tau, binding = _helix.certify(candidate)
    combined = float(np.nextafter(REFERENCE / upper, -np.inf))
    cases = {"upper_bound": float(upper), "combined_score": combined, "validity": 1.0,
             "length_per_crossing": float(length), "thickness": float(tau)}
    cases.update(_behavior(candidate, tau, binding))
    witness = {
        "schema": "sciel-full-output/1",
        "source_sha256": source_sha, "evaluator_sha256": evaluator_sha,
        "invocation_id": invocation_id, "stage": stage, "confirmation": None,
        "input_sha256": None, "seed_sha256": None,
        "output": {**candidate, "upper_bound": repr(float(upper))},
    }
    artifacts = {"eval_time": f"{eval_time:.4f}", "upper_bound": repr(float(upper)), "binding": binding,
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
        "cases": {"combined_score": 0.0, "upper_bound": 0.0, "validity": 0.0, "length_per_crossing": 0.0,
                  "thickness": 0.0, **_BEHAVIOR_ZERO},
        "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]},
    }
