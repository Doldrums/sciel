"""Evaluator for the centred Hardy-Littlewood maximal constant C_47 over squares in the plane from below;
smoke/full/confirm on the sciel contract.

C_47 is the best c with |{M f >= a}| <= (c / a) ||f||_1, M the centred maximal operator over axis-parallel squares. A
candidate is a periodic atomic measure: a cell [0, Px) x [0, Py) of rational atoms repeated over the lattice. Every
box with sides on atom coordinates and mass mu > 4 rho^2 gives a rectangle of centres whose square of radius rho holds
that box, so M mu > 1 there; the union area over the cell mass is a lower bound for C_47 by Lin's truncation and
smearing (Tomar's Lemma 6.2). It is scored exactly: Tomar's enumeration (check_exact2.py: box side at most 1396/125,
rho rounded down on the 2^-40 grid) in integer arithmetic, the union area as a fraction (assets/hl.py).
"""

from sciel.candidate_modules import load_candidate
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path
import time
import traceback
import uuid

import numpy as np

_hl = load_candidate(str(Path(__file__).resolve().parent / "assets" / "hl.py"), "hl")

# The 59-atom half cell of PR 321 (Pinho, after Tomar's hole-filling), reproduced exactly by this evaluator and by
# Tomar's check_exact2.py.
REFERENCE = Fraction(215956002657486428307163448091951082893881, 126800499112387565339223240212480000000000)


def _load(program_path, module_name="program"):
    return load_candidate(program_path, module_name)


# --- behavior readout -------------------------------------------------------
# The score is one number. These describe the SHAPE of the measure instead, so two measures with the same bound can
# land in different cells of the diversity map.

_BEHAVIOR_ZERO = {"b_dispersion": 0.0, "b_boundary_frac": 0.0, "b_contact": 0.0,
                  "b_offset": 0.0, "b_concentration": 0.0, "b_structure": 0.0}


def _behavior(px, py, pts, area, nrect):
    """Six bounded descriptors: atoms over the cap, uncovered share of the cell (x 50), rectangles per atom over 2000,
    aspect Py / (Px + Py), the heaviest atom's share of the mass, and the share of atoms lighter than a tenth of the
    heaviest. Never raises."""
    try:
        m = [float(p[2]) for p in pts]
        cell = float(px * py)
        out = {"b_dispersion": len(pts) / _hl.MAX_ATOMS, "b_boundary_frac": 50 * (1 - float(area) / cell),
               "b_contact": nrect / len(pts) / 2000, "b_offset": float(py / (px + py)),
               "b_concentration": max(m) / sum(m), "b_structure": sum(1 for v in m if v < 0.1 * max(m)) / len(m)}
        return {k: (min(max(float(v), 0.0), 1.0) if math.isfinite(v) else 0.0) for k, v in out.items()}
    except (ValueError, ZeroDivisionError, IndexError, TypeError):
        return dict(_BEHAVIOR_ZERO)


def _down(q):
    f = float(q)
    return float(np.nextafter(f, -np.inf)) if Fraction(f) > q else f


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
    px, py, pts = _hl.canonical(raw)
    output = _hl.make(px, py, pts)
    score_start = time.time()
    ratio, area, mass, nrect = _hl.exact(output)
    score_time = time.time() - score_start
    digits = "%d/%d" % (ratio.numerator, ratio.denominator)
    cases = {"lower_bound": _down(ratio), "combined_score": _down(ratio / REFERENCE), "validity": 1.0,
             "atoms": float(len(pts)), "rectangles": float(nrect)}
    cases.update(_behavior(px, py, pts, area, nrect))
    witness = {
        "schema": "sciel-full-output/1",
        "source_sha256": source_sha, "evaluator_sha256": evaluator_sha,
        "invocation_id": invocation_id, "stage": stage, "confirmation": None,
        "input_sha256": None, "seed_sha256": None,
        "output": {**output, "lower_bound": digits},
    }
    artifacts = {"eval_time": f"{eval_time:.4f}", "score_time": f"{score_time:.4f}", "lower_bound": digits,
                 "lower_bound_float": repr(float(ratio)), "cell_mass": str(mass), "rectangles": str(nrect),
                 "full_output": _canonical(witness)}
    if source_sha != _source_hash(program_path) or evaluator_sha != _source_hash(__file__):
        raise ValueError("source or evaluator changed during invocation")
    return {"ok": True, "score": cases["combined_score"], "cases": cases, "artifacts": artifacts}


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
        "cases": {"combined_score": 0.0, "lower_bound": 0.0, "validity": 0.0, "atoms": 0.0, "rectangles": 0.0,
                  **_BEHAVIOR_ZERO},
        "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]},
    }
