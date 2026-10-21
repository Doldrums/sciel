"""Evaluator for coverings of the sphere by N equal caps, N = 4..130; smoke/full/confirm on the sciel contract.

The best known coverings are Hardin, Sloane and Smith's of 1994 (the table Henry Cohn keeps at
https://cohn.mit.edu/sloane#cov). For each N the candidate gives N vectors; the covering radius is the largest
angular radius of a hull face's circumcircle, valid only when the origin is strictly inside the hull. Each N scores
its published radius over the candidate's, and combined_score is their mean, so a score above 1 means some covering
is better than its published one; n_improved counts the N beaten by more than 1e-6 degrees, ten times the table's
rounding.
"""

from sciel.candidate_modules import load_candidate
import hashlib
import json
from pathlib import Path
import time
import traceback
import uuid

import numpy as np
from scipy.spatial import ConvexHull

DATA = load_candidate(str(Path(__file__).resolve().with_name("assets") / "caps_data.py"), "caps_data_evaluator")
SIZES = tuple(range(4, 131))
REFERENCE = {n: float(DATA.RADIUS_DEG[n]) for n in SIZES}
MARGIN_DEG = 1e-6


def _load(program_path, module_name="program"):
    return load_candidate(program_path, module_name)


def radius(points) -> float:
    """Covering radius in degrees of the normalized points, or ValueError if they do not cover the sphere."""
    p = np.asarray(points, dtype=np.float64)
    p = p / np.linalg.norm(p, axis=1, keepdims=True)
    hull = ConvexHull(p)
    offsets = hull.equations[:, 3]
    if not (offsets < -1e-12).all():
        raise ValueError("the points lie in a closed hemisphere")
    return float(np.degrees(np.arccos(np.clip(-offsets, -1.0, 1.0))).max())


def _configurations(value):
    if not isinstance(value, dict):
        raise ValueError("coverings() must return a dict {N: list of N (x, y, z)}")
    out = {}
    for n in SIZES:
        if n not in value:
            raise ValueError(f"missing N={n}")
        p = np.array(value[n], dtype=np.float64)
        if p.shape != (n, 3) or not np.isfinite(p).all():
            raise ValueError(f"N={n}: expected {n} finite (x, y, z) triples, got shape {p.shape}")
        if (np.linalg.norm(p, axis=1) == 0).any():
            raise ValueError(f"N={n}: a zero vector")
        out[n] = p
    return out


# --- behavior readout -------------------------------------------------------
# The score is one mean. These describe WHERE the configurations stand against the table, so runs that win at
# different N land in different cells of the diversity map.

_BEHAVIOR_ZERO = {"b_dispersion": 0.0, "b_boundary_frac": 0.0, "b_contact": 0.0,
                  "b_offset": 0.0, "b_concentration": 0.0, "b_structure": 0.0}


def _behavior(ratios):
    """Six bounded descriptors of the per-N ratios: their spread, share at or above the table, mean gain at small,
    middle and large N, and the share of N within 1e-9 of the table. Never raises."""
    try:
        n = np.array(SIZES)
        r = np.array([ratios[k] for k in SIZES])
        squash = lambda x: float(0.5 + 0.5 * np.tanh(x * 1e4))
        out = {"b_dispersion": float(min(r.std() * 1e4, 1.0)), "b_boundary_frac": float((r >= 1.0).mean()),
               "b_contact": squash(r[n < 40].mean() - 1), "b_offset": squash(r[(n >= 40) & (n < 90)].mean() - 1),
               "b_concentration": squash(r[n >= 90].mean() - 1), "b_structure": float((np.abs(r - 1) < 1e-9).mean())}
        return {k: (min(max(v, 0.0), 1.0) if np.isfinite(v) else 0.0) for k, v in out.items()}
    except (ValueError, FloatingPointError, ZeroDivisionError):
        return dict(_BEHAVIOR_ZERO)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _evaluate(program_path, stage):
    source_sha = _source_hash(program_path)
    evaluator_sha = _source_hash(__file__)
    invocation_id = uuid.uuid4().hex
    program = _load(program_path, "program_" + stage)
    if not hasattr(program, "coverings"):
        raise AttributeError("missing coverings")
    start_time = time.time()
    configs = _configurations(program.coverings())
    eval_time = time.time() - start_time
    radii, ratios = {}, {}
    for n, p in configs.items():
        try:
            radii[n] = radius(p)
        except Exception as e:
            raise ValueError(f"N={n}: {e}") from None
        ratios[n] = REFERENCE[n] / radii[n]
    gains = {n: REFERENCE[n] - radii[n] for n in SIZES}
    improved = sorted(n for n in SIZES if gains[n] > MARGIN_DEG)
    best = max(SIZES, key=lambda n: gains[n])
    combined = float(np.mean([ratios[n] for n in SIZES]))
    cases = {"combined_score": combined, "n_improved": float(len(improved)), "best_gain_deg": float(gains[best]),
             "best_gain_n": float(best), "validity": 1.0}
    cases.update(_behavior(ratios))
    witness = {
        "schema": "sciel-full-output/1",
        "source_sha256": source_sha, "evaluator_sha256": evaluator_sha,
        "invocation_id": invocation_id, "stage": stage, "confirmation": None,
        "input_sha256": None, "seed_sha256": None,
        "output": {"coverings": {str(n): configs[n].tolist() for n in SIZES},
                   "radius_deg": {str(n): radii[n] for n in SIZES}, "improved": improved},
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
    """Re-run the program n times and keep the worst result; a record should survive its own repetition."""
    n = max(1, int(n or 1))
    runs = [full(program_path) for _ in range(n)]
    failures = [r for r in runs if not r.get("ok")]
    selected = (next(i for i, result in enumerate(runs) if not result.get("ok"))
                if failures else min(range(n), key=lambda i: runs[i]["score"]))
    worst = dict(runs[selected])
    worst["cases"] = {**worst["cases"], "n_trials": float(n)}
    worst["artifacts"] = dict(worst.get("artifacts", {}))
    selection = {"trials": n, "selected_trial": selected, "selection": "first_failure_else_lowest_score"}
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
        "cases": {"combined_score": 0.0, "n_improved": 0.0, "best_gain_deg": 0.0, "best_gain_n": 0.0,
                  "validity": 0.0, **_BEHAVIOR_ZERO},
        "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]},
    }
