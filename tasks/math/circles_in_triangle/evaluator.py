"""Evaluator for N equal circles of largest radius in the isosceles right triangle with legs 1, N = 150..299;
smoke/full/confirm on the sciel contract.

The best known packings are Eckard Specht's on Packomania (https://www.packomania.com/crt/crt.html, last updated
18 Mar 2011), except that local search from the published centres (maximizing on the active contacts, then shaking
and re-maximizing for two minutes per N) beats 125 of them by 1e-12 to 1.8e-4 relative; those packings
(assets/crt_best.py) are the bar.
For each N the candidate gives N centres in the triangle with sides x = 0, y = 0 and x + y = 1; the radius is the
largest r with every centre at distance >= r from the sides and every pair at distance >= 2r. It is computed exactly:
binary64 centres are rationals over one power of two, the squared side and pair distances are integers over its
square (the hypotenuse's squared distance (1 - x - y)^2 / 2 included), and only the square root of the exact minimum
is rounded. Each N scores radius / best radius and combined_score is their mean, so a score above 1 means some
packing beats the best known. n_improved counts the N beaten by more than 1e-12 relative, compared exactly: rounding
the published 30-digit centres to binary64 moves their radius by at most 3e-15 relative and a fresh maximizing pass
on a bar packing gains about 1e-13 or less, so anything above the margin is a new packing, not rounding or polish. n_beats_published counts the N
beating Packomania's own radius by the same margin.
"""

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

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

ASSETS = Path(__file__).resolve().with_name("assets")
DATA = load_candidate(str(ASSETS / "crt_data.py"), "crt_data_evaluator")
BEST = load_candidate(str(ASSETS / "crt_best.py"), "crt_best_evaluator")
SIZES = tuple(range(150, 300))
PUBLISHED_R2 = {n: Fraction(DATA.RADIUS[n]) ** 2 for n in SIZES}
REFERENCE_R2 = {n: Fraction(BEST.R2[n]) if n in BEST.R2 else PUBLISHED_R2[n] for n in SIZES}
REFERENCE = {n: math.sqrt(float(REFERENCE_R2[n])) for n in SIZES}
MARGIN = Fraction(1, 10 ** 12)


def _load(program_path, module_name="program"):
    return load_candidate(program_path, module_name)


def radius2(p) -> Fraction:
    """The squared radius of the (n, 2) binary64 centres, exactly. Raises ValueError when a centre is outside the
    closed triangle or the radius is zero."""
    ratios = [v.as_integer_ratio() for v in p.ravel().tolist()]
    d = max(q for _, q in ratios)
    z = [a * (d // q) for a, q in ratios]
    xs, ys = z[0::2], z[1::2]
    if min(xs) < 0 or min(ys) < 0 or max(x + y for x, y in zip(xs, ys)) > d:
        raise ValueError("a centre is outside the triangle")
    best = min(min(4 * x * x, 4 * y * y, 2 * (d - x - y) ** 2) for x, y in zip(xs, ys))
    # With every coordinate in [0, 1], each floating squared distance is within a relative 4e-16 of the exact one, so
    # the closest pair is among those within a relative 1e-9 of the floating minimum; they are compared exactly.
    sq = ((p[:, None, :] - p[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(sq, np.inf)
    i, j = np.nonzero(np.triu(sq <= sq.min() * (1 + 1e-9) + 1e-300, 1))
    for a, b in zip(i.tolist(), j.tolist()):
        best = min(best, (xs[a] - xs[b]) ** 2 + (ys[a] - ys[b]) ** 2)
    if best == 0:
        raise ValueError("two centres coincide or a centre is on a side")
    return Fraction(best, 4 * d * d)


def _configurations(value):
    if not isinstance(value, dict):
        raise ValueError("packings() must return a dict {N: list of N (x, y)}")
    out = {}
    for n in SIZES:
        if n not in value:
            raise ValueError(f"missing N={n}")
        p = np.array(value[n], dtype=np.float64)
        if p.shape != (n, 2) or not np.isfinite(p).all():
            raise ValueError(f"N={n}: expected {n} finite (x, y) pairs, got shape {p.shape}")
        out[n] = p
    return out


# --- behavior readout -------------------------------------------------------
# The score is one mean. These describe WHERE the packings stand against the table, so runs that win at different
# N land in different cells of the diversity map.

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
               "b_contact": squash(r[n < 200].mean() - 1), "b_offset": squash(r[(n >= 200) & (n < 250)].mean() - 1),
               "b_concentration": squash(r[n >= 250].mean() - 1), "b_structure": float((np.abs(r - 1) < 1e-9).mean())}
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
    if not hasattr(program, "packings"):
        raise AttributeError("missing packings")
    start_time = time.time()
    configs = _configurations(program.packings())
    eval_time = time.time() - start_time
    radii, ratios, improved, beats_published = {}, {}, [], 0
    for n, p in configs.items():
        try:
            exact = radius2(p)
        except Exception as e:
            raise ValueError(f"N={n}: {e}") from None
        radii[n] = math.sqrt(float(exact))
        ratios[n] = radii[n] / REFERENCE[n]
        if exact > REFERENCE_R2[n] * (1 + MARGIN) ** 2:
            improved.append(n)
        beats_published += exact > PUBLISHED_R2[n] * (1 + MARGIN) ** 2
    best = max(SIZES, key=lambda n: ratios[n])
    combined = float(np.mean([ratios[n] for n in SIZES]))
    cases = {"combined_score": combined, "n_improved": float(len(improved)), "best_gain_rel": ratios[best] - 1.0,
             "best_gain_n": float(best), "n_beats_published": float(beats_published), "validity": 1.0}
    cases.update(_behavior(ratios))
    witness = {
        "schema": "sciel-full-output/1",
        "source_sha256": source_sha, "evaluator_sha256": evaluator_sha,
        "invocation_id": invocation_id, "stage": stage, "confirmation": None,
        "input_sha256": None, "seed_sha256": None,
        # Base64 of the little-endian binary64 centres keeps the witness exact and under the 1 MiB artifact limit.
        "output": {"packings": {str(n): base64.b64encode(configs[n].astype("<f8").tobytes()).decode() for n in SIZES},
                   "radius": {str(n): radii[n] for n in SIZES}, "improved": improved},
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
        "cases": {"combined_score": 0.0, "n_improved": 0.0, "best_gain_rel": 0.0, "best_gain_n": 0.0,
                  "n_beats_published": 0.0, "validity": 0.0, **_BEHAVIOR_ZERO},
        "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]},
    }
