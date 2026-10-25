"""Evaluator for N equal circles of largest radius in the circular quadrant {x >= 0, y >= 0, x^2 + y^2 <= 1}
(assets/ccq_data.py: SIZES); smoke/full/confirm on the sciel contract.

The best known packings are Packomania's quadrant table (https://www.packomania.com/ccq/ccq.html, D. A. Paterson,
E. Specht, D. W. Cantrell and C. Lopez Soto, 2009-2011), except that local search from the published centres
(maximizing on the active contacts, then shaking and re-maximizing for 90 s per N) beats many of them; those packings
(assets/ccq_best.py) are the bar.
For each N the candidate gives N centres in the quadrant; the radius is the largest r with every centre at distance
>= r from the axes and >= r inside the arc (sqrt(x^2 + y^2) <= 1 - r), and every pair at distance >= 2r. It is
computed exactly: binary64 centres are rationals over one power of two d, so x d, y d, (x^2 + y^2) d^2 and the
squared pair distances times d^2 are integers, and the radius is the least of a rational side distance, a half pair
distance sqrt(s) and an arc distance 1 - sqrt(q), each of the form a + b sqrt(c) with rationals a, b, c. Two such
numbers are compared exactly by squaring with their signs tracked (_sign3), never in floating point; only the score
rounds. Each N scores radius / best radius and combined_score is their mean, so a score above 1 means some packing
beats the best known. n_improved counts the N beaten by more than 1e-12 relative, compared exactly: rounding the
published 30-digit centres to binary64 moves their radius by at most 7e-15 relative and a fresh maximizing pass on a
bar packing gains nothing, so anything above the margin is a new packing, not rounding or polish. n_beats_published
counts the N beating Packomania's own radius by the same margin.
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
DATA = load_candidate(str(ASSETS / "ccq_data.py"), "ccq_data_evaluator")
BEST = load_candidate(str(ASSETS / "ccq_best.py"), "ccq_best_evaluator")


def _load(program_path, module_name="program"):
    return load_candidate(program_path, module_name)


def _sign(x):
    return (x > 0) - (x < 0)


def _pair(b, u, c, v):
    """The sign of b sqrt(u) + c sqrt(v) for rationals b, c and u, v >= 0, exactly."""
    s, t = _sign(b) * (u > 0), _sign(c) * (v > 0)
    if s == t or t == 0:
        return s
    if s == 0:
        return t
    d = b * b * u - c * c * v
    return s if d > 0 else t if d < 0 else 0


def _sign3(a, b, u, c, v):
    """The sign of a + b sqrt(u) + c sqrt(v), exactly: the sign of the roots, then, when a has the other sign, the
    sign of a^2 - (b sqrt(u) + c sqrt(v))^2 = a^2 - b^2 u - c^2 v - 2bc sqrt(uv)."""
    roots, sa = _pair(b, u, c, v), _sign(a)
    if roots == 0 or sa == 0 or sa == roots:
        return sa or roots
    d = _pair(a * a - b * b * u - c * c * v, 1, -2 * b * c, u * v)
    return sa if d > 0 else roots if d < 0 else 0


def _above(x, y):
    """Whether a + b sqrt(c) of x exceeds that of y, exactly."""
    return _sign3(x[0] - y[0], x[1], x[2], -y[1], y[2]) > 0


def _float(x):
    return float(x[0]) + float(x[1]) * math.sqrt(float(x[2]))


SIZES = tuple(DATA.SIZES)
MARGIN = 1 + Fraction(1, 10 ** 12)
# Radii are triples (a, b, c) for a + b sqrt(c); a candidate improves when it exceeds the bar times MARGIN.
PUBLISHED_ABOVE = {n: (Fraction(DATA.RADIUS[n]) * MARGIN, Fraction(0), Fraction(0)) for n in SIZES}
REFERENCE_EXACT = {n: tuple(Fraction(v) for v in BEST.R[n]) for n in SIZES}
REFERENCE_ABOVE = {n: (a * MARGIN, b * MARGIN, c) for n, (a, b, c) in REFERENCE_EXACT.items()}
REFERENCE = {n: _float(REFERENCE_EXACT[n]) for n in SIZES}


def radius_exact(p):
    """The radius of the (n, 2) binary64 centres exactly, as (a, b, c) for a + b sqrt(c). Raises ValueError when a
    centre is outside the closed quadrant or the radius is zero."""
    ratios = [v.as_integer_ratio() for v in p.ravel().tolist()]
    d = max(q for _, q in ratios)
    z = [a * (d // q) for a, q in ratios]
    xs, ys = z[0::2], z[1::2]
    side = min(min(xs), min(ys))
    q = max(x * x + y * y for x, y in zip(xs, ys))
    if side < 0 or q > d * d:
        raise ValueError("a centre is outside the quadrant")
    best = (Fraction(side, d), Fraction(0), Fraction(0))
    arc = (Fraction(1), Fraction(-1), Fraction(q, d * d))
    best = arc if _above(best, arc) else best
    # With every coordinate in [0, 1], each floating squared distance is within a relative 4e-16 of the exact one,
    # so the closest pair is among those within a relative 1e-9 of the floating minimum; they are compared exactly.
    sq = ((p[:, None, :] - p[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(sq, np.inf)
    i, j = np.nonzero(np.triu(sq <= sq.min() * (1 + 1e-9) + 1e-300, 1))
    pair = min((xs[u] - xs[v]) ** 2 + (ys[u] - ys[v]) ** 2 for u, v in zip(i.tolist(), j.tolist()))
    pair = (Fraction(0), Fraction(1), Fraction(pair, 4 * d * d))
    best = pair if _above(best, pair) else best
    if _sign3(best[0], best[1], best[2], 0, 0) <= 0:
        raise ValueError("two centres coincide or a centre is on the boundary")
    return best


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
    """Six bounded descriptors of the per-N ratios: their spread, share at or above the table, mean gain in the lower,
    middle and upper third of the N, and the share of N within 1e-9 of the table. Never raises."""
    try:
        r = np.array([ratios[k] for k in SIZES])
        low, mid, high = np.array_split(r, 3)
        squash = lambda x: float(0.5 + 0.5 * np.tanh(x * 1e4))
        out = {"b_dispersion": float(min(r.std() * 1e4, 1.0)), "b_boundary_frac": float((r >= 1.0).mean()),
               "b_contact": squash(low.mean() - 1), "b_offset": squash(mid.mean() - 1),
               "b_concentration": squash(high.mean() - 1), "b_structure": float((np.abs(r - 1) < 1e-9).mean())}
        return {k: (min(max(v, 0.0), 1.0) if np.isfinite(v) else 0.0) for k, v in out.items()}
    except (ValueError, FloatingPointError, ZeroDivisionError):
        return dict(_BEHAVIOR_ZERO)


def _feedback(ratios, improved):
    """radius / best - 1 for the N that moved, largest moves first: which N rose or fell, in one line."""
    moved = [n for n in SIZES if abs(ratios[n] - 1) > 1e-12]
    better = sorted((n for n in moved if ratios[n] > 1), key=lambda n: -ratios[n])
    worse = sorted((n for n in moved if ratios[n] < 1), key=lambda n: ratios[n])
    items = lambda ns: ", ".join(f"{n} {ratios[n] - 1:+.2e}" for n in ns[:8]) or "none"
    return (f"radius / best - 1: above the bar {items(better)}; below {items(worse)}; "
            f"{len(SIZES) - len(moved)} N at the bar; {len(improved)} N improved by more than 1e-12")


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
            exact = radius_exact(p)
        except Exception as e:
            raise ValueError(f"N={n}: {e}") from None
        radii[n] = _float(exact)
        ratios[n] = radii[n] / REFERENCE[n]
        if _above(exact, REFERENCE_ABOVE[n]):
            improved.append(n)
        beats_published += _above(exact, PUBLISHED_ABOVE[n])
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
    artifacts = {"eval_time": f"{eval_time:.4f}", "feedback": _feedback(ratios, improved),
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
