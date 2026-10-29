"""Evaluator for N equal circles of largest radius in the regular pentagon with circumradius 1 and a flat bottom edge,
N = 100..200 (assets/cpt_data.py: SIZES); smoke/full/confirm on the sciel contract.

The best known packings are Packomania's pentagon table (https://www.packomania.com/cpt/cpt.html, E. Specht's
program cpt and P. Amore, 2023); the published centres as binary64 (assets/cpt_best.py) are the bar, unpolished.
For each N the candidate gives N centres in the pentagon, whose sides have outer unit normals n_k at angles -pi/2 +
2 pi k / 5 and lie at distance h = cos(pi/5) from the origin; the radius is the largest r with every centre at
distance >= r inside every side (h - n_k . c >= r) and every pair at distance >= 2r. It is computed exactly: binary64
centres are rationals over one power of two d, a side distance is then a + b sqrt 5 + (c + d sqrt 5) t with rationals
a, b, c, d and t = sqrt(10 + 2 sqrt 5) (h = (1 + sqrt 5) / 4, sin(pi/10) = (sqrt 5 - 1) / 4, cos(pi/10) = t / 4,
cos(3 pi/10) = (sqrt 5 - 1) t / 8), so is its square, and a squared pair distance is rational. The squared radius is
the least of these, compared exactly through signs in Q(sqrt 5) and squaring with signs tracked (_sign4), never in
floating point; only the score rounds. Each N scores radius / best radius and combined_score is their mean, so a score
above 1 means some packing beats the best known. The published centres have 12 digits, which leaves their radius 2e-12
to 2.4e-11 relative below the published one, and re-maximizing them on their contacts wins that back, so n_improved
counts the N beaten by more than 1e-10 relative, compared exactly: anything above that margin is a packing better than
the published one (a polish of a published packing that was not fully converged counts too). n_beats_published
counts the N beating Packomania's own 12-digit radius by the same margin.
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
DATA = load_candidate(str(ASSETS / "cpt_data.py"), "cpt_data_evaluator")
BEST = load_candidate(str(ASSETS / "cpt_best.py"), "cpt_best_evaluator")


def _load(program_path, module_name="program"):
    return load_candidate(program_path, module_name)


def _sign(x):
    return (x > 0) - (x < 0)


def _sign5(a, b):
    """The sign of a + b sqrt 5 for rationals a, b, exactly."""
    sa, sb = _sign(a), _sign(b)
    if sb == 0:
        return sa
    if sa == 0 or sa == sb:
        return sb
    return sa if a * a > 5 * b * b else sb


def _sign4(x):
    """The sign of a + b sqrt 5 + (c + d sqrt 5) t for x = (a, b, c, d), t = sqrt(10 + 2 sqrt 5), exactly: the signs
    of u = a + b sqrt 5 and v = c + d sqrt 5, then, when they differ, the sign of u^2 - v^2 t^2 in Q(sqrt 5)."""
    a, b, c, d = x
    su, sv = _sign5(a, b), _sign5(c, d)
    if sv == 0:
        return su
    if su == 0 or su == sv:
        return sv
    p, q = c * c + 5 * d * d, 2 * c * d
    return su if _sign5(a * a + 5 * b * b - 10 * p - 10 * q, 2 * a * b - 2 * p - 10 * q) > 0 else sv


def _sub(x, y):
    return tuple(u - v for u, v in zip(x, y))


def _scale(x, k):
    return tuple(u * k for u in x)


def _square(x):
    """(u + v t)^2 = u^2 + v^2 (10 + 2 sqrt 5) + 2 u v t for u = a + b sqrt 5, v = c + d sqrt 5."""
    a, b, c, d = x
    p, q = c * c + 5 * d * d, 2 * c * d
    return (a * a + 5 * b * b + 10 * p + 10 * q, 2 * a * b + 2 * p + 10 * q,
            2 * (a * c + 5 * b * d), 2 * (a * d + b * c))


def _above(x, y):
    """Whether x exceeds y, both (a, b, c, d), exactly."""
    return _sign4(_sub(x, y)) > 0


# Fractions within 1e-40 of sqrt 5 and t, so the float of an exact number is correctly rounded but for ties.
_S5 = Fraction(math.isqrt(5 * 10 ** 80), 10 ** 40)
_T = Fraction(math.isqrt(int((10 + 2 * _S5) * 10 ** 80)), 10 ** 40)


def _float(x):
    a, b, c, d = x
    return float(a + b * _S5 + (c + d * _S5) * _T)


_Q4, _Q8 = Fraction(1, 4), Fraction(1, 8)
H = (_Q4, _Q4, 0, 0)
# The outer unit normal n_k = (nx, ny) of each side k = 0..4, in the same field.
NORMALS_EXACT = (((0, 0, 0, 0), (-1, 0, 0, 0)), ((0, 0, _Q4, 0), (_Q4, -_Q4, 0, 0)),
                 ((0, 0, -_Q8, _Q8), (_Q4, _Q4, 0, 0)), ((0, 0, _Q8, -_Q8), (_Q4, _Q4, 0, 0)),
                 ((0, 0, -_Q4, 0), (_Q4, -_Q4, 0, 0)))
H_FLOAT = math.cos(math.pi / 5)
NORMALS = np.array([[math.cos(a), math.sin(a)] for a in -math.pi / 2 + 2 * math.pi * np.arange(5) / 5])

SIZES = tuple(DATA.SIZES)
MARGIN = 1 + Fraction(1, 10 ** 10)
# Squared radii are (a, b, c, d); a candidate improves when its squared radius exceeds the bar's times MARGIN^2.
PUBLISHED_ABOVE = {n: ((Fraction(DATA.RADIUS[n]) * MARGIN) ** 2, 0, 0, 0) for n in SIZES}
REFERENCE_R2 = {n: tuple(Fraction(v) for v in BEST.R2[n]) for n in SIZES}
REFERENCE_ABOVE = {n: _scale(r2, MARGIN ** 2) for n, r2 in REFERENCE_R2.items()}
REFERENCE = {n: math.sqrt(_float(REFERENCE_R2[n])) for n in SIZES}


def radius2(p):
    """The squared radius of the (n, 2) binary64 centres exactly, as (a, b, c, d). Raises ValueError when a centre is
    outside the closed pentagon or the radius is zero."""
    ratios = [v.as_integer_ratio() for v in p.ravel().tolist()]
    d = max(q for _, q in ratios)
    z = [a * (d // q) for a, q in ratios]
    xs, ys = z[0::2], z[1::2]
    # With every coordinate in [-1, 1], a floating side distance is within a few 1e-16 of the exact one, so the
    # nearest sides are among those within 1e-12 of the floating minimum; they are compared exactly.
    side = H_FLOAT - p @ NORMALS.T
    wall = None
    for i, k in np.argwhere(side <= side.min() + 1e-12).tolist():
        (nx, ny), x, y = NORMALS_EXACT[k], Fraction(xs[i], d), Fraction(ys[i], d)
        s = tuple(h - u * x - v * y for h, u, v in zip(H, nx, ny))
        wall = s if wall is None or _above(wall, s) else wall
    if _sign4(wall) < 0:
        raise ValueError("a centre is outside the pentagon")
    # With every coordinate in [-1, 1], each floating squared distance is within a relative 4e-16 of the exact one,
    # so the closest pair is among those within a relative 1e-9 of the floating minimum; they are compared exactly.
    sq = ((p[:, None, :] - p[None, :, :]) ** 2).sum(-1)
    np.fill_diagonal(sq, np.inf)
    i, j = np.nonzero(np.triu(sq <= sq.min() * (1 + 1e-9) + 1e-300, 1))
    pair = min((xs[u] - xs[v]) ** 2 + (ys[u] - ys[v]) ** 2 for u, v in zip(i.tolist(), j.tolist()))
    if _sign4(wall) == 0 or pair == 0:
        raise ValueError("two centres coincide or a centre is on a side")
    wall, pair = _square(wall), (Fraction(pair, 4 * d * d), 0, 0, 0)
    return pair if _above(wall, pair) else wall


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
            f"{len(SIZES) - len(moved)} N at the bar; {len(improved)} N improved by more than 1e-10")


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
        radii[n] = math.sqrt(_float(exact))
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
