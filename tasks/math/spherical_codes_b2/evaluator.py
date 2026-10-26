"""Evaluator for spherical codes on a slice of Henry Cohn's table (https://spherical-codes.org) marked "needs more
optimization": N points on S^(d-1) for the (d, N) of assets/sph_data.py; smoke/full/confirm on the sciel contract.

For each instance the candidate gives N nonzero points in R^d; only their directions count. The largest pair cosine
c is computed exactly: a binary64 point is an integer vector over a power of two, so the signed square of each
cosine is a rational and the near-maximal pairs found in floating point are compared as rationals. An instance
scores (1 - c) / (1 - c_bar), the minimal squared chord over the bar's, and combined_score is the mean, so a score
above 1 means some code beats the bar. The bar is the table's code after local search (assets/sph_best.py): descent
on a smooth maximum of the cosines at rising sharpness, sequential linear programming on the near-active pairs and
basin hopping, about eight minutes per instance. These codes do not converge: a further minute of descent and linear
programming still lowers c by up to about 1e-7 on 30 of the 33 instances, so n_improved counts only c <= c_bar -
1e-6, more than a short polish of the bar reaches. n_beats_published counts c <= the table's displayed cosine -
1e-12: the table shows cosines rounded up to 12 digits and its upload form ignores codes that do not improve the
displayed value, so exactly these would show up as new entries.
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
DATA = load_candidate(str(ASSETS / "sph_data.py"), "sph_data_evaluator")
BEST = load_candidate(str(ASSETS / "sph_best.py"), "sph_best_evaluator")
INSTANCES = tuple(DATA.INSTANCES)
DIM = {k: int(k.split("_")[0]) for k in INSTANCES}
SIZE = {k: int(k.split("_")[1]) for k in INSTANCES}
MARGIN = Fraction(1, 10 ** 6)
SITE_MARGIN = Fraction(1, 10 ** 12)
BAR_COS2 = {k: Fraction(BEST.COS2[k]) for k in INSTANCES}
PUBLISHED = {k: Fraction(DATA.COS[k]) for k in INSTANCES}


def _load(program_path, module_name="program"):
    return load_candidate(program_path, module_name)


def _sqrt(c2, bits=160):
    """sign(c2) sqrt(|c2|) rounded toward zero to a multiple of 2^-bits."""
    root = math.isqrt(abs(c2.numerator) * 4 ** bits // c2.denominator)
    return Fraction(root if c2 > 0 else -root, 2 ** bits)


BAR_COS = {k: _sqrt(BAR_COS2[k]) for k in INSTANCES}


def _integers(row):
    ratios = [v.as_integer_ratio() for v in row]
    q = max(b for _, b in ratios)
    return [a * (q // b) for a, b in ratios]


def cos2(p) -> Fraction:
    """The largest pair cosine of the (n, d) binary64 points squared with its sign, exactly. Raises ValueError on a
    zero point or two points with the same direction."""
    top = np.abs(p).max(axis=1, keepdims=True)
    if (top == 0).any():
        raise ValueError("a point is the zero vector")
    u = p / top
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    g = u @ u.T
    np.fill_diagonal(g, -np.inf)
    # A floating cosine of unit rows is within about 2 (d + 2) units of 2^-53 of the exact one, so every pair that can
    # have the largest cosine is within 1e-10 of the floating maximum; those are compared exactly.
    i, j = np.nonzero(np.triu(g >= g.max() - 1e-10, 1))
    ints, norms, best = {}, {}, None
    for a, b in zip(i.tolist(), j.tolist()):
        for k in (a, b):
            if k not in ints:
                ints[k] = _integers(p[k].tolist())
                norms[k] = sum(v * v for v in ints[k])
        ip = sum(x * y for x, y in zip(ints[a], ints[b]))
        value = Fraction(ip * abs(ip), norms[a] * norms[b])
        if best is None or value > best:
            best = value
    if best >= 1:
        raise ValueError("two points have the same direction")
    return best


def _at_most(c2, t):
    """Whether the cosine with signed square c2 is <= t."""
    if t >= 0:
        return c2 <= t * t
    return c2 < 0 and -c2 >= t * t


def _codes(value):
    if not isinstance(value, dict):
        raise ValueError('codes() must return a dict {"d_n": list of n points in R^d}')
    out = {}
    for k in INSTANCES:
        if k not in value:
            raise ValueError(f"missing {k}")
        p = np.array(value[k], dtype=np.float64)
        if p.shape != (SIZE[k], DIM[k]) or not np.isfinite(p).all():
            raise ValueError(f"{k}: expected {SIZE[k]} finite points in R^{DIM[k]}, got shape {p.shape}")
        out[k] = p
    return out


# --- behavior readout -------------------------------------------------------
# The score is one mean. These describe WHERE the codes stand against the bar, so runs that win in different
# parts of the slice land in different cells of the diversity map.

_BEHAVIOR_ZERO = {"b_dispersion": 0.0, "b_boundary_frac": 0.0, "b_contact": 0.0,
                  "b_offset": 0.0, "b_concentration": 0.0, "b_structure": 0.0}


def _behavior(ratios):
    """Six bounded descriptors of the per-instance ratios: their spread, share at or above the bar, mean gain in
    the first, middle and last third of the instances (ordered by d, then N), and the share within 1e-9 of the bar.
    Never raises."""
    try:
        r = np.array([ratios[k] for k in INSTANCES])
        third = np.arange(len(r)) * 3 // len(r)
        squash = lambda x: float(0.5 + 0.5 * np.tanh(x * 1e4))
        part = lambda m: squash(r[m].mean() - 1) if m.any() else 0.5
        out = {"b_dispersion": float(min(r.std() * 1e4, 1.0)), "b_boundary_frac": float((r >= 1.0).mean()),
               "b_contact": part(third == 0), "b_offset": part(third == 1),
               "b_concentration": part(third == 2), "b_structure": float((np.abs(r - 1) < 1e-9).mean())}
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
    if not hasattr(program, "codes"):
        raise AttributeError("missing codes")
    start_time = time.time()
    configs = _codes(program.codes())
    eval_time = time.time() - start_time
    cos, ratios, improved, beats_published = {}, {}, [], 0
    for k, p in configs.items():
        try:
            exact = cos2(p)
        except Exception as e:
            raise ValueError(f"{k}: {e}") from None
        c = _sqrt(exact)
        cos[k] = float(c)
        ratios[k] = float((1 - c) / (1 - BAR_COS[k]))
        if _at_most(exact, BAR_COS[k] - MARGIN):
            improved.append(k)
        beats_published += _at_most(exact, PUBLISHED[k] - SITE_MARGIN)
    best = max(INSTANCES, key=lambda k: ratios[k])
    combined = float(np.mean([ratios[k] for k in INSTANCES]))
    cases = {"combined_score": combined, "n_improved": float(len(improved)),
             "best_gain_cos": float(BAR_COS[best]) - cos[best], "best_gain_dim": float(DIM[best]),
             "best_gain_n": float(SIZE[best]), "n_beats_published": float(beats_published), "validity": 1.0}
    cases.update(_behavior(ratios))
    witness = {
        "schema": "sciel-full-output/1",
        "source_sha256": source_sha, "evaluator_sha256": evaluator_sha,
        "invocation_id": invocation_id, "stage": stage, "confirmation": None,
        "input_sha256": None, "seed_sha256": None,
        # Base64 of the little-endian binary64 rows keeps the witness exact and under the 1 MiB artifact limit.
        "output": {"codes": {k: base64.b64encode(configs[k].astype("<f8").tobytes()).decode() for k in INSTANCES},
                   "cos": cos, "improved": improved},
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
        "cases": {"combined_score": 0.0, "n_improved": 0.0, "best_gain_cos": 0.0, "best_gain_dim": 0.0,
                  "best_gain_n": 0.0, "n_beats_published": 0.0, "validity": 0.0, **_BEHAVIOR_ZERO},
        "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]},
    }
