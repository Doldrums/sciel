"""Evaluator for the Kakeya needle problem with 128 triangles, EinsteinArena's kakeya-needle-128 (AlphaEvolve problem
6.9); smoke/full/confirm on the sciel contract.

Triangle j = 1..128 is conv{(x_j, 0), (x_j + 1/128, 0), (x_j + j/128, 1)}, and the score is the area of the union. At
height y its section is [x_j + j y / 128, x_j + 1/128 + (j - 1) y / 128], so every section endpoint is affine in y,
and between two consecutive heights where endpoints cross the union length is affine too. The area is integrated
exactly slab by slab, as EinsteinArena's verifier does: offsets are decimals with at most 25 fractional digits, read
as exact rationals, every crossing height is an integer over one common denominator, and the area is one rational
rounded once. A superset of the crossings EinsteinArena uses gives the same rational, so the two agree exactly.
"""

from sciel.candidate_modules import load_candidate
from fractions import Fraction
import hashlib
import json
import math
from numbers import Real
from pathlib import Path
import time
import traceback
import uuid

import numpy as np

N = 128
# Ghazt's construction on EinsteinArena, 6 October 2026, the best on the board.
REFERENCE = 0.10675394011230244
MAX_FRAC_DIGITS = 25
MAX_INT_DIGITS = 4
# Every crossing height has a denominator den * m with m <= N, so all of them are integers over den * lcm(1..N).
_LCM = math.lcm(*range(1, N + 1))


def _load(program_path, module_name="program"):
    return load_candidate(program_path, module_name)


def _parse(s):
    """A decimal string as EinsteinArena reads it: optional sign, digits, at most one point, at most 25 fractional and
    4 integer digits."""
    t = s.strip()
    neg = t.startswith("-")
    if neg or t.startswith("+"):
        t = t[1:]
    if t.count(".") > 1:
        raise ValueError(f"malformed decimal string {s!r}")
    intpart, _, fracpart = t.partition(".")
    digits = intpart + fracpart
    if digits == "" or not digits.isdigit():
        raise ValueError(f"malformed decimal string {s!r}")
    if len(fracpart) > MAX_FRAC_DIGITS:
        raise ValueError(f"offset {s!r} has more than {MAX_FRAC_DIGITS} fractional digits")
    if len(intpart) > MAX_INT_DIGITS:
        raise ValueError(f"offset {s!r} has more than {MAX_INT_DIGITS} integer digits")
    value = Fraction(int(digits), 10 ** len(fracpart))
    return -value if neg else value


def decimal(value: Fraction) -> str:
    """A rational with denominator dividing 10**25 as the shortest decimal string that reads back exactly."""
    units = value * 10 ** MAX_FRAC_DIGITS
    if units.denominator != 1:
        raise ValueError("not a multiple of 10**-25")
    sign, units = ("-" if units < 0 else ""), abs(int(units))
    whole, frac = divmod(units, 10 ** MAX_FRAC_DIGITS)
    frac = str(frac).rjust(MAX_FRAC_DIGITS, "0").rstrip("0")
    return f"{sign}{whole}.{frac}" if frac else f"{sign}{whole}"


def offsets_of(values):
    """The candidate's offsets as exact decimal strings. A string is read as it is written; a number is its binary64
    value rounded to 25 decimal places, which the witness records as a string ready to submit."""
    if isinstance(values, (str, bytes)) or not hasattr(values, "__len__"):
        raise ValueError(f"offsets must be a sequence of {N} values")
    if len(values) != N:
        raise ValueError(f"expected exactly {N} offsets, got {len(values)}")
    out = []
    for v in values:
        if isinstance(v, str):
            out.append(decimal(_parse(v)))
        elif isinstance(v, (Real, np.floating, np.integer)) and not isinstance(v, (bool, np.bool_)):
            f = float(v)
            if not math.isfinite(f):
                raise ValueError("offsets must be finite")
            s = decimal(Fraction(round(Fraction(f) * 10 ** MAX_FRAC_DIGITS), 10 ** MAX_FRAC_DIGITS))
            _parse(s)
            out.append(s)
        else:
            raise ValueError(f"an offset must be a decimal string or a real number, got {type(v).__name__}")
    return out


def _union(intervals):
    intervals.sort()
    total = 0
    lo, hi = intervals[0]
    for left, right in intervals[1:]:
        if left > hi:
            total += hi - lo
            lo, hi = left, right
        elif right > hi:
            hi = right
    return total + hi - lo


def area(strings) -> Fraction:
    """The exact area of the union of the triangles with these decimal offsets."""
    xs = [_parse(s) for s in strings]
    xs = [x - xs[0] for x in xs]
    den = math.lcm(*(x.denominator for x in xs))
    # Endpoint lines (slope, intercept) of every section, scaled by N * den so that they are integers.
    lines = []
    for j, x in enumerate(xs, start=1):
        b = N * x.numerator * (den // x.denominator)
        lines += [(den * j, b), (den * (j - 1), b + den)]
    q = den * _LCM
    ys = set()
    for a in range(len(lines)):
        sa, ba = lines[a]
        for sb, bb in lines[a + 1:]:
            if sa != sb:
                y = (bb - ba) * (_LCM // ((sa - sb) // den))
                if 0 < y < q:
                    ys.add(y)
    total, prev_y, prev_u = 0, 0, None
    for y in [0, *sorted(ys), q]:
        u = _union([(lines[k][1] * q + lines[k][0] * y, lines[k + 1][1] * q + lines[k + 1][0] * y)
                    for k in range(0, len(lines), 2)])
        if prev_u is not None:
            total += (prev_u + u) * (y - prev_y)
        prev_y, prev_u = y, u
    value = Fraction(total, 2 * q * q * N * den)
    # The union is no larger than the 128 triangles of area 1/256 together.
    if not 0 < value <= Fraction(1, 2):
        raise ValueError(f"area check failed: {float(value)}")
    return value


# --- behavior readout -------------------------------------------------------
# The score is one number. These describe the SHAPE of the offset sequence instead, so two
# constructions with the same area can land in different cells of the diversity map.

_BEHAVIOR_ZERO = {"b_dispersion": 0.0, "b_rise": 0.0, "b_jump": 0.0,
                  "b_trend": 0.0, "b_overlap": 0.0, "b_structure": 0.0}


def _behavior(strings, value):
    """Six bounded descriptors: spread of the offsets, share of steps where the next offset is larger, the largest
    step relative to the spread, the fitted drift per triangle, union area over the triangles' total, and the share of
    the offsets' variance a straight line does not explain. Never raises."""
    try:
        x = np.array([float(s) for s in strings])
        x -= x.mean()
        j = np.arange(N) / N
        span = float(x.max() - x.min())
        steps = np.diff(x)
        slope, intercept = np.polyfit(j, x, 1)
        residual = x - slope * j - intercept
        var = float(x.var())
        out = {"b_dispersion": float(x.std() / (x.std() + 0.05)),
               "b_rise": float((steps > 0).mean()),
               "b_jump": float(np.abs(steps).max() / span) if span > 0 else 0.0,
               "b_trend": 0.5 + 0.5 * math.tanh(float(slope) / 0.2),
               "b_overlap": float(value * 2),
               "b_structure": float(residual.var() / var) if var > 0 else 0.0}
        return {k: (min(max(float(v), 0.0), 1.0) if math.isfinite(v) else 0.0) for k, v in out.items()}
    except (ValueError, FloatingPointError, ZeroDivisionError, np.linalg.LinAlgError):
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
    if not hasattr(program, "offsets"):
        raise AttributeError("missing offsets")
    start_time = time.time()
    raw = program.offsets()
    eval_time = time.time() - start_time
    strings = offsets_of(list(raw) if isinstance(raw, np.ndarray) else raw)
    value = area(strings)
    combined = float(Fraction(REFERENCE) / value)
    cases = {"area": float(value), "inverse_area": float(1 / value), "combined_score": combined,
             "validity": 1.0}
    cases.update(_behavior(strings, value))
    witness = {
        "schema": "sciel-full-output/1",
        "source_sha256": source_sha, "evaluator_sha256": evaluator_sha,
        "invocation_id": invocation_id, "stage": stage, "confirmation": None,
        "input_sha256": None, "seed_sha256": None,
        "output": {"offsets": strings, "area": [value.numerator, value.denominator]},
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
        "cases": {"combined_score": 0.0, "area": 0.0, "inverse_area": 0.0, "validity": 0.0, **_BEHAVIOR_ZERO},
        "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]},
    }
