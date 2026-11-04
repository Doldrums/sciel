"""Evaluator for the Ramsey multiplicity constant of K6; smoke/full/confirm on the sciel contract.

A function q: {0, ..., n} -> [0, 1] (n <= 10) defines a step graphon on the 2^n parts F_2^n, red with probability
q(|x XOR y|) between parts x and y (q(0) inside a part). Colouring K_m by sampling a part per vertex gives expected
monochromatic K6 density t(K6, W) + t(K6, 1 - W), so this bounds c6 from above. Fix the first vertex at 0: the other
five are n independent uniform columns in F_2^5, each adding a cut of the six vertices to the 15 pair weights, and
the colouring only sees those weights. The weight vectors are counted by a walk over the 32 cuts, merged up to the
720 relabellings of the six vertices (about 1.1 million classes at n = 10), so

    t(K6, W) = 32^-n sum_classes m(w) prod_e q(w_e),

with integer class sizes m summing to 32^n. With integer numerators Q over D the numerator is an integer below
2 32^n D^15; it is computed modulo primes below 2^31 and recombined by CRT, with one more prime as a check.
"""

from sciel.candidate_modules import load_candidate
from fractions import Fraction
from functools import lru_cache
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
import traceback
import uuid

import numpy as np

# Jeetu SK (teorth/optimizationproblems PR 279), 0.742254861593625 * 2^-14, reproduced exactly by this evaluator.
REFERENCE = Fraction(
    180244428715156181493112659521741405180580225001966676111887908798425648907356856409,
    3978585891278293137243057985174566720803649206378781739523711815145275976100267004264448)
MAX_DIMENSION = 10
MAX_DENOMINATOR = 2 ** 32
PAIRS = list(itertools.combinations(range(6), 2))


def _load(program_path, module_name="program"):
    return load_candidate(program_path, module_name)


def _kernel(value):
    """The candidate's q as integer numerators over one denominator, with the checks of the task. A pair
    (numerators, denominator) is taken as given; an array of binary64 values must be multiples of 2**-32."""
    if isinstance(value, tuple) and len(value) == 2:
        numerators, denominator = value
        if isinstance(denominator, bool) or not isinstance(denominator, (int, np.integer)):
            raise ValueError("the denominator must be an integer")
        denominator = int(denominator)
        if not 1 <= denominator <= MAX_DENOMINATOR:
            raise ValueError(f"the denominator must be in 1..2**32, got {denominator}")
        raw = list(numerators)
        if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in raw):
            raise ValueError("numerators must be integers")
        q = [int(x) for x in raw]
    else:
        h = np.array(value, dtype=np.float64, copy=True)
        if h.ndim != 1 or not np.isfinite(h).all():
            raise ValueError("q must be a 1-D sequence of finite numbers")
        scaled = np.ldexp(h, 32)
        if (scaled != np.round(scaled)).any():
            raise ValueError("binary64 values of q must be multiples of 2**-32; return (numerators, denominator) "
                             "or use weights.snap(q)")
        q, denominator = [int(x) for x in scaled], MAX_DENOMINATOR
        g = math.gcd(denominator, *q)
        q, denominator = [x // g for x in q], denominator // g
    if not 2 <= len(q) <= MAX_DIMENSION + 1:
        raise ValueError(f"q must have n + 1 entries, 1 <= n <= {MAX_DIMENSION}, got {len(q)}")
    if any(x < 0 or x > denominator for x in q):
        raise ValueError("q must lie in [0, 1]")
    return q, denominator


@lru_cache(maxsize=None)
def orbits(n: int) -> tuple:
    """(W, m): the 15 pair weights of one representative per class, as an int8 array (classes, 15), and the class
    sizes as int64, over the 32^n tuples (x_1, ..., x_5) with x_0 = 0. About 5 s at n = 10."""
    index = {p: i for i, p in enumerate(PAIRS)}
    cuts = np.array([[(t >> i & 1) ^ (t >> j & 1) for i, j in PAIRS] for t in range(0, 64, 2)], dtype=np.int64)
    base = float(n + 1)
    codes = np.zeros((len(PAIRS), 720))
    for k, s in enumerate(itertools.permutations(range(6))):
        for e, (i, j) in enumerate(PAIRS):
            codes[e, k] = base ** index[tuple(sorted((s[i], s[j])))]
    W, m = np.zeros((1, len(PAIRS)), dtype=np.int64), np.ones(1, dtype=np.int64)
    for _ in range(n):
        step = (W[:, None, :] + cuts[None]).reshape(-1, len(PAIRS))
        sizes = np.repeat(m, len(cuts))
        key = np.concatenate([(step[a:a + 4096].astype(np.float64) @ codes).min(axis=1)
                              for a in range(0, len(step), 4096)])
        key, first, inverse = np.unique(key, return_index=True, return_inverse=True)
        m = np.zeros(key.size, dtype=np.int64)
        np.add.at(m, inverse, sizes)
        W = step[first]
    return W.astype(np.int8), m


def _primes(count):
    """The largest primes below 2**31, so that products of two residues fit in int64."""
    out, p = [], 2 ** 31
    while len(out) < count:
        p -= 1
        if all(p % d for d in range(2, math.isqrt(p) + 1)):
            out.append(p)
    return out


def _count(W, m, q, d):
    """sum_classes m prod_e (Q(w_e) + (d - Q(w_e)) on the blue side) exactly: red plus blue numerators."""
    bits = (2 * int(m.sum()) * d ** 15).bit_length() + 1
    primes = _primes(bits // 30 + 2)
    residues = []
    for p in primes:
        total = 0
        for side in (q, [d - x for x in q]):
            values = np.array([x % p for x in side], dtype=np.int64)
            prod = values[W[:, 0]]
            for e in range(1, W.shape[1]):
                prod = prod * values[W[:, e]] % p
            total += int((prod * (m % p) % p).sum())
        residues.append(total % p)
    value, modulus = 0, 1
    for r, p in zip(residues[:-1], primes[:-1]):
        value += modulus * ((r - value) * pow(modulus, -1, p) % p)
        modulus *= p
    if value % primes[-1] != residues[-1]:
        raise ArithmeticError("CRT check prime disagrees")
    return value


def density(q, d) -> Fraction:
    """t(K6, W) + t(K6, 1 - W) exactly, for integer numerators q over d."""
    n = len(q) - 1
    W, m = orbits(n)
    return Fraction(_count(W, m, q, d), 32 ** n * d ** 15)


# --- behavior readout -------------------------------------------------------
# The score is one number. These describe the SHAPE of the kernel instead, so two kernels with the same density can
# land in different cells of the diversity map.

_BEHAVIOR_ZERO = {"b_dispersion": 0.0, "b_boundary_frac": 0.0, "b_contact": 0.0,
                  "b_offset": 0.0, "b_concentration": 0.0, "b_structure": 0.0}


def _behavior(q, d):
    """Six bounded descriptors: spread of the fractional values, share of weights at 0 or 1, red edge density, the
    inside-part value q(0), share of the red mass on weights up to n/2, and the dimension. Never raises."""
    try:
        x = np.asarray(q, dtype=np.float64) / d
        n = x.size - 1
        mass = np.array([math.comb(n, k) for k in range(n + 1)], dtype=np.float64) / 2 ** n
        frac = x[(x > 0) & (x < 1)]
        red = float(mass @ x)
        out = {"b_dispersion": float(frac.std() * 2) if frac.size else 0.0,
               "b_boundary_frac": float(((x == 0) | (x == 1)).mean()), "b_contact": red,
               "b_offset": float(x[0]),
               "b_concentration": min(float(mass[:n // 2 + 1] @ x[:n // 2 + 1]) / max(red, 1e-300), 1.0),
               "b_structure": n / MAX_DIMENSION}
        return {k: (float(v) if np.isfinite(v) else 0.0) for k, v in out.items()}
    except (ValueError, FloatingPointError, ZeroDivisionError):
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
    if not hasattr(program, "kernel"):
        raise AttributeError("missing kernel")
    start_time = time.time()
    q, d = _kernel(program.kernel())
    eval_time = time.time() - start_time
    value = density(q, d)
    combined = float(REFERENCE / value)
    cases = {"upper_bound": float(value), "inverse_bound": float(1 / value),
             "dimension": float(len(q) - 1), "combined_score": combined, "validity": 1.0}
    cases.update(_behavior(q, d))
    witness = {
        "schema": "sciel-full-output/1",
        "source_sha256": source_sha, "evaluator_sha256": evaluator_sha,
        "invocation_id": invocation_id, "stage": stage, "confirmation": None,
        "input_sha256": None, "seed_sha256": None,
        "output": {"numerators": q, "denominator": d, "density": [value.numerator, value.denominator]},
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
        "cases": {"combined_score": 0.0, "upper_bound": 0.0, "inverse_bound": 0.0, "dimension": 0.0,
                  "validity": 0.0, **_BEHAVIOR_ZERO},
        "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]},
    }
