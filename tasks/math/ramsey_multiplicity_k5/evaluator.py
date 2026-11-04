"""Evaluator for the Ramsey multiplicity constant of K5; smoke/full/confirm on the sciel contract.

A function q: F_2^n -> [0, 1] (n <= 8) defines a step graphon on N = 2^n equal parts, red with probability
q(x XOR y) between parts x and y (q(0) inside a part). Colouring K_m by sampling a part per vertex gives expected
monochromatic K5 density t(K5, W) + t(K5, 1 - W), so this bounds c5 from above. Translations are automorphisms, so
fixing the first vertex at 0,

    t(K5, W) = N^-4 sum_z q(z) G(z),   G(z) = trace((D_u A)^3),   u(a) = q(a) q(a XOR z),   A[a, b] = q(a XOR b),

and dt/dq(z) = 10 N^-4 G(z). With integer numerators Q over a denominator D the numerator sum_z Q(z) G(z) is an
integer below N^4 D^10; it is computed modulo primes below 2^22, where binary64 matrix products are exact, and
recombined by CRT, with one more prime as a check.
"""

from sciel.candidate_modules import load_candidate
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
import uuid

import numpy as np

# Pinho round 4 (teorth/optimizationproblems PR 307), reproduced exactly by this evaluator.
REFERENCE = Fraction(15058298592194934908767847, 9131939205947984819238469632)
MAX_DIMENSION = 8
MAX_DENOMINATOR = 2 ** 32


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
                             "or use cayley.snap(q)")
        q, denominator = [int(x) for x in scaled], MAX_DENOMINATOR
        g = math.gcd(denominator, *q)
        q, denominator = [x // g for x in q], denominator // g
    n = len(q).bit_length() - 1
    if len(q) != 1 << n or not 0 <= n <= MAX_DIMENSION:
        raise ValueError(f"q must have 2**n entries, n <= {MAX_DIMENSION}, got {len(q)}")
    if any(x < 0 or x > denominator for x in q):
        raise ValueError("q must lie in [0, 1]")
    return q, denominator


def _primes(count):
    """The largest primes below 2**22, so that N p^2 < 2**53 for N <= 256."""
    out, p = [], 2 ** 22
    while len(out) < count:
        p -= 1
        if all(p % d for d in range(2, math.isqrt(p) + 1)):
            out.append(p)
    return out


def _residue(q, p):
    """sum_z q(z) trace((D_u A)^3) mod p, u = q * q(. XOR z), for integer q already reduced mod p."""
    N = len(q)
    idx = np.arange(N)
    xor = idx[:, None] ^ idx[None, :]
    qf = np.asarray(q, dtype=np.float64)
    A = qf[xor]
    total = 0
    for lo in range(0, N, 32):
        z = np.arange(lo, min(lo + 32, N))
        u = np.fmod(qf[None, :] * qf[xor[z]], p)
        P = np.fmod(u[:, :, None] * A[None], p)
        PP = np.fmod(P @ P, p)
        G = np.fmod(PP * np.swapaxes(P, 1, 2), p).astype(np.int64).sum(axis=(1, 2)) % p
        total += int((G * (qf[z].astype(np.int64)) % p).sum())
    return total % p


def _count(q, d):
    """sum_z Q(z) G(z) exactly, Q = q, for an integer kernel with denominator d."""
    N = len(q)
    bits = (N ** 4 * d ** 10).bit_length() + 1
    primes = _primes(bits // 21 + 2)
    residues = [_residue([x % p for x in q], p) for p in primes]
    value, modulus = 0, 1
    for r, p in zip(residues[:-1], primes[:-1]):
        value += modulus * ((r - value) * pow(modulus, -1, p) % p)
        modulus *= p
    if value % primes[-1] != residues[-1]:
        raise ArithmeticError("CRT check prime disagrees")
    return value


def density(q, d) -> Fraction:
    """t(K5, W) + t(K5, 1 - W) exactly, for integer numerators q over d."""
    N = len(q)
    red = _count(q, d)
    blue = _count([d - x for x in q], d)
    return Fraction(red + blue, N ** 4 * d ** 10)


# --- behavior readout -------------------------------------------------------
# The score is one number. These describe the SHAPE of the kernel instead, so two kernels with the same density can
# land in different cells of the diversity map.

_BEHAVIOR_ZERO = {"b_dispersion": 0.0, "b_boundary_frac": 0.0, "b_contact": 0.0,
                  "b_offset": 0.0, "b_concentration": 0.0, "b_structure": 0.0}


def _behavior(q, d):
    """Six bounded descriptors: spread of the fractional values, share of entries at 0 or 1, red edge density, the
    inside-part value q(0), weight of the red set on low-weight z, and the dimension. Never raises."""
    try:
        x = np.asarray(q, dtype=np.float64) / d
        n = int(round(math.log2(x.size)))
        frac = x[(x > 0) & (x < 1)]
        weight = np.array([bin(z).count("1") for z in range(x.size)])
        low = float(x[weight <= n // 2].sum() / max(x.sum(), 1e-300))
        out = {"b_dispersion": float(frac.std() * 2) if frac.size else 0.0,
               "b_boundary_frac": float(((x == 0) | (x == 1)).mean()), "b_contact": float(x.mean()),
               "b_offset": float(x[0]), "b_concentration": min(low, 1.0),
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
             "dimension": float(len(q).bit_length() - 1), "combined_score": combined, "validity": 1.0}
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
