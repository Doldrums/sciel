"""Evaluator for the Ramsey multiplicity constant of K5 over mixed abelian groups; smoke/full/confirm on the sciel
contract.

A finite abelian group Z_{m_1} x ... x Z_{m_k} of order N <= 256 and a symmetric function q on it, q(-z) = q(z),
define a step graphon on N equal parts, red with probability q(x - y) between parts x and y (q(0) inside a part).
Colouring K_m by sampling a part per vertex gives expected monochromatic K5 density t(K5, W) + t(K5, 1 - W), so this
bounds c5 from above. Translations are automorphisms, so fixing the first vertex at 0,

    t(K5, W) = N^-4 sum_z q(z) G(z),   G(z) = trace((D_u A)^3),   u(a) = q(a) q(a - z),   A[a, b] = q(a - b),

and dt/dq(z) = 10 N^-4 G(z) along symmetric q. With integer numerators Q over a denominator D the numerator
sum_z Q(z) G(z) is an integer below N^4 D^10; it is computed modulo primes below 2^22, where binary64 matrix products
are exact, and recombined by CRT, with one more prime as a check. Group elements are indexed in mixed radix, the
first modulus most significant: z = (z_1, ..., z_k) has index ((z_1 m_2 + z_2) m_3 + ...) m_k + z_k.
"""

from sciel.candidate_modules import load_candidate
from fractions import Fraction
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
import traceback
import uuid

import numpy as np

# Pinho round 4 (teorth/optimizationproblems PR 307), a Cayley graphon on Z_2^8; the record to beat.
REFERENCE = Fraction(15058298592194934908767847, 9131939205947984819238469632)
MAX_ORDER = 256
MAX_DENOMINATOR = 2 ** 32


def _load(program_path, module_name="program"):
    return load_candidate(program_path, module_name)


def group(moduli):
    """(difference table D, negation): D[a, b] is the index of a - b, negation[z] the index of -z."""
    moduli = [int(m) for m in moduli]
    elements = np.array(list(itertools.product(*[range(m) for m in moduli])), dtype=np.int64).reshape(-1, len(moduli))
    m = np.array(moduli, dtype=np.int64)
    weights = np.array([math.prod(moduli[i + 1:]) for i in range(len(moduli))], dtype=np.int64)
    difference = ((elements[:, None, :] - elements[None, :, :]) % m) @ weights
    return difference, ((-elements) % m) @ weights


def _moduli(value):
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("moduli must be a non-empty list of integers")
    if any(isinstance(m, bool) or not isinstance(m, (int, np.integer)) or int(m) < 2 for m in value):
        raise ValueError("every modulus must be an integer >= 2")
    moduli = [int(m) for m in value]
    if math.prod(moduli) > MAX_ORDER:
        raise ValueError(f"the group order {math.prod(moduli)} exceeds {MAX_ORDER}")
    return moduli


def _kernel(value):
    """The candidate's (moduli, q) with q as integer numerators over one denominator, with the checks of the task.
    q is a pair (numerators, denominator), taken as given, or an array of binary64 multiples of 2**-32."""
    if not isinstance(value, dict) or set(value) != {"moduli", "q"}:
        raise ValueError('kernel() must return {"moduli": [m_1, ..., m_k], "q": q}')
    moduli = _moduli(value["moduli"])
    kernel = value["q"]
    if isinstance(kernel, tuple) and len(kernel) == 2:
        numerators, denominator = kernel
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
        h = np.array(kernel, dtype=np.float64, copy=True)
        if h.ndim != 1 or not np.isfinite(h).all():
            raise ValueError("q must be a 1-D sequence of finite numbers")
        scaled = np.ldexp(h, 32)
        if (scaled != np.round(scaled)).any():
            raise ValueError("binary64 values of q must be multiples of 2**-32; return (numerators, denominator) "
                             "or use group.snap(q)")
        q, denominator = [int(x) for x in scaled], MAX_DENOMINATOR
    g = math.gcd(denominator, *q)
    q, denominator = [x // g for x in q], denominator // g
    if len(q) != math.prod(moduli):
        raise ValueError(f"q must have one entry per group element, {math.prod(moduli)}, got {len(q)}")
    if any(x < 0 or x > denominator for x in q):
        raise ValueError("q must lie in [0, 1]")
    negation = group(moduli)[1]
    if any(q[z] != q[int(negation[z])] for z in range(len(q))):
        raise ValueError("q must be symmetric, q(-z) = q(z); use group.symmetrize")
    return moduli, q, denominator


def _primes(count):
    """The largest primes below 2**22, so that N p^2 < 2**53 for N <= 256."""
    out, p = [], 2 ** 22
    while len(out) < count:
        p -= 1
        if all(p % d for d in range(2, math.isqrt(p) + 1)):
            out.append(p)
    return out


def _residue(q, p, difference):
    """sum_z q(z) trace((D_u A)^3) mod p, u = q * q(. - z), for integer q already reduced mod p."""
    N = len(q)
    qf = np.asarray(q, dtype=np.float64)
    A = qf[difference]
    total = 0
    for lo in range(0, N, 32):
        z = np.arange(lo, min(lo + 32, N))
        u = np.fmod(qf[None, :] * qf[difference[:, z].T], p)
        P = np.fmod(u[:, :, None] * A[None], p)
        PP = np.fmod(P @ P, p)
        G = np.fmod(PP * np.swapaxes(P, 1, 2), p).astype(np.int64).sum(axis=(1, 2)) % p
        total += int((G * (qf[z].astype(np.int64)) % p).sum())
    return total % p


def _count(q, d, difference):
    """sum_z Q(z) G(z) exactly, Q = q, for an integer kernel with denominator d."""
    N = len(q)
    bits = (N ** 4 * d ** 10).bit_length() + 1
    primes = _primes(bits // 21 + 2)
    residues = [_residue([x % p for x in q], p, difference) for p in primes]
    value, modulus = 0, 1
    for r, p in zip(residues[:-1], primes[:-1]):
        value += modulus * ((r - value) * pow(modulus, -1, p) % p)
        modulus *= p
    if value % primes[-1] != residues[-1]:
        raise ArithmeticError("CRT check prime disagrees")
    return value


def density(moduli, q, d) -> Fraction:
    """t(K5, W) + t(K5, 1 - W) exactly, for integer numerators q over d on the group of the moduli."""
    N = len(q)
    difference = group(moduli)[0]
    red = _count(q, d, difference)
    blue = _count([d - x for x in q], d, difference)
    return Fraction(red + blue, N ** 4 * d ** 10)


# --- behavior readout -------------------------------------------------------
# The score is one number. These describe the SHAPE of the kernel instead, so two kernels with the same density can
# land in different cells of the diversity map.

_BEHAVIOR_ZERO = {"b_dispersion": 0.0, "b_boundary_frac": 0.0, "b_contact": 0.0,
                  "b_offset": 0.0, "b_concentration": 0.0, "b_structure": 0.0}


def _behavior(moduli, q, d):
    """Six bounded descriptors: spread of the fractional values, share of entries at 0 or 1, red edge density, the
    inside-part value q(0), weight of the red set on elements of order 2 or less, and the share of the order that is
    not a power of two. Never raises."""
    try:
        x = np.asarray(q, dtype=np.float64) / d
        frac = x[(x > 0) & (x < 1)]
        involution = group(moduli)[1] == np.arange(x.size)
        low = float(x[involution].sum() / max(x.sum(), 1e-300))
        order = math.prod(moduli)
        odd = order // (order & -order)
        out = {"b_dispersion": float(frac.std() * 2) if frac.size else 0.0,
               "b_boundary_frac": float(((x == 0) | (x == 1)).mean()), "b_contact": float(x.mean()),
               "b_offset": float(x[0]), "b_concentration": min(low, 1.0),
               "b_structure": math.log2(odd) / math.log2(order) if order > 1 else 0.0}
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
    moduli, q, d = _kernel(program.kernel())
    eval_time = time.time() - start_time
    value = density(moduli, q, d)
    combined = float(REFERENCE / value)
    cases = {"upper_bound": float(value), "inverse_bound": float(1 / value),
             "order": float(len(q)), "combined_score": combined, "validity": 1.0}
    cases.update(_behavior(moduli, q, d))
    witness = {
        "schema": "sciel-full-output/1",
        "source_sha256": source_sha, "evaluator_sha256": evaluator_sha,
        "invocation_id": invocation_id, "stage": stage, "confirmation": None,
        "input_sha256": None, "seed_sha256": None,
        "output": {"moduli": moduli, "numerators": q, "denominator": d, "density": [value.numerator, value.denominator]},
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
        "cases": {"combined_score": 0.0, "upper_bound": 0.0, "inverse_bound": 0.0, "order": 0.0,
                  "validity": 0.0, **_BEHAVIOR_ZERO},
        "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]},
    }
