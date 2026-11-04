"""Evaluator for the Ramsey multiplicity constant of K4; smoke/full/confirm on the sciel contract.

The step graphon has m = 3840 equal parts and a fixed group G of permutations of the parts (seven generators in
assets/frontier.npz, automorphisms of the Van Seters-Jothimani construction and its refinements). The candidate
gives one red probability per class of part pairs, a class being a G-orbit of ordered pairs joined with its
transpose, so W is symmetric and G-invariant. Colouring K_n by sampling a part per vertex gives expected
monochromatic K4 density t(K4, W) + t(K4, 1 - W), so this bounds c4 from above. With

    t(K4, W) = m^-4 sum_{i,j} W_ij s_ij,   s_ij = sum_{k,l} W_ik W_jk W_il W_jl W_kl,

s is constant on a class, so the sum runs over one representative (i, j) per class, weighted by the class size,
with i one of the 128 vertex-orbit representatives; only k, l where W_ik W_jk W_il W_jl is nonzero contribute. With
integer numerators Q over a denominator D the numerator is an integer below 2 m^4 D^6; it is computed modulo primes
p with m p^2 < 2^53, where binary64 matrix products are exact, and recombined by CRT, with one more prime as a check.
"""

from sciel.candidate_modules import load_candidate
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import time
import traceback
import uuid

import numpy as np

# Second L-BFGS round of vidigoat on J. SK's graphon (teorth/optimizationproblems PR 270, optimization-records commit
# 9f82508), reproduced exactly by this evaluator.
REFERENCE = Fraction(31383468324407329838135970970371081393990727093313271428891812741,
                     1041295852679825698551191435837073366434387539971249757275291648000)
MAX_DENOMINATOR = 2 ** 30
STRUCTURE = Path(__file__).resolve().parent / "assets" / "frontier.npz"


def _load(program_path, module_name="program"):
    return load_candidate(program_path, module_name)


@lru_cache(maxsize=1)
def structure():
    """(classes, rows, cols, sizes): classes[i, j] is the class of the pair (i, j), numbered by first appearance in
    row-major order; (rows[c], cols[c]) is the representative of class c and sizes[c] its number of ordered pairs."""
    generators = np.load(STRUCTURE)["generators"].astype(np.int64)
    m = generators.shape[1]
    label = np.arange(m * m, dtype=np.int64).reshape(m, m)
    while True:
        new = label
        for g in generators:
            new = np.minimum(new, label[np.ix_(g, g)])
        new = new.ravel()[new.ravel()].reshape(m, m)
        if (new == label).all():
            break
        label = new
    label = np.minimum(label, label.T)
    _, first, classes = np.unique(label.ravel(), return_index=True, return_inverse=True)
    order = np.argsort(first)
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    classes = rank[classes].reshape(m, m).astype(np.int32)
    sizes = np.bincount(classes.ravel())
    vertex = np.diag(classes)
    heads = np.unique([np.nonzero(vertex == v)[0][0] for v in np.unique(vertex)])
    flat = classes[heads].ravel()
    _, at = np.unique(flat, return_index=True)
    rows, cols = heads[at // m], at % m
    if rows.size != sizes.size:
        raise ArithmeticError("a class has no representative in a vertex-orbit row")
    for a in (classes, rows, cols, sizes):
        a.setflags(write=False)
    return classes, rows, cols, sizes


def _kernel(value):
    """The candidate's per-class red probabilities as integer numerators over one denominator. A pair
    (numerators, denominator) is taken as given; an array of binary64 values must be multiples of 2**-30."""
    count = structure()[3].size
    if isinstance(value, tuple) and len(value) == 2:
        numerators, denominator = value
        if isinstance(denominator, bool) or not isinstance(denominator, (int, np.integer)):
            raise ValueError("the denominator must be an integer")
        denominator = int(denominator)
        if not 1 <= denominator <= MAX_DENOMINATOR:
            raise ValueError(f"the denominator must be in 1..2**30, got {denominator}")
        raw = list(numerators)
        if any(isinstance(x, bool) or not isinstance(x, (int, np.integer)) for x in raw):
            raise ValueError("numerators must be integers")
        q = [int(x) for x in raw]
    else:
        h = np.array(value, dtype=np.float64, copy=True)
        if h.ndim != 1 or not np.isfinite(h).all():
            raise ValueError("q must be a 1-D sequence of finite numbers")
        scaled = np.ldexp(h, 30)
        if (scaled != np.round(scaled)).any():
            raise ValueError("binary64 values of q must be multiples of 2**-30; return (numerators, denominator) "
                             "or use graphon.snap(q)")
        q, denominator = [int(x) for x in scaled], MAX_DENOMINATOR
        g = math.gcd(denominator, *q)
        q, denominator = [x // g for x in q], denominator // g
    if len(q) != count:
        raise ValueError(f"q must have one entry per class, {count}, got {len(q)}")
    if any(x < 0 or x > denominator for x in q):
        raise ValueError("q must lie in [0, 1]")
    return q, denominator


def _primes(count, m):
    """The largest primes p with m p^2 < 2^53."""
    out, p = [], math.isqrt(2 ** 53 // m)
    while len(out) < count:
        p -= 1
        if all(p % d for d in range(2, math.isqrt(p) + 1)):
            out.append(p)
    return out


def _residue(q, p):
    """sum_{i,j} Q_ij s_ij mod p over red and blue, for integer class numerators q (red) and d - q (blue) given as
    a pair of int64 arrays."""
    classes, rows, cols, sizes = structure()
    total = 0
    for colour in q:
        M = colour[classes]
        Mf = (M % p).astype(np.float64)
        for i in np.unique(rows):
            mine = np.nonzero((rows == i) & (colour != 0))[0]
            if not mine.size:
                continue
            S = np.nonzero(M[i])[0]
            X = np.fmod(Mf[np.ix_(cols[mine], S)] * Mf[i, S], p)
            Y = np.fmod(X @ Mf[np.ix_(S, S)], p)
            s = np.fmod((Y * X).sum(axis=1), p).astype(np.int64)
            w = (colour[mine] % p) * (sizes[mine] % p) % p
            total += int((s * w % p).sum())
    return total % p


def numerator(q, d):
    """sum_{i,j} (Q_ij s_ij + B_ij s'_ij) exactly, B = d - Q, for integer class numerators q over d."""
    m = structure()[0].shape[0]
    red = np.asarray(q, dtype=np.int64)
    bits = (2 * m ** 4 * d ** 6).bit_length() + 1
    primes = _primes(bits // (math.isqrt(2 ** 53 // m).bit_length() - 1) + 2, m)
    # Primes are independent and run in fresh processes, one group per core: a forked child of a process that has
    # already used a threaded BLAS can hang.
    workers = max(1, min(len(primes), os.cpu_count() or 1))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "q.npy")
        np.save(path, np.stack([red, d - red]))
        procs = [subprocess.Popen([sys.executable, __file__, path, *map(str, primes[k::workers])],
                                  stdout=subprocess.PIPE, text=True) for k in range(workers)]
        found = {}
        for proc in procs:
            out, _ = proc.communicate()
            if proc.returncode:
                raise RuntimeError("a residue worker failed")
            found.update((int(a), int(b)) for a, b in (line.split() for line in out.splitlines()))
    residues = [found[p] for p in primes]
    value, modulus = 0, 1
    for r, p in zip(residues[:-1], primes[:-1]):
        value += modulus * ((r - value) * pow(modulus, -1, p) % p)
        modulus *= p
    if modulus.bit_length() <= bits or value % primes[-1] != residues[-1]:
        raise ArithmeticError("CRT check prime disagrees")
    return value


def density(q, d) -> Fraction:
    """t(K4, W) + t(K4, 1 - W) exactly, for integer class numerators q over d."""
    m = structure()[0].shape[0]
    return Fraction(numerator(q, d), m ** 4 * d ** 6)


# --- behavior readout -------------------------------------------------------
# The score is one number. These describe the SHAPE of the graphon instead, so two graphons with the same density
# can land in different cells of the diversity map.

_BEHAVIOR_ZERO = {"b_dispersion": 0.0, "b_boundary_frac": 0.0, "b_contact": 0.0,
                  "b_offset": 0.0, "b_concentration": 0.0, "b_structure": 0.0}


def _behavior(q, d):
    """Six bounded descriptors over the m^2 pairs: spread of the fractional values, share at 0 or 1, red edge
    density, mean inside-part value, share of mass within 2^-10 of 0 or 1 among fractional pairs, and the share of
    classes that are fractional. Never raises."""
    try:
        _, rows, _, sizes = structure()
        x = np.asarray(q, dtype=np.float64) / d
        w = sizes / sizes.sum()
        frac = (x > 0) & (x < 1)
        diag = np.isin(np.arange(x.size), np.unique(structure()[0].diagonal()))
        mean = float((w * x)[frac].sum() / max(w[frac].sum(), 1e-300))
        spread = math.sqrt(max(float((w * (x - mean) ** 2)[frac].sum() / max(w[frac].sum(), 1e-300)), 0.0))
        near = frac & ((x < 2 ** -10) | (x > 1 - 2 ** -10))
        out = {"b_dispersion": min(2 * spread, 1.0), "b_boundary_frac": float(w[~frac].sum()),
               "b_contact": float(w @ x), "b_offset": float(x[diag].mean()),
               "b_concentration": float(w[near].sum() / max(w[frac].sum(), 1e-300)),
               "b_structure": float(frac.mean())}
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
    kernel = program.kernel()
    eval_time = time.time() - start_time
    q, d = _kernel(kernel)
    value = density(q, d)
    combined = float(REFERENCE / value)
    cases = {"upper_bound": float(value), "inverse_bound": float(1 / value),
             "fractional_classes": float(sum(0 < x < d for x in q)), "combined_score": combined, "validity": 1.0}
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
        "cases": {"combined_score": 0.0, "upper_bound": 0.0, "inverse_bound": 0.0, "fractional_classes": 0.0,
                  "validity": 0.0, **_BEHAVIOR_ZERO},
        "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]},
    }


if __name__ == "__main__":
    _pair = tuple(np.load(sys.argv[1]))
    for _p in map(int, sys.argv[2:]):
        print(_p, _residue(_pair, _p), flush=True)
