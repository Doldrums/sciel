"""Evaluator for the Fourier entropy-influence constant from below; smoke/full/confirm on the sciel contract.

C is the least constant with H(f^2) <= C Inf(f) for every Boolean f: {-1, 1}^n -> {-1, 1} (Tao's optimization
constants, problem 71a), with spectral entropy H = sum_S fhat(S)^2 log2(1 / fhat(S)^2) over all S and total
influence Inf = sum_S fhat(S)^2 |S|. By the O'Donnell-Tan composition rule (arXiv 1304.1347; Hod 2017, Prop. 1.2) a
balanced g with Inf(g) > 1 proves C >= H(g) / (Inf(g) - 1): its iterated compositions have H / Inf tending to it.

The truth table follows the record certificates: entry x is 1 when f(x) = -1 (TRUE), and bit i of x set means
x_{i+1} = -1. The integer Walsh-Hadamard transform gives a_S = 2^n fhat(S) exactly, the influence is an exact
fraction over 4^n, and H = 2n - 2 sum_S a_S^2 log2|a_S| / 4^n is enclosed in outward-rounded intervals, with the
power-of-two part of each |a_S| exact. The score is the certified lower end of H / (Inf - 1).
"""

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

from sciel.candidate_modules import load_candidate
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time
import traceback
import uuid

from mpmath import iv
import numpy as np

# Numaro (July 2026, doi 10.5281/zenodo.21497769): a balanced monotone function on 18 variables with Inf = 261/128,
# the certified floor of its H / (Inf - 1).
REFERENCE = Fraction(65218457109230465756581439729485784683666, 10 ** 40)
MAX_N = 20
DIGITS = 60


def _load(program_path, module_name="program"):
    return load_candidate(program_path, module_name)


def _construction(result):
    """n and the truth table as uint8 bits, entry x = 1 where f(x) = -1."""
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise ValueError("boolean_function() must return a pair (n, table)")
    n, table = result
    if isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or not 1 <= int(n) <= MAX_N:
        raise ValueError(f"n must be an integer in 1..{MAX_N}")
    n = int(n)
    if isinstance(table, (bytes, bytearray)):
        table = np.frombuffer(bytes(table), dtype=np.uint8)
    t = np.asarray(table)
    if t.ndim != 1 or t.size != 1 << n:
        raise ValueError(f"table must be a 1-D sequence of 2**n = {1 << n} bits, got shape {t.shape}")
    if t.dtype.kind not in "biu":
        raise ValueError(f"table entries must be 0 or 1 (bool or integer), got dtype {t.dtype}")
    if ((t != 0) & (t != 1)).any():
        raise ValueError("table entries must be 0 or 1")
    return n, t.astype(np.uint8)


def popcounts(n) -> np.ndarray:
    """|S| for every subset S of [n], as an int64 array indexed by its bitmask."""
    idx = np.arange(1 << n, dtype=np.int64)
    pc = np.zeros(1 << n, dtype=np.int64)
    for i in range(n):
        pc += (idx >> i) & 1
    return pc


def spectrum(n, bits) -> np.ndarray:
    """The integers a_S = 2^n fhat(S), by the butterfly Walsh-Hadamard transform in int64."""
    a = 1 - 2 * np.asarray(bits, dtype=np.int64)
    h = 1
    while h < a.size:
        v = a.reshape(-1, 2, h)
        x, y = v[:, 0, :].copy(), v[:, 1, :]
        v[:, 0, :] += y
        v[:, 1, :] = x - y
        h *= 2
    return a


def influence(n, a) -> Fraction:
    return Fraction(int((a * a * popcounts(n)).sum()), 4 ** n)


def _endpoint(raw):
    sign, man, exp, _ = raw
    if man == 0:
        return Fraction(0)
    value = Fraction(int(man)) * Fraction(2) ** int(exp)
    return -value if sign else value


def entropy(n, a):
    """H enclosed as an mpmath interval: H = 2n - 2 sum_S a_S^2 log2|a_S| / 4^n, with |a_S| = 2^e o for odd o, so
    only log2 o is rounded, once per distinct o and outward."""
    v, counts = np.unique(np.abs(a[a != 0]), return_counts=True)
    weight = v * v * counts
    exact = 0
    odd_weight = {}
    for value, w in zip(v.tolist(), weight.tolist()):
        e = (value & -value).bit_length() - 1
        exact += w * e
        o = value >> e
        if o > 1:
            odd_weight[o] = odd_weight.get(o, 0) + w
    iv.dps = DIGITS
    ln2 = iv.log(2)
    total = iv.mpf(exact)
    for o, w in odd_weight.items():
        total += iv.mpf(w) * iv.log(o) / ln2
    return iv.mpf(2 * n) - 2 * total / iv.mpf(4 ** n)


def certify(n, bits) -> dict:
    """Balance, Inf > 1 and H > 0 checked exactly, then H / (Inf - 1) as certified [lower, upper] fractions."""
    a = spectrum(n, bits)
    if int((a * a).sum()) != 4 ** n:
        raise ValueError("Parseval fails; the transform overflowed")
    if a[0] != 0:
        raise ValueError(f"f must be balanced (fhat(empty) = 0), got a_0 = {int(a[0])} (FALSE minus TRUE entries)")
    inf = influence(n, a)
    if inf <= 1:
        raise ValueError(f"the composition rule needs Inf > 1, got {inf}")
    H = entropy(n, a)
    ratio = H / (iv.mpf(inf.numerator - inf.denominator) / iv.mpf(inf.denominator))
    lo, hi = _endpoint(ratio._mpi_[0]), _endpoint(ratio._mpi_[1])
    h_lo, h_hi = _endpoint(H._mpi_[0]), _endpoint(H._mpi_[1])
    if h_lo <= 0:
        raise ValueError("H must be positive")
    return {"a": a, "influence": inf, "entropy": (h_lo, h_hi), "bound": (lo, hi)}


def encode(n, bits) -> str:
    """The truth table as the hex of the integer with bit x set where f(x) = -1, 2^n / 4 digits; the certificates'
    true_hex."""
    mask = int.from_bytes(np.packbits(np.asarray(bits, dtype=np.uint8), bitorder="little").tobytes(), "little")
    return format(mask, f"0{max(1, (1 << n) // 4)}x")


def decode(n, text) -> np.ndarray:
    N = 1 << n
    raw = int(text, 16).to_bytes((N + 7) // 8, "little")
    return np.unpackbits(np.frombuffer(raw, dtype=np.uint8), bitorder="little")[:N].copy()


def table_sha256(bits) -> str:
    """SHA-256 of the comma-joined +-1 table, the binding the record certificates publish."""
    return hashlib.sha256(",".join(np.where(np.asarray(bits) == 1, "-1", "1").tolist()).encode()).hexdigest()


def _decimal(q, floor, digits=40):
    """A positive fraction rounded down (floor) or up to digits decimals."""
    scaled = q.numerator * 10 ** digits // q.denominator if floor else -(-q.numerator * 10 ** digits // q.denominator)
    s = str(scaled).rjust(digits + 1, "0")
    return f"{s[:-digits]}.{s[-digits:]}"


# --- behavior readout -------------------------------------------------------
# The score is one number. These describe the SHAPE of the function instead, so two
# functions with the same bound can land in different cells of the diversity map.

_BEHAVIOR_ZERO = {"b_structure": 0.0, "b_sparsity": 0.0, "b_level1": 0.0,
                  "b_max_influence": 0.0, "b_degree": 0.0, "b_monotone": 0.0}


def _behavior(n, bits, a):
    """Six bounded descriptors: variables used of MAX_N, support size of the spectrum, level-1 weight, largest
    single-variable influence, mean level per variable, and the share of variables f is monotone in (either
    direction). Never raises."""
    try:
        N = 1 << n
        w = (a * a).astype(np.float64) / float(N) ** 2
        idx = np.arange(N, dtype=np.int64)
        pc = popcounts(n)
        nnz = int(np.count_nonzero(a))
        per_var = [float(w[((idx >> i) & 1) == 1].sum()) for i in range(n)]
        t = np.asarray(bits, dtype=np.int8)
        unate = 0
        for i in range(n):
            low = idx[((idx >> i) & 1) == 0]
            step = t[low | (1 << i)] - t[low]
            unate += int(not ((step > 0).any() and (step < 0).any()))
        out = {"b_structure": n / MAX_N, "b_sparsity": float(np.log2(max(nnz, 1)) / n),
               "b_level1": float(w[pc == 1].sum()), "b_max_influence": min(max(per_var), 1.0),
               "b_degree": float((w * pc).sum() / n), "b_monotone": unate / n}
        return {k: (float(v) if np.isfinite(v) else 0.0) for k, v in out.items()}
    except (ValueError, FloatingPointError, ZeroDivisionError, OverflowError, IndexError):
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
    if not hasattr(program, "boolean_function"):
        raise AttributeError("missing boolean_function")
    start_time = time.time()
    n, bits = _construction(program.boolean_function())
    eval_time = time.time() - start_time
    cert = certify(n, bits)
    lo, hi = cert["bound"]
    inf = cert["influence"]
    cases = {"lower_bound": float(lo), "entropy": float(cert["entropy"][0]), "influence": float(inf),
             "n": float(n), "nonzero": float(np.count_nonzero(cert["a"])),
             "combined_score": float(lo / REFERENCE), "validity": 1.0}
    cases.update(_behavior(n, bits, cert["a"]))
    witness = {
        "schema": "sciel-full-output/1",
        "source_sha256": source_sha, "evaluator_sha256": evaluator_sha,
        "invocation_id": invocation_id, "stage": stage, "confirmation": None,
        "input_sha256": None, "seed_sha256": None,
        "output": {"n": n, "true_hex": encode(n, bits), "sha256_table": table_sha256(bits),
                   "influence": [inf.numerator, inf.denominator],
                   "bound_floor": _decimal(lo, floor=True), "bound_ceiling": _decimal(hi, floor=False)},
    }
    artifacts = {"eval_time": f"{eval_time:.4f}", "full_output": _canonical(witness)}
    if source_sha != _source_hash(program_path) or evaluator_sha != _source_hash(__file__):
        raise ValueError("source or evaluator changed during invocation")
    return {"ok": True, "score": cases["combined_score"], "cases": cases, "artifacts": artifacts}


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
        "cases": {"combined_score": 0.0, "lower_bound": 0.0, "entropy": 0.0, "influence": 0.0, "n": 0.0,
                  "nonzero": 0.0, "validity": 0.0, **_BEHAVIOR_ZERO},
        "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]},
    }
