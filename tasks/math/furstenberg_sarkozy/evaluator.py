"""Evaluator for the Furstenberg-Sarkozy square-difference exponent (Tao's 4b); smoke/full/confirm on the sciel contract.

Naslund's interval-moment criterion: take components in pairwise coprime square bases b_i, finite digit sets C_i
mod b_i and an interval [a_x, a_x + w_x] in [0, 1] (0 < w_x < 1) per digit, with a_x + w_x <= a_y whenever
y - x is a nonzero square mod b_i. If moment powers f_i satisfy sum_x w_x^f_i >= b_i^alpha and sum_i f_i > alpha,
then square-difference-free A in [1, N] of size N^(alpha - o(1)) exist, so C_4b >= alpha. Three component kinds:

  chain   a prime p = 3 mod 4 and digits c_1..c_t mod p with c_j - c_i a nonzero square exactly when j > i; equal
          widths 1/t on the low digit, the next digit free (base p^2), so f = 1 - (2 alpha - 1) log p / log t;
  code    odd primes p_k with low depths e_k: points are tuples of low words (digit 0 first), x -> y when in every
          prime the first differing digit of y minus that of x is a square mod p_k (or the words agree); rational
          starts and widths, high digits free (base prod p_k^(2 e_k)), so the moment test is
          log sum_x w_x^f >= (2 alpha - 1) sum_k e_k log p_k;
  binary  Naslund's grid-policy recursion at 2: states with windows and four residue branches (scale, shift, swap,
          reflect, child); exact containment and ordering of the placed windows, a positive vector v with
          sum_branches u^f v_child > a v_state for every state, log a > alpha log 4, and the fixed-depth surplus.

Every power x^f is replaced by a proved rational lower bound and every logarithm by a directed 2^-256 fixed-point
enclosure, so a pass is a proof. combined_score = alpha / ALPHA_BEST.
"""

from sciel.candidate_modules import load_candidate
from decimal import Decimal, localcontext, ROUND_FLOOR
from fractions import Fraction
from functools import lru_cache
import hashlib
import json
import math
from math import isqrt
from pathlib import Path
import time
import traceback
import uuid

# Naslund, Hexagon 2610.00151v1 (teorth/optimizationproblems constants/4b.md, [Nas2026b]); reproduced exactly here.
ALPHA_BEST = Fraction(473797394875551, 625000000000000)
MAX_CODE_POINTS = 40000
MAX_STATES = 400

BITS = 256
SCALE = 1 << BITS
TERMS = 90


def _ceildiv(a, b):
    return -(-a // b)


def _unit_log(n, d):
    """Integers (lo, hi) with lo <= SCALE log(n/d) <= hi, for 1 <= n/d <= 2: 2 atanh z, z = (n-d)/(n+d) <= 1/3."""
    zlo, zhi = (n - d) * SCALE // (n + d), _ceildiv((n - d) * SCALE, n + d)
    zzlo, zzhi = zlo * zlo // SCALE, _ceildiv(zhi * zhi, SCALE)
    plo, phi, slo, shi = zlo, zhi, 0, 0
    for j in range(TERMS):
        slo += plo // (2 * j + 1)
        shi += _ceildiv(phi, 2 * j + 1)
        plo, phi = plo * zzlo // SCALE, _ceildiv(phi * zzhi, SCALE)
    return 2 * slo, 2 * shi + _ceildiv(2 * phi * SCALE, (2 * TERMS + 1) * (SCALE - zzhi))


_LOG2 = _unit_log(2, 1)


@lru_cache(maxsize=None)
def log_bounds(x):
    """Fractions lo <= log x <= hi, for a positive Fraction x."""
    if x <= 0:
        raise ValueError("log of a non-positive number")
    n, d = x.numerator, x.denominator
    k = n.bit_length() - d.bit_length()
    if k >= 0:
        d <<= k
    else:
        n <<= -k
    if n < d:
        n, k = 2 * n, k - 1
    lo, hi = _unit_log(n, d)
    if k >= 0:
        lo, hi = lo + k * _LOG2[0], hi + k * _LOG2[1]
    else:
        lo, hi = lo + k * _LOG2[1], hi + k * _LOG2[0]
    return Fraction(lo, SCALE), Fraction(hi, SCALE)


@lru_cache(maxsize=None)
def power_lower(x, f):
    """A rational y <= x^f, proved by log y <= f log x, for 0 < x <= 1 and 0 < f < 1."""
    if x == 1:
        return Fraction(1)
    with localcontext() as ctx:
        ctx.prec = 90
        approx = ((Decimal(x.numerator) / Decimal(x.denominator)).ln()
                  * Decimal(f.numerator) / Decimal(f.denominator)).exp()
        y = Fraction(int((approx * 10 ** 60).to_integral_value(rounding=ROUND_FLOOR)) - 1, 10 ** 60)
    if not (y > 0 and log_bounds(y)[1] <= f * log_bounds(x)[0]):
        raise ArithmeticError("power lower bound not proved")
    return y


def _rational(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, str, Fraction)):
        raise ValueError(f"{name} must be an integer, a Fraction or a decimal/fraction string")
    return Fraction(value)


def _int(value, name, lo=None):
    if isinstance(value, bool) or not isinstance(value, int) or (lo is not None and value < lo):
        raise ValueError(f"{name} must be an integer" + ("" if lo is None else f" >= {lo}"))
    return value


def _prime(p):
    return p > 2 and all(p % d for d in range(2, isqrt(p) + 1))


def check_chain(p, chain, alpha):
    """Lower bound on the chain's moment power 1 - (2 alpha - 1) log p / log t."""
    p = _int(p, "chain prime", 3)
    if not _prime(p) or p % 4 != 3:
        raise ValueError(f"chain modulus {p} must be a prime 3 mod 4")
    chain = [_int(x, "chain digit", 0) for x in chain]
    if len(chain) < 2 or len(set(chain)) != len(chain) or max(chain) >= p:
        raise ValueError(f"chain mod {p} must have at least two distinct digits in 0..p-1")
    squares = {x * x % p for x in range(1, p)}
    for i, x in enumerate(chain):
        for j, y in enumerate(chain):
            if i != j and ((y - x) % p in squares) != (j > i):
                raise ValueError(f"chain mod {p}: {y} - {x} breaks the order")
    reward = 1 - (2 * alpha - 1) * log_bounds(Fraction(p))[1] / log_bounds(Fraction(len(chain)))[0]
    return reward


def incoming(primes, words):
    """Bitmask per point of the points x with y - x a square in every prime coordinate (first differing digit)."""
    n = len(words)
    out = [(1 << n) - 1] * n
    for z, p in enumerate(primes):
        qr = {a * a % p for a in range(1, p)}
        groups = {}
        for i, point in enumerate(words):
            word = point[z]
            for d in range(1, len(word) + 1):
                groups[word[:d]] = groups.get(word[:d], 0) | 1 << i
        memo = {}
        for i, point in enumerate(words):
            word = point[z]
            if word not in memo:
                mask = groups[word]
                for d, b in enumerate(word):
                    for q in qr:
                        mask |= groups.get(word[:d] + ((b - q) % p,), 0)
                memo[word] = mask
            out[i] &= memo[word]
    return out


def check_code(code, alpha):
    """Exact geometry of an odd low-digit code and its moment test; returns the power f."""
    primes = [_int(p, "code prime", 3) for p in code["primes"]]
    depths = [_int(e, "code depth", 1) for e in code["depths"]]
    points = code["points"]
    widths = [_int(w, "width numerator", 1) for w in code["width_numerators"]]
    starts = [_int(s, "start numerator", 0) for s in code["start_numerators"]]
    D = _int(code["width_denominator"], "width denominator", 1)
    f = _rational(code["power"], "code power")
    n = len(points)
    if not primes or len(primes) != len(set(primes)) or len(depths) != len(primes) or not all(map(_prime, primes)):
        raise ValueError("code primes must be distinct odd primes, one depth each")
    if not 0 < n <= MAX_CODE_POINTS or len(widths) != n or len(starts) != n:
        raise ValueError(f"a code has 1..{MAX_CODE_POINTS} points, one width and start each")
    if not 0 < f < 1:
        raise ValueError("code power must lie in (0, 1)")
    words = []
    for point in points:
        if len(point) != len(primes):
            raise ValueError("a point has one word per prime")
        point = tuple(tuple(_int(a, "digit", 0) for a in word) for word in point)
        if any(len(word) != e or any(a >= p for a in word) for word, p, e in zip(point, primes, depths)):
            raise ValueError("a word has the prime's depth and digits below the prime")
        words.append(point)
    if len(set(words)) != n:
        raise ValueError("code points must be distinct")
    if any(w >= D or s + w > D for s, w in zip(starts, widths)):
        raise ValueError("intervals must lie in [0, 1] with width below 1")
    incoming_ = incoming(primes, words)
    ends = [s + w for s, w in zip(starts, widths)]
    order = sorted(range(n), key=ends.__getitem__)
    cursor, early = 0, (1 << n) - 1
    for j in sorted(range(n), key=starts.__getitem__):
        while cursor < n and ends[order[cursor]] <= starts[j]:
            early &= ~(1 << order[cursor])
            cursor += 1
        if incoming_[j] & ~(1 << j) & early:
            raise ValueError("a square difference between code points is not ordered by their intervals")
    cache = {}
    for w in widths:
        if w not in cache:
            cache[w] = power_lower(Fraction(w, D), f)
    total = sum((cache[w] for w in widths), Fraction(0))
    log_free = sum(e * log_bounds(Fraction(p))[1] for p, e in zip(primes, depths))
    if not log_bounds(total)[0] > (2 * alpha - 1) * log_free:
        raise ValueError(f"code on {primes}: moment test fails at power {f}")
    return f


def check_binary(data, alpha):
    """Naslund's binary grid-policy component at 2; returns its power f."""
    states = data["states"]
    if not 0 < len(states) <= MAX_STATES:
        raise ValueError(f"binary component has 1..{MAX_STATES} states")
    f, a = _rational(data["f"], "binary f"), _rational(data["a"], "binary a")
    depth, root = _int(data["depth"], "binary depth", 2), _int(data["root"], "binary root", 0)
    v = [_rational(x, "binary vector") for x in data["vector"]]
    if not (0 < f < 1 and a > 1 and len(v) == len(states) and min(v) > 0 and root < len(states) and v[root] == 1):
        raise ValueError("binary: need 0 < f < 1, a > 1 and a positive vector equal to 1 at the root")
    windows = []
    for s in states:
        lo, hi, lo2, hi2, q = (_int(x, "window") for x in s["windows"])
        if q <= 0:
            raise ValueError("window denominator must be positive")
        W = [(Fraction(lo, q), Fraction(hi, q)), None]
        if lo2 >= 0:
            W[1] = (Fraction(lo2, q), Fraction(hi2, q))
        elif not lo2 == hi2 == -1:
            raise ValueError("an absent second window is -1, -1")
        if any(w is not None and not 0 <= w[0] < w[1] <= 1 for w in W):
            raise ValueError("windows lie in [0, 1]")
        windows.append(W)
    key = states[root]["windows"]
    if not (key[0] == key[2] == 0 and key[1] == key[3] == key[4]):
        raise ValueError("the root has both windows [0, 1]")
    for i, s in enumerate(states):
        placed_at, residues = {}, set()
        for b in s["branches"]:
            r, child = _int(b["r"], "residue", 0), _int(b["child"], "child", 0)
            if r > 3 or r in residues or child >= len(states) or b["swap"] not in (0, 1) or b["reflect"] not in (0, 1):
                raise ValueError("binary branch: distinct residues 0..3, a valid child, swap and reflect in {0, 1}")
            residues.add(r)
            u, t = Fraction(*b["scale"]), Fraction(*b["shift"])
            if u <= 0:
                raise ValueError("branch scale must be positive")
            for p in range(2):
                w = windows[child][p ^ b["swap"]]
                if w is None:
                    continue
                if b["reflect"]:
                    w = (1 - w[1], 1 - w[0])
                placed, parent = (t + u * w[0], t + u * w[1]), windows[i][r % 2]
                if parent is None or not parent[0] <= placed[0] < placed[1] <= parent[1]:
                    raise ValueError(f"binary state {i}: a placed window leaves its parent window")
                placed_at[r + 4 * p] = placed
        for j, w in placed_at.items():
            nxt = placed_at.get((j + 1) % 8)
            if nxt is not None and not w[1] <= nxt[0]:
                raise ValueError(f"binary state {i}: placed windows {j}, {(j + 1) % 8} out of order")
    for i, s in enumerate(states):
        lhs = sum((power_lower(Fraction(*b["scale"]), f) * v[b["child"]] for b in s["branches"]), Fraction(0))
        if not lhs > a * v[i]:
            raise ValueError(f"binary growth row {i} fails")
    la, l4, l2 = log_bounds(a)[0], log_bounds(Fraction(4))[1], log_bounds(Fraction(2))[1]
    if not la > alpha * l4:
        raise ValueError("binary growth a is not above 4^alpha")
    c = min(max(Fraction(s["windows"][1] - s["windows"][0], s["windows"][4]),
                Fraction(0) if s["windows"][2] < 0 else Fraction(s["windows"][3] - s["windows"][2], s["windows"][4]))
            / v[i] for i, s in enumerate(states))
    if not log_bounds(c)[0] + (depth - 1) * la - alpha * depth * l4 - f * l2 > 0:
        raise ValueError("binary fixed-depth surplus fails")
    return f


def certify(cert):
    """The exponent alpha proved by the certificate (an exact Fraction) and its parts; raises if any check fails."""
    alpha = _rational(cert["alpha"], "alpha")
    if not Fraction(1, 2) < alpha < 1:
        raise ValueError("alpha must lie in (1/2, 1)")
    chains, codes, binary = cert.get("chains") or [], cert.get("codes") or [], cert.get("binary")
    moduli = [2] if binary else []
    moduli += [p for p, _ in chains] + [p for c in codes for p in c["primes"]]
    if len(moduli) != len(set(moduli)) or not moduli:
        raise ValueError("components need pairwise coprime bases: every prime used once")
    parts = {"chains": [check_chain(p, c, alpha) for p, c in chains],
             "codes": [check_code(c, alpha) for c in codes],
             "binary": [check_binary(binary, alpha)] if binary else []}
    surplus = sum(parts["chains"]) + sum(parts["codes"]) + sum(parts["binary"]) - alpha
    if not surplus > 0:
        raise ValueError(f"contribution budget fails: sum f_i - alpha = {float(surplus):.3e}")
    return alpha, parts, surplus


# --- behavior readout -------------------------------------------------------
_BEHAVIOR_ZERO = {"b_dispersion": 0.0, "b_boundary_frac": 0.0, "b_contact": 0.0,
                  "b_offset": 0.0, "b_concentration": 0.0, "b_structure": 0.0}


def _behavior(cert, parts, surplus):
    """Shares of the exponent from chains, codes and the binary part, component count, code width spread and the
    log of the surplus. Never raises."""
    try:
        total = float(sum(parts["chains"]) + sum(parts["codes"]) + sum(parts["binary"]))
        widths = [w / c["width_denominator"] for c in cert.get("codes") or [] for w in c["width_numerators"]]
        mean = sum(widths) / len(widths) if widths else 0.0
        spread = (sum((w - mean) ** 2 for w in widths) / len(widths)) ** 0.5 / mean if widths else 0.0
        count = sum(len(v) for v in parts.values())
        out = {"b_dispersion": spread / (1 + spread), "b_boundary_frac": float(sum(parts["chains"])) / total,
               "b_contact": float(sum(parts["codes"])) / total, "b_offset": float(sum(parts["binary"])) / total,
               "b_concentration": min(max(-math.log10(float(surplus)) / 30, 0.0), 1.0),
               "b_structure": min(count / 16, 1.0)}
        return {k: (float(v) if v == v else 0.0) for k, v in out.items()}
    except (ValueError, ZeroDivisionError, OverflowError, KeyError, TypeError):
        return dict(_BEHAVIOR_ZERO)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False, default=str)


def _source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _evaluate(program_path, stage):
    source_sha = _source_hash(program_path)
    evaluator_sha = _source_hash(__file__)
    invocation_id = uuid.uuid4().hex
    program = load_candidate(program_path, "program_" + stage)
    if not hasattr(program, "certificate"):
        raise AttributeError("missing certificate")
    start_time = time.time()
    cert = program.certificate()
    eval_time = time.time() - start_time
    cert = json.loads(_canonical(cert))
    alpha, parts, surplus = certify(cert)
    combined = float(alpha / ALPHA_BEST)
    cases = {"alpha": float(alpha), "surplus": float(surplus), "components": float(sum(len(v) for v in parts.values())),
             "combined_score": combined, "validity": 1.0}
    cases.update(_behavior(cert, parts, surplus))
    witness = {
        "schema": "sciel-full-output/1",
        "source_sha256": source_sha, "evaluator_sha256": evaluator_sha,
        "invocation_id": invocation_id, "stage": stage, "confirmation": None,
        "input_sha256": None, "seed_sha256": None,
        "output": {"certificate": cert, "alpha": [alpha.numerator, alpha.denominator]},
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
        "cases": {"combined_score": 0.0, "alpha": 0.0, "surplus": 0.0, "components": 0.0, "validity": 0.0,
                  **_BEHAVIOR_ZERO},
        "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]},
    }
