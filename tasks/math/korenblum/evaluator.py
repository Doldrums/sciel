"""Evaluator for Korenblum's constant from above; smoke/full/confirm on the sciel contract.

Korenblum's constant kappa is the largest c in (0, 1) such that |f| <= |g| on the annulus c < |z| < 1 implies
||f|| <= ||g|| for all f, g in the Bergman space A^2 of the unit disk, ||f||^2 = (1/pi) int_D |f|^2 dA (Tao's
optimization constants, problem 68). A pair of polynomials with |f| <= |g| on c < |z| < 1 and ||f|| > ||g|| proves
kappa <= c. Wang (2008) proves kappa <= 0.6778994...: z^10 + a fails to be dominated by z(1 + a z^10) exactly when
a > 2/3, and c -> 0.67789942295 as a -> 2/3.

The candidate gives c and the real rational coefficients of f and g. Every step is certified:
  * ||f||^2 - ||g||^2 = sum_n (f_n^2 - g_n^2) / (n + 1) must be positive, in exact rationals;
  * on |z| = 1 and |z| = c, |g|^2 - |f|^2 is the cosine polynomial sum_k d_k cos(k theta) with d_k the correlations
    of the coefficients scaled by r^n; it must vanish identically (checked exactly, as polynomial identity at |z| = 1)
    or be positive, by bisection in theta with a third-order Taylor bound and every binary64 rounding bounded;
  * g has no zeros on c <= |z| <= 1: |g|^2 is certified positive on both circles the same way, and g's winding
    numbers on them, the counts of its zeros inside, agree; steps are short enough that no sample-to-sample piece of
    the curve can reach 0.
Then f/g is analytic on a neighbourhood of the closed annulus and |f/g| <= 1 on its boundary, so |f/g| <= 1 inside
by the maximum principle. c is rounded up to binary64 and scored as the upper bound.
"""

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

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

# Wang, Integral Equations Operator Theory 61 (2008) 423-432, as recorded in Tao's table and Chakraborty-Solynin.
REFERENCE = 0.6778994
MAX_DEGREE = 400
# Numerators and denominators of c and of the coefficients; any binary64 value fits.
MAX_BITS = 1100
# The common denominator of all coefficients; binary64 values share one power of two.
MAX_SCALE_BITS = 4096
U = 2.0 ** -53
# Bisection gives up after this many interval evaluations on one circle.
MAX_INTERVALS = 400000
MAX_SAMPLES = 2000000


def _load(program_path, module_name="program"):
    return load_candidate(program_path, module_name)


def exact(value) -> Fraction:
    """A real number as an exact rational: int, binary64, Fraction or a string like '2/3'."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError("booleans are not numbers")
    if isinstance(value, (int, np.integer)):
        out = Fraction(int(value))
    elif isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError("values must be finite")
        out = Fraction(float(value))
    elif isinstance(value, (Fraction, str)):
        out = Fraction(value)
    else:
        raise ValueError(f"cannot read {type(value).__name__} as an exact real number")
    if out.numerator.bit_length() > MAX_BITS or out.denominator.bit_length() > MAX_BITS:
        raise ValueError(f"numerators and denominators are limited to {MAX_BITS} bits")
    return out


def _polynomial(values, name):
    if isinstance(values, (str, bytes)) or not hasattr(values, "__len__"):
        raise ValueError(f"{name} must be a sequence of coefficients, constant term first")
    if not 1 <= len(values) <= MAX_DEGREE + 1:
        raise ValueError(f"{name} must have 1..{MAX_DEGREE + 1} coefficients")
    p = [exact(v) for v in values]
    while len(p) > 1 and p[-1] == 0:
        p.pop()
    if not 1 <= len(p) <= MAX_DEGREE + 1:
        raise ValueError(f"{name} must have degree at most {MAX_DEGREE}")
    if not any(p):
        raise ValueError(f"{name} must not be zero")
    if any(not math.isfinite(float(x)) for x in p):
        raise ValueError(f"{name} has a coefficient outside the binary64 range")
    return p


def construction(result):
    """c rounded up to binary64 and the exact coefficient lists of f and g."""
    if not isinstance(result, (tuple, list)) or len(result) != 3:
        raise ValueError("korenblum_pair() must return (c, f, g)")
    c = exact(result[0])
    if not 0 < c < 1:
        raise ValueError("c must lie in (0, 1)")
    r = float(c)
    if Fraction(r) < c:
        r = math.nextafter(r, 1.0)
    if not 0 < r < 1:
        raise ValueError("c must lie in (0, 1) after rounding up to binary64")
    return r, _polynomial(result[1], "f"), _polynomial(result[2], "g")


def norm2(p) -> Fraction:
    """Exact ||p||^2 = sum |p_n|^2 / (n + 1) in A^2 with normalized area measure."""
    return sum((Fraction(x) ** 2 / (n + 1) for n, x in enumerate(p)), Fraction(0))


def _integers(polys):
    scale = math.lcm(*(x.denominator for p in polys for x in p))
    if scale.bit_length() > MAX_SCALE_BITS:
        raise ValueError(f"the coefficients' common denominator exceeds 2**{MAX_SCALE_BITS}")
    return [[int(x * scale) for x in p] for p in polys]


def _identical_on_circle(f, g) -> bool:
    """Whether |f| = |g| on |z| = 1 identically, exactly: g(z) g*(z) = f(z) f*(z) as integer polynomials, with p*
    the reversed coefficients padded to one length, compared through a power of two above every coefficient."""
    F, G = _integers([f, g])
    n = max(len(F), len(G))
    F, G = F + [0] * (n - len(F)), G + [0] * (n - len(G))
    bits = 2 * max(abs(x) for x in F + G).bit_length() + n.bit_length() + 3

    def at(p):
        return sum(x << (bits * j) for j, x in enumerate(p))
    return at(G) * at(G[::-1]) == at(F) * at(F[::-1])


def _floats(p, n):
    out = np.zeros(n)
    out[:len(p)] = [float(x) for x in p]
    return out


def cosine_series(f, g, r):
    """For |z| = r, the cosine coefficients a_k (|g|^2 - |f|^2 = sum_k a_k cos k theta; f = 0 gives |g|^2) and a
    bound on each one's error: r^n by repeated products and the correlations in binary64, each rounding bounded."""
    n = max(len(f), len(g))
    power = np.cumprod(np.r_[1.0, np.full(n - 1, r)])
    F, G = _floats(f, n) * power, _floats(g, n) * power
    corr = np.correlate(G, G, "full")[n - 1:] - np.correlate(F, F, "full")[n - 1:]
    size = np.correlate(np.abs(G), np.abs(G), "full")[n - 1:] + np.correlate(np.abs(F), np.abs(F), "full")[n - 1:]
    weight = np.r_[1.0, np.full(n - 1, 2.0)]
    return weight * corr, weight * size * (4 * n + 16) * U * 1.01


def _derivative_bounds(a, e):
    k = np.arange(a.size, dtype=float)
    n = a.size
    errs = [float(np.sum(k ** d * e) + U * np.sum(k ** d * np.abs(a) * (math.pi * k + n + 8)) * 1.01)
            for d in range(3)]
    return k, errs, float(np.sum(k ** 3 * (np.abs(a) + e)) * 1.01)


def _taylor(a, k, m):
    """Values, first and second derivatives of sum a_k cos(k theta) at the points m."""
    phase = np.outer(m, k)
    cos, sin = np.cos(phase), np.sin(phase)
    return cos @ a, -(sin @ (k * a)), -(cos @ (k * k * a))


def certify_positive(a, e, fraction=0.0, where="circle"):
    """A rigorous positive lower bound on sum a_k cos(k theta) over [0, pi] (the series is even), or ValueError.
    Each interval [m - h, m + h] is bounded below by the exact minimum of the Taylor quadratic at m less the
    evaluation errors and M3 h^3 / 6; an interval counts once that bound exceeds fraction times the value at m."""
    k, (err0, err1, err2), m3 = _derivative_bounds(a, e)
    pieces = 64
    h = np.full(pieces, math.pi / (2 * pieces))
    mid = (np.arange(pieces) + 0.5) * (2 * h[0])
    lowest, spent = math.inf, 0
    while mid.size:
        spent += mid.size
        if spent > MAX_INTERVALS:
            raise ValueError(f"could not certify positivity on {where} within {MAX_INTERVALS} intervals; "
                             "leave a larger margin")
        lows = np.empty(mid.size)
        for s in range(0, mid.size, 2048):
            d0, d1, d2 = _taylor(a, k, mid[s:s + 2048])
            # Inflated so that the float intervals cover [0, pi] without gaps.
            hh = h[s:s + 2048] * (1 + 1e-9)
            if (d0 - err0 <= 0).any():
                theta = float(mid[s + int(np.argmax(d0 - err0 <= 0))])
                raise ValueError(f"not positive on {where} near theta = {theta:.12g} "
                                 f"(value {float(d0[np.argmax(d0 - err0 <= 0)]):.3e})")
            low = d0 - np.abs(d1) * hh + d2 * hh * hh / 2
            vertex = (d2 > 0) & (np.abs(d1) < d2 * hh)
            low = np.where(vertex, np.minimum(low, d0 - d1 * d1 / (2 * np.where(vertex, d2, 1.0))), low)
            slack = 8 * U * (np.abs(d0) + np.abs(d1) * hh + np.abs(d2) * hh * hh)
            lows[s:s + 2048] = low - slack - err0 - err1 * hh - err2 * hh * hh / 2 - m3 * hh ** 3 / 6
            lows[s:s + 2048] = np.where(lows[s:s + 2048] > fraction * d0, lows[s:s + 2048], -np.inf)
        done = lows > -np.inf
        if done.any():
            lowest = min(lowest, float(lows[done].min()))
        h, mid = h[~done] / 2, np.concatenate([mid[~done] - h[~done] / 2, mid[~done] + h[~done] / 2])
        h = np.concatenate([h, h])
    if not lowest > 0:
        raise ValueError(f"not positive on {where}")
    return lowest


def winding(g, r, floor):
    """Zeros of g in |z| < r: the winding number of g on |z| = r, given |g|^2 >= floor > 0 there."""
    n = len(g)
    coef = _floats(g, n) * np.cumprod(np.r_[1.0, np.full(n - 1, r)])
    k = np.arange(n, dtype=float)
    size = float(np.sum(np.abs(coef))) * (1 + 4 * (n + 4) * U)
    lipschitz = float(np.sum(k * np.abs(coef))) * (1 + 4 * (n + 4) * U)
    error = 2 * U * float(np.sum(np.abs(coef) * (math.pi * k + 2 * n + 16))) + 2 * (n + 4) * U * size
    room = math.sqrt(floor) - 3 * error
    if room <= 0:
        raise ValueError(f"|g| on |z| = {r!r} is too small against rounding to count its zeros")
    samples = int(math.ceil(2 * math.pi * lipschitz / room * 1.01)) + 8
    if samples > MAX_SAMPLES:
        raise ValueError(f"|g| on |z| = {r!r} varies too fast against its minimum to count its zeros")
    step = 2 * math.pi / samples
    total, first, previous = 0.0, None, None
    for s in range(0, samples + 1, 4096):
        theta = np.arange(s, min(s + 4096, samples + 1)) * step
        values = np.exp(1j * np.outer(theta, k)) @ coef
        if (np.abs(values) <= lipschitz * step * (1 + 1e-9) + 3 * error).any():
            raise ValueError(f"g comes too close to 0 on |z| = {r!r} to count its zeros")
        chain = values if previous is None else np.r_[previous, values]
        total += float(np.sum(np.angle(chain[1:] / chain[:-1])))
        previous = values[-1:]
    count = total / (2 * math.pi)
    if abs(count - round(count)) > 1e-6:
        raise ValueError("winding number did not close")
    return int(round(count))


def certify(c, f, g) -> dict:
    """Every condition of the bound kappa <= c for binary64 c and exact real coefficients; ValueError if one fails."""
    gap = norm2(f) - norm2(g)
    if gap <= 0:
        raise ValueError(f"need ||f||^2 > ||g||^2; the difference is {float(gap):.6e}")
    report = {"norm_gap": gap, "outer_identical": _identical_on_circle(f, g)}
    if not report["outer_identical"]:
        report["outer_margin"] = certify_positive(*cosine_series(f, g, 1.0), where="|z| = 1")
    report["inner_margin"] = certify_positive(*cosine_series(f, g, c), where=f"|z| = c = {c!r}")
    zeros = []
    for r in (1.0, c):
        floor = certify_positive(*cosine_series([0], g, r), fraction=0.5, where=f"|g|^2 at |z| = {r!r}")
        zeros.append(winding(g, r, floor))
    if zeros[0] != zeros[1]:
        raise ValueError(f"g has {zeros[0] - zeros[1]} zeros in c < |z| < 1")
    report["zeros_inside"] = zeros[1]
    return report


# --- behavior readout -------------------------------------------------------
# The score is one number. These describe the SHAPE of the pair instead, so two pairs with the same c can land in
# different cells of the diversity map.

_BEHAVIOR_ZERO = {"b_dispersion": 0.0, "b_boundary_frac": 0.0, "b_contact": 0.0,
                  "b_offset": 0.0, "b_concentration": 0.0, "b_structure": 0.0}


def _behavior(c, f, g, gap):
    """Six bounded descriptors: spread of |g| on the unit circle, share of it where |f| = |g|, share of zero
    coefficients, the norm margin on a log scale, share of ||f||^2 in the constant term, and the degree."""
    try:
        n = max(len(f), len(g))
        F, G = _floats(f, n), _floats(g, n)
        z = np.exp(1j * np.linspace(0, np.pi, 2048))
        vf, vg = np.abs(np.polyval(F[::-1], z)), np.abs(np.polyval(G[::-1], z))
        spread = float(vg.std() / vg.mean())
        relative = float(gap / norm2(g))
        out = {"b_dispersion": spread / (1.0 + spread),
               "b_boundary_frac": float((vf >= (1 - 1e-9) * vg).mean()),
               "b_contact": float(((F == 0) & (G == 0)).mean()),
               "b_offset": min(1.0, max(0.0, (math.log10(relative) + 12) / 12)),
               "b_concentration": float(Fraction(f[0]) ** 2 / norm2(f)),
               "b_structure": (n - 1) / MAX_DEGREE}
        return {k: (float(v) if np.isfinite(v) else 0.0) for k, v in out.items()}
    except (ValueError, FloatingPointError, ZeroDivisionError, OverflowError):
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
    if not hasattr(program, "korenblum_pair"):
        raise AttributeError("missing korenblum_pair")
    start_time = time.time()
    c, f, g = construction(program.korenblum_pair())
    eval_time = time.time() - start_time
    report = certify(c, f, g)
    cases = {"upper_bound": c, "inverse_bound": 1.0 / c, "degree": float(max(len(f), len(g)) - 1),
             "norm_gap": float(report["norm_gap"]), "combined_score": REFERENCE / c, "validity": 1.0}
    cases.update(_behavior(c, f, g, report["norm_gap"]))
    witness = {
        "schema": "sciel-full-output/1",
        "source_sha256": source_sha, "evaluator_sha256": evaluator_sha,
        "invocation_id": invocation_id, "stage": stage, "confirmation": None,
        "input_sha256": None, "seed_sha256": None,
        "output": {"c": c.hex(), "f": [str(x) for x in f], "g": [str(x) for x in g],
                   "norm_gap": str(report["norm_gap"]), "outer_identical": report["outer_identical"],
                   "inner_margin": report["inner_margin"], "zeros_inside": report["zeros_inside"]},
    }
    artifacts = {"eval_time": f"{eval_time:.4f}", "full_output": _canonical(witness)}
    if source_sha != _source_hash(program_path) or evaluator_sha != _source_hash(__file__):
        raise ValueError("source or evaluator changed during invocation")
    return {"ok": True, "score": REFERENCE / c, "cases": cases, "artifacts": artifacts}


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
        "cases": {"combined_score": 0.0, "upper_bound": 0.0, "inverse_bound": 0.0, "degree": 0.0,
                  "norm_gap": 0.0, "validity": 0.0, **_BEHAVIOR_ZERO},
        "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]},
    }
