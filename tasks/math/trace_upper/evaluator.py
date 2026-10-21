"""Evaluator for the Schur-Siegel-Smyth trace constant from above; smoke/full/confirm on the sciel contract.

C is the smallest limit point of tr(alpha)/deg(alpha) over totally positive algebraic integers (Tao's constant 86a).
A certificate is a step density mu = sum_i d_i 1[i/N, (i+1)/N) dx on [0, 6], of mass m, with integer polynomials Q and
weights b_Q >= 0. Write U(x) = int log|x - y| dmu(y) and f = U - sum_Q b_Q log|Q|. If
  (E1) d_i >= 0, the support lies in [a, b] with a > 0 and b - a > 4 (capacity above 1),
  (E2) each Q is primitive, irreducible, has all its roots real and in [0, 6], and no two are equal up to sign,
  (E3) sum_Q b_Q deg Q < m,
  (E4) int log|Q| dmu > 0 for each Q,
  (E5) f >= 0 on the support of mu,
then f >= 0 on all of C by the minimum principle (f is superharmonic off the support and tends to +infinity), so for
every other irreducible P the resultants with the Q give int log|P| dmu >= 0, and Smith's theorem (Ann. Math. 2024)
makes mu/m a limit of conjugate distributions: C <= int x dmu / m (Orloski-Talebizadeh Sardari; Pinho's certificate).

E1-E3 and the bound are exact rationals, E2 is SymPy, E4 is the closed form sum_j c_j G(r - j/N) over the jumps c_j of
the density and the roots r, with G(u) = u log|u| - u, in mpmath interval arithmetic. E5 is proved by bisection: on a
box [s, t] inside a cell, the far edge terms and the far roots are smooth with second derivative at most K, so they are
at least their chord less K w^2/8; an edge at s or t contributes a convex or concave G term, which lies at most |c| w/e
below its chord or above it; a root within 2w of the box is bounded by its largest distance. Points of the boxes are
dyadic, x - j/N = n / (N 2^40) with integer n, and each logarithm is evaluated with an a priori error bound of
2^-48 (1 + |exponent|), at least ten times the rounding analysis of the series used; every sum carries the IEEE bound.
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

import mpmath
import numpy as np
import sympy

# Pinho's certificate (rpinho/c86-trace-certificate @116a58f), replayed in teorth/optimizationproblems PR 242.
REFERENCE = Fraction(144938007612277893, 79999999999994000)
MAX_N = 800
MAX_POLYS = 40
MAX_DEGREE = 16
MAX_TOTAL_DEGREE = 120
MAX_BITS = 400
MAX_WITNESS = 1000000
LEVEL = 40
MAX_DEPTH = 34
MAX_BOXES = 400000
U = 2.0 ** -52
LN2 = 0.6931471805599453
INV_E = 0.3678794412
HORNER = [1.0 / (2 * i + 1) for i in range(12, -1, -1)]
X = sympy.Symbol("x")


def _load(program_path, module_name="program"):
    return load_candidate(program_path, module_name)


def _rational(value, what):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{what} must be a number, not a bool")
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"{what} must be finite")
    if isinstance(value, str) and len(value) > 300:
        raise ValueError(f"{what} is a string over 300 characters")
    if not isinstance(value, (int, float, str, Fraction)):
        raise ValueError(f"{what} must be an int, float, Fraction or rational string")
    q = Fraction(value)
    if q.numerator.bit_length() > MAX_BITS or q.denominator.bit_length() > MAX_BITS:
        raise ValueError(f"{what} has more than {MAX_BITS} bits")
    return q


def _certificate(result):
    """(A, b, d): integer coefficient lists (constant term first), rational weights and the 6N cell densities."""
    if not isinstance(result, (tuple, list)) or len(result) != 3:
        raise ValueError("certificate() must return (polynomials, weights, densities)")
    A, b, d = result
    if not 1 <= len(A) <= MAX_POLYS or len(b) != len(A):
        raise ValueError(f"1..{MAX_POLYS} polynomials with one weight each")
    polys = []
    for q in A:
        integers = isinstance(q, (list, tuple)) and all(
            isinstance(a, (int, np.integer)) and not isinstance(a, (bool, np.bool_)) for a in q)
        q = [int(a) for a in q] if integers else []
        if not 2 <= len(q) <= MAX_DEGREE + 1 or q[-1] == 0 or max(map(abs, q)) >= 2 ** 64:
            raise ValueError(f"polynomials are integer lists of degree 1..{MAX_DEGREE}, constant term first, "
                             "with a nonzero leading coefficient below 2**64")
        polys.append(q)
    if sum(len(q) - 1 for q in polys) > MAX_TOTAL_DEGREE:
        raise ValueError(f"the degrees of the polynomials sum to more than {MAX_TOTAL_DEGREE}")
    weights = [_rational(w, "weight") for w in b]
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")
    if len(d) % 6 or not 6 <= len(d) <= 6 * MAX_N:
        raise ValueError(f"densities are 6N values for the cells of width 1/N on [0, 6], N <= {MAX_N}")
    densities = [_rational(x, "density") for x in d]
    if any(x < 0 for x in densities):
        raise ValueError("densities must be non-negative")
    return polys, weights, densities


def exact_checks(A, b, d):
    """E1-E3, E2 and the bound, exactly; returns the support cells, the mass, the bound and the SymPy polynomials."""
    N = len(d) // 6
    live = [i for i, x in enumerate(d) if x > 0]
    if not live:
        raise ValueError("the density is zero")
    lo, hi = Fraction(live[0], N), Fraction(live[-1] + 1, N)
    if lo <= 0:
        raise ValueError("the support touches 0: the approximating integers would not be totally positive")
    if hi - lo <= 4:
        raise ValueError("the support hull is shorter than 4, so its capacity is not above 1")
    m = sum(d[i] for i in live) / N
    bound = sum(d[i] * (2 * i + 1) for i in live) / (2 * N * N) / m
    if sum(w * (len(q) - 1) for w, q in zip(b, A)) >= m:
        raise ValueError("sum_Q b_Q deg Q must be below the mass (E3)")
    polys, seen = [], set()
    for q in A:
        P = sympy.Poly(list(reversed(q)), X, domain="ZZ")
        content, factors = P.factor_list()
        if abs(content) != 1 or len(factors) != 1 or factors[0][1] != 1:
            raise ValueError(f"{P.as_expr()} is not primitive and irreducible (E2)")
        if P.count_roots(0, 6) != P.degree():
            raise ValueError(f"{P.as_expr()} has a root that is not real or not in [0, 6] (E2)")
        key = tuple(q) if q[-1] > 0 else tuple(-a for a in q)
        if key in seen:
            raise ValueError(f"{P.as_expr()} appears twice up to sign (E2)")
        seen.add(key)
        polys.append(P)
    return N, live, m, bound, polys


def _roots(P):
    """Rational isolating intervals of the roots, refined to 1e-30."""
    return [(Fraction(int(a.p), int(a.q)), Fraction(int(c.p), int(c.q)))
            for (a, c), _ in P.intervals(eps=sympy.Rational(1, 10 ** 30))]


def _jumps(d):
    """Edges j with jump c_j = d_j - d_{j-1} != 0, the density being 0 outside [0, 6]."""
    padded = [Fraction(0)] + d + [Fraction(0)]
    return [(j, padded[j + 1] - padded[j]) for j in range(len(d) + 1) if padded[j + 1] != padded[j]]


def _iv(q):
    return mpmath.iv.mpf(q.numerator) / q.denominator


def check_E4(N, m, jumps, polys, roots):
    """Lower ends of int log|Q| dmu = m log|lc Q| + sum_r sum_j c_j G(r - j/N), in interval arithmetic."""
    mpmath.iv.prec = 96
    edges = [(_iv(Fraction(j, N)), _iv(c)) for j, c in jumps]
    out = []
    for P, rs in zip(polys, roots):
        total = _iv(m) * mpmath.iv.log(abs(int(P.LC())))
        for a, c in rs:
            r = mpmath.iv.mpf([_iv(a).a, _iv(c).b])
            for (j, cj), (e, ce) in zip(jumps, edges):
                if a == c == Fraction(j, N):
                    continue
                u = r - e
                if 0 in u:
                    raise ValueError("a root enclosure contains a grid edge")
                total += ce * (u * mpmath.iv.log(abs(u)) - u)
        out.append(float(total.a))
    return out


def _log(y):
    """ln y for positive binary64 y, with |error| <= rad. y = m 2^k with m in [0.707, 1.414), so z = (m - 1)/(m + 1)
    is below 0.1716 (m - 1 is exact by Sterbenz) and ln m = 2 atanh z is summed to z^25, a tail below 1e-21."""
    m, k = np.frexp(y)
    low = m < 0.7071067811865476
    m = np.where(low, 2 * m, m)
    k = np.where(low, k - 1, k).astype(np.float64)
    z = (m - 1) / (m + 1)
    w = z * z
    s = np.full_like(z, HORNER[0])
    for c in HORNER[1:]:
        s = s * w + c
    return k * LN2 + 2 * z * s, 2.0 ** -48 * (1 + np.abs(k))


def _float_down(q):
    f = float(q)
    return f if Fraction(f) <= q else math.nextafter(f, -math.inf)


def _float_up(q):
    f = float(q)
    return f if Fraction(f) >= q else math.nextafter(f, math.inf)


class Potential:
    """f's pieces at dyadic points x = key / (N 2^LEVEL), as rigorous floats."""

    def __init__(self, N, jumps, weights, polys, roots):
        self.D = float(N * 2 ** LEVEL)
        mpmath.iv.prec = 96
        ld = mpmath.iv.log(N) + LEVEL * mpmath.iv.log(2)
        self.ld = float(ld.mid)
        self.ld_rad = float(ld.delta) + abs(self.ld) * U + 1e-300
        self.ej = np.array([j * 2 ** LEVEL for j, _ in jumps], dtype=np.int64)
        self.c = np.array([float(c) for _, c in jumps])
        rs = [(w, a, c) for w, P, rr in zip(weights, polys, roots) if w > 0 for a, c in rr]
        self.b = np.array([float(w) for w, _, _ in rs])
        self.r_lo = np.array([_float_down(a) for _, a, _ in rs])
        self.r_hi = np.array([_float_up(c) for _, _, c in rs])
        const = sum((_iv(w) * mpmath.iv.log(abs(int(P.LC()))) for w, P in zip(weights, polys) if w > 0),
                    mpmath.iv.mpf(0))
        self.const = float(const.b) + abs(float(const.b)) * U + 1e-300
        self.keys, self.u_mid, self.u_rad, self.root_up = {}, [], [], []

    def _evaluate(self, keys):
        n = keys[:, None] - self.ej[None, :]
        nf = n.astype(np.float64)
        an = np.abs(nf)
        lm, lr = _log(np.where(an > 0, an, 1.0))
        a = lm - self.ld - 1
        ra = lr + self.ld_rad + U * (np.abs(a) + 1)
        q = nf / self.D
        g = q * a
        rg = np.abs(q) * ra + 3 * U * np.abs(g)
        t = self.c * g
        rt = np.abs(self.c) * rg + 3 * U * np.abs(t)
        mid = t.sum(axis=1)
        rad = (rt.sum(axis=1) + self.c.size * U * np.abs(t).sum(axis=1)) * (1 + 2.0 ** -30)
        x = keys.astype(np.float64) / self.D
        dist = np.maximum(np.abs(x[:, None] - self.r_lo), np.abs(x[:, None] - self.r_hi)) * (1 + U) + U * 8
        lm, lr = _log(dist)
        up = lm + lr
        return mid, rad, self.b * up + U * self.b * np.abs(up)

    def index(self, keys):
        new = np.array(sorted({int(k) for k in keys} - self.keys.keys()), dtype=np.int64)
        for start in range(0, new.size, 1024):
            chunk = new[start:start + 1024]
            mid, rad, up = self._evaluate(chunk)
            for k in chunk.tolist():
                self.keys[k] = len(self.keys)
            self.u_mid.extend(mid.tolist())
            self.u_rad.extend(rad.tolist())
            self.root_up.extend(up)
        return np.array([self.keys[int(k)] for k in keys], dtype=np.int64)

    def estimate(self, keys):
        """f at the points up to about 1e-9 away from the roots: an early sign of a counterexample, never a proof."""
        i = self.index(keys)
        return np.array(self.u_mid)[i] - self._roots_up()[i].sum(axis=1)

    def _roots_up(self):
        return np.array(self.root_up).reshape(len(self.u_mid), self.b.size)

    def bounds(self, ks, width):
        """lower_bounds of the boxes [ks, ks + width], 2048 at a time."""
        return np.concatenate([self.lower_bounds(ks[i:i + 2048], ks[i:i + 2048] + width[i:i + 2048])
                               for i in range(0, ks.size, 2048)] + [np.zeros(0)])

    def lower_bounds(self, ks, kt):
        """Rigorous lower bounds of f on the boxes [ks, kt] / (N 2^LEVEL), each inside one cell."""
        i_s, i_t = self.index(ks), self.index(kt)
        um, ur, up = np.array(self.u_mid), np.array(self.u_rad), self._roots_up()
        s, t = ks.astype(np.float64) / self.D, kt.astype(np.float64) / self.D
        w = (kt - ks).astype(np.float64) / self.D * (1 + U)
        near = (self.r_hi[None, :] >= (s - 2 * w)[:, None]) & (self.r_lo[None, :] <= (t + 2 * w)[:, None])
        far = ~near
        e_s = um[i_s] - ur[i_s] - (up[i_s] * far).sum(axis=1)
        e_t = um[i_t] - ur[i_t] - (up[i_t] * far).sum(axis=1)
        values = np.minimum(e_s, e_t)
        dmax = np.maximum(np.abs(t[:, None] - self.r_lo), np.abs(self.r_hi - s[:, None])) * (1 + U) + U * 8
        lm, lr = _log(dmax)
        lu = lm + lr
        near_lb = -((self.b * lu + U * self.b * np.abs(lu)) * near).sum(axis=1)
        dmin = np.maximum(s[:, None] - self.r_hi, self.r_lo - t[:, None]) - U * 16
        curv_root = (np.where(far, self.b / np.where(far, dmin, 1.0) ** 2, 0.0)).sum(axis=1) * (1 + 2.0 ** -30)
        ns, nt = ks[:, None] - self.ej, kt[:, None] - self.ej
        edge = (ns == 0) | (nt == 0)
        with np.errstate(divide="ignore", invalid="ignore"):
            cs = self.c * self.D / np.where(ns == 0, 1, ns).astype(np.float64)
            ct = self.c * self.D / np.where(nt == 0, 1, nt).astype(np.float64)
        top = np.where(edge, 0.0, np.maximum(cs, ct))
        curv = top.sum(axis=1) + 1e-12 * np.abs(top).sum(axis=1) + curv_root
        pen = np.maximum(curv, 0.0) * w * w / 8 * (1 + 2.0 ** -30)
        convex = ((ns == 0) & (self.c > 0)) | ((nt == 0) & (self.c < 0))
        dips = (np.abs(self.c) * convex).sum(axis=1) * w * INV_E * (1 + 2.0 ** -30)
        lb = values - pen - dips + near_lb - self.const
        size = np.abs(um[i_s]) + np.abs(um[i_t]) + (np.abs(up[i_s]) + np.abs(up[i_t])).sum(axis=1)
        slop = 2.0 ** -40 * (size + pen + dips + np.abs(near_lb) + abs(self.const) + 1)
        return lb - slop


def check_E5(N, live, jumps, weights, polys, roots):
    """Bisection of every support cell until each box has a positive certified lower bound."""
    pot = Potential(N, jumps, weights, polys, roots)
    step = 2 ** (LEVEL - 3)
    ks = np.array([i * 2 ** LEVEL + k * step for i in live for k in range(8)], dtype=np.int64)
    width = np.full(ks.size, step, dtype=np.int64)
    boxes, smallest, depth = 0, math.inf, 3
    while ks.size:
        boxes += ks.size
        if boxes > MAX_BOXES:
            raise ValueError(f"E5 not proved within {MAX_BOXES} boxes")
        lb = pot.bounds(ks, width)
        good = lb > 0
        if good.any():
            smallest = min(smallest, float(lb[good].min()))
        bad = ~good
        if bad.any():
            low = pot.estimate(ks[bad])
            if low.min() < -1e-6:
                x = float(Fraction(int(ks[bad][low.argmin()]), N * 2 ** LEVEL))
                raise ValueError(f"E5 fails: f is about {low.min():.3e} at x = {x:.12f}")
        if bad.any() and depth >= MAX_DEPTH:
            x = float(Fraction(int(ks[bad][0]), N * 2 ** LEVEL))
            raise ValueError(f"E5 fails: f is not certified nonnegative near x = {x:.12f}")
        half = width[bad] // 2
        ks = np.concatenate([ks[bad], ks[bad] + half])
        width = np.concatenate([half, half])
        depth += 1
    return {"boxes": boxes, "points": len(pot.keys), "min_lower_bound": smallest, "depth": depth - 1}


def certify(A, b, d):
    """All checks of one certificate; returns its exact bound and a report. Raises ValueError when invalid."""
    N, live, m, bound, polys = exact_checks(A, b, d)
    roots = [_roots(P) for P in polys]
    jumps = _jumps(d)
    e4 = check_E4(N, m, jumps, polys, roots)
    if min(e4) <= 0:
        raise ValueError(f"int log|Q| dmu is not certified positive for {polys[e4.index(min(e4))].as_expr()} (E4)")
    e5 = check_E5(N, live, jumps, b, polys, roots)
    report = {"N": N, "cells": len(live), "mass": m, "e4_margin": min(e4),
              "b_deg_ratio": float(sum(w * (len(q) - 1) for w, q in zip(b, A)) / m), **e5}
    return bound, report


# --- behavior readout -------------------------------------------------------
# The score is one number. These describe the SHAPE of the certificate instead, so two
# certificates with the same bound can land in different cells of the diversity map.

_BEHAVIOR_ZERO = {"b_dispersion": 0.0, "b_boundary_frac": 0.0, "b_contact": 0.0,
                  "b_offset": 0.0, "b_concentration": 0.0, "b_structure": 0.0}


def _behavior(A, d):
    """Six bounded descriptors: density spread, mass in the outer tenths of the support, empty cells inside it,
    centre of mass against the middle of the support, mass in the heaviest cell, and total degree of the
    polynomials. Never raises."""
    try:
        x = np.array([float(v) for v in d])
        live = np.flatnonzero(x > 0)
        hull = x[live[0]:live[-1] + 1]
        mass = float(hull.sum())
        pos = (np.arange(hull.size) + 0.5) / hull.size
        nonzero = hull[hull > 0]
        spread = float(nonzero.std() / nonzero.mean())
        out = {"b_dispersion": spread / (1 + spread),
               "b_boundary_frac": float(hull[(pos < 0.1) | (pos > 0.9)].sum() / mass),
               "b_contact": float((hull == 0).mean()),
               "b_offset": min(abs(float((hull * pos).sum() / mass) - 0.5) / 0.5, 1.0),
               "b_concentration": float(hull.max() / mass),
               "b_structure": min(sum(len(q) - 1 for q in A) / 100.0, 1.0)}
        return {k: (float(v) if np.isfinite(v) else 0.0) for k, v in out.items()}
    except (ValueError, IndexError, FloatingPointError, ZeroDivisionError, OverflowError):
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
    if not hasattr(program, "certificate"):
        raise AttributeError("missing certificate")
    start_time = time.time()
    A, b, d = _certificate(program.certificate())
    eval_time = time.time() - start_time
    start_time = time.time()
    value, report = certify(A, b, d)
    check_time = time.time() - start_time
    live = [i for i, x in enumerate(d) if x > 0]
    output = {"A": A, "b": [str(w) for w in b], "N": report["N"], "first": live[0],
              "densities": [str(x) for x in d[live[0]:live[-1] + 1]], "bound": [value.numerator, value.denominator]}
    witness = {
        "schema": "sciel-full-output/1",
        "source_sha256": source_sha, "evaluator_sha256": evaluator_sha,
        "invocation_id": invocation_id, "stage": stage, "confirmation": None,
        "input_sha256": None, "seed_sha256": None, "output": output,
    }
    text = _canonical(witness)
    if len(text) > MAX_WITNESS:
        raise ValueError(f"the certificate takes {len(text)} bytes as JSON, over {MAX_WITNESS}")
    combined = float(REFERENCE / value)
    cases = {"upper_bound": float(value), "inverse_bound": float(1 / value), "N": float(report["N"]),
             "polynomials": float(len(A)), "e4_margin": report["e4_margin"],
             "e5_margin": report["min_lower_bound"], "b_deg_ratio": report["b_deg_ratio"],
             "combined_score": combined, "validity": 1.0}
    cases.update(_behavior(A, d))
    artifacts = {"eval_time": f"{eval_time:.4f}", "check_time": f"{check_time:.4f}",
                 "check": _canonical({k: (str(v) if isinstance(v, Fraction) else v) for k, v in report.items()}),
                 "full_output": text}
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
        "cases": {"combined_score": 0.0, "upper_bound": 0.0, "inverse_bound": 0.0, "N": 0.0, "polynomials": 0.0,
                  "e4_margin": 0.0, "e5_margin": 0.0, "b_deg_ratio": 0.0, "validity": 0.0, **_BEHAVIOR_ZERO},
        "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]},
    }
