"""Doche's limit-point bound on the essential minimum C_82 of the Zhang-Zagier height; a program uses it as:
from sciel.task import asset; doche = asset("doche").

A candidate is {"denominators": [{"coeffs": [...], "multiplicity": m}, ...], "numerators": [{"coeffs": [...],
"exponent": q}, ...]}: integer polynomials R_j (leading coefficient +-1) and S_i, coefficients from the leading one
down, positive integer multiplicities m_j and exponents q_i >= 0 (an int or an [numerator, denominator] pair). With
T(z) = z(1 - z), Q = prod R_j^m_j, A = prod S_i^q_i, D = sum m_j deg R_j and deg A = sum q_i deg S_i < D, Doche
(JTNB 13, 2001) gives infinitely many algebraic numbers of Zhang-Zagier height at most

    B = (1/D) [ sum_j m_j log M(R_j o T) + int_0^1 max(0, log|A(T(u))| - log|Q(T(u))|) dt ],   u = exp(2 pi i t),

so C_82 <= B (Pinho's certificate format). Only the ratios m : q matter; B is homogeneous of degree 0.

Certification (certify): every root of every R_j o T and S_i o T is refined in 320-bit fixed point and enclosed in a
disc of radius n |P / P'| plus an a priori Horner error bound; the discs of each polynomial are pairwise disjoint (so
each holds exactly one root) and numerator discs miss denominator discs (so gcd(S_i, R_j) = 1). The Mahler part is
summed from the enclosures at 40 digits; the integral is bounded above on a dyadic tiling of [0, 1/2] as in
Patankar's checker (zero panels, Taylor panels with a bound on g'', Lipschitz panels at crossings), in binary64 with
outward margins on every distance and sum.
"""

from fractions import Fraction
import math

import mpmath
import numpy as np
from scipy.optimize import linprog

MAX_DENOMINATORS = 8
MAX_NUMERATORS = 64
MAX_DEGREE = 128
MAX_BITS = 256
F = 320
TOL = 1e-10
TOP_LEVEL = 10
MAX_DEPTH = 52
_ROOTS = {}


def _poly(coeffs, what):
    raw = list(coeffs)
    if any(isinstance(c, bool) or not isinstance(c, (int, np.integer)) for c in raw):
        raise ValueError(f"{what} coefficients must be integers")
    p = [int(c) for c in raw]
    if len(p) < 2 or p[0] == 0:
        raise ValueError(f"{what} must have degree >= 1 and a nonzero leading coefficient")
    if len(p) - 1 > MAX_DEGREE or max(abs(c) for c in p).bit_length() > MAX_BITS:
        raise ValueError(f"{what} degree must be at most {MAX_DEGREE} and coefficients below 2^{MAX_BITS}")
    return p


def _exponent(q):
    if isinstance(q, (list, tuple)):
        if len(q) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in q):
            raise ValueError("an exponent pair must be [numerator, denominator] integers")
        return Fraction(int(q[0]), int(q[1]))
    if isinstance(q, Fraction):
        return q
    if isinstance(q, bool) or not isinstance(q, (int, np.integer)):
        raise ValueError("an exponent must be an integer or an [numerator, denominator] pair")
    return Fraction(int(q))


def canonical(candidate):
    """(denominators, numerators) as [(coeffs, m)] and [(coeffs, q)] with q > 0 only; raises ValueError."""
    if not isinstance(candidate, dict) or not {"denominators", "numerators"} <= set(candidate):
        raise ValueError('the candidate must be {"denominators": [...], "numerators": [...]}')
    den, num = [], []
    for r in candidate["denominators"]:
        p, m = _poly(r["coeffs"], "a denominator"), r["multiplicity"]
        if abs(p[0]) != 1:
            raise ValueError("denominators must have leading coefficient +-1")
        if isinstance(m, bool) or not isinstance(m, (int, np.integer)) or m <= 0:
            raise ValueError("multiplicities must be positive integers")
        den.append((p, int(m)))
    for s in candidate["numerators"]:
        p, q = _poly(s["coeffs"], "a numerator"), _exponent(s["exponent"])
        if q < 0:
            raise ValueError("exponents must be non-negative")
        if q > 0:
            num.append((p, q))
    if not 1 <= len(den) <= MAX_DENOMINATORS or len(num) > MAX_NUMERATORS:
        raise ValueError(f"1 to {MAX_DENOMINATORS} denominators and at most {MAX_NUMERATORS} active numerators")
    if len({tuple(p) for p, _ in den}) < len(den) or len({tuple(p) for p, _ in num}) < len(num):
        raise ValueError("factors must be distinct")
    D = sum((len(p) - 1) * m for p, m in den)
    if sum((len(p) - 1) * q for p, q in num) >= D:
        raise ValueError("deg A must be below D")
    return den, num


# --- roots ------------------------------------------------------------------------------------------------------


def _mul(a, b):
    return (a[0] * b[0] - a[1] * b[1]) >> F, (a[0] * b[1] + a[1] * b[0]) >> F


def _horner(p, w):
    """(P(w), P'(w)) in fixed point for integer coefficients p (leading first)."""
    v, dv = (p[0] << F, 0), (0, 0)
    for c in p[1:]:
        dv = _mul(dv, w)
        dv = (dv[0] + v[0], dv[1] + v[1])
        v = _mul(v, w)
        v = (v[0] + (c << F), v[1])
    return v, dv


def _fix(z):
    return _fixf(z.real), _fixf(z.imag)


def _fixf(x):
    m, e = math.frexp(x)
    k = F + e - 53
    m = int(m * 2 ** 53)
    return m << k if k >= 0 else m >> -k


def _flt(a):
    return complex(a[0] / (1 << F), a[1] / (1 << F))


def _abs(a):
    """|a| / 2^F as a float rounded up."""
    return math.ldexp(math.isqrt(a[0] * a[0] + a[1] * a[1]) + 1, -F) * (1 + 1e-15)


def roots(coeffs):
    """The 2 deg P roots of P o T as (centres, radii, exact): float centres, float radii of discs that each hold exactly
    one root, and the fixed-point centres. Cached per polynomial; about 0.5 s at degree 104. Raises ValueError when the
    discs cannot be separated (a repeated or very close pair of roots)."""
    key = tuple(coeffs)
    if key in _ROOTS:
        return _ROOTS[key]
    p = list(key)
    n = len(p) - 1
    beta = np.roots(np.array([float(c) for c in p]) / float(max(abs(c) for c in p)))
    s = np.sqrt(1 - 4 * beta.astype(complex))
    start = np.concatenate([(1 + s) / 2, (1 - s) / 2])
    one = 1 << F
    zs = [_fix(z0) for z0 in start]
    # Aberth iteration on all 2n roots at once: the ratio P / P' in fixed point, the repulsion sum in binary64.
    for _ in range(80):
        flt = np.array([_flt(z) for z in zs])
        diff = flt[:, None] - flt[None, :]
        np.fill_diagonal(diff, 1.0)
        rep = (1 / diff).sum(axis=1) - 1.0
        moved = 0.0
        for i, z in enumerate(zs):
            zz = _mul(z, z)
            v, dv = _horner(p, (z[0] - zz[0], z[1] - zz[1]))
            dz = _mul(dv, (one - 2 * z[0], -2 * z[1]))
            den = dz[0] * dz[0] + dz[1] * dz[1]
            if den == 0:
                continue
            ratio = _flt((((v[0] * dz[0] + v[1] * dz[1]) << F) // den, ((v[1] * dz[0] - v[0] * dz[1]) << F) // den))
            step = ratio / (1 - ratio * rep[i])
            zs[i] = (z[0] - _fixf(step.real), z[1] - _fixf(step.imag))
            moved = max(moved, abs(step))
        if moved < 2.0 ** (48 - F) * (1 + np.abs(flt).max()):
            break
    absum = [abs(c) for c in p]
    centres, radii, exact = [], [], []
    for z in zs:
        zz = _mul(z, z)
        w = (z[0] - zz[0], z[1] - zz[1])
        v, dv = _horner(p, w)
        dz = _mul(dv, (one - 2 * z[0], -2 * z[1]))
        aw, az = _abs(w) + 1, _abs(z) + 1
        # A priori bounds on the fixed-point Horner error of P(w) and of P'(w) (1 - 2z), with a 2^16 margin.
        bound = sum(a * aw ** (n - k) for k, a in enumerate(absum))
        e0 = math.ldexp((n + 4) * bound, 16 - F)
        e1 = math.ldexp((n + 4) ** 2 * bound * 2 * az, 16 - F)
        lo = _abs(dz) * (1 - 1e-15) - e1 - math.ldexp(4, -F)
        if not lo > 0:
            raise ValueError("a root could not be enclosed (vanishing derivative)")
        radii.append(2 * n * (_abs(v) + e0) / lo * (1 + 1e-12) + math.ldexp(4, -F))
        centres.append(_flt(z))
        exact.append(z)
    centres, radii = np.array(centres), np.array(radii)
    _separate(centres, radii, centres, radii, same=True)
    _ROOTS[key] = (centres, radii, exact)
    return _ROOTS[key]


def _separate(a, ra, b, rb, same=False):
    gap = np.abs(a[:, None] - b[None, :]) - ra[:, None] - rb[None, :] - 1e-13 * (1 + np.abs(a[:, None]))
    if same:
        np.fill_diagonal(gap, 1.0)
    if (gap <= 0).any():
        raise ValueError("root discs overlap: a repeated root, or a numerator sharing a root with a denominator")


def log_mahler(coeffs):
    """(lo, hi) mpf bounds on log M(P o T) = log|lead| + sum log max(1, |z|) over the enclosed roots."""
    centres, radii, exact = roots(coeffs)
    lo = hi = mpmath.log(abs(coeffs[0]))
    for z, r in zip(exact, radii):
        a = mpmath.sqrt(mpmath.mpf(z[0] * z[0] + z[1] * z[1])) / mpmath.mpf(2) ** F
        hi += mpmath.log(max(mpmath.mpf(1), a + r))
        lo += mpmath.log(max(mpmath.mpf(1), a - r))
    return lo, hi


# --- the integral -----------------------------------------------------------------------------------------------


def _system(den, num):
    """Roots, radii, weights c_r and the constant of g = sum c_r log|u - r| + const, all divided by D."""
    D = sum((len(p) - 1) * m for p, m in den)
    parts = [(p, -m / D) for p, m in den] + [(p, float(q) / D) for p, q in num]
    rs, rad, c = [], [], []
    const = sum(float(q) / D * math.log(abs(p[0])) for p, q in num)
    for p, wgt in parts:
        z, r, _ = roots(p)
        rs.append(z), rad.append(r), c.append(np.full(z.size, wgt))
    rs, rad, c = np.concatenate(rs), np.concatenate(rad), np.concatenate(c)
    nd = sum(2 * (len(p) - 1) for p, _ in den)
    _separate(rs[nd:], rad[nd:], rs[:nd], rad[:nd])
    return rs, rad, c, const


def _panels(centre, width, rs, eta, c, const, tol):
    """Upper bounds of int max(0, g) over each panel, or nan where the panel must be bisected."""
    u = np.exp(2j * np.pi * centre)
    dist = np.abs(u[:, None] - rs[None, :])
    rad = (np.pi * width)[:, None]
    pos, neg = c > 0, c < 0
    ac = np.abs(c)
    with np.errstate(divide="ignore", invalid="ignore"):
        eps = 1e-12 * (1 + np.abs(np.log(np.maximum(dist, 1e-300))) @ ac + abs(const))
        lo_all = dist - eta - rad
        gup = (const + np.log(dist[:, pos] + eta[pos] + rad) @ c[pos]
               + np.log(np.maximum(lo_all[:, neg], 1e-300)) @ c[neg] + eps)
        gup[(lo_all[:, neg] <= 0).any(axis=1)] = np.inf
        out = np.full(centre.size, np.nan)
        out[gup < 0] = 0.0
        sep = (lo_all > 0).all(axis=1) & np.isnan(out)
        if sep.any():
            d, lo, u_s, w = dist[sep], lo_all[sep], u[sep], width[sep]
            dlo, dhi = d - eta, d + eta
            gc_up = const + np.log(dhi[:, pos]) @ c[pos] + np.log(dlo[:, neg]) @ c[neg] + eps[sep]
            gc_lo = const + np.log(dlo[:, pos]) @ c[pos] + np.log(dhi[:, neg]) @ c[neg] - eps[sep]
            gp = np.abs(np.real(2j * np.pi * u_s * ((1 / (u_s[:, None] - rs[None, :])) @ c)))
            gp += 2 * np.pi * ((eta / dlo ** 2) @ ac + 1e-11 * (1 / dlo) @ ac)
            mb = 4 * np.pi ** 2 * ((np.abs(rs) + eta) / lo ** 2) @ ac * (1 + 1e-12)
            h = w / 2
            reach = gp * h + mb * h * h / 2
            taylor = mb * w ** 3 / 24
            val = np.where(gc_lo - reach > 0, gc_up * w + taylor,
                           np.where(gc_up + reach < 0, 0.0, np.maximum(gc_up, 0) * w + gp * w * w / 4 + taylor))
            err = np.where(gc_lo - reach > 0, taylor,
                           np.where(gc_up + reach < 0, 0.0, gp * w * w / 4 + taylor))
            ok = err < tol * w
            res = np.full(sep.sum(), np.nan)
            res[ok] = val[ok]
            out[sep] = res
    return out


def integral_upper(den, num, tol=TOL):
    """(upper, panels): an upper bound on int_0^1 max(0, g) dt (twice the bound on [0, 1/2])."""
    rs, rad, c, const = _system(den, num)
    eta = rad + 4e-16 * (1 + np.abs(rs)) + 1e-15
    k = np.arange(2 ** TOP_LEVEL)
    centre, width = (k + 0.5) / 2 ** (TOP_LEVEL + 1), np.full(k.size, 0.5 ** (TOP_LEVEL + 1))
    total, panels = 0.0, 0
    chunk = max(64, 2 ** 22 // rs.size)
    while centre.size:
        if width[0] < 0.5 ** MAX_DEPTH:
            raise ValueError("the integral did not resolve: a denominator root on the unit circle?")
        vals = np.concatenate([_panels(centre[i:i + chunk], width[i:i + chunk], rs, eta, c, const, tol)
                               for i in range(0, centre.size, chunk)])
        done = ~np.isnan(vals)
        total += float(np.sum(vals[done]))
        panels += int(done.sum())
        left = centre[~done]
        w = width[~done] / 2
        centre, width = np.concatenate([left - w / 2, left + w / 2]), np.concatenate([w, w])
    return 2 * total * (1 + 1e-12) + 1e-300 * panels, panels


def certify(candidate, tol=TOL):
    """(upper, mahler, integral, panels): upper is an mpf above B. About 5 s for the frontier certificate."""
    den, num = canonical(candidate)
    D = sum((len(p) - 1) * m for p, m in den)
    with mpmath.workdps(40):
        mahler = sum(m * log_mahler(p)[1] for p, m in den) / D
        integral, panels = integral_upper(den, num, tol)
        upper = mahler + mpmath.mpf(integral)
    return upper, mahler, integral, panels


# --- floating point tools -----------------------------------------------------------------------------------------


def _profile(coeffs, t):
    """log|P(T(u))| at u = exp(2 pi i t) for every t."""
    z, _, _ = roots(coeffs)
    u = np.exp(2j * np.pi * np.asarray(t))
    return math.log(abs(coeffs[0])) + np.log(np.abs(u[:, None] - z[None, :])).sum(axis=1)


def value(candidate, grid=2 ** 15):
    """Float B on a midpoint grid of [0, 1/2] (no certification)."""
    den, num = canonical(candidate)
    D = sum((len(p) - 1) * m for p, m in den)
    t = (np.arange(grid) + 0.5) / (2 * grid)
    g = sum(float(q) * _profile(p, t) for p, q in num) - sum(m * _profile(p, t) for p, m in den)
    mahler = sum(m * float(log_mahler(p)[1]) for p, m in den)
    return (mahler + np.maximum(g, 0).mean()) / D


def resolve(denominators, numerators, grid=2 ** 13, scale=10 ** 12):
    """The joint LP of Pinho: optimal multiplicities and exponents for the given factor lists (coefficient lists) on a
    midpoint grid, returned as a candidate with integer multiplicities summing to about `scale` over the degrees and
    exponents rounded down to multiples of 2^-20. Factors the LP leaves at zero are dropped. A few seconds."""
    t = (np.arange(grid) + 0.5) / (2 * grid)
    b = np.array([_profile(p, t) for p in denominators])
    a = np.array([_profile(p, t) for p in numerators]).reshape(len(numerators), grid)
    L = np.array([float(log_mahler(p)[1]) for p in denominators])
    dd = np.array([len(p) - 1 for p in denominators], dtype=float)
    dn = np.array([len(p) - 1 for p in numerators], dtype=float)
    J, I = len(denominators), len(numerators)
    cost = np.concatenate([L, np.zeros(I), np.full(grid, 1 / grid)])
    from scipy.sparse import csr_matrix, hstack, identity, vstack
    rows = hstack([csr_matrix(-b.T), csr_matrix(a.T), -identity(grid)])
    deg = csr_matrix(np.concatenate([-dd, dn, np.zeros(grid)])[None, :])
    res = linprog(cost, A_ub=vstack([rows, deg]).tocsr(), b_ub=np.r_[np.zeros(grid), -1e-9],
                  A_eq=np.concatenate([dd, np.zeros(I + grid)])[None, :], b_eq=[1.0],
                  bounds=[(0, None)] * (J + I + grid), method="highs")
    if res.status != 0:
        raise ValueError("the LP failed: " + res.message)
    m, q = res.x[:J], res.x[J:J + I]
    dens = [{"coeffs": list(p), "multiplicity": int(round(v * scale))} for p, v in zip(denominators, m)
            if round(v * scale) > 0]
    D = sum((len(r["coeffs"]) - 1) * r["multiplicity"] for r in dens)
    nums = [{"coeffs": list(p), "exponent": [int(math.floor(v * D * 2 ** 20)), 2 ** 20]}
            for p, v in zip(numerators, q) if math.floor(v * D * 2 ** 20) > 0]
    return {"denominators": dens, "numerators": nums}


def load(output):
    """A candidate from the output object the evaluator scored (population view)."""
    return {"denominators": [dict(r) for r in output["denominators"]],
            "numerators": [dict(s) for s in output["numerators"]]}
