"""Four-strand superhelix knots and a certified upper bound on Tao's constant 22b; a program uses it as:
from sciel.task import asset; helix = asset("helix").

C_22b is the largest b with L >= b C for every alternating knot, L the ropelength (unit radius thickness), C the
crossing number. Huh-Kim-Oh's four-strand superhelix with Li's unequal radii (R. Pinho, PR 324 on
teorth/optimizationproblems) closes into T(2, 4 P n + 3): two coaxial axes alpha_c(t) = c R e(t) + h t ez, c = +-1,
each carrying an outer and an inner strand on opposite sides,

    P_c,r(t) = alpha_c(t) + r(u) (cos u n_c(t) - sin u b_c(t)),   u = N t,  N = P / Q,

n_c, b_c the Frenet normal and binormal of the axis, r = +a(u) (outer) and r = -i(u) (inner) with a, i > 0
trigonometric polynomials in u (PR 324: constants). One minor period u in [0, 2 pi] holds four crossings, so the
length per crossing of the limit is c = (1/2) int_0^2pi (|P'_a| + |P'_-i|) du and C_22b <= c / thickness; the
caps cost O(1) and drop out as n -> oo.

A candidate is {"P": int, "Q": int, "R": x, "h": x, "outer": [a0, a1, b1, a2, b2, ...], "inner": [...]} with
a(u) = a0 + sum_k (a_k cos k u + b_k sin k u), k <= MAX_ORDER.

Certification (certify): every strand is a trigonometric polynomial in u, so the sup norms of its derivatives are
bounded by sums of |coefficient| frequency^k. Thickness = min(curvature radius, half the doubly critical
self-distance) (Cantarella-Kusner-Sullivan) is bounded below by
  - curvature: branch and bound on tau^2 |v x a|^2 - |v|^6 <= 0 with a second-order Taylor bound per cell;
  - other bundle: axes at distance >= 2R when h >= R, so tau <= R - max r;
  - same bundle: screw symmetry (u -> u + 2 pi is an isometry) reduces pairs to u in [0, 2 pi], phase gap delta
    bounded by the vertical rise; a same-strand arc shorter than pi tau cannot be doubly critical (curvature
    <= 1 / tau), the rest is a branch and bound on the squared distance with a second-order Taylor bound;
  - length: per cell w sqrt(|v|^2 + |a|^2 w^2 / 12) + J w^3 / 24, an upper bound on int |v|.
Floating point is binary64 with an absolute margin on every cell.
"""

from math import gcd

import numpy as np

MAX_ORDER = 4
MAX_P = 4000
MAX_Q = 64
SLACK = 1e-11
TOL = 2e-10
LENGTH_CELLS = 1 << 15
MAX_CELLS = 4_000_000


def canonical(candidate):
    """(P, Q, R, h, outer, inner) checked; raises ValueError."""
    if not isinstance(candidate, dict) or not {"P", "Q", "R", "h", "outer", "inner"} <= set(candidate):
        raise ValueError('the candidate must be {"P", "Q", "R", "h", "outer", "inner"}')
    P, Q = candidate["P"], candidate["Q"]
    if any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) for v in (P, Q)):
        raise ValueError("P and Q must be integers")
    P, Q = int(P), int(Q)
    if not (1 <= Q <= MAX_Q and 2 * Q < P <= MAX_P and gcd(P, Q) == 1):
        raise ValueError(f"need 1 <= Q <= {MAX_Q}, 2Q < P <= {MAX_P}, gcd(P, Q) = 1")
    R, h = float(candidate["R"]), float(candidate["h"])
    radii = []
    for key in ("outer", "inner"):
        c = [float(x) for x in candidate[key]]
        if len(c) % 2 != 1 or len(c) > 2 * MAX_ORDER + 1:
            raise ValueError(f"{key} must list a0, a1, b1, ... up to order {MAX_ORDER}")
        radii.append(c)
    if not all(np.isfinite([R, h, *radii[0], *radii[1]])):
        raise ValueError("non-finite parameter")
    if not 0 < R <= h:
        raise ValueError("need 0 < R <= h (the axes then stay 2R apart)")
    for key, c in zip(("outer", "inner"), radii):
        if c[0] - sum(np.hypot(c[1::2], c[2::2])) <= 0:
            raise ValueError(f"the {key} radius must stay positive")
    return P, Q, R, h, radii[0], radii[1]


class Strand:
    """The c = +1 strand of signed radius r(u) as trig polynomials: horizontal sum G_m e^{i (m + nu) u}, vertical
    (h / N) u + sum Z_m e^{i m u}."""

    def __init__(self, P, Q, R, h, coeffs, sign):
        N = P / Q
        self.nu, s = 1 / N, np.hypot(R, h)
        K = (len(coeffs) - 1) // 2
        rho = np.zeros(2 * K + 1, complex)
        rho[K] = sign * coeffs[0]
        for k in range(1, K + 1):
            z = sign * complex(coeffs[2 * k - 1], -coeffs[2 * k]) / 2
            rho[K + k], rho[K - k] = z, np.conj(z)
        M = K + 1
        up, down = np.zeros(2 * M + 1, complex), np.zeros(2 * M + 1, complex)
        up[2:], down[:-2] = rho, rho      # rho e^{iu}, rho e^{-iu}
        cos_r, sin_r = (up + down) / 2, (up - down) / 2j
        self.m = np.arange(-M, M + 1)
        self.G = -cos_r + 1j * (h / s) * sin_r
        self.G[M] += R
        self.Z = -(R / s) * sin_r
        self.rise, self.tilt = h / N, R / s
        self.rmax = sum(abs(rho))
        self.lean = (R / s) * sum(abs(sin_r) * abs(self.m))
        self.bound = [np.hypot(sum(abs(self.G) * abs(self.m + self.nu) ** k),
                               sum(abs(self.Z) * abs(self.m) ** k) + (self.rise if k == 1 else 0)) for k in range(5)]

    def eval(self, u, k=0):
        """k-th derivative in u at the points u, as an (n, 3) array."""
        u = np.asarray(u, float)[:, None]
        w = self.m + self.nu
        hz = (self.G * (1j * w) ** k * np.exp(1j * w * u)).sum(1)
        z = (self.Z * (1j * self.m) ** k * np.exp(1j * self.m * u)).sum(1).real
        z = z + (self.rise * u[:, 0] if k == 0 else self.rise if k == 1 else 0)
        return np.stack([hz.real, hz.imag, z], 1)


def strands(candidate):
    P, Q, R, h, outer, inner = canonical(candidate)
    a, i = Strand(P, Q, R, h, outer, 1.0), Strand(P, Q, R, h, inner, -1.0)
    if a.rise <= a.lean or i.rise <= i.lean:
        raise ValueError("every strand must climb monotonically (a graph over Z)")
    if R <= max(a.rmax, i.rmax):
        raise ValueError("the bundles overlap")
    return a, i, R


def length(a, i, cells=LENGTH_CELLS):
    """Certified upper bound on (1/2) int_0^2pi (|P'_a| + |P'_i|) du, the length per crossing."""
    w = 2 * np.pi / cells
    u = (np.arange(cells) + 0.5) * w
    total = 0.0
    for st in (a, i):
        v, acc = st.eval(u, 1), st.eval(u, 2)
        cell = w * np.sqrt((v * v).sum(1) + (acc * acc).sum(1) * w * w / 12) + st.bound[3] * w ** 3 / 24
        total += cell.sum() * (1 + 1e-14) + cells * 1e-15
    return total / 2


def curvature_radius(st):
    """Certified lower bound on the curvature radius of a strand over one period, by branch and bound."""
    u = np.linspace(0, 2 * np.pi, 1 << 17, endpoint=False)
    v, acc = st.eval(u, 1), st.eval(u, 2)
    rho = np.min(np.linalg.norm(v, axis=1) ** 3 / np.linalg.norm(np.cross(v, acc), axis=1))
    V, A, J, Qd = st.bound[1:5]
    for f in (1e-9, 1e-7, 1e-5, 1e-3, 3e-2):
        tau = rho * (1 - f)
        t2 = tau * tau
        F2 = 2 * t2 * (V * V * J * J + V * A * (A * J + V * Qd)) + 6 * (4 * V ** 4 * A * A + V ** 4 * (A * A + V * J))
        c, r = (np.arange(1 << 12) + 0.5) * (2 * np.pi / (1 << 12)), np.pi / (1 << 12)
        while c.size and c.size < MAX_CELLS:
            v, acc, j = st.eval(c, 1), st.eval(c, 2), st.eval(c, 3)
            vxa = np.cross(v, acc)
            vv = (v * v).sum(1)
            F = t2 * (vxa * vxa).sum(1) - vv ** 3
            dF = 2 * t2 * (vxa * np.cross(v, j)).sum(1) - 6 * vv * vv * (v * acc).sum(1)
            bad = F + np.abs(dF) * r + F2 * r * r / 2 + SLACK >= 0
            if np.any(F[bad] >= 0):
                break
            c, r = np.concatenate([c[bad] - r / 2, c[bad] + r / 2]), r / 2
        else:
            if not c.size:
                return tau
    raise ValueError("curvature certification failed")


def _distance_min(X, Y, lo, hi):
    """Certified lower bound on min |X(u) - Y(u + d)|^2 over u in [0, 2 pi], d in [lo, hi]."""
    if hi <= lo:
        return np.inf
    VX, AX, VY, AY = X.bound[1], X.bound[2], Y.bound[1], Y.bound[2]
    nu, nd = 64, max(4, int(np.ceil((hi - lo) / (2 * np.pi / 64))))
    ru, rd = np.pi / nu, (hi - lo) / (2 * nd)
    U, D = np.meshgrid((np.arange(nu) + 0.5) * 2 * ru, lo + (np.arange(nd) + 0.5) * 2 * rd, indexing="ij")
    U, D = U.ravel(), D.ravel()
    best, floor = np.inf, np.inf
    while U.size:
        if U.size > MAX_CELLS:
            raise ValueError("distance certification did not converge")
        x, y = X.eval(U), Y.eval(U + D)
        diff = x - y
        f = (diff * diff).sum(1)
        g1, g2 = 2 * (diff * X.eval(U, 1)).sum(1), -2 * (diff * Y.eval(U + D, 1)).sum(1)
        r1, r2 = ru, ru + rd
        dmax = np.sqrt(f) + VX * r1 + VY * r2
        H11, H22, H12 = 2 * VX * VX + 2 * dmax * AX, 2 * VY * VY + 2 * dmax * AY, 2 * VX * VY
        lb = f - np.abs(g1 + g2) * ru - np.abs(g2) * rd - (H11 * r1 * r1 + 2 * H12 * r1 * r2 + H22 * r2 * r2) / 2 - SLACK
        best = min(best, f.min())
        keep = lb < best - TOL
        floor = min(floor, lb[~keep].min(initial=np.inf))
        U, D = U[keep], D[keep]
        ru, rd = ru / 2, rd / 2
        U = np.concatenate([U - ru, U - ru, U + ru, U + ru])
        D = np.concatenate([D - rd, D + rd, D - rd, D + rd])
    return min(floor, best - TOL)


def certify(candidate):
    """(coefficient, length per crossing, thickness, binding) for the candidate, all certified; raises ValueError."""
    a, i, R = strands(candidate)
    cross = R - max(a.rmax, i.rmax)
    curv = min(curvature_radius(a), curvature_radius(i))
    lo = min(curv, cross)
    for _ in range(6):
        reach = {}
        for X, Y, same in ((a, a, True), (i, i, True), (a, i, False)):
            top = (2 * cross + (X.rmax + Y.rmax) * X.tilt) / X.rise
            if same:
                start = np.pi * lo / X.bound[1] * (1 - 1e-12)
                reach[(id(X), id(Y))] = _distance_min(X, Y, start, top)
            else:
                reach[(id(X), id(Y))] = _distance_min(X, Y, -top, top)
        dist = 0.5 * np.sqrt(max(min(reach.values()), 0.0))
        if dist >= lo:
            break
        lo = dist * (1 - 1e-12)
    else:
        raise ValueError("thickness did not settle")
    tau = min(curv, cross, dist)
    binding = ("curvature", "bundles", "contact")[int(np.argmin([curv, cross, dist]))]
    L = length(a, i)
    return L / tau * (1 + 1e-13), L, tau, binding


def value(candidate):
    """Fast float estimate of the coefficient (grid minima, no certification), about 50 ms."""
    a, i, R = strands(candidate)
    u = np.linspace(0, 2 * np.pi, 4096, endpoint=False)
    rad = []
    for st in (a, i):
        v, acc = st.eval(u, 1), st.eval(u, 2)
        rad.append(np.min(np.linalg.norm(v, axis=1) ** 3 / np.linalg.norm(np.cross(v, acc), axis=1)))
    tau = min(rad + [R - max(a.rmax, i.rmax)])
    du = np.linspace(0, 2 * np.pi, 256, endpoint=False)
    top = (2 * tau + 2 * a.rmax * a.tilt) / a.rise
    d = np.linspace(-top, top, 1201)
    U, D = np.meshgrid(du, d, indexing="ij")
    U, D = U.ravel(), D.ravel()
    for X, Y, same in ((a, a, True), (i, i, True), (a, i, False)):
        m = np.abs(D) * X.bound[1] >= np.pi * tau if same else np.ones_like(D, bool)
        diff = X.eval(U[m]) - Y.eval(U[m] + D[m])
        if m.any():
            tau = min(tau, 0.5 * np.sqrt((diff * diff).sum(1).min()))
    return length(a, i, 4096) / tau


def make(P, Q, R, h, outer, inner):
    return {"P": int(P), "Q": int(Q), "R": float(R), "h": float(h),
            "outer": [float(x) for x in outer], "inner": [float(x) for x in inner]}


def load(output):
    return make(output["P"], output["Q"], output["R"], output["h"], output["outer"], output["inner"])
