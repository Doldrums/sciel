"""Weak-type ratio of the centred square maximal function for periodic atomic measures; a program uses it as:
from sciel.task import asset; hl = asset("hl").

A candidate is {"Px": Px, "Py": Py, "atoms": [[x, y, m], ...]}: a period cell [0, Px) x [0, Py) holding distinct
atoms of positive mass m, repeated over the lattice Px Z x Py Z. Every number is an int, a binary64 (read exactly), a
string "p/q" or a [p, q] pair. With level 1, every box B whose sides lie on atom coordinates (side at most REACH)
carries the mass mu(B) of the atoms in it; with rho = (isqrt(floor(mu(B) DR^2 / 4)) - 1) / DR < sqrt(mu(B) / 4) the
square Q(z, rho) contains B for every centre z in the rectangle [b - rho, a + rho] x [f - rho, e + rho] (B = [a, b] x
[e, f]), so M mu > 1 there. The ratio |union of these rectangles in the cell| / mu(cell) is a lower bound for C_47
(Lin's truncation and smearing; Tomar's Lemma 6.2). exact() computes it as a fraction with Tomar's box enumeration
(check_exact2.py, same REACH and DR); value() is the float of the same enumeration, a few times faster.
"""

from fractions import Fraction
import math

import numpy as np

REACH = Fraction(1396, 125)
DR = 2 ** 40
MAX_ATOMS = 128
MAX_PERIOD = 8
MIN_PERIOD = Fraction(1, 2)
MAX_DENOMINATOR_BITS = 128


def _number(v, what):
    if isinstance(v, bool):
        raise ValueError(what + " must be a number")
    if isinstance(v, Fraction):
        q = v
    elif isinstance(v, (list, tuple)):
        if len(v) != 2 or not all(isinstance(t, int) and not isinstance(t, bool) for t in v) or v[1] == 0:
            raise ValueError(what + " pair must be [int, nonzero int]")
        q = Fraction(v[0], v[1])
    elif isinstance(v, str):
        q = Fraction(v)
    elif isinstance(v, (int, float, np.integer, np.floating)):
        if not math.isfinite(float(v)):
            raise ValueError(what + " must be finite")
        q = Fraction(v.item() if hasattr(v, "item") else v)
    else:
        raise ValueError(what + " has unsupported type " + type(v).__name__)
    if q.denominator.bit_length() > MAX_DENOMINATOR_BITS:
        raise ValueError(what + " denominator above 2^128")
    return q


def canonical(raw):
    """(Px, Py, [(x, y, m), ...]) as fractions, sorted; raises ValueError on anything invalid."""
    if not isinstance(raw, dict) or "atoms" not in raw:
        raise ValueError("candidate must be a dict with Px, Py, atoms")
    px, py = _number(raw["Px"], "Px"), _number(raw["Py"], "Py")
    if not (MIN_PERIOD <= px <= MAX_PERIOD and MIN_PERIOD <= py <= MAX_PERIOD):
        raise ValueError("periods must lie in [1/2, 8]")
    atoms = raw["atoms"]
    if not isinstance(atoms, (list, tuple)) or not 1 <= len(atoms) <= MAX_ATOMS:
        raise ValueError("between 1 and %d atoms" % MAX_ATOMS)
    pts = []
    for a in atoms:
        if not isinstance(a, (list, tuple)) or len(a) != 3:
            raise ValueError("an atom is [x, y, m]")
        x, y, m = _number(a[0], "x"), _number(a[1], "y"), _number(a[2], "m")
        if not (0 <= x < px and 0 <= y < py and m > 0):
            raise ValueError("atoms must lie in [0, Px) x [0, Py) with positive mass")
        pts.append((x, y, m))
    if len({(x, y) for x, y, _ in pts}) != len(pts):
        raise ValueError("duplicate atom position")
    return px, py, sorted(pts)


def _window(px, py, pts, dx, dm):
    """Integer atoms of the periodic window that can bound or fill a box whose rectangle meets the cell."""
    pxi, pyi, reachi = int(px * dx), int(py * dx), int(REACH * dx)
    P = [(int(x * dx), int(y * dx), int(m * dm)) for x, y, m in pts]
    mcell = sum(p[2] for p in P)
    nb = (int(REACH / px) + 2) * (int(REACH / py) + 2)
    rho_max = Fraction(math.isqrt(nb * mcell * DR * DR // (4 * dm)) + 1, DR)
    rmi = int(math.ceil(rho_max * dx)) + 1
    ext = reachi + rmi
    mx, my = ext // pxi + 2, ext // pyi + 2
    lo, hix, hiy = -rmi - reachi, pxi + rmi + reachi, pyi + rmi + reachi
    W = [(x + i * pxi, y + j * pyi, m) for i in range(-mx, mx + 1) for j in range(-my, my + 1) for x, y, m in P]
    W = [w for w in W if lo <= w[0] <= hix and lo <= w[1] <= hiy]
    return pxi, pyi, reachi, rmi, mcell, W


def _pairs(V, cell, rmi, reachi):
    """Index pairs a <= b of Tomar's loops (exact integer tests): V[a] >= -rmi, V[b] - V[a] <= reachi,
    V[b] <= cell + rmi."""
    out_a, out_b = [], []
    for a, va in enumerate(V):
        if va < -rmi:
            continue
        for b in range(a, len(V)):
            if V[b] - va > reachi or V[b] > cell + rmi:
                break
            out_a.append(a)
            out_b.append(b)
    return np.array(out_a, dtype=np.int64), np.array(out_b, dtype=np.int64)


def _boxes(px, py, pts):
    """Every box of Tomar's enumeration whose rectangle meets the cell, by a float screen with margin and an exact
    integer test: returns (rectangles as integers over L, L, mass of the cell over dm, dm)."""
    dx = 1
    for v in [px, py, REACH] + [p[0] for p in pts] + [p[1] for p in pts]:
        dx = math.lcm(dx, v.denominator)
    dm = 1
    for p in pts:
        dm = math.lcm(dm, p[2].denominator)
    L = math.lcm(dx, DR)
    sx, sr = L // dx, L // DR
    pxi, pyi, reachi, rmi, mcell, W = _window(px, py, pts, dx, dm)
    X = sorted({w[0] for w in W})
    Y = sorted({w[1] for w in W})
    xi = {v: i for i, v in enumerate(X)}
    yi = {v: i for i, v in enumerate(Y)}
    nx, ny = len(X), len(Y)
    G = [[0] * (ny + 1) for _ in range(nx + 1)]
    for x, y, m in W:
        G[xi[x] + 1][yi[y] + 1] += m
    for i in range(1, nx + 1):
        row, prev, acc = G[i], G[i - 1], 0
        for j in range(1, ny + 1):
            acc += row[j]
            row[j] = prev[j] + acc
    # float screen in cell units: mass relative error ~1e-15, margin 1e-9 of a period
    Gf = np.array([[v / dm for v in row] for row in G])
    Xf = np.array([v / dx for v in X])
    Yf = np.array([v / dx for v in Y])
    xa, xb = _pairs(X, pxi, rmi, reachi)
    ya, yb = _pairs(Y, pyi, rmi, reachi)
    pxf, pyf, tol = float(px), float(py), 1e-9
    yaf, ybf = Yf[ya], Yf[yb]
    cand = []
    step = max(1, 4_000_000 // max(len(ya), 1))
    for s in range(0, len(xa), step):
        a, b = xa[s:s + step], xb[s:s + step]
        M = (Gf[b + 1][:, yb + 1] - Gf[a][:, yb + 1] - Gf[b + 1][:, ya] + Gf[a][:, ya])
        rho = np.sqrt(np.maximum(M, 0.0) / 4)
        ok = (M > 0)
        ok &= (np.minimum(Xf[a][:, None] + rho, pxf) - np.maximum(Xf[b][:, None] - rho, 0.0) > -tol)
        ok &= (np.minimum(yaf[None, :] + rho, pyf) - np.maximum(ybf[None, :] - rho, 0.0) > -tol)
        r, c = np.nonzero(ok)
        cand.extend(zip((a[r]).tolist(), (b[r]).tolist(), ya[c].tolist(), yb[c].tolist()))
    R = []
    four_dm = 4 * dm
    for ia, ib, ja, jb in cand:
        Ms = G[ib + 1][jb + 1] - G[ia][jb + 1] - G[ib + 1][ja] + G[ia][ja]
        if Ms <= 0:
            continue
        k = math.isqrt(Ms * DR * DR // four_dm) - 1
        if k <= 0:
            continue
        rho = k * sr
        x0, x1 = max(X[ib] * sx - rho, 0), min(X[ia] * sx + rho, pxi * sx)
        if x1 <= x0:
            continue
        y0, y1 = max(Y[jb] * sx - rho, 0), min(Y[ja] * sx + rho, pyi * sx)
        if y1 <= y0:
            continue
        R.append((x0, x1, y0, y1))
    return R, L, mcell, dm


def _union_area(R):
    """Area of a union of axis rectangles (x0, x1, y0, y1) with exact integer (or float) coordinates."""
    if not R:
        return 0
    ys = sorted({r[2] for r in R} | {r[3] for r in R})
    yi = {v: i for i, v in enumerate(ys)}
    ev = []
    for x0, x1, y0, y1 in R:
        ev.append((x0, 1, yi[y0], yi[y1]))
        ev.append((x1, -1, yi[y0], yi[y1]))
    ev.sort()
    n = len(ys) - 1
    cnt = [0] * (4 * n + 4)
    ln = [0] * (4 * n + 4)

    def upd(node, lo, hi, a, b, d):
        if b <= lo or hi <= a:
            return
        if a <= lo and hi <= b:
            cnt[node] += d
        else:
            mid = (lo + hi) // 2
            upd(2 * node, lo, mid, a, b, d)
            upd(2 * node + 1, mid, hi, a, b, d)
        if cnt[node] > 0:
            ln[node] = ys[hi] - ys[lo]
        elif hi - lo == 1:
            ln[node] = 0
        else:
            ln[node] = ln[2 * node] + ln[2 * node + 1]

    area, last = 0, ev[0][0]
    for x, d, a, b in ev:
        area += ln[1] * (x - last)
        last = x
        if n > 0:
            upd(1, 0, n, a, b, d)
    return area


def exact(candidate):
    """(ratio, area, cell mass, rectangles) as fractions, exactly as scored."""
    px, py, pts = canonical(candidate)
    R, L, mcell, dm = _boxes(px, py, pts)
    area = Fraction(_union_area(R), L * L)
    mass = Fraction(mcell, dm)
    return area / mass, area, mass, len(R)


def rectangles(candidate):
    """The scored rectangles in the cell as floats (x0, x1, y0, y1)."""
    px, py, pts = canonical(candidate)
    R, L, _, _ = _boxes(px, py, pts)
    return [tuple(float(Fraction(v, L)) for v in r) for r in R]


def value(candidate):
    """Float ratio of the same enumeration (rectangles exact, union in binary64)."""
    px, py, pts = canonical(candidate)
    R, L, mcell, dm = _boxes(px, py, pts)
    Rf = [tuple(float(Fraction(v, L)) for v in r) for r in R]
    return float(_union_area(Rf)) / float(Fraction(mcell, dm))


def holes(candidate, n=400):
    """Centres (x, y) of the n x n grid points of the cell left uncovered, i.e. where the bound loses area."""
    px, py = float(_number(candidate["Px"], "Px")), float(_number(candidate["Py"], "Py"))
    cover = np.zeros((n + 1, n + 1), dtype=np.int32)
    for x0, x1, y0, y1 in rectangles(candidate):
        i0, i1 = int(np.ceil(x0 / px * n - 0.5)), int(np.ceil(x1 / px * n - 0.5))
        j0, j1 = int(np.ceil(y0 / py * n - 0.5)), int(np.ceil(y1 / py * n - 0.5))
        if i1 > i0 and j1 > j0:
            cover[i0, j0] += 1
            cover[i1, j0] -= 1
            cover[i0, j1] -= 1
            cover[i1, j1] += 1
    cover = cover.cumsum(0).cumsum(1)[:n, :n]
    i, j = np.nonzero(cover == 0)
    return np.stack([(i + 0.5) / n * px, (j + 0.5) / n * py], axis=1)


def make(px, py, atoms):
    """A candidate from periods and [x, y, m] atoms (floats are read exactly); atoms are wrapped into the cell."""
    px, py = _number(px, "Px"), _number(py, "Py")
    out = []
    for x, y, m in atoms:
        x, y = _number(x, "x") % px, _number(y, "y") % py
        out.append([_text(x), _text(y), _text(_number(m, "m"))])
    return {"Px": _text(px), "Py": _text(py), "atoms": out}


def _text(q):
    return str(q.numerator) if q.denominator == 1 else "%d/%d" % (q.numerator, q.denominator)


def floats(candidate):
    """(Px, Py, array of [x, y, m]) in binary64, for search."""
    px, py, pts = canonical(candidate)
    return float(px), float(py), np.array([[float(x), float(y), float(m)] for x, y, m in pts])


def load(output):
    return {"Px": output["Px"], "Py": output["Py"], "atoms": [list(a) for a in output["atoms"]]}
