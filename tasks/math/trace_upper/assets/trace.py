"""The task's objective for a candidate to build on: from sciel.task import asset; trace = asset("trace").

A certificate is (A, b, d): integer polynomials Q as coefficient lists (constant term first), weights b_Q >= 0 and the
6N densities d_i of the cells [i/N, (i+1)/N) on [0, 6]. Its bound is int x dmu / m. It is valid when
f = U_mu - sum_Q b_Q log|Q| >= 0 on the support (hence everywhere), int log|Q| dmu > 0 for each Q, sum_Q b_Q deg Q < m,
and the support is a hull of length above 4 away from 0. For fixed A and N, the densities and weights enter all of these
linearly, so the best certificate is a linear program over f >= eta at sample points; lp() solves it with cutting
planes and rounds the solution to rationals.
"""

from fractions import Fraction
from pathlib import Path
import time

import numpy as np
from scipy.optimize import linprog

from sciel.candidate_modules import load_candidate

_evaluator = load_candidate(str(Path(__file__).resolve().parents[1] / "evaluator.py"), "evaluator")


def load(output) -> tuple:
    """(A, b, d) of a scored certificate, from the output its evaluation recorded (a population entry's "output")."""
    N, first = int(output["N"]), int(output["first"])
    d = [Fraction(0)] * (6 * N)
    for i, x in enumerate(output["densities"]):
        d[first + i] = Fraction(x)
    return [list(q) for q in output["A"]], [Fraction(w) for w in output["b"]], d


def bound(cert) -> Fraction:
    """int x dmu / m exactly."""
    d = cert[2]
    N = len(d) // 6
    return sum(x * (2 * i + 1) for i, x in enumerate(d)) / (2 * N * N) / (sum(d) / N)


def check(cert) -> tuple:
    """(bound, report) exactly as the evaluator scores the certificate; raises ValueError when it is not valid."""
    return _evaluator.certify(*_evaluator._certificate(cert))


def refine(cert, k: int = 2) -> tuple:
    """Every cell split into k (N becomes kN): the same measure, so the same bound, with k times the freedom."""
    A, b, d = cert
    return A, b, [x for x in d for _ in range(int(k))]


def _G(u):
    u = np.asarray(u, dtype=float)
    a = np.abs(u)
    return np.where(a > 0, u * np.log(np.where(a > 0, a, 1.0)) - u, 0.0)


def roots(q) -> np.ndarray:
    """Real parts of the roots of Q in floating point."""
    return np.sort(np.roots(list(reversed(q))).real)


def _log_abs(q, x):
    return np.log(abs(q[-1])) + sum(np.log(np.abs(x - r)) for r in roots(q))


def potential(cert, x) -> np.ndarray:
    """f(x) = U_mu(x) - sum_Q b_Q log|Q(x)| in floating point (+inf at roots of Q)."""
    A, b, d = cert
    N = len(d) // 6
    x = np.atleast_1d(np.asarray(x, dtype=float))
    pad = np.concatenate([[0.0], [float(v) for v in d], [0.0]])
    jumps = np.flatnonzero(np.diff(pad))
    c = np.diff(pad)[jumps]
    out = np.zeros(x.size)
    for i in range(0, x.size, 2048):
        out[i:i + 2048] = (c * _G(x[i:i + 2048, None] - jumps / N)).sum(axis=1)
    with np.errstate(divide="ignore"):
        return out - sum(float(w) * _log_abs(q, x) for w, q in zip(b, A))


def energies(cert, polys=None) -> np.ndarray:
    """int log|Q| dmu for each Q of polys (default the certificate's own), in floating point. A valid certificate makes
    every value >= 0; a negative one is a polynomial the measure is not arithmetic for."""
    A, b, d = cert
    N = len(d) // 6
    pad = np.concatenate([[0.0], [float(v) for v in d], [0.0]])
    jumps = np.flatnonzero(np.diff(pad))
    c = np.diff(pad)[jumps]
    m = sum(float(v) for v in d) / N
    return np.array([m * np.log(abs(q[-1])) + sum((c * _G(r - jumps / N)).sum() for r in roots(q))
                     for q in (A if polys is None else polys)])


def _cells(q, N, cells):
    """int over each cell of log|Q|."""
    lo, hi = np.asarray(cells, dtype=float) / N, (np.asarray(cells, dtype=float) + 1) / N
    return np.log(abs(q[-1])) / N + sum(_G(hi - r) - _G(lo - r) for r in roots(q))


def _rows(A, N, cells, x):
    """f at the points x as a linear function of (densities of the cells, b)."""
    rows = np.empty((x.size, cells.size))
    for i in range(0, x.size, 512):
        y = x[i:i + 512, None]
        rows[i:i + 512] = _G(y - cells[None, :] / N) - _G(y - (cells[None, :] + 1) / N)
    with np.errstate(divide="ignore"):
        logs = np.stack([_log_abs(q, x) for q in A], axis=1)
    return np.hstack([rows, -logs])


def _probe(cells, N):
    """Dense check points of the cells: 257 uniform and 10^-2..10^-6 of a cell from each edge."""
    near = 10.0 ** -np.arange(2, 7)
    offsets = np.unique(np.concatenate([np.linspace(0, 1, 257), near, 1 - near]))
    return ((cells[:, None] + offsets[None, :]) / N).ravel()


def _lowest(x, values, N):
    """The lowest of the points x in each cell."""
    cell = np.floor(x * N + 1e-12).astype(int)
    order = np.lexsort((values, cell))
    return x[order[np.r_[True, np.diff(cell[order]) != 0]]]


def weak_points(cert, depth: int = 30, max_boxes: int = 400000) -> tuple:
    """(points, proved): the evaluator's bisection of E5, run until it proves f >= 0 (proved True) or meets boxes whose
    midpoint has f < 0 in floating point; points are the lowest such midpoint of each cell, the places a linear
    program must add as constraints. Raises ValueError when the exact checks E1-E3 fail."""
    ev = _evaluator
    A, b, d = ev._certificate(cert)
    N, live, _, _, polys = ev.exact_checks(A, b, d)
    pot = ev.Potential(N, ev._jumps(d), b, polys, [ev._roots(P) for P in polys])
    step = 2 ** (ev.LEVEL - 3)
    ks = np.array([i * 2 ** ev.LEVEL + k * step for i in live for k in range(8)], dtype=np.int64)
    width, boxes = np.full(ks.size, step, dtype=np.int64), 0
    for _ in range(depth):
        boxes += ks.size
        bad = pot.bounds(ks, width) <= 0
        if not bad.any():
            return np.zeros(0), True
        ks, width = ks[bad], width[bad]
        mids = (ks + width / 2) / (N * 2.0 ** ev.LEVEL)
        values = potential(cert, mids)
        if (values < 0).any() or boxes > max_boxes:
            return _lowest(mids[values < 0], values[values < 0], N), False
        ks, width = np.concatenate([ks, ks + width // 2]), np.concatenate([width // 2, width // 2])
    return np.zeros(0), False


def lp(A, N: int, window=(0.05, 5.2), per_cell: int = 4, delta: float = 1e-7, eta: float = 1e-7,
       slack: float = 1e-9, rounds: int = 16, deadline: float = None, denominator: int = 10 ** 13,
       verbose: bool = False) -> tuple:
    """The certificate with polynomials A on the grid 1/N minimizing the bound, with support inside window: f >= eta
    at per_cell points of each cell and at cutting planes, int log|Q| dmu >= delta and sum b_Q deg Q <= (1 - slack) m,
    rounded to rationals of the given denominator. Each round adds the lowest point of each cell where a dense check
    finds f < eta/2, or else the points weak_points() finds, until the evaluator's E5 bisection proves the result.
    Raises RuntimeError when the LP fails, the rounds run out, or the deadline (a time.time() value) passes."""
    A = [list(map(int, q)) for q in A]
    cells = np.arange(int(np.ceil(window[0] * N - 1e-9)), int(np.floor(window[1] * N + 1e-9)))
    n, k = cells.size, len(A)
    trace_cost = np.concatenate([(2 * cells + 1) / (2.0 * N * N), np.zeros(k)])
    mass = np.concatenate([np.full(n, 1.0 / N), np.zeros(k)])
    energy = np.hstack([np.stack([_cells(q, N, cells) for q in A]), np.zeros((k, k))])
    degree = np.concatenate([-(1 - slack) / N * np.ones(n), [len(q) - 1 for q in A]])
    rts = np.concatenate([roots(q) for q in A])
    x = np.append(((cells[:, None] + np.arange(per_cell)[None, :] / per_cell) / N).ravel(), (cells[-1] + 1) / N)
    probe = _probe(cells, N)
    probe = probe[np.abs(probe[:, None] - rts).min(axis=1) > 1e-9]
    for _ in range(rounds):
        x = x[np.abs(x[:, None] - rts).min(axis=1) > 1e-7]
        left = None if deadline is None else deadline - time.time()
        if left is not None and left < 1:
            raise RuntimeError("deadline passed")
        res = linprog(trace_cost, A_ub=np.vstack([-_rows(A, N, cells, x), -energy, degree[None, :]]),
                      b_ub=np.concatenate([-eta * np.ones(x.size), -delta * np.ones(k), [0.0]]),
                      A_eq=mass[None, :], b_eq=[1.0], bounds=(0, None), method="highs-ds",
                      options={"primal_feasibility_tolerance": 1e-10, "dual_feasibility_tolerance": 1e-10,
                               **({} if left is None else {"time_limit": left})})
        if res.status != 0:
            raise RuntimeError(res.message)
        z = np.maximum(res.x, 0.0)
        d = [Fraction(0)] * (6 * N)
        for i, v in zip(cells, z[:n]):
            d[i] = Fraction(round(v * denominator), denominator)
        cert = (A, [Fraction(int(v * denominator), denominator) for v in z[n:]], d)
        values = potential(cert, probe)
        low = values < eta / 2
        if low.any():
            new = _lowest(probe[low], values[low], N)
        else:
            new, proved = weak_points(cert)
            if proved:
                return cert
        if verbose:
            print(f"lp: {x.size} points, bound {res.fun:.10f}, dense min {values.min():.3e}, {new.size} new",
                  flush=True)
        if new.size == 0:
            raise RuntimeError("E5 is not proved and no point with f < 0 was found")
        x = np.concatenate([x, new])
    raise RuntimeError("the cutting-plane rounds ran out")
