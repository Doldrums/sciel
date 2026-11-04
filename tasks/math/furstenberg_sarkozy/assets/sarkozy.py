"""The task's objective for a candidate to build on: from sciel.task import asset; sarkozy = asset("sarkozy").

A certificate is a dict {alpha, chains, codes, binary} as described in the task (frontier.json is Naslund's).
certify(cert) is the scorer: the exact alpha it proves, the parts and the surplus, or an exception. The geometry fixes
how far alpha can go: capacity(cert) is that largest alpha (an mpmath number, 50 digits), with the best power of
every component; finalize(cert) rewrites alpha, the code powers and the binary f, a, vector and depth just below it,
so the result certifies (about 1e-30 below capacity). Changing a code's widths: relayout(code, widths) puts float
widths on the code's square-difference order (longest path, scaled into [0, 1)) as exact integer starts and widths.
"""

import copy
import json
import math
from fractions import Fraction
from pathlib import Path

import mpmath as mp
import numpy as np

from sciel.candidate_modules import load_candidate

_evaluator = load_candidate(str(Path(__file__).resolve().parents[1] / "evaluator.py"), "evaluator")
certify = _evaluator.certify
mp.mp.dps = 50
_SLACK = mp.mpf("1e-30")


def load(output):
    """The certificate of a population output."""
    if isinstance(output, str):
        output = json.loads(output)
    return output.get("output", output)["certificate"]


def incoming(code):
    """Bitmask per point of the points with a square difference into it (the evaluator's order relation)."""
    words = [tuple(tuple(word) for word in point) for point in code["points"]]
    return _evaluator.incoming(code["primes"], words)


def relayout(code, widths, denominator=10 ** 16):
    """A copy of code with the given positive float widths, placed by longest path along the square-difference order
    (points taken by current start), all scaled into [0, 1 - 1e-12]; exact integers over denominator."""
    n = len(code["points"])
    inc = incoming(code)
    order = sorted(range(n), key=lambda i: (code["start_numerators"][i], code["start_numerators"][i]
                                              + code["width_numerators"][i]))
    rank = {i: r for r, i in enumerate(order)}
    w = np.asarray(widths, dtype=np.float64)
    end = np.zeros(n)
    for i in order:
        preds = [j for j in _bits(inc[i] & ~(1 << i))]
        if any(rank[j] > rank[i] for j in preds):
            raise ValueError("the current starts do not order the code")
        end[i] = max((end[j] for j in preds), default=0.0) + w[i]
    scale = (1 - 1e-12) / end.max()
    wi = [max(1, int(x * scale * denominator)) for x in w]
    ends = [0] * n
    for i in order:
        ends[i] = max((ends[j] for j in _bits(inc[i] & ~(1 << i))), default=0) + wi[i]
    out = dict(code, width_numerators=wi, start_numerators=[e - x for e, x in zip(ends, wi)],
               width_denominator=denominator)
    if max(ends) > denominator:
        raise ValueError("relayout overflow")
    return out


def _bits(mask):
    while mask:
        low = mask & -mask
        yield low.bit_length() - 1
        mask ^= low


def _chain_f(p, t, alpha):
    return 1 - (2 * alpha - 1) * mp.log(p) / mp.log(t)


class _Code:
    def __init__(self, code):
        counts = {}
        for w in code["width_numerators"]:
            counts[w] = counts.get(w, 0) + 1
        D = code["width_denominator"]
        self.logs = [mp.log(mp.mpf(w) / D) for w in counts]
        self.mult = list(counts.values())
        self.L = mp.fsum(e * mp.log(p) for p, e in zip(code["primes"], code["depths"]))
        self.flog = np.array([float(x) for x in self.logs])
        self.fmult = np.array(self.mult, dtype=np.float64)

    def solve(self, alpha):
        """f with log sum w^f = (2 alpha - 1) L, and df/dalpha."""
        target = (2 * alpha - 1) * self.L
        f, tf = 0.05, float(target)
        for _ in range(60):
            e = self.fmult * np.exp(f * self.flog)
            f -= (math.log(e.sum()) - tf) / ((e * self.flog).sum() / e.sum())
        f = mp.mpf(f)
        for _ in range(3):
            terms = [m * mp.exp(f * lw) for m, lw in zip(self.mult, self.logs)]
            Z = mp.fsum(terms)
            dZ = mp.fsum(t * lw for t, lw in zip(terms, self.logs))
            f -= (mp.log(Z) - target) / (dZ / Z)
        return f, 2 * self.L / (dZ / Z)


class _Binary:
    def __init__(self, binary):
        self.states = binary["states"]
        self.n = len(self.states)
        self.edges = [(i, b["child"], mp.log(mp.mpf(b["scale"][0]) / b["scale"][1]))
                      for i, s in enumerate(self.states) for b in s["branches"]]

    def matrix(self, f, derivative=False):
        M = [[mp.mpf(0)] * self.n for _ in range(self.n)]
        for i, j, lu in self.edges:
            M[i][j] += mp.exp(f * lu) * (lu if derivative else 1)
        return M

    def perron(self, f):
        """rho(f), right vector v, left vector u, d log rho / df."""
        M = self.matrix(f)
        A = np.array([[float(x) for x in row] for row in M])
        vals, R = np.linalg.eig(A)
        k = int(np.argmax(vals.real))
        vals, Lv = np.linalg.eig(A.T)
        k2 = int(np.argmax(vals.real))
        v = [mp.mpf(abs(x)) for x in R[:, k].real]
        u = [mp.mpf(abs(x)) for x in Lv[:, k2].real]
        for _ in range(400):
            v2 = [mp.fsum(M[i][j] * v[j] for j in range(self.n)) for i in range(self.n)]
            u2 = [mp.fsum(M[i][j] * u[i] for i in range(self.n)) for j in range(self.n)]
            rho = v2[0] / v[0]
            res = max(abs(v2[i] - rho * v[i]) / v[i] for i in range(self.n))
            v, u = [x / v2[0] for x in v2], [x / u2[0] for x in u2]
            if res < mp.mpf("1e-45") * rho:
                break
        Mv = [mp.fsum(M[i][j] * v[j] for j in range(self.n)) for i in range(self.n)]
        uv = mp.fsum(a * b for a, b in zip(u, v))
        rho = mp.fsum(a * b for a, b in zip(u, Mv)) / uv
        dM = self.matrix(f, True)
        drho = mp.fsum(u[i] * dM[i][j] * v[j] for i in range(self.n) for j in range(self.n)) / uv
        return rho, v, u, drho / rho

    def solve(self, alpha, f=0.155):
        """f with rho(f) = 4^alpha, and df/dalpha."""
        f, target = mp.mpf(f), alpha * mp.log(4)
        for _ in range(30):
            rho, _, _, dlog = self.perron(f)
            step = (mp.log(rho) - target) / dlog
            f -= step
            if abs(step) < mp.mpf("1e-45"):
                break
        return f, mp.log(4) / dlog


def _powers(cert, alpha, models, guess):
    chains = [_chain_f(p, len(c), alpha) for p, c in cert.get("chains") or []]
    dchains = [-2 * mp.log(p) / mp.log(len(c)) for p, c in cert.get("chains") or []]
    codes = [m.solve(alpha) for m in models[0]]
    binary = [models[1].solve(alpha, guess)] if models[1] else []
    total = mp.fsum(chains) + mp.fsum(f for f, _ in codes + binary) - alpha
    slope = mp.fsum(dchains) + mp.fsum(d for _, d in codes + binary) - 1
    return total, slope, chains, codes, binary


def capacity(cert, detail=False):
    """The largest alpha the certificate's geometry supports (sum of best powers = alpha)."""
    models = ([_Code(c) for c in cert.get("codes") or []], _Binary(cert["binary"]) if cert.get("binary") else None)
    alpha = Fraction(cert["alpha"])
    alpha = mp.mpf(alpha.numerator) / alpha.denominator
    guess = float(Fraction(cert["binary"]["f"])) if cert.get("binary") else 0.155
    for _ in range(8):
        total, slope, chains, codes, binary = _powers(cert, alpha, models, guess)
        if binary:
            guess = binary[0][0]
        alpha -= total / slope
        if abs(total) < mp.mpf("1e-45"):
            break
    if not detail:
        return alpha
    return alpha, [f for f, _ in codes], (binary[0][0] if binary else None), models


def _floor(x, digits):
    return Fraction(int(mp.floor(x * mp.mpf(10) ** digits)), 10 ** digits)


def finalize(cert):
    """A copy of cert with alpha just below capacity(cert) and every power, a, vector and depth set to certify it."""
    alpha, fcodes, fbin, models = capacity(cert, True)
    out = copy.deepcopy(cert)
    q = _floor(alpha - _SLACK, 40)
    out["alpha"] = str(q)
    for code, f in zip(out.get("codes") or [], fcodes):
        code["power"] = str(_floor(f, 45))
    if fbin is not None:
        f = _floor(fbin, 45)
        rho, v, _, _ = models[1].perron(mp.mpf(f.numerator) / f.denominator)
        b = out["binary"]
        root = b["root"]
        b["f"] = str(f)
        b["a"] = str(_floor(rho * (1 - mp.mpf("1e-36")), 50))
        b["vector"] = [str(_floor(x / v[root], 60)) if i != root else "1" for i, x in enumerate(v)]
        b["depth"] = 10 ** 60
    return out
