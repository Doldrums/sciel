"""The task's objective for a candidate to build on: from sciel.task import asset; c7 = asset("c7").

A certificate is a recipe for an independent set in a strong power of C_7, built from the 367-word base gadget G5 by
exact profile algebra; value(certificate) returns (M, d), the independent set size and the dimension, and the bound is
M^(1/d). Everything is integer arithmetic.

Profiles. A Gao gadget has profile (a, t, s, o, h, v): main code size, private pairs, auxiliary set size and its
split into the neutral part X^0 and the one-sided parts X^H, X^V. gao(L, R) is Gao's product lemma (Tandon's
Lemma 1), sigma swaps H and V, phi maps a gadget to the BPZ seven-family vector (B, N, A, D, O, H, V) =
(a - t, o, v, h, t, t, t).

Theorem 1 (Tandon, heterogeneous Gao product). hetgao(L, R) with a left pattern (L, J0, JH, JV) keeps a, t of
gao(L, R) and replaces the auxiliary set by J0 x X_R^0 + JH x X_R^H + JV x X_R^V. The codebooks and their counts
q = |J minus N(X_L^0)| depend on L only, so a left pattern applies to every right factor R. Patterns:
  Tandon's ten, with his certified counts: P10, P10A, P10D, P15X, P15A8, P15Ah, P15A5, P15D, P30L, P15w;
  Ppp: L = G30^{++}, J0 = X(G30^het), JH = JV = T(I_30^{++}), q counted here;
  S<A,B,P>: L = Gao(A, B) with A the left factor of a Tandon pattern P (G10, G10A, G10D, G15X, G15AX, G15DX, G15h)
    and B any of GADGETS (prefix 's' for sigma); J0 = X(HetGao_P(A, B)); JH = JV = T(I), I the main code of the
    product of the plain gadgets G5 / G10 / G15h of the same dimensions; q counted here;
  N<Q,C>: L = Gao(L_Q, C) for a pattern Q of the S or N kind and C in GADGETS, J0 = X(HetGao_Q(L_Q, C)),
    JH = JV = T(I) with I the main code of the product of Q's main code and the plain gadget of C's dimension.
The S and N patterns and Ppp are those of L. Tang's certificate (teorth/optimizationproblems PR 296); each new count
is an exact block count over the base data, 1 to 30 s, cached per pattern within a process.

BPZ top level (Buys, Polak, Zuiddam). apply_rule(rule, vectors) combines seven-family vectors by one of RULES
(S2a, S2b, S3a..S3h, Lean-transcribed, admissibility re-checked) and apply_code(code, vectors) gives the size of the
independent set of a terminal code (K3a, K4a, K4b). Dimensions add.

Certificate format (a dict):
  "gadgets": steps in order, each {"name", "op": "hetgao", "pattern", "R"} | {"name", "op": "gao", "L", "R"} |
    {"name", "op": "sigma", "of"}; operands are earlier names or library gadgets (library(): G5, sG5 and the
    gadgets of Tandon's Theorem 2 chain, e.g. G15h, G30_6, G40_8, G55, with dimensions).
  "vectors" (optional): steps {"name", "rule", "inputs": [vector names]}; a vector name is "Phi(<gadget>)", one of
    Tandon's n6, n8, n11, or an earlier vector step.
  "top": {"terminal_code", "inputs": [k vector names]} or, as in PR 296, {"rule", "rule_inputs": [...],
    "terminal_code"} (k equal rule subtrees) or {"rule", "subtrees": [[...], ...], "terminal_code"}.
"""

from collections import Counter
from decimal import Decimal, getcontext
from fractions import Fraction
from functools import lru_cache, reduce
from itertools import combinations, product
from pathlib import Path
import re

from sciel.candidate_modules import load_candidate

_base = load_candidate(str(Path(__file__).resolve().parent / "base.py"), "c7_base")
_counts = load_candidate(str(Path(__file__).resolve().parent / "counts.py"), "c7_counts")

# Tandon's ledger (data/certificates.json); derive_counts() recomputes all five from the base data.
COUNTS = {"C1": 27488, "C2": 12839823, "C3": 14045805, "C4": 841760069965664, "q15": 12872271}
MAX_STEPS = 400
MAX_DIMENSION = 20000


# --- exact profile algebra -------------------------------------------------------------------------------------------

def profile(a, t, s, o, h, v):
    p = tuple(int(x) for x in (a, t, s, o, h, v))
    if min(p) < 0 or p[1] > p[0] or p[3] + p[4] + p[5] != p[2]:
        raise ValueError(f"not a gadget profile: {p}")
    return p


def sigma(p):
    a, t, s, o, h, v = p
    return (a, t, s, o, v, h)


def gao(L, R):
    a1, t1, s1, o1, h1, v1 = L
    a2, t2, s2, o2, h2, v2 = R
    return profile((a1 - t1) * (a2 - t2) + t1 * s2 + s1 * t2, t1 * o2 + o1 * t2, s1 * s2,
                   o1 * o2 + (h1 + v1) * (h2 + v2), h1 * o2 + o1 * v2, v1 * o2 + o1 * h2)


def neutral_from(p):
    """X(G') as J0 for a product from the same ordered input pair: (j0, o0, h0, v0)."""
    return (p[2], p[3], p[4], p[5])


def own_auxiliary(p):
    """X(G) as a side codebook over itself: q = s - o."""
    return (p[2], p[2] - p[3])


def hetgao(L, R, J0, JH, JV):
    aL, tL, sL, oL, hL, vL = L
    aR, tR, sR, oR, hR, vR = R
    (j0, o0, h0, v0), (jH, qH), (jV, qV) = J0, JH, JV
    if o0 + h0 + v0 != j0 or not (0 <= qH <= jH and 0 <= qV <= jV):
        raise ValueError("invalid codebook")
    return profile((aL - tL) * (aR - tR) + tL * sR + sL * tR, tL * oR + oL * tR, j0 * oR + jH * hR + jV * vR,
                   o0 * oR + qH * hR + qV * vR, h0 * oR + (jV - qV) * vR, v0 * oR + (jH - qH) * hR)


def phi(p):
    a, t, s, o, h, v = p
    return (a - t, o, v, h, t, t, t)


# --- BPZ rules and terminal codes ------------------------------------------------------------------------------------

LABELS = "BNADOHV"
PERP = {"B": "OHV", "N": "ADOHV", "A": "NDH", "D": "NAV", "O": "BN", "H": "BNA", "V": "BND"}


def _words(text):
    return [tuple(m.split(",")) for m in re.findall(r"\(([A-Z](?:,[A-Z])*)\)", text)]


_S3b = dict(
    B="(B,B,B),(B,H,A),(N,A,V),(N,D,H),(N,V,D),(A,B,H),(A,H,N),(A,H,D),(D,B,V),(D,V,N),(D,V,D),(H,N,D),(H,D,N),"
      "(H,D,D),(V,N,A),(V,A,B),(V,D,A)",
    N="(B,V,D),(N,N,N),(N,V,A),(A,N,H),(A,D,H),(A,H,N),(D,N,V),(D,D,H),(D,H,N),(H,V,D)",
    A="(N,B,A),(N,A,N),(A,B,B),(A,A,V),(A,H,A)", D="(N,B,D),(N,D,N),(D,B,B),(D,A,V),(D,H,A)", O="",
    H="(N,N,H),(N,H,N),(H,N,N),(H,V,V)", V="(N,N,V),(N,V,N),(V,N,N),(V,V,V)")
_RULES = {
    "S2a": dict(B="(B,B),(H,D),(V,A),(D,V),(A,H)", N="(N,N),(A,A),(A,D),(D,A),(D,D)", A="(A,N),(N,D)",
                D="(D,N),(N,A)", O="(O,N),(N,O)", H="(H,N),(N,V)", V="(V,N),(N,H)"),
    "S2b": dict(B="(A,V),(B,B),(D,H)", N="(A,A),(N,N)", A="(A,N),(B,D),(D,V),(V,H)", D="(D,N),(N,A)", O="",
                H="(H,B),(N,V)", V="(H,A),(N,H),(V,D),(V,N)"),
    "S3a": dict(
        B="(A,H,N),(A,N,V),(B,A,H),(B,B,B),(B,D,V),(B,H,D),(B,V,A),(D,N,H),(D,V,N),(H,D,N),(H,N,A),(V,A,N),(V,N,D)",
        N="(A,A,N),(A,D,N),(A,N,A),(A,N,D),(D,A,N),(D,D,N),(D,N,A),(D,N,D),(N,A,A),(N,A,D),(N,D,A),(N,D,D),(N,N,N)",
        A="(A,A,A),(A,A,D),(A,D,A),(A,D,D),(A,N,N),(N,A,N),(N,N,A)",
        D="(D,A,A),(D,A,D),(D,D,A),(D,D,D),(D,N,N),(N,D,N),(N,N,D)", O="(O,A,A),(O,A,D),(O,D,A),(O,D,D)",
        H="(H,A,A),(H,A,D),(H,D,A),(H,D,D),(H,N,N),(N,H,N),(N,N,H)",
        V="(N,N,V),(N,V,N),(V,A,A),(V,A,D),(V,D,A),(V,D,D),(V,N,N)"),
    "S3b": _S3b,
    "S3c": dict(
        B="(A,B,V),(A,H,H),(A,V,A),(B,B,B),(D,A,H),(D,H,A),(D,N,H),(H,A,A),(H,A,N),(H,H,H),(N,H,D),(N,V,A),(V,A,D),"
          "(V,D,B)",
        N="(A,A,D),(A,D,A),(B,H,D),(D,A,B),(D,B,H),(H,H,A),(H,V,H),(N,N,N),(V,A,A),(V,D,H)",
        A="(A,B,N),(A,N,D),(B,N,A),(B,V,V),(H,D,N),(N,A,B),(N,D,A),(V,N,V),(V,V,N)", D="(D,N,N),(N,D,N),(N,N,D)",
        O="", H="(B,H,N),(H,N,B),(N,B,H)",
        V="(A,V,N),(D,H,N),(H,N,A),(N,A,H),(N,D,V),(N,N,V),(N,V,N),(V,N,D),(V,N,N)"),
    "S3d": dict(
        B="(A,B,V),(B,B,B),(B,H,D),(B,V,A),(D,A,H),(D,D,V),(D,N,H),(H,A,A),(H,A,N),(V,A,D),(V,D,N),(V,H,A),(V,H,H)",
        N="(A,A,D),(B,H,D),(D,A,B),(D,B,H),(H,V,H),(N,N,N),(V,A,A),(V,D,H),(V,H,A)",
        A="(A,B,N),(A,N,D),(B,D,A),(B,N,A),(B,V,V),(H,D,N),(N,A,B),(V,N,V),(V,V,N)", D="(D,N,N),(N,D,N),(N,N,D)",
        O="", H="(B,H,N),(H,N,B),(N,B,H)",
        V="(A,V,N),(D,H,N),(H,N,A),(N,A,H),(N,D,V),(N,N,V),(N,V,N),(V,N,D),(V,N,N)"),
    "S3e": dict(
        B="(A,H,N),(B,A,H),(B,B,B),(B,D,V),(B,H,D),(B,V,A),(D,V,N),(H,D,N),(H,N,A),(V,A,N),(V,N,D)",
        N="(A,A,N),(A,H,N),(D,A,N),(D,D,N),(H,N,A),(N,A,A),(N,A,D),(N,D,A),(N,D,D),(N,N,N)",
        A="(A,A,A),(A,A,D),(A,D,A),(A,D,D),(A,N,N),(B,N,A),(H,N,D),(N,A,N)",
        D="(A,N,D),(D,A,A),(D,A,D),(D,D,A),(D,D,D),(D,N,N),(N,D,N),(N,N,D)", O="",
        H="(B,N,H),(H,A,A),(H,A,D),(H,D,D),(H,H,A),(H,N,N),(N,H,N)",
        V="(A,N,V),(D,N,H),(N,N,V),(N,V,N),(V,A,A),(V,A,D),(V,D,A),(V,D,D),(V,N,N)"),
    "S3f": dict(
        B="(A,H,D),(A,H,N),(B,A,H),(B,B,B),(B,D,V),(B,V,A),(H,D,B),(H,H,H),(H,N,A),(N,H,D),(V,N,D)",
        N="(A,H,B),(B,H,H),(D,H,A),(H,D,D),(H,N,A),(N,A,A),(N,A,H),(N,H,A),(N,N,N),(V,H,V)",
        A="(A,A,A),(A,A,H),(A,B,N),(B,N,A),(H,H,N),(H,N,H),(N,A,N)",
        D="(A,N,D),(D,A,A),(D,A,D),(D,B,N),(N,D,N),(N,N,D)", O="", H="(B,N,H),(H,A,B),(H,A,V),(H,N,N),(N,H,N)",
        V="(A,N,V),(D,N,H),(D,V,N),(N,N,V),(N,V,N),(V,A,B),(V,A,V),(V,N,N)"),
    "S3g": dict(
        B="(A,H,N),(B,A,H),(B,B,B),(B,D,V),(B,H,D),(B,V,A),(H,D,N),(H,N,A),(V,H,A),(V,H,H),(V,N,D)",
        N="(A,H,B),(B,H,H),(D,H,A),(H,D,D),(H,N,A),(N,A,A),(N,A,H),(N,H,A),(N,N,N),(V,H,V)",
        A="(A,A,A),(A,A,H),(A,B,N),(B,N,A),(H,H,N),(H,N,H),(N,A,N)",
        D="(A,N,D),(D,A,A),(D,A,H),(D,B,N),(N,D,N),(N,N,D)", O="", H="(B,N,H),(H,A,B),(H,A,V),(H,N,N),(N,H,N)",
        V="(A,N,V),(D,N,H),(D,V,N),(N,N,V),(N,V,N),(V,A,B),(V,A,V),(V,N,N)"),
    "S3h": dict(B=_S3b["B"], N=_S3b["N"], A=_S3b["D"], D=_S3b["A"], O="", H=_S3b["V"], V=_S3b["H"]),
}
RULES = {name: {lam: _words(text) for lam, text in rule.items()} for name, rule in _RULES.items()}
CODES = {
    "K3a": _words("(A,B,V),(A,V,H),(A,H,N),(B,A,B),(B,H,A),(B,D,D),(B,N,B),(D,D,N),(D,A,H),(D,N,H),(H,H,H),(H,B,A),"
                  "(H,V,N),(N,H,N),(N,B,V),(N,V,H),(V,B,N),(V,V,A),(V,B,D)"),
    "K4a": _words("(B,B,B,B),(B,N,V,B),(B,A,H,B),(B,D,V,B),(B,H,N,B),(B,H,D,B),(B,V,A,B),(N,B,B,V),(N,N,V,H),"
                  "(N,A,V,H),(N,D,H,H),(N,H,N,H),(N,H,A,H),(N,V,D,V),(A,B,B,V),(A,N,H,H),(A,A,V,H),(A,D,H,H),"
                  "(A,H,A,H),(A,V,N,H),(A,V,D,H),(D,B,B,H),(D,N,V,V),(D,A,V,V),(D,D,H,V),(D,H,N,V),(D,H,A,V),"
                  "(D,V,D,V),(H,A,H,D),(H,B,B,A),(H,N,V,D),(H,D,V,D),(H,A,V,N),(H,D,H,N),(H,H,A,N),(H,H,N,N),"
                  "(H,H,N,D),(H,H,D,D),(H,V,D,N),(H,V,A,D),(V,N,V,N),(V,B,B,N),(V,B,B,D),(V,D,V,A),(V,N,V,A),"
                  "(V,A,H,A),(V,V,A,A),(V,H,D,A),(V,V,N,A)"),
    "K4b": _words("(A,A,A,V),(A,A,N,V),(A,B,H,A),(A,H,A,A),(A,H,A,N),(A,H,N,A),(A,H,N,N),(A,N,A,V),(A,N,D,H),"
                  "(A,N,H,N),(A,N,N,H),(A,V,D,B),(B,A,D,H),(B,A,V,N),(B,B,B,B),(B,B,V,D),(B,D,B,H),(B,D,H,N),"
                  "(B,H,H,H),(B,V,A,D),(B,V,N,D),(D,A,V,A),(D,H,H,A),(D,N,B,V),(D,N,V,A),(D,N,V,N),(D,V,B,A),"
                  "(D,V,B,N),(H,B,V,H),(H,A,B,H),(H,B,D,B),(H,N,N,D),(H,D,A,B),(H,D,N,B),(N,A,A,V),(H,V,H,D),"
                  "(H,V,H,N),(N,H,A,N),(N,A,N,V),(N,B,H,A),(N,N,D,H),(N,H,B,A),(N,N,A,V),(N,V,D,N),(N,N,N,V),"
                  "(N,N,V,N),(V,A,A,B),(N,V,H,A),(N,V,N,N),(V,H,A,H),(V,A,N,B),(V,D,D,H),(V,N,N,A),(V,H,N,H),"
                  "(V,N,A,B),(V,N,N,N),(V,V,H,V)"),
}


def _separated(w1, w2):
    return any(y in PERP[x] for x, y in zip(w1, w2))


def rule_problems(rule):
    """[] when a rule is admissible (BPZ Definition 6): words of each output family pairwise separated, and words of
    perpendicular output families separated."""
    bad = []
    if set(rule) != set(LABELS) or len({len(w) for ws in rule.values() for w in ws}) != 1:
        return ["schema"]
    for lam, ws in rule.items():
        bad += [(lam, w1, w2) for w1, w2 in combinations(ws, 2) if not _separated(w1, w2)]
    for lam in LABELS:
        for mu in PERP[lam]:
            bad += [(lam, mu, w1, w2) for w1 in rule[lam] for w2 in rule[mu] if not _separated(w1, w2)]
    return bad


def code_problems(code):
    return [(w1, w2) for w1, w2 in combinations(code, 2) if not _separated(w1, w2)]


def _word(word, vectors):
    return reduce(lambda acc, iv: acc * vectors[iv[0]][LABELS.index(iv[1])], enumerate(word), 1)


def apply_rule(rule, vectors):
    if {len(w) for ws in rule.values() for w in ws} != {len(vectors)}:
        raise ValueError("rule arity does not match the number of inputs")
    return tuple(sum(_word(w, vectors) for w in rule[lam]) for lam in LABELS)


def apply_code(code, vectors):
    if {len(w) for w in code} != {len(vectors)}:
        raise ValueError("terminal code arity does not match the number of inputs")
    return sum(_word(w, vectors) for w in code)


if any(rule_problems(r) for r in RULES.values()) or any(code_problems(c) for c in CODES.values()):
    raise ArithmeticError("a BPZ rule or terminal code is not admissible")


# --- block sets over C_7^(box 5) -------------------------------------------------------------------------------------

def _code(w):
    return sum(c * 7 ** j for j, c in enumerate(w))


def _decode(n):
    return tuple((n // 7 ** j) % 7 for j in range(5))


NB = [frozenset(_code([(w + e) % 7 for w, e in zip(_decode(c), d)]) for d in product((-1, 0, 1), repeat=5))
      for c in range(7 ** 5)]
BITS = [sum(1 << x for x in nb) for nb in NB]


def T(n):
    """The automorphism (w0, ..., w4) -> (2 - w1, w3, w0, 2 - w2, w4) of C_7^(box 5)."""
    w = _decode(n)
    return _code(((2 - w[1]) % 7, w[3], w[0], (2 - w[2]) % 7, w[4]))


def _nbhd(S):
    out = set()
    for c in S:
        out |= NB[c]
    return out


_uid = [0]


class Atom:
    """An explicit set of words of C_7^(box 5) (width 1, codes) or C_7^(box 10) (width 2, pairs of codes)."""

    def __init__(self, items, width, name=""):
        _uid[0] += 1
        self.uid, self.width, self.name, self.items = _uid[0], width, name, frozenset(items)
        self._test = None

    def __len__(self):
        return len(self.items)

    def in_nbhd(self, w):
        if self._test is None:
            if self.width == 1:
                self._test = _nbhd(self.items)
            else:
                by_first = {}
                for x, y in self.items:
                    by_first[x] = by_first.get(x, 0) | 1 << y
                dil = {}
                for c in range(7 ** 5):
                    m = 0
                    for c2 in NB[c]:
                        m |= by_first.get(c2, 0)
                    if m:
                        dil[c] = m
                self._test = dil
        if self.width == 1:
            return w in self._test
        m = self._test.get(w[0])
        return m is not None and BITS[w[1]] & m != 0

    def image(self):
        if self.width == 1:
            return Atom({T(c) for c in self.items}, 1, "T" + self.name)
        return Atom({(T(x), T(y)) for x, y in self.items}, 2, "T" + self.name)


def size(blocks):
    return sum(reduce(lambda n, atom: n * len(atom), blk, 1) for blk in blocks)


def cross(left, right):
    return [l + r for l in left for r in right]


def image_blocks(blocks):
    """T applied to every five-dimensional coordinate."""
    cache, out = {}, []
    for blk in blocks:
        for atom in blk:
            if atom.uid not in cache:
                cache[atom.uid] = atom.image()
        out.append(tuple(cache[atom.uid] for atom in blk))
    return out


def count_outside(source, reference):
    """|union(source) minus N(union(reference))|, exactly. Source blocks must be disjoint in some coordinate (checked);
    a word lies in N(reference) iff some reference block contains its neighbourhood test in every coordinate, so
    counting the flag vectors of each coordinate and combining them is exact."""
    if len({tuple(a.width for a in b) for b in source + reference}) != 1:
        raise ValueError("inconsistent block shapes")
    disjoint = {}
    for i, b1 in enumerate(source):
        for b2 in source[i + 1:]:
            ok = False
            for x, y in zip(b1, b2):
                if x.uid == y.uid:
                    continue
                key = (min(x.uid, y.uid), max(x.uid, y.uid))
                if key not in disjoint:
                    disjoint[key] = not (x.items & y.items)
                if disjoint[key]:
                    ok = True
                    break
            if not ok:
                raise ArithmeticError("source blocks are not provably disjoint")
    memo, total, full = {}, 0, (True,) * len(reference)
    for blk in source:
        acc = Counter({full: 1})
        for i, atom in enumerate(blk):
            refs = tuple(r[i] for r in reference)
            key = (atom.uid, tuple(r.uid for r in refs))
            if key not in memo:
                memo[key] = Counter(tuple(r.in_nbhd(e) for r in refs) for e in atom.items)
            nxt = Counter()
            for f1, n1 in acc.items():
                for f2, n2 in memo[key].items():
                    nxt[tuple(a and b for a, b in zip(f1, f2))] += n1 * n2
            acc = nxt
        total += sum(n for flags, n in acc.items() if not any(flags))
    return total


@lru_cache(maxsize=None)
def blocks():
    """The physical sets of Tandon's construction as block sets: base atoms, ten-dimensional atoms, J^+ and the
    fifteen-dimensional block sets. Checks the base gadget axioms."""
    I, X, PH, PV = set(_base.I0), set(_base.X), set(_base.PH), set(_base.PV)
    pairs = [tuple(p) for p in _base.PAIRS]
    indep = lambda S: all(not (NB[u] & (S - {u})) for u in S)  # noqa: E731
    if not (indep(I) and indep(X) and indep(PH) and indep(PV) and len(PH) == len(PV) == len(pairs)
            and all(r in I and q not in I and {w for w in I if w in NB[q]} == {r} for r, q in pairs)
            and len({e for p in pairs for e in p}) == 2 * len(pairs) and not PH & PV
            and all((r in PH and q in PV) or (q in PH and r in PV) for r, q in pairs)
            and not X & _nbhd(PH) & _nbhd(PV)
            and len({T(c) for c in range(7 ** 5)}) == 7 ** 5
            and all({T(x) for x in NB[c]} == NB[T(c)] for c in range(7 ** 5))):
        raise ArithmeticError("base gadget axiom fails")
    R = {r for r, _ in pairs}
    NH, NV = _nbhd(PH), _nbhd(PV)
    g = dict(I=Atom(I, 1, "I"), B=Atom(I - R, 1, "B"), R=Atom(R, 1, "R"), PH=Atom(PH, 1, "PH"),
             PV=Atom(PV, 1, "PV"), X=Atom(X, 1, "X"), X0=Atom(X - NH - NV, 1, "X0"), XH=Atom(X & NH, 1, "XH"),
             XV=Atom(X & NV, 1, "XV"), Xr=Atom(X & (NH | NV), 1, "Xr"))
    P = lambda A, B: {(a, b) for a in A.items for b in B.items}  # noqa: E731
    A2 = lambda items, name: Atom(items, 2, name)  # noqa: E731
    a = dict(R2=A2(P(g["R"], g["X0"]) | P(g["X0"], g["R"]), "R2"),
             PH2=A2(P(g["PH"], g["X0"]) | P(g["X0"], g["PV"]), "PH2"),
             PV2=A2(P(g["PV"], g["X0"]) | P(g["X0"], g["PH"]), "PV2"),
             X2=A2(P(g["X"], g["X"]), "X2"),
             X2_0=A2(P(g["X0"], g["X0"]) | P(g["Xr"], g["Xr"]), "X2_0"),
             X2_H=A2(P(g["XH"], g["X0"]) | P(g["X0"], g["XV"]), "X2_H"),
             X2_V=A2(P(g["XV"], g["X0"]) | P(g["X0"], g["XH"]), "X2_V"))
    a["X2_r"] = A2(a["X2"].items - a["X2_0"].items, "X2_r")
    I2 = (P(g["B"], g["B"]) | a["R2"].items | P(g["PH"], g["XH"]) | P(g["PV"], g["XV"]) | P(g["XH"], g["PV"])
          | P(g["XV"], g["PH"]))
    a["B2"] = A2(I2 - a["R2"].items, "B2")
    PA_H = A2(P(g["PV"], g["X0"]) | P(g["X0"], g["PV"]), "PA_H")
    PA_V = A2(P(g["PH"], g["X0"]) | P(g["X0"], g["PH"]), "PA_V")
    a["X2A_0"] = A2({p for p in a["X2"].items if not PA_H.in_nbhd(p) and not PA_V.in_nbhd(p)}, "X2A_0")
    a["X2A_H"] = A2({p for p in a["X2"].items if PA_H.in_nbhd(p)}, "X2A_H")
    a["X2A_V"] = A2({p for p in a["X2"].items if PA_V.in_nbhd(p)}, "X2A_V")
    if a["X2A_0"].items != a["X2_0"].items:
        raise ArithmeticError("X_10^0 depends on the orientation")
    # J^+ = (T x T)(I_10) with the eight exchanges, each replacing its only J-neighbour, from inside N(X_10^0) to
    # outside, with the inserted words pairwise non-confusable.
    raw = {(T(x), T(y)) for x, y in I2}
    removed, inserted = {x for x, _ in _base.EXCHANGES}, {y for _, y in _base.EXCHANGES}
    index = {}
    for x, y in raw:
        index.setdefault(x, set()).add(y)
    if not removed <= raw or inserted & raw:
        raise ArithmeticError("bad exchange")
    for rem, ins in _base.EXCHANGES:
        if {(x, y) for x in NB[ins[0]] for y in index.get(x, ()) if y in NB[ins[1]]} != {rem} \
                or not a["X2_0"].in_nbhd(rem) or a["X2_0"].in_nbhd(ins):
            raise ArithmeticError("bad exchange")
    if any(p[0] in NB[q[0]] and p[1] in NB[q[1]] for p, q in combinations(sorted(inserted), 2)):
        raise ArithmeticError("inserted words are confusable")
    a["Jp"] = A2((raw - removed) | inserted, "Jp")
    a["Jq"] = A2({p for p in a["Jp"].items if not a["X2_0"].in_nbhd(p)}, "Jq")
    a["JpN"] = A2({p for p in a["Jp"].items if a["X2_0"].in_nbhd(p)}, "JpN")
    a["X2A_r"] = A2(a["X2A_H"].items | a["X2A_V"].items, "X2A_r")
    G = dict(
        B15=[(a["B2"], g["B"]), (a["PH2"], g["XH"]), (a["PV2"], g["XV"]), (a["X2_H"], g["PV"]), (a["X2_V"], g["PH"])],
        R15=[(a["R2"], g["X0"]), (a["X2_0"], g["R"])], PH15=[(a["PH2"], g["X0"]), (a["X2_0"], g["PV"])],
        PV15=[(a["PV2"], g["X0"]), (a["X2_0"], g["PH"])],
        X15_0=[(a["X2_0"], g["X0"]), (a["Jq"], g["XH"]), (a["Jq"], g["XV"])],
        X15_H=[(a["X2_H"], g["X0"]), (a["JpN"], g["XV"])], X15_V=[(a["X2_V"], g["X0"]), (a["JpN"], g["XH"])],
        X15X_0=[(a["X2_0"], g["X0"]), (a["X2_r"], g["Xr"])], X15X_r=[(a["X2_r"], g["X0"]), (a["X2_0"], g["Xr"])],
        X15A_0=[(a["X2A_0"], g["X0"]), (a["Jq"], g["XH"]), (a["X2_r"], g["XV"])],
        X15A_r=[(a["X2A_H"], g["X0"]), (a["X2_0"], g["XV"]), (a["X2A_V"], g["X0"]), (a["JpN"], g["XH"])])
    G["X15het"] = [(a["X2"], g["X0"]), (a["Jp"], g["XH"]), (a["Jp"], g["XV"])]
    one = lambda *atoms: [(x,) for x in atoms]  # noqa: E731
    # main-code split (B = I minus R, R, PH, PV) and auxiliary classes of the plain gadgets of each dimension
    main = {5: dict(B=one(g["B"]), R=one(g["R"]), PH=one(g["PH"]), PV=one(g["PV"]), X0=one(g["X0"]),
                    XH=one(g["XH"]), XV=one(g["XV"])),
            10: dict(B=one(a["B2"]), R=one(a["R2"]), PH=one(a["PH2"]), PV=one(a["PV2"]), X0=one(a["X2_0"]),
                     XH=one(a["X2_H"]), XV=one(a["X2_V"])),
            15: dict(B=G["B15"], R=G["R15"], PH=G["PH15"], PV=G["PV15"], X0=G["X15_0"], XH=G["X15_H"],
                     XV=G["X15_V"])}
    # physical (X^0, X^H u X^V) of each library gadget kind
    aux = {"5": (one(g["X0"]), one(g["XH"], g["XV"])), "10": (one(a["X2_0"]), one(a["X2_r"])),
           "X": (G["X15X_0"], G["X15X_r"]), "h": (G["X15_0"], G["X15_H"] + G["X15_V"]),
           "A": (G["X15A_0"], G["X15A_r"])}
    return dict(g=g, a=a, G=G, main=main, aux=aux)


def gao_sets(L, R):
    """Main-code split and auxiliary classes of the Gao product of two gadgets given by their block sets."""
    c = cross
    return dict(B=c(L["B"], R["B"]) + c(L["PH"], R["XH"]) + c(L["PV"], R["XV"]) + c(L["XH"], R["PV"])
                + c(L["XV"], R["PH"]),
                R=c(L["R"], R["X0"]) + c(L["X0"], R["R"]),
                PH=c(L["PH"], R["X0"]) + c(L["X0"], R["PV"]), PV=c(L["PV"], R["X0"]) + c(L["X0"], R["PH"]),
                X0=c(L["X0"], R["X0"]) + c(L["XH"] + L["XV"], R["XH"] + R["XV"]),
                XH=c(L["XH"], R["X0"]) + c(L["X0"], R["XV"]), XV=c(L["XV"], R["X0"]) + c(L["X0"], R["XH"]))


def derive_counts():
    """Tandon's five counts C1, C2, C3, C4, q15 recomputed from the base data (about a minute)."""
    b = blocks()
    a, g, G = b["a"], b["g"], b["G"]
    I15 = [(a["B2"], g["B"]), (a["R2"], g["X0"]), (a["PH2"], g["XH"]), (a["PV2"], g["XV"]), (a["X2_0"], g["R"]),
           (a["X2_H"], g["PV"]), (a["X2_V"], g["PH"])]
    J15 = image_blocks(I15)
    I30 = gao_sets(b["main"][15], b["main"][15])
    ref = cross(G["X15X_0"], G["X15A_0"]) + cross(G["X15X_r"], G["X15A_r"])
    return {"C1": len(a["Jq"]), "C2": count_outside(J15, G["X15X_0"]), "C3": count_outside(G["X15het"], G["X15X_0"]),
            "C4": count_outside(image_blocks(I30["B"] + I30["R"]), ref), "q15": count_outside(J15, G["X15_0"])}


# --- Tandon's gadgets and left patterns ------------------------------------------------------------------------------

G5 = profile(367, 8, 367, 322, 26, 19)


@lru_cache(maxsize=None)
def _tandon():
    c = COUNTS
    sG5 = sigma(G5)
    G10, G10A, G10D = gao(G5, G5), gao(sG5, G5), gao(G5, sG5)
    Jplus, X10 = (G10[0], c["C1"]), own_auxiliary(G10)
    t = dict(G5=G5, sG5=sG5, G10=G10, G10A=G10A, G10D=G10D)
    t["G15X"] = gao(G10, G5)
    t["G15h"] = hetgao(G10, G5, neutral_from(G10), Jplus, Jplus)
    t["G15AX"] = gao(G10A, G5)
    t["G15Ah"] = hetgao(G10A, G5, neutral_from(G10A), Jplus, X10)
    t["G15DX"] = gao(G10D, sG5)
    t["G15Dh"] = hetgao(G10D, sG5, neutral_from(G10D), X10, Jplus)
    J15, X15, X15X = (t["G15X"][0], c["C2"]), (t["G15h"][2], c["C3"]), own_auxiliary(t["G15X"])
    t["Gpp30"] = gao(t["G15h"], t["G15h"])
    J30pp, J15w = (t["Gpp30"][0], c["C4"]), (t["G15h"][0], c["q15"])
    t["G30L"] = gao(t["G15AX"], sigma(t["G15Ah"]))
    pats = {
        "P10": (G10, neutral_from(G10), Jplus, Jplus, 10),
        "P10A": (G10A, neutral_from(G10A), Jplus, X10, 10),
        "P10D": (G10D, neutral_from(G10D), X10, Jplus, 10),
        "P15X": (t["G15X"], neutral_from(t["G15h"]), J15, J15, 15),
        "P15A8": (t["G15AX"], neutral_from(t["G15Ah"]), J15, J15, 15),
        "P15Ah": (t["G15AX"], neutral_from(t["G15Ah"]), J15, X15X, 15),
        "P15A5": (t["G15AX"], neutral_from(t["G15Ah"]), J15, X15, 15),
        "P15D": (t["G15DX"], neutral_from(t["G15Dh"]), X15, J15, 15),
        "P15w": (t["G15h"], neutral_from(t["G15h"]), J15w, J15w, 15),
    }
    het = lambda p, R: hetgao(pats[p][0], R, *pats[p][1:4])  # noqa: E731
    t["G30_6"] = het("P15X", t["G15h"])
    t["G25_8"] = het("P15A8", G10D)
    t["G40_8"] = het("P15D", t["G25_8"])
    t["Ghat30"] = het("P15Ah", sigma(t["G15Ah"]))
    t["Ghat25"] = het("P15A5", G10D)
    t["G25R"] = sigma(t["Ghat25"])
    pats["P30L"] = (t["G30L"], neutral_from(t["Ghat30"]), J30pp, J30pp, 30)
    t["G55"] = het("P30L", t["G25R"])
    t["G30h"] = het("P15w", t["G15h"])
    vec = dict(n6=phi(t["G30_6"]), n8=phi(t["G40_8"]), n11=phi(sigma(t["G55"])))
    M = apply_code(CODES["K4a"], [apply_rule(RULES["S3b"], [vec["n6"], vec["n11"], vec["n8"]])] * 4)
    dims = dict(G5=5, sG5=5, G10=10, G10A=10, G10D=10, G15X=15, G15h=15, G15AX=15, G15Ah=15, G15DX=15, G15Dh=15,
                Gpp30=30, G30L=30, G30_6=30, G25_8=25, G40_8=40, Ghat30=30, Ghat25=25, G25R=25, G55=55, G30h=30)
    return t, pats, vec, M, dims


def library():
    """{name: (profile, dimension)} of the gadgets a certificate may use as operands."""
    t, _, _, _, dims = _tandon()
    return {k: (t[k], dims[k]) for k in dims}


def tandon():
    """(M, d) of Tandon's Theorem 2: 3.2588326203532663..., d = 500."""
    return _tandon()[3], 500


# physical kind of each gadget usable inside S and N patterns, and the left factor of each Tandon pattern
_KIND = {"G5": "5", "G10": "10", "G10A": "10", "G10D": "10", "G15X": "X", "G15AX": "X", "G15DX": "X", "G15h": "h",
         "G15Ah": "A", "G15Dh": "A"}
_KIND_DIM = {"5": 5, "10": 10, "X": 15, "h": 15, "A": 15}
_LEFT = {"P10": "G10", "P10A": "G10A", "P10D": "G10D", "P15X": "G15X", "P15A8": "G15AX", "P15Ah": "G15AX",
         "P15A5": "G15AX", "P15D": "G15DX", "P15w": "G15h"}
GADGETS = sorted(_KIND) + ["s" + k for k in sorted(_KIND)]


def _gadget_profile(name):
    """(profile, dimension) of a gadget usable inside S and N patterns, sigma for an 's' prefix."""
    flip = name.startswith("s") and name[1:] in _KIND
    key = name[1:] if flip else name
    if key not in _KIND:
        raise ValueError(f"unknown gadget {name} (allowed: {GADGETS})")
    p = library()[key][0]
    return (sigma(p) if flip else p), _KIND_DIM[_KIND[key]]


def _gadget(name):
    """(profile, X^0 blocks, X^H u X^V blocks, dimension) of a gadget usable inside S and N patterns."""
    p, d = _gadget_profile(name)
    key = name[1:] if name.startswith("s") and name[1:] in _KIND else name
    X0, Xr = blocks()["aux"][_KIND[key]]
    if size(X0) != p[3] or size(Xr) != p[4] + p[5]:
        raise ArithmeticError(f"block sets do not match the profile of {name}")
    return p, X0, Xr, _KIND_DIM[_KIND[key]]


_PLAIN = {5: "G5", 10: "G10", 15: "G15h"}
_COUNTED = {}


def _parse(name):
    m = re.fullmatch(r"S<(\w+),(\w+),(\w+)>", name)
    if m:
        return ("S",) + m.groups()
    m = re.fullmatch(r"N<(.+),(\w+)>", name)
    if m and m.group(1).startswith(("S<", "N<")):
        return ("N",) + m.groups()
    raise ValueError(f"unknown pattern {name}")


@lru_cache(maxsize=None)
def _shape(name):
    """(L, J0, j, d, main-code profile) of an S or N pattern, from profiles alone."""
    kind, x, y, *rest = _parse(name)
    lib = library()
    if kind == "S":
        A_name, B_name, P_name = x, y, rest[0]
        if _LEFT.get(P_name) != A_name:
            raise ValueError(f"pattern {P_name} does not have left factor {A_name}")
        A, dA = _gadget_profile(A_name)
        B, dB = _gadget_profile(B_name)
        PL, J0, JH, JV, _ = _tandon()[1][P_name]
        PM = gao(lib[_PLAIN[dA]][0], lib[_PLAIN[dB]][0])
        L, Lhet, d = gao(A, B), hetgao(PL, B, J0, JH, JV), dA + dB
    else:
        L1, J01, d1 = pattern(x)[0], pattern(x)[1:4], pattern(x)[4]
        C, dC = _gadget_profile(y)
        PM = gao(_shape(x)[4], lib[_PLAIN[dC]][0])
        L, Lhet, d = gao(L1, C), hetgao(L1, C, *J01), d1 + dC
    if Lhet[:2] != L[:2]:
        raise ArithmeticError("HetGao and Gao must share code and private pairs")
    return L, neutral_from(Lhet), PM[0], d, PM


@lru_cache(maxsize=None)
def _sets(name):
    """(main-code sets, X^0 blocks, X^H u X^V blocks) of the left gadget of an S or N pattern."""
    kind, x, y, *rest = _parse(name)
    main = blocks()["main"]
    if kind == "S":
        _, XA0, XAr, dA = _gadget(x)
        _, XB0, XBr, dB = _gadget(y)
        return gao_sets(main[dA], main[dB]), cross(XA0, XB0) + cross(XAr, XBr), cross(XAr, XB0) + cross(XA0, XBr)
    sets1, X01, Xr1 = _sets(x)
    _, XC0, XCr, dC = _gadget(y)
    return gao_sets(sets1, main[dC]), cross(X01, XC0) + cross(Xr1, XCr), cross(Xr1, XC0) + cross(X01, XCr)


def count(name) -> int:
    """q = |T(I) minus N(X^0(L))| of an S or N pattern, counted exactly from the base data (1 to 60 s)."""
    if name == "Ppp":
        t, G, main = _tandon()[0], blocks()["G"], blocks()["main"]
        HV = G["X15_H"] + G["X15_V"]
        ref = cross(G["X15_0"], G["X15_0"]) + cross(HV, HV)
        I = gao_sets(main[15], main[15])
        I = I["B"] + I["R"]
        if size(ref) != t["Gpp30"][3] or size(I) != t["Gpp30"][0]:
            raise ArithmeticError("block sets do not match the profile of G30^{++}")
        return count_outside(image_blocks(I), ref)
    L, _, j, _, _ = _shape(name)
    sets, X0, Xr = _sets(name)
    I = sets["B"] + sets["R"]
    if size(X0) != L[3] or size(Xr) != L[4] + L[5] or size(I) != j:
        raise ArithmeticError(f"block sets do not match the profiles of {name}")
    return count_outside(image_blocks(I), X0)


def counted() -> dict:
    """{pattern: q} of the patterns whose counts are known without counting: the table shipped with the task (every
    entry recomputed by count) and those counted in this process."""
    if not _COUNTED:
        _COUNTED.update(_counts.Q)
    return _COUNTED


def pattern(name):
    """(L, J0, JH, JV, dimension of L) of a left pattern; a new count is computed live and remembered."""
    pats = _tandon()[1]
    if name in pats:
        return pats[name]
    if name not in counted():
        _COUNTED[name] = count(name)
    if name == "Ppp":
        t = _tandon()[0]
        J = (t["Gpp30"][0], _COUNTED[name])
        return t["Gpp30"], neutral_from(t["G30h"]), J, J, 30
    L, J0, j, d, _ = _shape(name)
    J = (j, _COUNTED[name])
    return L, J0, J, J, d


def patterns() -> list:
    """Names of all patterns usable without a live count."""
    return sorted(_tandon()[1]) + sorted(counted())

# --- certificates ----------------------------------------------------------------------------------------------------

def value(certificate) -> tuple:
    """(M, d): an independent set of size M in C_7^(box d), exactly, with the checks of the task."""
    if not isinstance(certificate, dict):
        raise ValueError("the certificate must be a dict")
    steps, vsteps, top = certificate.get("gadgets", []), certificate.get("vectors", []), certificate["top"]
    if len(steps) + len(vsteps) > MAX_STEPS:
        raise ValueError(f"more than {MAX_STEPS} steps")
    prof = library()
    for st in steps:
        name, op = str(st["name"]), st["op"]
        if name in prof or name.startswith("Phi("):
            raise ValueError(f"duplicate name {name}")
        if op == "hetgao":
            L, J0, JH, JV, dL = pattern(str(st["pattern"]))
            R, dR = prof[st["R"]]
            prof[name] = (hetgao(L, R, J0, JH, JV), dL + dR)
        elif op == "gao":
            (L, dL), (R, dR) = prof[st["L"]], prof[st["R"]]
            prof[name] = (gao(L, R), dL + dR)
        elif op == "sigma":
            p, d = prof[st["of"]]
            prof[name] = (sigma(p), d)
        else:
            raise ValueError(f"unknown op {op}")
        if prof[name][1] > MAX_DIMENSION:
            raise ValueError("dimension too large")
    vec = {k: (v, {"n6": 30, "n8": 40, "n11": 55}[k]) for k, v in _tandon()[2].items()}

    def vector(key):
        if key in vec:
            return vec[key]
        if key.startswith("Phi(") and key.endswith(")") and key[4:-1] in prof:
            p, d = prof[key[4:-1]]
            return phi(p), d
        raise ValueError(f"unknown vector {key}")

    for st in vsteps:
        name = str(st["name"])
        if name in vec or name.startswith("Phi("):
            raise ValueError(f"duplicate name {name}")
        ins = [vector(x) for x in st["inputs"]]
        vec[name] = (apply_rule(RULES[st["rule"]], [v for v, _ in ins]), sum(d for _, d in ins))
        if vec[name][1] > MAX_DIMENSION:
            raise ValueError("dimension too large")
    code = CODES[top["terminal_code"]]
    k = len(code[0])
    if "inputs" in top:
        ins = [vector(x) for x in top["inputs"]]
    else:
        subs = top["subtrees"] if "subtrees" in top else [top["rule_inputs"]] * k
        ins = []
        for sub in subs:
            parts = [vector(x) for x in sub]
            ins.append((apply_rule(RULES[top["rule"]], [v for v, _ in parts]), sum(d for _, d in parts)))
    if len(ins) != k:
        raise ValueError("number of inputs != terminal code arity")
    d = sum(dd for _, dd in ins)
    if d > MAX_DIMENSION:
        raise ValueError("dimension too large")
    return apply_code(code, [v for v, _ in ins]), d


def rate(M, d) -> float:
    """M^(1/d) as a float."""
    getcontext().prec = 50
    return float((Decimal(M).ln() / d).exp())


def ratio(M, d, M0, d0) -> Fraction:
    """M^(1/d) / M0^(1/d0) to about 40 digits, exactly 1 when the two bounds are equal."""
    if M ** d0 == M0 ** d:
        return Fraction(1)
    getcontext().prec = 60
    return Fraction((Decimal(M).ln() / d - Decimal(M0).ln() / d0).exp())


def estimate(certificate) -> float:
    """value(certificate) as a rate M^(1/d); fast once its patterns are counted."""
    return rate(*value(certificate))


def load(output) -> dict:
    """The certificate of a population output."""
    return output["certificate"]
