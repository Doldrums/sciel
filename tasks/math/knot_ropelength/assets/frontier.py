"""The best known construction for this task: R. Pinho's phase-modulated superhelix (teorth/optimizationproblems PR 347,
open, 8 Nov 2026; rpinho/math-certificates commit e4e7b4b, c22b/round4/candidate_high/CONSTRUCTION.json), stacked on
SCiEL's even radius modulation (PR 342) and his PR 324 caps; claimed C_22b <= 7.087559024. N = 331/6, six cosine
orders in each radius, four sine orders in the phase v(u) = u + sum p_k sin k u; h = (N - 1) b for his reference
pitch b. His 1000001/1000000 scale is dropped: the bound is scale free."""

from fractions import Fraction as F

N, PITCH = F(331, 6), F("40523356633/125000000000")
OUTER = ("1931566317919/1000000000000", "856026859/1000000000000", "276591/200000000000", "-8283669/1000000000000",
         "-791183/125000000000", "286361/200000000000", "-12009/1000000000000")
INNER = ("39141104299/500000000000", "-229157223/500000000000", "-28104853/1000000000000",
         "-38357531/1000000000000", "-27421237/1000000000000", "-3084633/1000000000000",
         "203917/1000000000000")
PHASE = ("-48416661/40000000000", "561411/15625000000", "71834467/1000000000000", "122280043/500000000000")


def candidate() -> dict:
    def radius(c):
        return [float(c[0])] + [x for k in c[1:] for x in (float(k), 0.0)]

    return {"P": 331, "Q": 6, "R": float(F("586482106879/200000000000")), "h": float((N - 1) * PITCH),
            "outer": radius([F(x) for x in OUTER]), "inner": radius([F(x) for x in INNER]),
            "phase": [float(F(x)) for x in PHASE]}
