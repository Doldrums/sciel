"""The best known construction for this task: R. Pinho's four-strand superhelix with unequal radii (teorth/
optimizationproblems PR 324, open, 2026; rpinho/math-certificates commit e8460e5, c22b/round1/CONSTRUCTION.json),
claimed C_22b <= 7.090412 after a thickness margin (scale 10001/10000). Exact parameters: N = 276/5 minor periods
per axis turn, axis radius R = 293279789/10^8, outer radius a = 193279689/10^8, inner radius i = 3846709/(5 10^7),
pitch b = 16209211/(5 10^7) with axis rise h = (N - 1) b per radian. The radii are constants here."""

from fractions import Fraction


def candidate() -> dict:
    N, b = Fraction(276, 5), Fraction(16209211, 50000000)
    return {"P": 276, "Q": 5, "R": float(Fraction(293279789, 10 ** 8)), "h": float((N - 1) * b),
            "outer": [float(Fraction(193279689, 10 ** 8))], "inner": [float(Fraction(3846709, 5 * 10 ** 7))]}
