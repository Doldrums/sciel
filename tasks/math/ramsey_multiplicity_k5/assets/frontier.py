"""The best known construction for this task: R. Pinho's 256-part binary Cayley step graphon (round 4 of
github.com/rpinho/math-certificates, commit 1d38853, teorth/optimizationproblems PR 307, 3 November 2026). Parts are
indexed by z in F_2^8 and the red probability between parts x and y is q(x XOR y): 1 on the 120 set bits of HARD,
2648/4041 on the 16 set bits of FRACTIONAL, 0 elsewhere. Its monochromatic K5 density is exactly DENSITY."""

from fractions import Fraction

HARD = 0x173e5ee876e8e880167c7ae06ec8a801173e5ee876e8e880e88181178117177e
FRACTIONAL = 0x0040200008000001010204081020408000402000080000010000000000000000
P = Fraction(2648, 4041)
DENSITY = Fraction(15058298592194934908767847, 9131939205947984819238469632)


def kernel() -> tuple:
    """(numerators, denominator) of q on F_2^8, numerators[z] / denominator the red probability at z."""
    d = P.denominator
    return [d if HARD >> z & 1 else P.numerator if FRACTIONAL >> z & 1 else 0 for z in range(256)], d
