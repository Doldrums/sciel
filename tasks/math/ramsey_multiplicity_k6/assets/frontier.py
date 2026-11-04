"""The best known construction for this task: Jeetu SK's weight-class refinement of Franek's Cayley graph on F_2^10
(github.com/JeetuSK0808/tao-constants-certificates, commit 98b6ee3, teorth/optimizationproblems PR 279, 26 October
2026). The red probability between parts x and y is NUMERATORS[|x XOR y|] / 2^20. Its monochromatic K6 density is
exactly DENSITY = 0.742254861593625... * 2^-14. FRANEK is the 0/1 kernel it started from, whose density is
51162598917 / 2^50 (Combinatorica 22, 2002)."""

from fractions import Fraction

NUMERATORS = [1048576, 461327, 0, 1048576, 1048576, 13579, 0, 1048576, 1048576, 0, 799568]
DENOMINATOR = 2 ** 20
FRANEK = [1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1]
DENSITY = Fraction(
    180244428715156181493112659521741405180580225001966676111887908798425648907356856409,
    3978585891278293137243057985174566720803649206378781739523711815145275976100267004264448)


def kernel() -> tuple:
    """(numerators, denominator) of q on the weights 0..10."""
    return list(NUMERATORS), DENOMINATOR


def franek() -> tuple:
    return list(FRANEK), 1
