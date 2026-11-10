"""The best known construction for this task: SCiEL's weight-class kernel on F_2^10 (teorth/optimizationproblems PR
331, 5 November 2026), found from Jeetu SK's refinement of Franek's Cayley graph (PR 279, 0.742254861593625 * 2^-14).
The red probability between parts x and y is NUMERATORS[|x XOR y|] / 2^32. Its monochromatic K6 density is exactly
DENSITY = 0.7422042115280503... * 2^-14. FRANEK is the 0/1 kernel SK started from, whose density is 51162598917 / 2^50
(Combinatorica 22, 2002)."""

from fractions import Fraction

NUMERATORS = [4294967296, 1917548695, 0, 4294967296, 4294967296, 0, 81376308, 4294967296, 4294967296, 0, 0]
DENOMINATOR = 2 ** 32
FRANEK = [1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1]
DENSITY = Fraction(
    67432845284234356713012367110897548622422530589772294031336898680659579871445572986246952507783885816983659288359622409120016266716137,
    1488565707357402911845015158554633286356257506687627387456491927921949262056238946972039271861787782268441644249633132407878864135402815488)


def kernel() -> tuple:
    """(numerators, denominator) of q on the weights 0..10."""
    return list(NUMERATORS), DENOMINATOR


def franek() -> tuple:
    return list(FRANEK), 1
