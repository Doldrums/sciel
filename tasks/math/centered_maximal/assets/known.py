"""The best known construction, as exact data: from sciel.task import asset; known = asset("known").

Lin's lattice (github.com/CoolRmal/centered-maximal-constant, commit 8e1ee3d, Lean-checked): with u = (2 + sqrt 22)/3,
w = u^2 - 1 = (17 + 4 sqrt 22)/9, h = (1 + u)/2 = (5 + sqrt 22)/6 and V = h + 1, masses 1 at (ih, jV) for even i and
w for odd i. A period box is 2h x V with one atom of each kind; M mu >= 1 on it except four open a x b slots, so
c_2 >= Phi = (2hV - 4ab)/(1 + w) = 1.6855099933555252, with a = 2h - sqrt(2(2 + w))/2 - u/2 and
b = V - sqrt(2) u/2 - sqrt(w)/2. lin() places the heavy atom at (0, V/2) and the unit atom at (h, V/2), so every
witness of Lin's proof lies in the 3 x 3 block of periods the evaluator searches; sqrt 22 is rounded to 2**-bits.
"""

from decimal import Decimal, getcontext
from fractions import Fraction
import math

PHI = 1.6855099933555252


def lin(bits: int = 88) -> tuple:
    """((P_x, P_y), [(x, y, mass), ...]) of Lin's lattice with sqrt 22 rounded down to a multiple of 2**-bits."""
    s = Fraction(math.isqrt(22 << (2 * bits)), 1 << bits)
    h = (5 + s) / 6
    v = h + 1
    w = (17 + 4 * s) / 9
    return (2 * h, v), [(Fraction(0), v / 2, w), (h, v / 2, Fraction(1))]


def phi(digits: int = 40) -> Decimal:
    """Phi from Lin's closed form, to the given number of digits."""
    getcontext().prec = digits + 10
    r22, r2 = Decimal(22).sqrt(), Decimal(2).sqrt()
    u = (2 + r22) / 3
    w = u * u - 1
    h = (1 + u) / 2
    v = h + 1
    a = 2 * h - (2 * (2 + w)).sqrt() / 2 - u / 2
    b = v - r2 * u / 2 - w.sqrt() / 2
    return +((2 * h * v - 4 * a * b) / (1 + w))
