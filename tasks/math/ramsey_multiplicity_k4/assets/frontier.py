"""The best known construction for this task: the 3840-part step graphon of the second L-BFGS round in
github.com/vidigoat/optimization-records (commit 9f82508, tao/certificates/c89, teorth/optimizationproblems PR 270),
itself a refinement of J. SK's gradient refinement of the Van Seters-Jothimani twelve-Clebsch construction. Its red
probabilities are multiples of 2**-30, constant on the classes of the task; its monochromatic K4 density is exactly
DENSITY."""

from fractions import Fraction
from pathlib import Path

import numpy as np

DENSITY = Fraction(31383468324407329838135970970371081393990727093313271428891812741,
                   1041295852679825698551191435837073366434387539971249757275291648000)


def kernel() -> tuple:
    """(numerators, denominator), one numerator per class."""
    data = np.load(Path(__file__).resolve().parent / "frontier.npz")
    return [int(x) for x in data["numerators"]], int(data["denominator"])
