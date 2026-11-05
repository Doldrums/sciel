# EVOLVE-BLOCK-START
import os
import time

import numpy as np
from scipy.optimize import minimize
from sciel.population import population
from sciel.task import asset

group = asset("group")


def kernel():
    """
    Construct a finite abelian group Z_{m_1} x ... x Z_{m_k} of order at most 256 and a symmetric q on it, the red
    probability between parts x and y of a step graphon when x - y = z, whose monochromatic K5 density
    t(K5, W) + t(K5, 1 - W) is as small as possible.

    Returns:
        {"moduli": [m_1, ..., m_k], "q": (numerators, denominator)}: one integer in [0, denominator] per group
        element in mixed radix, denominator <= 2**32, q(-z) = q(z); q may also be an array of binary64 multiples of
        2**-32.
    """
    deadline = time.time() + 0.4 * float(os.environ.get("SCIEL_STAGE_SECONDS", "30"))
    # The first evaluation returns the best known mixed-group construction unchanged, so it enters the population.
    views = [program for program in population() if program["output"] is not None]
    if not views:
        return asset("frontier").kernel()
    # Afterwards jump out of the parent's basin (or the view's best): round one to five orbits {z, -z} to 0 or 1 and
    # flip them, then descend with L-BFGS-B in the box [0, 1] and keep hopping from the best descent until the time
    # is up.
    moduli, start = group.as_float(group.load(next((p for p in views if p["parent"]), views[0])["output"]))
    orbit = group.orbits(moduli)
    free = np.unique(orbit)
    rng = np.random.default_rng(int(time.time() * 1e6) % 2**32)

    def objective(x):
        v, g = group.value_and_gradient(moduli, x)
        return 1e3 * v, 1e3 * g

    def hop(q):
        y = q.copy()
        for j in rng.choice(free, size=int(rng.integers(1, 6)), replace=False):
            y[orbit == j] = 1 - np.round(y[j])
        result = minimize(objective, y, jac=True, method="L-BFGS-B", bounds=[(0.0, 1.0)] * y.size,
                          options={"maxiter": 100, "ftol": 1e-16, "gtol": 1e-14})
        return group.symmetrize(moduli, result.x), result.fun

    best, value = hop(start)
    while time.time() < deadline:
        q, v = hop(best)
        if v < value:
            best, value = q, v
    # The descended jump is returned even when its density is above the start's: a distinct output is admitted.
    return group.snap(moduli, best, 32)


# EVOLVE-BLOCK-END
