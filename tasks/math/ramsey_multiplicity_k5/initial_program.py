# EVOLVE-BLOCK-START
import os
import time

import numpy as np
from scipy.optimize import minimize
from sciel.population import population
from sciel.task import asset

cayley = asset("cayley")


def kernel():
    """
    Construct q: F_2^n -> [0, 1], n <= 8, the red probability between parts x and y of a step graphon on 2^n equal
    parts when x XOR y = z, whose monochromatic K5 density t(K5, W) + t(K5, 1 - W) is as small as possible.

    Returns:
        (numerators, denominator): 2^n integers in [0, denominator], denominator <= 2**32; or an array of 2^n
        binary64 values in [0, 1] that are multiples of 2**-32.
    """
    deadline = time.time() + 0.4 * float(os.environ.get("SCIEL_STAGE_SECONDS", "30"))
    # The first evaluation returns the best known construction unchanged, so the record enters the population.
    views = [program for program in population() if program["output"] is not None]
    if not views:
        return asset("frontier").kernel()
    # Afterwards jump out of the parent's basin (or the view's best): flip a few entries between red and blue, move
    # the fractional ones, then descend from there with L-BFGS-B in the box [0, 1].
    q = cayley.as_float(cayley.load(next((p for p in views if p["parent"]), views[0])["output"]))
    rng = np.random.default_rng(int(time.time() * 1e6) % 2**32)
    flips = rng.choice(np.arange(1, q.size), size=int(rng.integers(2, 9)), replace=False)
    q[flips] = 1 - q[flips]
    q = np.clip(q + rng.uniform(0.0, 0.1) * rng.standard_normal(q.size) * ((q > 0) & (q < 1)), 0, 1)

    def objective(x):
        v, g = cayley.value_and_gradient(x)
        return 1e3 * v, 1e3 * g

    while time.time() < deadline:
        result = minimize(objective, q, jac=True, method="L-BFGS-B", bounds=[(0.0, 1.0)] * q.size,
                          options={"maxiter": 50, "ftol": 1e-16, "gtol": 1e-14})
        if np.array_equal(result.x, q):
            break
        q = result.x
    # The descended jump is returned even when its density is above the start's: a distinct output is admitted.
    return cayley.snap(q, 32)


# EVOLVE-BLOCK-END
