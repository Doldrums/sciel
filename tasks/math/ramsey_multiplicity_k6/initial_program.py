# EVOLVE-BLOCK-START
import os
import time

import numpy as np
from scipy.optimize import minimize
from sciel.population import population
from sciel.task import asset

weights = asset("weights")


def kernel():
    """
    Construct q: {0, ..., n} -> [0, 1], n <= 10, the red probability between parts x and y of a step graphon on the
    2^n parts F_2^n when |x XOR y| = k, whose monochromatic K6 density t(K6, W) + t(K6, 1 - W) is as small as possible.

    Returns:
        (numerators, denominator): n + 1 integers in [0, denominator], denominator <= 2**32; or an array of n + 1
        binary64 values in [0, 1] that are multiples of 2**-32.
    """
    deadline = time.time() + 0.5 * float(os.environ.get("SCIEL_STAGE_SECONDS", "30"))
    # The first evaluation returns the best known construction unchanged, so the record enters the population.
    views = [program for program in population() if program["output"] is not None]
    if not views:
        return asset("frontier").kernel()
    # Afterwards jump out of the parent's basin (or the view's best): flip one or two weight classes between red and
    # blue and shake the fractional ones, then descend from there with L-BFGS-B in the box [0, 1].
    q = weights.as_float(weights.load(next((p for p in views if p["parent"]), views[0])["output"]))
    rng = np.random.default_rng(int(time.time() * 1e6) % 2**32)
    flips = rng.choice(q.size, size=int(rng.integers(1, 3)), replace=False)
    q[flips] = 1 - q[flips]
    q = np.clip(q + rng.uniform(0.0, 0.2) * rng.standard_normal(q.size) * ((q > 0) & (q < 1)), 0, 1)

    def objective(x):
        v, g = weights.value_and_gradient(x)
        return 1e5 * v, 1e5 * g

    while time.time() < deadline:
        result = minimize(objective, q, jac=True, method="L-BFGS-B", bounds=[(0.0, 1.0)] * q.size,
                          options={"maxiter": 20, "ftol": 1e-18, "gtol": 1e-16})
        if np.array_equal(result.x, q):
            break
        q = result.x
    # The descended jump is returned even when its density is above the start's: a distinct output is admitted.
    return weights.snap(q, 32)


# EVOLVE-BLOCK-END
