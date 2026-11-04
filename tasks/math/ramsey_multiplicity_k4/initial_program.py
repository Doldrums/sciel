# EVOLVE-BLOCK-START
import os
import time

import numpy as np
from scipy.optimize import minimize
from sciel.population import population
from sciel.task import asset

graphon = asset("graphon")


def kernel():
    """
    Construct q, one red probability per class of pairs of the 3840 parts (graphon.structure() gives the classes),
    for the step graphon W[i, j] = q[classes[i, j]] whose monochromatic K4 density t(K4, W) + t(K4, 1 - W) is as
    small as possible.

    Returns:
        (numerators, denominator): one integer in [0, denominator] per class, denominator <= 2**30; or an array of
        binary64 values in [0, 1] that are multiples of 2**-30.
    """
    deadline = time.time() + 0.6 * float(os.environ.get("SCIEL_STAGE_SECONDS", "30"))
    # The first evaluation returns the best known construction unchanged, so the record enters the population.
    views = [program for program in population() if program["output"] is not None]
    if not views:
        return asset("frontier").kernel()
    # Afterwards jump out of the parent's basin (or the view's best): flip a few classes between red and blue, move
    # the fractional ones, then descend from there with L-BFGS-B in the box [0, 1].
    q = graphon.as_float(graphon.load(next((p for p in views if p["parent"]), views[0])["output"]))
    rng = np.random.default_rng(int(time.time() * 1e6) % 2**32)
    flips = rng.choice(q.size, size=int(rng.integers(1, 5)), replace=False)
    q[flips] = 1 - q[flips]
    q = np.clip(q + rng.uniform(0.0, 0.02) * rng.standard_normal(q.size) * ((q > 0) & (q < 1)), 0, 1)

    def objective(x):
        v, g = graphon.value_and_gradient(x)
        return 1e3 * v, 1e3 * g

    while time.time() < deadline:
        result = minimize(objective, q, jac=True, method="L-BFGS-B", bounds=[(0.0, 1.0)] * q.size,
                          options={"maxiter": 5, "ftol": 1e-16, "gtol": 1e-14})
        if np.array_equal(result.x, q):
            break
        q = result.x
    # The descended jump is returned even when its density is above the start's: a distinct output is admitted.
    return graphon.snap(q, 30)


# EVOLVE-BLOCK-END
