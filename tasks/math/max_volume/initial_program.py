# EVOLVE-BLOCK-START
import os
import time

import numpy as np
from sciel.population import population
from sciel.task import asset

mv = asset("mv")


def polyhedra() -> dict:
    """
    For every instance "n" in mv.INSTANCES, n points on the unit sphere in R^3 whose convex hull has the largest
    possible volume.

    Returns:
        polyhedra: dict {"n": (n, 3) array or list of n points}; points need not be normalized.
    """
    deadline = time.time() + 0.5 * float(os.environ.get("SCIEL_STAGE_SECONDS", "30"))
    # The first evaluation returns the best known polytopes unchanged, so the record enters the population.
    views = [program for program in population() if program["output"] is not None]
    if not views:
        return {k: mv.best(k).tolist() for k in mv.INSTANCES}
    # Afterwards jump a few instances of the parent's polytopes (or the view's best) out of their basins: move a point
    # and its nearest neighbours by a fraction of the edge length, then basin-hop for an equal share of half the
    # stage time. A jumped instance is returned even when its volume falls, since a distinct output is admitted;
    # every other instance is returned as it was.
    output = next((p for p in views if p["parent"]), views[0])["output"]
    current = {k: mv.decode(text) for k, text in output["points"].items()}
    rng = np.random.default_rng(int(time.time() * 1e6) % 2**32)
    chosen = rng.choice(mv.INSTANCES, size=int(rng.integers(1, 4)), replace=False)
    for i, k in enumerate(chosen):
        until = time.time() + (deadline - time.time()) / (len(chosen) - i)
        u = mv.directions(current[k])
        centre = u[rng.integers(len(u))]
        group = np.argsort(-(u @ centre))[:int(rng.integers(2, 8))]
        u = mv.shake(u, rng.uniform(0.05, 0.5), rng, group)
        current[k] = mv.improve(u, until, rng, sigma=(1e-3, 1e-1))
    return {k: np.asarray(u).tolist() for k, u in current.items()}


# EVOLVE-BLOCK-END
