# EVOLVE-BLOCK-START
import os
import time

import numpy as np
from sciel.population import population
from sciel.task import asset

sph = asset("sph")


def codes() -> dict:
    """
    For every instance "d_n" in sph.INSTANCES, n points on the unit sphere in R^d whose largest pair cosine is as
    small as possible (the minimal angle as large as possible).

    Returns:
        codes: dict {"d_n": (n, d) array or list of n points}; points need not be normalized.
    """
    deadline = time.time() + 0.5 * float(os.environ.get("SCIEL_STAGE_SECONDS", "30"))
    # The first evaluation returns the best known codes unchanged, so the record enters the population.
    views = [program for program in population() if program["output"] is not None]
    if not views:
        return {k: sph.best(k).tolist() for k in sph.INSTANCES}
    # Afterwards jump a few instances of the parent's codes (or the view's best) out of their basins: move a point in
    # contact and its nearest neighbours by a fraction of the minimal chord, then re-maximize for an equal share of
    # half the stage time. A jumped instance is returned even when its cosine rises, since a distinct output is
    # admitted; every other instance is returned as it was.
    output = next((p for p in views if p["parent"]), views[0])["output"]
    current = {k: sph.decode(text, sph.DIM[k]) for k, text in output["codes"].items()}
    rng = np.random.default_rng(int(time.time() * 1e6) % 2**32)
    chosen = rng.choice(sph.INSTANCES, size=int(rng.integers(1, 4)), replace=False)
    for i, k in enumerate(chosen):
        until = time.time() + (deadline - time.time()) / (len(chosen) - i)
        u = sph.directions(current[k])
        held = np.unique(sph.contacts(u))
        centre = int(rng.choice(held)) if held.size else int(rng.integers(len(u)))
        group = np.argsort(-(u @ u[centre]))[:int(rng.integers(2, 2 * sph.DIM[k]))]
        u = sph.shake(u, rng.uniform(0.02, 0.2), rng, group)
        u = sph.descend(u, betas=(1e3, 1e4, 1e5), deadline=until)
        current[k] = sph.improve(u, until, rng, sigma=(1e-4, 1e-2))
    return {k: np.asarray(u).tolist() for k, u in current.items()}


# EVOLVE-BLOCK-END
