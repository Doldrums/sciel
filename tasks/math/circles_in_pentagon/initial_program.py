# EVOLVE-BLOCK-START
import os
import time

import numpy as np
from sciel.population import population
from sciel.task import asset

cpt = asset("cpt")


def packings() -> dict:
    """
    For every N in cpt.SIZES, N centres in the regular pentagon (circumradius 1, flat bottom edge) whose equal
    circles have the largest radius: r = min(h - n_k . c over the sides k, half the distance of two centres).

    Returns:
        packings: dict {N: (N, 2) array or list of (x, y)}.
    """
    deadline = time.time() + 0.5 * float(os.environ.get("SCIEL_STAGE_SECONDS", "30"))
    # The first evaluation returns the best known packings unchanged, so the record enters the population.
    views = [program for program in population() if program["output"] is not None]
    if not views:
        return {n: cpt.best(n).tolist() for n in cpt.SIZES}
    # Afterwards jump a few N of the parent's packings (or the view's best) out of their basins: move a centre in
    # contact and its nearest neighbours by a fraction of the radius (centres pushed out of the pentagon are put back on
    # its boundary), then re-maximize the radius for an equal share of half the stage time. A jumped N is returned
    # even when its radius falls, since a distinct output is admitted; every other N is returned as it was.
    output = next((p for p in views if p["parent"]), views[0])["output"]
    best = {int(n): cpt.decode(p) for n, p in output["packings"].items()}
    rng = np.random.default_rng(int(time.time() * 1e6) % 2**32)
    chosen = rng.choice(cpt.SIZES, size=int(rng.integers(1, 4)), replace=False)
    for k, n in enumerate(int(n) for n in chosen):
        until = time.time() + (deadline - time.time()) / (len(chosen) - k)
        p, r = best[n].copy(), cpt.radius(best[n])
        held = np.unique(cpt.contacts(p)[0])
        centre = int(rng.choice(held)) if held.size else int(rng.integers(n))
        group = np.argsort(((p - p[centre]) ** 2).sum(axis=1))[:int(rng.integers(4, 13))]
        p[group] += rng.normal(scale=rng.uniform(0.1, 0.3) * r, size=(len(group), 2))
        p = cpt.maximize(cpt.clip(p), rounds=10**6, step=0.05 * r, deadline=until)
        # Then shake slightly and re-maximize, keeping rises, for the rest of this N's share.
        while time.time() < until:
            q = cpt.maximize(cpt.clip(p + rng.normal(scale=1e-3 * r, size=p.shape)),
                             rounds=30, step=0.01 * r, deadline=until)
            if cpt.radius_fast(q) > cpt.radius_fast(p):
                p = q
        best[n] = p
    return {n: np.asarray(p).tolist() for n, p in best.items()}


# EVOLVE-BLOCK-END
