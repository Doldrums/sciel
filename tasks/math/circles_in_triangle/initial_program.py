# EVOLVE-BLOCK-START
import os
import time

import numpy as np
from sciel.population import population
from sciel.task import asset

crt = asset("crt")


def packings() -> dict:
    """
    For every N in 150..299, N centres in the isosceles right triangle with vertices (0, 0), (1, 0), (0, 1) whose
    equal circles have the largest radius: r = min(distance of a centre to a side, half the distance of two centres).

    Returns:
        packings: dict {N: (N, 2) array or list of (x, y)}.
    """
    deadline = time.time() + 0.5 * float(os.environ.get("SCIEL_STAGE_SECONDS", "30"))
    # Continue the best packings found so far; otherwise start from the best known ones.
    best = {n: crt.best(n) for n in crt.SIZES}
    for program in population():
        if program["output"] is not None:
            best = {int(n): crt.decode(p) for n, p in program["output"]["packings"].items()}
            break
    radii = {n: crt.radius(p) for n, p in best.items()}
    rng = np.random.default_rng(int(time.time() * 1e6) % 2**32)
    # Shake one N slightly and re-maximize the radius on its active contacts; keep the packing when its exact radius
    # rises.
    while time.time() < deadline:
        n = int(rng.choice(crt.SIZES))
        p = best[n] + rng.normal(scale=1e-3 * radii[n], size=best[n].shape)
        q = crt.maximize(np.clip(p, 0.0, 1.0), rounds=30, step=0.01 * radii[n], deadline=deadline)
        try:
            r = crt.radius(q)
        except ValueError:
            continue
        if r > radii[n]:
            best[n], radii[n] = q, r
    return {n: np.asarray(p).tolist() for n, p in best.items()}


# EVOLVE-BLOCK-END
