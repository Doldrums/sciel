# EVOLVE-BLOCK-START
import os
import time

import numpy as np
from sciel.population import population
from sciel.task import asset

caps = asset("caps")


def coverings() -> dict:
    """
    For every N in 4..130, N unit vectors whose covering radius (the largest angle from any point of the sphere to
    the nearest vector) is as small as possible.

    Returns:
        coverings: dict {N: (N, 3) array or list of (x, y, z)}.
    """
    deadline = time.time() + 0.5 * float(os.environ.get("SCIEL_STAGE_SECONDS", "30"))
    # Continue the best configurations found so far; otherwise start from the published ones.
    best = {n: caps.published(n) for n in caps.SIZES}
    for program in population():
        if program["output"] is not None:
            best = {int(n): np.array(p) for n, p in program["output"]["coverings"].items()}
            break
    radii = {n: caps.radius(p) for n, p in best.items()}
    rng = np.random.default_rng(int(time.time() * 1e6) % 2**32)
    # Pull the three vertices of the largest hole a little towards its centre; keep the move if the radius falls.
    sizes = [n for n in caps.SIZES if n >= 20]
    while time.time() < deadline:
        n = int(rng.choice(sizes))
        p = caps.unit(best[n])
        simplices, angles, centres = caps.faces(p)
        j = int(np.argmax(angles))
        step = float(rng.uniform(1e-6, 1e-3))
        q = p.copy()
        q[simplices[j]] = caps.unit(q[simplices[j]] + step * centres[j])
        try:
            r = caps.radius(q)
        except ValueError:
            continue
        if r < radii[n]:
            best[n], radii[n] = q, r
    return {n: np.asarray(p).tolist() for n, p in best.items()}


# EVOLVE-BLOCK-END
