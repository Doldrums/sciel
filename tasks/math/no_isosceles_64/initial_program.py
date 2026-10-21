# EVOLVE-BLOCK-START
import os
import time

import numpy as np
from sciel.population import population
from sciel.task import asset

iso = asset("iso")


def points() -> list:
    """
    A subset of the 64 x 64 grid {0..63}^2 with no three points forming a possibly flat isosceles triangle, as large
    as possible.

    Returns:
        points: list of (x, y) integer pairs.
    """
    deadline = time.time() + 0.5 * float(os.environ.get("SCIEL_STAGE_SECONDS", "30"))
    # Continue the best set found so far; otherwise start from the best known 112 points.
    best = [tuple(p) for p in iso.BEST]
    for program in population():
        if program["output"] is not None:
            best = [tuple(p) for p in program["output"]["points"]]
            break
    rng = np.random.default_rng(int(time.time() * 1e6) % 2**32)
    # Remove a few points and regrow; keep any set at least as large, so the search drifts along plateaus.
    while time.time() < deadline:
        k = int(rng.integers(1, 6))
        keep = [best[i] for i in sorted(rng.choice(len(best), len(best) - k, replace=False))]
        trial = iso.grow(keep, rng)
        if len(trial) >= len(best):
            best = trial
    return best


# EVOLVE-BLOCK-END
