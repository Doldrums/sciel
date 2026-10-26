# EVOLVE-BLOCK-START
import os
import time

import numpy as np
from sciel.population import population
from sciel.task import asset

steps = asset("steps")


def step_heights() -> np.ndarray:
    """
    Construct a step function with n equal steps on [-1/4, 1/4], possibly negative, whose bound
    2n max(abs(convolve(h, h))) / sum(h)^2 on the Matolcsi-Vinuesa autoconvolution constant is as small as possible.

    Returns:
        heights: 1-D np.ndarray of at most 65536 step heights.
    """
    deadline = time.time() + 0.5 * float(os.environ.get("SCIEL_STAGE_SECONDS", "30"))
    # The first evaluation returns the best earlier construction unchanged, so the record enters the population.
    views = [program for program in population() if program["output"] is not None]
    if not views:
        return asset("frontier").heights()
    # Afterwards jump out of the basin of the parent's construction (or the view's best): noise of a few percent of
    # the typical height on a random block of steps, sometimes on a doubled grid, then descend from there.
    h = np.array(next((p for p in views if p["parent"]), views[0])["output"]["heights"], dtype=float)
    rng = np.random.default_rng(int(time.time() * 1e6) % 2**32)
    if rng.random() < 0.3 and 2 * h.size <= steps.MAX_STEPS:
        h = steps.refine(h, 2)
    h = steps.clean(h)
    lo = int(rng.integers(h.size))
    width = int(rng.integers(h.size // 8, h.size // 2 + 1))
    block = (lo + np.arange(width)) % h.size
    h[block] += rng.uniform(0.01, 0.05) * np.abs(h).mean() * rng.standard_normal(width)
    h = steps.clean(h)
    # L-BFGS on the soft maximum with beta doubled between rounds, keeping the lowest bound seen.
    best, value, beta = h, steps.bound(h), 1e4
    while time.time() < deadline:
        h = steps.descend(h, beta, iters=50)
        current = steps.bound(h)
        if current < value:
            best, value = h, current
        beta = min(beta * 2, 1e10)
    # The descended jump is returned even when its bound is above the start's: a distinct output is admitted.
    return best


# EVOLVE-BLOCK-END
