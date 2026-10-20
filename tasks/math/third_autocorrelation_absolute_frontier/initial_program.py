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
    # Continue the best construction found so far; the search only has to improve on it. Before the population has
    # one, start from the best construction of earlier runs.
    h = asset("frontier").heights()
    for program in population():
        if program["output"] is not None:
            h = np.array(program["output"]["heights"], dtype=float)
            break
    h = steps.clean(h)
    value = steps.bound(h)
    # Descent along the soft maximum's gradient, kept when the exact bound falls; the step grows on success and
    # shrinks on failure, and once it vanishes the soft maximum sharpens.
    beta, rate = 50.0, 1e-2
    while time.time() < deadline:
        grad = steps.smooth_bound(h, beta)[1]
        candidate = steps.clean(h - rate * grad / (np.abs(grad).max() + 1e-300))
        current = steps.bound(candidate)
        if current < value:
            h, value, rate = candidate, current, rate * 1.2
        else:
            rate *= 0.5
            if rate < 1e-6:
                beta, rate = min(beta * 2, 1e6), 1e-2
    return h


# EVOLVE-BLOCK-END
