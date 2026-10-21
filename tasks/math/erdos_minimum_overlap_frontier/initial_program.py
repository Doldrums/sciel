# EVOLVE-BLOCK-START
import os
import time

import numpy as np
from sciel.population import population
from sciel.task import asset

steps = asset("steps")


def step_heights() -> np.ndarray:
    """
    Construct a step function on [0, 2] with n equal steps, values in [0, 1] and sum(h) exactly n/2,
    whose bound (2/n) max(correlate(h, 1 - h, 'full')) on the minimum overlap constant is as small as possible.

    Returns:
        heights: 1-D np.ndarray of at most 65536 step heights.
    """
    deadline = time.time() + 0.5 * float(os.environ.get("SCIEL_STAGE_SECONDS", "30"))
    # Continue the best construction found so far; the search only has to improve on it. Before the population has
    # one, start from the best known construction.
    h = asset("frontier").heights()
    for program in population():
        if program["output"] is not None:
            h = np.array(program["output"]["heights"], dtype=float)
            break
    h = steps.clean(h)
    value = steps.bound(h)
    # Projected descent along the soft maximum's gradient, kept when the bound falls by more than the FFT's rounding;
    # the step grows on success and shrinks on failure, and once it vanishes the soft maximum sharpens.
    beta, rate = 1e3, 1e-3
    while time.time() < deadline:
        grad = steps.smooth_bound(h, beta)[1]
        candidate = steps.clean(h - rate * grad / (np.abs(grad).max() + 1e-300))
        current = steps.bound(candidate)
        if current < value * (1 - 1e-12):
            h, value, rate = candidate, current, rate * 1.2
        else:
            rate *= 0.5
            if rate < 1e-9:
                beta, rate = min(beta * 2, 1e8), 1e-3
    return h


# EVOLVE-BLOCK-END
