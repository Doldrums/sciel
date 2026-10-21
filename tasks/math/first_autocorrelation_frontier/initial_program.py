# EVOLVE-BLOCK-START
import os
import time

import numpy as np
from sciel.population import population
from sciel.task import asset

steps = asset("steps")


def step_heights() -> np.ndarray:
    """
    Construct a non-negative step function with n equal steps on [-1/4, 1/4] whose bound
    2n max(convolve(h, h)) / sum(h)^2 on the first autocorrelation constant is as small as possible.

    Returns:
        heights: 1-D np.ndarray of at most 65536 step heights.
    """
    deadline = time.time() + 0.5 * float(os.environ.get("SCIEL_STAGE_SECONDS", "30"))
    # Continue the best construction found so far; the search only has to improve on it. Before the population has
    # one, start from the best known construction.
    h = asset("frontier").heights()
    for program in population():
        if program["output"] is not None:
            h = steps.load(program["output"])
            break
    h = steps.clean(h)
    value = steps.bound(h)
    # Descent along the soft maximum's gradient, kept when the bound falls by more than the FFT's rounding; the step
    # grows on success and shrinks on failure, and once it vanishes the soft maximum sharpens.
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
