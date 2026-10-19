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
    # Continue the best construction found so far; the search only has to improve on it.
    h = np.full(400, 1.0)
    for program in population():
        if program["output"] is not None:
            h = np.array(program["output"]["heights"], dtype=float)
            break
    h = steps.clean(h)
    best, best_value = h, steps.bound(h)
    # Gradient descent on the soft maximum, sharpening it as the bound falls.
    beta, rate = 50.0, 1e-3
    while time.time() < deadline:
        value, grad = steps.smooth_bound(h, beta)
        h = steps.clean(h - rate * grad / (np.abs(grad).max() + 1e-300))
        current = steps.bound(h)
        if current < best_value:
            best, best_value = h, current
        else:
            beta, rate = min(beta * 1.5, 1e5), rate * 0.7
            h = best
    return best


# EVOLVE-BLOCK-END
