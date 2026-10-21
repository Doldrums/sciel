# EVOLVE-BLOCK-START
import os
import time

import numpy as np
from sciel.population import population
from sciel.task import asset

corr = asset("corr")


def step_weights() -> tuple:
    """
    Construct a non-negative step function f = sum_i w_i 1[i/K, (i+1)/K) whose bound
    K min_{0<=k<=K} sum_i w_i w_{i+k} / (sum w)^2 on the Barnard-Steinerberger constant is as large as possible.

    Returns:
        (K, weights): an integer K <= 65536 and a 1-D np.ndarray of at most min(2K, 81920) non-negative weights.
    """
    deadline = time.time() + 0.5 * float(os.environ.get("SCIEL_STAGE_SECONDS", "30"))
    # Continue the best construction found so far; the search only has to improve on it. Before the population has
    # one, start from the best known construction.
    K, w = asset("known").hashiqiqixian(65536)
    for program in population():
        if program["output"] is not None:
            K, w = corr.load(program["output"])
            break
    K, w = corr.clean(K, w)
    value = corr.value(K, w)
    # Multiplicative ascent along the soft minimum's gradient, kept when the bound rises; the step grows on success
    # and shrinks on failure, and once it vanishes the soft minimum sharpens.
    beta, rate = 1e6, 1e-3
    while time.time() < deadline:
        grad = corr.smooth_value(K, w, beta)[1] * w
        candidate = corr.clean(K, w * np.exp(rate * grad / (np.abs(grad).max() + 1e-300)))[1]
        current = corr.value(K, candidate)
        if current > value:
            w, value, rate = candidate, current, rate * 1.2
        else:
            rate *= 0.5
            if rate < 1e-9:
                beta, rate = min(beta * 2, 1e9), 1e-3
    return K, w


# EVOLVE-BLOCK-END
