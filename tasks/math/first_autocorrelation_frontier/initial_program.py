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
    # The first evaluation returns the best known construction unchanged, so the record enters the population.
    views = [program for program in population() if program["output"] is not None]
    if not views:
        return asset("frontier").heights()
    # Afterwards jump out of the basin of the parent's construction (or the view's best): multiplicative noise of a
    # few percent on a random block of steps, sometimes on a grid coarsened by two, then descend from there.
    h = np.array(steps.load(next((p for p in views if p["parent"]), views[0])["output"]), dtype=float)
    rng = np.random.default_rng(int(time.time() * 1e6) % 2**32)
    if rng.random() < 0.3 and h.size >= 2048:
        h = steps.resample(h, h.size // 2)
    lo = int(rng.integers(h.size))
    width = int(rng.integers(h.size // 8, h.size // 2 + 1))
    block = (lo + np.arange(width)) % h.size
    h[block] *= np.exp(rng.uniform(0.01, 0.05) * rng.standard_normal(width))
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
    # The descended jump is returned even when its bound is above the start's: a distinct output is admitted.
    return h


# EVOLVE-BLOCK-END
