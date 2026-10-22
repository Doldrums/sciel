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
    # The first evaluation returns the best known construction unchanged, so the record enters the population.
    views = [program for program in population() if program["output"] is not None]
    if not views:
        return asset("known").hashiqiqixian(65536)
    # Afterwards jump out of the basin of the parent's construction (or the view's best): multiplicative noise of up
    # to a percent on a random block of cells (the minimum over lags drops fast), then ascend from there.
    K, w = corr.load(next((p for p in views if p["parent"]), views[0])["output"])
    w = np.array(w, dtype=float)
    rng = np.random.default_rng(int(time.time() * 1e6) % 2**32)
    lo = int(rng.integers(w.size))
    width = int(rng.integers(w.size // 8, w.size // 2 + 1))
    block = (lo + np.arange(width)) % w.size
    w[block] *= np.exp(rng.uniform(0.002, 0.01) * rng.standard_normal(width))
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
    # The ascended jump is returned even when its bound is below the start's: a distinct output is admitted.
    return K, w


# EVOLVE-BLOCK-END
