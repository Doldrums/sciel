# EVOLVE-BLOCK-START
import os
import time

import numpy as np
from sciel.population import population
from sciel.task import asset

kakeya = asset("kakeya")


def offsets() -> list:
    """
    Choose offsets x_1..x_128 so that the union of the triangles conv{(x_j, 0), (x_j + 1/128, 0), (x_j + j/128, 1)}
    has as small an area as possible.

    Returns:
        offsets: 128 decimal strings (at most 25 fractional digits) or numbers.
    """
    deadline = time.time() + 0.5 * float(os.environ.get("SCIEL_STAGE_SECONDS", "30"))
    # The first evaluation returns the best construction on the EinsteinArena board unchanged, so the record enters
    # the population.
    views = [program for program in population() if program["output"] is not None]
    if not views:
        return asset("frontier").offsets()
    # Afterwards jump out of the basin of the parent's construction (or the view's best): noise of a few percent of
    # a triangle's base 1/128 on a random block of offsets and a few block slides of that size, then descend.
    x = kakeya.values(next((p for p in views if p["parent"]), views[0])["output"]["offsets"])
    rng = np.random.default_rng(int(time.time() * 1e6) % 2**32)
    size = rng.uniform(0.01, 0.05) / 128
    lo = int(rng.integers(0, 128))
    hi = int(rng.integers(lo + 1, min(128, lo + 64) + 1))
    x = x.copy()
    x[lo:hi] += size * rng.standard_normal(hi - lo)
    for _ in range(int(rng.integers(1, 4))):
        lo = int(rng.integers(0, 128))
        x = kakeya.slide(x, lo, int(rng.integers(lo + 1, min(128, lo + 32) + 1)), size * rng.standard_normal())
    value = kakeya.area(x)
    # Alternate a descent step along the subgradient with a slide of a random block of consecutive triangles; a move
    # is kept when the area falls, its scale grows on success and shrinks on failure.
    rate = 1e-5
    while time.time() < deadline - 3:
        if rng.random() < 0.5:
            grad = kakeya.subgradient(x)[1]
            candidate = x - rate * grad / (np.abs(grad).max() + 1e-300)
        else:
            lo = int(rng.integers(0, 128))
            hi = int(rng.integers(lo + 1, min(128, lo + 32) + 1))
            candidate = kakeya.slide(x, lo, hi, rate * rng.standard_normal())
        current = kakeya.area(candidate)
        if current < value - 1e-15:
            x, value, rate = candidate, current, rate * 1.5
        else:
            rate = max(rate * 0.9, 1e-10)
    # The descended jump is returned even when its area is above the start's: a distinct output is admitted.
    return kakeya.decimals(x)


# EVOLVE-BLOCK-END
