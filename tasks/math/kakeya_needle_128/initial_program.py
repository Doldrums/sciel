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
    # Continue the best construction found so far; the search only has to improve on it. Before the population has
    # one, start from the best construction on the EinsteinArena board.
    start = asset("frontier").offsets()
    for program in population():
        if program["output"] is not None:
            start = program["output"]["offsets"]
            break
    start = kakeya.decimals(start)
    x = kakeya.values(start)
    value = kakeya.area(x)
    best = value
    # Alternate a descent step along the subgradient with a slide of a random block of consecutive triangles; a move
    # is kept when the area falls, its scale grows on success and shrinks on failure.
    rng = np.random.default_rng(0)
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
    # The floating-point area guides the search; the exact area decides what is returned.
    if value < best and kakeya.exact_area(x) < kakeya.exact_area(start):
        return kakeya.decimals(x)
    return start


# EVOLVE-BLOCK-END
