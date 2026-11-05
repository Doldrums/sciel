# EVOLVE-BLOCK-START
import os
import time

import numpy as np
from sciel.population import population
from sciel.task import asset

ghr = asset("ghr")


def candidate():
    """
    Construct a digit mask M (finite set of integers in [0, 2000] containing 0) and tilts x, y in (0, 1) for which
    Olafson's two-temperature bound 2 - sigma / rho on C_3a is as large as possible.

    Returns:
        {"mask": [m, ...], "x": x, "y": y}: distinct integers; x and y binary64 numbers or [numerator, denominator].
    """
    deadline = time.time() + 0.4 * float(os.environ.get("SCIEL_STAGE_SECONDS", "30"))
    # The first evaluation returns the best known construction unchanged, so the record enters the population.
    views = [program for program in population() if program["output"] is not None]
    if not views:
        return asset("frontier").candidate()
    # Afterwards jump out of the parent's basin (or the view's best): toggle one to three elements below the top or
    # move the cut, then re-optimize both tilts and climb by single toggles until the time is up.
    start = ghr.load(next((p for p in views if p["parent"]), views[0])["output"])
    mask = set(start["mask"])
    rng = np.random.default_rng(int(time.time() * 1e6) % 2**32)
    top = max(mask)
    if rng.random() < 0.7:
        for e in rng.choice(np.arange(1, top + 1), size=int(rng.integers(1, 4)), replace=False):
            mask ^= {int(e)}
    else:
        cut = int(np.clip(top + rng.integers(-10, 11), 2, ghr.MAX_ELEMENT))
        mask = {m for m in mask if m <= cut} | {cut}
    if len(mask) < 2 or mask == set(start["mask"]):
        mask = set(start["mask"]) ^ {top + 1 if top < ghr.MAX_ELEMENT else top - 1}
    value, x, y = ghr.best(sorted(mask))
    while time.time() < deadline:
        e = int(rng.integers(1, max(mask) + 2))
        trial = mask ^ {e}
        if len(trial) < 2 or trial == set(start["mask"]):
            continue
        v, tx, ty = ghr.best(sorted(trial), x)
        if v > value:
            mask, value, x, y = trial, v, tx, ty
    # The climbed jump is returned even when its bound is below the start's: a distinct output is admitted.
    return ghr.make(sorted(mask), x, y)


# EVOLVE-BLOCK-END
