# EVOLVE-BLOCK-START
import os
import time

import numpy as np
from sciel.population import population
from sciel.task import asset

fei = asset("fei")


def candidate():
    """
    Construct a balanced Boolean function f on n <= 22 variables whose spectral entropy over total influence minus
    one, H / (I - 1), is as large as possible.

    Returns:
        {"n": n, "table": hex}: the truth table, bit x of int(hex, 16) set when f(x) = -1.
    """
    deadline = time.time() + 0.4 * float(os.environ.get("SCIEL_STAGE_SECONDS", "30"))
    # The first evaluation returns the best known construction unchanged, so the record enters the population.
    views = [program for program in population() if program["output"] is not None]
    if not views:
        return asset("frontier").candidate()
    # Afterwards jump out of the parent's basin (or the view's best): swap the values of two to six points of
    # opposite sign, which keeps f balanced, then climb by single swaps until the time is up.
    start = fei.array(fei.load(next((p for p in views if p["parent"]), views[0])["output"]))
    rng = np.random.default_rng(int(time.time() * 1e6) % 2**32)
    f = start.copy()
    minus, plus = np.flatnonzero(f < 0), np.flatnonzero(f > 0)
    for _ in range(int(rng.integers(1, 4))):
        a, b = rng.choice(minus), rng.choice(plus)
        f[a], f[b] = f[b], f[a]
    if np.array_equal(f, start):
        f[minus[0]], f[plus[0]] = 1, -1
    value = fei.rates(f)[0]
    while time.time() < deadline:
        a, b = rng.choice(np.flatnonzero(f < 0)), rng.choice(np.flatnonzero(f > 0))
        f[a], f[b] = 1, -1
        v = fei.rates(f)[0]
        if v > value and not np.array_equal(f, start):
            value = v
        else:
            f[a], f[b] = -1, 1
    # The climbed jump is returned even when its bound is below the start's: a distinct output is admitted.
    return fei.make(f)


# EVOLVE-BLOCK-END
