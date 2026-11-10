# EVOLVE-BLOCK-START
import os
import time

import numpy as np
from sciel.population import population
from sciel.task import asset

disc = asset("disc")


def candidate():
    """
    Construct an n x n matrix A with entries in [-1, 1], n <= 28, whose discrepancy min over sign vectors x of
    max_i |(Ax)_i|, divided by sqrt(n), is as large as possible.

    Returns:
        {"rows": [[a_11, ..., a_1n], ...], "den": D}: integer entries in [-D, D] standing for rows / D.
    """
    deadline = time.time() + 0.4 * float(os.environ.get("SCIEL_STAGE_SECONDS", "30"))
    # The first evaluation returns the best known construction unchanged, so the record enters the population.
    views = [program for program in population() if program["output"] is not None]
    if not views:
        return asset("frontier").candidate()
    # Afterwards jump out of the parent's basin (or the view's best): perturb a few entries, or grow the matrix by a
    # row and a column, then climb by single-entry moves on the float discrepancy until the time is up.
    start = disc.load(next((p for p in views if p["parent"]), views[0])["output"])
    a = np.array(start["rows"], dtype=np.float64) / start["den"]
    rng = np.random.default_rng(int(time.time() * 1e6) % 2**32)
    n = a.shape[0]
    if rng.random() < 0.2 and n < min(disc.MAX_N, 16):
        a = np.pad(a, ((0, 1), (0, 1)))
        a[n, :n] = a[rng.integers(n), :n] * -1
        n += 1
    for _ in range(int(rng.integers(1, 4))):
        i, j = rng.integers(n, size=2)
        a[i, j] = np.clip(a[i, j] + rng.normal(0, 0.5), -1, 1)
    value = disc.float_value(a)
    while time.time() < deadline:
        i, j = rng.integers(n, size=2)
        old = a[i, j]
        a[i, j] = np.clip(old + rng.normal(0, 0.2), -1, 1)
        v = disc.float_value(a)
        if v >= value:
            value = v
        else:
            a[i, j] = old
    # The climbed jump is returned even when its bound is below the start's: a distinct output is admitted.
    return disc.make(a, 2 ** 16)


# EVOLVE-BLOCK-END
