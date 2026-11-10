# EVOLVE-BLOCK-START
import os
import time

import numpy as np
from sciel.population import population
from sciel.task import asset

bh = asset("bh")


def candidate():
    """
    Construct a function f on {-1, 1}^n, n <= 22, of degree d whose ratio (sum |f^(S)|^p)^(1 / p) / max |f|,
    p = 2d / (d + 1), is as large as possible.

    Returns:
        {"n": n, "terms": [[mask, a], ...]}: f = sum a x^S with S the set bits of mask, a nonzero integers.
    """
    deadline = time.time() + 0.4 * float(os.environ.get("SCIEL_STAGE_SECONDS", "30"))
    # The first evaluation returns the best known construction unchanged, so the record enters the population.
    views = [program for program in population() if program["output"] is not None]
    if not views:
        return asset("frontier").candidate()
    # Afterwards jump out of the parent's basin (or the view's best): rescale a few coefficients or add one new term
    # of degree at most d, then climb by single-coefficient moves on the float ratio at the same d.
    a = bh.array(bh.load(next((p for p in views if p["parent"]), views[0])["output"])).astype(np.float64)
    a /= np.abs(a).max()
    n = len(a).bit_length() - 1
    d = bh.degree(a)
    allowed = np.flatnonzero(bh.popcounts(n) <= d)
    rng = np.random.default_rng(int(time.time() * 1e6) % 2**32)
    for _ in range(int(rng.integers(1, 4))):
        if rng.random() < 0.3:
            a[rng.choice(allowed)] += rng.normal(0, 0.3)
        else:
            a[rng.choice(np.flatnonzero(a))] *= np.exp(rng.normal(0, 0.3))
    value = bh.ratio(a, d)
    while time.time() < deadline:
        i = rng.choice(allowed) if rng.random() < 0.1 else rng.choice(np.flatnonzero(a))
        old = a[i]
        a[i] = old * np.exp(rng.normal(0, 0.1)) if old and rng.random() < 0.8 else old + rng.normal(0, 0.1)
        v = bh.ratio(a, d)
        if v >= value:
            value = v
        else:
            a[i] = old
    # The climbed jump is returned even when its ratio is below the start's: a distinct output is admitted.
    return bh.make(a / np.abs(a).max(), 2 ** 20)


# EVOLVE-BLOCK-END
