# EVOLVE-BLOCK-START
import os
import time

import numpy as np
from sciel.population import population
from sciel.task import asset

entropy = asset("entropy")


def law():
    """
    Construct a mixture of finite laws on Z^2 for which sum_k t_k H_k(x - y) / max over (1,0), (0,1), (1,1)
    of sum_k t_k H_k(a x + b y) is as large as possible: a lower bound on the constant C_3b.

    Returns:
        {"laws": [{"points": [[x, y], ...], "weights": [w, ...]}, ...], "mix": [t, ...]}: at most 4 laws of at most
        512 distinct points each, weights and shares as binary64 numbers, ints or [numerator, denominator] pairs.
    """
    deadline = time.time() + 0.4 * float(os.environ.get("SCIEL_STAGE_SECONDS", "30"))
    # The first evaluation returns the best known construction unchanged, so the record enters the population.
    views = [program for program in population() if program["output"] is not None]
    if not views:
        return asset("frontier").law()
    # Afterwards jump out of the parent's basin (or the view's best): in one law add a few lattice points next to the
    # support or drop its lightest points, shake all weights by a few percent, then re-optimize weights and shares.
    laws, mix = entropy.as_float(entropy.load(next((p for p in views if p["parent"]), views[0])["output"]))
    rng = np.random.default_rng(int(time.time() * 1e6) % 2**32)
    k = int(rng.integers(len(laws)))
    points, p = laws[k]
    if rng.random() < 0.6 or p.size < 8:
        near = entropy.neighbours(points, radius=2)
        new = near[rng.choice(len(near), size=min(len(near), int(rng.integers(1, 5))), replace=False)]
        points = np.concatenate([points, new])
        p = np.concatenate([p, np.full(len(new), 10 ** rng.uniform(-5, -3))])
    else:
        keep = np.argsort(p)[int(rng.integers(1, 3)):]
        points, p = points[keep], p[keep]
    laws[k] = (points, p / p.sum())
    laws = [(q, w * np.exp(rng.uniform(0.01, 0.05) * rng.standard_normal(w.size))) for q, w in laws]
    start = entropy.make([(q, w / w.sum()) for q, w in laws], mix if len(laws) > 1 else None)
    # The polished jump is returned even when its bound is below the start's: a distinct output is admitted.
    return entropy.polish(start, seconds=max(1.0, deadline - time.time()))


# EVOLVE-BLOCK-END
