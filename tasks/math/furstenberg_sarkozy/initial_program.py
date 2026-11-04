# EVOLVE-BLOCK-START
import os
import time

import numpy as np
from sciel.population import population
from sciel.task import asset

sarkozy = asset("sarkozy")


def certificate():
    """
    Construct an interval-moment certificate {alpha, chains, codes, binary} proving square-difference-free sets in
    [1, N] of size N^(alpha - o(1)); alpha is to be maximized.

    Returns:
        dict: alpha (fraction string), chains [[p, digits]], codes [{primes, depths, points, width_numerators,
        start_numerators, width_denominator, power}], binary {f, a, depth, root, vector, states}.
    """
    deadline = time.time() + 0.4 * float(os.environ.get("SCIEL_STAGE_SECONDS", "30"))
    # The first evaluation returns the best known certificate unchanged, so the record enters the population.
    views = [program for program in population() if program["output"] is not None]
    if not views:
        return asset("frontier").certificate()
    # Afterwards move the widths of one code (a random handful, log-normal steps), lay the code out again along its
    # square-difference order, keep the best capacity found before the deadline and set every power for it.
    cert = sarkozy.load(next((p for p in views if p["parent"]), views[0])["output"])
    rng = np.random.default_rng(int(time.time() * 1e6) % 2**32)
    best, best_alpha = None, None
    while best is None or time.time() < deadline:
        trial = dict(cert, codes=list(cert["codes"]))
        k = int(rng.integers(len(trial["codes"])))
        code = trial["codes"][k]
        w = np.asarray(code["width_numerators"], dtype=np.float64)
        idx = rng.choice(w.size, size=int(rng.integers(3, 30)), replace=False)
        w[idx] *= np.exp(10 ** rng.uniform(-4, -2) * rng.standard_normal(idx.size))
        trial["codes"][k] = sarkozy.relayout(code, w)
        alpha = sarkozy.capacity(trial)
        if best is None or alpha > best_alpha:
            best, best_alpha = trial, alpha
            cert = trial
    # The jump is returned even when its alpha is below the start's: a distinct output is admitted.
    return sarkozy.finalize(best)


# EVOLVE-BLOCK-END
