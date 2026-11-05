# EVOLVE-BLOCK-START
import os
import time

import numpy as np
from sciel.population import population
from sciel.task import asset

doche = asset("doche")


def certificate():
    """
    Construct integer polynomials R_j (leading coefficient +-1) and S_i with multiplicities m_j and exponents q_i for
    which Doche's limit-point bound B on the essential minimum C_82 of the Zhang-Zagier height is as small as possible.

    Returns:
        {"denominators": [{"coeffs": [...], "multiplicity": m}, ...], "numerators": [{"coeffs": [...], "exponent": q},
        ...]}: coefficients from the leading one down, m positive integers, q an int or [numerator, denominator] >= 0,
        sum q_i deg S_i < sum m_j deg R_j.
    """
    deadline = time.time() + 0.4 * float(os.environ.get("SCIEL_STAGE_SECONDS", "30"))
    # The first evaluation returns the best known construction unchanged, so the record enters the population.
    views = [program for program in population() if program["output"] is not None]
    if not views:
        return asset("frontier").certificate()
    # Afterwards jump out of the parent's basin (or the view's best): add one to three lattice neighbours of its
    # numerators (one coefficient moved by +-1) or drop one numerator, then re-solve the joint LP for multiplicities
    # and exponents on the factors that survive.
    start = doche.load(next((p for p in views if p["parent"]), views[0])["output"])
    den = [list(r["coeffs"]) for r in start["denominators"]]
    num = [list(s["coeffs"]) for s in start["numerators"]]
    rng = np.random.default_rng(int(time.time() * 1e6) % 2**32)
    if rng.random() < 0.3 and len(num) > 2:
        num.pop(int(rng.integers(len(num))))
    else:
        added = 0
        while added < int(rng.integers(1, 4)) and time.time() < deadline:
            p = list(num[int(rng.integers(len(num)))])
            p[int(rng.integers(1, len(p)))] += int(rng.choice([-1, 1]))
            if p in num or p in den:
                continue
            try:
                doche.roots(p)
            except ValueError:
                continue
            num.append(p)
            added += 1
    # The re-solved jump is returned even when its bound is above the start's: a distinct output is admitted.
    return doche.resolve(den, num, grid=2 ** 17)


# EVOLVE-BLOCK-END
