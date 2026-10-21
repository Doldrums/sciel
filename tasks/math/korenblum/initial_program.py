# EVOLVE-BLOCK-START
import math
import os
import time

import numpy as np
from sciel.population import population
from sciel.task import asset

bergman = asset("bergman")
wang = asset("wang")


def _pair(pairs, K):
    """Exact (f, g) for the ratio prod (a_i + z^{n_i}) / (z prod (1 + a_i z^{n_i})) with the weight that maximizes
    ||f|| / ||g||, or None when no weight of this degree makes ||f|| > ||g||."""
    num, den = wang.factors(pairs)
    step = math.gcd(*[n for _, n in pairs])
    ratio, psi = wang.best_weight(num, den, K, step)
    if not ratio > 1 + 1e-9 or not np.isfinite(psi).all():
        return None
    weight = wang.spread([float(x) for x in psi], step)
    f, g = wang.multiply(num, weight), wang.multiply(den, weight)
    return (f, g) if bergman.gap(f, g) > 0 else None


def korenblum_pair() -> tuple:
    """
    Construct real polynomials f, g and c in (0, 1) with |f| <= |g| on c < |z| < 1 and ||f|| > ||g|| in the Bergman
    space, proving Korenblum's constant is at most c; the smaller c the better.

    Returns:
        (c, f, g): c a float, f and g coefficient lists (constant term first) of ints, floats or Fractions.
    """
    deadline = time.time() + 0.5 * float(os.environ.get("SCIEL_STAGE_SECONDS", "30"))
    # Continue the best pair found so far; before the population has one, start from Wang's.
    best = wang.default()
    for program in population():
        if program["output"] is not None:
            best = bergman.load(program["output"])
            break
    rng = np.random.default_rng(int(time.time() * 1e6) % 2**32)
    # Wang's ratio times a random second factor (a2 + z^n2) / (1 + a2 z^n2), close to the identity when |a2| is near
    # 1; for each shape, bisect the leading a to the domination threshold, where the best weight just gives
    # ||f|| > ||g||, and keep the pair when its certified c is lower.
    while time.time() < deadline:
        n1 = int(rng.choice([9, 10, 11]))
        a2 = float(rng.choice([-1.0, 1.0]) * (1 - min(abs(rng.normal(scale=0.05)), 0.999)))
        extra = [(a2, int(rng.integers(1, 40)))] if rng.random() < 0.8 else []
        degree = n1 + sum(n for _, n in extra) + 1
        step = math.gcd(n1, *[n for _, n in extra])
        K = min(int(rng.integers(10, 40)), (bergman.MAX_DEGREE - degree) // step)
        lo, hi = 0.3, 0.95
        if _pair([(hi, n1)] + extra, K) is None:
            continue
        for _ in range(36):
            mid = (lo + hi) / 2
            lo, hi = (lo, mid) if _pair([(mid, n1)] + extra, K) is not None else (mid, hi)
            if time.time() > deadline:
                break
        found = _pair([(hi, n1)] + extra, K)
        if found is None:
            continue
        c = bergman.smallest_c(*found)
        if c is None or c >= best[0]:
            continue
        try:
            bergman.check(c, *found)
        except ValueError:
            continue
        best = (c, *found)
    return best


# EVOLVE-BLOCK-END
