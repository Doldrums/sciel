# EVOLVE-BLOCK-START
import os
import time

import numpy as np
from scipy.optimize import minimize
from sciel.population import population
from sciel.task import asset

helix = asset("helix")


def candidate():
    """
    Construct a four-strand superhelix (two coaxial axes of radius R and rise h per radian, each with an outer strand
    of radius a(u) and an inner strand of radius i(u) on the opposite side, N = P / Q twists per axis turn) whose
    length per crossing over its thickness, an upper bound on C_22b, is as small as possible.

    Returns:
        {"P": int, "Q": int, "R": x, "h": x, "outer": [a0, a1, b1, ...], "inner": [i0, ...]}: Fourier coefficients
        of the radii in u = N t, order <= 4.
    """
    deadline = time.time() + 0.4 * float(os.environ.get("SCIEL_STAGE_SECONDS", "30"))
    # The first evaluation returns the best known construction unchanged, so the record enters the population.
    views = [program for program in population() if program["output"] is not None]
    if not views:
        return asset("frontier").candidate()
    # Afterwards jump out of the parent's basin (or the view's best): move N to a nearby fraction, or add a random
    # first or second harmonic to one radius, then re-optimize the continuous parameters on the float estimate.
    start = helix.load(next((p for p in views if p["parent"]), views[0])["output"])
    rng = np.random.default_rng(int(time.time() * 1e6) % 2**32)
    c = dict(start, outer=list(start["outer"]), inner=list(start["inner"]))
    if rng.random() < 0.5:
        while (c["P"], c["Q"]) == (start["P"], start["Q"]) or np.gcd(c["P"], c["Q"]) != 1:
            c["Q"] = int(np.clip(start["Q"] + rng.integers(-2, 3), 1, helix.MAX_Q))
            c["P"] = int(round(start["P"] / start["Q"] * c["Q"] * (1 + rng.normal(0, 0.02))))
    else:
        key = ["outer", "inner"][int(rng.integers(2))]
        r = c[key] + [0.0] * max(0, 5 - len(c[key]))
        r[int(rng.integers(1, 5))] += rng.normal(0, 0.003) * r[0]
        c[key] = r
    no, ni = len(c["outer"]), len(c["inner"])

    def unpack(x):
        return helix.make(c["P"], c["Q"], x[0], x[1], x[2:2 + no], x[2 + no:])

    def f(x):
        if time.time() > deadline:
            raise StopIteration
        try:
            return helix.value(unpack(x))
        except ValueError:
            return 1e3

    x0 = np.r_[c["R"], c["h"], c["outer"], c["inner"]]
    best = [f(x0) if time.time() < deadline else 1e3, x0]

    def track(x):
        v = f(x)
        if v < best[0]:
            best[:] = [v, np.array(x)]
        return v

    try:
        minimize(track, x0, method="Nelder-Mead", options={"maxfev": 4000, "xatol": 1e-10, "fatol": 1e-11})
    except StopIteration:
        pass
    # The re-optimized jump is returned even when its bound is above the start's: a distinct output is admitted.
    return unpack(best[1])


# EVOLVE-BLOCK-END
