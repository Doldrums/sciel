# EVOLVE-BLOCK-START
import math
import os
import time

import numpy as np
from sciel.population import population
from sciel.task import asset

sis = asset("sis")
squeeze = asset("squeeze")


def _angles(p):
    """The two most common tilts of a packing, in radians, beside 45 degrees."""
    a = np.round(np.abs(sis.normalize(p)[:, 2]), 3)
    a = a[a > 1e-3]
    values, counts = np.unique(a, return_counts=True)
    return sorted({math.pi / 4, *(float(v) for v in values[np.argsort(-counts)][:2])})


def _build(n, p, until, rng):
    """Search the parameters of the construction families for the smallest L, then squeeze the best construction."""
    tilts = _angles(p) + [float(rng.uniform(0.25, 0.75))]
    best, best_side = None, math.inf
    search = time.time() + 0.6 * (until - time.time())
    while time.time() < search or best is None and time.time() < until:
        family = "band" if rng.random() < 0.7 else "corners"
        width = rng.uniform(0.8, 0.45 * math.sqrt(n)) if family == "band" else rng.uniform(0.0, 1.3 * math.sqrt(n))
        q, side = sis.construct(n, family, float(rng.choice(tilts)), float(width), float(rng.normal(scale=0.5)),
                                (rng.uniform(0, 0.25), rng.uniform(0.75, 1.0)), tuple(rng.uniform(size=2)), tol=1e-3)
        if q is not None and side < best_side:
            best, best_side = q, side
    if best is None:
        return None
    return squeeze.squeeze(best, rounds=10**6, step=0.02, deadline=until)


def _hop(p, until, rng):
    """Shake a small group on the box, repair and squeeze; keep a shake only when it is apart and its side falls."""
    while time.time() < until:
        held = squeeze.boundary(p)
        centre = int(rng.choice(held)) if len(held) else int(rng.integers(len(p)))
        q = squeeze.jiggle(p, squeeze.group(p, centre, int(rng.integers(2, 7))), 10 ** rng.uniform(-3, -1.5),
                           rng, 10 ** rng.uniform(-3, -1.5))
        q = squeeze.squeeze(q, rounds=30, step=0.01, deadline=until, free=squeeze.group(q, centre, min(len(p), 40)))
        q = squeeze.squeeze(q, rounds=10**6, step=0.005, deadline=until)
        if sis.min_gap(q) > 0 and sis.side(q) < sis.side(p):
            p = q
    return p


def packings() -> dict:
    """
    For every n in sis.targets(), n unit squares (centre x, centre y, rotation angle in radians) with disjoint
    interiors whose bounding square side L = max(max_x - min_x, max_y - min_y) over all corners is as small as possible.

    Returns:
        packings: dict {n: (n, 3) array or list of [x, y, angle]}.
    """
    deadline = time.time() + 0.8 * float(os.environ.get("SCIEL_STAGE_SECONDS", "75")) - 15
    targets = sis.targets()
    # The first evaluation returns the merged public packings unchanged, so the bar enters the population.
    views = [program for program in population() if program["output"] is not None]
    if not views:
        return {n: sis.seed(n).tolist() for n in targets}
    # Afterwards rebuild one or two n from a construction family (a tilted band along the diagonal between two
    # staircase grids, or tilted blocks in two corners) with searched parameters, squeeze the best construction and
    # keep it when its side is below the parent's; the rest of each n's share hops on the better packing.
    output = next((p for p in views if p["parent"]), views[0])["output"]
    best = {int(n): sis.decode(p) for n, p in output["packings"].items()}
    rng = np.random.default_rng(int(time.time() * 1e6) % 2**32)
    chosen = rng.choice(targets, size=min(len(targets), int(rng.integers(1, 3))), replace=False)
    for k, n in enumerate(int(n) for n in chosen):
        until = time.time() + (deadline - time.time()) / (len(chosen) - k)
        p = best[n]
        q = _build(n, p, time.time() + 0.7 * (until - time.time()), rng)
        if q is not None and sis.min_gap(q) > 0 and sis.side(q) < sis.side(p):
            p = q
        best[n] = _hop(p, until, rng)
    return {n: best[n].tolist() for n in targets}


# EVOLVE-BLOCK-END
