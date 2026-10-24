# EVOLVE-BLOCK-START
import os
import time

import numpy as np
from sciel.population import population
from sciel.task import asset

sis = asset("sis")
squeeze = asset("squeeze")


def packings() -> dict:
    """
    For every n in sis.SLICE, n unit squares (centre x, centre y, rotation angle in radians) with disjoint interiors
    whose bounding square side L = max(max_x - min_x, max_y - min_y) over all corners is as small as possible.

    Returns:
        packings: dict {n: (n, 3) array or list of [x, y, angle]}.
    """
    deadline = time.time() + 0.5 * float(os.environ.get("SCIEL_STAGE_SECONDS", "30"))
    # The first evaluation returns the merged public packings unchanged, so the bar enters the population.
    views = [program for program in population() if program["output"] is not None]
    if not views:
        return {n: sis.seed(n).tolist() for n in sis.SLICE}
    # Afterwards jump one or two n of the parent's packings (or the view's best) out of their basins: sometimes start
    # from the public packing of a slightly larger m with the extra squares dropped, then move a group of squares
    # around one on the box by a fraction of a side, repair the overlaps locally and squeeze the whole packing for an
    # equal share of half the stage time. A jumped n is returned even when its side grows, since a distinct output is
    # admitted; every other n is returned as it was.
    output = next((p for p in views if p["parent"]), views[0])["output"]
    best = {int(n): sis.decode(p) for n, p in output["packings"].items()}
    rng = np.random.default_rng(int(time.time() * 1e6) % 2**32)
    chosen = rng.choice(sis.SLICE, size=int(rng.integers(1, 3)), replace=False)
    for k, n in enumerate(int(n) for n in chosen):
        until = time.time() + (deadline - time.time()) / (len(chosen) - k)
        p = best[n]
        if rng.random() < 0.3:
            m = n + int(rng.integers(1, 4))
            p = sis.remove(sis.best(m), m - n)
        held = squeeze.boundary(p)
        centre = int(rng.choice(held)) if len(held) else int(rng.integers(n))
        moved = squeeze.group(p, centre, int(rng.integers(4, 13)))
        p = squeeze.jiggle(p, moved, rng.uniform(0.02, 0.1), rng, rng.uniform(0.01, 0.05))
        local = squeeze.group(p, centre, min(n, 60))
        p = squeeze.squeeze(p, rounds=50, step=0.02, deadline=until, free=local)
        p = squeeze.squeeze(p, rounds=10**6, step=0.01, deadline=until)
        # Then hop for the rest of this n's share: shake a small group on the box, repair and squeeze, keeping a
        # shake only when it is apart and its side falls.
        while time.time() < until:
            held = squeeze.boundary(p)
            centre = int(rng.choice(held)) if len(held) else int(rng.integers(n))
            q = squeeze.jiggle(p, squeeze.group(p, centre, int(rng.integers(2, 7))), 10 ** rng.uniform(-3, -1.5),
                               rng, 10 ** rng.uniform(-3, -1.5))
            q = squeeze.squeeze(q, rounds=30, step=0.01, deadline=until, free=squeeze.group(q, centre, min(n, 40)))
            q = squeeze.squeeze(q, rounds=10**6, step=0.005, deadline=until)
            if sis.min_gap(q) > 0 and sis.side(q) < sis.side(p):
                p = q
        best[n] = p
    return {n: p.tolist() for n, p in best.items()}


# EVOLVE-BLOCK-END
