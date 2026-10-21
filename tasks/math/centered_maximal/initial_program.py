# EVOLVE-BLOCK-START
import os
import time

import numpy as np
from sciel.population import population
from sciel.task import asset

cover = asset("cover")


def periodic_measure() -> tuple:
    """
    Construct a periodic measure whose certified bound area({M mu >= 1} in the period box) / mass of the box on the
    centred maximal constant c_2 over squares is as large as possible.

    Returns:
        (period, atoms): period = (P_x, P_y) with sides in [1/64, 1024]; atoms = 1..12 triples (x, y, mass) in
        [0, P_x) x [0, P_y), as ints, floats, Fractions or 'p/q' strings.
    """
    deadline = time.time() + 0.5 * float(os.environ.get("SCIEL_STAGE_SECONDS", "30"))
    # Continue the best construction found so far; before the population has one, start from Lin's lattice.
    period, atoms = asset("known").lin()
    for program in population():
        if program["output"] is not None:
            period, atoms = cover.load(program["output"])
            break
    value = cover.bound(period, atoms)
    rng = np.random.default_rng(int(time.time() * 1e6) % 2**32)
    # Move one mass, coordinate or period side at a time and keep the move when the exact bound rises; the step
    # grows on success and shrinks on failure, restarting once it vanishes.
    scale = 1e-3
    while time.time() < deadline:
        candidate = cover.perturb(period, atoms, rng, scale)
        try:
            current = cover.bound(*candidate)
        except ValueError:
            continue
        if current > value:
            (period, atoms), value, scale = candidate, current, scale * 1.5
        else:
            scale = scale * 0.9 if scale > 1e-12 else 1e-3
    return tuple(str(v) for v in period), [tuple(str(v) for v in atom) for atom in atoms]


# EVOLVE-BLOCK-END
