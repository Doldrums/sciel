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
    # The first evaluation returns Lin's lattice unchanged, so the record enters the population.
    views = [program for program in population() if program["output"] is not None]
    if not views:
        period, atoms = asset("known").lin()
        return tuple(str(v) for v in period), [tuple(str(v) for v in atom) for atom in atoms]
    # Afterwards jump out of the basin of the parent's measure (or the view's best): move every number by a few
    # percent, or split an atom into two of half its mass a few percent of the period apart, then climb from there.
    start = cover.load(next((p for p in views if p["parent"]), views[0])["output"])
    rng = np.random.default_rng(int(time.time() * 1e6) % 2**32)
    while True:
        period, atoms = start
        if rng.random() < 0.5 or len(atoms) >= cover.MAX_ATOMS:
            s = rng.uniform(0.01, 0.05)
            period = [float(v) * np.exp(s * rng.normal()) for v in period]
            atoms = [(float(x) + s * period[0] * rng.normal(), float(y) + s * period[1] * rng.normal(),
                      float(m) * np.exp(s * rng.normal())) for x, y, m in atoms]
        else:
            atoms = list(atoms)
            x, y, m = atoms.pop(int(rng.integers(len(atoms))))
            for sign in (1, -1):
                atoms.append((float(x) + sign * 0.03 * float(period[0]) * rng.normal(),
                              float(y) + sign * 0.03 * float(period[1]) * rng.normal(), float(m) / 2))
        period, atoms = cover.clean(period, atoms)
        try:
            value = cover.bound(period, atoms)
            break
        except ValueError:
            continue
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
    # The climbed jump is returned even below the start: a distinct output is admitted.
    return tuple(str(v) for v in period), [tuple(str(v) for v in atom) for atom in atoms]


# EVOLVE-BLOCK-END
