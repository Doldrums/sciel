# EVOLVE-BLOCK-START
import os
import time

import numpy as np
from sciel.population import population
from sciel.task import asset

hl = asset("hl")


def candidate():
    """
    Construct a periodic atomic measure (a cell [0, Px) x [0, Py) of atoms repeated over the lattice) whose exact
    weak-type ratio for the centred square maximal function, a lower bound for C_47, is as large as possible.

    Returns:
        {"Px": Px, "Py": Py, "atoms": [[x, y, m], ...]}: at most 80 distinct atoms in the cell, masses m > 0, periods in
        [1/2, 8]; numbers are ints, binary64 (read exactly), "p/q" strings or [p, q] pairs.
    """
    deadline = time.time() + 0.4 * float(os.environ.get("SCIEL_STAGE_SECONDS", "30"))
    # The first evaluation returns the best known construction unchanged, so the record enters the population.
    views = [program for program in population() if program["output"] is not None]
    if not views:
        return asset("frontier").candidate()
    # Afterwards jump out of the parent's basin (or the view's best): insert a light atom in an uncovered hole, move
    # one atom or drop a light one, then re-optimize by single-atom moves scored by the float ratio.
    start = hl.load(next((p for p in views if p["parent"]), views[0])["output"])
    px, py, atoms = hl.floats(start)
    rng = np.random.default_rng(int(time.time() * 1e6) % 2**32)
    move = rng.random()
    holes = hl.holes(start, 300) if move < 0.5 else []
    if len(holes) and len(atoms) < hl.MAX_ATOMS:
        x, y = holes[int(rng.integers(len(holes)))]
        atoms = np.vstack([atoms, [x, y, 10 ** rng.uniform(-4, -2)]])
    elif move < 0.8 or len(atoms) < 3:
        k = int(rng.integers(len(atoms)))
        atoms[k, :2] += rng.normal(0, 0.01, 2)
    else:
        light = np.argsort(atoms[:, 2])[: max(1, len(atoms) // 4)]
        atoms = np.delete(atoms, int(rng.choice(light)), axis=0)
    atoms[:, 0] %= px
    atoms[:, 1] %= py
    best = hl.value(hl.make(px, py, atoms.tolist()))
    while time.time() < deadline:
        trial = atoms.copy()
        k = int(rng.integers(len(trial)))
        trial[k, :2] = (trial[k, :2] + rng.normal(0, 1e-3, 2)) % (px, py)
        trial[k, 2] *= np.exp(rng.normal(0, 0.05))
        try:
            v = hl.value(hl.make(px, py, trial.tolist()))
        except ValueError:
            continue
        if v > best:
            atoms, best = trial, v
    # The re-optimized jump is returned even when its ratio is below the start's: a distinct output is admitted.
    return hl.make(px, py, atoms.tolist())


# EVOLVE-BLOCK-END
