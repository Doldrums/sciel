# EVOLVE-BLOCK-START
import os
import time

import numpy as np
from sciel.population import population
from sciel.task import asset

sf = asset("sunflower")


def block() -> dict:
    """
    Construct a Sperner, 3-sunflower-free block with ports p, q whose chain rate rho^(1/(n - 2)) is as large as
    possible, rho the Perron root of [[N10, N11], [N00, N01]].

    Returns:
        {"nodes": [...], "ports": [p, q]}: leaves {"n", "sets"} and glue nodes {"glue": [i, a, j, b], "sets"}, sets
        as integer bitmasks; the block is the last node.
    """
    deadline = time.time() + 0.3 * float(os.environ.get("SCIEL_STAGE_SECONDS", "300"))
    # The first evaluation returns the best known block unchanged, so the record enters the population.
    views = [program for program in population() if program["output"] is not None]
    if not views:
        return asset("known").record()
    # Afterwards jump: drop a few of the top node's explicit sets (the rate falls) and bar them, then re-add sets
    # greedily. Candidates are members of the block with one or two points flipped; every new set raises a port count.
    b = sf.load(next((p for p in views if p["parent"]), views[0])["output"])
    top = b["nodes"][-1]
    n, fam = sf.members(b)
    rng = np.random.default_rng(int(time.time() * 1e6) % 2**32)
    sets = list(top["sets"])
    drop = set(rng.choice(sets, size=min(len(sets), int(rng.integers(1, 6))), replace=False).tolist()) if sets else set()
    keep = [E for E in sets if E not in drop]
    fam = fam[~np.isin(fam, list(drop))]
    tried = set(drop)
    while time.time() < deadline:
        E = int(fam[rng.integers(fam.size)])
        for x in rng.choice(n, size=int(rng.integers(1, 3)), replace=False):
            E ^= 1 << int(x)
        if E in tried or E == 0:
            continue
        tried.add(E)
        if sf.addable(fam, n, E):
            keep.append(E)
            fam = np.append(fam, E)
    # The re-optimized jump is returned even when its rate is below the start's: a distinct output is admitted.
    nodes = b["nodes"][:-1] + [{**top, "sets": keep}]
    return {"nodes": nodes, "ports": b["ports"]}


# EVOLVE-BLOCK-END
