# EVOLVE-BLOCK-START
import os
import time

import numpy as np
from sciel.population import population
from sciel.task import asset

trace = asset("trace")
polys = asset("polys")


def certificate() -> tuple:
    """
    Construct a certificate (A, b, d) whose bound int x dmu / mu(R) on the Schur-Siegel-Smyth trace constant is as
    small as possible.

    Returns:
        (A, b, d): integer polynomials as coefficient lists with the constant term first, their weights b_Q >= 0, and
        the 6N densities of the cells [i/N, (i+1)/N) on [0, 6], as exact rationals.
    """
    budget = 0.5 * float(os.environ.get("SCIEL_STAGE_SECONDS", "30"))
    deadline = time.time() + budget
    # The first evaluation returns Pinho's certificate unchanged, so the record enters the population.
    views = [program for program in population() if program["output"] is not None]
    if not views:
        return asset("pinho").certificate()
    # Afterwards jump from the parent's certificate (or the view's best): add one of the nearly active polynomials,
    # remove one, or keep A, and re-solve the linear program on another grid whose solve fits half the stage time
    # (about 6 s at N = 50, 30 s at N = 100, 4 min at N = 200), on the support widened by two cells. The new
    # certificate is returned even when its bound is above the start's: a distinct output is admitted. When a solve
    # fails, the grid halves and the polynomials jump again.
    cert = trace.load(next((p for p in views if p["parent"]), views[0])["output"])
    rng = np.random.default_rng(int(time.time() * 1e6) % 2**32)
    A, _, d = cert
    N = len(d) // 6
    live = [i for i, x in enumerate(d) if x > 0]
    window = (max(live[0] - 2, 1) / N, min(live[-1] + 3, 6 * N) / N)
    grids = [n for n in (40, 50, 60, 80, 100, 120, 160, 200, 240, 320) if 6 * (n / 50) ** 2.2 < 0.6 * budget]
    M = int(rng.choice([n for n in grids if n != N] or [40]))
    while True:
        move = rng.integers(3)
        if move == 0:
            pool = [q for q, _ in polys.candidates(cert, max_degree=5)[:8]]
            jumped = A + [pool[int(rng.integers(len(pool)))]] if pool and len(A) < 40 else A
        elif move == 1 and len(A) > 1:
            k = int(rng.integers(len(A)))
            jumped = A[:k] + A[k + 1:]
        else:
            jumped = A
        if jumped == A and M == N:
            continue
        try:
            new = trace.lp(jumped, M, window=window, deadline=max(deadline, time.time() + 10 * (M / 50) ** 2.2))
            trace.check(new)
            return new
        except (RuntimeError, ValueError):
            M = max(M // 2, 20)


# EVOLVE-BLOCK-END
