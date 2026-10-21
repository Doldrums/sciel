# EVOLVE-BLOCK-START
import os
import time

from sciel.population import population
from sciel.task import asset

trace = asset("trace")


def certificate() -> tuple:
    """
    Construct a certificate (A, b, d) whose bound int x dmu / mu(R) on the Schur-Siegel-Smyth trace constant is as
    small as possible.

    Returns:
        (A, b, d): integer polynomials as coefficient lists with the constant term first, their weights b_Q >= 0, and
        the 6N densities of the cells [i/N, (i+1)/N) on [0, 6], as exact rationals.
    """
    deadline = time.time() + 0.5 * float(os.environ.get("SCIEL_STAGE_SECONDS", "30"))
    # Continue the best certificate found so far; the search only has to improve on it. Before the population has one,
    # start from Pinho's.
    cert = asset("pinho").certificate()
    for program in population():
        if program["output"] is not None:
            cert = trace.load(program["output"])
            break
    # Re-solve the linear program for the same polynomials and grid, on the support widened by two cells, and keep the
    # result only when the evaluator's own check proves it with a lower bound.
    A, b, d = cert
    N = len(d) // 6
    live = [i for i, x in enumerate(d) if x > 0]
    window = (max(live[0] - 2, 1) / N, min(live[-1] + 3, 6 * N) / N)
    try:
        new = trace.lp(A, N, window=window, deadline=deadline)
        if trace.bound(new) < trace.bound(cert):
            trace.check(new)
            return new
    except (RuntimeError, ValueError):
        pass
    return cert


# EVOLVE-BLOCK-END
