# EVOLVE-BLOCK-START
import os
import time

import numpy as np
from sciel.population import population
from sciel.task import asset

fei = asset("fei")


def boolean_function() -> tuple:
    """
    Construct a balanced Boolean function f on n <= 20 variables with total influence above 1 whose bound
    H / (Inf - 1) on the Fourier entropy-influence constant is as large as possible.

    Returns:
        (n, bits): an integer n and 2^n bits (np.ndarray, list or bytes of 0/1), entry x = 1 where f(x) = -1, with
        bit i of x set when x_{i+1} = -1.
    """
    deadline = time.time() + 0.5 * float(os.environ.get("SCIEL_STAGE_SECONDS", "30"))
    # The first evaluation returns the best known function unchanged, so the record enters the population.
    views = [program for program in population() if program["output"] is not None]
    if not views:
        return asset("known").record()
    # Afterwards jump out of the parent's basin (or the view's best) in the cell picture: a core cell decided by the
    # auxiliaries moves to another boundary cell, two auxiliaries merge, or an auxiliary's cells split onto a new
    # variable.
    n, start = fei.load(next((p for p in views if p["parent"]), views[0])["output"])
    start = np.asarray(start, dtype=np.uint8)
    rng = np.random.default_rng(int(time.time() * 1e6) % 2**32)
    variables = fei.core(n, start)
    aux = [i for i in range(n) if i not in variables]
    bits = start
    while np.array_equal(bits, start):
        kind = rng.random()
        try:
            if kind < 0.5 or len(aux) < 2:
                bits = start
                for _ in range(int(rng.integers(1, 3))):
                    bits = fei.move_cell(n, bits, variables, rng)
            elif kind < 0.75 or n == fei.MAX_N:
                i, j = (int(v) for v in rng.choice(aux, size=2, replace=False))
                m, merged = fei.merge(n, start, i, j)
                k = i if i < j else i - 1
                cells = rng.random(1 << m) < 0.5
                n, bits = fei.rehost(m, merged, k, cells[np.arange(1 << m) & ((1 << len(variables)) - 1)])
            else:
                i = int(rng.choice(aux))
                cells = rng.random(1 << len(variables)) < 0.5
                n, bits = fei.rehost(n, start, i, cells[np.arange(1 << n) & ((1 << len(variables)) - 1)])
                variables = fei.core(n, bits)
        except ValueError:
            bits = start
    bits = bits.copy()
    a = fei.spectrum(n, bits)
    current = fei.value(n, a)
    # Re-optimize: propose further cell moves and balanced swaps across cube edges on the boundary, keeping those
    # that raise the bound. The climbed jump is returned even below the start: a distinct output is admitted.
    while time.time() < deadline:
        if rng.random() < 0.3:
            try:
                trial = fei.move_cell(n, bits, variables, rng)
            except ValueError:
                continue
            candidate = fei.spectrum(n, trial)
            if fei.value(n, candidate) > current:
                bits, a, current = trial, candidate, fei.value(n, candidate)
            continue
        x = int(rng.integers(1 << n))
        y = x ^ (1 << int(rng.integers(n)))
        if bits[x] == bits[y]:
            continue
        candidate = fei.swap(n, a, bits, x, y)
        if fei.value(n, candidate) > current:
            bits[[x, y]] ^= 1
            a, current = candidate, fei.value(n, candidate)
    return n, bits


# EVOLVE-BLOCK-END
