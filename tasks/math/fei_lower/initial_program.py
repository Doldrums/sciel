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
    # Afterwards jump out of the basin of the parent's function (or the view's best): swap a chunk of TRUE and FALSE
    # entries across edges of the cube, which keeps f balanced, then climb from there.
    n, bits = fei.load(next((p for p in views if p["parent"]), views[0])["output"])
    bits = np.asarray(bits, dtype=np.uint8).copy()
    rng = np.random.default_rng(int(time.time() * 1e6) % 2**32)
    a = fei.spectrum(n, bits)
    for _ in range(int(rng.integers(8, 65))):
        x = int(rng.integers(1 << n))
        y = x ^ (1 << int(rng.integers(n)))
        if bits[x] != bits[y]:
            candidate = fei.swap(n, a, bits, x, y)
            if fei.value(n, candidate) > float("-inf"):
                a = candidate
                bits[[x, y]] ^= 1
    current = fei.value(n, a)
    # Swap a TRUE and a FALSE entry, half the time across an edge of the cube, otherwise anywhere, and keep the swap
    # when the bound rises. The climbed jump is returned even below the start: a distinct output is admitted.
    while time.time() < deadline:
        x = int(rng.integers(1 << n))
        y = x ^ (1 << int(rng.integers(n))) if rng.random() < 0.5 else int(rng.integers(1 << n))
        if bits[x] == bits[y]:
            continue
        candidate = fei.swap(n, a, bits, x, y)
        if fei.value(n, candidate) > current:
            bits[[x, y]] ^= 1
            a, current = candidate, fei.value(n, candidate)
    return n, bits


# EVOLVE-BLOCK-END
