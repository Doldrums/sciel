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
    # Continue the best construction found so far; the search only has to improve on it. Before the population has
    # one, start from the best known function.
    n, bits = asset("known").record()
    for program in population():
        if program["output"] is not None:
            n, bits = fei.load(program["output"])
            break
    bits = np.asarray(bits, dtype=np.uint8).copy()
    a = fei.spectrum(n, bits)
    best, current = fei.exact_value(n, bits), fei.value(n, a)
    # Swap a TRUE and a FALSE entry, which keeps f balanced: half the time across an edge of the cube, otherwise
    # anywhere. A swap is kept only when its certified bound beats the current one.
    rng = np.random.default_rng(0)
    while time.time() < deadline:
        x = int(rng.integers(1 << n))
        y = x ^ (1 << int(rng.integers(n))) if rng.random() < 0.5 else int(rng.integers(1 << n))
        if bits[x] == bits[y]:
            continue
        candidate = fei.swap(n, a, bits, x, y)
        if fei.value(n, candidate) <= current:
            continue
        trial = bits.copy()
        trial[[x, y]] ^= 1
        certified = fei.exact_value(n, trial)
        if certified > best:
            bits, a, best, current = trial, candidate, certified, fei.value(n, candidate)
    return n, bits


# EVOLVE-BLOCK-END
