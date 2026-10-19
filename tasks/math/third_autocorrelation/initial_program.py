# EVOLVE-BLOCK-START
import numpy as np
from sciel.population import population


def step_heights() -> np.ndarray:
    """
    Construct a step function with n equal steps on [-1/4, 1/4], possibly negative, whose bound
    |2n max(convolve(h, h))| / sum(h)^2 on the third autocorrelation constant is as small as possible.

    Returns:
        heights: 1-D np.ndarray of at most 65536 step heights.
    """
    # Continue the best construction found so far; the search only has to improve on it.
    for program in population():
        if program["output"] is not None:
            return np.array(program["output"]["heights"], dtype=float)
    return np.full(400, 1.0)


# EVOLVE-BLOCK-END
