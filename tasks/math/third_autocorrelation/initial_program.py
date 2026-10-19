# EVOLVE-BLOCK-START
import numpy as np


def step_heights() -> np.ndarray:
    """
    Construct a step function with n equal steps on [-1/4, 1/4], possibly negative, whose bound
    |2n max(convolve(h, h))| / sum(h)^2 on the third autocorrelation constant is as small as possible.

    Returns:
        heights: 1-D np.ndarray of at most 65536 step heights.
    """
    return np.full(400, 1.0)


# EVOLVE-BLOCK-END
