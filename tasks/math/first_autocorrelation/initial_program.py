# EVOLVE-BLOCK-START
import numpy as np


def step_heights() -> np.ndarray:
    """
    Construct a non-negative step function with n equal steps on [-1/4, 1/4] whose bound
    2n max(convolve(h, h)) / sum(h)^2 on the first autocorrelation constant is as small as possible.

    Returns:
        heights: 1-D np.ndarray of at most 65536 step heights.
    """
    return np.full(600, 1.0)


# EVOLVE-BLOCK-END
