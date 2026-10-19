# EVOLVE-BLOCK-START
import numpy as np


def step_heights() -> np.ndarray:
    """
    Construct a step function on [0, 2] with n equal steps, values in [0, 1] and sum(h) exactly n/2,
    whose bound (2/n) max(correlate(h, 1 - h, 'full')) on the minimum overlap constant is as small as possible.

    Returns:
        heights: 1-D np.ndarray of at most 10000 step heights.
    """
    return np.full(600, 0.5)


# EVOLVE-BLOCK-END
