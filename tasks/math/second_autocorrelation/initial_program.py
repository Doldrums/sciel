# EVOLVE-BLOCK-START
import numpy as np


def step_heights() -> np.ndarray:
    """
    Construct a non-negative step function with n equal steps whose ratio
    ||f*f||_2^2 / (||f*f||_1 ||f*f||_inf) is as large as possible.

    Returns:
        heights: 1-D np.ndarray of at most 65536 step heights.
    """
    return np.full(50, 1.0)


# EVOLVE-BLOCK-END
