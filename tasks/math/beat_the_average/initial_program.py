# EVOLVE-BLOCK-START
import numpy as np


def beat_the_average_measure() -> tuple[np.ndarray, np.ndarray]:
    """
    Beat-the-average game. Construct a discrete probability measure mu on [0, inf)
    that maximizes P[X1 + X2 + X3 < 2*X4], where X1, X2, X3, X4 are i.i.d. with law mu.
    The goal is to push the achievable lower bound on the constant
    C = sup_mu P[X1 + X2 + X3 < 2*X4] as high as possible.

    Baseline (from the upstream notebook): the uniform measure on {0, 1, ..., n-1}.
    Here n = 32.

    Returns:
        support: np.ndarray of shape (n,), non-negative support points of the measure.
        weights: np.ndarray of shape (n,), non-negative weights (need not be normalized;
            the evaluator normalizes them to sum to 1).
    """
    n = 32
    support = np.arange(n, dtype=np.float64)
    weights = np.ones(n, dtype=np.float64) / n
    return support, weights


# EVOLVE-BLOCK-END
