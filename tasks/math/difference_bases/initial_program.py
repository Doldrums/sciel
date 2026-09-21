# EVOLVE-BLOCK-START
import numpy as np


def difference_basis() -> np.ndarray:
    """
    Construct a Sidon set (B_2 set): a set of distinct non-negative integers, all in the
    range [0, 60], such that all pairwise differences |a_i - a_j| (i < j) are distinct
    (equivalently, all pairwise sums are distinct). Maximize the number of elements.

    Returns:
        basis: 1-D np.ndarray of distinct non-negative integers in [0, 60].
    """
    basis = np.array([0, 1, 3, 7, 12, 20])
    return basis


# EVOLVE-BLOCK-END
