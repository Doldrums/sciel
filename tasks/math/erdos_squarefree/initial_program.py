# EVOLVE-BLOCK-START
import numpy as np


def erdos_squarefree500() -> np.ndarray:
    """
    Erdos squarefree problem. For N = 500, find a subset A of {1, ..., N} of maximum
    size such that a*b + 1 is NOT square-free for every pair a, b in A (a square-free
    integer is one not divisible by any perfect square > 1). Larger feasible subsets
    are better; pairs whose a*b + 1 is square-free are penalized as violations.

    Baseline construction: the residue class 7 (mod 25). If a, b == 7 (mod 25) then
    a*b == 49 == -1 (mod 25), so a*b + 1 == 0 (mod 25) is divisible by 5^2 and is never
    square-free. This yields zero violations for any pair drawn from the class.

    Returns:
        subset: 1-D np.ndarray of distinct integers in {1, ..., 500} forming the subset A.
    """
    N = 500
    candidates = np.arange(1, N + 1)
    subset = candidates[candidates % 25 == 7]
    return subset


# EVOLVE-BLOCK-END
