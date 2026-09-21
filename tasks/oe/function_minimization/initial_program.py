# EVOLVE-BLOCK-START
"""Function minimization example for OpenEvolve"""

import numpy as np


def search_algorithm(iterations=1000, bounds=(-5, 5)):
    """
    A simple random search algorithm that often gets stuck in local minima.

    Args:
        iterations: Number of iterations to run
        bounds: Bounds for the search space (min, max)

    Returns:
        Tuple of (best_x, best_y, best_value)
    """
    # Initialize with a random point
    best_x = np.random.uniform(bounds[0], bounds[1])
    best_y = np.random.uniform(bounds[0], bounds[1])
    best_value = evaluate_function(best_x, best_y)

    for _ in range(iterations):
        # Simple random search
        x = np.random.uniform(bounds[0], bounds[1])
        y = np.random.uniform(bounds[0], bounds[1])
        value = evaluate_function(x, y)

        if value < best_value:
            best_value = value
            best_x, best_y = x, y

    return best_x, best_y, best_value


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved).
#
# Hard objective-evaluation budget. The evolvable algorithm above may only
# query the objective MAX_EVALS times per run_search(); further calls return a
# large penalty. This is enforced here, OUTSIDE the EVOLVE-BLOCK, so an evolved
# program cannot lift the budget by rewriting its own loop. With brute-force
# random search no longer viable, scoring becomes discriminating: a
# sample-efficient optimizer (local refinement, restarts, annealing, surrogate)
# scores well, a naive one does not. Tune MAX_EVALS to set difficulty.
MAX_EVALS = 50
_OVER_BUDGET_PENALTY = 1e9
_eval_count = 0


def _objective(x, y):
    """The complex function we're trying to minimize (true, unbudgeted)."""
    return np.sin(x) * np.cos(y) + np.sin(x * y) + (x**2 + y**2) / 20


def evaluate_function(x, y):
    """Budgeted objective: real value for the first MAX_EVALS calls of a run,
    then a large penalty so extra sampling cannot improve the result."""
    global _eval_count
    _eval_count += 1
    if _eval_count > MAX_EVALS:
        return _OVER_BUDGET_PENALTY
    return _objective(x, y)


def run_search():
    global _eval_count
    _eval_count = 0  # fresh budget per evaluation trial
    x, y, value = search_algorithm()
    return x, y, value


if __name__ == "__main__":
    x, y, value = run_search()
    print(f"Found minimum at ({x}, {y}) with value {value}")
