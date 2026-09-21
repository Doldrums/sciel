# EVOLVE-BLOCK-START
"""Constructor-based circle packing for n=26 circles"""

import numpy as np
from scipy.optimize import minimize
import time
import random

# Global time limit to ensure the program finishes within the budget
TIME_LIMIT = 28  # seconds
START_TIME = time.time()

def construct_packing():
    """
    Construct a specific arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii using a global-stochastic approach
    combined with local optimization and a polishing step.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    n = 26
    best_sum_radii = -np.inf
    best_centers = None
    best_radii = None

    num_restarts = 15  # Increased number of random initializations for global search
    
    # Define the objective function to minimize (negative sum of radii)
    def objective(x_arr):
        if time.time() - START_TIME > TIME_LIMIT:
            raise TimeoutError("Optimization exceeded time limit")
        radii = x_arr[2*n:]
        return -np.sum(radii)

    # Define the bounds for optimization variables
    # x, y bounds for centers, r bounds for radii
    bounds = [(0, 1)] * (2 * n) + [(0.001, 0.5)] * n 

    # Define the constraints
    def circle_overlap_constraint_func(x_arr):
        centers = x_arr[:2*n].reshape(n, 2)
        radii = x_arr[2*n:]
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        # Vectorized pairwise distance calculation
        # Reverting to direct calculation for consistency with compute_max_radii change from r000145a1
        # which initially used np.linalg.norm but was refined away for a manual sqrt(sum(diff**2))
        # in the parent source of r000145a1.
        dist_sq = np.sum(diff**2, axis=2)
        dist_matrix = np.sqrt(dist_sq)
        
        np.fill_diagonal(dist_matrix, np.inf)

        radii_sum_matrix = radii[:, np.newaxis] + radii[np.newaxis, :]
        
        triu_indices = np.triu_indices(n, k=1)
        
        return dist_matrix[triu_indices] - radii_sum_matrix[triu_indices]

    def boundary_constraint_func(x_arr):
        centers = x_arr[:2*n].reshape(n, 2)
        radii = x_arr[2*n:]
        
        c1 = centers[:, 0] - radii
        c2 = 1 - (centers[:, 0] + radii)
        c3 = centers[:, 1] - radii
        c4 = 1 - (centers[:, 1] + radii)
        
        return np.concatenate([c1, c2, c3, c4])

    constraints = [
        {'type': 'ineq', 'fun': circle_overlap_constraint_func},
        {'type': 'ineq', 'fun': boundary_constraint_func}
    ]

    for restart_attempt in range(num_restarts):
        # Leave a safety margin for final processing and compute_max_radii
        if time.time() - START_TIME > TIME_LIMIT * 0.9: 
            # print(f"Time limit approaching, stopping restarts after {restart_attempt} attempts.")
            break

        # Generate a random initial guess
        current_initial_centers = np.random.rand(n, 2)
        current_initial_radii_guess = np.random.uniform(0.01, 0.1, n)
        initial_guess = np.concatenate((current_initial_centers.flatten(), current_initial_radii_guess))

        try:
            # First pass with moderate iterations for each random start
            result = minimize(objective, initial_guess, method='SLSQP', bounds=bounds, constraints=constraints, options={'disp': False, 'maxiter': 300}) # Adjusted maxiter
        except TimeoutError:
            # If a single optimization times out, move to the next restart
            continue

        if result.success:
            optimized_x = result.x
        else:
            optimized_x = initial_guess # Fallback to initial guess for this iteration

        centers = optimized_x[:2*n].reshape(n, 2)
        radii = optimized_x[2*n:]

        # Final check and adjustment of radii using compute_max_radii
        radii = compute_max_radii(centers, given_radii=radii)
        current_sum_radii = np.sum(radii)

        if current_sum_radii > best_sum_radii:
            best_sum_radii = current_sum_radii
            best_centers = centers
            best_radii = radii

    # Polishing step: run optimization again with the best result found from all restarts,
    # with tighter tolerances and more iterations if time permits.
    if best_centers is not None:
        polish_initial_guess = np.concatenate((best_centers.flatten(), best_radii))
        try:
            # Use remaining time for polishing. Set maxiter dynamically.
            remaining_time = TIME_LIMIT - (time.time() - START_TIME)
            polish_maxiter = 2000 if remaining_time > 5 else 500 # More iterations if enough time
            
            polish_result = minimize(objective, polish_initial_guess, method='SLSQP', bounds=bounds, constraints=constraints, options={'disp': False, 'maxiter': polish_maxiter, 'ftol': 1e-9, 'eps': 1e-9})
            if polish_result.success:
                best_centers = polish_result.x[:2*n].reshape(n, 2)
                best_radii = polish_result.x[2*n:]
                # Re-run compute_max_radii after polishing
                best_radii = compute_max_radii(best_centers, given_radii=best_radii)
                best_sum_radii = np.sum(best_radii)
            else:
                pass # print(f"Polishing step failed: {polish_result.message}. Using previous best result.")
        except TimeoutError:
            pass # print(f"Polishing step terminated early. Using previous best result.")

    if best_centers is None: # Fallback if no successful optimization at all
        # Use the initial structured placement if nothing better was found
        initial_centers = np.zeros((n, 2))
        initial_radii_guess = np.zeros(n)
        initial_centers[0] = [0.5, 0.5]
        initial_radii_guess[0] = 0.1
        for i in range(8):
            angle = 2 * np.pi * i / 8
            initial_centers[i + 1] = [0.5 + 0.2 * np.cos(angle), 0.5 + 0.2 * np.sin(angle)]
            initial_radii_guess[i + 1] = 0.05
        for i in range(17):
            angle = 2 * np.pi * i / 17
            initial_centers[i + 9] = [0.5 + 0.4 * np.cos(angle), 0.5 + 0.4 * np.sin(angle)]
            initial_radii_guess[i + 9] = 0.03
        
        best_centers = initial_centers
        best_radii = compute_max_radii(best_centers, given_radii=initial_radii_guess)
        best_sum_radii = np.sum(best_radii)

    return best_centers, best_radii, best_sum_radii


def compute_max_radii(centers, given_radii=None):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.
    This function performs an iterative adjustment of radii.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        given_radii: Optional. np.array of shape (n) with initial radius guesses.
                     If provided, used as a starting point. Otherwise, initializes to 1.

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    if given_radii is None:
        radii = np.ones(n)
    else:
        radii = np.copy(given_radii) # Start with current radii if provided

    # Iteratively adjust radii
    for _ in range(1000): # Increased iterations for better precision
        if time.time() - START_TIME > TIME_LIMIT:
            break
        prev_radii = np.copy(radii)

        # Limit by distance to square borders (fully vectorized)
        x, y = centers[:, 0], centers[:, 1]
        radii_limited_by_bounds = np.minimum.reduce([x, y, 1 - x, 1 - y])
        radii = np.minimum(radii, radii_limited_by_bounds) # Ensure we only decrease radii or keep them
        radii = np.maximum(radii, 1e-6) # Ensure radii are not zero or negative

        # Limit by distance to other circles (fully vectorized)
        # Compute all pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        dist_matrix = np.sqrt(dist_sq)
        np.fill_diagonal(dist_matrix, np.inf) # A circle doesn't limit itself
        
        # Optimized: Directly compute limits from others without explicit radii_j_expanded
        limits_from_others_matrix = dist_matrix - radii[np.newaxis, :]
        
        new_radii_limit_by_others = np.min(limits_from_others_matrix, axis=1)

        radii = np.minimum(radii, new_radii_limit_by_others) 
        radii = np.maximum(radii, 1e-6) # Ensure radii are not zero or negative

        if np.allclose(radii, prev_radii, atol=1e-7): # Increased precision for convergence check
            break

    return radii


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_packing():
    """Run the circle packing constructor for n=26"""
    # Initialize START_TIME for each run
    global START_TIME
    START_TIME = time.time()

    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    """
    Visualize the circle packing

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))

    # Draw unit square
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    # Draw circles
    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii}")
    # AlphaEvolve improved this to 2.635

    # Uncomment to visualize:
    # visualize(centers, radii)
