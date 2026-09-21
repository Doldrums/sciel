# EVOLVE-BLOCK-START
import numpy as np


def graph_edges() -> np.ndarray:
    """
    Construct a triangle-free graph on 16 vertices that maximizes the number of edges.
    By Turan's theorem the maximum is floor(16^2/4) = 64, achieved by the complete
    bipartite graph K_{8,8}.

    Returns:
        adjacency: np.ndarray of shape (16,16), a symmetric 0/1 adjacency matrix with
        zero diagonal describing an undirected simple graph.
    """
    n = 16
    adjacency = np.zeros((n, n), dtype=int)
    for j in range(1, n):
        adjacency[0, j] = 1
        adjacency[j, 0] = 1
    return adjacency


# EVOLVE-BLOCK-END
