# EVOLVE-BLOCK-START
"""Rank-R decomposition of the 4x4 matrix-multiplication tensor.

Return factors (U, V, W), each shape (R, 16), such that
    T[p,q,r] ~= sum_s U[s,p] V[s,q] W[s,r]
where T is the 4x4 matmul tensor. The goal is to reach an EXACT decomposition
(residual ~ 0) at as LOW a rank R as possible. Naive is rank 64; Strassen^2 is
49; AlphaEvolve reached 48 over the complex field.

This starting point is the exact naive rank-64 decomposition: one rank-1 term per
(i,j,k) triple. Evolution should REDUCE the rank while keeping the residual near
zero -- e.g. by a numerical search (gradient descent / ALS on the factors with a
rank budget, rounding toward the half-integer lattice), under the time budget.
"""

import numpy as np


def run_search():
    n = 4
    terms = []
    for i in range(n):
        for j in range(n):
            for k in range(n):
                u = np.zeros(n * n); u[i * n + j] = 1.0
                v = np.zeros(n * n); v[j * n + k] = 1.0
                w = np.zeros(n * n); w[i * n + k] = 1.0
                terms.append((u, v, w))
    U = np.array([t[0] for t in terms])
    V = np.array([t[1] for t in terms])
    W = np.array([t[2] for t in terms])
    return U, V, W


# EVOLVE-BLOCK-END
