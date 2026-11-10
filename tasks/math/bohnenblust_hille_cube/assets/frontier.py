"""The best known construction for this task: Koshta's cycle function f_d on 3d variables (teorth/optimizationproblems
PR 284 [Kos2026], October 2026), here at d = 7, n = 21. For each proper subset J of Z_d it places one block
chi_beta 1_A, A the affine subspace cut out by the cycle edges z_i + z_{i+1} (i in J) and the pairs a_i + b_i
(i not in J) at values 1_J(i - 1) + c_i (c = 0 for odd d, e_0 for even d), and beta any vector of the coset (a_i or
b_i off J, one vertex of each odd path of J). The live sets and the Fourier cosets are pairwise disjoint, so f_d is
{0, +-1}-valued, zero on 2^-d of the cube, with (2^d - 1) 2^d coefficients +-2^-d, and
R = 2 (1 - 2^-d)^((d + 1) / (2d)) = 1.99105642531... at d = 7. Every {0, +-1}-valued f of degree d has
R <= 2 Pr[f != 0]^((d + 1) / (2d)) < 2, so this family cannot reach 2: beating 2 needs other values."""

import numpy as np

D = 7


def table(d=D):
    """f_d at all 2^(3d) points, int64, coordinates z_i = i, a_i = d + i, b_i = 2d + i."""
    n = 3 * d
    m = np.arange(1 << n, dtype=np.int64)
    bit = [(m >> i) & 1 for i in range(n)]
    c = [1 if (d % 2 == 0 and i == 0) else 0 for i in range(d)]
    out = np.zeros(1 << n, dtype=np.int64)
    for code in range((1 << d) - 1):
        inside = [(code >> i) & 1 for i in range(d)]
        live = np.ones(1 << n, dtype=bool)
        beta = np.zeros(1 << n, dtype=np.int64)
        for i in range(d):
            w = inside[(i - 1) % d] ^ c[i]
            if inside[i]:
                live &= (bit[i] ^ bit[(i + 1) % d]) == w
            else:
                live &= (bit[d + i] ^ bit[2 * d + i]) == w
                beta ^= bit[d + i]
        # Paths of J: maximal runs of edges; a run of m edges covers m + 1 vertices starting at its first edge.
        for i in range(d):
            if inside[i] and not inside[(i - 1) % d]:
                length = 0
                while inside[(i + length) % d]:
                    length += 1
                if length % 2:
                    beta ^= bit[i]
        out[live] = 1 - 2 * beta[live]
    return out


def candidate(d=D) -> dict:
    """The integer coefficients of 2^d f_d (all +-1) as a candidate dict."""
    a = np.array(table(d), copy=True)
    h = 1
    while h < len(a):
        a = a.reshape(-1, 2, h)
        a = np.stack([a[:, 0] + a[:, 1], a[:, 0] - a[:, 1]], axis=1).reshape(-1)
        h *= 2
    # f_d has coefficients +-2^-d, so the transform of its table is +-2^(n - d) = +-4^d.
    a //= 1 << (2 * d)
    nz = np.flatnonzero(a)
    return {"n": 3 * d, "terms": [[int(m), int(a[m])] for m in nz]}
