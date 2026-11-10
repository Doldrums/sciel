"""The best known construction for this task: Y. H.'s 8 x 8 matrix over {-1, 0, 1} (teorth/optimizationproblems
constant 10c [H2026], 2026). Its first seven columns hold the eight rows T_i of the perfect [7, 4] Hamming code as
sign patterns and the eighth column is zero; the sixteen codewords T_i and their complements have radius-1 balls
that partition {0, 1}^7, so every sign vector meets a row with |(Ax)_i| >= 5 and disc = 5, giving 5 / sqrt(8). The
counting bound of the page (Kleitman's diameter theorem) shows 5 is optimal at n = 8; sign matrices are tied to
covering-code tables, where 5 / sqrt(8), 7 / 4 and 7 / sqrt(17) are the best rows."""

ROWS = [
    [1, 1, 1, 1, 1, 1, 1, 0],
    [1, 1, 1, -1, -1, -1, -1, 0],
    [1, 1, -1, 1, -1, -1, 1, 0],
    [1, 1, -1, -1, 1, 1, -1, 0],
    [1, -1, 1, 1, -1, 1, -1, 0],
    [1, -1, 1, -1, 1, -1, 1, 0],
    [1, -1, -1, 1, 1, -1, -1, 0],
    [1, -1, -1, -1, -1, 1, 1, 0],
]


def candidate() -> dict:
    return {"rows": [list(r) for r in ROWS], "den": 1}
