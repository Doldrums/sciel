"""The best known construction for this task: C. Olafson's two-temperature cyclic quotient on the truncated numerical
semigroup M = <4, 5> cut at 80 (github.com/carson-olaf/c3a-two-temperature-bound, teorth/optimizationproblems PR 148,
2026), with x = 8149/10000 and y = 42571/50000. Here |M| = 75, M + M misses 1, 2, 3, 6, 7, 11 below 160 and M - M is
all of [-80, 80]; the six gap differences cost kappa = 9, 18, 13, 14, 17, 19. The bound 2 - sigma / rho is
1.21418118095053104745...; the PR states C_3a > 1.21418."""


def mask() -> list:
    ok = [True] + [False] * 80
    for n in range(1, 81):
        ok[n] = (n >= 4 and ok[n - 4]) or (n >= 5 and ok[n - 5])
    return [n for n in range(81) if ok[n]]


def candidate() -> dict:
    return {"mask": mask(), "x": [8149, 10000], "y": [42571, 50000]}
