"""The best known construction for this task: SCiEL's four-strand superhelix with even modulated radii (Qwen3.8-27B,
run kr1, 6 Nov 2026; teorth/optimizationproblems PR 342, candidate, unrefereed; zenodo 24038188), found from R. Pinho's
constant-radius superhelix of PR 324 (claim 7.090412) on PR 324's caps. Claimed C_22b <= 7.087662112: length per
crossing 7.0895361607 over thickness 1.00026441. N = 276/5, binary64 parameters read exactly; outer and inner radii
carry cos u and cos 2u harmonics."""

from fractions import Fraction as F


def candidate() -> dict:
    return {"P": 276, "Q": 5, "R": float(F("6606242138010249/2251799813685248")),
            "h": float(F("4947045984950869/281474976710656")),
            "outer": [float(F(x)) for x in ("8701839008452619/4503599627370496", "2113683943087381/2305843009213693952",
                                            "0", "-3539682830458125/295147905179352825856", "0")],
            "inner": [float(F(x)) for x in ("2805350842905619/36028797018963968", "-1197882095636579/9223372036854775808",
                                            "0", "-7523285038527103/73786976294838206464", "0")]}
