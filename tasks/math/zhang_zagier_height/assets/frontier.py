"""The best known construction for this task: V. Patankar's certificate [Pat26c] for C_82 (vidigoat/optimization-records
commit 197962c, tao/certificates/c82/v3, teorth/optimizationproblems PR 280, 2026), certified C_82 <= 0.2520591385260
by its Arb checker. Doche's limit-point construction with three monic degree-104 denominators (multiplicities summing
to D = 999999999960 over the degrees) and 21 numerators: Doche's P1, P2, P3, P4, P6, R. Pinho's MS360 (PR 301), two
earlier library factors and 13 from Patankar's dual-weighted LLL search, with dyadic exponents. Coefficients are listed
from the leading one down; the data is the certificate's JSON, zlib-compressed and base64-encoded."""

import base64
import json
import zlib

DATA = (
    "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"
    "wg==")


def certificate() -> dict:
    """{"denominators": [{"name", "coeffs", "multiplicity"}], "numerators": [{"name", "coeffs", "exponent": [p, q]}]}."""
    return json.loads(zlib.decompress(base64.b64decode("".join(DATA))))
