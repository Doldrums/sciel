"""The best known construction for this task: R. Pinho's 98-atom measure for C_47 (teorth/optimizationproblems PR 350,
candidate, unrefereed; rpinho/math-certificates c47/round2/certificates/final.json, sha256 bc342d18...), found from
SCiEL's 63-atom PR 341, itself from Pinho's PR 321 half cell, hole-filling on T. S. Tomar's measure (zenodo 23716077)
refining Y. Lin's weighted lattice. Its decimal coordinates and masses rounded to binary64; assets/hl.py gives the ratio
47861206742878272615637096835775228687/28087100168700099164727324743556923392 = 1.704028057556977 from 1802408
rectangles (Pinho's periodic ratio on the 2^-48 grid: 1.7040280575637). The data is the measure's JSON, zlib-compressed and base64-encoded."""

import base64
import json
import zlib

DATA = (
    "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"
    "6AKnzaKOfP/+r38DBPDQAg=="
)


def candidate() -> dict:
    return json.loads(zlib.decompress(base64.b64decode("".join(DATA))))
