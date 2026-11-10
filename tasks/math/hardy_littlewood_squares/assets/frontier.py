"""The best known construction for this task: SCiEL's 98-atom measure for C_47 (hlz1, 9 Nov 2026), found from R. Pinho's
98-atom measure (teorth/optimizationproblems PR 350, candidate, unrefereed), itself from SCiEL's 63-atom PR 341, Pinho's
PR 321 half cell, and hole-filling on T. S. Tomar's measure (zenodo 23716077) refining Y. Lin's weighted lattice. Its
coordinates and masses are binary64; assets/hl.py gives the ratio
6837325052916568364798811725856251555/4012442881242871309246760677650989056 = 1.704030500939786 from 2002525
rectangles (Pinho's PR 350 periodic ratio: 1.7040280575637). The data is the measure's JSON, zlib-compressed and
base64-encoded."""

import base64
import json
import zlib

DATA = (
    "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"
)


def candidate() -> dict:
    return json.loads(zlib.decompress(base64.b64decode("".join(DATA))))
