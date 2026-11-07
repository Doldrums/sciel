"""The best known construction for this task: SCiEL's 63-atom measure for C_47 (Qwen3.8-27B, run hl1, 6 Nov 2026;
teorth/optimizationproblems PR 341, candidate, unrefereed; zenodo 24022791), found from R. Pinho's 59-atom half cell
of PR 321, itself hole-filling on T. S. Tomar's measure (zenodo 23716077) refining Y. Lin's weighted lattice. Binary64
coordinates and masses, read exactly. Tomar's check_exact2.py and assets/hl.py both give the ratio
3132581668909806427917067970204393210052649/1839223215401194968772738622942534413844480 = 1.70320907...
from 165083 rectangles. The data is the measure's JSON, zlib-compressed and base64-encoded."""

import base64
import json
import zlib

DATA = (
    "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")


def candidate() -> dict:
    """{"Px": "p/q", "Py": "p/q", "atoms": [["x", "y", "m"], ...]}."""
    return json.loads(zlib.decompress(base64.b64decode("".join(DATA))))
