"""The best known construction for this task: E. Naslund's exponent 473797394875551/625000000000000 = 0.7580758318008816
(Hexagon 2610.00151v1; certificates at github.com/enaslund/sarkozy-lower-bound-0.758, commit 5a83f41). Six Paley
chains (p = 3, 7, 11, 31, 59, 103), two low-digit codes (5 * 43 at depth 3, 4913 points; 19 * 23 at depths 3, 2, 3645
points) and the 25-state binary recursion, repacked in frontier.json; the 5 * 43 starts are the longest-path positions
of its published widths."""

import json
from pathlib import Path


def certificate() -> dict:
    return json.loads(Path(__file__).with_name("frontier.json").read_text())
