"""The task's objective for a candidate to build on: from sciel.task import asset; sf = asset("sunflower").

A block is {"nodes": [...], "ports": [p, q]}. A node is a leaf {"n": n, "sets": [bitmask, ...]} or a glue node
{"glue": [i, a, j, b], "sets": [...]}: the DEGKM glue of earlier nodes i and j at their points a and b (members
(X - a) | (Y - b) with [a in X] != [b in Y]; the points of i other than a first, then those of j other than b), plus
explicit extra sets. The block is the last node. Its rate is rho^(1/(n - 2)), rho the Perron root of
[[N10, N11], [N00, N01]]; any valid set added to the block raises some N_ab and so the rate.
"""

from pathlib import Path

import numpy as np

from sciel.candidate_modules import load_candidate

_evaluator = load_candidate(str(Path(__file__).resolve().parents[1] / "evaluator.py"), "evaluator")

glue = _evaluator.glue
port_counts = _evaluator.port_counts


def load(output) -> dict:
    """The block of a scored construction, from the output its evaluation recorded (a population entry's "output")."""
    return {"nodes": output["nodes"], "ports": list(output["ports"])}


def families(block) -> list:
    """(n, members as an int64 array) of every node, without checks; fast (the record's 252292 members in 0.1 s)."""
    return _evaluator.families(_evaluator.spec(block)[0])


def members(block) -> tuple:
    """(n, members) of the block itself, without checks."""
    return families(block)[-1]


def rate(N, n) -> float:
    """rho^(1/(n - 2)) for port counts N = (N00, N01, N10, N11)."""
    n00, n01, n10, n11 = N
    rho = (n10 + n01 + ((n10 - n01) ** 2 + 4 * n11 * n00) ** 0.5) / 2
    return float(rho ** (1.0 / (n - 2)))


def value(block) -> float:
    """The block's rate in floating point, without validity checks."""
    n, fam = members(block)
    return rate(port_counts(fam, *block["ports"]), n)


def exact_value(block) -> float:
    """The rate as scored, after the evaluator's checks (about 75 s for the record); raises if the block is invalid."""
    nodes, (p, q) = _evaluator.spec(block)
    n, fam = _evaluator.validate(nodes)
    return float(_evaluator.rate(port_counts(fam, p, q), n))


def why_not(fam, n, E):
    """None if E can join fam (a node's members, which may already hold E) with no containment and no 3-sunflower
    through it, else the reason. Exact, as the evaluator checks an explicit set; about 0.1 s against 252292 members."""
    return _evaluator.through(fam, n, int(E))


def addable(fam, n, E) -> bool:
    """E is new to fam and why_not(fam, n, E) is None."""
    fam = np.asarray(fam, dtype=np.int64)
    return not (fam == E).any() and why_not(fam, n, E) is None
