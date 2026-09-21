"""arch-v4 #4a: learned MAP-Elites descriptor axis (behavior, not code length)."""

import pytest

from sciel.engine.db import Database

torch = pytest.importorskip("torch")


def test_default_grid_uses_length():
    db = Database(num_islands=1)
    assert db.descriptor_fn is None
    # short vs long code land in different length bins at same score
    short = db._cell(_prog("x=1", 0.5))
    long = db._cell(_prog("x=1\n" * 300, 0.5))
    assert short[0] != long[0]


def test_descriptor_grid_uses_behavior():
    calls = []

    def desc(code, cases):
        calls.append((code, cases))
        return 0.42

    db = Database(num_islands=1, descriptor_fn=desc)
    cell = db._cell(_prog("x=1", 0.5))
    assert calls, "descriptor_fn not consulted"
    assert cell[0] == int(0.42 * db.grid_bins)          # descriptor bin
    assert cell[1] == int(0.5 * db.grid_bins)           # score bin unchanged


def test_learned_descriptor_deterministic():
    from sciel.components.policy.code_embed import featurize
    # stable token hashing => identical features across calls/processes
    a = featurize("import numpy as np\ndef run(): return 1\n", {"validity": 1.0})
    b = featurize("import numpy as np\ndef run(): return 1\n", {"validity": 1.0})
    assert (a == b).all()


class _P:
    def __init__(self, code, score):
        self.code, self.score, self.cases = code, score, {}


def _prog(code, score):
    return _P(code, score)
