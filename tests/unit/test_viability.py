"""Viability contract: typed violations + declarative repair table."""

from sciel.engine.viability import (Violation, mechanical_repair, precheck,
                                     pre_eval_gate, REPAIR_TABLE)


def test_empty_violation():
    v = precheck("   \n  ")
    assert v == Violation("empty")
    assert mechanical_repair(v, "") is None


def test_syntax_violation_and_repair():
    broken = 'Improved packing.\n"""\nimport numpy as np\ndef run(): return 1\n'
    v = precheck(broken)
    assert v.kind == "syntax"
    fixed = mechanical_repair(v, broken)
    assert fixed is not None and precheck(fixed) is None


def test_syntax_without_repair_resamples():
    v = precheck("def run(:\n  pass\n")   # genuine syntax error, not mid-docstring
    assert v.kind == "syntax"
    assert mechanical_repair(v, "def run(:\n  pass\n") is None


def test_resource_violation_no_repair():
    code = ("import multiprocessing as mp\n"
            "def run():\n    def w(x): return x\n"
            "    return mp.Pool(2).map(w, [1])\n")
    v = precheck(code)
    assert v.kind == "resource"
    assert mechanical_repair(v, code) is None


def test_valid_passes():
    assert precheck("import numpy as np\ndef run(): return 1\n") is None


def test_repair_table_is_declarative():
    assert set(REPAIR_TABLE) == {"syntax", "resource", "empty", "duplicate"}
    assert REPAIR_TABLE["syntax"] is not None       # syntax has a mechanical repair
    assert REPAIR_TABLE["resource"] is None         # resource resamples


def test_backcompat_facade():
    # pre_eval_gate still returns a reason string for the two pre-eval kinds,
    # None for empty/valid (empty is handled by the empty branch, not the gate)
    assert pre_eval_gate("   ") is None
    assert pre_eval_gate("import numpy\ndef run(): pass\n") is None
    assert "syntax" in pre_eval_gate("def x(:\n pass\n")
