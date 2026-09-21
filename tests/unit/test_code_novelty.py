"""Pre-evaluation source novelty has no evaluator or model dependency."""

import json
import os
import subprocess
import sys

import pytest

from sciel.components.code_novelty import CodeNoveltyGate


def test_spacing_and_comments_are_duplicate():
    reference = "def run(x):\n    return x * 2\n"
    formatted = "# rewritten formatting\ndef run( x ):\n\treturn x*2 # no behaviour change\n"
    result = CodeNoveltyGate(threshold=1.0).inspect(formatted, [reference])
    assert result["accept"] is False
    assert result["similarity"] == 1.0
    assert result["matched_index"] == 0


def test_meaningfully_different_algorithm_is_accepted():
    reference = "def run(x):\n    return sum(v * v for v in x)\n"
    alternative = (
        "import math\ndef run(x):\n"
        "    ordered = sorted(x)\n"
        "    return math.sin(ordered[len(ordered) // 2])\n"
    )
    result = CodeNoveltyGate().inspect(alternative, [reference])
    assert result["accept"] is True
    assert result["similarity"] < 0.98


def test_numeric_constants_are_not_erased():
    gate = CodeNoveltyGate()
    result = gate.inspect("def run(x): return x * 0.25\n", ["def run(x): return x * 250\n"])
    assert result["accept"] is True
    assert result["similarity"] < 1.0


def test_nearest_reference_and_original_index_are_reported():
    code = "def run(x): return x ** 2\n"
    result = CodeNoveltyGate().inspect(code, ["# empty", "x = 37\n", code])
    assert result["matched_index"] == 2
    assert result["similarity"] == 1.0
    assert result["accept"] is False


@pytest.mark.parametrize("code, refs", [("", [""]), ("# comment", ["x=1"]),
                                        ("x=1", ["", "# comment"]), ("x=1", [])])
def test_zero_vectors_and_missing_references_fail_open(code, refs):
    result = CodeNoveltyGate().inspect(code, refs)
    assert result["accept"] is True
    assert result["similarity"] == 0.0
    assert result["matched_index"] is None


def test_source_is_never_executed(tmp_path):
    marker = tmp_path / "would_be_written"
    code = f"open({str(marker)!r}, 'w').write('executed')\nraise RuntimeError('executed')\n"
    assert CodeNoveltyGate().inspect(code, [code])["accept"] is False
    assert not marker.exists()


def test_limit_does_not_consume_extra_references():
    def references():
        yield "def run(x): return sum(x)\n"
        raise AssertionError("max_entries exceeded")

    result = CodeNoveltyGate(max_entries=1).inspect("def run(): return 23\n", references())
    assert result["accept"] is True


def test_malformed_prefix_does_not_reject_complete_program():
    result = CodeNoveltyGate().inspect("x = [1, 2]\n", ["x = [1, 2"])
    assert result["accept"] is True
    assert result["matched_index"] is None


def test_descriptor_is_independent_of_python_hash_seed():
    script = (
        "import json; from sciel.components.code_novelty import CodeNoveltyGate; "
        "print(json.dumps(CodeNoveltyGate().inspect('def f(x): return x+1\\n', "
        "['def f(x): return x*2\\n'])))"
    )
    outputs = []
    for seed in ("11", "972"):
        env = dict(os.environ, PYTHONHASHSEED=seed)
        outputs.append(json.loads(subprocess.check_output([sys.executable, "-c", script], env=env)))
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize("kwargs", [{"threshold": 0}, {"threshold": 1.1},
                                    {"threshold": float("nan")}, {"max_entries": 0},
                                    {"max_entries": 1.5}, {"max_entries": True}])
def test_invalid_configuration_rejected(kwargs):
    with pytest.raises(ValueError):
        CodeNoveltyGate(**kwargs)
