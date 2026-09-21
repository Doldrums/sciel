"""Conservative static proof tests; candidate bodies are never executed.

Authored source variants challenge false positive proofs, not numerical quality.
"""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from sciel.hypothesis import circle_baseline_equivalence as checker
from sciel.hypothesis import circle_restart_body as builder
from sciel.hypothesis import circle_proposals as proposals


@pytest.fixture(scope="module")
def equivalent_bodies():
    return {"off": BASELINE, "on": "x = ordinary_params.copy()\nreturn x\n"}


@pytest.fixture
def verify():
    def run(body, parent=None, **changes):
        args = {"builder_source": Path(builder.__file__).read_bytes(),
            "proposal_source": Path(proposals.__file__).read_bytes()}
        args.update(changes)
        return checker.verify(body, builder.source_for() if parent is None else parent, **args)
    return run


BASELINE = """x = best_params + uniform_noise
x[52:] = np.maximum(x[52:], 0.001)
x[:52] = np.clip(x[:52], 0.001, 0.999)
return x
"""


@pytest.mark.parametrize("arm", ["off", "on"])
def test_distinct_authored_bodies_are_statically_baseline(verify, equivalent_bodies, arm):
    body = equivalent_bodies[arm]
    result = verify(body)
    assert result["verdict"] == "equivalent" and result["proof"]
    assert builder.source_for(body) != builder.source_for()
    assert result == verify(body)


@pytest.mark.parametrize("body", ["return None\n", "return ordinary_params\n",
    "return ordinary_params.copy()\n", "return np.copy(ordinary_params)\n", BASELINE,
    "# Different bytes, same supported transformation.\n" + BASELINE,
    BASELINE.replace("x[52:] = np.maximum(x[52:], 0.001)\nx[:52] = np.clip(x[:52], 0.001, 0.999)",
        "x[:52] = np.clip(x[:52], 0.001, 0.999)\nx[52:] = np.maximum(x[52:], 0.001)")])
def test_supported_source_variants_prove_initializer_values_only(verify, body):
    assert verify(body)["verdict"] == "equivalent"


@pytest.mark.parametrize("body", [
    "ordinary_params[:] = best_params\nreturn None\n",
    "best_params[:] = initial_params\nreturn ordinary_params\n",
    "x = ordinary_params.copy()\ny = x.copy()\ny[:52] = best_params[:52]\nreturn x\n",
])
def test_detached_argument_and_explicit_copy_mutation_does_not_change_saved_default(verify, body):
    assert verify(body)["verdict"] == "equivalent"


@pytest.mark.parametrize("body", [
    "ordinary_params[:52] = best_params[:52]\nreturn ordinary_params\n",
    "x = ordinary_params.copy()\ny = x\ny[:52] = best_params[:52]\nreturn x\n",
    "x = ordinary_params.copy()\ny = x[52:]\ny[:] = best_params[52:]\nreturn x\n",
    "x = ordinary_params.copy()\ny = x[:52].reshape(26, 2)\ny[:] = best_params[:52].reshape(26, 2)\nreturn x\n",
    "x = ordinary_params.copy()\ny = x[:52].reshape(26, 2).ravel()\ny[:] = initial_params[:52]\nreturn x\n",
    "if restart_index == 0:\n    best_params[:] = ordinary_params\n    x = initial_params.copy()\nelse:\n    x = ordinary_params.copy()\nreturn x\n",
    "if restart_index == 0:\n    initial_params[:] = ordinary_params\n    x = best_params.copy()\nelse:\n    x = ordinary_params.copy()\nreturn x\n",
])
def test_real_alias_writes_and_equal_but_independent_index_zero_inputs_are_not_erased(verify, body):
    result = verify(body)
    assert result["verdict"] == "unknown" and not result.get("proof")


@pytest.mark.parametrize("body", [
    BASELINE.replace("best_params + uniform_noise", "best_params + uniform_noise * 1.0000000000000002"),
    BASELINE.replace("best_params + uniform_noise", "best_params + uniform_noise * 0.5"),
    BASELINE.replace("maximum(x[52:], 0.001)", "maximum(x[52:], 0.0010000000000000002)"),
    BASELINE.replace("maximum(x[52:], 0.001)", "maximum(x[52:], 0.0)"),
    BASELINE.replace("clip(x[:52], 0.001, 0.999)", "clip(x[:52], 0.0010000000000000002, 0.999)"),
    BASELINE.replace("clip(x[:52], 0.001, 0.999)", "clip(x[:52], 0.001, 0.9990000000000001)"),
    BASELINE.replace("best_params + uniform_noise", "initial_params + uniform_noise"),
    BASELINE.replace("x = best_params + uniform_noise", "x = (best_params + uniform_noise) + 1e-16"),
    BASELINE.replace("x[:52] = np.clip(x[:52], 0.001, 0.999)\n", ""),
])
def test_small_real_algorithm_changes_are_not_approximately_folded_into_baseline(verify, body):
    assert verify(body)["verdict"] == "unknown"


@pytest.mark.parametrize("body", [
    "if restart_index == 4:\n    return best_params\nreturn ordinary_params\n",
    "if restart_index > 4:\n    return best_params\nreturn ordinary_params\n",
    "if np.sum(uniform_noise) > 0:\n    return best_params\nreturn ordinary_params\n",
    "for i in range(0):\n    ordinary_params[i] = 0\nreturn ordinary_params\n",
    "def f():\n    return ordinary_params\nreturn f()\n",
    "np = best_params\nreturn np.copy(ordinary_params)\n",
    "np = 0\nreturn ordinary_params\n",
    "return np.asarray(ordinary_params, dtype=np.float32)\n",
    "return ordinary_params.astype(np.float32)\n",
    "x = ordinary_params.copy()\nx[:52] = initial_params\nreturn x\n",
    "return ordinary_params.reshape(26, 3)\n",
    "return ordinary_params[:52]\n",
    "x = ordinary_params.copy()\nx[:52] = initial_params[:51]\nreturn None\n",
    "return np.maximum(ordinary_params, float('nan'))\n",
])
def test_unsupported_control_flow_rebinding_dtype_and_runtime_error_shapes_are_unknown(verify, body):
    result = verify(body)
    assert result["verdict"] == "unknown" and not result.get("proof")


@pytest.mark.parametrize("parent", [
    "not a source family",
    builder.source_for("return ordinary_params\n"),
    builder.source_for("# Same default values but not the pinned control source.\nreturn None\n"),
    builder.source_for() + "\n# byte drift\n",
])
def test_exact_parent_control_restriction_cannot_be_bypassed_by_equal_behavior(verify, parent):
    result = verify(BASELINE, parent)
    assert result["verdict"] == "unknown" and not result.get("proof")


@pytest.mark.parametrize("key", ["builder_source", "proposal_source"])
def test_source_pin_drift_refuses_proof_before_assuming_kernel_invariants(verify, key):
    path = builder.__file__ if key == "builder_source" else proposals.__file__
    result = verify(BASELINE, **{key: Path(path).read_bytes() + b"\n# drift\n"})
    assert result["verdict"] == "unknown" and not result.get("proof")


def test_proof_is_deterministic_detached_and_does_not_mutate_input(verify, equivalent_bodies):
    body = equivalent_bodies["on"]
    expected = copy.deepcopy(verify(body))
    for field in ("identity", "states", "assumptions"):
        result = verify(body)
        result["proof"][field].clear()
        assert verify(body) == expected
    assert equivalent_bodies["on"] == body


def test_proof_binds_exact_raw_body_parent_sources_and_both_call_domains(verify, equivalent_bodies):
    body = equivalent_bodies["off"]
    proof = verify(body)["proof"]
    assert proof["body_sha256"] == hashlib.sha256(body.encode()).hexdigest()
    assert proof["parent_source_sha256"] == hashlib.sha256(builder.source_for().encode()).hexdigest()
    assert proof["identity"]["builder_source_sha256"] == hashlib.sha256(Path(builder.__file__).read_bytes()).hexdigest()
    assert proof["identity"]["proposal_builder_sha256"] == hashlib.sha256(Path(proposals.__file__).read_bytes()).hexdigest()
    assert proof["checker_source_sha256"] == hashlib.sha256(Path(checker.__file__).read_bytes()).hexdigest()
    assert len(proof["states"]) == 2 and all(state["elements_proven_equal"] == 78 for state in proof["states"])
    assert "restart_index == 0" in proof["states"][0]["domain"]
    assert "4,8,12" in proof["states"][1]["domain"]
    canonical = json.dumps({key: value for key, value in proof.items() if key != "sha256"},
        sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    assert proof["sha256"] == hashlib.sha256(canonical).hexdigest()
    assert verify(body + "\n")["proof"]["body_sha256"] != proof["body_sha256"]
    assert "Not runtime" in proof["scope"] and "quality equivalence" in proof["scope"]


@pytest.mark.parametrize("body", [None, b"return None\n", True, "", "return (\n",
    "#" + "x" * 8192 + "\nreturn None\n", "pass\n" * 513 + "return None\n",
    "x = ordinary_params\n" + "x = x.copy()\n" * 160 + "return x\n",
    "# isolated surrogate \ud800\nreturn None\n"])
def test_invalid_or_unbounded_input_returns_finite_unknown_not_a_crash(verify, body):
    result = verify(body)
    assert result["verdict"] == "unknown" and not result.get("proof")
    json.dumps(result, allow_nan=False)


def test_loaded_checker_source_drift_refuses_a_fresh_proof(verify, tmp_path, monkeypatch):
    path = tmp_path / "checker-copy.py"
    path.write_bytes(Path(checker.__file__).read_bytes())
    monkeypatch.setattr(checker, "__file__", str(path))
    assert verify(BASELINE)["verdict"] == "equivalent"
    path.write_bytes(path.read_bytes() + b"\n# changed after loading\n")
    result = verify(BASELINE)
    assert result["verdict"] == "unknown" and not result.get("proof")
