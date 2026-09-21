"""Task/toolbox contracts without importing or executing candidate source."""
import ast
from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from sciel.methods.examples import circle_pipeline, circle_state_schema, circle_toolbox
from sciel.methods.models import validate_manifest, validate_value
from sciel.methods.registry import Registry
from sciel.methods.tasks import Circle26Task, SortTask


RUNTIME = {"image_id": "sha256:" + "a" * 64}
RESOURCES = {"wall_seconds": 5, "cpu_seconds": 3, "memory_bytes": 268435456,
             "pids": 8, "max_output_bytes": 65536, "max_stderr_bytes": 4096}


def geometry(dtype="float64"):
    centers = np.asarray([[(j + .5) / 6, (i + .5) / 5] for i in range(5) for j in range(6)][:26])
    radii = np.full(26, .01, dtype=dtype)
    return {"centers": centers.tolist(), "radii": radii.astype(float).tolist(), "radii_dtype": dtype}


def test_sort_checks_values_multiplicity_and_input_contract():
    task = SortTask()
    assert task.measure([3, -1, 3], [-1, 3, 3])["metrics"] == {"correct": 1.0}
    assert task.measure([3, -1, 3], [-1, 3])["valid"] is False
    assert task.measure([3, -1, 3], [3, -1, 3])["valid"] is False
    assert task.measure([1], [True])["valid"] is False
    with pytest.raises(ValueError):
        task.measure([float("inf")], [1])
    identity = task.identity
    identity["metric"] = "pretend"
    assert task.identity["metric"] == "correct"


def test_source_wrapping_does_not_execute_and_is_not_a_toolbox_claim(tmp_path):
    marker = tmp_path / "MUST_NOT_EXIST"
    source = f"raise RuntimeError('never import')\ndef solve(values):\n    open({str(marker)!r}, 'w').close()\n"
    bundle = SortTask().solution_bundle(source, "author", RUNTIME, RESOURCES)
    assert bundle["files"]["candidate.py"] == source
    assert bundle["manifest"]["protocol"]["permanent_toolbox_method"] is False
    assert not marker.exists()
    validate_manifest(bundle["manifest"], bundle["files"])


def test_circle_identity_pins_canonical_task_and_evaluator_bytes():
    task = Circle26Task()
    evaluator = Path(task.task.evaluator_path)
    seed = Path(task.task.initial_program_path)
    assert task.identity["task_hash"] == task.task.task_hash
    assert task.identity["evaluator_sha256"] == hashlib.sha256(evaluator.read_bytes()).hexdigest()
    assert task.identity["canonical_seed_sha256"] == hashlib.sha256(seed.read_bytes()).hexdigest()
    assert task.canonical_seed_source == seed.read_text()
    assert task.identity["validity_tolerance"] == 1e-6


def test_circle_measurement_ignores_candidate_quality_claims():
    task = Circle26Task()
    value = geometry()
    value.update(valid=True, score=10000, sum_radii=999)
    measured = task.measure({}, value)
    assert measured["valid"] is True
    assert measured["observables"]["sum_radii"] == float(np.sum(np.array(value["radii"])))
    assert measured["metrics"][task.metric] < 1
    value["radii"][0] = .9
    invalid = task.measure({}, value)
    assert invalid["valid"] is False
    assert invalid["metrics"][task.metric] == 0


@pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
def test_circle_measurement_preserves_exact_declared_native_score_dtype(dtype):
    task = Circle26Task()
    value = geometry(dtype)
    measured = task.measure({}, value)
    expected = float(np.sum(np.asarray(value["radii"], dtype=dtype)))
    assert measured["valid"] is True
    assert measured["observables"]["radii_dtype"] == dtype
    assert measured["metrics"][task.metric] == expected / 2.635


def test_circle_does_not_allow_dtype_rounding_to_hide_invalid_geometry():
    task = Circle26Task()
    value = geometry()
    value["radii_dtype"] = "float16"  # .01 in this payload is not exactly float16.
    assert task.measure({}, value)["valid"] is False
    value = geometry("float16")
    value["centers"][0] = [float(np.float16(.8)), .5]
    value["radii"][0] = float(np.float16(.2003))
    assert task.measure({}, value)["valid"] is False


@pytest.mark.parametrize("bad", [{}, {"centers": [], "radii": []},
                                  {"centers": [[False, 0]] * 26, "radii": [0] * 26},
                                  {"centers": [[.5, .5]] * 26, "radii": [float("nan")] * 26}])
def test_circle_malformed_output_is_invalid_data_not_candidate_execution(bad):
    assert Circle26Task().measure({}, bad)["valid"] is False


def test_circle_solution_wrapper_snapshots_using_pinned_actual_evaluator():
    task = Circle26Task()
    bundle = task.solution_bundle("def run_packing():\n    raise RuntimeError('fixture')\n", "lab", RUNTIME, RESOURCES)
    assert bundle["files"]["canonical_evaluator.py"] == Path(task.task.evaluator_path).read_text()
    wrapper = bundle["files"]["adapter.py"]
    assert "_snapshot(centers, radii)" in wrapper
    assert "candidate.run_packing()" in wrapper
    assert "random.seed(seed)" in wrapper
    ast.parse(wrapper)
    validate_manifest(bundle["manifest"], bundle["files"])


def test_toolbox_packages_exact_existing_helpers_and_honest_provenance():
    toolbox = circle_toolbox(RUNTIME, RESOURCES)
    assert set(toolbox) == {"lp", "relocation", "slsqp", "greedy_control", "certifier"}
    project = Path(__file__).resolve().parents[2]
    for bundle in toolbox.values():
        manifest = bundle["manifest"]
        assert manifest["protocol"]["provenance"] == "authored_common_toolbox"
        assert manifest["protocol"]["automatically_discovered"] is False
        validate_manifest(manifest, bundle["files"])
        for name, source in bundle["files"].items():
            ast.parse(source, filename=name)
        for origin, sha in manifest["protocol"]["sources"].items():
            if origin.startswith("sciel/") or origin.startswith("tasks/"):
                path = project / ("src/" + origin if origin.startswith("sciel/") else origin)
                original = path.read_text()
                assert hashlib.sha256(original.encode()).hexdigest() == sha
                assert original in bundle["files"].values()
    assert toolbox["certifier"]["manifest"]["scope"]["not_optimizer"] is True
    assert toolbox["slsqp"]["manifest"]["scope"]["not_high_precision"] is True
    assert "solve_packing(" not in toolbox["slsqp"]["files"]["method.py"]
    assert "params, method=\"SLSQP\"" in toolbox["slsqp"]["files"]["method.py"]


def test_composition_rejects_mismatch_and_freezes_exact_dependency_order(tmp_path):
    registry = Registry(tmp_path)
    toolbox = circle_toolbox(RUNTIME, RESOURCES)
    methods = [registry.get_method(registry.register(toolbox[key]["files"], toolbox[key]["manifest"]))
               for key in ("relocation", "lp", "slsqp", "certifier")]
    composed = circle_pipeline(methods, RUNTIME, RESOURCES)
    result = registry.get_method(registry.register(composed["files"], composed["manifest"]))
    assert result["manifest"]["dependencies"] == [method["id"] for method in methods]
    assert result["manifest"]["protocol"]["sources"]["ordered_methods"] == result["manifest"]["dependencies"]
    assert "unattested" in result["manifest"]["scope"]["nested_invocations"]
    corrupted = deepcopy(methods)
    corrupted[1]["files"]["method.py"] += "\n# unauthenticated edit\n"
    with pytest.raises(ValueError, match="hash differs"):
        circle_pipeline(corrupted, RUNTIME, RESOURCES)
    with pytest.raises(ValueError, match="runtime"):
        circle_pipeline(methods, {"image_id": "sha256:" + "f" * 64}, RESOURCES)


def test_common_state_has_explicit_remaining_budget_and_geometry_contract():
    state = {**geometry(), "budget_seconds": .25, "seed": 1}
    validate_value(state, circle_state_schema())
    state["budget_seconds"] = -1
    with pytest.raises(ValueError):
        validate_value(state, circle_state_schema())


def test_ready_solution_export_contains_only_valid_data_never_algorithm_or_metadata():
    task=SortTask()
    assert task.transfer_payload([2,1],[1,2])=={'input':[2,1],'output':[1,2]}
    with pytest.raises(ValueError,match='invalid sorting'):
        task.transfer_payload([2,1],[2,1])
    task=Circle26Task()
    value=geometry()
    value.update(code='SECRET_ALGORITHM',diagnostics={'algorithm':'SECRET'},artifacts={'source':'SECRET'})
    payload=task.transfer_payload({},value)
    assert set(payload)=={'centers','radii','centers_dtype','radii_dtype'}
    assert 'SECRET' not in str(payload)
    payload['radii'][0]=9
    assert value['radii'][0]==.01
    value['radii'][0]=9
    with pytest.raises(ValueError,match='invalid packing'):
        task.transfer_payload({},value)
