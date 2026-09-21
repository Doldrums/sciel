"""Metric metadata never changes evaluator fitness or manufactures observations."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.eval.registry import TaskSpec, load_task
from sciel.hypothesis.measurement import (
    SCHEMA, evaluation_observation, metric_context, validate_cases,
    validate_evaluation, validate_metric_context, validate_research_metric,
)


ROOT = Path(__file__).resolve().parents[2]
CIRCLE = ROOT / "tasks/oe/circle_packing"


def _signed(value):
    result = copy.deepcopy(value)
    result.pop("sha256", None)
    result["sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    return result


def _observation(**updates):
    value = {"ok": True, "score": 0.887793997246478,
             "cases": {"sum_radii": 2.3393371827444693, "validity": 1.0},
             "reached": "full", "confirmed": False,
             "raw_first_score": 0.887793997246478}
    value.update(updates)
    return value


def test_circle_declaration_has_exact_units_scale_and_bound_identity():
    task = load_task(str(CIRCLE))
    context = metric_context(task)
    assert context["schema"] == SCHEMA
    assert context["task_name"] == "oe/circle_packing"
    assert context["task_hash"] == task.task_hash
    assert context["primary_metric"] == "combined_score"
    assert context["score_source"] == "EvalResult.score" and context["goal"] == "maximize"
    semantics = context["semantics"]
    assert semantics == task.research_metric
    assert semantics["score_units"] == "dimensionless"
    assert semantics["numerator_case"] == "sum_radii"
    assert semantics["numerator_units"] == "unit_square_side_length"
    assert semantics["denominator"] == 2.635
    assert semantics["validity_case"] == "validity" and semantics["valid_value"] == 1.0
    assert semantics["invalid_score"] == semantics["invalid_numerator_value"] == 0.0
    assert semantics["invalid_numerator_semantics"] == "penalty_sentinel_not_physical_measurement"
    assert _signed(context) == context
    assert validate_metric_context(json.loads(json.dumps(context))) == context
    semantics["denominator"] = 9
    assert task.research_metric["denominator"] == 2.635  # No shared mutable state.


@pytest.mark.parametrize("name,known", [("circle_packing", 2.635), ("matmul_tensor", 48),
                                       ("function_minimization", 1.5)])
def test_name_or_known_best_never_infers_metric_semantics(name, known):
    task = TaskSpec("unused", "oe", name, "oe", "combined_score",
                    known_best=known, task_hash="a" * 64)
    value = metric_context(task)
    assert value["semantics"] == {"kind": "unknown", "score_units": None}
    assert "denominator" not in value["semantics"]


def test_no_task_and_undeclared_real_task_are_explicitly_unknown():
    value = metric_context()
    assert all(value[k] is None for k in ("task_name", "task_hash", "primary_metric"))
    assert value["semantics"]["kind"] == "unknown"
    assert validate_metric_context(value) == value
    task = load_task(str(ROOT / "tasks/oe/matmul_tensor"))
    assert task.known_best == 48 and task.research_metric is None
    assert metric_context(task)["semantics"] == value["semantics"]


def test_metric_metadata_changes_task_identity_without_changing_evaluator_bytes(tmp_path):
    for name in ("task.yaml", "evaluator.py", "initial_program.py"):
        (tmp_path / name).write_bytes((CIRCLE / name).read_bytes())
    before = load_task(str(tmp_path))
    cfg = yaml.safe_load((tmp_path / "task.yaml").read_text())
    cfg["research_metric"]["denominator"] = 2.0
    (tmp_path / "task.yaml").write_text(yaml.safe_dump(cfg))
    after = load_task(str(tmp_path))
    assert before.task_hash != after.task_hash
    assert metric_context(before)["sha256"] != metric_context(after)["sha256"]
    for name in ("evaluator.py", "initial_program.py"):
        assert (tmp_path / name).read_bytes() == (CIRCLE / name).read_bytes()


@pytest.mark.parametrize("key,value", [
    ("kind", "ratio"), ("score_units", "radius"), ("denominator", 0),
    ("denominator", -1), ("denominator", True), ("denominator", float("nan")),
    ("denominator", float("inf")), ("valid_value", False), ("invalid_score", "0"),
    ("invalid_numerator_value", None), ("numerator_case", ""),
    ("validity_case", "sum_radii"), ("numerator_units", None),
    ("invalid_numerator_semantics", "physical_radius"), ("extra", 1),
])
def test_invalid_task_metric_metadata_is_rejected_atomically(key, value):
    raw = load_task(str(CIRCLE)).research_metric
    raw[key] = value
    before = copy.deepcopy(raw)
    with pytest.raises(ValueError):
        validate_research_metric(raw)
    assert raw.keys() == before.keys()  # Validator never cleans the input in place.


def test_registry_rejects_partial_metadata_instead_of_treating_it_as_unknown(tmp_path):
    for name in ("evaluator.py", "initial_program.py"):
        (tmp_path / name).write_text("# Deliberately never imported.\n")
    cfg = dict(suite="test", name="fixture", origin="math", primary_metric="score",
               research_metric={"kind": "validity_gated_ratio", "denominator": 2.635})
    (tmp_path / "task.yaml").write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match="exactly"):
        load_task(str(tmp_path))


@pytest.mark.parametrize("change", [
    {"goal": "minimize"}, {"score_source": "cases.combined_score"},
    {"task_hash": "not-sha256"}, {"schema": "future"}, {"unknown": True},
    {"semantics": {"kind": "unknown", "score_units": "dimensionless"}},
])
def test_resigning_malformed_context_does_not_bypass_schema(change):
    context = metric_context()
    context.update(change)
    with pytest.raises(ValueError):
        validate_metric_context(_signed(context))


def test_context_tamper_or_unbound_declared_semantics_is_rejected():
    context = metric_context(load_task(str(CIRCLE)))
    changed = copy.deepcopy(context)
    changed["semantics"]["denominator"] = 2.0
    with pytest.raises(ValueError, match="hash"):
        validate_metric_context(changed)
    context["task_hash"] = None
    with pytest.raises(ValueError, match="bound task"):
        validate_metric_context(_signed(context))


def test_returned_values_and_explicit_score_are_preserved_without_conversion():
    cases = {"sum_radii": 2.3393371827444693, "combined_score": 123.25,
             "n_trials": 3, "signed_zero": -0.0}
    result = EvalResult(True, 0.887793997246478, cases)
    outcome = CascadeOutcome(result, 0.91, "confirm", confirmed=True)
    value = evaluation_observation(outcome)
    assert value["score"] == result.score != cases["combined_score"]
    assert value["raw_first_score"] == 0.91
    assert value["cases"] == cases and type(value["cases"]["n_trials"]) is int
    assert json.dumps(value["cases"]["signed_zero"]) == "-0.0"
    assert validate_evaluation(json.loads(json.dumps(value))) == value
    value["cases"]["sum_radii"] = 99
    assert result.cases["sum_radii"] == 2.3393371827444693


def test_invalid_packing_failure_and_no_returned_result_remain_distinct():
    invalid = _observation(score=0., cases={"validity": 0., "sum_radii": 0.}, raw_first_score=0.)
    failed = {**invalid, "ok": False, "reached": "smoke"}
    assert validate_evaluation(invalid)["ok"] is True
    assert validate_evaluation(failed)["ok"] is False
    assert validate_evaluation(None) is None
    assert validate_evaluation(failed)["cases"] == failed["cases"]


@pytest.mark.parametrize("cases", [None, [], {1: 1}, {"": 0}, {"x": True},
                                    {"x": "1"}, {"x": None}, {"x": float("nan")},
                                    {"x": float("inf")}, {"x": 10 ** 1000}])
def test_cases_reject_bad_entry_without_silently_dropping_it(cases):
    with pytest.raises(ValueError):
        validate_cases(cases)


@pytest.mark.parametrize("changes", [
    {"ok": 1}, {"confirmed": 0}, {"score": True}, {"raw_first_score": float("nan")},
    {"reached": "mini"}, {"confirmed": True}, {"extra": 0},
    {"ok": False, "reached": "confirm", "confirmed": True},
])
def test_malformed_returned_evaluation_fails_without_repair(changes):
    with pytest.raises(ValueError):
        validate_evaluation(_observation(**changes))


def test_missing_cascade_result_is_not_fabricated():
    with pytest.raises(ValueError, match="CascadeOutcome"):
        evaluation_observation(SimpleNamespace())


@pytest.mark.parametrize("valid", [True, False])
def test_real_circle_scalar_contract_on_fixed_arrays_only(monkeypatch, tmp_path, valid):
    # Import only the repository evaluator. Its candidate loader is replaced
    # before full() is called; no source generation or candidate execution occurs.
    import numpy as np
    spec = importlib.util.spec_from_file_location("circle_metric_fixture", CIRCLE / "evaluator.py")
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    centers = np.array([[.1 + .15 * (i % 6), .1 + .15 * (i // 6)] for i in range(26)])
    radii = np.full(26, .02)
    if not valid:
        centers[0, 0] = -.1
    monkeypatch.setattr(evaluator, "_run", lambda *args: (centers, radii, 999.))
    monkeypatch.setattr(evaluator, "_load", lambda *args: pytest.fail("candidate import forbidden"))
    source = tmp_path / "never_execute_this_candidate.py"
    source.write_text("# Inert source identity; loader is replaced.\n")
    result = evaluator.full(str(source))
    definition = metric_context(load_task(str(CIRCLE)))["semantics"]
    assert result["ok"] is True
    if valid:
        assert result["score"] == float(np.sum(radii)) / definition["denominator"]
        assert result["cases"][definition["numerator_case"]] == float(np.sum(radii))
        assert result["cases"][definition["validity_case"]] == definition["valid_value"]
    else:
        assert result["score"] == definition["invalid_score"]
        assert result["cases"][definition["numerator_case"]] == definition["invalid_numerator_value"]
        assert result["cases"]["validity"] == 0.
    assert result["artifacts"]["reported_sum"] == "999.000000"  # Not a measurement authority.
