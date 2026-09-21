"""Frozen data roles and prefix-causal representation/container diagnostics."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


@pytest.fixture
def pipeline():
    torch.set_num_threads(1)
    return module("train_review_models")


@pytest.fixture
def baseline(tmp_path, pipeline):
    """Small authored model artifacts, independent of retired campaign weights."""
    model_path = tmp_path / "external-encoder.pt"
    axes = tmp_path / "external-axes"
    axes.mkdir()
    with torch.random.fork_rng():
        torch.manual_seed(17)
        model = pipeline.Embedder(len(pipeline.featurize("def f(): return 1", {})))
    torch.save({"state": model.state_dict(), "in_dim": model.in_dim}, model_path)
    for task in pipeline.TASKS:
        axis = pipeline.BehaviorAxis(np.ones(64), np.zeros(64),
            embed_sha=pipeline.file_sha(model_path), fit_task=task)
        axis.save(axes / f"{task.replace('/', '_')}_v3.npz")
    return model_path, axes


@pytest.fixture
def rows(pipeline):
    result = []
    for task_no, task in enumerate(pipeline.TASKS):
        for part in pipeline.PARTS:
            for i in range(12):
                code = f"def solve_{task_no}_{part}_{i}():\n    return {i}\n"
                score = (i // 3) * .3 + (i % 3) * .001
                result.append({"task": task, "split": part, "code": code,
                               "code_hash": pipeline.normalized_hash(code), "task_hash": task,
                               "source_sha256": hashlib.sha256(code.encode()).hexdigest(),
                               "representative_run_id": f"{task}/{part}/{i // 4}",
                               "score": score, "ok": True,
                               "evaluation": {"recomputed": True},
                               "cases": {"combined_score": score, **{
                                   k: float(((i + j * 3) % 11) / 10)
                                   for j, k in enumerate(pipeline.BEHAVIOR_EXTRA_KEYS)}}})
    return result


def jsonl(path, records):
    path.write_text("".join(json.dumps(row) + "\n" for row in records))
    return path


def test_loader_rejects_program_and_run_leakage(tmp_path, pipeline, rows):
    path = jsonl(tmp_path / "rows.jsonl", rows)
    assert len(pipeline.read_corpus(path)[0]) == len(rows)
    leaked = copy.deepcopy(rows)
    leaked[12].update(code=rows[0]["code"], code_hash=rows[0]["code_hash"])
    with pytest.raises(ValueError, match="program .* leaks"):
        pipeline.read_corpus(jsonl(path, leaked))
    leaked = copy.deepcopy(rows)
    leaked[12]["representative_run_id"] = rows[0]["representative_run_id"]
    with pytest.raises(ValueError, match="run .* leaks"):
        pipeline.read_corpus(jsonl(path, leaked))


def test_fresh_behavior_is_required(tmp_path, pipeline, rows):
    rows[0]["cases"].pop("b_structure")
    with pytest.raises(ValueError, match="six recomputed"):
        pipeline.read_corpus(jsonl(tmp_path / "rows.jsonl", rows))


def test_neither_training_loss_sees_calibration_or_test(pipeline, rows):
    model, _ = pipeline.fit_candidate(rows, epochs=3, seed=7)
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["split"] != "train":
            row["score"] += 100
            row["code"] = "def adversarial_heldout_input(): return 12345"
            row["cases"] = {k: 42. for k in row["cases"]}
    other, _ = pipeline.fit_candidate(changed, epochs=3, seed=7)
    for key, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, other.state_dict()[key], rtol=0, atol=0)


def test_axis_and_temperature_do_not_see_test_values(pipeline, rows):
    rng = np.random.default_rng(4)
    Z = rng.normal(size=(len(rows), 8)).astype(np.float32)
    axis, info = pipeline.fit_axis(Z, rows, pipeline.TASKS[0], "frozen")
    changed_rows, changed_Z = copy.deepcopy(rows), Z.copy()
    for i, row in enumerate(changed_rows):
        if row["split"] == "test":
            row["score"] = i * 1000.
            changed_Z[i] *= -100
    other, other_info = pipeline.fit_axis(changed_Z, changed_rows, pipeline.TASKS[0], "frozen")
    np.testing.assert_array_equal(axis.axis, other.axis)
    np.testing.assert_array_equal(axis.mean, other.mean)
    assert axis.scale == other.scale
    assert info == other_info


def test_pipeline_saves_bound_artifacts_and_refuses_partial_results(tmp_path, pipeline, rows, baseline):
    for i, row in enumerate(rows):
        row.update(row_id=str(i), evaluation_id="evaluation", memberships=[{
            "run_id": row["representative_run_id"], "split": row["split"]}])
        row["evaluation"].update(evaluation_id="evaluation", task_hash=row["task_hash"], reached="full")
    selected = jsonl(tmp_path / "selected.jsonl", rows)
    results = jsonl(tmp_path / "results.jsonl", rows)
    heldout = [r for r in rows if r["task"] == pipeline.TASKS[0] and r["split"] == "test"][:2]
    trajectories = jsonl(tmp_path / "trajectories.jsonl", [
        {"run_id": r["representative_run_id"], "task": r["task"], "code_hash": r["code_hash"],
         "split": "test", "source_mapping": "exact_child_id", "iteration_no": i + 1,
         "B_evals": i + 1, "historical_score": r["score"]} for i, r in enumerate(heldout)])
    manifest = {"files": {"selected.jsonl": pipeline.sha256(selected),
                          "trajectories.jsonl": pipeline.sha256(trajectories)},
                "evaluation_id": "evaluation", "tasks": {t: {"task_hash": t} for t in pipeline.TASKS},
                "trajectory_anchors": [],
                "runs": [{"run_id": r["representative_run_id"], "split": r["split"],
                          "task": r["task"], "budget": 30, "trajectory_source_complete": True} for r in rows]}
    mp = tmp_path / "manifest.json"
    pipeline.write_json(mp, manifest)
    status = {"complete": True, "manifest_sha256": pipeline.sha256(mp),
              "result_files": {"results.jsonl": pipeline.sha256(results)}}
    pipeline.write_json(tmp_path / "status.json", status)
    assert pipeline.verify_frozen_corpus(results, mp) == manifest
    out = tmp_path / "models"
    assert pipeline.main(["--corpus", str(results), "--corpus-manifest", str(mp),
                          "--out", str(out), "--epochs", "1", "--baseline", str(baseline[0]),
                          "--axis-dir", str(baseline[1])]) == 0
    artifacts = json.loads((out / "manifest.json").read_text())
    assert artifacts["default_promoted"] is False
    for task, info in artifacts["baseline_axes"].items():
        path = baseline[1] / f"{task.replace('/', '_')}_v3.npz"
        assert info == {"path": str(path.resolve()), "sha256": pipeline.sha256(path)}
    assert artifacts["samples_sha256"] == pipeline.sha256(out / "samples.jsonl")
    assert artifacts["model_sha256"] == pipeline.sha256(artifacts["model_path"])
    assert all(not info["selection_used_test"] for info in artifacts["axes"].values())
    assert json.loads((out / "metrics.json").read_text())["v3_exposure"]["holdout"] is False
    with pytest.raises(ValueError, match="not empty"):
        pipeline.main(["--corpus", str(results), "--corpus-manifest", str(mp), "--out", str(out)])
    jsonl(results, rows[:-1])
    status["result_files"]["results.jsonl"] = pipeline.sha256(results)
    pipeline.write_json(tmp_path / "status.json", status)
    with pytest.raises(ValueError, match="incomplete"):
        pipeline.verify_frozen_corpus(results, mp)


def test_small_effective_run_count_does_not_fabricate_ci(pipeline, rows):
    test = pipeline.indices(rows, "test", pipeline.TASKS[0])
    for i in test:
        rows[i]["representative_run_id"] = "one_source_run"
    ci = pipeline.cluster_ci(rows, test, lambda ix: float(np.mean(ix)))
    assert ci["n_runs"] == 1 and ci["low"] is None and ci["high"] is None


def test_simple_baseline_fit_never_uses_calibration_or_test(pipeline, rows):
    metrics, ridge = pipeline.simple_baselines(rows)
    changed = copy.deepcopy(rows)
    for row in changed:
        if row["split"] != "train":
            row["score"] *= -100
            row["cases"] = {k: v * -99 for k, v in row["cases"].items()}
    other_metrics, other_ridge = pipeline.simple_baselines(changed)
    for key in ridge:
        np.testing.assert_array_equal(ridge[key], other_ridge[key])
    for task in pipeline.TASKS:
        assert metrics["task_train_mean"][task]["train_mean"] == other_metrics["task_train_mean"][task]["train_mean"]
