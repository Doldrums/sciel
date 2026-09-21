"""Corpus provenance, disjoint holdouts, frozen sampling and resumable labels."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "prepare_review_corpus.py"
spec = importlib.util.spec_from_file_location("prepare_review_corpus", SCRIPT)
corpus = importlib.util.module_from_spec(spec)
spec.loader.exec_module(corpus)


def make_run(root, name, *, task="oe/circle_packing", code="def f(): return 1\n", tail=None,
             event_code=None):
    folder = root / name
    folder.mkdir()
    state = {"config": {"task": task}, "db": {"programs": [
        {"id": "p2", "code": code, "score": .5, "cases": {"b_contact": 1.0}}]}}
    events = [{"event": "run_start", "task": task, "task_hash": "historical-contract", "budget": 1},
              {"event": "iteration", "child_id": "p2", "iteration_no": 1, "B_evals": 1,
               "child_score": .5, "llm_response": "never evaluated"},
              {"event": "run_end", "iteration_no": 1, "B_evals": 1, "reason": "budget"}]
    if event_code:
        events[1]["child_code"] = event_code
    if tail:
        events.extend(tail)
    (folder / "checkpoint.json").write_text(json.dumps(state))
    corpus.write_jsonl(folder / "sciel_events.jsonl", events)
    return folder


def test_collect_excludes_open_resumes_and_uses_final_evaluated_source(tmp_path):
    make_run(tmp_path, "good", event_code="def f(): return 7\n")
    make_run(tmp_path, "dirty", tail=[{"event": "run_start", "task": "oe/circle_packing"}])
    items, runs, excluded, traces, _ = corpus.collect([tmp_path])
    assert len(items) == len(runs) == len(excluded) == len(traces) == 1
    membership = items[0]["memberships"][0]
    assert membership["source_kind"] == "journal.child_code"
    assert membership["source_task_hash"] == "historical-contract"
    assert items[0]["sources"][membership["source_sha256"]] == "def f(): return 7\n"
    assert traces[0]["code_hash"] == corpus.normalized_hash("def f(): return 7\n")
    assert "resumed segment" in excluded[0]["reason"]


def test_legacy_checkpoint_without_child_id_is_training_source_not_invented_trajectory(tmp_path):
    folder = make_run(tmp_path, "old")
    path = folder / "sciel_events.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    events[1].pop("child_id")
    corpus.write_jsonl(path, events)
    items, runs, _, traces, _ = corpus.collect([tmp_path])
    assert items[0]["memberships"][0]["child_id"] is None
    assert traces[0]["source_mapping"] == "unavailable"
    assert not runs[0]["trajectory_source_complete"]


def test_source_changed_during_inventory_is_excluded(tmp_path, monkeypatch):
    folder = make_run(tmp_path, "changing")
    original = corpus.completed_events

    def change_checkpoint(path):
        result = original(path)
        (folder / "checkpoint.json").write_text('{"changed": true}')
        return result

    monkeypatch.setattr(corpus, "completed_events", change_checkpoint)
    items, runs, excluded, _, _ = corpus.collect([tmp_path])
    assert not items and not runs
    assert "source changed" in excluded[0]["reason"]


def test_normalized_duplicates_do_not_cross_splits_even_across_tasks(tmp_path):
    for task_index, task in enumerate(corpus.TASKS):
        for i in range(10):
            make_run(tmp_path, f"{task_index}-{i}", task=task,
                     code="def f(): return 1\n" if i < 3 else f"def f(): return {task_index*10+i}\n")
    items, runs, _, _, _ = corpus.collect([tmp_path])
    corpus.assign_splits(items, runs, seed=12)
    shared_hash = corpus.normalized_hash("def f(): return 1\n")
    shared = [item for item in items if item["code_hash"] == shared_hash]
    assert len(shared) == 3
    assert len({item["split"] for item in shared}) == 1
    by_hash = {}
    for item in items:
        assert by_hash.setdefault(item["code_hash"], item["split"]) == item["split"]
    assert {run["split"] for run in runs} == set(corpus.SPLITS)
    first = [(item["row_id"], item["split"]) for item in corpus.sample(items, runs, 10, 12)[0]]
    second = [(item["row_id"], item["split"]) for item in corpus.sample(list(reversed(items)), runs, 10, 12)[0]]
    assert first == second


def test_sampling_spans_scores_and_is_fixed_before_current_results():
    rows = [{"historical_score": i / 100, "code_hash": str(i)} for i in range(100)]
    selected = corpus.stratify(rows, 4)
    assert [row["historical_score"] for row in selected] == [.12, .37, .62, .87]
    for row in selected:
        row["ok"] = False
    assert corpus.stratify(rows, 4) == selected


def source_row():
    code = "def f(): return 1\n"
    return {"row_id": "row1", "task": "oe/circle_packing", "code": code,
            "source_sha256": corpus.digest(code.encode()), "task_hash": "new-contract",
            "split": "train", "evaluation_id": "runtime1"}


def result_row():
    return {**source_row(), "ok": False, "score": 0.0, "cases": {},
            "evaluation": {"recomputed": True, "evaluation_id": "runtime1", "task_hash": "new-contract",
                           "wall_seconds": 1.0}}


def test_failed_evaluation_is_reused_and_interrupted_final_append_is_recoverable(tmp_path):
    path = tmp_path / "results.jsonl"
    row = result_row()
    path.write_text(json.dumps(row) + '\n{"row_id":')
    loaded = corpus.load_results(path, [source_row()], "runtime1")
    assert loaded == {"row1": row}
    assert not loaded["row1"]["ok"]
    assert path.with_suffix(".interrupted-tail").exists()
    assert len(path.read_text().splitlines()) == 1


@pytest.mark.parametrize("change", [
    {"task_hash": "old-contract"}, {"evaluation_id": "old-runner"},
    {"source_sha256": "unknown"}, {"code": "different source"}, {"split": "test"},
    {"evaluation": {}}, {"score": float("nan")}, {"ok": "true"},
    {"evaluation": {"recomputed": True, "evaluation_id": "runtime1", "task_hash": "other-task"}},
])
def test_unidentified_backfill_and_mismatched_results_never_reused(tmp_path, change):
    path = tmp_path / "results.jsonl"
    path.write_text(json.dumps({**result_row(), **change}) + "\n")
    with pytest.raises(ValueError):
        corpus.load_results(path, [source_row()], "runtime1")


def test_frozen_source_hash_drift_prevents_evaluation(tmp_path, monkeypatch):
    (tmp_path / "selected.jsonl").write_text("original\n")
    manifest = {"schema_version": corpus.VERSION, "runtime": {"runtime": 1}, "tasks": {},
                "files": {"selected.jsonl": corpus.file_hash(tmp_path / "selected.jsonl")}}
    corpus.write_json(tmp_path / "manifest.json", manifest)
    monkeypatch.setattr(corpus, "runtime_identity", lambda: {"runtime": 1})
    assert corpus.verify_manifest(tmp_path) == manifest
    (tmp_path / "selected.jsonl").write_text("mutated\n")
    with pytest.raises(ValueError, match="source manifest changed"):
        corpus.verify_manifest(tmp_path)


def test_runtime_drift_prevents_reuse(tmp_path, monkeypatch):
    corpus.write_json(tmp_path / "manifest.json", {"schema_version": corpus.VERSION, "runtime": {"version": 1}})
    monkeypatch.setattr(corpus, "runtime_identity", lambda: {"version": 2})
    with pytest.raises(ValueError, match="runtime changed"):
        corpus.verify_manifest(tmp_path)


def test_finalize_keeps_failures_and_partial_status(tmp_path):
    selected = [source_row(), {**source_row(), "row_id": "pending"}]
    corpus.write_json(tmp_path / "manifest.json", {"frozen": True})
    status = corpus.finalize(tmp_path, {"tasks": ["oe/circle_packing"]}, selected, {"row1": result_row()})
    assert not status["complete"] and status["pending"] == 1
    assert json.loads((tmp_path / "train.jsonl").read_text())["ok"] is False
    assert status["tasks"]["oe/circle_packing"]["failed"] == 1


@pytest.mark.parametrize("smoke_ok", [True, False])
def test_local_sandbox_rescores_full_and_retains_smoke_failure(tmp_path, monkeypatch, smoke_ok):
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text(
        f"def smoke(path): return {{'ok': {smoke_ok!r}, 'score': 1.0}}\n"
        "def full(path): return {'ok': True, 'score': 3.0, 'cases': {'b_contact': 0.25}}\n")
    task = SimpleNamespace(task_hash="new-contract", evaluator_path=str(evaluator),
                           full_timeout=3, smoke_timeout=3, cpu_seconds=None, mem_bytes=None)
    monkeypatch.setattr(corpus, "load_task", lambda path: task)
    result = corpus.evaluate_one(source_row(), tmp_path)
    assert result["ok"] is smoke_ok
    assert result["score"] == (3 if smoke_ok else 1)
    assert result["evaluation"]["reached"] == ("full" if smoke_ok else "smoke")
    assert result["evaluation"]["task_hash"] == "new-contract"
    assert result["evaluation"]["recomputed"] is True
    if smoke_ok:
        assert result["cases"] == {"b_contact": .25}


def test_concurrent_resume_cannot_append_same_cache(tmp_path):
    with (tmp_path / ".evaluation.lock").open("a") as lock:
        corpus.fcntl.flock(lock, corpus.fcntl.LOCK_EX | corpus.fcntl.LOCK_NB)
        with pytest.raises(ValueError, match="another evaluation owns"):
            corpus.evaluate(tmp_path)
