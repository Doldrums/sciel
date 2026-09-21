"""Offline consumers use selected ancestry and report every provenance skip.

Selection itself is covered by journal replay tests. The contract fixtures here
leave contradictory raw events on disk so an accidental raw-file join fails.
No candidate execution or model training is needed.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def module(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def start():
    return {"event": "run_start", "schema_version": 1, "task": "circle",
            "seed": 1, "config": {"arm": "pool"}}


def step(number, score=.3, **extra):
    return {"event": "iteration", "iteration_no": number, "B_evals": number,
            "child_id": f"p{number}", "child_score": score, "island": 0,
            "island_best": score, "d_isl": score, "is_pop_event": False,
            **extra}


def end(number):
    return {"event": "run_end", "iteration_no": number, "B_evals": number,
            "reason": "budget", "pop_best": .8}


def journal(tmp_path, rows):
    path = tmp_path / "sciel_events.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return str(path)


def test_dataset_extract_uses_selected_events_even_when_raw_suffix_disagrees(tmp_path, monkeypatch):
    consumer = module("dataset_pipeline")
    path = journal(tmp_path, [start(), step(1, is_island_event=False), end(1)])
    selected = [start(), step(1, is_island_event=True, op_used="diff"), end(1)]
    monkeypatch.setattr(consumer, "completed_events", lambda p: (selected, selected[-1]))
    rows = consumer.extract(path)
    assert len(rows) == 1 and rows[0]["y"] == 1.0 and rows[0]["op_diff"] == 1.0


@pytest.mark.parametrize("rows, reason", [
    ([start(), step(1)], "no run_end"),
    ([start(), step(1), {**end(1), "reason": "paused"}], "paused"),
    ([{"event": "run_start"}, step(1), end(1)], "not a v1 journal"),
    ([start(), step(1), step(1), end(1)], "rollback|repeated"),
])
def test_dataset_extract_rejects_unusable_trajectory(tmp_path, rows, reason):
    consumer = module("dataset_pipeline")
    with pytest.raises(consumer.DirtyRun, match=reason):
        consumer.extract(journal(tmp_path, rows))


def test_dataset_cli_reports_malformed_journal(tmp_path, capsys):
    consumer = module("dataset_pipeline")
    (tmp_path / "sciel_events.jsonl").write_text('{"event":')
    with pytest.raises(SystemExit, match="no gen rows"):
        consumer.main(["dataset_pipeline.py", str(tmp_path)])
    assert "malformed journal" in capsys.readouterr().err


def test_record_ratio_counts_selected_branch_and_legacy_gain(tmp_path, monkeypatch):
    consumer = module("record_ratio")
    path = journal(tmp_path, [start()] + [step(i, is_pop_event=True)
                                       for i in range(1, 21)] + [end(20)])
    selected = [start()]
    for i in range(1, 21):
        row = step(i, d_pop=.1 if i == 3 else 0.)
        row.pop("is_pop_event")
        selected.append(row)
    selected.append(end(20))
    monkeypatch.setattr(consumer, "completed_events", lambda p: (selected, selected[-1]))
    result = consumer.ratio_for(path)
    assert result["records"] == 1 and result["n"] == 20
    assert result["task"] == "circle" and result["arm"] == "pool"


@pytest.mark.parametrize("indices, final_n", [
    (list(range(1, 21)) + [20], 20),
    (list(range(1, 20)), 20),
    (list(range(1, 22)), 20),
])
def test_record_ratio_never_deduplicates_or_truncates_selected_children(
        tmp_path, monkeypatch, capsys, indices, final_n):
    consumer = module("record_ratio")
    selected = [start()] + [step(i) for i in indices] + [end(final_n)]
    monkeypatch.setattr(consumer, "completed_events", lambda p: (selected, selected[-1]))
    assert consumer.ratio_for(str(tmp_path / "unused.jsonl")) is None
    assert "B_evals" in capsys.readouterr().err


def test_record_ratio_reports_malformed_input_instead_of_ignoring_line(tmp_path, capsys):
    consumer = module("record_ratio")
    path = journal(tmp_path, [start()] + [step(i) for i in range(1, 21)] + [end(20)])
    with open(path, "a") as fh:
        fh.write('{"event":\n')
    assert consumer.ratio_for(path) is None
    assert "malformed journal" in capsys.readouterr().err


@pytest.mark.parametrize("consumer_name", ["lineage_labels", "train_code_embedder"])
def test_source_consumers_use_selected_code_and_report_missing_source(
        tmp_path, monkeypatch, capsys, consumer_name):
    if consumer_name == "train_code_embedder":
        pytest.importorskip("torch")
    consumer = module(consumer_name)
    path = journal(tmp_path, [start(), step(1, child_code="stale suffix"), end(1)])
    selected = [start(), step(1, child_code="def solve(): return 7"),
                step(2, response="def solve(): return 99"), end(2)]
    monkeypatch.setattr(consumer, "completed_events", lambda p: (selected, selected[-1]))
    monkeypatch.setattr(consumer, "checkpoint_programs", lambda p, **kw: {})
    if consumer_name == "lineage_labels":
        records = consumer.labels_for(path, with_code=True)
        assert [(r["id"], r["code"]) for r in records] == [
            ("p1", "def solve(): return 7")]
    else:
        observed = []
        monkeypatch.setattr(consumer, "featurize",
                            lambda code, cases: observed.append(code) or [1.])
        assert len(consumer.load([str(tmp_path)])[0]) == 1
        assert observed == ["def solve(): return 7"]
    assert "child p2: missing evaluated source" in capsys.readouterr().err


@pytest.mark.parametrize("consumer_name", ["lineage_labels", "train_code_embedder"])
def test_source_consumers_report_rejected_checkpoint_binding(
        tmp_path, monkeypatch, capsys, consumer_name):
    if consumer_name == "train_code_embedder":
        pytest.importorskip("torch")
    consumer = module(consumer_name)
    path = journal(tmp_path, [start(), step(1), end(1)])

    def rejected(_, **kwargs):
        raise consumer.DirtyRun("checkpoint does not bind selected final state")

    monkeypatch.setattr(consumer, "checkpoint_programs", rejected)
    if consumer_name == "lineage_labels":
        assert consumer.labels_for(path, with_code=True) == []
    else:
        assert len(consumer.load([str(tmp_path)])[0]) == 0
    assert "checkpoint does not bind selected final state" in capsys.readouterr().err


def test_lineage_reports_dirty_run(tmp_path, capsys):
    consumer = module("lineage_labels")
    path = journal(tmp_path, [start(), step(1)])
    assert consumer.labels_for(path) == []
    assert "no run_end" in capsys.readouterr().err


def test_three_head_eviction_targets_exclude_abandoned_decision(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    consumer = module("train_three_head")
    raw = [start(), {"event": "decision", "policy": "eviction", "victim_id": "p1"},
           end(0)]
    journal(tmp_path, raw)
    selected = [start(), {"event": "decision", "policy": "eviction", "victim_id": "p2"},
                end(0)]
    monkeypatch.setattr(consumer, "completed_events", lambda p: (selected, selected[-1]))
    monkeypatch.setattr(consumer, "labels_for", lambda *a, **kw: [
        {"id": "p1", "island": 0, "subtree_max": .7},
        {"id": "p2", "island": 0, "subtree_max": .8}])
    monkeypatch.setattr(consumer, "_embed", lambda *a: np.eye(2))
    sets, labels = consumer.build_sets(None, [str(tmp_path)], min_set=2)
    assert len(sets) == 1
    np.testing.assert_array_equal(labels[0]["eviction"], [1., 0.])


def test_three_head_skips_dirty_eviction_source(tmp_path, monkeypatch, capsys):
    pytest.importorskip("torch")
    consumer = module("train_three_head")
    journal(tmp_path, [start()])
    monkeypatch.setattr(consumer, "labels_for", lambda *a, **kw: [{"id": "p1"}])
    assert consumer.build_sets(None, [str(tmp_path)]) == ([], [])
    assert "no run_end" in capsys.readouterr().err


def test_mode_trainer_prints_specific_corpus_skip_reasons(monkeypatch, capsys):
    consumer = module("train_mode_policy")
    monkeypatch.setattr(consumer, "load_corpus", lambda dirs: ([], {
        "loaded": [], "skipped": [("broken-run", "checkpoint prefix hash mismatch")]}))
    monkeypatch.setattr(consumer, "held_out_split", lambda *a, **kw: ([], []))
    consumer.main("unused.npz", ["broken-run"])
    assert "skip broken-run: checkpoint prefix hash mismatch" in capsys.readouterr().err


@pytest.mark.parametrize("case, reason", [
    ("extra_restore", "unexpected checkpoint_restore"),
    ("future_checkpoint", "terminal run_end|matching checkpoint identity"),
])
def test_v2_reader_rejects_unpaired_restore_and_future_end_binding(tmp_path, case, reason):
    """Both journals were accepted despite semantically impossible event order."""
    from sciel.run_log import RunLog

    reader = module("journal_data")
    path = tmp_path / "sciel_events.jsonl"
    log = RunLog(str(path))
    counters = {"iteration_no": 0, "B_evals": 0}
    checkpoint_id, checkpoint_hash = "saved-1", "a" * 64
    try:
        log.run_start(counters=counters)
        if case == "extra_restore":
            log.emit("checkpoint_restore", ancestry="legacy_unbound",
                     restored_counters={"iteration_no": 99, "B_evals": 99})
        else:
            # A later declaration must not retroactively validate this run_end.
            log.run_end(**counters, reason="budget", checkpoint_id=checkpoint_id,
                        checkpoint_sha256=checkpoint_hash)
        provenance = {"schema": reader.PROVENANCE_SCHEMA,
                      "checkpoint_id": checkpoint_id, "run_id": log.run_id,
                      "counters": counters, "journal_anchor": log.anchor()}
        log.checkpoint(**counters, checkpoint_id=checkpoint_id,
                       checkpoint_sha256=checkpoint_hash, provenance=provenance)
        log.run_end(**counters, reason="budget", checkpoint_id=checkpoint_id,
                    checkpoint_sha256=checkpoint_hash)
    finally:
        log.close()
    with pytest.raises(reader.DirtyRun, match=reason):
        reader.completed_events(str(path))
