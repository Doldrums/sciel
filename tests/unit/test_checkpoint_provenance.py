"""Checkpoint ancestry using real producers, with no evaluator or LLM calls."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from sciel.engine import checkpoint
from sciel.run_log import RunLog

spec = importlib.util.spec_from_file_location("journal_fixtures", Path(__file__).with_name("test_journal_replay_regressions.py"))
fixtures = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixtures)
journal = fixtures.journal


def engine(folder, offset=0.):
    result = fixtures._fast_engine(folder, offset)
    result._evaluate = lambda *a, **k: pytest.fail("evaluator called")
    result.llm.generate = lambda *a, **k: pytest.fail("LLM called")
    original_log = result.log.iteration
    def log_source(**event):
        if event.get("child_id"):
            event["child_code"] = result.db.get(event["child_id"]).code
        original_log(**event)
    result.log.iteration = log_source
    return result


def fresh(folder):
    result = engine(folder)
    result.db.add("def solve(): return 0\n", 0., {}, 0, origin="seed")
    return result


def rollback_run(folder):
    initial = fresh(folder)
    initial._stop_at = 1
    initial.run()
    earlier = folder / "earlier.json"
    earlier.write_bytes((folder / "checkpoint.json").read_bytes())
    middle = engine(folder)
    checkpoint.load(middle, str(earlier))
    middle.run()
    final = engine(folder, .5)
    checkpoint.load(final, str(earlier))
    final.run()
    return earlier, final


def rows(folder):
    return [json.loads(line) for line in (folder / "sciel_events.jsonl").read_bytes().splitlines()]


def rewrite(folder, events):
    (folder / "sciel_events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))


def test_sequential_resume_uses_exact_checkpoint_hash_and_preserves_search_state(tmp_path):
    continuous = fresh(tmp_path / "continuous")
    continuous.run()
    first = fresh(tmp_path / "split")
    first._stop_at = 1
    first.run()
    data = (tmp_path / "split/checkpoint.json").read_bytes()
    expected = json.loads(data)
    second = engine(tmp_path / "split")
    checkpoint.load(second, str(tmp_path / "split/checkpoint.json"))
    second.run()
    events, end = journal.completed_events(tmp_path / "split/sciel_events.jsonl")
    restore = next(e for e in events if e["event"] == "checkpoint_restore")
    assert restore["checkpoint_sha256"] == hashlib.sha256(data).hexdigest()
    assert restore["restored_counters"] == expected["provenance"]["counters"]
    assert second.db.state() == continuous.db.state()
    assert second.rng.state() == continuous.rng.state()
    assert checkpoint.counters(second) == checkpoint.counters(continuous)
    assert [e["iteration_no"] for e in events if e["event"] == "iteration"] == [1, 2, 3, 4]
    assert end["B_evals"] == 4


def test_repeated_rollback_drops_abandoned_source_and_eviction_suffixes(tmp_path):
    earlier, final = rollback_run(tmp_path)
    more = engine(tmp_path, .8)
    checkpoint.load(more, str(earlier))
    more.run()
    selected, _ = journal.completed_events(tmp_path / "sciel_events.jsonl")
    generated = [e for e in selected if e.get("child_id")]
    assert len(generated) == 4
    assert [e["child_score"] for e in generated] == pytest.approx([.1, 1., 1.1, 1.2])
    assert len([e for e in rows(tmp_path) if e.get("child_id") == "p3"]) == 3
    assert len([e for e in generated if e.get("child_id") == "p3"]) == 1


def test_resume_at_completed_budget_needs_no_new_iteration_to_prove_ancestry(tmp_path):
    fresh(tmp_path).run()
    second = engine(tmp_path)
    checkpoint.load(second, str(tmp_path / "checkpoint.json"))
    second.run()
    selected, end = journal.completed_events(tmp_path / "sciel_events.jsonl")
    assert len([e for e in selected if e["event"] == "iteration"]) == 4
    assert len([e for e in selected if e["event"] == "run_start"]) == 2
    assert end["B_evals"] == 4


@pytest.mark.parametrize("change,reason", [
    ("restore_hash", "hash/provenance mismatch"),
    ("counter", "restored counters disagree"),
    ("prefix", "prefix hash mismatch"),
    ("sequence", "event_seq"),
    ("checkpoint_id", "declaration is missing"),
    ("final_hash", "matching checkpoint"),
])
def test_provenance_mutation_is_not_silently_joined(tmp_path, change, reason):
    rollback_run(tmp_path)
    events = rows(tmp_path)
    restore = [e for e in events if e["event"] == "checkpoint_restore"][-1]
    if change == "restore_hash":
        restore["checkpoint_sha256"] = "0" * 64
    elif change == "counter":
        restore["restored_counters"]["db_counter"] += 1
    elif change == "prefix":
        events[1]["child_score"] = 999
    elif change == "sequence":
        events[-1]["event_seq"] -= 1
    elif change == "checkpoint_id":
        restore["checkpoint_id"] = "absent"
    else:
        events[-1]["checkpoint_sha256"] = "0" * 64
    # Keep byte lengths/offsets stable for mutations after a parent's anchor.
    raw = (tmp_path / "sciel_events.jsonl").read_bytes()
    old_lines = raw.splitlines(keepends=True)
    original = rows(tmp_path)
    changed = [i for i, (before, after) in enumerate(zip(original, events)) if before != after]
    for i in changed:
        old_lines[i] = (json.dumps(events[i]) + "\n").encode()
    (tmp_path / "sciel_events.jsonl").write_bytes(b"".join(old_lines))
    # Mutating a restore before later checkpoints also invalidates those exact
    # byte-prefix anchors. Either mismatch must refuse the entire training run.
    with pytest.raises(journal.DirtyRun, match=reason + "|prefix hash mismatch|event boundary"):
        journal.completed_events(tmp_path / "sciel_events.jsonl")


def test_checkpoint_code_lookup_cannot_use_a_replaced_file_at_same_counters(tmp_path):
    fresh(tmp_path).run()
    selected, _ = journal.completed_events(tmp_path / "sciel_events.jsonl")
    original = json.loads((tmp_path / "checkpoint.json").read_bytes())
    original["db"]["programs"][-1]["code"] = "def solve(): return 'different'\n"
    (tmp_path / "checkpoint.json").write_text(json.dumps(original))
    with pytest.raises(journal.DirtyRun, match="not bound"):
        journal.checkpoint_programs(tmp_path / "sciel_events.jsonl")
    assert selected[-1]["B_evals"] == original["B_evals"]


def test_legacy_checkpoint_resume_is_operational_but_unbound_training_is_rejected(tmp_path):
    first = fresh(tmp_path)
    first._stop_at = 1
    first.run()
    path = tmp_path / "checkpoint.json"
    state = json.loads(path.read_bytes())
    state.pop("provenance")
    path.write_text(json.dumps(state))
    second = engine(tmp_path)
    checkpoint.load(second, str(path))
    second.run()
    assert second.B_evals == 4
    with pytest.raises(journal.DirtyRun, match="unbound ancestry"):
        journal.completed_events(tmp_path / "sciel_events.jsonl")


def test_copied_checkpoint_without_ancestor_journal_is_rejected(tmp_path):
    fresh(tmp_path / "source").run()
    second = engine(tmp_path / "destination")
    checkpoint.load(second, str(tmp_path / "source/checkpoint.json"))
    second.run()
    with pytest.raises(journal.DirtyRun, match="declaration is missing"):
        journal.completed_events(tmp_path / "destination/sciel_events.jsonl")


def test_failed_log_write_cannot_supply_checkpoint_anchor(tmp_path):
    log = RunLog(str(tmp_path / "events.jsonl"))
    log.run_start(counters={"iteration_no": 0, "B_evals": 0})
    log.close()
    assert log.emit("iteration", iteration_no=1, B_evals=1) is False
    with pytest.raises(OSError, match="unverifiable checkpoint"):
        log.anchor()


def test_open_or_truncated_restored_segment_does_not_inherit_old_completion(tmp_path):
    earlier, final = rollback_run(tmp_path)
    log = RunLog(str(tmp_path / "sciel_events.jsonl"))
    data = earlier.read_bytes()
    state = json.loads(data)
    restore = {"checkpoint_id": state["provenance"]["checkpoint_id"],
        "checkpoint_sha256": hashlib.sha256(data).hexdigest(), "provenance": state["provenance"],
        "restored_counters": state["provenance"]["counters"], "ancestry": "bound"}
    log.run_start(restored_checkpoint=restore, counters=restore["restored_counters"])
    log.close()
    with pytest.raises(journal.DirtyRun, match="no run_end"):
        journal.completed_events(tmp_path / "sciel_events.jsonl")
    with (tmp_path / "sciel_events.jsonl").open("ab") as stream:
        stream.write(b'{"event":')
    with pytest.raises(journal.DirtyRun, match="malformed"):
        journal.completed_events(tmp_path / "sciel_events.jsonl")


@pytest.mark.parametrize("state", [[], {"db": [1]}, {"db": {"programs": None}},
    {"db": {"programs": [{"id": "p1"}, {"id": "p1"}]}}])
def test_invalid_checkpoint_shapes_have_a_reportable_rejection(tmp_path, state):
    fixtures._write(tmp_path, [fixtures._start(), fixtures._step(1), fixtures._end(1)])
    (tmp_path / "checkpoint.json").write_text(json.dumps(state))
    with pytest.raises(journal.DirtyRun, match="checkpoint"):
        journal.checkpoint_programs(tmp_path / "sciel_events.jsonl")


def test_corpus_joins_only_selected_rollback_and_rejects_checkpoint_replacement(tmp_path):
    _, final = rollback_run(tmp_path / "run")
    corpus = fixtures._module("prepare_review_corpus")
    items, runs, excluded, traces, _ = corpus.collect([tmp_path], tasks=(final.task.qualified_name,))
    assert not excluded and len(runs) == 1
    p3 = next(item for item in items if any(m["program_id"] == "p3" for m in item["memberships"]))
    membership = p3["memberships"][0]
    assert membership["historical_score"] == pytest.approx(.7)
    assert "0.7" in p3["sources"][membership["source_sha256"]]
    assert len([r for r in traces if r["child_id"] == "p3"]) == 1
    path = tmp_path / "run/checkpoint.json"
    state = json.loads(path.read_bytes())
    state["db"]["programs"][-1]["code"] = "changed source"
    path.write_text(json.dumps(state))
    items, runs, excluded, traces, _ = corpus.collect([tmp_path], tasks=(final.task.qualified_name,))
    assert not items and not runs and not traces
    assert "not bound" in excluded[0]["reason"]




def test_terminal_score_mutation_cannot_poison_return_to_go_labels(tmp_path):
    fresh(tmp_path).run()
    events = rows(tmp_path)
    events[-1]["pop_best"] = 999
    rewrite(tmp_path, events)
    with pytest.raises(journal.DirtyRun, match="pop_best differs"):
        journal.completed_events(tmp_path / "sciel_events.jsonl")
    loader = fixtures._module("dt_dataloader")
    with pytest.raises(loader.DirtyRun, match="pop_best differs"):
        loader.build_samples(tmp_path / "sciel_events.jsonl")


def test_source_lookup_remains_bound_to_first_snapshot_during_concurrent_resume(tmp_path):
    first = fresh(tmp_path)
    first.run()
    selected, original_end = journal.completed_events(tmp_path / "sciel_events.jsonl")
    earlier = tmp_path / "earlier.json"
    # Another writer can publish a newer final checkpoint between consumer reads.
    earlier.write_bytes((tmp_path / "checkpoint.json").read_bytes())
    second = engine(tmp_path)
    checkpoint.load(second, str(earlier))
    second.run()
    assert journal.completed_events(tmp_path / "sciel_events.jsonl")[1] != original_end
    with pytest.raises(journal.DirtyRun, match="not bound"):
        journal.checkpoint_programs(tmp_path / "sciel_events.jsonl", end=original_end)
    assert selected[-1] == original_end


def test_malformed_start_counters_are_a_reportable_dirty_run(tmp_path):
    log = RunLog(str(tmp_path / "sciel_events.jsonl"))
    log.run_start(counters=[1])
    log.close()
    with pytest.raises(journal.DirtyRun, match="counters shape"):
        journal.trajectory_events(tmp_path / "sciel_events.jsonl", require_complete=False)
