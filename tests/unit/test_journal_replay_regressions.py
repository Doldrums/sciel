"""Offline readers must not join abandoned suffixes to a checkpoint replay."""

import importlib.util
import json
from pathlib import Path
from types import MethodType

import pytest

import conftest
from sciel.engine import checkpoint
from sciel.engine.factory import build_engine
from sciel.engine.llm.mock import MockLLM


def _module(name):
    path = Path(__file__).resolve().parents[2] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


journal = _module("journal_data")


def _write(tmp_path, events):
    path = tmp_path / "sciel_events.jsonl"
    path.write_text("".join(json.dumps(event) + "\n" for event in events))
    return str(path)


def _start():
    return {"event": "run_start", "schema_version": 1}


def _step(number, child=None):
    return {"event": "iteration", "iteration_no": number, "B_evals": number,
            "child_id": child or f"p{number}", "child_score": number / 10}


def _end(number, reason="budget"):
    return {"event": "run_end", "iteration_no": number, "B_evals": number,
            "reason": reason, "pop_best": number / 10}


def test_sequential_resume_keeps_prefix_and_uses_latest_end(tmp_path):
    rows = [_start(), _step(1), _end(1, "paused"), _start(), _step(2), _end(2)]
    events, end = journal.completed_events(_write(tmp_path, rows))
    assert [e["child_id"] for e in events if e["event"] == "iteration"] == ["p1", "p2"]
    assert end == rows[-1]


@pytest.mark.parametrize("tail", [
    [_start(), _step(2), _end(2)],  # a replay must not retain the previous p2
    [_start(), _step(4), _end(4)],  # a missing prefix is not evidence of continuation
    [_start(), _end(2)],           # no first iteration to establish restored state
])
def test_ambiguous_resumed_branches_are_rejected(tmp_path, tail):
    rows = [_start(), _step(1), _step(2), _end(2)] + tail
    with pytest.raises(journal.DirtyRun, match="resumed segment"):
        journal.completed_events(_write(tmp_path, rows))


def test_checkpoint_at_matching_number_does_not_prove_restore_provenance(tmp_path):
    rows = [_start(), _step(1), {"event": "checkpoint", "iteration_no": 1,
            "B_evals": 1, "path": "/same/overwritten/checkpoint.json"},
            _step(2), _end(2), _start(), _step(2), _end(2)]
    with pytest.raises(journal.DirtyRun, match="checkpoint branch is unknown"):
        journal.completed_events(_write(tmp_path, rows))


def test_duplicate_child_id_is_not_accepted_even_with_monotone_counters(tmp_path):
    rows = [_start(), _step(1, "p7"), _step(2, "p7"), _end(2)]
    with pytest.raises(journal.DirtyRun, match="child_id"):
        journal.completed_events(_write(tmp_path, rows))


def test_counter_rollback_without_run_start_is_rejected(tmp_path):
    rows = [_start(), _step(1), _step(2), _step(1, "p8"), _end(1)]
    with pytest.raises(journal.DirtyRun, match="rollback"):
        journal.completed_events(_write(tmp_path, rows))


def test_legacy_missing_iteration_numbers_do_not_prove_a_resume(tmp_path):
    rows = [_start(), {"event": "iteration", "child_id": "p1"}, _end(1),
            _start(), {"event": "iteration", "child_id": "p2"}, _end(2)]
    with pytest.raises(journal.DirtyRun, match="checkpoint provenance"):
        journal.completed_events(_write(tmp_path, rows))


def test_no_schema_is_invented_for_unknown_restore_events(tmp_path):
    rows = [_start(), _step(1), {"event": "checkpoint_restore", "iteration_no": 0},
            _step(2), _end(2)]
    with pytest.raises(journal.DirtyRun, match="unsupported checkpoint_restore"):
        journal.completed_events(_write(tmp_path, rows))


def test_non_object_json_records_are_dirty_runs(tmp_path):
    path = _write(tmp_path, [_start(), ["not an event"], _end(0)])
    with pytest.raises(journal.DirtyRun, match="records must be objects"):
        journal.completed_events(path)


def _fast_engine(run_dir, offset):
    """Exercise real run/checkpoint producers without evaluation or model calls."""
    cfg = conftest.mock_config(budget=4)
    cfg.update(num_islands=1, milestone_fracs=[.25, .75], kb=False,
               policy={"enabled": False}, embed_grid=False)
    engine = build_engine(cfg, conftest.TASKS_ROOT, str(run_dir),
                          python=conftest.PYTHON, llm=MockLLM())

    def step(self, island):
        self.iteration_no += 1
        self.B_evals += 1
        score = offset + self.iteration_no / 10
        parent = self.db.pop_best_id
        child = self.db.add(f"def solve(): return {score}\n", score, {}, island,
                            origin="gen", parent_id=parent)
        self.log.iteration(iteration_no=self.iteration_no, B_evals=self.B_evals,
                           island=island, child_id=child.id, parent_id=parent,
                           child_score=score)

    engine._step = MethodType(step, engine)
    return engine


def test_actual_checkpoint_rollback_selects_bound_prefix_and_correct_reused_source(tmp_path):
    engine = _fast_engine(tmp_path, 0.0)
    engine.db.add("def solve(): return 0\n", 0.0, {}, 0, origin="seed")
    earlier = tmp_path / "earlier.json"
    original_milestone = engine._maybe_milestone

    def capture_checkpoint():
        original_milestone()
        if engine.iteration_no == 1:
            earlier.write_bytes((tmp_path / "checkpoint.json").read_bytes())

    engine._maybe_milestone = capture_checkpoint
    engine.run()
    replay = _fast_engine(tmp_path, 0.5)
    checkpoint.load(replay, str(earlier))
    replay.run()

    path = tmp_path / "sciel_events.jsonl"
    raw = [json.loads(line) for line in path.read_text().splitlines()]
    starts = [e for e in raw if e["event"] == "run_start"]
    assert len(starts) == 2 and all("counters" in e for e in starts)
    assert len({e["segment_id"] for e in starts}) == 2
    restored = next(e for e in raw if e["event"] == "checkpoint_restore")
    assert restored["restored_counters"]["iteration_no"] == 1
    assert restored["checkpoint_id"] == json.loads(earlier.read_bytes())["provenance"]["checkpoint_id"]
    p3 = [e for e in raw if e.get("child_id") == "p3"]
    assert [e["child_score"] for e in p3] == pytest.approx([.2, .7])
    # The old reader returned both p3 rows. Fallback then paired the first row's
    # .2 score with final checkpoint source that actually evaluated to .7.
    programs = journal.checkpoint_programs(str(path))
    assert programs["p3"]["score"] == pytest.approx(.7)
    selected, end = journal.completed_events(str(path))
    selected_p3 = [e for e in selected if e.get("child_id") == "p3"]
    assert len(selected_p3) == 1 and selected_p3[0]["child_score"] == pytest.approx(.7)
    labels = _module("lineage_labels").labels_for(str(path), with_code=True)
    p3_label = next(r for r in labels if r["id"] == "p3")
    assert p3_label["code"] == programs["p3"]["code"]
    assert p3_label["score"] == programs["p3"]["score"] == pytest.approx(.7)
