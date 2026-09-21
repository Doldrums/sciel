"""Engine lifecycle telemetry with a real local evaluator and unpaid MockLLM."""
import hashlib
import json
from pathlib import Path
import sys

import pytest

from sciel.campaign.budget import CampaignStopped
from sciel.engine.factory import build_engine
from sciel.engine.llm.mock import MockLLM
from sciel.engine.runtime import process_matches


ROOT = Path(__file__).resolve().parents[2]


def engine(directory, llm=None):
    cfg = dict(task="test/toy_quadratic", budget=2, max_iterations=4, num_islands=1,
        embed_grid=False, portfolio_seed=False, idea_island=False, dynamic_islands=False,
        crossover=False, policy={"enabled": False}, milestone_fracs=[],
        schedule_policy="round_robin", llm={"mock": True})
    return build_engine(cfg, str(ROOT / "tasks"), str(directory),
                        python=sys.executable, llm=llm or MockLLM())


def test_actual_mock_search_records_observed_stages_and_matching_terminal_identity(tmp_path, monkeypatch):
    from sciel.engine import loop
    seen = []
    runtime = tmp_path / "runtime.json"

    class ObservedMock(MockLLM):
        def generate(self, *args, **kwargs):
            record = json.loads(runtime.read_text())
            assert record["status"] == "running"
            assert record["stage"] == "generating"
            assert process_matches(record) is True
            events = [json.loads(line) for line in
                      (tmp_path / "sciel_events.jsonl").read_text().splitlines()]
            seeds = [row for row in events if row.get("action") == "seed_program"]
            assert len(seeds) == 1
            assert seeds[0]["source"] == Path(search.task.initial_program_path).read_text()
            assert seeds[0]["measured"] is True
            assert seeds[0]["score"] == search.seed_score
            seen.append("generating")
            return super().generate(*args, **kwargs)

    original = loop.cascade_evaluate

    def observed_evaluate(*args, **kwargs):
        record = json.loads(runtime.read_text())
        assert record["stage"] == "evaluating"
        assert record["run_id"] and record["segment_id"]
        seen.append("evaluating")
        return original(*args, **kwargs)

    monkeypatch.setattr(loop, "cascade_evaluate", observed_evaluate)
    search = engine(tmp_path, ObservedMock())
    result = search.run()
    record = json.loads(runtime.read_text())
    rows = [json.loads(line) for line in (tmp_path / "sciel_events.jsonl").read_text().splitlines()]
    assert {"generating", "evaluating"}.issubset(seen)
    assert record["status"] == "completed"
    assert record["reason"] == result["reason"]
    assert record["B_evals"] == result["B_evals"]
    assert record["run_id"] == rows[-1]["run_id"]
    assert record["segment_id"] == rows[-1]["segment_id"]
    evaluator = rows[0]["evaluator"]
    assert evaluator["backend"] == "subprocess"
    assert evaluator["scope"] == "ordinary_evaluator_process"
    assert evaluator["full_timeout"] == search.task.full_timeout
    assert evaluator["primary_metric"] == search.task.primary_metric
    assert evaluator["confirm_enabled"] is False
    seed = next(row for row in rows if row.get("action") == "seed_program")
    assert seed["schema"] == "sciel-initialization/1"
    assert seed["program_id"] == "p1"
    assert seed["island"] == seed["B_evals"] == seed["iteration_no"] == 0
    assert seed["source_sha256"] == hashlib.sha256(seed["source"].encode()).hexdigest()
    assert seed["score"] == seed["seed_score"] == search.seed_score
    assert search._runtime_status is None


def test_failed_ordinary_seed_does_not_present_penalty_as_measured_score(tmp_path):
    from sciel.eval.cascade import CascadeOutcome
    from sciel.eval.facade import EvalResult
    search = engine(tmp_path)
    search._evaluate = lambda code: CascadeOutcome(EvalResult(False, 0.0), 0.0, "smoke")
    try:
        search.seed()
        rows = [json.loads(line) for line in (tmp_path / "sciel_events.jsonl").read_text().splitlines()]
        seed = next(row for row in rows if row.get("action") == "seed_program")
        assert seed["measured"] is False
        assert seed["score"] is seed["seed_score"] is None
        assert seed["program_id"] == "p1" and seed["source"]
        assert search.db.get("p1").score == search.seed_score == 0.0
    finally:
        search.log.close()


@pytest.mark.parametrize("exception,status", [
    (RuntimeError("sensitive failure text"), "failed"),
    (KeyboardInterrupt(), "interrupted"),
    (CampaignStopped("sensitive stop text"), "stopped"),
])
def test_engine_exit_records_failure_type_without_error_contents(tmp_path, exception, status):
    search = engine(tmp_path)
    def fail():
        raise exception
    search._run_dispatch = fail
    try:
        with pytest.raises(type(exception)):
            search.run()
        raw = (tmp_path / "runtime.json").read_text()
        record = json.loads(raw)
        assert record["status"] == status
        assert record["error_type"] == type(exception).__name__
        assert record["finished_at"]
        assert "sensitive" not in raw
        assert search._runtime_status is None
    finally:
        search.log.close()
