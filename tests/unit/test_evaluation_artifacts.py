"""Returned evaluation diagnostics survive every engine route without execution."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import pytest

from sciel.engine.llm.client import LLMResponse
from sciel.engine.loop import Engine
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.eval.registry import load_task


CANDIDATE = "def run_search(): return (0.25, 0.75)\n"


class FixtureLLM:
    models = ["fixture"]
    weights = [1.0]

    def __init__(self):
        self.calls = 0

    def choose_model(self, rng):
        return self.models[0]

    def generate(self, *args, **kwargs):
        self.calls += 1
        return LLMResponse(f"```python\n{CANDIDATE}```", "fixture", .125, 10, 20)


def make_engine(tmp_path):
    task = load_task(str(Path(__file__).resolve().parents[2] / "tasks/test/toy_quadratic"))
    cfg = dict(task="test/toy_quadratic", num_islands=1, budget=1, dup_cap=1,
               embed_grid=False, embed_niche=False, three_head=False,
               portfolio_seed=False, idea_island=False, dynamic_islands=False,
               crossover=False, policy={"enabled": False}, milestone_fracs=[],
               op_fixed="full_rewrite", llm={"mock": True}, confirm_records=True)
    engine = Engine(cfg, task, str(tmp_path), FixtureLLM(), python=sys.executable)
    # Exercise the identified-run persistence path while calling each route
    # directly; a full run would initialize this identity in run_start.
    engine.log.run_id = 'evaluation-artifact-fixture'
    return engine


def events(tmp_path):
    return [json.loads(line) for line in (tmp_path / "sciel_events.jsonl").read_text().splitlines()]


@pytest.mark.parametrize("route", ["seed", "standard", "portfolio", "spawn"])
@pytest.mark.parametrize("kind", ["valid", "invalid_geometry", "eval_failed"])
def test_complete_returned_diagnostics_once_before_admission(tmp_path, monkeypatch, route, kind):
    engine = make_engine(tmp_path)
    initial = Path(engine.task.initial_program_path).read_text()
    if route != "seed":
        engine.seed_score = .4
        engine.db.add(initial, .4, {"validity": 1.0}, 0, origin="seed")
    artifacts = {"diagnostic": {"numbers": [0, 1.25, None], "flag": True,
                               "nested": {"reported_sum": "2.123"}},
                 "long_text": "begin:" + "x" * 10000 + ":untruncated-end"}
    if kind == "eval_failed":
        artifacts.update(error="fixture timeout", error_type="TimeoutError")
    result = EvalResult(ok=kind != "eval_failed", score=.7 if kind == "valid" else 0.,
                        cases={"validity": float(kind == "valid")}, artifacts=artifacts,
                        timing=.0125, stderr="fixture stderr", cost=.03,
                        confirmed_sampled=kind == "valid")
    outcome = CascadeOutcome(result, raw_first_score=.6 if kind == "valid" else 0.,
                             reached="confirm" if kind == "valid" else "full",
                             confirmed=kind == "valid")
    original = deepcopy(outcome)
    seen = []

    def fake_cascade(task, path, *, record_candidate, python):
        assert record_candidate is True and python == sys.executable
        assert task is engine.task
        data = Path(path).read_bytes()
        assert data.decode() == (initial if route == "seed" else CANDIDATE)
        assert not [e for e in events(tmp_path) if e.get("policy") == "evaluation"]
        seen.append((path, data, engine.db.state(), engine.B_evals, engine.iteration_no))
        return outcome

    monkeypatch.setattr("sciel.engine.loop.cascade_evaluate", fake_cascade)
    try:
        if route == "seed":
            engine.seed()
        elif route == "standard":
            engine._step(0)
        else:
            assert engine._paradigm_child(initial, "fixture paradigm", 0, route) is result.ok
    finally:
        engine.log.close()

    assert len(seen) == 1 and not Path(seen[0][0]).exists()
    assert outcome == original  # Logging must not normalize or mutate evaluator data.
    recorded = events(tmp_path)
    diagnostics = [e for e in recorded if e.get("policy") == "evaluation"]
    assert len(diagnostics) == 1
    event = diagnostics[0]
    evaluation_id = event['evaluation_id']
    assert evaluation_id.startswith('evaluation:' + engine.log.run_id + ':')
    from datetime import datetime
    assert datetime.fromisoformat(event['timestamp'].replace('Z', '+00:00')).tzinfo is not None
    assert {k: v for k, v in event.items() if k != 'timestamp'} == {"event": "decision", "policy": "evaluation", "action": "result",
                     "schema": "sciel-evaluation-result/1",
                     "evaluation_id": evaluation_id,
                     "source_sha256": hashlib.sha256(seen[0][1]).hexdigest(),
                     "iteration_no_before": seen[0][4], "B_evals_before": seen[0][3],
                     **result.to_json(), "raw_first_score": outcome.raw_first_score,
                     "reached": outcome.reached, "confirmed": outcome.confirmed}
    assert "child_id" not in event and "path" not in event
    assert event["artifacts"] == artifacts
    assert event["artifacts"]["long_text"].endswith(":untruncated-end")
    from sciel.engine.research_artifacts import saved_evaluation, saved_evaluation_for_program
    saved = saved_evaluation(engine, seen[0][1].decode(), evaluation_id)
    assert saved['status'] == 'available'
    assert saved['record']['result'] == result.to_json()
    assert saved['record']['evaluation_id'] == evaluation_id
    if result.ok:
        admitted = max(engine.db._all.values(), key=lambda program: int(program.id[1:]))
        assert saved_evaluation_for_program(engine, admitted)['reference'] == saved['reference']
    if route == "seed":
        assert engine.seed_score == result.score and len(engine.db._all) == 1
        assert engine.B_evals == engine.iteration_no == engine.llm.calls == 0
        assert engine.cost_total == 0
    else:
        assert engine.llm.calls == 1 and engine.cost_total == .125
        assert len(engine.db._all) == (2 if result.ok else 1)
        if route == "standard":
            assert engine.B_evals == int(result.ok) and engine.iteration_no == 1
            assert engine.n_eval_fail == int(not result.ok)
            terminal = next(e for e in recorded if e["event"] == "iteration")
        else:
            assert engine.B_evals == engine.iteration_no == 0
            terminal = next(e for e in recorded if e.get("policy") in
                            ("portfolio_seed", "paradigm_validation"))
        assert recorded.index(event) < recorded.index(terminal)


def test_cascade_exception_does_not_fabricate_result_and_temp_source_is_removed(tmp_path, monkeypatch):
    engine = make_engine(tmp_path)
    seen = []

    def fail(task, path, **kwargs):
        seen.append(path)
        raise RuntimeError("fixture cascade exception")

    monkeypatch.setattr("sciel.engine.loop.cascade_evaluate", fail)
    try:
        with pytest.raises(RuntimeError, match="fixture cascade exception"):
            engine._evaluate(CANDIDATE)
    finally:
        engine.log.close()
    assert len(seen) == 1 and not Path(seen[0]).exists()
    assert not [e for e in events(tmp_path) if e.get("policy") == "evaluation"]
    assert engine.B_evals == engine.iteration_no == engine.cost_total == 0
