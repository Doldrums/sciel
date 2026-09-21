"""Exhausted generation batches retain their paid cost in the iteration journal."""

import json
from pathlib import Path
import sys

import pytest

from sciel.engine.llm.client import LLMResponse
from sciel.engine.loop import Engine
from sciel.eval.registry import load_task


@pytest.mark.parametrize("reason", ["gate_rejected", "dup_cap"])
def test_nochild_journal_cost_matches_paid_attempts_and_engine_total(tmp_path, reason):
    parent = "def run_search(): return (100, 0)\n"
    candidate = "# formatting only\n" + parent if reason == "gate_rejected" else parent

    class LocalLLM:
        models = ["mock"]
        weights = [1.0]

        def __init__(self):
            self.calls = 0

        def choose_model(self, rng):
            return "mock"

        def generate(self, *args, **kwargs):
            self.calls += 1
            return LLMResponse(text=f"```python\n{candidate}```", model="mock",
                               cost=0.125, prompt_tokens=10, completion_tokens=20)

    root = Path(__file__).resolve().parents[2]
    task = load_task(str(root / "tasks/test/toy_quadratic"))
    llm = LocalLLM()
    config = dict(task="test/toy_quadratic", num_islands=1, budget=1, dup_cap=2,
                  embed_grid=False, portfolio_seed=False, idea_island=False,
                  dynamic_islands=False, policy={"enabled": False}, milestone_fracs=[],
                  op_fixed="full_rewrite", llm={"mock": True},
                  container={"kind": "pool_novelty" if reason == "gate_rejected" else "pool"})
    engine = Engine(config, task, str(tmp_path), llm, python=sys.executable)
    try:
        engine.seed_score = 0.1
        engine.db.add(parent, 0.1, {}, 0, "seed")
        engine._evaluate = lambda code: pytest.fail("rejected source reached evaluator")
        engine._step(0)
        assert llm.calls == 2
        assert engine.B_evals == 0 and engine.n_nochild == 1
        assert engine.cost_total == pytest.approx(0.25)
        assert engine.schedule.saturation_counter[0] == int(reason == "dup_cap")
    finally:
        engine.log.close()

    events = [json.loads(line) for line in (tmp_path / "sciel_events.jsonl").read_text().splitlines()]
    iterations = [event for event in events if event["event"] == "iteration"]
    assert len(iterations) == 1 and iterations[0]["reason"] == reason
    assert iterations[0]["meta"]["dup_attempts"] == 2
    assert iterations[0]["cost"] == pytest.approx(0.25)
    assert sum(event["cost"] for event in iterations) == engine.cost_total
