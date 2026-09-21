"""Unbounded ordinary search, explicit stops and checkpoint continuation.

These tests use a real local toy evaluator and unpaid MockLLM. The provider has
an independent abort guard so a broken stop condition cannot hang the suite.
"""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

from sciel.config import validate
from sciel.engine import checkpoint
from sciel.engine.factory import build_engine
from sciel.engine.llm.client import LLMResponse
from sciel.engine.llm.mock import MockLLM


ROOT = Path(__file__).resolve().parents[2]


class TestAbort(BaseException):
    __test__ = False


class PeriodicInterruption(Exception):
    pass


class StoppingMock(MockLLM):
    def __init__(self, stop_at=6, *, fail=False):
        super().__init__(dup_modulo=2**65, fail_modulo=2**65)
        self.engine = None
        self.stop_at = stop_at
        self.fail = fail
        self.calls = 0

    def generate(self, *args, **kwargs):
        self.calls += 1
        if self.calls > 12:
            raise TestAbort("STOP was ignored; aborting this bounded test")
        if self.engine.iteration_no + 1 >= self.stop_at:
            (Path(self.engine.run_dir) / "STOP").write_text("test requested pause\n")
        if self.fail:
            return LLMResponse(text="```python\ndef run_search():\n    raise RuntimeError('fixture')\n```\n",
                               model="mock-model", cost=0, prompt_tokens=1, completion_tokens=1)
        return super().generate(*args, **kwargs)


def settings(**changes):
    cfg = dict(task="test/toy_quadratic", budget=None, max_iterations=None,
        schedule_horizon=2, checkpoint_interval=3, backstop_k=1,
        random_seed=131, num_islands=1, inflight_cap=1, dup_cap=1,
        schedule_policy="sh", milestone_fracs=[.5, .75], sh_min_fair_evals=0,
        embed_grid=False, embed_niche=False, portfolio_seed=False,
        dynamic_islands=False, idea_island=False, crossover=False,
        op_fixed="full_rewrite", policy={"enabled": False, "plateau_shift": False},
        llm={"mock": True}, wandb={"enabled": False})
    cfg.update(changes)
    return cfg


def construct(directory, cfg=None, *, stop_at=6, fail=False):
    provider = StoppingMock(stop_at, fail=fail)
    engine = build_engine(cfg or settings(), str(ROOT / "tasks"), str(directory),
                          python=sys.executable, llm=provider)
    provider.engine = engine
    return engine, provider


def journal(directory):
    return [json.loads(line) for line in (Path(directory) / "sciel_events.jsonl").read_text().splitlines()]


def semantic_state(engine):
    return dict(db=engine.db.state(), schedule=engine.schedule.state(),
        selection=engine.selection.state(), operators=engine.op_bandit.state(),
        reward=engine.reward.state(), rng=engine.rng.state(),
        B_evals=engine.B_evals, iteration_no=engine.iteration_no,
        seed_score=engine.seed_score, cost_total=engine.cost_total,
        noop_parents=sorted(engine.noop_parents), milestones=sorted(engine.milestones_fired))


def test_unlimited_crosses_horizon_and_backstop_then_consumes_stop_after_current_child(tmp_path):
    engine, provider = construct(tmp_path)
    summary = engine.run()
    rows = journal(tmp_path)
    assert summary["reason"] == "paused"
    assert engine.budget is None
    assert summary["iteration_no"] == provider.calls == 6
    assert summary["B_evals"] > engine.cfg["schedule_horizon"] * engine.cfg["backstop_k"]
    assert not (tmp_path / "STOP").exists()
    iterations = [row for row in rows if row["event"] == "iteration"]
    assert iterations[-1]["iteration_no"] == 6
    assert iterations[-1].get("child_id")
    assert all(0 <= row["progress_pre"] <= 1 for row in iterations if "progress_pre" in row)
    assert iterations[-1]["progress_pre"] == 1
    assert [row["milestone"] for row in rows if row["event"] == "sh_cut"] == [1, 2]
    saved = json.loads((tmp_path / "checkpoint.json").read_text())
    assert saved["budget"] is saved["config"]["budget"] is None
    assert saved["iteration_no"] == 6


def test_periodic_checkpoint_is_recoverable_without_replaying_or_changing_search(tmp_path):
    whole, _ = construct(tmp_path / "whole")
    whole.run()
    split, split_provider = construct(tmp_path / "split")
    original = split.log.checkpoint

    def interrupt_after_periodic_record(**fields):
        original(**fields)
        # SH occurs at 1 and 2; an iteration-3 save is the periodic checkpoint.
        if split.iteration_no == 3:
            raise PeriodicInterruption("simulated process interruption after durable checkpoint")

    split.log.checkpoint = interrupt_after_periodic_record
    try:
        with pytest.raises(PeriodicInterruption):
            split.run()
    finally:
        split.log.close()
    saved = json.loads((tmp_path / "split/checkpoint.json").read_text())
    assert saved["iteration_no"] == split_provider.calls == 3
    assert saved["milestone"] != "final"
    assert saved["provenance"]["journal_anchor"]["event_seq"]
    resumed, resumed_provider = construct(tmp_path / "split")
    checkpoint.load(resumed, str(tmp_path / "split/checkpoint.json"))
    resumed.run()
    assert resumed_provider.calls == 3
    assert semantic_state(resumed) == semantic_state(whole)
    rows = journal(tmp_path / "split")
    assert len([row for row in rows if row["event"] == "checkpoint_restore"]) == 1
    assert [row["iteration_no"] for row in rows if row["event"] == "iteration"] == list(range(1, 7))
    assert [row["milestone"] for row in rows if row["event"] == "sh_cut"] == [1, 2]


def test_finite_successful_evaluation_budget_still_stops_without_stop_file(tmp_path):
    engine, provider = construct(tmp_path, settings(budget=2, max_iterations=8))
    summary = engine.run()
    assert summary["reason"] == "budget"
    assert summary["B_evals"] == provider.calls == 2
    assert not (tmp_path / "STOP").exists()


def test_finite_backstop_still_bounds_failed_attempts(tmp_path):
    engine, provider = construct(tmp_path,
        settings(budget=2, max_iterations=20, backstop_k=2), fail=True)
    summary = engine.run()
    assert summary["reason"] == "backstop"
    assert summary["B_evals"] == 0
    assert summary["iteration_no"] == provider.calls == 4
    assert not (tmp_path / "STOP").exists()


@pytest.mark.parametrize("horizon", [None, 0, -1, True, 2.5])
def test_unlimited_requires_explicit_positive_integer_schedule_horizon(horizon):
    cfg = settings(schedule_horizon=horizon)
    if horizon is None:
        cfg.pop("schedule_horizon")
    with pytest.raises(ValueError, match="horizon"):
        validate(cfg)


@pytest.mark.parametrize("interval", [0, -1, True, 1.5])
def test_periodic_checkpoint_interval_rejects_invalid_values(interval):
    with pytest.raises(ValueError, match="checkpoint_interval"):
        validate(settings(checkpoint_interval=interval))


@pytest.mark.parametrize("route", ["repeat_promotion", "repeat_population"])
def test_finite_repeat_profiles_cannot_accidentally_enter_unlimited_mode(route):
    cfg = settings(task="oe/circle_packing", budget=2, max_iterations=4,
        schedule_policy="round_robin", milestone_fracs=[], archive_size=0,
        container={"kind": "pool", "selection": "shared"},
        research={"enabled": True, "max_aux_calls": 0, "reflection_interval": 0})
    cfg[route] = dict(enabled=True, draws_per_source=3, seed_master=157,
        max_comparisons=2, max_draw_attempts=15, max_wall_seconds=1000.,
        timeout_seconds=30., cpu_seconds=30, mem_bytes=2 * 1024**3,
        certification_seconds=5.)
    validate(deepcopy(cfg))  # The same route is otherwise a valid finite profile.
    cfg.update(budget=None, max_iterations=None)
    with pytest.raises(ValueError, match="unlimited|unbounded|finite|budget"):
        validate(cfg)
