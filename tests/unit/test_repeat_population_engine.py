"""Real Engine orchestration with inert programs and synthetic valid receipts.

Neither a candidate, evaluator, subprocess, model API nor optimizer is executed.
The geometric receipts describe a fixed separated grid and are checked by the
real pure validators; they do not claim a packing-search quality improvement.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import pytest
from sciel.config import apply_defaults
from sciel.engine import checkpoint
from sciel.engine.llm.client import LLMResponse
from sciel.engine.loop import Engine
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.eval.registry import load_task
from repeat_population_fixtures import (
    ROOT, sha, step_of, code_for,
    settings, Provider, Draws, task,
    numeric, draws, construct, projection,
    normalized_calls, interrupted_bootstrap,
)


@pytest.mark.parametrize('control', ['request', 'marker'])
def test_repeat_population_honors_graceful_stop_with_null_attempt_cap(
        task, tmp_path, numeric, draws, control):
    engine, provider = construct(task, tmp_path, cfg=settings(max_iterations=None))
    def stop_after_dispatch(prompt):
        if control == 'request':
            engine.request_stop()
        else:
            (tmp_path / 'STOP').write_text('pause\n')
    provider.before_call = stop_after_dispatch
    result = engine.run()
    assert result['reason'] == 'paused'
    assert result['iteration_no'] == result['B_evals'] == len(provider.calls) == 1
    assert not (tmp_path / 'STOP').exists()
    assert not engine.paired_batches.pending
    assert len(draws.calls) == 9  # Bootstrap + complete paired comparison.
    saved = json.loads((tmp_path / 'checkpoint.json').read_text())
    assert saved['iteration_no'] == 1 and saved['research']['pending'] == {}


@pytest.mark.parametrize("change", [
    {"num_islands": True}, {"pop_cap": 1}, {"inflight_cap": 2}, {"dup_cap": 2},
    {"confirm_records": True}, {"portfolio_seed": True}, {"dynamic_islands": True},
    {"container": {"kind": "grid"}}, {"schedule_policy": "sh"},
    {"milestone_fracs": [.5]}, {"archive_size": 1}, {"op_fixed": "error_repair"},
    {"research": {"enabled": False}}, {"policy": {"enabled": True}},
    {"crossover": True}, {"embed_grid": True}, {"three_head": True},
])
def test_unsupported_profiles_fail_before_any_runtime(change):
    with pytest.raises(ValueError, match="repeat_population"):
        apply_defaults(settings(**change))


def test_two_islands_use_repeat_admission_without_rewriting_forecast_or_single_score(
        task, tmp_path, monkeypatch, numeric, draws):
    import sciel.engine.repeat_population_support as route

    engine, provider = construct(task, tmp_path / "two-islands")
    legacy = {name: copy.deepcopy(getattr(engine, name).state())
              for name in ("reward", "selection", "op_bandit")}
    selected = []
    original_worker = route.run_worker

    def inspect_worker(op, snapshot, *args, **kwargs):
        active = set(engine.repeat_population.active_ids(snapshot.island))
        assert snapshot.parent_id in active
        expected = {engine.db.get(i).code for i in active if i != snapshot.parent_id}
        assert {item["code"] for item in snapshot.inspirations} == expected
        assert code_for(1) not in expected
        return original_worker(op, snapshot, *args, **kwargs)

    monkeypatch.setattr(route, "run_worker", inspect_worker)

    def capture_selection(prompt):
        snap = projection(engine)["selection"]
        parent = engine.db.get(snap["parent_id"])
        heads = projection(engine)["evidence_heads"]
        actual = json.JSONDecoder().raw_decode(
            prompt.split("REPEAT_SELECTION_EVIDENCE_JSON:\n", 1)[1])[0]
        assert actual == {"parent": heads[snap["parent_source_sha256"]],
                          "comparator": heads[snap["comparator_source_sha256"]]}
        selected.append({"snapshot": copy.deepcopy(snap), "score": parent.score,
                         "cases": copy.deepcopy(parent.cases), "code": parent.code})

    provider.before_call = capture_selection
    result = engine.run()
    state = projection(engine)
    assert result["B_evals"] == engine.iteration_no == 5
    assert len(provider.calls) == 5 and len(numeric) == 6
    assert len(draws.calls) == 33  # One shared bootstrap, five fresh six-draw comparisons.
    assert [s["snapshot"]["island"] for s in selected] == [0, 1, 0, 1, 0]
    assert [s["status"] for s in state["transitions"]] == [
        "bootstrap_confirmed", "rejected", "admitted", "admitted", "replaced", "replaced"]
    assert [[step_of(engine.db.get(i).code) for i in engine.repeat_population.active_ids(k)]
            for k in (0, 1)] == [[5, 3], [4, 2]]
    # Original best is retained as history, never exposed as an active parent.
    assert step_of(engine.db.get(engine.db.pop_best_id).code) == 1
    assert all(step_of(s["code"]) != 1 for s in selected)
    assert all(p.score == numeric[step_of(p.code)]["score"] for p in engine.db._all.values())
    assert [r["value"] for r in state["rewards"]] == [0, 1, 1, 1, 1]
    assert legacy == {name: getattr(engine, name).state() for name in legacy}
    assert engine.cost_total == pytest.approx(.005)
    assert engine.research.attempts == engine.research.outcomes == 5
    assert engine.research.aux_calls == 0
    assert engine.repeat_population.publication_source_sha256 == sha(numeric[0]["source"])
    assert "repeat_population" in result
    assert "repeat_promotion" not in result
    exported = result["repeat_population"]
    assert exported["observed_search_best"]["source_sha256"] == sha(code_for(1))
    assert exported["confirmed_output_incumbent"]["source_sha256"] == sha(numeric[0]["source"])
    assert Path(exported["confirmed_output_incumbent"]["path"]).read_text() == numeric[0]["source"]

    ordinary = engine.research.knowledge.experiments
    groups = engine.research.knowledge.comparisons
    assert len(ordinary) == len(groups) == 5
    assert any(s["snapshot"]["parent_id"] != s["snapshot"]["comparator_program_id"] for s in selected)
    for record, group, pick, call in zip(ordinary, groups, selected, provider.calls):
        assert record["parent_id"] == pick["snapshot"]["parent_id"]
        assert record["parent_score"] == pick["score"]
        assert record["parent_cases"] == pick["cases"]
        assert record["actual_delta"] == record["score"] - pick["score"]
        assert group["origin_experiment_id"] == record["id"]
        assert group["branch"] == record["branch"]
        assert group["comparator_source_sha256"] == pick["snapshot"]["comparator_source_sha256"]
        assert group["challenger_source_sha256"] == record["source_sha256"]
        assert group["forecast"] == "not_scored"
        # Check the actual sent context, not only an expected Python snapshot.
        from sciel.hypothesis.rendering import decode_evidence
        wire = json.JSONDecoder().raw_decode(call["prompt"].split("RESEARCH_CONTEXT_JSON:\n", 1)[1])[0]
        context = decode_evidence(wire)
        assert context["parent_score"] == pick["score"]
        assert context["parent_cases"] == pick["cases"]


def test_each_draw_follows_durable_checkpoint_and_history_only_candidate(
        task, tmp_path, monkeypatch, numeric, draws):
    engine, _ = construct(task, tmp_path / "durable", cfg=settings(budget=2, max_iterations=2))
    fsynced = set()
    original = os.fsync

    def fsync(fd):
        original(fd)
        st = os.fstat(fd)
        fsynced.add((st.st_dev, st.st_ino))

    monkeypatch.setattr(os, "fsync", fsync)
    observations = []

    def inspect(call):
        path = Path(engine.run_dir) / "checkpoint.json"
        saved = json.loads(path.read_text())
        for item in (path, path.parent, path.parent / "sciel_events.jsonl"):
            st = item.stat()
            assert (st.st_dev, st.st_ino) in fsynced
        assert not saved["research"]["pending"] and not saved["research"]["queue"]
        batch = saved["paired_batches"]["data"]["batches"][-1]
        assert batch["decision"] is None
        assert call["source_sha256"] in {d["source_sha256"] for d in batch["plan"]["draws"]}
        population = saved["repeat_population"]["projection"]
        pending = population["pending"]
        if pending is not None:
            assert pending["program_id"] in population["history"]
            assert all(pending["program_id"] not in ids for ids in population["active"].values())
            assert saved["B_evals"] == saved["research"]["outcomes"]
        observations.append(saved["B_evals"])

    draws.before_draw = inspect
    engine.run()
    assert observations == [0] * 3 + [1] * 6 + [2] * 6


def test_shared_comparator_refresh_after_rejection_reaches_other_island(
        task, tmp_path, numeric, draws):
    # Seed source appears once per island but is measured once per physical batch.
    draws.values = lambda step, batch: (.4 if batch == 0 else .5) if step == 0 else [.4, .3, .6][step]
    engine, provider = construct(task, tmp_path / "shared", cfg=settings(budget=2, max_iterations=2))
    views = []
    provider.before_call = lambda prompt: views.append(copy.deepcopy(projection(engine)))
    engine.run()
    source = sha(numeric[0]["source"])
    before, after = views
    first = before["evidence_heads"][source]
    second = after["evidence_heads"][source]
    assert first["id"] != second["id"]
    assert first["metrics"]["raw"]["minimum"] == pytest.approx(.4)
    assert second["metrics"]["raw"]["minimum"] == pytest.approx(.5)
    assert after["selection"]["island"] == 1
    assert after["selection"]["distribution"][0]["evidence_id"] == second["id"]
    assert len(after["evidence_heads"]) == 2  # Also retain measured rejected child.


@pytest.mark.parametrize("failure_at,status,expected_draws,attempts,rewards", [
    (4, "candidate_failure", 15, 2, [0, 1]),
    (5, "candidate_failure", 9, 1, []),
    (4, "infrastructure", 4, 1, []),
])
def test_challenger_failure_continues_but_unknown_comparator_or_infrastructure_stops(
        task, tmp_path, numeric, draws, failure_at, status, expected_draws, attempts, rewards):
    draws.failure_at, draws.failure_status = failure_at, status
    engine, provider = construct(task, tmp_path / f"failure-{failure_at}-{status}",
                                 cfg=settings(budget=2, max_iterations=2))
    engine.run()
    assert len(draws.calls) == expected_draws and len(provider.calls) == attempts
    assert engine.B_evals == engine.research.outcomes == attempts
    assert [r["value"] for r in projection(engine)["rewards"]] == rewards
    first = engine.research.knowledge.comparisons[0]
    failed = [v for pair in first["pairs"] for v in (pair["comparator"], pair["challenger"])
              if v is not None and v["status"] != "valid"]
    assert failed and all(r["raw_sum"] is None and r["certified_sum"] is None for r in failed)
    assert not first["derived"]["all_valid"]
    for role in ("comparator", "challenger"):
        if any(pair[role] is None or pair[role]["status"] != "valid" for pair in first["pairs"]):
            assert first["derived"]["raw_means"][role] is None
            assert first["derived"]["certified_means"][role] is None
    if not rewards:
        assert engine.repeat_population.stopped_reason is not None
        assert [len(engine.repeat_population.active_ids(i)) for i in (0, 1)] == [1, 1]


def test_whole_comparison_budget_is_checked_before_next_paid_generation(
        task, tmp_path, numeric, draws):
    cfg = settings()
    cfg["repeat_population"].update(max_comparisons=1, max_draw_attempts=9, max_wall_seconds=315.)
    engine, provider = construct(task, tmp_path / "budget", cfg=cfg)
    result = engine.run()
    assert result["reason"] == "repeat_budget"
    assert len(provider.calls) == 1 and len(draws.calls) == 9
    assert engine.B_evals == 1 and len(numeric) == 2
    assert engine.repeat_population.pending is None


def test_absent_and_disabled_profiles_preserve_default_engine_and_never_probe_adapter(
        task, tmp_path, monkeypatch, numeric):
    def forbidden(**kwargs):
        raise AssertionError("disabled population route probed a numeric runtime")

    monkeypatch.setattr("sciel.eval.repeat_draw.adapter_identity", forbidden)
    monkeypatch.setattr("sciel.eval.repeat_draw.draw", forbidden)
    cfg = settings(budget=1, max_iterations=1)
    cfg.pop("repeat_population")
    plain, plain_calls = construct(task, tmp_path / "plain", cfg=cfg)
    plain.run()
    disabled, disabled_calls = construct(task, tmp_path / "disabled",
        cfg={**cfg, "repeat_population": {"enabled": False}})
    disabled.run()
    assert plain.repeat_population is disabled.repeat_population is None
    assert plain.paired_batches is disabled.paired_batches is None
    assert plain_calls.calls == disabled_calls.calls
    for key in ("db", "rng", "schedule", "reward", "op_bandit", "selection"):
        assert getattr(plain, key).state() == getattr(disabled, key).state()
    for engine in (plain, disabled):
        saved = json.loads((Path(engine.run_dir) / "checkpoint.json").read_text())
        assert "repeat_population" not in saved and "paired_batches" not in saved
        assert "repeat_population" not in saved["identity"]
        assert not (Path(engine.run_dir) / "paired_batches").exists()


def test_post_observation_research_stop_retains_unexamined_source_without_admission(
        task, tmp_path, monkeypatch, numeric, draws):
    from sciel.campaign.budget import CampaignStopped

    path = tmp_path / "research-stop"
    engine, provider = construct(task, path, cfg=settings(budget=2, max_iterations=2))
    complete = engine.research.complete_step

    def stop_after_completion(**kwargs):
        complete(**kwargs)
        engine.research.stopped_reason = "fixture_auxiliary_stop"
        raise CampaignStopped("fixture_auxiliary_stop")

    monkeypatch.setattr(engine.research, "complete_step", stop_after_completion)
    with pytest.raises(CampaignStopped):
        engine.run()
    state = projection(engine)
    assert len(provider.calls) == 1 and len(draws.calls) == 3
    assert engine.B_evals == engine.research.outcomes == 1
    assert engine.cost_total == pytest.approx(.001)
    assert state["pending"]["source_sha256"] == sha(code_for(1))
    assert state["pending"]["program_id"] == "p3"
    assert state["stopped_reason"] == "research_stopped"
    assert state["active"] == {"0": ["p1"], "1": ["p2"]}
    assert state["rewards"] == [] and not engine.research.knowledge.comparisons
    saved = json.loads((path / "checkpoint.json").read_text())
    assert saved["repeat_population"]["projection"]["pending"] == state["pending"]
    exported = json.loads((path / "repeat_population.json").read_text())
    assert exported["status"] == "research_stopped" and exported["coverage_complete"] is False
    assert exported["repeat_summary"]["coverage_scope"] == "reserved_batches_only"
    assert exported["repeat_summary"]["coverage_complete"] is True
    later, later_provider = construct(task, path, cfg=settings(budget=2, max_iterations=2), start=1)
    checkpoint.load(later, str(path / "checkpoint.json"))
    with pytest.raises(CampaignStopped):
        later.run()
    assert len(draws.calls) == 3 and not later_provider.calls


@pytest.mark.parametrize("stale_output", [False, True])
def test_failed_bootstrap_has_no_generation_coverage_or_unbound_output(
        task, tmp_path, numeric, draws, stale_output):
    draws.failure_at = 2
    engine, provider = construct(task, tmp_path / "bootstrap", cfg=settings(budget=2, max_iterations=2))
    target = Path(engine.run_dir) / "confirmed_program.py"
    if stale_output:
        target.write_text("unbound old output")
        with pytest.raises(ValueError, match="unbound confirmed"):
            engine.run()
        assert target.read_text() == "unbound old output"
    else:
        result = engine.run()
        exported = result["repeat_population"]
        assert exported["coverage_complete"] is False
        assert exported["confirmed_output_incumbent"] is None
        assert exported["status"] == "bootstrap_incomplete"
        assert not target.exists()
    assert not provider.calls and len(draws.calls) == 3 and len(numeric) == 1
    assert engine.B_evals == engine.research.attempts == 0
    assert [engine.repeat_population.active_ids(i) for i in (0, 1)] == [[], []]
    assert projection(engine)["rewards"] == []


def test_null_v1_repeat_configuration_is_disabled_not_a_population_conflict():
    cfg = apply_defaults(settings(repeat_promotion=None))
    assert cfg["repeat_population"]["enabled"] is True
    assert cfg["repeat_promotion"] == {"enabled": False}
