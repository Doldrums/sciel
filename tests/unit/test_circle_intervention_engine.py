"""Circle intervention Engine lifecycle with inert sources and fake observations.

No candidate, numerical solver, task evaluator, or external model is executed.
The task files are copied only to exercise real identity and seed preflight.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import shutil
import sys

import pytest

from sciel.config import apply_defaults
from sciel.engine import checkpoint
from sciel.engine.llm.client import LLMResponse
from sciel.engine.loop import Engine
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.eval.registry import load_task
from sciel.hypothesis import circle_interventions as edits


ROOT = Path(__file__).resolve().parents[2]
COST = .031


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def plan(policy="incumbent", **changes):
    data = {"hypothesis": f"circle_restart_reference/1:{policy}",
            "predicted_low": -.2, "predicted_high": .2,
            "reasoning": "Changing the restart reference may change local refinement; the effect is uncertain.",
            "confidence": .5}
    data.update(changes)
    return "HYPOTHESIS_JSON: " + json.dumps(data)


class Model:
    models = ["fixture"]
    weights = [1.0]
    max_retries = 1

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []

    def choose_model(self, rng):
        return "fixture"

    def generate(self, prompt, model, temperature=None):
        self.calls.append((prompt, model, temperature))
        assert self.texts, "unexpected extra model call"
        return LLMResponse(self.texts.pop(0), "fixture", COST, 17, 11)


def config(**changes):
    value = {
        "task": "oe/circle_packing", "budget": 1, "max_iterations": 1,
        "num_islands": 1, "random_seed": 21, "inflight_cap": 1, "dup_cap": 1,
        "op_fixed": "intervention", "circle_intervention": {"enabled": True},
        "policy": {"enabled": False}, "portfolio_seed": False,
        "dynamic_islands": False, "embed_grid": False, "embed_niche": False,
        "idea_island": False, "crossover": False, "milestone_fracs": [],
        "confirm_records": False, "llm": {"mock": True, "max_retries": 1},
        "research": {"enabled": True, "max_aux_calls": 0, "reflection_interval": 0},
    }
    value.update(changes)
    return value


def task_copy(path, *, initial=None):
    path.mkdir(parents=True, exist_ok=True)
    canonical = ROOT / "tasks/oe/circle_packing"
    for filename in ("task.yaml", "evaluator.py"):
        shutil.copyfile(canonical / filename, path / filename)
    (path / "initial_program.py").write_text(
        edits.baseline_source() if initial is None else initial)
    return load_task(str(path))


def rows(engine):
    return [json.loads(line) for line in Path(engine.log.path).read_text().splitlines()]


def build_engine(path, texts, *, cfg=None, task=None, scores=None):
    task = task or task_copy(path.parent / "task")
    model = Model(texts)
    engine = Engine(config() if cfg is None else cfg, task, str(path), model,
                    python=sys.executable)
    evaluated = []
    scores = scores or {"initial": .5, "incumbent": .6, "alternating": .7}

    def evaluate(source):
        policy = edits.policy_of(source)  # Source-byte lookup only; no import.
        if policy != "initial" and engine.intervention is not None:
            events = rows(engine)
            binding = next(e for e in reversed(events)
                           if e.get("phase") == "intervention"
                           and e.get("action") == "source_binding")
            hypothesis = next(e for e in reversed(events) if e.get("phase") == "hypothesis")
            assert binding["experiment_id"] == hypothesis["experiment_id"]
            assert binding["proof"]["child_source_sha256"] == digest(source)
            assert hypothesis["evidence"]["source_sha256"] == digest(source)
            assert events.index(binding) < events.index(hypothesis)
        evaluated.append(source)
        score = scores[policy]
        cases = {"sum_radii": score * 2.635, "target_ratio": score,
                 "validity": 1., "combined_score": score}
        return CascadeOutcome(EvalResult(True, score, cases, timing=.01),
                              score, "full", False)

    engine._evaluate = evaluate
    engine.fixture_evaluated = evaluated
    return engine


def test_engine_builds_exact_child_and_persists_pre_evaluation_proof_and_measurement(tmp_path):
    engine = build_engine(tmp_path / "run", [plan()])
    result = engine.run()
    baseline, child = edits.baseline_source(), edits.source_for("incumbent")
    assert result["B_evals"] == 1 and result["reason"] == "budget"
    assert engine.fixture_evaluated == [baseline, child]
    assert len(engine.llm.calls) == 1 and engine.cost_total == COST
    generated, = [p for p in engine.db._all.values() if p.origin == "gen"]
    assert generated.code == child
    observation, = engine.research.knowledge.experiments
    assert observation["status"] == "evaluated"
    assert observation["parent_source_sha256"] == digest(baseline)
    assert observation["proposed_sha256"] == observation["source_sha256"] == digest(child)
    assert observation["source_repaired"] is False
    assert observation["score"] == .6
    assert observation["actual_delta"] == pytest.approx(.1)
    assert observation["evaluation"]["cases"]["sum_radii"] == .6 * 2.635
    events = rows(engine)
    binding, = [e for e in events if e.get("phase") == "intervention"
                and e.get("action") == "source_binding"]
    assert binding["source_origin"] == "deterministic_builder"
    assert binding["semantic_or_causal_claim"] is False
    assert binding["proof"] == edits.apply(baseline, "incumbent")["proof"]
    iteration, = [e for e in events if e["event"] == "iteration"]
    assert iteration["child_code"] == child
    assert iteration["meta"]["intervention"]["proof"] == binding["proof"]
    saved = json.loads((tmp_path / "run/checkpoint.json").read_text())
    assert saved["circle_intervention"] == engine.intervention.state()
    assert saved["identity"]["circle_intervention"]


@pytest.mark.parametrize("text", [
    "not a plan", plan("unknown"), plan("initial"),
    plan() + "\n```python\nraise RuntimeError('never execute')\n```",
])
def test_invalid_and_noop_plans_are_paid_once_without_candidate_evaluation(tmp_path, text):
    engine = build_engine(tmp_path / "run", [text])
    result = engine.run()
    assert result["B_evals"] == 0 and result["reason"] == "backstop"
    assert engine.fixture_evaluated == [edits.baseline_source()]
    assert len(engine.llm.calls) == 1 and engine.cost_total == COST
    assert not [p for p in engine.db._all.values() if p.origin == "gen"]
    record, = engine.research.knowledge.experiments
    assert record["status"] != "evaluated"
    assert record["score"] is record["actual_delta"] is None
    assert record["evaluation"] is None
    iteration, = [e for e in rows(engine) if e["event"] == "iteration"]
    assert iteration["reason"] == "intervention_plan_invalid"


def test_resume_retains_attempted_plan_after_child_eviction_and_matches_continuous_run(tmp_path):
    # A one-cell/one-program population evicts the lower-score child. Therefore
    # rejecting it later proves persisted plan memory, not active-source dedup.
    cfg = config(budget=2, max_iterations=3, pop_cap=1, grid_bins=1)
    scores = {"initial": .9, "incumbent": .8, "alternating": .7}
    continuous = build_engine(tmp_path / "continuous", [plan(), plan(), plan("alternating")],
                              cfg=cfg, scores=scores)
    expected = continuous.run()
    paused = build_engine(tmp_path / "split", [plan()], cfg=cfg, scores=scores)
    paused._stop_at = 1
    assert paused.run()["reason"] == "paused"
    assert [p.code for p in paused.db.islands[0].programs.values()] == [edits.baseline_source()]
    before = copy.deepcopy(paused.intervention.state())
    resumed = build_engine(tmp_path / "split", [plan(), plan("alternating")], cfg=cfg, scores=scores)
    checkpoint.load(resumed, str(tmp_path / "split/checkpoint.json"))
    assert resumed.intervention.state() == before
    assert resumed.run() == expected
    assert resumed.fixture_evaluated == [edits.source_for("alternating")]
    assert resumed.llm.calls == continuous.llm.calls[1:]
    assert resumed.cost_total == pytest.approx(3 * COST)
    for name in ("intervention", "research", "db", "rng", "selection", "schedule"):
        assert getattr(resumed, name).state() == getattr(continuous, name).state()
    rejected = [e for e in rows(resumed) if e["event"] == "iteration"
                and e.get("reason") == "intervention_plan_repeated"]
    assert len(rejected) == 1
    assert len(resumed.research.knowledge.experiments) == 3
    assert [r["status"] == "evaluated" for r in resumed.research.knowledge.experiments] == [True, False, True]


def test_exhausted_parent_does_not_buy_another_plan_or_evaluate_another_source(tmp_path):
    engine = build_engine(tmp_path / "run", [plan(), plan("alternating")],
        cfg=config(budget=3, max_iterations=3, pop_cap=1, grid_bins=1),
        scores={"initial": .9, "incumbent": .8, "alternating": .7})
    result = engine.run()
    assert result["reason"] == "backstop" and result["B_evals"] == 2
    assert len(engine.llm.calls) == 2 and engine.cost_total == 2 * COST
    assert engine.fixture_evaluated == [edits.source_for(p)
                                        for p in ("initial", "incumbent", "alternating")]
    assert len(engine.research.knowledge.experiments) == 2
    iteration = [e for e in rows(engine) if e["event"] == "iteration"][-1]
    assert iteration["reason"] == "intervention_exhausted" and iteration["cost"] == 0


def test_improved_child_cannot_revisit_evicted_seed_source_after_resume(tmp_path):
    # The source-binding history must retain BOTH ends of earlier edits. Once
    # incumbent wins the single grid cell, initial is absent from active hashes
    # and incumbent->initial is a new directed key, but its bytes are old work.
    cfg = config(budget=2, max_iterations=3, pop_cap=1, grid_bins=1)
    texts = [plan("incumbent"), plan("initial"), plan("alternating")]
    continuous = build_engine(tmp_path / "continuous", texts, cfg=cfg)
    expected = continuous.run()
    paused = build_engine(tmp_path / "split", texts[:1], cfg=cfg)
    paused._stop_at = 1
    assert paused.run()["reason"] == "paused"
    assert [p.code for p in paused.db.islands[0].programs.values()] == [edits.source_for("incumbent")]
    old_state = copy.deepcopy(paused.intervention.state())
    resumed = build_engine(tmp_path / "split", texts[1:], cfg=cfg)
    checkpoint.load(resumed, str(tmp_path / "split/checkpoint.json"))
    assert resumed.intervention.state() == old_state
    assert resumed.run() == expected
    assert resumed.fixture_evaluated == [edits.source_for("alternating")]
    assert continuous.fixture_evaluated == [edits.source_for(p)
                                            for p in ("initial", "incumbent", "alternating")]
    assert resumed.llm.calls == continuous.llm.calls[1:]
    assert resumed.cost_total == pytest.approx(3 * COST)
    events = rows(resumed)
    requests = [e for e in events if e.get("phase") == "intervention"
                and e.get("action") == "request_start"]
    assert [r["allowed_targets"] for r in requests] == [
        ["incumbent", "alternating"], ["alternating"], ["alternating"]]
    rejected, = [e for e in events if e["event"] == "iteration"
                 and e.get("reason") == "intervention_plan_repeated"]
    assert rejected["cost"] == COST
    observations = resumed.research.knowledge.experiments
    assert len(observations) == 3
    assert observations[1]["score"] is observations[1]["actual_delta"] is None
    assert observations[1]["evaluation"] is None
    assert observations[2]["parent_source_sha256"] == digest(edits.source_for("incumbent"))
    for name in ("intervention", "research", "db", "rng", "selection", "schedule"):
        assert getattr(resumed, name).state() == getattr(continuous, name).state()


@pytest.mark.parametrize("changes", [
    {"research": {"enabled": False}}, {"op_fixed": "full_rewrite"},
    {"inflight_cap": 2}, {"portfolio_seed": True}, {"dynamic_islands": True},
    {"idea_island": True}, {"confirm_records": True},
    {"idea_two_stage": {"enabled": True}},
    {"circle_intervention": {"enabled": True, "extra": True}},
    {"circle_intervention": {"enabled": 1}},
])
def test_incompatible_configuration_fails_before_any_model_call(tmp_path, changes):
    model = Model([])
    with pytest.raises(ValueError):
        Engine(config(**changes), task_copy(tmp_path / "task"), str(tmp_path / "run"), model)
    assert model.calls == []


@pytest.mark.parametrize("kind", ["foreign_task", "historical_seed", "modified_family"])
def test_wrong_task_or_warm_seed_is_refused_before_any_model_call(tmp_path, kind):
    model = Model([])
    if kind == "foreign_task":
        task = load_task(str(ROOT / "tasks/test/toy_quadratic"))
    else:
        source = ((ROOT / "tasks/oe/circle_packing/initial_program.py").read_text()
                  if kind == "historical_seed" else edits.baseline_source() + "\n")
        task = task_copy(tmp_path / "task", initial=source)
    with pytest.raises(ValueError):
        Engine(config(), task, str(tmp_path / "run"), model)
    assert model.calls == []


@pytest.mark.parametrize("force", [False, True])
def test_intervention_checkpoint_cannot_resume_with_component_disabled(tmp_path, force):
    original = build_engine(tmp_path / "run", [plan()])
    original.run()
    target = build_engine(tmp_path / "run", [],
                          cfg=config(op_fixed="full_rewrite", circle_intervention={"enabled": False}))
    try:
        with pytest.raises(checkpoint.IdentityMismatch, match="intervention"):
            checkpoint.load(target, str(tmp_path / "run/checkpoint.json"), allow_identity_mismatch=force)
        assert target.llm.calls == [] and target.fixture_evaluated == []
    finally:
        target.log.close()


@pytest.mark.parametrize("change", ["missing_state", "identity", "attempt_memory", "valid_key_substitution"])
def test_checkpoint_tampering_blocks_resume_before_dispatch(tmp_path, change):
    original = build_engine(tmp_path / "run", [plan()])
    original.run()
    saved_path = tmp_path / "run/checkpoint.json"
    state = json.loads(saved_path.read_text())
    if change == "missing_state":
        del state["circle_intervention"]
    elif change == "identity":
        state["identity"]["circle_intervention"]["fixture_tamper"] = True
    elif change == "attempt_memory":
        assert state["circle_intervention"]["seen_plans"]
        state["circle_intervention"]["seen_plans"] = []
    else:
        # Counts, syntax and membership in the six possible directed edits all
        # stay valid. Only the anchored source-binding history can reject this.
        replacement = edits.apply(edits.baseline_source(), "alternating")["proof"]["plan_key"]
        assert state["circle_intervention"]["seen_plans"] != [replacement]
        state["circle_intervention"]["seen_plans"] = [replacement]
    saved_path.write_text(json.dumps(state))
    target = build_engine(tmp_path / "run", [])
    try:
        with pytest.raises(checkpoint.IdentityMismatch, match="intervention"):
            checkpoint.load(target, str(saved_path))
        assert target.llm.calls == [] and target.fixture_evaluated == []
    finally:
        target.log.close()


def test_uncheckpointed_intervention_request_cannot_be_replayed(tmp_path):
    original = build_engine(tmp_path / "run", [plan()])
    original.run()
    with Path(original.log.path).open("a") as stream:
        stream.write(json.dumps({"event": "research", "phase": "intervention",
                                 "action": "request_start"}) + "\n")
    target = build_engine(tmp_path / "run", [])
    try:
        with pytest.raises(checkpoint.IdentityMismatch, match="uncheckpointed research activity"):
            checkpoint.load(target, str(tmp_path / "run/checkpoint.json"))
        assert target.llm.calls == [] and target.fixture_evaluated == []
    finally:
        target.log.close()


def test_default_off_and_explicit_off_preserve_original_worker_path(tmp_path):
    assert "circle_intervention" not in apply_defaults({})
    text = plan() + "\n```python\n" + edits.source_for("incumbent") + "```\n"
    absent = config(op_fixed="full_rewrite")
    del absent["circle_intervention"]
    disabled = {**absent, "circle_intervention": {"enabled": False}}
    left = build_engine(tmp_path / "left", [text], cfg=absent)
    right = build_engine(tmp_path / "right", [text], cfg=disabled)
    assert left.intervention is right.intervention is None
    assert left.run() == right.run()
    assert left.llm.calls == right.llm.calls
    assert left.fixture_evaluated == right.fixture_evaluated
    assert left.db.state() == right.db.state()
    assert left.research.state() == right.research.state()
    for engine in (left, right):
        assert not any(e.get("phase") == "intervention" for e in rows(engine))
        saved = json.loads((Path(engine.run_dir) / "checkpoint.json").read_text())
        assert "circle_intervention" not in saved
        assert "circle_intervention" not in saved["identity"]
