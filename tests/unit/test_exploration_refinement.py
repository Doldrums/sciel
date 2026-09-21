"""Resolved exploration must survive global-stall refinement.

Only inert source strings, a local provider and an evaluator fixture are used.
The regressions exercise actual engine plans, frozen jobs and delivered prompts.
"""
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from sciel.components.policy.tilt import Tilt, elite_target
from sciel.engine.llm.client import LLMResponse
from sciel.engine.loop import Engine
from sciel.engine.parallel import _intent, _save_intents
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.eval.registry import load_task


ROOT = Path(__file__).resolve().parents[2]
PARENT = "def run_search():\n    return (0, 0)\n"
CHILD = "def run_search():\n    return (2, 0)\n"


class Provider:
    models = ["fixture"]
    weights = [1.0]

    def __init__(self):
        self.prompts = []

    def choose_model(self, rng):
        return "fixture"

    def generate(self, prompt, model, temperature=None):
        self.prompts.append(prompt)
        return LLMResponse("```python\n" + CHILD + "```", model, .01, 10, 10)


@pytest.fixture
def make_engine(tmp_path, monkeypatch):
    opened = []

    def make(mode, source="controller", *, gradient=True, **overrides):
        cfg = dict(task="test/toy_quadratic", num_islands=1, budget=100,
            pop_cap=30, embed_grid=False, portfolio_seed=False,
            idea_island=False, dynamic_islands=False, milestone_fracs=[],
            policy={"enabled": source in {"controller", "bandit"}, "base_mix": {mode: 1.},
                    "mode_bandit": source == "bandit", "mode_floor": .001},
            preset_selection=source == "preset", llm={"mock": True},
            research={"enabled": False}, candidate_novelty={"enabled": False},
            refine_stall=8, container={"kind": "pool", "selection": "shared"})
        cfg.update(overrides)
        provider = Provider()
        engine = Engine(cfg, load_task(str(ROOT / "tasks/test/toy_quadratic")),
            str(tmp_path / str(len(opened))), provider, python=sys.executable)
        opened.append(engine)
        engine.seed_score = .9
        engine.db.add(PARENT, .9, {}, 0, "seed")
        engine.db.add(PARENT + "# second parent\n", .85 if gradient else .9, {}, 0, "seed")
        engine.B_evals = 24
        engine._last_pop_improve = 5
        if source == "preset":
            monkeypatch.setattr(engine.preset_weights, "sample", lambda rng: (mode, 1.))
            monkeypatch.setattr(engine.preset_weights, "probabilities", lambda: {mode: 1.})
        if source == "bandit":
            monkeypatch.setattr(engine.controller.mode_bandit, "dist", lambda phase: {mode: 1.})
        intensify = []
        original = engine.op_bandit.pick

        def capture(*args, **kwargs):
            intensify.append(kwargs["intensify"])
            return original(*args, **kwargs)

        monkeypatch.setattr(engine.op_bandit, "pick", capture)
        return engine, provider, intensify

    yield make
    for engine in opened:
        engine.log.close()


def evaluate_fixture(code):
    assert code == CHILD
    return CascadeOutcome(EvalResult(True, .95, cases={"value": .95}, timing=.01),
                          .95, "full", False)


@pytest.mark.parametrize("source", ["controller", "bandit", "preset"])
def test_explore_remains_structural_in_frozen_and_delivered_prompt(make_engine, tmp_path, source):
    engine, provider, intensify = make_engine("explore", source)
    plan = engine._prepare_step(0, spawn=False, iteration_no=28)
    assert plan["mode"] == plan["eff_mode"] == "explore"
    assert intensify == [False]
    directory = tmp_path / "frozen"
    directory.mkdir()
    _save_intents(directory, [_intent(engine, "G:29", plan)])
    frozen = json.loads((directory / "jobs.json").read_text())["jobs"][0]
    directive = frozen["snapshot"]["directive"]
    assert "STRUCTURAL change" in directive
    assert "population best has not improved" in directive
    assert "does not establish stagnation in this particular lineage" in directive
    assert "REFINEMENT:" not in directive and "KEEP this construction" not in directive
    result = engine._execute_step(plan, evaluate_fn=evaluate_fixture)
    assert len(provider.prompts) == 1 and directive in provider.prompts[0]
    assert result.cost == .01
    events = [json.loads(line) for line in Path(engine.log.path).read_text().splitlines()]
    decisions = [r for r in events if r.get("policy") == "refine"]
    assert len(decisions) == 1
    assert decisions[0]["action"] == "skip"
    assert decisions[0]["reason"] == "exploration_mode"
    assert decisions[0]["effective_mode"] == "explore"
    assert decisions[0]["iteration_no"] == 28


@pytest.mark.parametrize("source", ["controller", "bandit", "preset"])
@pytest.mark.parametrize("mode", ["base", "exploit"])
def test_base_and_exploit_keep_competitive_refinement(make_engine, source, mode):
    engine, provider, intensify = make_engine(mode, source)
    plan = engine._prepare_step(0, spawn=False)
    assert plan["eff_mode"] == mode
    assert intensify == [True]
    assert "KEEP this construction" in plan["snap"].directive
    assert "STAGNATION:" not in plan["snap"].directive
    engine._execute_step(plan, evaluate_fn=evaluate_fixture)
    assert plan["snap"].directive in provider.prompts[0]


def test_global_counter_does_not_claim_a_lineage_has_stalled(make_engine):
    engine, _, intensify = make_engine("base", refine_elite_frac=2.)
    assert engine.iteration_no == 0
    assert not engine.controller.in_plateau(0, 0)
    plan = engine._prepare_step(0, spawn=False)
    assert intensify == [False]
    assert "population-wide signal" in plan["snap"].directive
    assert "this line of programs has stopped improving" not in plan["snap"].directive


def test_forced_operator_still_has_no_refinement(make_engine):
    engine, _, intensify = make_engine("exploit", op_fixed="full_rewrite")
    plan = engine._prepare_step(0, spawn=False)
    assert intensify == []
    assert "REFINEMENT:" not in plan["snap"].directive


def test_selection_override_uses_resolved_mode_for_refinement(make_engine):
    engine, _, intensify = make_engine("explore")
    engine.three_head = SimpleNamespace(build=lambda view, db: Tilt(1., lambda v: elite_target(v, 1)))
    plan = engine._prepare_step(0, spawn=False)
    assert plan["mode"] == plan["eff_mode"] == "exploit"
    assert intensify == [True]
    assert "KEEP this construction" in plan["snap"].directive


def test_warmup_gated_exploit_keeps_base_mode_attribution(make_engine):
    engine, _, intensify = make_engine("exploit", gradient=False)
    plan = engine._prepare_step(0, spawn=False)
    assert plan["mode"] == "exploit" and plan["eff_mode"] == "base"
    assert intensify == [True]
    assert "KEEP this construction" in plan["snap"].directive


def test_no_population_stall_does_not_invent_one_for_explore(make_engine):
    engine, _, intensify = make_engine("explore")
    engine._last_pop_improve = engine.B_evals
    plan = engine._prepare_step(0, spawn=False)
    assert intensify == [False]
    assert "STAGNATION:" not in plan["snap"].directive
    assert "REFINEMENT:" not in plan["snap"].directive


def test_island_detector_reports_its_own_scope(make_engine):
    engine, _, intensify = make_engine("explore")
    engine._last_pop_improve = engine.B_evals
    engine.controller.plateau_until[0] = 100
    engine.controller.plateau_mix = {"explore": 1.}
    plan = engine._prepare_step(0, spawn=False)
    assert intensify == [False]
    assert "island-level signal" in plan["snap"].directive
    assert "population best has not improved" not in plan["snap"].directive
