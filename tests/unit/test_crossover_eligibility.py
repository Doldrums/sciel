"""Crossover requires a second exact source, not just another population ID.

All candidates are inert strings. Providers and evaluations are local fixtures;
no generated code, optimizer, subprocess or remote model is executed.
"""
import copy
import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pytest

from sciel.engine.llm.client import LLMResponse
from sciel.engine.loop import Engine
from sciel.engine.operators import OpBandit, usable_inspirations
from sciel.engine.parallel import _intent, _save_intents
from sciel.engine.prompt.sources import source_bundle, source_digest
from sciel.engine.types import NoChild, Snapshot
from sciel.engine.worker import run_worker
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.eval.registry import load_task


ROOT = Path(__file__).resolve().parents[2]
PARENT = "def run_search():\n    return (0, 0)\n"
DONOR = "def run_search():\n    return (0.000001, 0)\n"
CHILD = "def run_search():\n    return (2, 0)\n"


def row(ident, code, score=.3):
    return {"program_id": ident, "code": code, "score": score,
            "card": "Inert source fixture", "source_sha256": source_digest(code)}


class Provider:
    models = ["fixture"]
    weights = [1.0]

    def __init__(self, texts=None):
        self.prompts = []
        self.texts = iter(texts or ["```python\n" + CHILD + "```"])

    def choose_model(self, rng):
        return "fixture"

    def generate(self, prompt, model, temperature=None):
        self.prompts.append(prompt)
        return LLMResponse(next(self.texts), model, .01, 10, 10)


def outcome(code):
    assert code == CHILD
    return CascadeOutcome(EvalResult(True, .8, cases={"value": .8}, timing=.01),
                          .8, "full", False)


@pytest.fixture
def make_engine(tmp_path):
    opened = []

    def make(**overrides):
        cfg = dict(task="test/toy_quadratic", num_islands=1, budget=5,
            pop_cap=30, embed_grid=False, portfolio_seed=False,
            idea_island=False, dynamic_islands=False, crossover=True,
            policy={"enabled": False}, milestone_fracs=[], llm={"mock": True},
            research={"enabled": False}, candidate_novelty={"enabled": False},
            refine_stall=10000, container={"kind": "pool", "selection": "shared"})
        cfg.update(overrides)
        engine = Engine(cfg, load_task(str(ROOT / "tasks/test/toy_quadratic")),
                        str(tmp_path / str(len(opened))), Provider(), python=sys.executable)
        engine.seed_score = .2
        opened.append(engine)
        return engine

    yield make
    for engine in opened:
        engine.log.close()


def test_donors_exclude_same_id_exact_aliases_and_duplicate_sources_only():
    rows = [row("p1", DONOR), row("alias", PARENT), row("p2", DONOR),
            row("p3", DONOR), row("p4", PARENT + "# comment\n")]
    before = copy.deepcopy(rows)
    donors = usable_inspirations("p1", PARENT, rows)
    assert [r["program_id"] for r in donors] == ["p2", "p4"]
    assert donors[0]["code"] == DONOR  # Even a tiny constant change remains eligible.
    assert rows == before
    donors[0]["card"] = "mutated detached view"
    assert rows == before


def test_filtering_happens_before_donor_limit():
    rows = [row(f"alias-{i}", PARENT) for i in range(8)]
    rows += [row("later", DONOR), row("last", CHILD)]
    donors = usable_inspirations("p1", PARENT, rows, limit=1)
    assert [r["program_id"] for r in donors] == ["later"]


def test_donor_hash_cannot_claim_a_different_source():
    donor = row("p2", DONOR)
    donor["source_sha256"] = source_digest(CHILD)
    with pytest.raises(ValueError, match="source/provenance hash"):
        usable_inspirations("p1", PARENT, [donor])


@pytest.mark.parametrize("fresh_failure,expected", [
    (False, {"diff", "full_rewrite"}),
    (True, {"diff", "full_rewrite", "error_repair"}),
])
def test_cross_mask_precedes_probability_normalization(fresh_failure, expected):
    bandit = OpBandit(arms=["diff", "full_rewrite", "error_repair", "cross"])
    bandit.events["cross"].append(1000.)
    class Capture:
        def choices(self, arms, *, weights, k):
            assert set(arms) == expected
            assert weights == pytest.approx([1 / len(expected)] * len(expected))
            return [arms[0]]
    op, propensity, statistics = bandit.pick(
        fresh_failure, Capture(), cross_available=False)
    assert op in expected
    assert set(statistics) == expected
    assert propensity == pytest.approx(1 / len(expected))


def test_available_cross_retains_real_bandit_credit():
    bandit = OpBandit(arms=["diff", "full_rewrite", "cross"])
    bandit.observe("cross", SimpleNamespace(is_island_event=True, r_norm=.2))
    op, propensity, statistics = bandit.pick(False, random.Random(1), cross_available=True)
    assert statistics["cross"] == .2
    assert op in statistics and 0 < propensity < 1


def test_single_program_island_masks_cross_in_actual_engine(make_engine):
    engine = make_engine()
    parent = engine.db.add(PARENT, .2, {}, 0, "seed")
    for _ in range(8):
        plan = engine._prepare_step(0, spawn=False)
        assert plan["snap"].parent_id == parent.id
        assert plan["op"] in {"diff", "full_rewrite"}
        assert plan["op_prop"] == pytest.approx(.5)
        assert "cross" not in plan["op_stat"]
        assert plan["snap"].inspirations == []


def test_actual_engine_finds_source_distinct_donor_after_ranked_aliases(make_engine):
    engine = make_engine(op_fixed="cross")
    parent = engine.db.add(PARENT, .9, {}, 0, "seed")
    aliases = [engine.db.add(PARENT, .8 - i / 100, {}, 0, "migration") for i in range(5)]
    donor = engine.db.add(DONOR, .3, {}, 0, "gen")
    plan = engine._prepare_step(0, spawn=False,
        excluded_parents=[p.id for p in aliases] + [donor.id])
    assert plan["op"] == "cross"
    assert plan["snap"].parent_id == parent.id
    assert [r["program_id"] for r in plan["snap"].inspirations] == [donor.id]
    bundle = source_bundle(plan["snap"])
    assert bundle["inspirations"][0]["program_id"] == donor.id
    assert bundle["inspirations"][0]["source_ref"] == source_digest(DONOR)


def test_actual_bandit_cross_gets_credit_for_real_two_source_improvement(make_engine, monkeypatch):
    engine = make_engine()
    parent = engine.db.add(PARENT, .2, {}, 0, "seed")
    donor = engine.db.add(DONOR, .3, {}, 0, "gen")
    stream = engine.rng.stream
    class ChooseCross:
        def choices(self, arms, *, weights, k):
            assert "cross" in arms
            assert weights == pytest.approx([1 / 3] * 3)
            return ["cross"]
    monkeypatch.setattr(engine.rng, "stream", lambda name: ChooseCross() if name == "op" else stream(name))
    plan = engine._prepare_step(0, spawn=False, excluded_parents=[donor.id])
    assert plan["snap"].parent_id == parent.id
    assert plan["op"] == "cross"
    assert plan["op_prop"] == pytest.approx(1 / 3)
    assert plan["snap"].operator_decision["on_policy"] is True
    result = engine._execute_step(plan, evaluate_fn=outcome)
    assert result.meta["op_used"] == "cross"
    assert result.meta["op_on_policy"] is True
    engine._commit_step(plan, result)
    events = [json.loads(line) for line in Path(engine.log.path).read_text().splitlines()]
    committed = next(event for event in reversed(events) if event["event"] == "iteration")
    assert committed["op_used"] == "cross" and committed["r_norm"] > 0
    assert list(engine.op_bandit.events["cross"]) == [committed["r_norm"]]
    assert list(engine.op_bandit.events["full_rewrite"]) == []
    assert list(engine.op_bandit.events["diff"]) == []


@pytest.mark.parametrize("aliases", [0, 3])
def test_forced_empty_cross_rewrites_without_cross_or_on_policy_credit(make_engine, aliases):
    engine = make_engine(op_fixed="cross")
    engine.db.add(PARENT, .2, {}, 0, "seed")
    for _ in range(aliases):
        engine.db.add(PARENT, .2, {}, 0, "migration")
    plan = engine._prepare_step(0, spawn=False)
    assert plan["op"] == plan["snap"].op == "full_rewrite"
    assert plan["op_prop"] is None
    decision = plan["snap"].operator_decision
    assert decision["requested"] == "cross"
    assert decision["effective"] == "full_rewrite"
    assert decision["on_policy"] is False
    assert decision["fallback_reason"]
    result = engine._execute_step(plan, evaluate_fn=outcome)
    assert result.meta["op_used"] == "full_rewrite"
    assert result.meta["op_requested"] == "cross"
    assert result.meta["op_on_policy"] is False
    engine._commit_step(plan, result)
    assert list(engine.op_bandit.events["cross"]) == []
    assert list(engine.op_bandit.events["full_rewrite"]) == []


@pytest.mark.parametrize("donors", [[], [row("alias", PARENT)]])
def test_standalone_worker_downgrades_before_first_prompt_and_keeps_cost(donors):
    snap = Snapshot(0, "cross", "fixture", 0, "p1", PARENT, .2,
                    inspirations=copy.deepcopy(donors), required_entrypoints=["run_search"])
    before = copy.deepcopy(snap)
    provider = Provider()
    result = run_worker("cross", snap, provider, outcome, dup_cap=1)
    assert snap == before
    assert len(provider.prompts) == 1
    assert "You are combining two approaches" not in provider.prompts[0]
    assert result.meta["op_requested"] == "cross"
    assert result.meta["op_initial"] == result.meta["op_used"] == "full_rewrite"
    assert result.meta["op_fallback_reason"]
    assert result.meta["op_on_policy"] is False
    assert result.cost == .01


def test_no_child_keeps_fallback_attribution_and_spent_request_cost():
    snap = Snapshot(0, "cross", "fixture", 0, "p1", PARENT, .2,
                    required_entrypoints=["run_search"])
    provider = Provider([""])
    def forbidden_evaluation(code):
        pytest.fail("Empty generation must never reach evaluation")
    result = run_worker("cross", snap, provider, forbidden_evaluation, dup_cap=1)
    assert isinstance(result, NoChild)
    assert result.meta["op_requested"] == "cross"
    assert result.meta["op_used"] == "full_rewrite"
    assert result.meta["op_on_policy"] is False
    assert result.meta["op_fallback_reason"] == "no_usable_crossover_donor"
    assert result.cost == .01


def test_worker_preserves_genuine_cross_and_exact_donor_catalog():
    snap = Snapshot(0, "cross", "fixture", 0, "p1", PARENT, .2,
        inspirations=[row("p2", DONOR)], required_entrypoints=["run_search"],
        research_context={}, operator_decision={"requested": "cross", "effective": "cross",
            "selection": "bandit", "fallback_reason": None, "on_policy": True})
    provider = Provider()
    result = run_worker("cross", snap, provider, outcome, dup_cap=1)
    assert "You are combining two approaches" in provider.prompts[0]
    assert "INSPIRATIONS FROM OTHER SOLUTIONS:\n(none)" not in provider.prompts[0]
    assert result.meta["op_initial"] == result.meta["op_used"] == "cross"
    assert result.meta["op_on_policy"] is True
    assert result.meta["op_fallback_reason"] is None
    assert DONOR in source_bundle(snap)["catalog"].values()


def test_worker_rejects_wrong_frozen_operator_before_dispatch():
    snap = Snapshot(0, "diff", "fixture", 0, "p1", PARENT, .2,
        operator_decision={"requested": "diff", "effective": "diff",
            "selection": "bandit", "fallback_reason": None, "on_policy": True})
    provider = Provider()
    with pytest.raises(ValueError, match="frozen decision"):
        run_worker("cross", snap, provider, outcome, dup_cap=1)
    assert provider.prompts == []


def test_actual_frozen_job_records_donor_and_decision_detached_from_population(make_engine, tmp_path):
    engine = make_engine(op_fixed="cross")
    parent = engine.db.add(PARENT, .2, {}, 0, "seed")
    donor = engine.db.add(DONOR, .3, {}, 0, "gen")
    plan = engine._prepare_step(0, spawn=False, excluded_parents=[donor.id])
    frozen = _intent(engine, "G:1", plan)
    directory = tmp_path / "wave"
    directory.mkdir()
    _save_intents(directory, [frozen])
    saved = (directory / "jobs.json").read_bytes()
    donor.code = CHILD
    donor.card = "later replacement"
    assert plan["snap"].inspirations[0]["code"] == DONOR
    replay = _intent(engine, "G:1", plan)
    _save_intents(directory, [replay])
    assert (directory / "jobs.json").read_bytes() == saved
    restored = json.loads(saved)["jobs"][0]["snapshot"]
    assert restored["operator_decision"] == plan["snap"].operator_decision
    assert restored["parent_id"] == parent.id
    assert restored["inspirations"][0]["program_id"] == donor.id
    assert restored["inspirations"][0]["source_sha256"] == source_digest(DONOR)
    assert restored["inspirations"][0]["code"] == DONOR
