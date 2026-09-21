"""Two-stage ideas with fake LLM and evaluator functions; no source execution."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import re
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sciel.campaign.budget import BudgetedLLM, CampaignStopped, CostLedger
from sciel.config import apply_defaults
from sciel.engine import checkpoint
from sciel.engine.db import Database, normalized_hash
from sciel.engine.idea import TwoStageIdea, card_text, digest, parse_card
from sciel.engine.llm.client import LLMError, LLMResponse
from sciel.engine.loop import Engine
from sciel.engine.research import ResearchController
from sciel.engine.types import NoChild, Result, Snapshot
from sciel.engine.worker import run_worker
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.eval.registry import load_task


CARD = {"family": "grid", "strategy": "Improve the grid placement", "mechanism": "Change spacing"}
PARENT = "def run_search():\n    return 1.0\n"
CHILD = "def run_search():\n    return 1.5\n"


def code_response(code=CHILD):
    hyp = {"hypothesis": "The changed placement increases the score", "predicted_low": 0,
           "predicted_high": 1, "reasoning": "A proposed local change", "confidence": .5}
    return "HYPOTHESIS_JSON: " + json.dumps(hyp) + "\n```python\n" + code + "```\n"


def response(text, cost=.03):
    return LLMResponse(text, "fake", cost, 15, 8)


class Log:
    def __init__(self):
        self.events = []

    def research(self, **event):
        self.events.append(copy.deepcopy(event))


class Model:
    models = ["fake"]
    weights = [1.0]
    max_retries = 1

    def __init__(self, responses, log=None):
        self.responses, self.calls, self.log = list(responses), [], log

    def choose_model(self, rng):
        return "fake"

    def generate(self, prompt, model, temperature=None):
        if self.log:
            event = self.log.events[-1]
            assert event["phase"] == "idea" and event["action"] == "request_start"
            assert event["prompt_sha256"] == digest(prompt)
        self.calls.append(prompt)
        item = self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def setup(responses, references=()):
    cfg = apply_defaults({"research": {"enabled": True, "max_aux_calls": 0, "reflection_interval": 0}})
    log, charged = Log(), []
    model = Model(responses, log)
    research = ResearchController(cfg["research"], model, log, charged.append)
    db = Database(1)
    parent = db.add(PARENT, 1., {}, 0, "seed")
    snap = Snapshot(0, "idea", "fake", 0, parent.id, PARENT, 1.,
                    required_entrypoints=["run_search"], idea_cards=list(references))
    snap.research_context = research.context(db, parent, 0, 1)
    idea = TwoStageIdea(research)
    evaluated = []

    def evaluate(code):
        evaluated.append(code)
        assert log.events[-1]["action"] == "source_binding"
        assert log.events[-1]["card_sha256"] == digest(card_text(CARD))
        assert log.events[-1]["candidate_source_sha256"] == digest(code)
        assert any(e["phase"] == "hypothesis" for e in log.events)
        return CascadeOutcome(EvalResult(True, 1.5, {}, timing=.01), 1.5, "full", False)

    def run(evaluator=evaluate, **kwargs):
        return run_worker("idea", snap, model, evaluator, research=research, idea=idea, **kwargs)

    return research, idea, snap, model, log, charged, evaluated, run


@pytest.mark.parametrize("text", ["", "```json\n{}\n```", "[]", "null", None,
    json.dumps({**CARD, "extra": True}), json.dumps({**CARD, "family": ""}),
    json.dumps({**CARD, "family": "x" * 129}),
    '{"family":"grid","family":"other","strategy":"a","mechanism":"b"}',
    "prose\n" + json.dumps(CARD)])
def test_card_schema_rejects_ambiguous_or_unbounded_wire(text):
    assert parse_card(text) is None


def test_card_normalizes_whitespace_without_losing_math_or_order():
    assert parse_card("IDEA_CARD_JSON: " + json.dumps({**CARD, "strategy": " Improve the grid placement "})) == CARD
    less = {**CARD, "mechanism": "use x < 0"}
    greater = {**CARD, "mechanism": "use x > 0"}
    assert card_text(less) != card_text(greater)
    assert card_text({**CARD, "mechanism": "Use x"}) != card_text({**CARD, "mechanism": "use x"})
    assert card_text({**CARD, "strategy": "Match 'a b'"}) != card_text({**CARD, "strategy": "Match 'a  b'"})


def test_card_then_code_then_hypothesis_then_evaluation_and_sum_of_both_costs():
    research, idea, snap, model, log, charged, evaluated, run = setup([
        response(json.dumps(CARD), .03), response(code_response(), .07)])
    result = run(dup_cap=4)
    assert isinstance(result, Result) and result.ok
    assert len(model.calls) == 2 and evaluated == [CHILD] and charged == []
    assert result.cost == pytest.approx(.1)
    assert result.meta["idea_card"] == CARD
    assert result.meta["idea"]["card_sha256"] in model.calls[1]
    assert result.meta["idea"]["cost"] == pytest.approx(.1)
    assert [c["cost"] for c in result.meta["idea"]["calls"]] == [.03, .07]
    assert research.attempts == research.outcomes == 1
    assert research.knowledge.experiments[0]["generation_cost"] == .07  # Code request only.
    assert idea.calls_started == idea.calls_returned == 2 and idea.known_cost == pytest.approx(.1)
    state = json.loads(json.dumps(idea.state()))
    other = TwoStageIdea(research)
    other.load_state(state)
    assert other.state() == state


@pytest.mark.parametrize("text,reason", [("invalid", "idea_card_invalid"), (None, "idea_card_invalid"),
                                        (json.dumps(CARD), "idea_card_duplicate")])
def test_card_rejection_is_paid_but_has_no_code_call_or_fake_hypothesis(text, reason):
    refs = [{"program_id": "p9", "source_sha256": "a" * 64, "card": card_text(CARD)}]
    research, idea, _, model, _, charged, evaluated, run = setup([response(text, .03)], refs)
    result = run(dup_cap=4)
    assert isinstance(result, NoChild) and result.reason == reason
    assert len(model.calls) == 1 and result.cost == .03
    assert research.attempts == research.outcomes == 0 and research.knowledge.experiments == []
    assert not charged and not evaluated
    assert result.reason != "dup_cap" and not result.meta["escalated"]


@pytest.mark.parametrize("which", ["code_error", "code_stop", "card_error", "invalid_cost"])
def test_unknown_stage_stops_without_replay_and_accounts_known_costs(which):
    rows = [response(json.dumps(CARD), .03)]
    if which == "code_error":
        rows += [LLMError("private provider details")]
    elif which == "code_stop":
        rows += [CampaignStopped("local cap")]
    elif which == "card_error":
        rows = [LLMError("unknown")]
    else:
        rows += [response(code_response(), float("nan"))]
    research, idea, _, model, log, charged, evaluated, run = setup(rows)
    with pytest.raises(CampaignStopped):
        run()
    expected = 0. if which == "card_error" else .03
    assert sum(charged) == pytest.approx(expected)
    assert idea.known_cost == pytest.approx(expected) and not evaluated
    count = len(model.calls)
    with pytest.raises(CampaignStopped):
        run()
    assert len(model.calls) == count
    assert research.stopped_reason and idea.stopped_reason
    stopped = [e for e in log.events if e.get("action") == "stopped"][-1]
    assert stopped["current_call_cost_unknown"] is True
    assert "private provider details" not in json.dumps(log.events)


@pytest.mark.parametrize("text,reason", [(None, "idea_code_invalid"),
                                        ("```python\n" + CHILD + "```", "gate_rejected"),
                                        (code_response("def broken(:\n"), "gate_rejected")])
def test_code_stage_invalid_response_preserves_both_charges_without_evaluation(text, reason):
    research, _, _, model, _, _, evaluated, run = setup([response(json.dumps(CARD)), response(text, .07)])
    result = run()
    assert isinstance(result, NoChild) and result.reason == reason
    assert result.cost == pytest.approx(.1) and len(model.calls) == 2 and not evaluated
    assert research.outcomes == 1
    assert research.knowledge.experiments[0]["actual_delta"] is None


def test_evaluator_failure_is_single_measured_failure_not_resampled():
    research, idea, _, model, _, _, _, run = setup([response(json.dumps(CARD)), response(code_response(), .07)])
    def failed(code):
        return CascadeOutcome(EvalResult(False, 0., {}, artifacts={"error": "timed out", "error_type": "TimeoutError"}),
                              0., "smoke", False)
    result = run(failed)
    assert isinstance(result, Result) and not result.ok and result.cost == pytest.approx(.1)
    assert len(model.calls) == 2 and idea.evaluations == 1
    record = research.knowledge.experiments[0]
    assert record["actual_delta"] is None and record["forecast_verdict"] == "INCONCLUSIVE"
    assert "TimeoutError" in record["failure"]


def test_raised_evaluator_failure_is_not_mislabelled_as_generation_failure():
    research, idea, _, model, _, charged, _, run = setup([response(json.dumps(CARD)), response(code_response(), .07)])
    def crashed(code):
        raise RuntimeError("fixture evaluator failure")
    with pytest.raises(CampaignStopped):
        run(crashed)
    assert len(model.calls) == 2 and sum(charged) == pytest.approx(.1)
    assert idea.evaluations == 1
    record = research.knowledge.experiments[0]
    assert record['status'] == 'eval_failed' and record['failure'] == 'RuntimeError'
    assert record['source_sha256'] == digest(CHILD) and record['actual_delta'] is None
    assert record['forecast_verdict'] == 'INCONCLUSIVE'
    with pytest.raises(CampaignStopped):
        run(crashed)
    assert len(model.calls) == 2 and sum(charged) == pytest.approx(.1)


def test_duplicate_code_skips_novelty_and_evaluation_after_both_calls():
    _, _, snap, _, _, _, evaluated, run = setup([response(json.dumps(CARD)), response(code_response())])
    snap.code_hashes.add(normalized_hash(CHILD))
    result = run(novelty_check=lambda _: pytest.fail("duplicate must not reach novelty"))
    assert result.reason == "dup_cap" and result.cost == .06 and not evaluated


@pytest.mark.parametrize("config", [{"idea_two_stage": True}, {"idea_two_stage": {"enabled": 1}},
    {"idea_two_stage": {"enabled": True}}, {"idea_two_stage": {"extra": True}},
    {"idea_two_stage": {"enabled": True}, "research": {"enabled": True}, "llm": {"max_retries": 2}}])
def test_opt_in_requires_explicit_research_and_no_hidden_retries(config):
    with pytest.raises(ValueError):
        apply_defaults(config)


def test_disabled_defaults_and_direct_worker_path_remain_single_call():
    assert "idea_two_stage" not in apply_defaults({})
    snap = Snapshot(0, "idea", "fake", 0, "p1", PARENT, 1.)
    model = Model([response("```python\n" + CHILD + "```", .07)])
    result = run_worker("idea", snap, model, lambda _: CascadeOutcome(EvalResult(True, 1.5, {}), 1.5, "full", False))
    assert result.cost == .07 and len(model.calls) == 1 and "idea" not in result.meta


def build_engine(path, *, enabled=True):
    cfg = {"task": "test/toy_quadratic", "budget": 2, "max_iterations": 4, "num_islands": 1,
        "random_seed": 21, "dup_cap": 1, "op_fixed": "idea", "policy": {"enabled": False},
        "portfolio_seed": False, "dynamic_islands": False, "embed_grid": False, "idea_island": True,
        "crossover": False, "milestone_fracs": [], "confirm_records": False,
        "llm": {"mock": True}, "idea_two_stage": {"enabled": enabled},
        "research": {"enabled": True, "max_aux_calls": 0, "reflection_interval": 0}}

    class AutoModel(Model):
        def __init__(self):
            super().__init__([])

        def generate(self, prompt, model, temperature=None):
            self.calls.append(prompt)
            if "IDEA STAGE 1:" in prompt:
                iteration = int(re.search(r"ITERATION: (\d+)", prompt)[1])
                return response(json.dumps({**CARD, "strategy": f"fixture step {iteration}"}), .03)
            iteration = int(re.search(r"fixture step (\d+)", prompt)[1])
            return response(code_response(f"def run_search():\n    return {iteration / 10}, 0.0\n"), .07)

    task = load_task(str(Path(__file__).resolve().parents[2] / "tasks/test/toy_quadratic"))
    engine = Engine(cfg, task, str(path), AutoModel(), python=sys.executable)
    def evaluate(code):
        match = re.search(r"return (0\.\d+), 0\.0", code)
        value = float(match[1]) if match else 0.
        return CascadeOutcome(EvalResult(True, value, {}, timing=.01), value, "full", False)
    engine._evaluate = evaluate  # Numeric fixture only: candidate is never imported/executed.
    return engine


def test_engine_checkpoint_resume_preserves_cards_cost_counters_and_next_decisions(tmp_path):
    continuous = build_engine(tmp_path / "continuous")
    expected = continuous.run()
    paused = build_engine(tmp_path / "split")
    paused._stop_at = 1
    assert paused.run()["reason"] == "paused"
    resumed = build_engine(tmp_path / "split")
    checkpoint.load(resumed, str(tmp_path / "split/checkpoint.json"))
    assert resumed.run() == expected
    for name in ("idea", "research", "db", "rng", "selection", "schedule"):
        assert getattr(resumed, name).state() == getattr(continuous, name).state()
    assert resumed.cost_total == pytest.approx(.2) and resumed.idea.calls_returned == 4
    assert all(parse_card(p.card) is not None for p in resumed.db._all.values() if p.origin == "gen")


def test_post_checkpoint_card_start_blocks_resume_before_any_call(tmp_path):
    paused = build_engine(tmp_path / "split")
    paused._stop_at = 1
    paused.run()
    path = tmp_path / "split/sciel_events.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps({"event": "research", "phase": "idea", "action": "request_start"}) + "\n")
    resumed = build_engine(tmp_path / "split")
    with pytest.raises(checkpoint.IdentityMismatch, match="uncheckpointed research activity"):
        checkpoint.load(resumed, str(tmp_path / "split/checkpoint.json"))
    assert resumed.llm.calls == []


def test_idea_identity_and_unfinished_state_cannot_be_resumed(tmp_path):
    original = build_engine(tmp_path / "run")
    original._stop_at = 1
    original.run()
    changed = build_engine(tmp_path / "run", enabled=False)
    with pytest.raises(checkpoint.IdentityMismatch, match="idea_two_stage"):
        checkpoint.load(changed, str(tmp_path / "run/checkpoint.json"))
    state = original.idea.state()
    state["pending"] = {"stage": "code"}
    with pytest.raises(ValueError, match="unfinished"):
        original.idea.load_state(state)


@pytest.mark.parametrize("unknown_second", [False, True])
def test_both_stages_share_real_budget_wrapper_with_fake_provider_only(tmp_path, unknown_second):
    research, idea, snap, _, log, charged, _, _ = setup([])
    client = Mock()
    provider_calls = []
    def create(**kwargs):
        provider_calls.append(kwargs)
        assert log.events[-1]['action'] == 'request_start'
        if len(provider_calls) == 2 and unknown_second:
            raise RuntimeError('fake provider disconnect')
        text = json.dumps(CARD) if len(provider_calls) == 1 else code_response()
        return SimpleNamespace(id=f'fake-{len(provider_calls)}',
            usage=SimpleNamespace(cost=.0001, prompt_tokens=100, completion_tokens=20),
            choices=[SimpleNamespace(message=SimpleNamespace(content=text))])
    client.chat.completions.create.side_effect = create
    ledger = CostLedger(tmp_path / 'ledger.json', 1)
    llm = BudgetedLLM({'models': ['fake'], 'weights': [1], 'max_tokens': 100,
        'api_base': 'https://openrouter.ai/api/v1'},
        {'models': [{'id': 'fake', 'pricing': {'prompt': '.000001', 'completion': '.000002'}}]},
        ledger, 'idea-fixture', 'not-a-real-key', client=client)
    research.llm = llm
    def evaluate(code):
        return CascadeOutcome(EvalResult(True, 1.5, {}), 1.5, 'full', False)
    if unknown_second:
        with pytest.raises(CampaignStopped):
            idea.run(snap, llm, evaluate)
        assert sum(charged) == .0001
        with pytest.raises(CampaignStopped):
            idea.run(snap, llm, evaluate)
    else:
        result = idea.run(snap, llm, evaluate)
        assert result.cost == .0002 and charged == []
    saved = json.loads(ledger.path.read_text())
    assert len(saved['calls']) == len(provider_calls) == 2
    assert {row['run_id'] for row in saved['calls'].values()} == {'idea-fixture'}
    known, held = CostLedger.totals(saved)
    assert float(known) == (.0001 if unknown_second else .0002)
    assert (held > 0) is unknown_second
