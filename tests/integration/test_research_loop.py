"""Full research loop with local deterministic fixtures and no network calls."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re
import sys
import textwrap
from decimal import Decimal
from types import SimpleNamespace
import pytest
from sciel.hypothesis.rendering import decode_evidence
from sciel.knowledge_base.presentation import expand_reflection_payload
from sciel.campaign.budget import BudgetedLLM, CampaignStopped, CostLedger
from sciel.config import apply_defaults
from sciel.engine import checkpoint
from sciel.engine.llm.client import LLMResponse
from sciel.engine.loop import Engine
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.eval.registry import load_task
from .research_loop_fixtures import (
    pytestmark, LESSON, research_task, config,
    RecordingLLM, hypothesis_code, events, engine,
    fixed_evaluate, prompt_json, unknown_mechanism, successful_response,
    research_events, quick_engine,
)


def test_real_evaluator_sees_hypothesis_before_code_and_next_prompt_uses_knowledge(research_task, tmp_path):
    run_dir = tmp_path / "full"
    model = RecordingLLM(successful_response)
    instance = engine(research_task, run_dir, model)
    result = instance.run()

    assert result["reason"] == "budget" and result["B_evals"] == 3
    records = instance.research.knowledge.experiments
    assert len(records) == 3 and all(r["status"] == "evaluated" for r in records)
    assert records[0]["id"] == "r000001a1"
    assert len({r["id"] for r in records}) == 3
    assert all(r["actual_delta"] == r["score"] - r["parent_score"] for r in records)
    assert all(r["forecast_verdict"] == "CONFIRMED" for r in records)
    assert records[0]["injected_ids"] == [] and all(r["injected_ids"] for r in records[1:])

    observed = [json.loads(line) for line in (run_dir / "evaluation_observations.jsonl").read_text().splitlines()]
    for record in records:
        matching = [row for row in observed if row["source_sha256"] == record["source_sha256"]]
        assert len(matching) == 2, "both real smoke and full evaluations must be observed"
        for row in matching:
            planned = [e for e in row["research_before_evaluation"] if e["phase"] == "hypothesis"
                       and e["experiment_id"] == record["id"]]
            assert len(planned) == 1
            assert planned[0]["evidence"]["source_sha256"] == row["source_sha256"]
            assert planned[0]["evidence"]["status"] == "pending"
            assert not any(e["phase"] == "outcome" and e["experiment_id"] == record["id"]
                           for e in row["research_before_evaluation"])

    generation = [c for c in model.calls if "HYPOTHESIS_JSON" in c["prompt"]]
    assert len(generation) == 3 and LESSON not in generation[0]["prompt"]
    assert all(LESSON in c["prompt"] for c in generation[1:])
    assert all("Measured source observations:" in c["prompt"] for c in generation[1:])
    assert all("not causal proof" in c["prompt"] for c in generation[1:])
    assert len(research_events(run_dir, "conclusion")) == 3
    assert len(research_events(run_dir, "reflection")) == 1
    assert result["cost_total"] == pytest.approx(.12)
    assert result["cost_total"] == pytest.approx(sum(c["cost"] for c in model.calls))
    assert result["research"]["aux_cost"] == pytest.approx(.09)
    assert result["research"]["reflections"] == 1 and result["research"]["reflection_changes"] == 0
    saved = json.loads((run_dir / "checkpoint.json").read_text())
    assert saved["research"] == instance.research.state()


@pytest.mark.parametrize("proposal,status", [
    ("```python\ndef run_search():\n    return 0.8\n```", "hypothesis_invalid"),
    (hypothesis_code("def run_search():\n    return 0.1\n"), "no_child"),
])
def test_rejected_proposals_are_evidence_without_evaluation_or_invented_delta(
        research_task, tmp_path, proposal, status):
    instance = quick_engine(research_task, tmp_path / status, RecordingLLM(lambda _: proposal),
                            max_iterations=1)
    evaluated = []
    def observe(code):
        evaluated.append(code)
        return fixed_evaluate(code)
    instance._evaluate = observe
    instance.run()
    assert len(evaluated) == 1, "only the initial seed is evaluated"
    record, = instance.research.knowledge.experiments
    assert record["status"] == status
    assert record["score"] is None and record["actual_delta"] is None
    assert record["forecast_verdict"] == "INCONCLUSIVE"
    assert instance.research.knowledge.lessons == []
    assert instance.research.aux_calls == 0
    assert instance.cost_total == pytest.approx(.01)
    proposal_event, = research_events(tmp_path / status, "hypothesis")
    assert proposal_event["response"] == proposal
    assert proposal_event["response_sha256"] == hashlib.sha256(proposal.encode()).hexdigest()
    assert proposal_event["response_truncated"] is False
    assert proposal_event["completion_tokens"] == len(proposal) // 4


def test_evaluation_failure_is_retained_but_llm_cannot_invent_support(research_task, tmp_path):
    def respond(prompt):
        if "CONCLUSION_JSON" in prompt:
            record = prompt_json(prompt, "EVIDENCE_JSON:")
            return "CONCLUSION_JSON: " + json.dumps({"verdict": "CONFIRMED",
                "explanation": "An intentionally unsupported model claim.",
                "mechanism_review": unknown_mechanism(record), "lessons": [{
                    "rule": "Failures prove this improves the score.", "polarity": "positive",
                    "scope": "branch", "evidence_ids": [record["id"]]}]})
        return hypothesis_code("def run_search():\n    raise RuntimeError('fixture failure')\n")
    run_dir = tmp_path / "failure"
    instance = engine(research_task, run_dir, RecordingLLM(respond), max_iterations=1)
    result = instance.run()
    record, = instance.research.knowledge.experiments
    assert result["B_evals"] == 0 and record["status"] == "eval_failed"
    assert record["score"] is None and record["actual_delta"] is None
    assert record["forecast_verdict"] == "INCONCLUSIVE"
    assert record["failure"]
    assert instance.research.knowledge.lessons == []
    conclusion, = research_events(run_dir, "conclusion")
    assert conclusion["llm"]["verdict"] == "CONFIRMED"
    assert conclusion["observed_forecast_verdict"] == "INCONCLUSIVE"
    assert conclusion["accepted_lesson_ids"] == []
    assert result["cost_total"] == pytest.approx(.03)


def test_grafted_code_does_not_confirm_the_unexecuted_proposal(research_task, tmp_path):
    def respond(prompt):
        if "CONCLUSION_JSON" in prompt:
            return successful_response(prompt)
        return hypothesis_code("def helper():\n    return 0.8\n")
    instance = engine(research_task, tmp_path / "repair", RecordingLLM(respond), budget=1)
    instance.run()
    record, = instance.research.knowledge.experiments
    assert record["status"] == "evaluated" and record["score"] == .1
    assert record["source_repaired"] is True
    assert record["source_sha256"] != record["proposed_sha256"]
    assert record["actual_delta"] == 0 and record["forecast_verdict"] == "INCONCLUSIVE"
    assert instance.research.knowledge.lessons == []


def test_auxiliary_request_limit_preserves_outcomes_and_charges_only_dispatched_calls(research_task, tmp_path):
    research = {**config()["research"], "max_aux_calls": 1}
    model = RecordingLLM(successful_response)
    instance = quick_engine(research_task, tmp_path / "limit", model, research=research)
    result = instance.run()
    assert result["B_evals"] == 3
    assert len(instance.research.knowledge.experiments) == 3
    assert instance.research.aux_calls == 1 and len(model.calls) == 4
    assert result["cost_total"] == pytest.approx(.05)
    skipped = research_events(tmp_path / "limit", "aux_skipped")
    assert {row["kind"] for row in skipped} == {"conclusion", "reflection"}
    assert all(row["reason"] == "aux_call_limit" for row in skipped)


@pytest.mark.parametrize("phase", ["generation", "conclusion", "reflection"])
def test_campaign_stop_is_persisted_and_never_replayed(research_task, tmp_path, phase):
    def respond(prompt):
        here = "reflection" if "REFLECTION_JSON" in prompt else "conclusion" if "CONCLUSION_JSON" in prompt else "generation"
        if here == phase:
            raise CampaignStopped("fixture campaign exhausted")
        return successful_response(prompt)
    run_dir = tmp_path / phase
    model = RecordingLLM(respond)
    instance = quick_engine(research_task, run_dir, model)
    with pytest.raises(CampaignStopped):
        instance.run()
    assert instance.research.stopped_reason
    assert instance.cost_total == pytest.approx(sum(c["cost"] for c in model.calls))
    saved = json.loads((run_dir / "checkpoint.json").read_text())
    assert saved["research"]["stopped_reason"]
    if phase == "generation":
        record = instance.research.knowledge.experiments[-1]
        assert record["generation_cost"] is None, "no response does not establish a free request"
        assert research_events(run_dir, "outcome")[-1]["evidence"]["generation_cost"] is None
    assert not saved["research"]["pending"] and not saved["research"]["queue"]
    assert events(run_dir)[-1]["reason"] == "research_stopped"
    fresh = RecordingLLM(successful_response)
    resumed = quick_engine(research_task, run_dir, fresh)
    checkpoint.load(resumed, str(run_dir / "checkpoint.json"))
    with pytest.raises(CampaignStopped):
        resumed.run()
    assert fresh.calls == []


def test_checkpoint_resume_preserves_research_reflection_rng_population_and_prompts(research_task, tmp_path):
    whole_model, first_model, rest_model = [RecordingLLM(successful_response) for _ in range(3)]
    whole = quick_engine(research_task, tmp_path / "whole", whole_model, budget=4)
    whole_result = whole.run()
    split_dir = tmp_path / "split"
    first = quick_engine(research_task, split_dir, first_model, budget=4)
    first._stop_at = 2
    assert first.run()["reason"] == "paused"
    assert first.research.reflections == 1
    saved_research = first.research.state()
    resumed = quick_engine(research_task, split_dir, rest_model, budget=4)
    checkpoint.load(resumed, str(split_dir / "checkpoint.json"))
    assert resumed.research.state() == saved_research
    resumed_result = resumed.run()
    assert resumed_result == whole_result
    assert resumed.research.state() == whole.research.state()
    assert resumed.rng.state() == whole.rng.state()
    assert resumed.db.state() == whole.db.state()
    assert first_model.calls + rest_model.calls == whole_model.calls
    assert [e["through"] for e in research_events(split_dir, "reflection")] == [2, 4]


def test_default_legacy_run_does_not_enable_research_or_auxiliary_calls(research_task, tmp_path):
    raw = config()
    raw.pop("research")
    assert not apply_defaults(raw).get("research", {}).get("enabled", False)
    model = RecordingLLM(lambda _: "```python\ndef run_search():\n    return 0.2\n```")
    instance = Engine({**raw, "budget": 1}, research_task, str(tmp_path / "legacy"), model,
                      python=sys.executable)
    instance._evaluate = fixed_evaluate
    result = instance.run()
    assert instance.research is None and not instance.kb_on and not instance.c2_on
    assert "research" not in result and len(model.calls) == 1
    assert "HYPOTHESIS_JSON" not in model.calls[0]["prompt"]
    assert not [e for e in events(tmp_path / "legacy") if e["event"] == "research"]
    assert "research" not in json.loads((tmp_path / "legacy" / "checkpoint.json").read_text())


@pytest.mark.parametrize("old_flag", ["kb_free", "c2_event_bets", "bet_enabled"])
def test_research_cannot_mix_with_legacy_knowledge_and_bets(old_flag):
    with pytest.raises(ValueError, match="research replaces legacy"):
        apply_defaults(config(**{old_flag: True}))


@pytest.mark.parametrize("interrupt_conclusion", [False, True])
def test_generation_and_research_share_real_ledger_with_injected_local_client(
        research_task, tmp_path, interrupt_conclusion):
    prompts = []
    def create(**kwargs):
        prompt = kwargs["messages"][0]["content"]
        prompts.append(prompt)
        if interrupt_conclusion and "CONCLUSION_JSON" in prompt:
            raise TimeoutError("local simulated provider timeout")
        return SimpleNamespace(id=f"fixture-{len(prompts)}",
            usage=SimpleNamespace(cost=.0001, prompt_tokens=20, completion_tokens=40),
            choices=[SimpleNamespace(message=SimpleNamespace(content=successful_response(prompt)))])
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    ledger = CostLedger(tmp_path / "local_costs.json", 1)
    cfg = {"models": ["local-test-model"], "weights": [1], "max_tokens": 100,
           "api_base": "https://openrouter.ai/api/v1"}
    prices = {"models": [{"id": "local-test-model", "pricing": {
        "prompt": "0.000001", "completion": "0.000002"}}]}
    model = BudgetedLLM(cfg, prices, ledger, "local-research", "unused-fixture-key", client=client)
    run_dir = tmp_path / "ledger_run"
    instance = quick_engine(research_task, run_dir, model, budget=2)
    if interrupt_conclusion:
        with pytest.raises(CampaignStopped, match="outcome is unknown"):
            instance.run()
        assert len(prompts) == 2
        state = ledger.snapshot()
        assert Decimal(state["known_cost_usd"]) == Decimal(".0001")
        assert Decimal(state["reserved_or_unknown_usd"]) > 0 and state["halted"]
        assert instance.cost_total == pytest.approx(.0001)
        # Restore must refuse the stopped controller before invoking the client,
        # even if the caller constructs a fresh wrapper over the same ledger.
        restored = quick_engine(research_task, run_dir, model, budget=2)
        checkpoint.load(restored, str(run_dir / "checkpoint.json"))
        with pytest.raises(CampaignStopped):
            restored.run()
        assert len(prompts) == 2
    else:
        result = instance.run()
        assert len(prompts) == 5  # two generations, two conclusions, one reflection
        state = ledger.snapshot()
        assert Decimal(state["known_cost_usd"]) == Decimal(".0005")
        assert Decimal(state["reserved_or_unknown_usd"]) == 0
        assert result["cost_total"] == pytest.approx(.0005)
        assert result["research"]["aux_cost"] == pytest.approx(.0003)
    calls = json.loads((tmp_path / "local_costs.json").read_text())["calls"]
    assert len(calls) == len(prompts)
    assert {row["run_id"] for row in calls.values()} == {"local-research"}


def test_resume_refuses_research_calls_after_checkpoint_instead_of_repeating_them(research_task, tmp_path):
    run_dir = tmp_path / "uncheckpointed"
    instance = quick_engine(research_task, run_dir, RecordingLLM(successful_response),
                            milestone_fracs=[1 / 3])
    class SimulatedHostCrash(BaseException):
        pass
    count = 0
    def crash_after_checkpoint(code):
        nonlocal count
        count += 1
        if count == 3:  # seed, first successful mutation/checkpoint, next proposal
            raise SimulatedHostCrash()
        return fixed_evaluate(code)
    instance._evaluate = crash_after_checkpoint
    with pytest.raises(SimulatedHostCrash):
        instance.run()
    instance.log.close()
    fresh = RecordingLLM(successful_response)
    resumed = quick_engine(research_task, run_dir, fresh, milestone_fracs=[1 / 3])
    with pytest.raises(checkpoint.IdentityMismatch, match="uncheckpointed research"):
        checkpoint.load(resumed, str(run_dir / "checkpoint.json"))
    assert fresh.calls == []


def test_portfolio_bootstrap_uses_research_and_distinct_ids_without_spending_eval_budget(research_task, tmp_path):
    model = RecordingLLM(successful_response)
    instance = quick_engine(research_task, tmp_path / "portfolio", model, portfolio_seed=True, budget=2)
    result = instance.run()
    records = instance.research.knowledge.experiments
    assert result["B_evals"] == 2 and len(records) == 3
    assert len({r["id"] for r in records}) == 3
    assert records[0]["iteration"] == 0
    assert all(r["status"] == "evaluated" and r["actual_delta"] > 0 for r in records)
    assert all("HYPOTHESIS_JSON" in call["prompt"] for call in model.calls
               if "CONCLUSION_JSON" not in call["prompt"] and "REFLECTION_JSON" not in call["prompt"])
    assert result["cost_total"] == pytest.approx(.12)
    source_hashes = {hashlib.sha256(p.code.encode()).hexdigest() for p in instance.db._all.values()}
    assert {r["source_sha256"] for r in records} <= source_hashes
    assert instance.research.reflections == 1


def test_portfolio_auxiliary_stop_cannot_be_retried_on_resume(research_task, tmp_path):
    def respond(prompt):
        if "CONCLUSION_JSON" in prompt:
            raise CampaignStopped("portfolio conclusion interrupted")
        return successful_response(prompt)
    run_dir = tmp_path / "portfolio_stop"
    instance = quick_engine(research_task, run_dir, RecordingLLM(respond), portfolio_seed=True)
    with pytest.raises(CampaignStopped):
        instance.run()
    assert instance.B_evals == 0 and instance.cost_total == pytest.approx(.01)
    assert instance.research.knowledge.experiments[0]["status"] == "evaluated"
    fresh = RecordingLLM(successful_response)
    resumed = quick_engine(research_task, run_dir, fresh, portfolio_seed=True)
    checkpoint.load(resumed, str(run_dir / "checkpoint.json"))
    with pytest.raises(CampaignStopped):
        resumed.run()
    assert fresh.calls == []


def test_research_model_must_belong_to_injected_ensemble_before_any_call(research_task, tmp_path):
    model = RecordingLLM(successful_response)
    research = {**config()["research"], "model": "not-in-ensemble"}
    with pytest.raises(ValueError, match="configured ensemble"):
        engine(research_task, tmp_path / "invalid_model", model, research=research)
    assert model.calls == []


def test_research_cannot_disable_fragment_capacity():
    with pytest.raises(ValueError, match="max_fragments"):
        apply_defaults(config(research={**config()["research"], "max_fragments": 0}))


@pytest.mark.parametrize("where", ["config", "backend"])
def test_research_rejects_hidden_request_retries_before_calls(research_task, tmp_path, where):
    model = RecordingLLM(successful_response)
    updates = {}
    if where == "config":
        updates["llm"] = {"mock": True, "max_retries": 2}
    else:
        model.max_retries = 2
    with pytest.raises(ValueError, match="hidden"):
        engine(research_task, tmp_path / where, model, **updates)
    assert model.calls == []


@pytest.mark.parametrize("journal", ["empty", "different", "truncated"])
def test_research_checkpoint_requires_its_exact_journal_prefix_before_calls(research_task, tmp_path, journal):
    source_dir = tmp_path / "source"
    source = quick_engine(research_task, source_dir, RecordingLLM(successful_response))
    source._stop_at = 1
    source.run()
    saved = json.loads((source_dir / "checkpoint.json").read_text())
    data = (source_dir / "sciel_events.jsonl").read_bytes()
    target_dir = tmp_path / journal
    target_dir.mkdir()
    if journal == "different":
        data = data.replace(b'"seed": 41', b'"seed": 42', 1)
    elif journal == "truncated":
        data = data[:saved["provenance"]["journal_anchor"]["byte_offset"] - 1]
    else:
        data = b""
    (target_dir / "sciel_events.jsonl").write_bytes(data)
    model = RecordingLLM(successful_response)
    target = quick_engine(research_task, target_dir, model)
    with pytest.raises(checkpoint.IdentityMismatch, match="journal does not match checkpoint prefix"):
        checkpoint.load(target, str(source_dir / "checkpoint.json"))
    assert model.calls == []


def test_dynamic_spawn_uses_research_without_id_collision_or_hidden_cost(research_task, tmp_path):
    def respond(prompt):
        if "CONCLUSION_JSON" in prompt or "REFLECTION_JSON" in prompt:
            return successful_response(prompt)
        tag = hashlib.sha256(prompt.encode()).hexdigest()[:12]
        return hypothesis_code(f"def tag_{tag}():\n    pass\ndef run_search():\n    return 0.1\n", low=0, high=0)
    model = RecordingLLM(respond)
    instance = quick_engine(research_task, tmp_path / "spawn", model, budget=2,
                            dynamic_islands=True, max_islands=2, spawn_stall=1)
    result = instance.run()
    assert result["B_evals"] == 2 and len(instance.db.islands) == 2
    records = instance.research.knowledge.experiments
    assert len(records) == len({r["id"] for r in records}) == 3
    assert all(r["status"] == "evaluated" for r in records)
    spawned = [p for p in instance.db._all.values() if p.origin == "spawn"]
    assert len(spawned) == 1
    assert hashlib.sha256(spawned[0].code.encode()).hexdigest() in {r["source_sha256"] for r in records}
    assert result["cost_total"] == pytest.approx(.12)
    assert result["cost_total"] == pytest.approx(sum(c["cost"] for c in model.calls))
