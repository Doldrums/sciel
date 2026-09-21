"""Repeat population reflection ordering with inert Engine/provider/draw fixtures.

No source, evaluator, helper subprocess, optimizer or model service is executed.
"""
import copy
import json
import sys

import pytest

from sciel.engine import checkpoint
from sciel.engine.llm.client import LLMResponse
from sciel.engine.loop import Engine
from sciel.hypothesis.rendering import decode_evidence

from _mechanism_fixtures import unknown_mechanism
from test_repeat_population_engine import Provider, settings, task, numeric, draws


class ResearchProvider(Provider):
    def __init__(self, draws, start=0):
        super().__init__(start)
        self.draws = draws
        self.auxiliary = []

    def generate(self, prompt, model, temperature):
        if "CONCLUSION_JSON" not in prompt and "REFLECTION_JSON" not in prompt:
            return super().generate(prompt, model, temperature)
        evidence = decode_evidence(json.JSONDecoder().raw_decode(
            prompt.rsplit("EVIDENCE_JSON:\n", 1)[1])[0])
        kind = "conclusion" if "CONCLUSION_JSON" in prompt else "reflection"
        self.auxiliary.append({"kind": kind, "prompt": prompt, "evidence": evidence,
                               "draw_count": len(self.draws.calls)})
        if kind == "conclusion":
            value = {"verdict": evidence["forecast_verdict"],
                     "explanation": "Only the original ordinary forecast is assessed.",
                     "lessons": [], "mechanism_review": unknown_mechanism(evidence)}
            text = "CONCLUSION_JSON:\n" + json.dumps(value)
            cost = .002
        else:
            text = 'REFLECTION_JSON:\n{"lessons":[],"retire_ids":[]}'
            cost = .003
        return LLMResponse(text, model, cost, 10, 10)


def construct(task, path, draws, *, start=0):
    cfg = settings(budget=1, max_iterations=1,
                   research={"enabled": True, "max_aux_calls": 2,
                             "reflection_interval": 1, "retrieve_k": 2})
    provider = ResearchProvider(draws, start)
    return Engine(cfg, task, str(path), provider, python=sys.executable), provider


def reflection(provider):
    return [call for call in provider.auxiliary if call["kind"] == "reflection"]


def assert_terminal_reflection(engine, provider, numeric):
    calls = reflection(provider)
    assert len(calls) == 1
    assert calls[0]["draw_count"] == 9
    group = engine.research.knowledge.comparisons[0]
    assert group["id"] in calls[0]["evidence"]["observation_catalog"]
    ordinary = engine.research.knowledge.experiments[0]
    assert ordinary["score"] == numeric[1]["score"]
    assert ordinary["parent_score"] == numeric[0]["score"]
    assert ordinary["actual_delta"] == numeric[1]["score"] - numeric[0]["score"]
    assert ordinary["forecast_verdict"] == engine.research.conclusions[ordinary["id"]]["observed_forecast_verdict"]
    assert group["forecast"] == "not_scored"
    assert engine.research.outcomes == engine.research.reflected_through == 1
    assert engine.research.aux_calls == 2
    assert engine.research.aux_cost == pytest.approx(.005)
    assert engine.cost_total == pytest.approx(.006)


def test_actual_reflection_request_follows_complete_population_comparison(
        task, tmp_path, numeric, draws):
    engine, provider = construct(task, tmp_path / "ordinary-and-repeat", draws)
    engine.run()
    assert len(provider.calls) == 1 and len(numeric) == 2
    assert [call["kind"] for call in provider.auxiliary] == ["conclusion", "reflection"]
    assert provider.auxiliary[0]["draw_count"] == 3
    assert_terminal_reflection(engine, provider, numeric)


def test_final_budget_resume_flushes_owed_reflection_once_without_new_draws(
        task, tmp_path, monkeypatch, numeric, draws):
    path = tmp_path / "terminal-before-import-checkpoint"
    engine, first_provider = construct(task, path, draws)
    original_apply = engine.repeat_population.apply

    def crash_after_transition(packet):
        transition = original_apply(packet)
        if packet["kind"] == "comparison":
            raise KeyboardInterrupt("fixture crash after terminal pure transition")
        return transition

    monkeypatch.setattr(engine.repeat_population, "apply", crash_after_transition)
    with pytest.raises(KeyboardInterrupt):
        engine.run()
    assert len(draws.calls) == 9
    assert len(first_provider.calls) == 1
    assert reflection(first_provider) == []
    saved = json.loads((path / "checkpoint.json").read_text())
    assert saved["research"]["reflected_through"] == 0

    resumed, later = construct(task, path, draws, start=1)
    checkpoint.load(resumed, str(path / "checkpoint.json"))
    resumed.run()
    assert len(draws.calls) == 9 and len(numeric) == 2 and later.calls == []
    assert [call["kind"] for call in later.auxiliary] == ["reflection"]
    assert_terminal_reflection(resumed, later, numeric)

    again, third = construct(task, path, draws, start=1)
    checkpoint.load(again, str(path / "checkpoint.json"))
    again.run()
    assert len(draws.calls) == 9 and len(numeric) == 2
    assert third.calls == third.auxiliary == []


def test_unknown_draw_does_not_dispatch_owed_reflection_on_resume(
        task, tmp_path, numeric, draws):
    path = tmp_path / "unknown-draw"
    first, provider = construct(task, path, draws)

    def interrupt(call):
        if len(draws.calls) == 4:
            raise KeyboardInterrupt("fixture vanished after durable draw intent")

    draws.before_draw = interrupt
    with pytest.raises(KeyboardInterrupt):
        first.run()
    assert reflection(provider) == []
    assert first.research.reflected_through == 0
    draws.before_draw = None
    resumed, later = construct(task, path, draws, start=1)
    checkpoint.load(resumed, str(path / "checkpoint.json"))
    assert resumed.run()["reason"] == "repeat_stopped"
    assert len(draws.calls) == 4 and len(numeric) == 2
    assert later.calls == later.auxiliary == []
    assert resumed.research.aux_calls == 1
    assert resumed.research.reflected_through == 0
    assert len(resumed.research.knowledge.comparisons) == 1


def test_auxiliary_tail_after_terminal_comparison_cannot_replay(
        task, tmp_path, monkeypatch, numeric, draws):
    import sciel.engine.repeat_population_support as route

    path = tmp_path / "paid-reflection-tail"
    engine, provider = construct(task, path, draws)
    save = route.save

    def interrupt_after_auxiliary(engine, phase):
        if phase == "population_after_reflection":
            raise KeyboardInterrupt("fixture crash before auxiliary checkpoint")
        return save(engine, phase)

    monkeypatch.setattr(route, "save", interrupt_after_auxiliary)
    with pytest.raises(KeyboardInterrupt):
        engine.run()
    assert_terminal_reflection(engine, provider, numeric)
    saved = json.loads((path / "checkpoint.json").read_text())
    assert saved["research"]["reflected_through"] == 0
    assert saved["research"]["aux_calls"] == 1
    assert len(saved["research"]["knowledge"]["paired_comparisons"]["records"]) == 1

    resumed, later = construct(task, path, draws, start=1)
    try:
        with pytest.raises(checkpoint.IdentityMismatch,
                           match="uncheckpointed search activity|uncheckpointed research"):
            checkpoint.load(resumed, str(path / "checkpoint.json"), allow_identity_mismatch=True)
        assert resumed.B_evals == 0 and resumed.db._all == {}
        assert later.calls == later.auxiliary == []
        assert len(draws.calls) == 9 and len(numeric) == 2
    finally:
        resumed.paired_batches.close()
        resumed.log.close()


def test_defer_flag_is_strict_and_still_prunes_without_consuming_watermark(
        task, tmp_path, draws):
    engine, provider = construct(task, tmp_path / "strict-phase", draws)
    research = engine.research
    # An already evicted conclusion must still be pruned during the ordinary
    # phase; bad API arguments must not mutate it first.
    research.conclusions["evicted"] = {"status": "skipped"}
    before = copy.deepcopy(research.state())
    try:
        for invalid in (0, 1, None, "false"):
            with pytest.raises(ValueError, match="reflect.*boolean"):
                research.complete_step(reflect=invalid)
            assert research.state() == before
        research.complete_step(reflect=False)
        assert research.conclusions == {}
        assert research.reflected_through == before["reflected_through"]
        assert research.aux_calls == 0
        assert provider.calls == provider.auxiliary == []
    finally:
        engine.paired_batches.close()
        engine.log.close()
