"""Behavioral checks for research memory."""
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


def test_large_unicode_evidence_reflects_whole_records_within_context_limit(research_task, tmp_path):
    hypothesis_text, reasoning, rule = "Ж" * 2048, "界" * 4096, "Урок" * 512
    def respond(prompt):
        if "REFLECTION_JSON" in prompt:
            return successful_response(prompt)
        if "CONCLUSION_JSON" in prompt:
            record = prompt_json(prompt, "EVIDENCE_JSON:")
            return "CONCLUSION_JSON: " + json.dumps({"verdict": "CONFIRMED", "explanation": "Наблюдение",
                "mechanism_review": unknown_mechanism(record),
                "lessons": [{"rule": rule, "polarity": "positive", "scope": "branch",
                             "evidence_ids": [record["id"]]}]}, ensure_ascii=False)
        reply = successful_response(prompt)
        hyp = prompt_json(reply, "HYPOTHESIS_JSON:")
        hyp.update(hypothesis=hypothesis_text, reasoning=reasoning)
        return "HYPOTHESIS_JSON: " + json.dumps(hyp, ensure_ascii=False) + "\n```python" + reply.split("```python", 1)[1]
    model = RecordingLLM(respond)
    research = {**config()["research"], "reflection_interval": 4}
    instance = quick_engine(research_task, tmp_path / "unicode", model, budget=4, research=research)
    result = instance.run()
    assert result["B_evals"] == 4 and instance.research.reflections == 1
    reflection, = [c for c in model.calls if "REFLECTION_JSON" in c["prompt"]]
    evidence = prompt_json(reflection["prompt"], "EVIDENCE_JSON:")
    wire_text = reflection["prompt"].split("EVIDENCE_JSON:\n", 1)[1].strip()
    assert len(wire_text) <= 131072
    assert 0 < len(evidence["observation_catalog"]) < 4, "large records must be selected whole, not sliced"
    records = {r["id"]: r for r in instance.research.knowledge.experiments}
    for record in evidence["observation_catalog"].values():
        assert record == {k: v for k, v in records[record["id"]].items()
                          if k not in ("code_fragment", "code_fragment_sha256")}
        assert record["hypothesis"]["text"] == hypothesis_text
        assert record["hypothesis"]["reasoning"] == reasoning
    assert all(lesson["unverified_interpretation"] == rule for lesson in evidence["lessons"])
    assert instance.cost_total == pytest.approx(sum(c["cost"] for c in model.calls))


def test_controller_consolidates_old_lesson_with_supplied_evidence_and_bare_json(research_task, tmp_path):
    def respond(prompt):
        if "REFLECTION_JSON" not in prompt:
            return successful_response(prompt)
        data = prompt_json(prompt, "EVIDENCE_JSON:")
        old = data["lessons"][0]
        supplied = set(data["observation_catalog"])
        support = old['source_support']
        assert set(support["evidence_ids"]) <= supplied
        return json.dumps({"lessons": [{"rule": "Consolidated observation replacing " + old["id"],
            "polarity": support["polarity"], "scope": old["scope"], "evidence_ids": support["evidence_ids"]}],
            "retire_ids": [old["id"]]})
    model = RecordingLLM(respond)
    instance = quick_engine(research_task, tmp_path / "consolidate", model, budget=4)
    instance.run()
    reflections = research_events(tmp_path / "consolidate", "reflection")
    assert len(reflections) == 2 and all(e["applied"]["retired_ids"] and e["changed"] for e in reflections)
    assert instance.research.reflection_changes == 2
    context = research_events(tmp_path / "consolidate", "reflection_context")
    assert len(context) == 2 and all(e["experiment_ids"] and e["context_sha256"] for e in context)
    assert "r000001a1" in context[1]["experiment_ids"], "old support must survive the recent window"


def test_controller_reports_unsupported_claim_and_verdict_mismatch_without_rewriting_measurement(research_task, tmp_path):
    def respond(prompt):
        if "CONCLUSION_JSON" not in prompt:
            return successful_response(prompt)
        record = prompt_json(prompt, "EVIDENCE_JSON:")
        return json.dumps({"verdict": "REFUTED", "explanation": "Unsupported model opinion",
            "mechanism_review": unknown_mechanism(record),
            "lessons": [{"rule": "Claim a regression despite measured improvement",
                "polarity": "negative", "scope": "branch", "evidence_ids": [record["id"]]}]})
    model = RecordingLLM(respond)
    folder = tmp_path / "rejection"
    instance = quick_engine(research_task, folder, model, budget=1)
    instance.run()
    conclusion, = research_events(folder, "conclusion")
    assert conclusion["observed_forecast_verdict"] == "CONFIRMED"
    assert conclusion["verdict_matches_observation"] is False
    assert conclusion["lesson_error"] == "ValueError"
    assert "matching measured quality deltas" in conclusion["lesson_error_detail"]
    assert instance.research.knowledge.lessons == []
    result, = research_events(folder, "aux_result")
    assert result["response_truncated"] is False
    assert result["prompt_tokens"] > 0 and result["completion_tokens"] > 0


def test_controller_rejects_retirement_of_an_omitted_lesson_even_with_visible_evidence(research_task, tmp_path):
    before = []

    def respond(prompt):
        record = prompt_json(prompt, "EVIDENCE_JSON:") if "CONCLUSION_JSON" in prompt else None
        if record is not None:
            return json.dumps({"verdict": record["forecast_verdict"], "explanation": "Two observations",
                "mechanism_review": unknown_mechanism(record),
                "lessons": [{"rule": rule, "polarity": "positive", "scope": "branch",
                             "evidence_ids": [record["id"]]} for rule in ("Visible advice", "Omitted advice")]})
        if "REFLECTION_JSON" in prompt:
            data = prompt_json(prompt, "EVIDENCE_JSON:")
            assert len(data["lessons"]) == 1
            all_lessons = instance.research.knowledge.lessons
            hidden = next(l for l in all_lessons if l["id"] != data["lessons"][0]["id"])
            before.append(instance.research.knowledge.state())
            return json.dumps({"lessons": [{"rule": "Unseen replacement", "polarity": "positive",
                "scope": "branch", "evidence_ids": hidden["evidence_ids"]}], "retire_ids": [hidden["id"]]})
        return successful_response(prompt)

    folder = tmp_path / "hidden_retirement"
    instance = quick_engine(research_task, folder, RecordingLLM(respond), budget=1,
                            research={**config()["research"], "reflection_interval": 1})
    original = instance.research.knowledge.reflection_context
    instance.research.knowledge.reflection_context = lambda n, **kwargs: original(n, **{**kwargs, "max_lessons": 1})
    instance.run()
    event, = research_events(folder, "reflection")
    assert event["error"] == "ValueError" and "supplied" in event["error_detail"]
    assert event["applied"] is None and event["changed"] is False
    assert instance.research.knowledge.state() == before[0]
    assert instance.research.reflections == instance.research.reflection_changes == 0


def test_controller_rejects_retirement_marked_ineligible_by_source_count(research_task, tmp_path):
    def respond(prompt):
        data = prompt_json(prompt, "EVIDENCE_JSON:")
        old, = data["lessons"]
        assert old["retirement_eligible"] is False
        # This response fits the wire limit only by dropping required evidence.
        return json.dumps({"lessons": [{"rule": "Incomplete replacement", "polarity": "positive",
            "scope": "branch", "evidence_ids": old["source_support"]["evidence_ids"][:1]}], "retire_ids": [old["id"]]})

    folder = tmp_path / "oversized_retirement"
    instance = quick_engine(research_task, folder, RecordingLLM(respond))
    controller = instance.research
    for number in range(1, 34):
        ident = f"r{number:06d}a1"
        controller.knowledge.record({"id": ident, "branch": "p1", "parent_score": 0.,
            "score": 1., "actual_delta": 1., "status": "evaluated", "source_sha256": "0" * 64,
            "proposed_sha256": "0" * 64, "hypothesis": None})
        controller.knowledge.add_lessons([{"rule": "Repeated observation", "polarity": "positive",
            "scope": "branch", "evidence_ids": [ident]}], allowed_evidence_ids=[ident])
    controller.attempts = controller.outcomes = 33
    before = controller.knowledge.state()
    controller.complete_step()
    event, = research_events(folder, "reflection")
    assert event["error"] == "ValueError" and "evidence limit" in event["error_detail"]
    assert event["applied"] is None and event["changed"] is False
    assert controller.knowledge.state() == before
    assert controller.reflections == controller.reflection_changes == 0
