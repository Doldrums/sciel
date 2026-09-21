"""Engineering feedback uses local stubs, never candidate execution or an API."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from sciel.hypothesis.rendering import decode_evidence

from sciel.campaign.budget import CampaignStopped
from sciel.engine.llm.client import LLMError, LLMResponse
from sciel.engine.research import ResearchController
from sciel.engine.types import Program, Snapshot
from sciel.engine.worker import _evaluation_failure, run_worker
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.knowledge_base import KnowledgeBase


PARENT = 'def run_search():\n    """Seed."""\n    return 0.5\n'
CANDIDATE = 'def run_search():\n    """Proposed change."""\n    return 0.6\n'
ERROR = "cannot unpack non-iterable NoneType object"


class FakeLLM:
    models = ["local-test"]

    def __init__(self):
        self.prompts = []

    def generate(self, prompt, model, temperature=None):
        self.prompts.append(prompt)
        body = {"hypothesis": "Change the initial placement.", "predicted_low": .01,
                "predicted_high": .03, "reasoning": "A different start may help.", "confidence": .6}
        return LLMResponse("HYPOTHESIS_JSON: " + json.dumps(body) +
                           "\n```python\n" + CANDIDATE + "```", model, .01, 20, 20)


def setup():
    llm, rows = FakeLLM(), []
    config = dict(model=None, max_lessons=10, max_experiments=20, max_fragments=5,
                  retrieve_k=3, max_aux_calls=0, reflection_interval=0)
    controller = ResearchController(config, llm, SimpleNamespace(research=lambda **r: rows.append(r)),
                                    lambda cost: None)
    parent = Program("p1", PARENT, .5, {}, 0, "seed")
    db = SimpleNamespace(_all={parent.id: parent})

    def snapshot(iteration=1):
        return Snapshot(0, "full_rewrite", llm.models[0], iteration, parent.id, PARENT, .5,
                        required_entrypoints=["run_search"],
                        research_context=controller.context(db, parent, 0, iteration))

    return controller, llm, rows, snapshot


def failed(artifacts):
    return CascadeOutcome(EvalResult(ok=False, score=0, artifacts=artifacts), 0, "smoke")


def test_actual_evaluator_error_reaches_next_request_with_identity_and_no_quality_lesson():
    controller, llm, rows, snapshot = setup()
    # This is the exact error shape from the six archived NoneType outcomes;
    # no original program is imported or executed.
    artifacts = {"error": ERROR, "traceback": "Traceback (most recent call last):\n"
                 '  File "/private/unneeded/path.py", line 44\nTypeError: ' + ERROR + "\n"}
    result = run_worker("full_rewrite", snapshot(), llm, lambda code: failed(artifacts),
                        dup_cap=1, research=controller)
    record = controller.knowledge.experiments[0]
    assert not result.ok and result.artifacts == artifacts
    assert record["failure"] == "TypeError: " + ERROR
    assert record["score"] is record["actual_delta"] is None
    assert record["forecast_verdict"] == "INCONCLUSIVE"
    assert record["source_sha256"] == hashlib.sha256(result.child_code.encode()).hexdigest()
    assert record["code_fragment"] == ""
    with pytest.raises(ValueError, match="measured"):
        controller.knowledge.add_lessons([{"rule": "Avoid this algorithm", "polarity": "negative",
            "scope": "branch", "evidence_ids": [record["id"]]}], allowed_evidence_ids=[record["id"]])
    assert controller.knowledge.lessons == []
    run_worker("full_rewrite", snapshot(2), llm, lambda code: failed(artifacts),
               dup_cap=1, research=controller)
    context = decode_evidence(json.JSONDecoder().raw_decode(llm.prompts[1].split("RESEARCH_CONTEXT_JSON:", 1)[1].lstrip())[0])
    feedback = context["failed_hypotheses"][0]
    assert ERROR in feedback and "TypeError" in feedback and record["id"] in feedback
    assert record["source_sha256"] in feedback and "quality not measured" in feedback
    assert "/private/unneeded/path.py" not in feedback and "Traceback" not in feedback
    restored = KnowledgeBase(max_lessons=10, max_experiments=20, max_fragments=5)
    restored.load_state(controller.knowledge.state())
    assert restored.failed_hypotheses("p1") == controller.knowledge.failed_hypotheses("p1")


@pytest.mark.parametrize("artifacts,expected", [
    ({"error_type": "TimeoutError", "error": "Timed out after 30s"}, "TimeoutError: Timed out after 30s"),
    ({"error": "Unknown solver differential_evolution"}, "EvaluationError: Unknown solver differential_evolution"),
    ({"error": "bad shape", "traceback": "ValueError: bad shape\n"}, "ValueError: bad shape"),
    ({"error_type": "x\nprivate text", "error": "bad shape"}, "EvaluationError: bad shape"),
    ({"error": {"secret": "not a text diagnostic"}}, "EvaluationError"),
    (None, "EvaluationError"),
])
def test_bounded_evaluation_diagnostic_categories(artifacts, expected):
    assert _evaluation_failure(artifacts) == expected


def test_diagnostic_limits_and_recognizable_credential_redaction():
    message = "OPENROUTER_API_KEY=sk-or-v1-secretvalue Authorization: Bearer othercredential " + "x" * 10000
    result = _evaluation_failure({"error_type": "ValueError", "error": message,
                                 "stderr": "private stderr", "response": "private remote body"})
    assert len(result) <= 666 and result.endswith("...")
    assert "secretvalue" not in result and "othercredential" not in result
    assert "private stderr" not in result and "private remote body" not in result
    controller, llm, rows, snapshot = setup()
    run_worker("full_rewrite", snapshot(), llm, lambda code: failed({"error": message}),
               dup_cap=1, research=controller)
    feedback = controller.knowledge.failed_hypotheses("p1")[0]
    assert len(feedback) < 1500 and "secretvalue" not in feedback
    assert controller.knowledge.failed_hypotheses("p1", limit=0) == []


@pytest.mark.parametrize("score,low,high,expected", [
    (1.05, .01, .03, "Improved despite forecast miss"),
    (.4, -.2, -.05, "measured regression"),
    (.4, .01, .03, "Forecast refuted; measured regression"),
    (.5, .01, .03, "No measured improvement; forecast miss"),
])
def test_forecast_miss_and_measured_regression_are_different(score, low, high, expected):
    digest = hashlib.sha256(CANDIDATE.encode()).hexdigest()
    kb = KnowledgeBase()
    kb.record({"id": "r000001a1", "branch": "p1", "parent_score": .5,
        "score": score, "actual_delta": score - .5, "status": "evaluated",
        "source_sha256": digest, "proposed_sha256": digest,
        "hypothesis": {"text": "Change the starting point", "predicted_low": low,
            "predicted_high": high, "reasoning": "A local forecast", "confidence": .6}})
    text = kb.failed_hypotheses("p1")[0]
    assert text.startswith(expected) and digest in text and "r000001a1" in text
    assert "quality not measured" not in text
    assert ("regression" in text) == (score < .5)


def test_provider_exception_body_is_not_added_to_research_failure():
    controller, llm, rows, snapshot = setup()

    def fail(*args, **kwargs):
        raise LLMError("private provider body secret=credential")

    llm.generate = fail
    with pytest.raises(CampaignStopped):
        run_worker("full_rewrite", snapshot(), llm, lambda code: pytest.fail("no evaluation"),
                   dup_cap=1, research=controller)
    assert "private provider body" not in json.dumps(rows)
    assert "credential" not in json.dumps(controller.knowledge.state())


def test_gate_diagnostic_keeps_syntax_detail_without_evaluation():
    controller, llm, rows, snapshot = setup()
    original = llm.generate

    def malformed(*args, **kwargs):
        result = copy.copy(original(*args, **kwargs))
        result.text = result.text.replace("return 0.6", "return (")
        return result

    llm.generate = malformed
    run_worker("full_rewrite", snapshot(), llm, lambda code: pytest.fail("no evaluation"),
               dup_cap=1, research=controller)
    record = controller.knowledge.experiments[0]
    assert record["status"] == "no_child"
    assert "syntax" in record["failure"] and "line" in record["failure"]
    assert record["score"] is record["actual_delta"] is None
