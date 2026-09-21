"""Validation policy and diagnostics without executing generated programs."""

import hashlib
import json
from types import SimpleNamespace

import pytest

from sciel.engine.db import normalized_hash
from sciel.engine.llm.client import LLMError, LLMResponse
from sciel.engine.types import NoChild, Snapshot
from sciel.engine.validation import TRACE_SCHEMA, validate_candidate
from sciel.engine.viability import precheck
from sciel.engine.worker import run_worker
from sciel.campaign.budget import CampaignStopped
from sciel.run_log import RunLog


PARENT = "def solve(): return helper()\n"
HELPER = "def helper(): return 2\n"
RESOURCE = ("import multiprocessing as mp\n"
            "def solve():\n    def worker(x): return x\n"
            "    return mp.Pool(2).map(worker, [1])\n")


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def validate(code, **kwargs):
    return validate_candidate(code, parent_code=PARENT, required_entrypoints=["solve"],
                              code_hashes=kwargs.pop("code_hashes", set()), **kwargs)


def forbidden(*args, **kwargs):
    pytest.fail("an ineligible candidate reached a later stage")


@pytest.mark.parametrize("code,kind,detail", [
    ("def solve(:\n pass\n", "syntax", "invalid syntax (line 1)"),
    (RESOURCE, "resource", "multiprocessing cannot pickle a function worker defined locally; "
     "define the worker at module level or drop multiprocessing"),
])
def test_rejections_keep_exact_category_detail_and_source_hash(code, kind, detail):
    result = validate(code, novelty_check=forbidden)
    assert result.disposition == "resample"
    assert result.violation.kind == kind
    # Match the existing classifier verbatim, rather than flattening all gates.
    assert result.decisions[0]["violation"] == {
        "kind": kind, "detail": precheck(code).detail}
    assert result.violation.detail == detail
    assert result.decisions[0]["input_code_sha256"] == digest(code)
    assert result.decisions[0]["output_code_sha256"] == digest(code)
    assert result.decisions[0]["repair_applied"] is False
    assert len(result.decisions) == 1


def test_docstring_and_interface_repairs_have_connected_hashes():
    code = 'A proposed algorithm.\n"""\n' + HELPER
    result = validate(code)
    expected = ('"""' + code).rstrip() + "\n\n\n" + PARENT
    assert result.code == expected and result.disposition == "evaluate"
    pre, interface, duplicate = result.decisions
    assert pre["violation"]["kind"] == "syntax" and pre["repair_applied"]
    assert pre["repair"] == "prepend_docstring"
    assert pre["input_code_sha256"] == digest(code)
    assert pre["output_code_sha256"] == interface["input_code_sha256"] == digest('"""' + code)
    assert interface["grafted"] == result.grafted == ["solve"]
    assert interface["parent_code_sha256"] == digest(PARENT)
    assert interface["violation"]["kind"] == "interface"
    assert interface["output_code_sha256"] == duplicate["input_code_sha256"] == digest(expected)
    assert duplicate["normalized_code_hash"] == normalized_hash(expected)


def test_repair_policy_does_not_add_a_new_postrepair_resource_rejection():
    result = validate('A proposed algorithm.\n"""\n' + RESOURCE)
    assert result.disposition == "evaluate"
    assert precheck(result.code).kind == "resource"  # Fixture proves the distinction.
    assert [d["stage"] for d in result.decisions] == ["precheck", "interface", "duplicate"]


def test_duplicate_uses_grafted_source_and_bypasses_novelty():
    expected = HELPER.rstrip() + "\n\n\n" + PARENT
    result = validate(HELPER, code_hashes={normalized_hash(expected)}, novelty_check=forbidden)
    assert result.violation.kind == "duplicate" and result.code == expected
    assert result.grafted == ["solve"]
    assert result.decisions[-1]["action"] == "resample"
    assert result.decisions[-1]["input_code_sha256"] == digest(expected)


@pytest.mark.parametrize("code", ["x = 1\n", "solve = int\n", "def other(): return 3\n"])
def test_unchanged_graft_is_not_an_interface_rejection(code):
    # Missing parent definitions and no-def guards must still reach evaluation.
    result = validate_candidate(code, parent_code="x = 2\n", required_entrypoints=["solve"],
                                code_hashes=set())
    assert result.disposition == "evaluate" and result.code == code
    assert result.decisions[1]["action"] == "unchanged"
    assert result.decisions[1]["violation"] is None


class ScriptedLLM:
    def __init__(self, responses, events):
        self.responses = iter(responses)
        self.events = events

    def generate(self, prompt, model, temperature):
        self.events.append(("generate", prompt, temperature))
        text = next(self.responses)
        if isinstance(text, Exception):
            raise text
        return LLMResponse(text, model, .125, 10, 20)


def snapshot():
    return Snapshot(island=0, op="diff", model="mock", iteration_no=0,
                    parent_id="p1", parent_code=PARENT, parent_score=.2,
                    code_hashes={normalized_hash(PARENT)}, required_entrypoints=["solve"])


@pytest.mark.parametrize("ok", [True, False])
def test_mixed_attempts_preserve_order_calls_cost_and_final_source(monkeypatch, ok):
    monkeypatch.setattr("sciel.engine.worker.build", lambda op, snap: op)
    events = []
    novel = iter([False, True])

    def novelty(code):
        events.append(("novelty", code))
        return {"accept": next(novel), "similarity": .8, "matched_index": 0,
                "descriptor_version": "fixture"}

    def evaluate(code):
        events.append(("evaluate", code))
        result = SimpleNamespace(ok=ok, score=.3, cases={"fixture": 1}, timing=.01,
                                 artifacts={} if ok else {"error": "fixture eval failure"})
        return SimpleNamespace(result=result, raw_first_score=.25, reached="full", confirmed=True)

    final = 'A proposed algorithm.\n"""\n' + HELPER
    responses = ["", "def solve(:\n pass\n", PARENT, "def solve(): return 3\n", final]
    out = run_worker("diff", snapshot(), ScriptedLLM(responses, events), evaluate,
                     dup_cap=7, novelty_check=novelty)
    assert out.ok is ok  # A failed evaluator outcome must not cause another request.
    assert [x[0] for x in events] == ["generate"] * 4 + ["novelty", "generate", "novelty", "evaluate"]
    requests = [x for x in events if x[0] == "generate"]
    assert [x[1] for x in requests] == ["diff"] + ["full_rewrite"] * 4
    assert [x[2] for x in requests] == pytest.approx([.7, .7, .9, 1.1, 1.3])
    assert out.cost == .625 and out.meta["dup_attempts"] == 4
    assert out.meta["empty_responses"] == 1 and out.meta["gate_rejections"] == 2
    assert out.meta["grafted"] == ["solve"] and len(out.meta["novelty_checks"]) == 2
    assert out.prompt == "full_rewrite" and out.response == final
    assert out.child_code == ('"""' + final).rstrip() + "\n\n\n" + PARENT
    assert out.raw_first_score == .25 and out.confirmed and out.reached == "full"
    trace = out.meta["validation_trace"]
    assert trace["schema"] == TRACE_SCHEMA
    assert [a["attempt"] for a in trace["attempts"]] == [1, 2, 3, 4, 5]
    assert [a["disposition"] for a in trace["attempts"]] == ["resample"] * 4 + ["evaluate"]
    assert [a["decisions"][-1]["violation"]["kind"] for a in trace["attempts"][:4]] == [
        "empty", "syntax", "duplicate", "code_novelty"]
    assert trace["attempts"][-1]["candidate_code_sha256"] == digest(out.child_code)
    assert trace["attempts"][-1]["response_sha256"] == digest(final)
    assert trace["attempts"][-1]["completion_tokens"] == 20


def test_prior_rejection_trace_and_cost_survive_later_llm_error(monkeypatch):
    monkeypatch.setattr("sciel.engine.worker.build", lambda *args: "prompt")
    with pytest.raises(CampaignStopped) as stopped:
        run_worker("diff", snapshot(), ScriptedLLM([RESOURCE, LLMError("fixture error")], []),
                   forbidden, dup_cap=4)
    assert stopped.value.current_call_cost_unknown is True
    assert stopped.value.known_cost == .125
    attempts = stopped.value.validation_trace["attempts"]
    assert attempts[0]["decisions"][0]["violation"]["kind"] == "resource"
    assert attempts[0]["cost"] == .125
    assert attempts[1]["disposition"] == "llm_failed" and not attempts[1]["generation_returned"]
    assert "candidate_code_sha256" not in attempts[1] and attempts[1]["cost"] is None


def test_exhausted_trace_is_persisted_without_rejected_source(monkeypatch, tmp_path):
    monkeypatch.setattr("sciel.engine.worker.build", lambda *args: "private prompt body")
    malformed = "# PRIVATE_LONG_PAYLOAD\ndef solve(:\n pass\n"
    out = run_worker("diff", snapshot(), ScriptedLLM([malformed, RESOURCE], []),
                     forbidden, dup_cap=2)
    assert out.reason == "gate_rejected" and out.meta["gate_rejections"] == 2
    assert out.cost == .25
    path = tmp_path / "events.jsonl"
    log = RunLog(str(path))
    log.iteration(reason=out.reason, meta=out.meta, cost=out.cost)
    log.close()
    stored = path.read_text()
    event = json.loads(stored)
    assert event["meta"]["validation_trace"] == out.meta["validation_trace"]
    assert "PRIVATE_LONG_PAYLOAD" not in stored and "private prompt body" not in stored
    assert "finish_reason" not in stored and "truncated" not in stored
    attempts = event["meta"]["validation_trace"]["attempts"]
    assert [a["operator"] for a in attempts] == ["diff", "full_rewrite"]
    assert attempts[0]["resample"]["operator"] == "full_rewrite"
    # Exhaustion does not schedule or journal a nonexistent third request.
    assert "resample" not in attempts[-1]


@pytest.mark.parametrize("response", [None, 123, {}])
def test_malformed_response_is_not_coerced_into_an_eligible_candidate(monkeypatch, response):
    monkeypatch.setattr("sciel.engine.worker.build", lambda *args: "prompt")
    with pytest.raises(TypeError):
        run_worker("diff", snapshot(), ScriptedLLM([response], []), forbidden, dup_cap=2)
