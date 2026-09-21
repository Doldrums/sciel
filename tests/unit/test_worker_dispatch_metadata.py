"""A rejected final response must not invent another operator dispatch."""
from types import SimpleNamespace

import pytest

from sciel.config import apply_defaults
from sciel.engine.db import Database, normalized_hash
from sciel.engine.llm.client import LLMResponse
from sciel.engine.research import ResearchController
from sciel.engine.types import NoChild, Result, Snapshot
from sciel.engine.worker import run_worker
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult


PARENT = "def solve():\n    return 1\n"
CHILD = "def solve():\n    return 2\n"


@pytest.mark.parametrize("kind", ["duplicate", "empty", "gate", "hypothesis"])
@pytest.mark.parametrize("attempts", [1, 2])
def test_exhausted_metadata_matches_last_dispatched_operator(kind, attempts):
    text = {"duplicate": f"```python\n{PARENT}```", "empty": "",
            "gate": "```python\ndef broken(:\n```",
            "hypothesis": f"```python\n{CHILD}```"}[kind]
    calls, evaluations, rows = [], [], []

    def generate(prompt, model, temperature):
        calls.append((prompt, temperature))
        return LLMResponse(text, model, .01, 1, 1)

    model = SimpleNamespace(generate=generate, models=["fake"], max_retries=1)
    db = Database(1)
    parent = db.add(PARENT, 1., {}, 0, "seed")
    snap = Snapshot(0, "diff", "fake", 0, parent.id, PARENT, 1.,
                    code_hashes={normalized_hash(PARENT)})
    research = None
    if kind == "hypothesis":
        config = apply_defaults({"research": {"enabled": True}})["research"]
        log = SimpleNamespace(research=lambda **row: rows.append(row))
        research = ResearchController(config, model, log, lambda cost: None)
        snap.research_context = research.context(db, parent, 0, 1)
    result = run_worker("diff", snap, model, evaluations.append,
                        dup_cap=attempts, research=research)
    assert isinstance(result, NoChild) and len(calls) == attempts and not evaluations
    assert result.cost == pytest.approx(.01 * attempts)
    trace = result.meta["validation_trace"]["attempts"]
    expected = ["diff"] if attempts == 1 else ["diff", "full_rewrite"]
    assert [a["operator"] for a in trace] == expected
    assert result.meta["op_used"] == expected[-1]
    assert result.meta["escalated"] is (attempts > 1)
    assert "resample" not in trace[-1]
    if attempts > 1:
        assert trace[0]["resample"]["operator"] == "full_rewrite"
    if research:
        assert [r["evidence"]["operator"] for r in rows if r["phase"] == "attempt"] == expected


def test_actual_retry_keeps_operator_cost_and_successful_result():
    replies = iter([PARENT, CHILD])
    calls, evaluations = [], []

    def generate(prompt, model, temperature):
        calls.append(prompt)
        return LLMResponse(f"```python\n{next(replies)}```", model, .01, 1, 1)

    def evaluate(code):
        evaluations.append(code)
        return CascadeOutcome(EvalResult(True, 2., {}), 2., "full")

    snap = Snapshot(0, "diff", "fake", 0, "p1", PARENT, 1.,
                    code_hashes={normalized_hash(PARENT)})
    result = run_worker("diff", snap, SimpleNamespace(generate=generate), evaluate, dup_cap=2)
    assert isinstance(result, Result) and result.ok
    assert len(calls) == 2 and evaluations == [CHILD] and result.cost == pytest.approx(.02)
    assert result.meta["op_used"] == "full_rewrite" and result.meta["escalated"] is True
