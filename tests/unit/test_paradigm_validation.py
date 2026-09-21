"""Frozen pre-trace bootstrap method versus the current method, all work stubbed."""

import json
from types import SimpleNamespace

import pytest

from sciel.campaign.budget import CampaignStopped
from sciel.engine.llm.client import LLMError, LLMResponse
from sciel.engine.loop import Engine, _paradigm_prompt
from sciel.engine.observed_changes import ObservedChanges
from sciel.engine.prompt.build import compose_directive
from sciel.engine.evaluation_timing import remember

# Baseline method copied from 6d939c972e124fd59990542a70f1d40a547da972, SHA-256 9dc36c931420f9b3d598e5dfa74afe1b0d853cb80be52e001b33b83b0e9819cc.
# Only its function name and relative imports were changed; no git dependency at test time.
def legacy_paradigm_child(self, base_code: str, paradigm: str, island: int,
                    origin: str) -> bool:
    """One LLM shot at a paradigm-specific program for `island`; adds it on a
    successful eval. Returns True on success. Costs one call + one eval that
    do NOT count toward B_evals (seeding, like the initial eval)."""
    from sciel.engine.prompt.parse import extract_card, extract_code
    from sciel.engine.worker import _graft_entrypoints
    try:
        resp = self.llm.generate(
            _paradigm_prompt(self.task_prompt, base_code, paradigm),
            self.llm.choose_model(self.rng.stream("prompt")), 0.9)
    except CampaignStopped:
        raise
    except Exception:
        return False
    self.cost_total += getattr(resp, "cost", 0.0)
    code = extract_code(resp.text)
    if not code.strip():
        return False
    code, _ = _graft_entrypoints(code, base_code, self.required_entrypoints)
    if self.novelty_gate is not None and not self._novelty_check(code, island)["accept"]:
        return False
    outcome = self._evaluate(code)
    if not outcome.result.ok:
        return False
    prog = self.db.add(code, outcome.result.score, outcome.result.cases,
                       island, origin=origin, card=extract_card(resp.text))
    self.eval_times[prog.id] = outcome.result.timing
    self.log.decision(policy="portfolio_seed", action=origin, island=island,
                      score=outcome.result.score, paradigm=paradigm[:40])
    return True


PARENT = "def solve(): return helper()\n"
RESOURCE = ("import multiprocessing as mp\ndef solve():\n"
            "    def worker(x): return x\n    return mp.Pool(2).map(worker, [1])\n")
CASES = ["success", "empty", "syntax", "resource", "duplicate", "graft",
         "novelty_reject", "novelty_accept", "eval_failed", "llm_error",
         "generation_error", "malformed", "eval_raises", "novelty_raises",
         "campaign_stop", "model_error"]


def exercise(method, case, origin):
    work, logged = [], []
    text = {"empty": "", "syntax": "def solve(:\n pass\n", "resource": RESOURCE,
            "duplicate": PARENT, "graft": "def helper(): return 2\n",
            "malformed": None}.get(case, "def solve(): return 3\n")

    def stream(name):
        work.append(("rng_stream", name))
        return "fixture_rng"

    def choose(rng):
        work.append(("choose_model", rng))
        if case == "model_error":
            raise ValueError("PRIVATE_PROVIDER_BODY")
        return "fixture_model"

    def generate(prompt, model, temperature):
        if method is Engine._paradigm_child:
            budget = '\n\n' + compose_directive(6.0, None, plateau=False)
            assert prompt.endswith(budget)
            # Budget delivery changed intentionally; compare the remaining work
            # to the frozen method, including its original task/paradigm prompt.
            prompt = prompt[:-len(budget)]
        work.append(("generate", prompt, model, temperature))
        if case in ("llm_error", "generation_error", "campaign_stop"):
            error = {"llm_error": LLMError, "generation_error": RuntimeError,
                     "campaign_stop": CampaignStopped}[case]
            raise error("PRIVATE_PROVIDER_BODY")
        return LLMResponse(text, model, .125, 10, 20)

    def novelty(code, island):
        work.append(("novelty", code, island))
        if case == "novelty_raises":
            raise ValueError("fixture novelty failure")
        return {"accept": case != "novelty_reject", "similarity": .9,
                "matched_index": 0, "descriptor_version": "fixture"}

    def evaluate(code):
        work.append(("evaluate", code))
        if case == "eval_raises":
            raise ValueError("fixture evaluator failure")
        result = SimpleNamespace(ok=case != "eval_failed", score=.4, timing=.03,
                                 cases={"fixture": 1}, artifacts={"error": "fixture error"})
        return SimpleNamespace(result=result)

    def add(code, score, cases, island, **kwargs):
        work.append(("add", code, score, cases, island, kwargs))
        return SimpleNamespace(id="fixture-program", code=code, cases=cases)

    engine = SimpleNamespace(
        llm=SimpleNamespace(choose_model=choose, generate=generate), task_prompt="fixture task",
        rng=SimpleNamespace(stream=stream), cost_total=1.25, required_entrypoints=["solve"],
        novelty_gate=object() if case.startswith("novelty_") else None,
        _novelty_check=novelty, _evaluate=evaluate, db=SimpleNamespace(add=add, _all={}), eval_times={},
        observed_changes=ObservedChanges(),
        _observe_program=lambda program, result, **kwargs: remember(engine, program, result),
        _bind_novelty_program=lambda *args, **kwargs: None,
        _per_run_budget=lambda: 6.0,
        log=SimpleNamespace(decision=lambda **fields: logged.append(fields)))
    try:
        outcome = ("return", method(engine, PARENT, "constructive", 1, origin))
    except Exception as exc:
        outcome = ("exception", type(exc), str(exc))
    return outcome, work, engine.cost_total, engine.eval_times, logged


@pytest.mark.parametrize("origin", ["portfolio", "spawn"])
@pytest.mark.parametrize("case", CASES)
def test_paradigm_profile_matches_head_work_and_adds_attributable_trace(case, origin):
    old = exercise(legacy_paradigm_child, case, origin)
    new = exercise(Engine._paradigm_child, case, origin)
    if case in ('llm_error', 'generation_error'):
        assert new[0] == ('exception', CampaignStopped, 'generation outcome and cost are unknown')
        assert new[1:4] == old[1:4]
        assert len(new[-1]) == 1 and new[-1][0]['cost'] is None
        assert new[-1][0]['action'] == 'generation_failed'
        assert 'PRIVATE_PROVIDER_BODY' not in json.dumps(new[-1])
        return
    if case == 'model_error':
        # Model selection failed locally, before dispatch: no unknown charge.
        assert new[0] == ('exception', ValueError, 'PRIVATE_PROVIDER_BODY')
        assert new[1:4] == old[1:4] and not new[-1]
        return
    assert new[:3] == old[:3]  # return/exception, ordered work, total cost
    # Timing scope intentionally changed: this fixture reports no trial count,
    # so the old stage duration is no longer presented as one candidate run.
    assert new[3] == {ident: None for ident in old[3]}
    outcome, work, cost, _, events = new
    if outcome[0] == "exception":
        assert not events  # No new swallowing, retry, or success after unexpected failure.
        return
    assert len(events) == 1
    event = events[0]
    assert "child_id" not in event
    assert event["origin"] == origin and event["island"] == 1
    trace = event["meta"]["validation_trace"]
    assert trace["profile"] == "paradigm" and len(trace["attempts"]) == 1
    attempt = trace["attempts"][0]
    assert "resample" not in attempt
    assert attempt["operator"] == origin and attempt["temperature"] == .9
    if case in ("llm_error", "generation_error", "model_error"):
        assert event["policy"] == "paradigm_validation"
        assert event["action"] == "generation_failed"
        assert not attempt["generation_returned"]
        assert event["cost"] is None and attempt["cost"] is None
        assert "candidate_code_sha256" not in attempt
        assert "PRIVATE_PROVIDER_BODY" not in json.dumps(event)
        assert cost == 1.25
        return
    assert attempt["generation_returned"] and attempt["cost"] == event["cost"] == .125
    assert attempt["completion_tokens"] == 20 and cost == 1.375
    assert all(d["stage"] not in ("precheck", "duplicate") for d in attempt["decisions"])
    if outcome[1]:
        assert event["policy"] == "portfolio_seed" and event["action"] == origin
        assert attempt["evaluation"] == {"ok": True, "disposition": "admit"}
        # Preserve every legacy success event field exactly.
        assert all(event[k] == v for k, v in old[-1][0].items())
    else:
        assert event["policy"] == "paradigm_validation"
        if case == "empty":
            assert attempt["decisions"][0]["violation"]["kind"] == "empty"
        elif case == "novelty_reject":
            assert attempt["decisions"][-1]["violation"]["kind"] == "code_novelty"
        elif case == "eval_failed":
            assert event["artifacts"]["error"] == "fixture error"
            assert attempt["evaluation"] == {"ok": False, "disposition": "eval_failed"}
    # Syntax, resource, and exact duplicate sources still reach the stub evaluator.
    if case in ("syntax", "resource", "duplicate"):
        assert any(e[0] == "evaluate" for e in work)
