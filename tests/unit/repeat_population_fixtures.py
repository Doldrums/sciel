"""Inert engine fixtures for repeat population admission and recovery."""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import pytest
from sciel.config import apply_defaults
from sciel.engine import checkpoint
from sciel.engine.llm.client import LLMResponse
from sciel.engine.loop import Engine
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.eval.registry import load_task


ROOT = Path(__file__).resolve().parents[2]


def sha(code):
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def step_of(code):
    match = re.search(r"fixture_step = (\d+)", code)
    return int(match.group(1)) if match else 0


def code_for(step):
    return (f"# fixture_step = {step}\n"
            f"def construct_packing():\n    return {step}\n"
            "def run_packing():\n    return construct_packing()\n")


def settings(**changes):
    cfg = dict(task="oe/circle_packing", budget=5, max_iterations=5,
        dup_cap=1, random_seed=29, num_islands=2, pop_cap=2, inflight_cap=1,
        schedule_policy="round_robin", milestone_fracs=[], op_fixed="full_rewrite",
        container={"kind": "pool", "selection": "shared"}, archive_size=0,
        policy={"enabled": False, "plateau_shift": False}, llm={"mock": True},
        portfolio_seed=False, idea_island=False, dynamic_islands=False,
        crossover=False, embed_grid=False, embed_niche=False, three_head=False,
        confirm_records=False, research={"enabled": True, "max_aux_calls": 0,
            "reflection_interval": 0, "retrieve_k": 2},
        repeat_population={"enabled": True, "draws_per_source": 3,
            "seed_master": 157, "max_comparisons": 5, "max_draw_attempts": 33,
            "max_wall_seconds": 2000., "timeout_seconds": 30.,
            "cpu_seconds": 30, "mem_bytes": 2 * 1024**3,
            "certification_seconds": 5.})
    cfg.update(copy.deepcopy(changes))
    return cfg


class Provider:
    models = ["fixture"]
    weights = [1.]
    max_retries = 1

    def __init__(self, start=0):
        self.step = start
        self.calls = []
        self.before_call = None

    def choose_model(self, rng):
        return "fixture"

    def generate(self, prompt, model, temperature=None):
        assert "Review this one experiment." not in prompt
        assert "Review existing experimental evidence and lessons." not in prompt
        self.step += 1
        if self.before_call is not None:
            self.before_call(prompt)
        response = "HYPOTHESIS_JSON: " + json.dumps(dict(
            hypothesis="Change the inert fixture marker.", predicted_low=-1.,
            predicted_high=1., reasoning="Only a fake orchestration fixture.", confidence=.5))
        response += "\n```python\n" + code_for(self.step) + "```"
        self.calls.append({"prompt": prompt, "text": response, "model": model,
                           "temperature": temperature})
        return LLMResponse(response, model, .001, 10, 10)


class Draws:
    identity = {"schema": "fake-repeat-adapter/1", "runtime": "inert-fixture"}

    def __init__(self, monkeypatch):
        self.calls = []
        self.failure_at = None
        self.failure_status = "candidate_failure"
        self.before_draw = None
        self.values = lambda step, batch: [.4, .3, .6, .7, .8, .75][step]
        self.arrays = {}
        monkeypatch.setattr("sciel.eval.repeat_draw.adapter_identity", self.adapter_identity)
        monkeypatch.setattr("sciel.eval.repeat_draw.draw", self.draw)

    def adapter_identity(self, *, python=None, runtime=None):
        assert runtime == {"backend": "native"}
        return copy.deepcopy(self.identity)

    def draw(self, source, *, seed, timeout, cpu_seconds, mem_bytes,
             certification_seconds, python=None, runtime=None, expected_identity=None):
        from sciel.eval.packing_geometry import validate_packing, certify_packing

        assert runtime == {"backend": "native"} and expected_identity == self.identity
        batch = 0 if len(self.calls) < 3 else 1 + (len(self.calls) - 3) // 6
        call = {"source": source, "source_sha256": sha(source), "seed": seed,
                "step": step_of(source), "batch": batch}
        self.calls.append(call)
        if self.before_draw is not None:
            self.before_draw(call)
        total = self.values(call["step"], batch)
        if total not in self.arrays:
            array = {"centers": [[(i % 6 + .5) / 6, (i // 6 + .5) / 5]
                                 for i in range(26)], "radii": [total / 26] * 26}
            self.arrays[total] = validate_packing(array), certify_packing(array)
        capture, certificate = copy.deepcopy(self.arrays[total])
        assert capture["valid"] and certificate["status"] == "certified"
        status = self.failure_status if len(self.calls) == self.failure_at else "valid"
        valid = status == "valid"
        return dict(schema="sciel-repeat-draw/1", source_sha256=sha(source), seed=seed,
            adapter_identity=copy.deepcopy(self.identity), status=status,
            seed_applied=dict(python_random=seed, numpy_legacy=seed,
                python_hash_seed=str(seed), before_candidate_import=True, determinism_proven=False),
            limits=dict(candidate_wall_seconds=timeout,
                certification_wall_seconds=certification_seconds,
                candidate_cpu_seconds=cpu_seconds, candidate_mem_bytes=mem_bytes,
                certification_cpu_seconds=cpu_seconds, certification_mem_bytes=mem_bytes),
            raw_sum=capture["sum_radii"] if valid else None,
            certified_sum=certificate["sum_exact"] if valid else None,
            capture=capture if valid else None, certificate=certificate if valid else None,
            reason=None if valid else "inert injected failure",
            started_at="2026-09-16T00:00:00+00:00",
            completed_at="2026-09-16T00:00:00.010000+00:00",
            elapsed_seconds=.01, candidate_elapsed_seconds=.006,
            certification_elapsed_seconds=.003, identity_elapsed_seconds=.001)


@pytest.fixture
def task():
    return load_task(str(ROOT / "tasks/oe/circle_packing"))


@pytest.fixture
def numeric(monkeypatch):
    calls = []
    # The first child has the best ordinary observation but loses repeats. Later
    # children have lower ordinary observations but strictly dominate repeats.
    raw_values = [1., 1.8, .1, .2, .15, .3]

    def evaluate(task, path, *, record_candidate, python):
        source = Path(path).read_text()
        raw = raw_values[step_of(source)]
        score = raw / 2.635
        cases = dict(sum_radii=raw, validity=1., combined_score=score)
        calls.append({"source": source, "score": score, "cases": copy.deepcopy(cases)})
        return CascadeOutcome(EvalResult(ok=True, score=score, cases=cases,
            artifacts={"fixture_only": True}, timing=.01), score, "full", False)

    monkeypatch.setattr("sciel.engine.loop.cascade_evaluate", evaluate)
    return calls


@pytest.fixture
def draws(monkeypatch):
    return Draws(monkeypatch)


def construct(task, path, *, cfg=None, start=0):
    provider = Provider(start)
    engine = Engine(cfg or settings(), task, str(path), provider, python=sys.executable)
    return engine, provider


def projection(engine):
    return engine.repeat_population.state()["projection"]


def normalized_calls(calls, groups, experiments):
    """Verify complete cards/facts before replacing independent-run identities."""
    from sciel.hypothesis.rendering import decode_evidence
    from sciel.knowledge_base.comparison_facts import validate_comparison_facts
    from sciel.knowledge_base.presentation import comparison_card, omit_source_context
    from test_repeat_engine_integration import _split_observed_feedback

    expected = {g["id"]: comparison_card(g) for g in groups}
    batch_ids = {g["id"]: g["batch_id"] for g in groups}
    evidence = {r['id']: r for r in [*experiments, *groups]}
    result = copy.deepcopy(calls)
    for call in result:
        call['prompt'], _ = _split_observed_feedback(call['prompt'])
        prefix, suffix = call["prompt"].split("RESEARCH_CONTEXT_JSON:\n", 1)
        raw, end = json.JSONDecoder().raw_decode(suffix)
        context = decode_evidence(raw)
        block = context.get("paired_comparisons")
        if block is not None:
            selected, seen = [], set()
            for card in block['cards']:
                group = evidence[card['id']]
                for ident in (group['id'], group['origin_experiment_id']):
                    if ident in evidence and ident not in seen:
                        selected.append(evidence[ident])
                        seen.add(ident)
            if 'comparison_facts' in context:
                facts = context['comparison_facts']
                validate_comparison_facts(facts, records=selected, max_chars=facts['max_chars'])
                facts['paired'] = {batch_ids[ident]: value for ident, value in facts['paired'].items()}
                for field in ('presented_ids', 'omitted_ids'):
                    facts[field] = [batch_ids.get(ident, ident) for ident in facts[field]]
                # Preserve original rendered_chars: the independent runs must
                # still have used the same actual presentation budget.
            for card in block["cards"]:
                full = expected[card["id"]]
                assert card == full or card == omit_source_context(full)
                card["id"] = batch_ids[card["id"]]
                for field in ("namespace", "evidence_sha256", "view_sha256"):
                    card.pop(field)
            for field in ("presented_ids", "omitted_ids"):
                block[field] = [batch_ids[ident] for ident in block[field]]
        call["prompt"] = prefix + "RESEARCH_CONTEXT_JSON:\n" + json.dumps(context, sort_keys=True) + suffix[end:]
    return result


def interrupted_bootstrap(task, path, monkeypatch, numeric, draws):
    """Crash after durable plan/checkpoint but before the first draw intent."""
    engine, provider = construct(task, path, cfg=settings(budget=1, max_iterations=1))

    def before_dispatch():
        assert engine.paired_batches.pending
        raise KeyboardInterrupt("fixture crash before bootstrap dispatch")

    monkeypatch.setattr(engine.paired_batches, "run_pending", before_dispatch)
    with pytest.raises(KeyboardInterrupt):
        engine.run()
    assert not provider.calls and not draws.calls and len(numeric) == 1
    saved_path = path / "checkpoint.json"
    saved = json.loads(saved_path.read_text())
    assert saved["B_evals"] == saved["iteration_no"] == 0
    assert len(saved["db"]["programs"]) == 2
    assert saved["repeat_population"]["projection"]["history"] == {}
    batch = saved["paired_batches"]["data"]["batches"][0]
    assert batch["plan"]["kind"] == "bootstrap" and batch["decision"] is None
    assert batch["intents"] == [] or batch["intents"] == {}
    engine.log.close()
    return saved_path, saved
