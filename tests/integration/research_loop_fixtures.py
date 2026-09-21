"""Local task, mock model and evidence fixtures for research loop integration."""
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


pytestmark = pytest.mark.integration


LESSON = "Increase the returned test value in a small measured step."


@pytest.fixture
def research_task(tmp_path):
    root = tmp_path / "task"
    root.mkdir()
    (root / "task.yaml").write_text("suite: test\nname: research_fixture\norigin: math\n"
        "primary_metric: score\nstochastic: false\nsmoke_timeout: 3\nfull_timeout: 3\n")
    (root / "initial_program.py").write_text("def run_search():\n    return 0.1\n")
    (root / "evaluator.py").write_text(textwrap.dedent('''
        import hashlib
        import importlib.util
        import json
        from pathlib import Path

        def evaluate(program_path):
            path = Path(program_path)
            spec = importlib.util.spec_from_file_location("local_candidate", path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            assert hasattr(module, "run_search")
            journal = path.parent / "sciel_events.jsonl"
            events = [json.loads(line) for line in journal.read_text().splitlines()] if journal.exists() else []
            with (path.parent / "evaluation_observations.jsonl").open("a") as stream:
                stream.write(json.dumps({"source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "research_before_evaluation": [e for e in events if e.get("event") == "research"]}) + "\\n")
            value = float(module.run_search())
            return {"ok": True, "score": value, "cases": {"value": value}}

        smoke = evaluate
        full = evaluate
    '''))
    return load_task(str(root))


def config(**updates):
    base = {"task": "test/research_fixture", "budget": 3, "num_islands": 1,
        "random_seed": 41, "schedule_policy": "round_robin", "milestone_fracs": [],
        "max_iterations": 6, "dup_cap": 1, "op_fixed": "full_rewrite",
        "policy": {"enabled": False, "plateau_shift": False}, "llm": {"mock": True},
        "portfolio_seed": False, "idea_island": False, "dynamic_islands": False,
        "embed_grid": False, "embed_niche": False, "three_head": False,
        "crossover": False, "confirm_records": False, "checkpoint_on_milestone": True,
        "research": {"enabled": True, "reflection_interval": 2, "max_aux_calls": 8,
            "max_lessons": 50, "max_experiments": 200, "max_fragments": 20,
            "retrieve_k": 3, "model": None}}
    base.update(updates)
    return base


class RecordingLLM:
    """Prompt-aware deterministic fake with visible, nonzero request costs."""
    models = ["local-test-model"]
    weights = [1.0]

    def __init__(self, respond):
        self.respond = respond
        self.calls = []

    def choose_model(self, rng):
        return self.models[0]

    def generate(self, prompt, model, temperature=None):
        text = self.respond(prompt)
        cost = .03 if "REFLECTION_JSON" in prompt else .02 if "CONCLUSION_JSON" in prompt else .01
        self.calls.append({"prompt": prompt, "model": model, "temperature": temperature,
                           "text": text, "cost": cost})
        return LLMResponse(text=text, model=model, cost=cost,
                           prompt_tokens=len(prompt) // 4, completion_tokens=len(text) // 4)


def hypothesis_code(code, *, low=0, high=1):
    hypothesis = {"hypothesis": "A small increase improves the numerical output.",
                  "predicted_low": low, "predicted_high": high,
                  "reasoning": "The evaluator reads the returned test value.", "confidence": .8}
    return "HYPOTHESIS_JSON: " + json.dumps(hypothesis) + "\n```python\n" + code + "```\n"


def events(run_dir):
    return [json.loads(line) for line in (Path(run_dir) / "sciel_events.jsonl").read_text().splitlines()]


def engine(task, run_dir, model, **updates):
    return Engine(config(**updates), task, str(run_dir), model, python=sys.executable)


def fixed_evaluate(code):
    """Resume comparisons hold evaluator timing fixed; real subprocess has its own test."""
    match = re.search(r"return\s+([0-9.]+)", code)
    assert match, code
    value = float(match[1])
    return CascadeOutcome(result=EvalResult(ok=True, score=value, cases={"value": value}, timing=.01),
                          raw_first_score=value, reached="full", confirmed=False)


def prompt_json(prompt, marker):
    value = decode_evidence(json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1].lstrip())[0])
    return expand_reflection_payload(value) if value.get("schema") == "sciel-reflection-presentation/2" else value


def unknown_mechanism(record):
    """An inert current-contract reply for tests of measurements/admission."""
    hypothesis = record.get('hypothesis') or {}
    claim = hypothesis.get('text') or hypothesis.get('reasoning')
    return [{'claim_quote': claim[:160] if claim else None,
             'source_quotes': [], 'assessment': 'unknown',
             'direct_effect': 'This fixture does not assess the source mechanism.',
             'untested_downstream': 'No causal effect is established by this fixture.'}]


def successful_response(prompt):
    if "CONCLUSION_JSON" in prompt:
        record = prompt_json(prompt, "EVIDENCE_JSON:")
        lesson = {"rule": LESSON, "polarity": "positive", "scope": "branch",
                  "evidence_ids": [record["id"]]}
        return "CONCLUSION_JSON: " + json.dumps({
            "verdict": record["forecast_verdict"], "explanation": "Use the measured delta.",
            "mechanism_review": unknown_mechanism(record),
            "lessons": [lesson] if record["actual_delta"] is not None and record["actual_delta"] > 0 else []})
    if "REFLECTION_JSON" in prompt:
        return 'REFLECTION_JSON: {"lessons": [], "retire_ids": []}'
    context = prompt_json(prompt, "RESEARCH_CONTEXT_JSON:")
    # Depends on the prompt, never on process-local call counts: a fresh client
    # after checkpoint restore must generate the same responses as a continuous run.
    salt = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16) / 2**32 / 100
    value = context["parent_score"] + .1 + salt
    return hypothesis_code(f"def run_search():\n    return {value!r}\n")


def research_events(run_dir, phase):
    return [row for row in events(run_dir) if row["event"] == "research" and row["phase"] == phase]


def quick_engine(task, run_dir, model, **updates):
    instance = engine(task, run_dir, model, **updates)
    instance._evaluate = fixed_evaluate
    return instance
