"""Mock generation providers and local engine setup for Circle generation tests."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from sciel.config import apply_defaults
from sciel.campaign.budget import CampaignStopped
from sciel.engine import checkpoint
from sciel.engine.llm.client import LLMResponse
from sciel.engine.loop import Engine
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.hypothesis import circle_restart_body as body
from test_circle_intervention_engine import config as old_config, task_copy, rows, digest
from test_repeat_engine_integration import FakeDraws, config as repeat_config


COST = .003


BODIES = ['return best_params + uniform_noise * 0.25\n', 'return best_params + uniform_noise * 0.5\n']


def response(target=BODIES[0], **forecast):
    hypothesis = dict(hypothesis='Change the numeric restart initializer.',
        predicted_low=-.2, predicted_high=.2,
        reasoning='A different numeric starting point can change bounded local refinement; uncertain.',
        confidence=.5)
    hypothesis.update(forecast)
    return json.dumps(dict(hypothesis=hypothesis, body=target))


class Model:
    models = ['fixture']
    weights = [1.]
    max_retries = 1  # Research explicitly forbids hidden provider retries.

    def __init__(self, texts):
        self.texts = list(texts)
        self.calls = []

    def choose_model(self, rng):
        return 'fixture'

    def generate(self, prompt, model, temperature=None, *, response_format=None):
        assert response_format == body.response_format()
        self.calls.append(dict(prompt=prompt, model=model, temperature=temperature,
                               response_format=copy.deepcopy(response_format)))
        assert self.texts, 'unexpected extra model call'
        value = self.texts.pop(0)
        if isinstance(value, Exception):
            raise value
        if not isinstance(value, str):
            return value
        return LLMResponse(value, model, COST, 17, 11, finish_reason='stop')


def config(**changes):
    value = old_config()
    value.pop('circle_intervention')
    value.update(op_fixed='circle_generate', circle_generation={'enabled': True})
    value.update(changes)
    return value


def build(path, texts, *, cfg=None, task=None, scores=None):
    task = task or task_copy(path.parent / 'task', initial=body.source_for())
    model = Model(texts)
    engine = Engine(config() if cfg is None else cfg, task, str(path), model)
    evaluated = []
    scoremap = {body.DEFAULT_BODY: .5, BODIES[0]: .6, BODIES[1]: .7}
    scoremap.update(scores or {})

    def evaluate(source):
        parsed = body.body_of(source)  # Pure exact source lookup, never imports source.
        if source != body.source_for():
            events = rows(engine)
            binding = next(e for e in reversed(events) if e.get('phase') == 'circle_generation'
                           and e.get('action') == 'source_binding')
            hypothesis = next(e for e in reversed(events) if e.get('phase') == 'hypothesis')
            assert binding['experiment_id'] == hypothesis['experiment_id']
            assert binding['proof']['child_source_sha256'] == digest(source)
            assert hypothesis['evidence']['source_sha256'] == digest(source)
            assert events.index(binding) < events.index(hypothesis)
        evaluated.append(source)
        score = scoremap[parsed]
        cases = dict(sum_radii=score*2.635, target_ratio=score, validity=1., combined_score=score)
        return CascadeOutcome(EvalResult(True, score, cases, timing=.01), score, 'full', False)

    engine._evaluate = evaluate
    engine.fixture_evaluated = evaluated
    return engine


def read_checkpoint(path):
    return json.loads((path / 'checkpoint.json').read_text())
