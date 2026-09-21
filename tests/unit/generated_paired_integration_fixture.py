"""Shared public-controller fixtures with authored inert paired observations."""
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace as NS

import pytest

from sciel.config import apply_defaults
from sciel.engine.circle_generation import CircleGeneration, CONTEXT_SCOPE
from sciel.engine.llm.client import LLMResponse
from sciel.engine.research import ResearchController
from sciel.engine.types import Program, Snapshot
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.hypothesis import circle_optimizer_traced_body as builder
from sciel.hypothesis.rendering import decode_evidence
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
from sciel.run_log import RunLog


@pytest.fixture(autouse=True)
def forbid_runtime_execution(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail('Integration fixtures must never launch a draw or runtime probe.')
    monkeypatch.setattr('sciel.eval.repeat_draw.draw', forbidden)
    monkeypatch.setattr('sciel.eval.repeat_draw.adapter_identity', forbidden)


@pytest.fixture(scope='module')
def pilot_artifacts():
    from generated_paired_pilot_fixture import authored_pilot
    return authored_pilot()


@pytest.fixture(scope='module')
def component(pilot_artifacts):
    from sciel.knowledge_base.generated_paired_history import identity, proposal_id
    p = pilot_artifacts
    manifest_sha = hashlib.sha256(p.generation['manifest.json']).hexdigest()
    proposals = []
    for arm in ('off', 'on'):
        response = p.gj(arm + '/response.json')
        provenance = dict(manifest_sha256=manifest_sha,
            run_id=p.gj('manifest.json')['run_id'], request_key=arm,
            research_id=p.bundle['arms'][arm]['research_id'])
        proposals.append(dict(id=proposal_id(provenance, response), provenance=provenance,
            request=p.gj(arm + '/provider_request.json'), response=response,
            provider_response=p.gj(arm + '/provider_return.json'),
            parent_source=p.bundle['parent']['code'],
            source=p.generation[arm + '/candidate.py'].decode(),
            proof=p.gj(arm + '/choice.json')['proof']))
    return dict(schema='sciel-generated-paired-history/1', profile='paired_restart_trace/2',
        metric=p.metric,
        root_parent=dict(id=p.bundle['parent']['id'],
                         source_sha256=hashlib.sha256(p.bundle['parent']['code'].encode()).hexdigest()),
        policy=identity(), proposals=proposals, comparisons=copy.deepcopy(list(p.groups.values())))


def pilot_for_task(task):
    """Reauthor the entire evidence chain for this copied task's exact identity."""
    from generated_paired_pilot_fixture import authored_pilot
    from sciel.hypothesis.measurement import metric_context
    pilot = authored_pilot(task_metric=metric_context(task))
    return pilot, component.__wrapped__(pilot)


class InertModel:
    models = ['anthropic/claude-sonnet-4.5']
    max_retries = 1

    def __init__(self, text=None):
        self.text, self.calls = text, []

    def generate(self, prompt, model, temperature=None, *, response_format=None):
        assert self.text is not None, 'Memory import or preparation called a model.'
        self.calls.append(dict(prompt=prompt, model=model, temperature=temperature,
                               response_format=copy.deepcopy(response_format)))
        return LLMResponse(self.text, model, .001, 10, 10, finish_reason='stop')


def construct(pilot, capsule, path, *, llm=None, **overrides):
    raw_config = dict(enabled=True, structured_auxiliary=True,
        restart_predictions=True, max_aux_calls=0, reflection_interval=0,
        model=InertModel.models[0],
        external_component_sha256=canonical_sha(capsule))
    raw_config.update(overrides)
    from test_restart_prediction_engine import config as engine_config
    config = engine_config()
    config['research'] = raw_config
    cfg = apply_defaults(config)['research']
    model = llm or InertModel()
    log = RunLog(str(path / 'events.jsonl'))
    log.run_start(scope='inert generated-paired component integration test')
    charges = []
    metric = pilot.metric
    c = ResearchController(cfg, model, log, charges.append, metric=metric)
    worker = CircleGeneration(c, execution_profile=builder.EXECUTION_PROFILE)
    parent = Program(**pilot.bundle['parent'])
    return NS(c=c, worker=worker, model=model, log=log, charges=charges,
              cfg=cfg, parent=parent, db=NS(_all={parent.id: parent}))


def import_component(f, capsule):
    return f.worker.import_external_component(capsule,
        expected_sha256=canonical_sha(capsule), parent=f.parent)


def rows(log):
    return [json.loads(line) for line in Path(log.path).read_text().splitlines()]


def snapshot(f, context=None):
    context = f.c.context(f.db, f.parent, 0, f.c.attempts + 1) if context is None else context
    return Snapshot(island=0, op='circle_generate', model=f.model.models[0],
        iteration_no=f.c.attempts, parent_id=f.parent.id, parent_code=f.parent.code,
        parent_score=f.parent.score, task_prompt='Pack 26 circles in the unit square.',
        research_context=context, kb_lessons=context['lesson_texts'])


def prepare(f, context=None):
    snap = snapshot(f, context)
    return f.c.begin(snap, 0, 'circle_generate', f.worker.prepare_prompt(snap),
                     context_scope=CONTEXT_SCOPE)


def decoded_context(prompt):
    value, _ = json.JSONDecoder().raw_decode(prompt.split('RESEARCH_CONTEXT_JSON:\n', 1)[1])
    return decode_evidence(value)


def decoded_component(prompt):
    from sciel.knowledge_base.generated_paired_context import PREFIX
    value, _ = json.JSONDecoder().raw_decode(prompt.split(PREFIX, 1)[1])
    return value


