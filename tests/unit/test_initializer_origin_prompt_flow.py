"""Stored synthetic handoffs through real prompt paths; no helper/model API run."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from sciel.config import apply_defaults
from sciel.engine.research import ResearchController
from sciel.engine.types import Snapshot
from sciel.engine.llm.client import LLMResponse
from sciel.hypothesis import circle_initializer_witness as core
from sciel.hypothesis import initializer_evidence as evidence
from sciel.hypothesis.rendering import presentation_identity
from sciel.knowledge_base.catalogs import canonical_json
from sciel.knowledge_base.presentation import (
    INITIALIZER_CONTEXT_PREFIX, INITIALIZER_CONTEXT_RULES, INITIALIZER_CONTEXT_SCHEMA,
    ORIGIN_INITIALIZER_CONTEXT_SCHEMA, LEGACY_INITIALIZER_CONTEXT_SCHEMA,
    MAX_MEMORY_CHARS, validate_initializer_context)
from test_initializer_view_origins import synthetic_value


CLAIM = 'A different fixture initializer may improve the measured score.'


class Model:
    models = ['fixture']
    max_retries = 1

    def __init__(self):
        self.prompts = []

    def generate(self, prompt, model, temperature):
        self.prompts.append(prompt)
        if 'Review this one experiment.' in prompt:
            result = dict(verdict='REFUTED', explanation='The measured score decreased.', lessons=[],
                mechanism_review=[dict(claim_quote=CLAIM, source_quotes=[], assessment='unknown',
                    direct_effect='This inert response makes no source interpretation.',
                    untested_downstream='Synthetic outputs establish no solver gain.')])
        else:
            result = dict(lessons=[], retire_ids=[])
        return LLMResponse(json.dumps(result), model, .001, 10, 10)


@pytest.fixture(autouse=True)
def forbid_helper_execution(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail('prompt projection must never execute a helper')
    monkeypatch.setattr(core, '_launch', forbidden)
    monkeypatch.setattr(core, 'witness_sources', forbidden)


def stored_controller(mode):
    value = synthetic_value(mode=mode)
    cfg = apply_defaults({'research': {'enabled': True, 'model': 'fixture',
        'reflection_interval': 1, 'max_aux_calls': 2}})['research']
    cfg['initializer_witness'] = True
    model, events = Model(), []
    c = ResearchController(cfg, model, SimpleNamespace(research=lambda **e: events.append(e)), lambda _: None)
    candidate = value['sources'][value['roles']['candidate']]['source_sha256']
    parent = value['sources'][value['roles']['parent']]['source_sha256']
    c.knowledge.record(dict(id='r000001a1', branch='p1', parent_id='p1', status='evaluated',
        source_sha256=candidate, proposed_sha256=candidate, parent_source_sha256=parent,
        source_repaired=False, parent_score=1., score=.75, actual_delta=-.25,
        hypothesis=dict(text=CLAIM, reasoning='Unverified fixture interpretation.',
                        predicted_low=.1, predicted_high=.2, confidence=.5),
        metric=c.metric, parent_cases={}, evaluation=dict(ok=True, score=.75, cases={},
            reached='full', confirmed=False, raw_first_score=.75)))
    c.knowledge.register_initializer_evidence('r000001a1', value)
    # This is an explicitly synthetic, already consumed receipt. The test covers
    # projection/restore, not whether its fictional source really produced it.
    c._initializer_attempts['r000001a1'] = dict(status='completed', source_sha256=candidate,
        parent_source_sha256=parent, plan_sha256=value['plan_sha256'],
        evidence_sha256=value['evidence_sha256'], reason=None)
    c.attempts = c.outcomes = c.valid_hypotheses = 1
    c.queue = copy.deepcopy(c.knowledge.experiments)
    return c, model, events


def next_context(c):
    parent = SimpleNamespace(id='p1', parent_id=None, code='fixture_parent = 1\n',
                             card='fixture parent', score=1., cases={})
    return c.context(SimpleNamespace(_all={'p1': parent}), parent, 0, 2)


def begin(c, context):
    return c.begin(Snapshot(0, 'circle_generate', 'fixture', c.attempts,
        'p1', 'fixture_parent = 1\n', 1., research_context=context), 0, 'circle_generate',
        'Inert body instruction.', response_instruction='Return a typed body.',
        context_scope='circle_restart_body/1')


def aux_context(prompt):
    return json.JSONDecoder().raw_decode(prompt.split(INITIALIZER_CONTEXT_PREFIX, 1)[1])[0]


def generation_context(prompt):
    data = json.JSONDecoder().raw_decode(prompt.split('RESEARCH_CONTEXT_JSON:', 1)[1].lstrip())[0]
    return data.get('initializer_evidence')


def case(context, ident):
    view = context['cards'][0]['evidence']
    assert view['schema'] == evidence.VIEW_SCHEMA
    assert view['columns']['handoff'] == ['attempts', 'accepted', 'none', 'errors']
    return next(row for row in evidence.evidence_case_views(view) if row['case_id'] == ident)


@pytest.mark.parametrize('mode,origin,kind,counts', [
    ('fallback', 'saved_ordinary', 'none', [1, 0, 1, 0]),
    ('ordinary', 'return', 'value', [1, 1, 0, 0]),
    ('best', 'return', 'value', [1, 1, 0, 0]),
])
def test_origins_survive_actual_aux_and_next_generation_prompts(mode, origin, kind, counts):
    c, model, events = stored_controller(mode)
    ordinary, seen = c.knowledge.experiments, copy.deepcopy(c.knowledge._seen)
    c.complete_step()
    assert len(model.prompts) == 2
    context = next_context(c)
    ident, prompt = begin(c, context)
    shown = [aux_context(p) for p in model.prompts] + [generation_context(prompt)]
    for block in shown:
        assert block['schema'] == INITIALIZER_CONTEXT_SCHEMA
        row = case(block, 'w05')
        assert (row['origin'], row['return_kind'], row['handoff'], row['error']) == (origin, kind, counts, None)
        assert row['status'] == 'observed'
        # None fallback and an explicit ordinary-equal return have identical
        # geometry but different observed origins; unchanged best is accepted.
        assert row['facts']['changes'] == ([0, 0, 0, 78] if mode == 'best' else [52, 26, 26, 0])
        assert len(canonical_json(block)) == block['rendered_chars'] <= 16000
        assert len(canonical_json(block['cards'][0]['evidence'])) <= 8192
    assert all(INITIALIZER_CONTEXT_RULES in p for p in model.prompts + [prompt])
    assert c.knowledge.experiments == ordinary and c.knowledge._seen == seen
    assert c.knowledge.lessons == [] and c.outcomes == 1 and c.aux_calls == 2
    assert c.pending[ident]['injected_ids'] == []  # Witness is not a lesson exposure.
    presentations = [e for e in events if e['phase'] == 'initializer_witness_presentation']
    assert [e['role'] for e in presentations] == ['conclusion', 'reflection', 'generation']
    assert [e['prompt_sha256'] for e in presentations] == [hashlib.sha256(p.encode()).hexdigest()
                                                          for p in model.prompts + [prompt]]


def test_failed_and_missing_cases_remain_explicit_in_every_prompt():
    c, model, _ = stored_controller('mixed')
    ordinary = c.knowledge.experiments
    c.complete_step()
    _, prompt = begin(c, next_context(c))
    for block in [aux_context(p) for p in model.prompts] + [generation_context(prompt)]:
        assert case(block, 'w00')['origin'] == 'saved_ordinary'
        assert case(block, 'w01')['origin'] == 'return'
        for ident, kind in [('w03', None), ('w04', 'value')]:
            row = case(block, ident)
            assert row['status'] == 'failed' and row['facts'] is None and row['origin'] is None
            assert row['return_kind'] == kind and row['handoff'] == [1, 0, 0, 1] and row['error']
        for ident, status in [('w05', 'unknown'), ('w06', 'not_dispatched')]:
            row = case(block, ident)
            assert row['status'] == status and row['facts'] is None and row['origin'] is None
            assert row['return_kind'] is None and row['handoff'] is None and row['error']
    assert c.knowledge.experiments == ordinary and c.knowledge.lessons == []


def test_whole_card_budget_and_off_keep_stored_origins_without_exposure():
    c, _, _ = stored_controller('fallback')
    c.complete_step()
    saved = c.knowledge.state()
    full = c.knowledge.initializer_context(['r000001a1'])
    clipped = c.knowledge.initializer_context(['r000001a1'], max_chars=full['rendered_chars'] - 1)
    assert clipped['cards'] == [] and clipped['omitted_count'] == 1
    assert clipped['omitted'][0]['reason'] == 'presentation_budget'
    assert clipped['rendered_chars'] < full['rendered_chars'] <= MAX_MEMORY_CHARS
    assert c._initializer_context(['r000001a1'], 1) is None
    c.config['inject_lessons'] = False
    context = next_context(c)
    assert 'initializer_evidence' not in context
    ident, prompt = begin(c, context)
    assert generation_context(prompt) is None and c.pending[ident]['injected_ids'] == []
    assert c.knowledge.state() == saved


def reseal_context(value):
    value['context_sha256'] = '0' * 64
    while value['rendered_chars'] != len(canonical_json(value)):
        value['rendered_chars'] = len(canonical_json(value))
    value['context_sha256'] = core._hash({k: v for k, v in value.items() if k != 'context_sha256'})


def test_begin_rejects_rehashed_contradictory_handoff_before_attempt():
    c, _, _ = stored_controller('fallback')
    c.complete_step()
    context = next_context(c)
    block = context['initializer_evidence']
    view = block['cards'][0]['evidence']
    reference = view['candidate_cases'][view['case_ids'].index('w05')][2]
    view['handoffs'][reference]['return_kind'] = 'value'  # Still claims None fallback/counts.
    view['view_sha256'] = core._hash({k: v for k, v in view.items() if k != 'view_sha256'})
    reseal_context(block)
    before = c.state()
    with pytest.raises(ValueError):
        begin(c, context)
    assert c.state() == before


def test_legacy_context_remains_origin_free_and_old_presentation_cannot_resume():
    c, _, _ = stored_controller('fallback')
    c.complete_step()
    current = next_context(c)['initializer_evidence']
    origin = copy.deepcopy(current)
    origin['schema'] = ORIGIN_INITIALIZER_CONTEXT_SCHEMA
    view = origin['cards'][0]['evidence']
    view['candidate_cases'] = evidence.evidence_case_views(view)
    del view['handoffs']
    del view['columns']['case']
    view['schema'] = evidence.ORIGIN_VIEW_SCHEMA
    view['scope'] = evidence.ORIGIN_VIEW_SCOPE
    view['view_sha256'] = core._hash({k: v for k, v in view.items() if k != 'view_sha256'})
    reseal_context(origin)
    assert validate_initializer_context(origin) == origin
    old = copy.deepcopy(origin)
    old['schema'] = LEGACY_INITIALIZER_CONTEXT_SCHEMA
    view = old['cards'][0]['evidence']
    view['schema'] = evidence.LEGACY_VIEW_SCHEMA
    view['scope'] = evidence.SCOPE
    del view['columns']['handoff']
    for row in view['candidate_cases']:
        for key in ('origin', 'return_kind', 'handoff', 'error'):
            del row[key]
    view['view_sha256'] = core._hash({k: v for k, v in view.items() if k != 'view_sha256'})
    reseal_context(old)
    assert validate_initializer_context(old) == old
    assert all('origin' not in row for row in view['candidate_cases'])
    for outer in (old, origin, current):
        for inner in (old, origin, current):
            if outer['schema'] == inner['schema']:
                continue
            mixed = copy.deepcopy(outer)
            mixed['cards'][0]['evidence'] = copy.deepcopy(inner['cards'][0]['evidence'])
            reseal_context(mixed)
            with pytest.raises(ValueError):
                validate_initializer_context(mixed)
    saved = c.state()
    assert presentation_identity()['initializer_context'] == INITIALIZER_CONTEXT_SCHEMA
    for previous in (LEGACY_INITIALIZER_CONTEXT_SCHEMA, ORIGIN_INITIALIZER_CONTEXT_SCHEMA):
        changed = copy.deepcopy(saved)
        changed['presentation']['initializer_context'] = previous
        with pytest.raises(ValueError, match='presentation'):
            c.load_state(changed)
        assert c.state() == saved
