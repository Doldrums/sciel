"""Exact measurements survive bounded memory, consolidation and safe resume.

All code strings are inert fixtures. No model, candidate or evaluator is run.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sciel.hypothesis.rendering import decode_evidence
from sciel.engine.types import Snapshot
from sciel.engine.prompt.build import build
from sciel.knowledge_base.presentation import expand_reflection_payload

from sciel.config import apply_defaults
from sciel.engine.research import ResearchController, digest
from sciel.engine.types import Program
from sciel.eval.registry import load_task
from sciel.hypothesis.measurement import metric_context
from sciel.knowledge_base import KnowledgeBase
from sciel.knowledge_base.presentation import observation_card, reflection_payload


def begin_fixture(controller, context):
    snap = Snapshot(context['island'], 'diff', 'fake', context['iteration'] - 1,
        context['parent_id'], context['parent_code'], context['parent_score'],
        research_context=context, kb_lessons=context.get('lesson_texts', []))
    return controller.begin(snap, 0, 'diff', build('diff', snap))


ROOT = Path(__file__).resolve().parents[2]
PARENT_CASES = {'sum_radii': .9597642169962064, 'validity': 1., 'target_ratio': .36423689449571406}


def controller(*, metric=None, **settings):
    config = apply_defaults({'research': {'enabled': True, 'max_experiments': 2,
        'max_aux_calls': 0, 'reflection_interval': 0, **settings}})['research']
    events = []
    c = ResearchController(config, SimpleNamespace(models=['fake'], max_retries=1),
        SimpleNamespace(research=lambda **kw: events.append(copy.deepcopy(kw))),
        lambda _: pytest.fail('no model calls'), metric=metric)
    parent = Program('p1', 'def run_search():\n    return .5\n', .5,
                     copy.deepcopy(PARENT_CASES), 0, 'seed')
    return c, parent, SimpleNamespace(_all={'p1': parent}), events


def circle_metric():
    return metric_context(load_task(str(ROOT / 'tasks/oe/circle_packing')))


def observe(c, parent, db, score, *, status='evaluated', evaluation=True):
    ctx = c.context(db, parent, 0, c.attempts + 1)
    ident, prompt = begin_fixture(c, ctx)
    code = f'def run_search():\n    return {c.attempts}\n'
    c.pending[ident].update(proposed_sha256=digest(code), source_sha256=digest(code))
    observation = {'ok': status == 'evaluated', 'score': score,
        'cases': {'sum_radii': 2.4425261335140847, 'validity': 1., 'trial_count': 3},
        'reached': 'full', 'confirmed': False, 'raw_first_score': score} if evaluation else None
    result = c.resolve(ident, status=status, score=score if status == 'evaluated' else None,
                       source=code, evaluation=observation)
    return result, prompt


def lesson(c, evidence, rule):
    ids = [evidence['id']]
    return c.knowledge.add_lessons([{'rule': rule, 'polarity': 'positive',
        'scope': 'branch', 'evidence_ids': ids}], allowed_evidence_ids=ids)[0]


def test_measurements_survive_pruning_consolidation_fragments_and_resume():
    metric = circle_metric()
    c, parent, db, events = controller(metric=metric)
    first, prompt = observe(c, parent, db, .9269548893791594)
    a, b = lesson(c, first, 'First interpretation'), lesson(c, first, 'Second interpretation')
    failed, _ = observe(c, parent, db, 0., status='eval_failed')
    replacement, = c.knowledge.reflect([{'rule': 'Combined local interpretation',
        'polarity': 'positive', 'scope': 'branch', 'evidence_ids': [first['id']]}],
        [a, b], allowed_evidence_ids=[first['id']])['lesson_ids']
    observe(c, parent, db, .6)
    assert first['id'] not in {r['id'] for r in c.knowledge.experiments}
    card = observation_card(c.knowledge.lessons[0])
    assert card['lesson_id'] == replacement
    source, = card['source_observations']
    assert source['metric'] == metric and source['parent_cases'] == PARENT_CASES
    assert source['evaluation'] == first['evaluation']
    exposed = next(x for x in card['advice_exposures']['representatives']
                   if x['observation']['id'] == failed['id'])
    assert exposed['observation']['evaluation']['score'] == 0
    assert exposed['observation']['score'] is exposed['observation']['actual_delta'] is None
    assert {x['from_lesson_id'] for x in exposed['attribution']} == {a, b}
    fragment = next(x for x in c.knowledge.fragments if x['experiment_id'] == first['id'])
    assert fragment['evidence']['evaluation'] == first['evaluation']
    records, lessons = c.knowledge.reflection_context(1)
    assert expand_reflection_payload(reflection_payload(records, lessons))['lessons'][0]['source_support']['observations'][0] == source
    state = json.loads(json.dumps(c.state(), allow_nan=False))
    restored, parent2, db2, _ = controller(metric=metric)
    restored.load_state(state)
    assert restored.state() == state
    assert restored.context(db2, parent2, 0, 4) == c.context(db, parent, 0, 4)
    c.knowledge.fragments[0]['evidence']['evaluation']['cases'].clear()
    assert c.state() == state  # Returned projections are independent copies.


@pytest.mark.parametrize('field', ['metric', 'parent_cases', 'evaluation'])
def test_measurement_tampering_and_partial_triplets_cannot_load(field):
    c, parent, db, _ = controller(metric=circle_metric())
    observe(c, parent, db, .6)
    state = c.state()
    broken = copy.deepcopy(state)
    record = broken['knowledge']['experiments'][0]
    if field == 'metric':
        record[field] = metric_context()
    elif field == 'parent_cases':
        record[field]['sum_radii'] += .1
    else:
        record[field]['cases']['sum_radii'] += .1
    target, _, _, _ = controller(metric=circle_metric())
    before = target.state()
    with pytest.raises(ValueError, match='identity mismatch'):
        target.load_state(broken)
    assert target.state() == before
    del state['knowledge']['experiments'][0][field]
    with pytest.raises(ValueError, match='incomplete experiment measurement'):
        target.load_state(state)
    assert target.state() == before


def test_foreign_metric_evidence_cannot_be_resumed_even_with_valid_kb_digests():
    source, parent, db, _ = controller(metric=circle_metric())
    observe(source, parent, db, .6)
    target, _, _, _ = controller()
    state = source.state()
    state['metric'] = target.metric  # Valid descriptor, but retained evidence belongs elsewhere.
    before = target.state()
    with pytest.raises(ValueError, match='different metric'):
        target.load_state(state)
    assert target.state() == before


def test_metric_mismatch_before_attempt_or_checkpoint_mutation():
    c, parent, db, _ = controller(metric=circle_metric())
    before = c.state()
    ctx = c.context(db, parent, 0, 1)
    ctx['metric'] = metric_context()
    with pytest.raises(ValueError, match='differs from controller'):
        begin_fixture(c, ctx)
    assert c.state() == before
    saved = c.state()
    saved['metric'] = metric_context()
    with pytest.raises(ValueError, match='checkpoint metric mismatch'):
        c.load_state(saved)
    assert c.state() == before


@pytest.mark.parametrize('status,ok,score', [('evaluated', False, .6), ('evaluated', True, .9),
    ('eval_failed', True, .6), ('no_child', False, 0.)])
def test_returned_result_cannot_be_assigned_to_a_different_outcome(status, ok, score):
    c, parent, db, _ = controller()
    ident, _ = begin_fixture(c, c.context(db, parent, 0, 1))
    observation = {'ok': ok, 'score': score, 'cases': {}, 'reached': 'full',
                   'confirmed': False, 'raw_first_score': score}
    before = c.state()
    with pytest.raises(ValueError, match='returned evaluation'):
        c.resolve(ident, status=status, score=.6, evaluation=observation)
    assert c.state() == before


def test_legacy_kb_roundtrips_without_fabricating_measurement_fields():
    from research_memory_fixtures import positive_exposure_history
    state = positive_exposure_history().state()
    data = json.dumps(state, sort_keys=True)
    kb = KnowledgeBase(**state['limits'])
    kb.load_state(json.loads(data))
    assert kb.state() == state and json.dumps(state, sort_keys=True) == data
    for record in kb._evidence().values():
        assert not {'metric', 'parent_cases', 'evaluation'} & set(record)
    card = observation_card(kb.lessons[0])
    assert all(x['metric'] is x['parent_cases'] is x['evaluation'] is None
               for x in card['source_observations'])


def test_reflection_event_hash_matches_the_exact_payload_sent_to_model():
    from sciel.engine.llm.client import LLMResponse
    c, parent, db, events = controller(max_aux_calls=1, reflection_interval=1)
    observe(c, parent, db, .6)
    prompts = []
    def generate(prompt, model, temperature):
        prompts.append(prompt)
        return LLMResponse('REFLECTION_JSON: {"lessons": [], "retire_ids": []}', model, 0., 1, 1)
    c.llm.generate = generate
    c.charge = lambda _: None
    c.complete_step()
    payload = json.loads(prompts[0].split('EVIDENCE_JSON:\n', 1)[1])
    event, = [e for e in events if e['phase'] == 'reflection_context']
    from sciel.knowledge_base.catalogs import canonical_json
    assert event['context_sha256'] == digest(canonical_json(payload))
    payload = decode_evidence(payload)
    assert payload['observation_catalog']['r000001a1']['metric'] == c.metric
