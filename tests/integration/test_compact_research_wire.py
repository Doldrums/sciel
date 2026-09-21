"""Exercise actual research prompt boundaries without any API or code execution."""
import copy
import json
from types import SimpleNamespace

import pytest

from sciel.config import apply_defaults
from sciel.engine.llm.client import LLMResponse
from sciel.engine.prompt.build import build
from sciel.engine.research import ResearchController, digest
from sciel.engine.types import Program, Snapshot
from sciel.hypothesis import CONTRACT
from sciel.hypothesis import conclusion_source
from sciel.hypothesis.rendering import (REFERENCE_RULES, decode_evidence,
    presentation_identity, serialize_context, serialize_record, serialize_reflection)
from sciel.knowledge_base.catalogs import SCHEMA, canonical_json
from sciel.knowledge_base.presentation import (expand_reflection_payload,
    observation_card, reflection_inline_payload)


CASES = {**{f'measured_case_{i:03d}': i / 13 for i in range(100)},
         'signed_zero': -0.0, 'trial_count': 3}
CODE = 'def solve():\n    return 1\n'


def wire(prompt, marker):
    return json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1].lstrip())[0]


@pytest.fixture
def fixture():
    events, prompts = [], []
    def generate(prompt, model, temperature):
        prompts.append(prompt)
        if 'Review this one experiment.' in prompt:
            record = decode_evidence(wire(prompt, 'EVIDENCE_JSON:'))
            text = 'CONCLUSION_JSON: ' + json.dumps({'verdict': record['forecast_verdict'],
                'explanation': 'Synthetic observation, no causal claim.', 'lessons': []})
        else:
            text = 'REFLECTION_JSON: {"lessons":[],"retire_ids":[]}'
        return LLMResponse(text, model, 0., 0, 0)
    model = SimpleNamespace(models=['fake'], max_retries=1, generate=generate)
    cfg = apply_defaults({'research': {'enabled': True, 'reflection_interval': 2,
                                      'max_aux_calls': 8}})['research']
    controller = ResearchController(cfg, model,
        SimpleNamespace(research=lambda **kw: events.append(copy.deepcopy(kw))), lambda _: None)
    parent = Program('p1', CODE, .5, copy.deepcopy(CASES), 0, 'seed')
    return controller, parent, SimpleNamespace(_all={'p1': parent}), events, prompts


def snapshot(c, parent, db, *, repair=False):
    ctx = c.context(db, parent, 0, c.attempts + 1)
    target = 'def solve():\n    raise RuntimeError("failed child")\n' if repair else parent.code
    return Snapshot(0, 'error_repair' if repair else 'diff', 'fake', c.attempts,
        parent.id, target, 0. if repair else parent.score, research_context=ctx,
        kb_lessons=ctx['lesson_texts'], inspirations=[{'code': f['code'],
            'score': f['score'], 'card': 'same source, inspiration role'} for f in ctx['fragments']])


def observe(c, parent, db, score):
    snap = snapshot(c, parent, db)
    ident, prompt = c.begin(snap, 0, snap.op, build(snap.op, snap))
    code = f'def solve():\n    return {c.attempts + 1}\n'
    hypothesis = {'hypothesis': 'Synthetic constant change.', 'reasoning': 'Fixture only.',
                  'predicted_low': -1., 'predicted_high': 1., 'confidence': .5}
    response = LLMResponse('HYPOTHESIS_JSON: ' + json.dumps(hypothesis) +
                          '\n```python\n' + code + '```', 'fake', 0., 0, 0)
    assert c.proposal(ident, response, code, SimpleNamespace(code=code))
    result = c.resolve(ident, status='evaluated', score=score, source=code,
        evaluation={'ok': True, 'score': score, 'cases': copy.deepcopy(CASES),
                    'reached': 'full', 'confirmed': False, 'raw_first_score': score})
    return result, prompt


def test_production_cards_mutations_auxiliary_and_hashes_share_exact_transport(fixture):
    c, parent, db, events, prompts = fixture
    first, _ = observe(c, parent, db, .7)
    lesson, = c.knowledge.add_lessons([{'rule': 'Local unverified interpretation.',
        'polarity': 'positive', 'scope': 'branch', 'evidence_ids': [first['id']]}],
        allowed_evidence_ids=[first['id']])
    second, prompt = observe(c, parent, db, .4)
    assert second['injected_ids'] == [lesson]
    mutation = wire(prompt, 'RESEARCH_CONTEXT_JSON:')
    assert mutation['schema'] == SCHEMA and REFERENCE_RULES in prompt
    context = decode_evidence(mutation)
    sources = wire(prompt, 'SOURCE_CATALOG_JSON:')['catalog']
    fragment, = context['fragments']
    assert 'code' not in fragment
    assert sources[fragment['source_ref']] == c.knowledge.fragments[0]['code']
    assert fragment['evidence']['evaluation'] == first['evaluation']
    assert context['parent_cases'] == CASES
    assert canonical_json(context['parent_cases']).endswith('"trial_count":3}')
    assert '"signed_zero":-0.0' in canonical_json(context)
    assert events[-3]['phase'] == 'attempt' and events[-3]['prompt_sha256'] == digest(prompt)

    ctx = c.context(db, parent, 0, 3)
    text, = ctx['lesson_texts']
    card_wire = wire(text, 'RESEARCH_LESSON_JSON:')
    assert card_wire['schema'] == SCHEMA and REFERENCE_RULES in text
    expected = observation_card(c.knowledge.lessons[0])
    assert canonical_json(decode_evidence(card_wire)) == canonical_json(expected)
    assert expected['source_ids'] == [first['id']]
    assert expected['advice_exposures']['shown_ids'] == [second['id']]
    assert expected['source_observations'][0]['actual_delta'] > 0
    assert expected['advice_exposures']['representatives'][0]['observation']['actual_delta'] < 0
    presentation = ctx['lesson_presentation']
    assert presentation['schema'] == 'sciel-memory-presentation/5'
    assert presentation['rendered_chars'] == len(text)
    assert presentation['presented'] == [{'id': lesson, 'chars': len(text), 'sha256': digest(text)}]

    records, lessons = c.knowledge.reflection_context(2, selection_epoch=c.reflection_attempts)
    data = serialize_reflection(records, lessons)
    assert c.knowledge.reflection_context(2, max_chars=len(data),
        selection_epoch=c.reflection_attempts) == (records, lessons)
    before = c.knowledge.state()
    c.complete_step()
    assert c.knowledge.state() == before  # fake auxiliary replies are explicitly empty
    assert len(prompts) == 3
    for prompt, record in zip(prompts[:2], (first, second)):
        payload = wire(prompt, 'EVIDENCE_JSON:')
        assert payload['schema'] == SCHEMA
        # C additionally receives its source-bound diff, while ordinary memory
        # retains the original observation. Validate the full source pair
        # against the measured identities before extending the exact wire view.
        binding = c.conclusions[record['id']]['source_evidence']
        assert binding['parent_source'] == parent.code
        assert digest(binding['candidate_source']) == record['source_sha256']
        projection = conclusion_source.validate_binding(record, binding)
        assert projection == conclusion_source.project(parent.code, binding['candidate_source'])
        assert projection['coverage']['complete'] is True
        assert 'conclusion_source' not in record
        assert all('conclusion_source' not in r for r in c.knowledge.experiments)
        assert canonical_json(decode_evidence(payload)) == canonical_json(
            {**{k: v for k, v in record.items() if k != 'code_fragment'},
             'conclusion_source': projection})
    raw = prompts[-1].split('EVIDENCE_JSON:\n', 1)[1].rstrip('\n')
    assert raw == data
    payload = decode_evidence(json.loads(raw))
    assert payload['lessons'][0]['source_support']['observations'] == [{'observation_ref': first['id']}]
    assert expand_reflection_payload(payload) == reflection_inline_payload(records, lessons)
    event, = [e for e in events if e['phase'] == 'reflection_context']
    assert event['context_sha256'] == digest(raw)
    state = c.state()
    assert state['schema'] == CONTRACT == 'sciel-research/12'
    assert state['presentation'] == presentation_identity()
    c.load_state(state)
    assert c.state() == state


@pytest.mark.parametrize('corruption', ['missing', 'wrong_hash', 'missing_fragment'])
def test_source_reference_validation_precedes_attempt_mutation(fixture, corruption):
    c, parent, db, events, _ = fixture
    observe(c, parent, db, .7)
    snap = snapshot(c, parent, db)
    prompt = build(snap.op, snap)
    prefix, raw = prompt.rsplit('SOURCE_CATALOG_JSON:\n', 1)
    catalog = json.loads(raw)
    if corruption == 'missing':
        prompt = prefix
    else:
        ref = snap.research_context['fragments'][0]['source_sha256']
        if corruption == 'wrong_hash': catalog['catalog'][ref] += '# changed\n'
        else: del catalog['catalog'][ref]
        prompt = prefix + 'SOURCE_CATALOG_JSON:\n' + json.dumps(catalog)
    state, prior_events = c.state(), copy.deepcopy(events)
    with pytest.raises(ValueError, match='source'):
        c.begin(snap, 0, snap.op, prompt)
    assert c.state() == state and events == prior_events


def test_repair_catalog_does_not_replace_measured_baseline(fixture):
    c, parent, db, _, _ = fixture
    observe(c, parent, db, .7)
    snap = snapshot(c, parent, db, repair=True)
    ident, prompt = c.begin(snap, 0, snap.op, build(snap.op, snap))
    context = decode_evidence(wire(prompt, 'RESEARCH_CONTEXT_JSON:'))
    catalog = wire(prompt, 'SOURCE_CATALOG_JSON:')['catalog']
    assert catalog[digest(snap.parent_code)] == snap.parent_code
    assert digest(parent.code) not in catalog
    assert context['parent_score'] == parent.score != snap.parent_score
    assert context['parent_source_sha256'] == digest(parent.code)
    assert context['parent_cases'] == parent.cases
    assert c._parents[ident] == parent.code


@pytest.mark.parametrize('corruption', ['old_contract', 'missing_layout', 'wrong_layout'])
def test_resume_refuses_mixed_presentation_before_state_mutation(fixture, corruption):
    c, _, _, _, _ = fixture
    before = c.state(); changed = copy.deepcopy(before)
    if corruption == 'old_contract': changed['schema'] = 'sciel-research/7'
    elif corruption == 'missing_layout': del changed['presentation']
    else: changed['presentation']['memory'] = 'sciel-memory-presentation/1'
    with pytest.raises(ValueError, match='checkpoint'):
        c.load_state(changed)
    assert c.state() == before


def test_inline_fallback_and_opaque_lookalikes_remain_literal():
    record = {'metric': None, 'parent_cases': {}, 'evaluation': None,
              'artifact': {'catalog_ref': 'literal', 'metric': ['opaque']}}
    assert serialize_record(record) == canonical_json(record)
    assert decode_evidence(json.loads(serialize_record(record))) == record
    context = {'parent_cases': CASES, 'fragments': [{'evidence': {'parent_cases': CASES}}],
               'opaque': {'schema': SCHEMA, 'catalog_ref': 'literal'}}
    wire = serialize_context(context)
    assert json.loads(wire)['schema'] == SCHEMA
    assert len(wire) + len(REFERENCE_RULES) < len(canonical_json(context))
    assert canonical_json(decode_evidence(json.loads(wire))) == canonical_json(context)
