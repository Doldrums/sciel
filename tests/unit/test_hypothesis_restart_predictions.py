"""Shared hypothesis forecasts persist without creating extra evidence or credit."""
import copy
import hashlib
import json

import pytest

from sciel.hypothesis import Hypothesis
from sciel.hypothesis import circle_proposal_group as groups
from sciel.hypothesis import circle_restart_body as body
from sciel.hypothesis import circle_proposal_pair as pair
from sciel.hypothesis import restart_predictions as predictions
from sciel.hypothesis.parsing import parse_hypothesis, parse_hypothesis_value
from sciel.knowledge_base import KnowledgeBase
from sciel.knowledge_base.catalogs import canonical_json
from test_circle_proposal_group import make, record, wire
from test_research_knowledge import experiment, proposal


def spec():
    return {'schema': predictions.SCHEMA,
            'expected': dict(zip(predictions.NAMES, (True, False, True)))}


def hypothesis(*, extended=True):
    value = {'hypothesis': 'Move accepted restart locations.', 'predicted_low': -0.02,
             'predicted_high': 0.04, 'reasoning': 'The return may retain a different arrangement.',
             'confidence': .6}
    if extended:
        value['restart_predictions'] = spec()
    return value


def digest(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def bound_record(*, delta=0.0):
    result = experiment(1, delta=delta,
                        hypothesis=parse_hypothesis_value(hypothesis()).to_dict())
    result['restart_prediction_binding'] = {
        'schema': 'sciel-restart-prediction-binding/1', 'spec_sha256': digest(spec()),
        'response_sha256': 'a' * 64, 'proposed_sha256': result['proposed_sha256'],
        'source_sha256': result['source_sha256']}
    return result


def test_legacy_hypothesis_serialization_is_exactly_unchanged():
    original = hypothesis(extended=False)
    expected = {'text': original['hypothesis'], **{k: v for k, v in original.items() if k != 'hypothesis'}}
    parsed = parse_hypothesis_value(original)
    assert parsed == Hypothesis(expected['text'], -.02, .04, expected['reasoning'], .6)
    assert canonical_json(parsed.to_dict()) == canonical_json(expected)
    assert 'restart_predictions' not in parsed.to_dict()


def test_extended_parser_and_to_dict_are_detached_and_preserve_false_expectations():
    value = hypothesis()
    parsed = parse_hypothesis_value(value)
    assert parsed.restart_predictions == spec()
    value['restart_predictions']['expected']['result_changed'] = True
    shown = parsed.to_dict()
    shown['restart_predictions']['expected']['result_changed'] = True
    assert parsed.restart_predictions['expected']['result_changed'] is False
    assert parse_hypothesis('HYPOTHESIS_JSON: ' + json.dumps(hypothesis())) == parsed
    assert parse_hypothesis(json.dumps(hypothesis()) + '\n```python\npass\n```') == parsed


@pytest.mark.parametrize('bad', [None, {}, [], True,
    {'schema': predictions.SCHEMA, 'expected': {}},
    {'schema': 'future', 'expected': {name: True for name in predictions.NAMES}},
    {'schema': predictions.SCHEMA, 'expected': {name: 1 for name in predictions.NAMES}},
    {'schema': predictions.SCHEMA, 'expected': {name: None for name in predictions.NAMES}},
    {'schema': predictions.SCHEMA, 'expected': {name: 'true' for name in predictions.NAMES}},
    {'schema': predictions.SCHEMA, 'expected': {**dict.fromkeys(predictions.NAMES, True), 'extra': False}},
    {'schema': predictions.SCHEMA, 'expected': dict.fromkeys(predictions.NAMES, True), 'extra': 1},
])
def test_invalid_extension_invalidates_whole_forecast_without_partial_salvage(bad):
    value = {**hypothesis(), 'restart_predictions': bad}
    assert parse_hypothesis_value(value) is None
    assert parse_hypothesis('HYPOTHESIS_JSON: ' + json.dumps(value)) is None
    kb = KnowledgeBase()
    before = kb.state()
    stored = {'text': value['hypothesis'], **{k: v for k, v in value.items() if k != 'hypothesis'}}
    with pytest.raises(ValueError):
        kb.record(experiment(1, hypothesis=stored))
    assert kb.state() == before


def test_duplicate_prediction_json_key_is_rejected_by_all_transports():
    raw = json.dumps(hypothesis()).replace('"result_changed": false',
                                         '"result_changed": false, "result_changed": true')
    assert parse_hypothesis('HYPOTHESIS_JSON: ' + raw) is None
    assert body.decode_response('{"hypothesis":' + raw + ',"body":"return None"}')['hypothesis'] is None
    assert pair.decode_response('{"candidates":[{"hypothesis":' + raw +
        ',"body":"return None"},{"hypothesis":' + json.dumps(hypothesis()) +
        ',"body":"return None"}]}')['error'] == 'invalid_json_envelope'


def test_kb_roundtrip_preserves_bound_spec_without_quality_or_lesson_credit():
    kb = KnowledgeBase()
    original = bound_record()
    kb.record(original)
    saved = kb.experiments[0]
    assert saved['hypothesis']['restart_predictions'] == spec()
    assert saved['restart_prediction_binding'] == original['restart_prediction_binding']
    assert saved['actual_delta'] == 0 and saved['forecast_verdict'] == 'CONFIRMED'
    assert not kb.lessons and not kb.fragments
    for polarity in ('positive', 'negative'):
        with pytest.raises(ValueError):
            kb.add_lessons([proposal(original['id'], polarity=polarity)],
                           allowed_evidence_ids=[original['id']])
    state = kb.state()
    restored = KnowledgeBase()
    restored.load_state(json.loads(canonical_json(state)))
    assert canonical_json(restored.state()) == canonical_json(state)
    assert set(restored._evidence()) == {original['id']}


def test_bound_prediction_survives_lesson_snapshot_after_original_eviction():
    kb = KnowledgeBase(max_experiments=1)
    original = bound_record(delta=.02)
    kb.record(original)
    kb.add_lessons([proposal(original['id'])], allowed_evidence_ids=[original['id']])
    kb.record(experiment(2, delta=0))
    assert [e['id'] for e in kb.experiments] == ['r000002a1']
    assert kb._evidence()[original['id']]['restart_prediction_binding'] == original['restart_prediction_binding']
    state = kb.state()
    restored = KnowledgeBase(max_experiments=1)
    restored.load_state(json.loads(canonical_json(state)))
    assert canonical_json(restored.state()) == canonical_json(state)
    assert restored.lessons[0]['source_support_ids'] == [original['id']]


@pytest.mark.parametrize('key,value', [
    ('schema', 'future'), ('spec_sha256', 'b' * 64), ('response_sha256', None),
    ('response_sha256', True), ('source_sha256', 'b' * 64), ('proposed_sha256', 'c' * 64),
    ('extra', 'unsupported'),
])
def test_invalid_prediction_binding_rejected_atomically(key, value):
    original = bound_record()
    original['restart_prediction_binding'][key] = value
    kb = KnowledgeBase()
    before = kb.state()
    with pytest.raises(ValueError, match='restart prediction'):
        kb.record(original)
    assert kb.state() == before


@pytest.mark.parametrize('change', ['no_hypothesis', 'no_spec', 'no_source', 'missing_hash', 'null_binding'])
def test_binding_requires_complete_original_spec_and_source(change):
    original = bound_record()
    if change == 'no_hypothesis':
        original['hypothesis'] = None
    elif change == 'no_spec':
        del original['hypothesis']['restart_predictions']
    elif change == 'no_source':
        original['source_sha256'] = None
        original['code_fragment'] = ''
    elif change == 'missing_hash':
        del original['restart_prediction_binding']['response_sha256']
    else:
        original['restart_prediction_binding'] = None
    kb = KnowledgeBase()
    before = kb.state()
    with pytest.raises(ValueError, match='restart prediction'):
        kb.record(original)
    assert kb.state() == before


def test_uncommitted_rejected_hypothesis_can_persist_without_invented_binding():
    kb = KnowledgeBase()
    kb.record(experiment(1, status='proposal_rejected',
        hypothesis=parse_hypothesis_value(hypothesis()).to_dict()))
    stored = kb.experiments[0]
    assert stored['hypothesis']['restart_predictions'] == spec()
    assert 'restart_prediction_binding' not in stored
    assert stored['forecast_verdict'] == 'INCONCLUSIVE'
    assert stored['source_sha256'] is stored['score'] is stored['actual_delta'] is None
    restored = KnowledgeBase()
    restored.load_state(kb.state())
    assert restored.state() == kb.state()


def test_repaired_source_preserves_original_bound_forecast_without_scalar_verdict():
    original = bound_record()
    original['source_repaired'] = True
    original['proposed_sha256'] = original['restart_prediction_binding']['proposed_sha256'] = 'c' * 64
    kb = KnowledgeBase()
    kb.record(original)
    assert kb.experiments[0]['forecast_verdict'] == 'INCONCLUSIVE'
    assert kb.experiments[0]['restart_prediction_binding'] == original['restart_prediction_binding']


def test_extended_alternatives_and_selected_raw_response_survive_group_and_kb_restore():
    raw = json.loads(wire(first_hyp=hypothesis()))
    raw['candidates'][1]['hypothesis']['restart_predictions'] = spec()
    raw['candidates'][1]['hypothesis']['restart_predictions']['expected']['result_changed'] = True
    text = json.dumps(raw)
    group = make(text, selected=0)
    original = record(group)
    original['restart_prediction_binding'] = {
        'schema': 'sciel-restart-prediction-binding/1', 'spec_sha256': digest(spec()),
        'response_sha256': group['response_sha256'], 'proposed_sha256': group['selected_source_sha256'],
        'source_sha256': group['selected_source_sha256']}
    assert groups.validate_group(group, raw_response=text, experiment=original) == group
    assert group['entries'][1]['hypothesis']['restart_predictions']['expected']['result_changed'] is True
    assert group['entries'][1]['disposition'] == 'eligible_unselected'
    kb = KnowledgeBase()
    kb.record(original)
    assert len(kb.experiments) == 1 and not kb.lessons and not kb.fragments
    restored = KnowledgeBase()
    restored.load_state(json.loads(canonical_json(kb.state())))
    assert restored.state() == kb.state()
    tampered = copy.deepcopy(original)
    tampered['restart_prediction_binding']['response_sha256'] = 'f' * 64
    other = KnowledgeBase()
    before = other.state()
    with pytest.raises(ValueError, match='proposal response'):
        other.record(tampered)
    assert other.state() == before


def test_restore_rejects_changed_spec_before_replacing_live_kb():
    kb = KnowledgeBase()
    kb.record(bound_record())
    state = kb.state()
    state['experiments'][0]['hypothesis']['restart_predictions']['expected']['result_changed'] = True
    restored = KnowledgeBase()
    restored.record(experiment(90))
    before = restored.state()
    with pytest.raises(ValueError):
        restored.load_state(state)
    assert restored.state() == before
