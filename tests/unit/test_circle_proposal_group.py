"""Pure proposal grouping: no provider, candidate execution, or numerical calls."""
import copy
import hashlib
import json

import pytest

from sciel.hypothesis import circle_proposal_group as groups
from sciel.hypothesis import circle_restart_body as body

FIRST = 'x = ordinary_params.copy()\nx[52:] *= 0.99\nreturn x\n'
SECOND = 'x = ordinary_params.copy()\nx[:52] += uniform_noise[:52] * 0.2\nreturn x\n'


def hypothesis(label='first'):
    return {'hypothesis': label, 'predicted_low': -.02, 'predicted_high': .03,
            'reasoning': 'The initializer may explore another basin.', 'confidence': .6}


def wire(first=FIRST, second=SECOND, first_hyp=None):
    return json.dumps({'candidates': [
        {'hypothesis': hypothesis() if first_hyp is None else first_hyp, 'body': first},
        {'hypothesis': hypothesis('second'), 'body': second}]}, ensure_ascii=False)


def parent():
    return body.source_for()


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def make(raw=None, *, request_id='r000001a1', gates=None, selected=None, provider_error=None):
    raw = wire() if raw is None else raw
    if gates is None and selected is not None:
        gates = [groups.gate('passed') for _ in range(2)]
    proof = None
    if selected is not None:
        target = json.loads(raw)['candidates'][selected]['body']
        proof = body.apply(parent(), target)['proof']
    return groups.build_group(request_id=request_id, raw_response=raw,
        parent_source_sha256=sha(parent()), gates=gates, provider_error=provider_error,
        selected_proof=proof)


def record(group, *, status='evaluated'):
    chosen = group['selected_index']
    selected = group['entries'][chosen] if chosen is not None else None
    measured = status == 'evaluated'
    result = dict(id=group['request_id'], branch='p1', parent_id='p1', parent_source_sha256=sha(parent()),
        parent_score=.5, status=status, generation_dispatch='returned', generation_cost=.003,
        source_repaired=False, hypothesis=selected['hypothesis'] if selected else None,
        source_sha256=group['selected_source_sha256'], proposed_sha256=group['selected_source_sha256'],
        score=.6 if measured else None, actual_delta=.6-.5 if measured else None,
        child_id='p2' if measured else None, code_fragment='', injected_ids=[],
        proposal_group=copy.deepcopy(group))
    if selected:
        result['controlled_change'] = body.apply(parent(), selected['body'])['proof']
    return result


def test_pending_is_not_implicit_eligibility_and_inputs_are_detached():
    raw = wire(); value = make(raw)
    assert not value['assessment_complete'] and value['selected_index'] is None
    assert [e['eligible'] for e in value['entries']] == [False, False]
    assert [e['disposition'] for e in value['entries']] == ['gate_pending'] * 2
    assert value['entries'][0]['body'] == FIRST
    returned = groups.validate_group(value, raw_response=raw)
    returned['entries'][0]['body'] = 'return None'
    assert value['entries'][0]['body'] == FIRST
    with pytest.raises(ValueError, match='pending'):
        groups.validate_group(value, experiment=record(value, status='no_child'))
    groups.validate_group(value, experiment=record(value, status='generation_stopped'))


def test_selects_first_eligible_with_exact_forecast_source_and_proof():
    gates = [groups.gate('rejected', reason='baseline_initializer', evidence_sha256='a'*64), groups.gate('passed')]
    value = make(gates=gates, selected=1)
    assert value['selected_index'] == 1 and value['assessment_complete']
    assert [e['disposition'] for e in value['entries']] == ['gate_rejected', 'selected']
    assert value['entries'][1]['hypothesis']['text'] == 'second'
    assert value['selected_source_sha256'] == sha(body.source_for(SECOND))
    assert value['selected_proof_sha256'] == groups.digest(body.apply(parent(), SECOND)['proof'])
    groups.validate_group(value, raw_response=wire(), experiment=record(value))
    assert all(not set(e) & {'score','actual_delta','cost','outcome','forecast_verdict','child_id'} for e in value['entries'])


def test_two_eligible_choose_first_and_do_not_consume_the_other():
    value = make(selected=0)
    assert value['selected_index'] == 0
    assert [e['disposition'] for e in value['entries']] == ['selected', 'eligible_unselected']
    assert value['entries'][1]['eligible'] and value['entries'][1]['body'] == SECOND
    with pytest.raises(ValueError, match='body|binding'):
        make(selected=1)  # Can't supply the second proof while first remains eligible.


def test_card_temporal_labels_do_not_reassess_or_mutate_stored_group():
    value = make(selected=0)
    before = groups.canonical_json(value)
    card = groups.group_card(value)
    assert value['schema'] == 'sciel-circle-proposal-group/1'
    assert card['schema'] == 'sciel-circle-proposal-card/2'
    assert card['resubmission_eligibility'] == 'not_assessed'
    assert 'not permission to resubmit' in card['interpretation']
    assert 'not guaranteed eligible now' in card['interpretation']
    assert 'unmeasured in that original request' in card['interpretation']
    assert 'measurements in other requests remain separate' in card['interpretation']
    for original, shown in zip(value['entries'], card['entries']):
        assert not {'eligible', 'gate', 'disposition'} & shown.keys()
        assert shown['eligible_at_original_request'] is True
        assert shown['disposition_at_original_request'] == original['disposition']
        assert shown['gate_at_original_request'] == {
            key: original['gate'][key] for key in ('status', 'reason', 'evidence_sha256')}
        assert shown['hypothesis'] == original['hypothesis']
        assert shown['body'] is None and shown['body_sha256'] == original['body_sha256']
        assert not {'score', 'actual_delta', 'forecast_verdict'} & shown.keys()
    assert card['group_sha256'] == groups.digest(value)
    card['entries'][1]['gate_at_original_request']['reason'] = 'caller mutation'
    card['entries'][1]['hypothesis']['confidence'] = 0
    assert groups.canonical_json(value) == before
    assert groups.validate_group(value, raw_response=wire()) == value


def test_invalid_first_forecast_does_not_poison_identical_valid_second_source():
    raw = wire(SECOND, SECOND, first_hyp={'no': 'valid hypothesis'})
    value = make(raw, gates=[groups.gate('not_applicable'), groups.gate('passed')], selected=1)
    assert value['entries'][0]['error'] == 'invalid_hypothesis'
    assert value['entries'][0]['body'] == SECOND
    assert value['entries'][0]['hypothesis'] is None
    assert value['selected_index'] == 1
    assert value['entries'][0]['candidate_source_sha256'] == value['selected_source_sha256']
    groups.validate_group(value, raw_response=raw)


@pytest.mark.parametrize('invalid', ['return forbidden_name', 'x'*100000, 19, '\ud800'])
def test_invalid_body_omitted_but_valid_forecast_retained(invalid):
    raw = json.dumps({'candidates':[{'hypothesis':hypothesis(),'body':invalid},
                                  {'hypothesis':hypothesis('second'),'body':SECOND}]})
    value = make(raw, gates=[groups.gate('not_applicable'), groups.gate('passed')], selected=1)
    entry = value['entries'][0]
    assert entry['hypothesis']['text'] == 'first' and entry['body'] is None
    assert entry['body_omission_reason'] == 'invalid_body' and entry['body_error']
    assert entry['candidate_source_sha256'] is None
    assert len(groups.canonical_json(value)) < 10000
    groups.validate_group(value, raw_response=raw)


@pytest.mark.parametrize('raw', ['{', '{}', '{"candidates":[]}', '{"candidates":[{},{}],"candidates":[{},{}]}'])
def test_invalid_outer_envelope_has_no_invented_slots(raw):
    value = make(raw)
    assert value['entries'] == [] and value['envelope_error'] == 'invalid_json_envelope'
    assert value['assessment_complete'] and value['selected_index'] is None
    groups.validate_group(value, raw_response=raw, experiment=record(value,status='hypothesis_invalid'))


def test_provider_rejection_preserves_entries_but_cannot_select():
    value = make(provider_error='structured_response_not_complete_or_refused')
    assert len(value['entries']) == 2 and value['assessment_complete']
    assert value['selected_index'] is None
    assert [e['disposition'] for e in value['entries']] == ['provider_rejected']*2
    with pytest.raises(ValueError):
        make(gates=[groups.gate('passed')]*2, selected=0,
             provider_error='structured_response_not_complete_or_refused')


@pytest.mark.parametrize('mutation', [
    lambda x:x.update(selected_index=True),
    lambda x:x.update(assessment_complete=1),
    lambda x:x.update(selected_index=1),
    lambda x:x['entries'][0].update(index=False),
    lambda x:x['entries'][0].update(eligible=1),
    lambda x:x['entries'][1].update(disposition='selected'),
    lambda x:x['entries'][1].update(score=0),
    lambda x:x['entries'][1]['hypothesis'].update(confidence=True),
    lambda x:x['entries'][0].update(candidate_source_sha256='f'*64),
    lambda x:x['entries'][0].update(body=SECOND),
    lambda x:x['entries'][0].update(body_sha256='0'*64),
    lambda x:x['entries'][0]['gate'].update(status='pending'),
    lambda x:x['entries'][0].update(body_omission_reason='budget'),
    lambda x:x.update(entries=tuple(x['entries'])),
])
def test_rejects_typed_or_derived_tampering(mutation):
    value = make(selected=0); mutation(value)
    with pytest.raises(ValueError):groups.validate_group(value)


@pytest.mark.parametrize('mutation', [
    lambda e:e.update(id='different'),
    lambda e:e.update(parent_source_sha256='0'*64),
    lambda e:e.update(source_sha256='1'*64),
    lambda e:e.update(hypothesis=None),
    lambda e:e.update(source_repaired=True),
    lambda e:e['controlled_change'].update(target_body=SECOND),
    lambda e:e.update(generation_dispatch='not_dispatched'),
])
def test_selected_group_requires_exact_completed_experiment_binding(mutation):
    value = make(selected=0); e = record(value); mutation(e)
    with pytest.raises(ValueError):groups.validate_group(value, experiment=e)


def test_selected_then_stopped_before_source_commit_retains_only_proposal_facts():
    value = make(selected=0); e = record(value,status='generation_stopped')
    e.update(source_sha256=None,proposed_sha256=None);del e['controlled_change']
    groups.validate_group(value, experiment=e)
    e['status']='no_child'
    with pytest.raises(ValueError):groups.validate_group(value,experiment=e)


def test_raw_receipt_binding_rejects_changed_hypothesis_and_missing_invalid_entry_bytes():
    raw=wire();value=make(raw,selected=0)
    changed=json.loads(raw);changed['candidates'][1]['hypothesis']['reasoning']='Different'
    with pytest.raises(ValueError):groups.validate_group(value,raw_response=json.dumps(changed))
    value['entries'][1]['entry_sha256']='0'*64
    with pytest.raises(ValueError):groups.validate_group(value,raw_response=raw)


def test_complete_card_is_bounded_and_explicitly_omits_large_unicode_text_and_bodies():
    data=json.loads(wire())
    for entry in data['candidates']:
        entry['hypothesis'].update(hypothesis='α'*2000,reasoning='測'*4000)
    raw=json.dumps(data);value=make(raw,selected=0);card=groups.group_card(value)
    assert len(groups.canonical_json(card))<=groups.MAX_CARD_CHARS
    assert card['group_sha256']==groups.digest(value)
    assert all(e['body'] is None and e['body_omission_reason']=='presentation_body_omitted' for e in card['entries'])
    assert all(e['hypothesis_text_omissions']==['text','reasoning'] for e in card['entries'])
    assert card['entries'][0]['hypothesis']['predicted_low']==-.02
    assert 'unmeasured' in card['interpretation']
    assert value['entries'][0]['body']==FIRST


def test_deep_malformed_entry_is_bounded_indexed_receipt_not_infrastructure_failure():
    invalid = 0
    for _ in range(80):
        invalid = [invalid]
    raw = json.dumps({'candidates':[invalid,{'hypothesis':hypothesis('second'),'body':SECOND}]})
    value = make(raw,gates=[groups.gate('not_applicable'),groups.gate('passed')],selected=1)
    assert value['selected_index']==1 and value['entries'][0]['error']=='invalid_entry'
    assert value['entries'][0]['entry_sha256']==groups.digest({'response_sha256':value['response_sha256'],'index':0})
    groups.validate_group(value,raw_response=raw)


def test_raw_unpaired_surrogate_is_rejected_envelope_with_stable_receipt():
    raw='\ud800'
    value=make(raw)
    assert value['entries']==[] and value['envelope_error']=='invalid_json_envelope'
    assert value['response_sha256']==hashlib.sha256(raw.encode('utf-8',errors='surrogatepass')).hexdigest()
    groups.validate_group(value,raw_response=raw,experiment=record(value,status='hypothesis_invalid'))
