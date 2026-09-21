"""Typed future reviews; no provider, source execution or rewritten history."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from sciel.hypothesis import mechanism_review as mechanism
from sciel.hypothesis.auxiliary_schema import response_format
from sciel.hypothesis.conclusion_feedback import (
    conclusion_feedback, reflection_review_context, validate_conclusions, validate_conclusion_feedback,
    TYPED_FEEDBACK_SCHEMA, TYPED_REFLECTION_REVIEW_SCHEMA,
)
from sciel.hypothesis.observed_facts import observed_comparison
from sciel.hypothesis.parsing import decode_conclusion, parse_conclusion, MAX_ID_CHARS
from sciel.hypothesis.prompts import conclusion_prompt, reflection_prompt
from sciel.hypothesis.rendering import decode_evidence, presentation_identity, serialize_context
from sciel.knowledge_base.catalogs import canonical_json
from sciel.engine.db import Database
from sciel.engine.types import Snapshot
from sciel.hypothesis import circle_restart_body as body
from test_mechanism_review import controller, entry as legacy_entry, payload
from test_structured_auxiliary_flow import construct


def record():
    c, _, _, _ = controller()
    return deepcopy(c.knowledge.experiments[0])


def forecast(ident='r000001a1', **changes):
    value = dict(claim_quote=None, source_quotes=[], assessment='contradicted',
        direct_effect='The native score delta lies outside the predicted interval.',
        untested_downstream='This comparison does not verify a solver mechanism.',
        evidence_kind='forecast',
        forecast_ref=dict(experiment_id=ident, target=mechanism.FORECAST_TARGET))
    value.update(changes)
    return value


def source_effect(**changes):
    return legacy_entry(**dict(evidence_kind='source_effect', forecast_ref=None) | changes)


def unavailable(**changes):
    return legacy_entry(**dict(evidence_kind='unavailable', forecast_ref=None,
        assessment='unknown', source_quotes=[]) | changes)


def saved_review(row, entries):
    value = payload(mechanism_review=entries, verdict=row['forecast_verdict'])
    parsed = parse_conclusion(json.dumps(value), require_typed=True)
    assert parsed is not None
    return dict(status='parsed', llm=parsed.to_dict(),
        observed_forecast_verdict=row['forecast_verdict'], observed_comparison=observed_comparison(row),
        verdict_matches_observation=True, accepted_lesson_ids=[], lesson_error=None, lesson_error_detail=None)


def test_old_payload_is_exact_and_new_dispatch_requires_homogeneous_typed_entries():
    old = payload()
    assert parse_conclusion(json.dumps(old)).to_dict() == old
    result = decode_conclusion(json.dumps(old), require_typed=True)
    assert result['conclusion'] is None and result['error']['code'] == 'typed_mechanism_required'
    for entries in ([legacy_entry(), forecast()], [forecast(), legacy_entry()]):
        assert parse_conclusion(json.dumps(payload(mechanism_review=entries))) is None
    typed = payload(mechanism_review=[source_effect(), forecast(), unavailable()])
    assert parse_conclusion(json.dumps(typed), require_typed=True).to_dict() == typed
    assert mechanism.validate_binding(typed['mechanism_review'], record(), require_typed=True) == typed['mechanism_review']
    assert mechanism.forecast_grounding(old['mechanism_review'], record()) is None


@pytest.mark.parametrize('changes', [
    dict(claim_quote='Adding a step will improve the result.'),
    dict(source_quotes=[dict(field='change_excerpt', text='value = base + step\n')]),
    dict(forecast_ref=None), dict(forecast_ref={}),
    dict(forecast_ref=dict(experiment_id='r000001a1', target='paired_sum_radii')),
    dict(forecast_ref=dict(experiment_id='r000001a1', target=mechanism.FORECAST_TARGET, extra=True)),
    dict(forecast_ref=dict(experiment_id=True, target=mechanism.FORECAST_TARGET)),
    dict(forecast_ref=dict(experiment_id='bad id', target=mechanism.FORECAST_TARGET)),
    dict(forecast_ref=dict(experiment_id='x' * 129, target=mechanism.FORECAST_TARGET)),
    dict(evidence_kind='numeric_proof'), dict(evidence_kind=True),
])
def test_forecast_cannot_bind_prose_code_quotes_or_arbitrary_measurement_target(changes):
    with pytest.raises(mechanism.MechanismReviewError):
        mechanism.validate_review([forecast(**changes)], require_typed=True)


@pytest.mark.parametrize('value', [
    source_effect(claim_quote=None, assessment='unknown', source_quotes=[]),
    source_effect(forecast_ref=dict(experiment_id='r000001a1', target=mechanism.FORECAST_TARGET)),
    {**unavailable(), 'assessment': 'contradicted'},
    {**unavailable(), 'source_quotes': [dict(field='change_excerpt', text='value')]},
    {**unavailable(), 'forecast_ref': dict(experiment_id='r000001a1', target=mechanism.FORECAST_TARGET)},
])
def test_nonforecast_cross_fields_fail_closed(value):
    with pytest.raises(mechanism.MechanismReviewError):
        mechanism.validate_review([value], require_typed=True)


def test_unknown_claim_rules_and_exact_source_whitespace_stay_literal():
    row = record()
    empty = unavailable(claim_quote=None)
    with pytest.raises(ValueError, match='claim quote'):
        mechanism.validate_binding([empty], row)
    row['hypothesis'] = None
    assert mechanism.validate_binding([empty], row) == [empty]
    with pytest.raises(ValueError, match='claim unavailable'):
        mechanism.validate_binding([unavailable()], row)
    row = record()
    bad = source_effect(source_quotes=[dict(field='change_excerpt', text='value = base  + step\n')])
    with pytest.raises(ValueError, match='source quote'):
        mechanism.validate_binding([bad], row)


@pytest.mark.parametrize('damage', ['id', 'paired', 'absent_id'])
def test_forecast_ref_must_name_this_ordinary_record(damage):
    row = record()
    if damage == 'id': row['id'] = 'r000002a1'
    elif damage == 'paired': row['schema'] = 'sciel-paired-circle-comparison/1'
    else: row.pop('id')
    with pytest.raises(mechanism.MechanismReviewError) as exc:
        mechanism.validate_binding([forecast()], row)
    assert exc.value.diagnostic['code'] == 'forecast_ref_mismatch'


@pytest.mark.parametrize('low,high,sign', [
    (-1., -0.1, 'negative'), (-1., 0., 'nonpositive'), (-1., 1., 'spans_zero'),
    (0., 0., 'zero'), (0., 1., 'nonnegative'), (.1, 1., 'positive'),
])
def test_forecast_assessment_and_interval_sign_come_from_same_saved_measurement(low, high, sign):
    row = record()
    row['hypothesis'].update(predicted_low=low, predicted_high=high)
    expected = 'directly_supported' if low <= row['actual_delta'] <= high else 'contradicted'
    entry = forecast(assessment=expected)
    facts = mechanism.forecast_grounding([entry], row)['forecasts'][0]
    assert facts['observed_comparison']['forecast']['interval_sign'] == sign
    assert facts['observed_comparison']['measurement']['direction'] == 'worsened'
    assert facts['required_assessment'] == expected
    wrong = 'contradicted' if expected == 'directly_supported' else 'directly_supported'
    with pytest.raises(ValueError, match='assessment conflicts'):
        mechanism.validate_binding([forecast(assessment=wrong)], row)


@pytest.mark.parametrize('changes', [
    dict(status='eval_failed'), dict(source_repaired=True), dict(source_repaired=None),
    dict(proposed_sha256='c' * 64), dict(score=None), dict(actual_delta=0),
])
def test_failed_unbound_or_inconsistent_measurements_remain_unknown(changes):
    row = record(); row.update(changes)
    good = forecast(assessment='unknown')
    facts = mechanism.forecast_grounding([good], row)['forecasts'][0]
    assert facts['computed_verdict'] == 'INCONCLUSIVE'
    assert facts['observed_comparison']['forecast']['position'] == 'unavailable'
    with pytest.raises(ValueError, match='assessment conflicts'):
        mechanism.validate_binding([forecast()], row)


def test_actual_r2_interval_is_spans_zero_with_positive_native_delta_outside_it():
    # Saved actual measurement; this is a synthetic future typed review, NOT a
    # converted or repaired historical provider response.
    fixture = json.loads((Path(__file__).parents[1] / 'fixtures/restart_r2_scope_signs.json').read_text())
    row = next(r for r in fixture['records'] if 'schema' not in r)
    before = deepcopy(row)
    value = forecast(row['id'], direct_effect='Unverified prose can still misdescribe a negative forecast.')
    facts = mechanism.forecast_grounding([value], row)['forecasts'][0]
    comparison = facts['observed_comparison']
    assert comparison['forecast']['predicted_low'] == -.001
    assert comparison['forecast']['predicted_high'] == .0002
    assert comparison['forecast']['interval_sign'] == 'spans_zero'
    assert comparison['forecast']['position'] == 'above'
    assert comparison['measurement']['direction'] == 'improved'
    assert facts['computed_verdict'] == 'REFUTED'
    assert row == before and value['direct_effect'].startswith('Unverified prose')


def test_grounded_feedback_and_reflection_are_separate_from_exact_unverified_prose():
    row = record(); entries = [source_effect(), forecast()]
    entries[0]['direct_effect'] = 'This real quote proves every solution optimal.'
    reviews = {row['id']: saved_review(row, entries)}; rows = {row['id']: row}
    before = deepcopy((reviews, rows))
    assert validate_conclusions(reviews, rows) == reviews
    feedback = conclusion_feedback(reviews, rows, 'p1')
    reflection = reflection_review_context(reviews, rows, [row['id']])
    assert feedback['schema'] == TYPED_FEEDBACK_SCHEMA
    assert reflection['schema'] == TYPED_REFLECTION_REVIEW_SCHEMA
    for block in (feedback, reflection):
        card = block['cards'][0]
        assert card['unverified_interpretation']['mechanism_review'] == entries
        assert card['mechanism_grounding']['forecasts'][0]['entry_index'] == 1
        assert card['mechanism_grounding']['forecasts'][0]['computed_verdict'] == 'REFUTED'
        assert 'unverified' in card['mechanism_grounding']['interpretation_limits']
        assert len(canonical_json(block)) <= block['max_chars']
    data = serialize_context({'conclusion_feedback': feedback})
    assert decode_evidence(json.loads(data))['conclusion_feedback'] == feedback
    prompt = reflection_prompt([row], [], review_context=reflection, structured=True)
    assert canonical_json(reflection) in prompt
    assert (reviews, rows) == before
    reflection['cards'][0]['mechanism_grounding']['forecasts'][0]['computed_verdict'] = 'CONFIRMED'
    assert reflection_review_context(reviews, rows, [row['id']])['cards'][0]['mechanism_grounding']['forecasts'][0]['computed_verdict'] == 'REFUTED'
    changed = deepcopy(reviews)
    changed[row['id']]['llm']['mechanism_review'][1]['assessment'] = 'directly_supported'
    with pytest.raises(ValueError, match='assessment conflicts'):
        validate_conclusions(changed, rows)


def test_grounding_budget_is_counted_before_whole_card_admission():
    row = record(); reviews = {row['id']: saved_review(row, [forecast()])}; rows = {row['id']: row}
    for make in (lambda limit: conclusion_feedback(reviews, rows, 'p1', max_chars=limit),
                 lambda limit: reflection_review_context(reviews, rows, [row['id']], max_chars=limit)):
        full = make(32000); boundary = len(canonical_json(full))
        # The embedded max_chars itself changes digit count from 32000.
        boundary = len(canonical_json(make(boundary)))
        bounded = make(boundary - 1)
        assert bounded is not None and bounded['cards'] == [] and bounded['omitted_count'] == 1
        assert len(canonical_json(bounded)) <= boundary - 1
        assert 'mechanism_grounding' not in canonical_json(bounded)


@pytest.mark.parametrize('damage', ['sign', 'prose', 'missing_grounding', 'extra_field',
    'max_items_bool', 'max_items_large', 'max_chars_bool', 'max_chars_large', 'disabled', 'other_branch'])
def test_snapshot_grounding_and_review_are_rederived_from_authoritative_record(damage):
    row = record(); reviews = {row['id']: saved_review(row, [forecast()])}; rows = {row['id']: row}
    value = conclusion_feedback(reviews, rows, 'p1')
    original = deepcopy(value)
    result = validate_conclusion_feedback(value, reviews, rows, 'p1')
    assert result == value and result is not value
    kwargs = {}
    branch = 'p1'
    if damage == 'sign':
        value['cards'][0]['mechanism_grounding']['forecasts'][0]['observed_comparison']['forecast']['interval_sign'] = 'spans_zero'
    elif damage == 'prose': value['cards'][0]['unverified_interpretation']['explanation'] = 'Different review.'
    elif damage == 'missing_grounding': value['cards'][0].pop('mechanism_grounding')
    elif damage == 'extra_field': value['cards'][0]['trusted'] = True
    elif damage == 'max_items_bool': value['max_items'] = True
    elif damage == 'max_items_large': value['max_items'] = 3
    elif damage == 'max_chars_bool': value['max_chars'] = False
    elif damage == 'max_chars_large': value['max_chars'] = 32001
    elif damage == 'disabled': kwargs['enabled'] = False
    else: branch = 'other'
    with pytest.raises(ValueError, match='conclusion feedback'):
        validate_conclusion_feedback(value, reviews, rows, branch, **kwargs)
    assert conclusion_feedback(reviews, rows, 'p1') == original


def test_typed_wire_and_prompt_are_explicit_without_changing_legacy_instruction():
    old = conclusion_prompt(record())
    new = conclusion_prompt(record(), structured=True)
    assert 'evidence_kind' not in old and 'forecast_ref' not in old
    assert mechanism.TYPED_RULES in new
    entry = response_format('conclusion')['json_schema']['schema']['properties']['mechanism_review']['items']
    assert set(entry['required']) == set(entry['properties']) == {
        'claim_quote', 'source_quotes', 'assessment', 'direct_effect', 'untested_downstream',
        'evidence_kind', 'forecast_ref'}
    ref = entry['properties']['forecast_ref']
    assert ref['type'] == ['object', 'null'] and ref['additionalProperties'] is False
    assert set(ref['required']) == {'experiment_id', 'target'}
    assert mechanism.MAX_REFERENCE_ID_CHARS == MAX_ID_CHARS
    assert response_format('reflection')['json_schema']['name'] == 'sciel_reflection_v1'
    identity = presentation_identity()['typed_mechanism']
    assert identity['schema'] == mechanism.SCHEMA and identity['grounding'] == mechanism.GROUNDING_SCHEMA
    assert identity['source_sha256'] == hashlib.sha256(Path(mechanism.__file__).read_bytes()).hexdigest()


def test_production_structured_route_delivers_grounding_to_reflection_and_next_generation(tmp_path):
    entry = forecast(assessment='directly_supported')
    f = construct(tmp_path, mechanism_entry=entry)
    original = deepcopy(f.research.knowledge.experiments)
    f.research.complete_step()
    assert f.research.conclusions[f.ident]['status'] == 'parsed'
    assert f.research.conclusions[f.ident]['llm']['mechanism_review'] == [entry]
    assert f.research.knowledge.experiments == original and f.research.knowledge.lessons == []
    assert len(f.calls) == 2 and f.research.reflections == 1
    delivered = f.calls[1]['messages'][0]['content']
    assert 'mechanism_grounding' in delivered and 'native_delta_interval' in delivered
    db = Database(1); source = body.source_for(); parent = db.add(source, .8, {}, 0, 'seed')
    snap = Snapshot(0, 'circle_generate', 'fake', 0, parent.id, source, .8)
    snap.research_context = f.research.context(db, parent, 0, 2)
    card = snap.research_context['conclusion_feedback']['cards'][0]
    assert card['mechanism_grounding']['forecasts'][0]['computed_verdict'] == 'CONFIRMED'
    _, prompt = f.research.begin(snap, 1, 'circle_generate', 'Inert next generation.',
        response_instruction=body.response_instruction(), context_scope=body.OPERATION)
    data, _ = json.JSONDecoder().raw_decode(prompt.split('RESEARCH_CONTEXT_JSON:', 1)[1].lstrip())
    shown = decode_evidence(data)
    assert shown['conclusion_feedback'] == snap.research_context['conclusion_feedback']
    assert len(f.calls) == 2  # No actual next model call or numerical evaluation.


@pytest.mark.parametrize('entry,reason', [
    (legacy_entry(), 'invalid_conclusion_structure'),
    (forecast(assessment='contradicted'), 'mechanism_binding_invalid'),
    (forecast('different_record', assessment='directly_supported'), 'mechanism_binding_invalid'),
])
def test_structured_route_rejects_wrong_contract_or_grounding_before_lessons_without_retry(tmp_path, entry, reason):
    f = construct(tmp_path, mechanism_entry=entry)
    f.research.complete_step(reflect=False)
    result = f.research.conclusions[f.ident]
    assert result['status'] == 'invalid' and result['llm'] is None
    assert not f.research.knowledge.lessons and len(f.calls) == 1 and f.research.aux_cost == .001
    rejection = next(e for e in f.events if e['phase'] == 'conclusion_rejected')
    returned = next(e for e in f.events if e['phase'] == 'aux_result')
    assert rejection['reason'] == reason and f.events.index(returned) < f.events.index(rejection)
    f.research.complete_step(reflect=False)
    assert len(f.calls) == 1


def test_new_runtime_refuses_old_presentation_identity_but_parser_keeps_old_payload(tmp_path):
    f = construct(tmp_path, mechanism_entry=forecast(assessment='directly_supported'))
    f.research.complete_step()
    before = f.research.state()
    old_identity = deepcopy(before)
    old_identity['presentation']['mechanism_review'] = mechanism.LEGACY_SCHEMA
    old_identity['presentation'].pop('typed_mechanism')
    with pytest.raises(ValueError, match='presentation mismatch'):
        f.research.load_state(old_identity)
    assert f.research.state() == before
    assert parse_conclusion(json.dumps(payload())).to_dict() == payload()


@pytest.mark.parametrize('require_typed', [None, 1, 'yes'])
def test_typed_requirement_is_boolean(require_typed):
    with pytest.raises(ValueError):
        mechanism.validate_review([forecast()], require_typed=require_typed)
    assert decode_conclusion(json.dumps(payload()), require_typed=require_typed)['error']['code'] == 'invalid_typed_requirement'
