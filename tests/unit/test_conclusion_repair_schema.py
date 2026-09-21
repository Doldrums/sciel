"""Evidence-bound repair wire prevents the observed valid-shape/invalid-kind combinations."""
from copy import deepcopy
import json

import pytest

from sciel.engine.llm.client import validate_response_format
from sciel.hypothesis import conclusion_refs as refs
from test_conclusion_source_refs import evidence, entry, payload, wire
from test_conclusion_format_repair import setup, response


def matches(schema, value):
    """Evaluate only the documented wire subset emitted here, independently."""
    if 'anyOf' in schema and not any(matches(s, value) for s in schema['anyOf']):
        return False
    if 'enum' in schema and value not in schema['enum']:
        return False
    kind = schema.get('type')
    if kind == 'null': return value is None
    if kind == 'string': return type(value) is str
    if kind == 'array':
        return (type(value) is list and len(value) >= schema.get('minItems', 0)
                and len(value) <= schema.get('maxItems', float('inf'))
                and all(matches(schema.get('items', {}), v) for v in value))
    if kind == 'object':
        return (type(value) is dict and all(k in value for k in schema.get('required', []))
                and (schema.get('additionalProperties') is not False or set(value) <= set(schema.get('properties', {})))
                and all(matches(s, value[k]) for k, s in schema.get('properties', {}).items() if k in value))
    return True


@pytest.mark.parametrize('model', ['google/gemini-2.5-flash', 'anthropic/claude-sonnet-4', 'fixture'])
def test_non_unknown_source_effect_requires_real_catalog_source_on_wire(model):
    data = evidence()
    review = refs.repair_response_format(data, model=model)['json_schema']['schema']['properties']['mechanism_review']
    assert matches(review, [entry()])
    assert not matches(review, [entry(source_refs=[], assessment='contradicted')])
    assert not matches(review, [entry(source_refs=[refs.catalog(data)['claims'][0]['id']])])
    assert not matches(review, [entry(claim_ref='foreign-claim')])
    assert matches(review, [entry(source_refs=[], assessment='unknown')])
    assert matches(review, [entry(source_refs=[], assessment='unknown', evidence_kind='unavailable')])


def test_forecast_identity_is_constrained_and_array_caps_remain_local():
    data = evidence()
    fmt = refs.repair_response_format(data, model='google/gemini-2.5-flash')
    assert validate_response_format(fmt) == fmt
    review = fmt['json_schema']['schema']['properties']['mechanism_review']
    forecast = entry(claim_ref=None, source_refs=[], evidence_kind='forecast')
    assert matches(review, [forecast])
    assert not matches(review, [{**forecast, 'assessment': 'contradicted'}])
    assert not matches(review, [{**forecast, 'claim_ref': entry()['claim_ref']}])
    nonempty = {**forecast, 'source_refs': entry()['source_refs']}
    assert matches(review, [nonempty])
    assert refs.decode(wire(payload(mechanism_review=[nonempty])), data)['conclusion'] is None
    unavailable = entry(evidence_kind='unavailable', assessment='unknown')
    assert matches(review, [unavailable])
    assert refs.decode(wire(payload(mechanism_review=[unavailable])), data)['conclusion'] is None
    too_many_sources = entry(source_refs=[r['id'] for r in refs.catalog(data)['sources'][:3]])
    assert matches(review, [too_many_sources])
    assert refs.decode(wire(payload(mechanism_review=[too_many_sources])), data)['conclusion'] is None
    assert matches(review, [entry()] * 5)
    assert refs.decode(wire(payload(mechanism_review=[entry()] * 5)), data)['conclusion'] is None


def test_portable_profile_retains_unsupported_array_caps_as_local_checks():
    data = evidence()
    fmt = refs.repair_response_format(data, model='anthropic/claude-sonnet-4')
    assert 'maxItems' not in json.dumps(fmt)
    assert 'minItems' in json.dumps(fmt) and 'anyOf' in json.dumps(fmt)
    review = fmt['json_schema']['schema']['properties']['mechanism_review']
    invalid = entry(evidence_kind='forecast', claim_ref=None)
    assert matches(review, [invalid])  # Portable wire cannot guarantee empty arrays.
    assert refs.decode(wire(payload(mechanism_review=[invalid])), data)['conclusion'] is None
    assert fmt == refs.repair_response_format(data, model='google/gemini-2.5-flash')
    assert fmt == refs.repair_response_format(data)


@pytest.mark.parametrize('no_claims,no_sources', [(True, False), (False, True), (True, True)])
def test_empty_catalog_has_no_impossible_source_judgment_or_empty_enum(no_claims, no_sources):
    data = evidence()
    if no_claims: data['hypothesis'] = None
    if no_sources:
        data.pop('change_excerpt')
        data.pop('controlled_change')
    fmt = refs.repair_response_format(data, model='google/gemini-2.5-flash')
    assert '"enum": []' not in json.dumps(fmt)
    review = fmt['json_schema']['schema']['properties']['mechanism_review']
    kinds = [s['properties']['evidence_kind']['enum'][0] for s in review['items']['anyOf']]
    if no_claims: assert 'source_effect' not in kinds
    if no_sources:
        assert all(s['properties']['assessment']['enum'] == ['unknown'] for s in review['items']['anyOf']
                   if s['properties']['evidence_kind']['enum'] == ['source_effect'])
    unavailable = entry(evidence_kind='unavailable', assessment='unknown', source_refs=[],
                        claim_ref=None if no_claims else refs.catalog(data)['claims'][0]['id'])
    assert matches(review, [unavailable])
    assert refs.decode(wire(payload(mechanism_review=[unavailable])), data)['error'] is None


def test_schema_is_detached_and_catalog_specific_without_mutating_original_wire():
    data = evidence()
    first = refs.repair_response_format(data)
    original = deepcopy(first)
    first['json_schema']['schema']['properties'].clear()
    assert refs.repair_response_format(data) == original
    data['hypothesis']['text'] = 'A different claim.'
    assert refs.repair_response_format(data) != original
    assert refs.response_format()['json_schema']['name'] == 'sciel_conclusion_refs_v2'


def test_controller_emits_and_preserves_exact_correction_schema_and_two_charges():
    c, model, events, charges = setup([response(bad_ref=True), response(lessons=True)])
    data = deepcopy(c.queue[0])
    c.complete_step()
    assert len(model.calls) == 2 and charges == [.03, .03]
    assert c.conclusions[data['id']]['status'] == 'parsed'
    starts = [e for e in events if e['phase'] == 'aux_start']
    assert starts[0]['response_format'] == refs.evidence_response_format(data, model='fixture')
    assert starts[1]['response_format'] == model.calls[1][1]['response_format'] == refs.repair_response_format(data, model='fixture')
    restored, _, _, _ = setup([])
    restored.queue = []
    restored.load_state(c.state())
    assert restored.state() == c.state()


def test_repair_evidence_must_match_reference_before_call_or_charge():
    c, model, events, charges = setup([])
    with pytest.raises(ValueError, match='repair evidence'):
        c.prepare_aux('conclusion', 'prompt', 'different', repair_evidence=evidence())
    with pytest.raises(ValueError, match='repair evidence'):
        c.prepare_aux('reflection', 'prompt', 'r1', repair_evidence=evidence())
    assert c.aux_calls == 0 and not model.calls and not charges
