"""Pure quote-catalog grounding and size guards; no provider or candidate calls."""
from copy import deepcopy
import hashlib
import json

import pytest

from sciel.hypothesis import conclusion_refs as refs
from sciel.hypothesis.mechanism_review import MAX_REVIEW_CHARS, SOURCE_FIELDS
from sciel.hypothesis.parsing import MAX_BLOCK_CHARS


def wire(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def evidence():
    return dict(id='r1', hypothesis=dict(text='Adding a step could help.', reasoning='Reduce the residual.',
            predicted_low=.1, predicted_high=.4),
        change_excerpt='-value = base\n+value = base + step\n',
        controlled_change={'target_body': '\treturn value\r\n'},
        code_fragment='PRIVATE_SOURCE_NOT_PRESENTED', status='evaluated', parent_score=.5,
        score=.75, actual_delta=.25, source_repaired=False,
        proposed_sha256='a' * 64, source_sha256='a' * 64)


def entry(data=None, **changes):
    selected = refs.catalog(data or evidence())
    result = dict(claim_ref=selected['claims'][0]['id'], source_refs=[selected['sources'][0]['id']],
        assessment='directly_supported', direct_effect='The source adds a step.',
        untested_downstream='Search quality causality is unverified.', evidence_kind='source_effect')
    result.update(changes)
    return result


def payload(data=None, **changes):
    result = dict(verdict='CONFIRMED', explanation='The observed delta is within the forecast.',
                  lessons=[], mechanism_review=[entry(data)])
    result.update(changes)
    return result


def test_reference_decode_is_exact_detached_and_binds_catalog_and_raw_response():
    data = evidence()
    before = deepcopy(data)
    catalog = refs.catalog(data)
    raw = wire(payload(data))
    result = refs.decode(raw, data)
    assert result['error'] is None and data == before
    actual, = result['conclusion'].mechanism_review
    assert actual == dict(claim_quote=catalog['claims'][0]['text'],
        source_quotes=[dict(field=catalog['sources'][0]['field'], text=catalog['sources'][0]['text'])],
        assessment='directly_supported', direct_effect='The source adds a step.',
        untested_downstream='Search quality causality is unverified.', evidence_kind='source_effect', forecast_ref=None)
    resolution = result['resolution']
    assert resolution['schema'] == refs.SCHEMA
    assert resolution['raw_response_sha256'] == hashlib.sha256(raw.encode()).hexdigest()
    assert resolution['catalog_sha256'] == refs._sha(refs._json(catalog))
    assert resolution['used_reference_ids'] == [catalog['claims'][0]['id'], catalog['sources'][0]['id']]
    actual['source_quotes'][0]['text'] = 'changed externally'
    assert data == before and refs.catalog(data) == catalog
    assert 'PRIVATE_SOURCE_NOT_PRESENTED' not in refs.instruction(data)


@pytest.mark.parametrize('changes,code', [
    ({'claim_ref': 'unknown-claim'}, 'unknown_claim_reference'),
    ({'claim_ref': ['not-a-string']}, 'unknown_claim_reference'),
    ({'source_refs': ['unknown-source']}, 'invalid_source_references'),
    ({'source_refs': [{'injected': 'object'}]}, 'invalid_source_references'),
    ({'source_refs': 'not-a-list'}, 'invalid_source_references'),
    ({'source_refs': []}, 'invalid_source_quote_count'),
    ({'claim_quote': 'model must not inject literal claims'}, 'invalid_reference_entry_fields'),
    ({'assessment': 'certain'}, 'invalid_assessment'),
])
def test_unknown_or_forged_refs_never_fall_back_to_literal_quotes(changes, code):
    raw = payload(mechanism_review=[entry(**changes)])
    result = refs.decode(wire(raw), evidence())
    assert result['conclusion'] is result['resolution'] is None
    assert result['error']['code'] == code
    assert 'model must not' not in wire(result)


def test_duplicate_source_refs_and_unlisted_fields_are_rejected():
    item = entry()
    item['source_refs'] *= 2
    assert refs.decode(wire(payload(mechanism_review=[item])), evidence())['error']['code'] == 'invalid_source_references'
    data = evidence()
    assert all(row['field'] in SOURCE_FIELDS for row in refs.catalog(data)['sources'])
    assert refs.decode(wire(payload(mechanism_review=[entry(source_refs=['code_fragment'])])), data)['conclusion'] is None


@pytest.mark.parametrize('field', ['claim', 'source'])
def test_response_refs_from_changed_evidence_cannot_be_rebound(field):
    data = evidence()
    raw = wire(payload(data))
    if field == 'claim':
        data['hypothesis']['text'] = 'A different original claim.'
    else:
        data['controlled_change']['target_body'] = '\treturn other\r\n'
    result = refs.decode(raw, data)
    assert result['conclusion'] is None
    assert result['error']['code'] == ('unknown_claim_reference' if field == 'claim' else 'invalid_source_references')


def test_forecast_uses_this_experiment_and_rederived_observation_not_source_prose():
    data = evidence()
    forecast = entry(claim_ref=None, source_refs=[], evidence_kind='forecast')
    valid = refs.decode(wire(payload(mechanism_review=[forecast])), data)
    assert valid['error'] is None
    assert valid['conclusion'].mechanism_review[0]['forecast_ref'] == {'experiment_id': 'r1', 'target': 'native_delta_interval'}
    data.update(score=.4, actual_delta=.4 - .5)
    assert refs.decode(wire(payload(mechanism_review=[forecast])), data)['error']['code'] == 'forecast_assessment_mismatch'


def test_exact_crlf_unicode_offsets_and_boundary_characters_survive_decode():
    data = evidence()
    data['hypothesis']['text'] = 'Café e\u0301\r\n可能性'
    data['controlled_change']['target_body'] = '\tvalue = "α β"\r\nreturn value\x85# 終\u2028'
    selected = refs.catalog(data)
    fields = {**{f'hypothesis.{k}': v for k, v in data['hypothesis'].items() if type(v) is str},
              **refs.source_fields(data)}
    for row in selected['claims'] + selected['sources']:
        assert row['text'] == fields[row['field']][row['start']:row['end']]
        assert 0 < len(row['text']) <= refs.MAX_FRAGMENT_CHARS
    result = refs.decode(wire(payload(data)), data)
    assert result['error'] is None
    assert result['conclusion'].mechanism_review[0]['source_quotes'][0]['text'] == '\tvalue = "α β"\r\n'


def test_large_unicode_payload_does_not_inflate_transport_during_resolution():
    data = evidence()
    raw = payload(explanation='界' * 4096, lessons=[dict(rule='漢' * 2048, polarity='positive', scope='global', evidence_ids=['r1'])])
    assert len(wire(raw)) < MAX_BLOCK_CHARS < len(refs._json(raw))
    result = refs.decode(wire(raw), data)
    assert result['error'] is None and result['conclusion'].explanation == raw['explanation']
    assert result['conclusion'].lessons == raw['lessons']


@pytest.mark.parametrize('field', ['direct_effect', 'untested_downstream'])
@pytest.mark.parametrize('text', ['', None, 'x' * 513])
def test_effect_fields_keep_existing_hard_limits(field, text):
    result = refs.decode(wire(payload(mechanism_review=[entry(**{field: text})])), evidence())
    assert result['conclusion'] is None and result['resolution'] is None
    assert result['error']['path'] == 'mechanism_review[0].' + field
    assert result['error']['code'] == ('text_too_long' if text is not None and len(text) > 512 else 'invalid_text')


def test_expanded_unicode_review_retains_escaped_canonical_aggregate_cap():
    row = entry(direct_effect='界' * 512, untested_downstream='界' * 512)
    result = refs.decode(wire(payload(mechanism_review=[row])), evidence())
    assert result['conclusion'] is None and result['resolution'] is None
    assert result['error']['code'] == 'review_too_long'
    assert result['error']['actual'] > result['error']['maximum'] == MAX_REVIEW_CHARS


def test_sample_zero_one_and_even_end_coverage_are_defined():
    assert refs._sample(list(range(10)), 0) == []
    assert refs._sample(list(range(10)), 1) == [0]
    assert refs._sample(list(range(10)), 3) == [0, 4, 9]
    assert refs._sample([], 1) == []


def test_catalog_cap_and_omission_metadata_cover_large_escaped_fields():
    data = evidence()
    source = ('😀' * 150 + '\n') * 100
    data['hypothesis'].update(text=source, reasoning=source)
    data['controlled_change'] = dict(parent_body=source, target_body=source,
                                    before={'text': source}, after={'text': source})
    data['change_excerpt'] = source
    selected = refs.catalog(data)
    assert len(refs._json(selected)) <= refs.MAX_CATALOG_CHARS
    assert len(selected['claims']) <= refs.MAX_CLAIM_REFS and len(selected['sources']) <= refs.MAX_SOURCE_REFS
    assert len(selected['fields']) == 7
    assert any(row['omitted_fragments'] > 0 for row in selected['fields'])
    for meta in selected['fields']:
        count = sum(row['field'] == meta['field'] for row in selected['claims'] + selected['sources'])
        assert meta['total_fragments'] == 100
        assert meta['shown_fragments'] == count
        assert meta['shown_fragments'] + meta['omitted_fragments'] == meta['total_fragments']


def test_newline_pathology_materializes_only_bounded_selected_fragments(monkeypatch):
    data = dict(id='r1', hypothesis=None, change_excerpt='x\n' * 25000)
    original, calls = refs._fragment, []
    def counted(*args):
        calls.append(args[3:])
        return original(*args)
    monkeypatch.setattr(refs, '_fragment', counted)
    selected = refs.catalog(data)
    assert len(calls) == len(selected['sources']) == refs.MAX_SOURCE_REFS
    assert selected['fields'][0]['total_fragments'] == 25000
    assert selected['sources'][0]['start'] == 0 and selected['sources'][-1]['end'] == 50000
    assert len(refs._json(selected)) <= refs.MAX_CATALOG_CHARS


@pytest.mark.parametrize('bad', [None, {'id': 'x' * 18001}, {'id': ['not-an-id']}])
def test_malformed_evidence_fails_closed_without_catalog_pop_or_attribute_errors(bad):
    result = refs.decode(wire(payload()), bad)
    assert result['conclusion'] is result['resolution'] is None
    assert result['error']['code'] in ('invalid_evidence', 'invalid_evidence_id')


def test_transport_duplicate_keys_and_injected_extra_fields_fail_closed():
    raw = wire(payload())
    duplicate = raw.replace('"verdict":"CONFIRMED"', '"verdict":"CONFIRMED","verdict":"REFUTED"')
    assert refs.decode(duplicate, evidence())['conclusion'] is None
    assert refs.decode(wire({**payload(), 'resolution': {'verified': True}}), evidence())['conclusion'] is None


def test_response_schema_is_detached_and_only_allows_reference_wire_fields():
    first = refs.response_format()
    fields = first['json_schema']['schema']['properties']['mechanism_review']['items']['properties']
    assert set(fields) == {'claim_ref', 'source_refs', 'assessment', 'direct_effect', 'untested_downstream', 'evidence_kind'}
    fields.clear()
    assert refs.response_format()['json_schema']['schema']['properties']['mechanism_review']['items']['properties']
