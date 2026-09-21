"""One correction sees all reference defects without weakening scientific gates."""
from copy import deepcopy
import json

import pytest

from sciel.hypothesis import conclusion_refs as refs
from test_conclusion_source_refs import evidence, entry, payload, wire
from test_conclusion_format_repair import setup, response


def test_real_failure_pattern_reports_count_and_claim_as_source_together():
    data = evidence()
    catalog = refs.catalog(data)
    source_ids = [row['id'] for row in catalog['sources']]
    claim_id = catalog['claims'][0]['id']
    raw = wire(payload(mechanism_review=[
        entry(source_refs=source_ids[:3]), entry(source_refs=[claim_id])]))
    before = deepcopy(data)
    result = refs.decode(raw, data)
    assert result['conclusion'] is result['resolution'] is None
    assert result['error']['code'] == 'invalid_source_references'
    diagnostics = result['error']['reference_repair']
    assert [(row['code'], row['path']) for row in diagnostics['issues']] == [
        ('source_ref_count', 'mechanism_review[0].source_refs'),
        ('claim_id_used_as_source', 'mechanism_review[1].source_refs[0]')]
    assert diagnostics['issues'][1]['value'] == claim_id
    assert diagnostics['allowed_source_ids'] == source_ids
    assert claim_id not in diagnostics['allowed_source_ids']
    assert data == before
    prompt = refs.repair_prompt('ORIGINAL EVIDENCE', raw, result['error'])
    assert 'ALL diagnostic.reference_repair.issues' in prompt
    assert 'A score increase/decrease alone does not establish a source mechanism' in prompt
    encoded = json.loads(prompt.split('REJECTED_RESPONSE_JSON:\n')[1])
    assert encoded['previous_response'] == raw
    assert encoded['diagnostic'] == result['error']


def test_all_empty_source_effects_are_reported_without_becoming_accepted():
    raw = wire(payload(mechanism_review=[
        entry(source_refs=[], assessment='contradicted'),
        entry(source_refs=[], assessment='directly_supported')]))
    result = refs.decode(raw, evidence())
    assert result['conclusion'] is None
    assert result['error']['code'] == 'invalid_source_quote_count'
    issues = result['error']['reference_repair']['issues']
    assert [row['path'] for row in issues] == [
        'mechanism_review[0].source_refs', 'mechanism_review[1].source_refs']
    assert all(row['code'] == 'non_unknown_source_effect_requires_source' for row in issues)


def test_structural_entry_count_error_still_explains_later_reference_errors():
    claim = refs.catalog(evidence())['claims'][0]['id']
    entries = [entry(source_refs=[claim]), *[entry() for _ in range(4)]]
    result = refs.decode(wire(payload(mechanism_review=entries)), evidence())
    assert result['conclusion'] is result['resolution'] is None
    assert result['error']['code'] == 'invalid_entry_count'
    issues = result['error']['reference_repair']['issues']
    assert [row['code'] for row in issues] == ['review_entry_count', 'claim_id_used_as_source']
    assert issues[0]['actual'] == 5 and issues[1]['value'] == claim


def test_advisory_catalog_failure_preserves_original_structure_rejection(monkeypatch):
    raw = wire(payload(mechanism_review=[]))
    def broken(_):
        raise ValueError('unavailable evidence')
    monkeypatch.setattr(refs, 'catalog', broken)
    result = refs.decode(raw, None)
    assert result == {'conclusion': None, 'resolution': None,
                      'error': {'code': 'invalid_entry_count', 'path': 'mechanism_review'}}


@pytest.mark.parametrize('changes', [
    {'source_refs': [{}]}, {'source_refs': {'malformed': []}},
    {'claim_ref': []}, {'source_refs': ['x' * 1000] * 100},
    {'source_refs': [None, True, [], {}]}, {'assessment': {}, 'evidence_kind': []},
])
def test_bad_provider_fields_produce_bounded_safe_diagnostics(changes):
    result = refs.decode(wire(payload(mechanism_review=[entry(**changes)])), evidence())
    assert result['conclusion'] is result['resolution'] is None
    if 'reference_repair' in result['error']:
        details = result['error']['reference_repair']
        assert len(details['issues']) <= 16
        assert len(json.dumps(details)) < 12000
        assert all(len(issue.get('value', '')) <= 128 for issue in details['issues'])


def test_diagnostics_are_bounded_for_many_independent_defects():
    prepared = refs.catalog(evidence())
    bad = entry(claim_ref='wrong', source_refs=['wrong'] * 100)
    details = refs._reference_repair_context({'mechanism_review': [bad] * 100}, prepared)
    assert len(details['issues']) == 16
    assert details['uninspected_or_omitted'] > 0
    assert len(json.dumps(details)) < 12000


@pytest.mark.parametrize('kind', ['source_effect', 'unavailable'])
def test_honest_unknown_option_still_passes_unchanged_authoritative_binding(kind):
    raw = wire(payload(mechanism_review=[entry(
        evidence_kind=kind, assessment='unknown', source_refs=[])]))
    result = refs.decode(raw, evidence())
    assert result['error'] is None
    assert result['conclusion'].mechanism_review[0]['assessment'] == 'unknown'


def test_enriched_diagnostics_keep_one_correction_and_both_charges():
    c, model, events, charges = setup([response(bad_ref=True), response(bad_ref=True)])
    c.complete_step()
    assert len(model.calls) == c.aux_calls == 2
    assert charges == [.03, .03]
    assert c.conclusions['r000001a1']['status'] == 'invalid'
    assert not c.knowledge.lessons
    rejected = [event for event in events if event['phase'] == 'conclusion_rejected']
    assert len(rejected) == 2
    assert all(event['diagnostic']['reference_repair']['allowed_claim_ids'] for event in rejected)
