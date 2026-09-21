"""Replay one saved conclusion without executing candidate/provider code."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

from sciel.hypothesis import conclusion_refs as refs


def saved():
    return json.loads((Path(__file__).parents[1] / 'fixtures' /
                       'conclusion_refs_g82.json').read_text())


def test_saved_v19_catalog_and_response_remain_explicitly_reproducible():
    row = saved()
    evidence, response = row['evidence'], row['response_v1']
    assert hashlib.sha256(response.encode()).hexdigest() == row['provenance']['response_sha256']
    assert refs.catalog(evidence, schema=refs.LEGACY_SCHEMA) == row['catalog_v1']
    legacy = refs.decode(response, evidence, schema=refs.LEGACY_SCHEMA)
    assert legacy['error'] is None
    assert legacy['resolution']['schema'] == refs.LEGACY_SCHEMA
    quoted = [item['source_quotes'] for item in legacy['conclusion'].mechanism_review
              if item['evidence_kind'] == 'source_effect']
    assert len(quoted) == 3
    assert all(items == [{'field': 'change_excerpt', 'text': '@@ -17,6 +18,6 @@\n'}]
               for items in quoted)


def test_saved_control_only_support_cannot_enter_the_new_catalog_or_decoder():
    row = saved()
    before = deepcopy(row)
    evidence = row['evidence']
    prepared = refs.catalog(evidence)
    assert prepared['schema'] == refs.SCHEMA != refs.LEGACY_SCHEMA
    assert 's74e1b74a40731ef690c4' not in {item['id'] for item in prepared['sources']}
    for item in prepared['sources']:
        source = refs.source_fields(evidence)[item['field']]
        assert item['text'] == source[item['start']:item['end']]
        assert not item['text'].startswith(('@@ ', '--- evaluated_parent', '+++ candidate'))
    # This real edit was present in the saved evidence but absent from v1's
    # catalog. Removing control spans before sampling makes it selectable.
    assert not any('+    num_starts = 10' in item['text'] for item in row['catalog_v1']['sources'])
    assert any('+    num_starts = 10' in item['text'] for item in prepared['sources'])
    rejected = refs.decode(row['response_v1'], evidence)
    assert rejected['conclusion'] is None and rejected['resolution'] is None
    assert rejected['error']['code'] == 'invalid_source_references'
    assert row == before


def test_saved_measurement_can_still_support_a_forecast_without_code_citations():
    row = saved()
    result = refs.decode(row['response_v1'], row['evidence'], schema=refs.LEGACY_SCHEMA)
    assert result['conclusion'].verdict == 'CONFIRMED'
    # A new response can retain the numerical comparison while declining a
    # mechanism claim. This does not migrate or rewrite the historical answer.
    wire = {'verdict': 'CONFIRMED', 'explanation': 'The measured delta is inside the forecast.',
            'lessons': [], 'mechanism_review': [{
                'claim_ref': None, 'source_refs': [], 'assessment': 'directly_supported',
                'direct_effect': 'The observed delta is within the declared interval.',
                'untested_downstream': 'The causal effect of the added starts is unverified.',
                'evidence_kind': 'forecast'}]}
    decoded = refs.decode(json.dumps(wire), row['evidence'])
    assert decoded['error'] is None
    assert decoded['resolution']['schema'] == refs.SCHEMA
    assert decoded['conclusion'].lessons == []
