"""Deterministic comparison keeps evidence and shared identity references."""
import copy
import json

import pytest

import conftest


def rows(suffix):
    first, second = 'first-' + suffix, 'second-' + suffix
    return [
        {'event': 'evaluation', 'evaluation_id': first, 'score': .25, 'cost': .1,
         'timestamp': 'wall-' + suffix, 'timing': 3.,
         'cases': {'combined_score': .25}},
        {'event': 'evaluation', 'evaluation_id': second, 'score': .5, 'cost': None,
         'cases': {'combined_score': .5}},
        {'event': 'iteration', 'meta': {
            'evaluation_ids': [first, second, first],
            'nested': [{'evaluation_id': second, 'evaluation_ids': [first]}],
            'generation_cost': None, 'accounted_cost': .1, 'unknown_cost_calls': 1,
            'measurement': {'score': .5, 'timing': 7., 'path': 'logical-path'},
            'receipt_sha256': 'fixed-content-hash',
         }},
    ]


def normalize(events):
    identities = {}
    return [conftest.strip_volatile(json.dumps(event), identities) for event in events]


def test_plural_references_use_scalar_namespace_without_mutating_evidence():
    first, second = rows('a'), rows('b')
    original = copy.deepcopy(first)
    normalized = normalize(first)
    assert normalized == normalize(second)
    assert first == original
    ids = [normalized[0]['evaluation_id'], normalized[1]['evaluation_id']]
    assert ids[0] != ids[1]
    assert normalized[-1]['meta']['evaluation_ids'] == [*ids, ids[0]]
    assert normalized[-1]['meta']['nested'][0] == {
        'evaluation_id': ids[1], 'evaluation_ids': [ids[0]]}
    assert normalized[-1]['meta']['measurement'] == first[-1]['meta']['measurement']
    assert [event['cost'] for event in normalized[:2]] == [.1, None]


@pytest.mark.parametrize('mutation', [
    lambda r: r[-1]['meta'].update(evaluation_ids=['first-a', 'first-a', 'first-a']),
    lambda r: r[-1]['meta'].update(evaluation_ids=['second-a', 'first-a', 'first-a']),
    lambda r: r[-1]['meta'].update(evaluation_ids=['first-a', 'second-a']),
    lambda r: r[-1]['meta'].update(evaluation_ids=['first-a', 'second-a', 'unseen-a']),
    lambda r: r[-1]['meta']['nested'][0].update(evaluation_id='unseen-a'),
    lambda r: r[-1]['meta']['nested'][0].update(evaluation_ids=['second-a']),
    lambda r: r[0].update(cost=.2),
    lambda r: r[1].update(cost=0.),
    lambda r: r[-1]['meta'].update(generation_cost=0.),
    lambda r: r[-1]['meta'].update(accounted_cost=.2),
    lambda r: r[-1]['meta'].update(unknown_cost_calls=0),
    lambda r: r[-1]['meta']['measurement'].update(score=.6),
    lambda r: r[-1]['meta']['measurement'].update(timing=8.),
    lambda r: r[-1]['meta'].update(receipt_sha256='changed-content-hash'),
    lambda r: r[1]['cases'].update(combined_score=.6),
])
def test_changed_alias_or_substantive_evidence_remains_detectable(mutation):
    baseline = rows('a')
    changed = copy.deepcopy(baseline)
    mutation(changed)
    assert normalize(changed) != normalize(baseline)


@pytest.mark.parametrize('value', ['', None, 7, {}, [], ['first-a', None, '']])
def test_non_identifier_values_are_not_replaced_or_discarded(value):
    event = {'event': 'iteration', 'meta': {'evaluation_ids': value}}
    normalized = normalize([event])[0]
    expected = copy.deepcopy(value)
    if value == ['first-a', None, '']:
        expected[0] = '<evaluation_id:1>'
    assert normalized['meta']['evaluation_ids'] == expected


def test_read_journal_preserves_links_across_lines_and_raw_file(tmp_path):
    raw = ''.join(json.dumps(row) + '\n' for row in rows('a'))
    path = tmp_path / 'sciel_events.jsonl'
    path.write_text(raw)
    assert conftest.read_journal(tmp_path) == normalize(rows('a'))
    assert path.read_text() == raw
