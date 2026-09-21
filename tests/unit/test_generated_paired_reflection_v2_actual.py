"""Authored reflection passes shape but fails group-local source binding."""
from decimal import Decimal
import json

import pytest

from sciel.hypothesis.parsing import parse_reflection
from paired_reflection_rejection_fixture import build, sha


@pytest.fixture(scope='module')
def saved(tmp_path_factory):
    return build(tmp_path_factory.mktemp('foreign-target-reflection'), 'foreign_target')


def test_raw_response_is_shape_valid_with_bounded_reasons_and_matching_named_claims(saved):
    payload = json.loads(saved.raw)
    parsed = parse_reflection(saved.raw, comparison_reviews=True, comparison_review_version=3)
    assert parsed is not None and parsed.to_dict() == payload
    reviews = payload['comparison_reviews']
    assert [len(row['next_hypothesis_reason']) for row in reviews] == [489, 370, 358]
    assert [len(row['claims']) for row in reviews] == [10, 10, 9]
    for row in reviews:
        catalog = saved.catalogs[row['comparison_id']]
        assert row['comparison_sha256'] == catalog['comparison_sha256']
        ids = [claim['fact_id'] for claim in row['claims']]
        assert len(ids) == len(set(ids)) and not {'comparison_id', 'comparison_sha256'} & set(ids)
        assert all(catalog['facts'][claim['fact_id']] == claim['value'] for claim in row['claims'])
    assert saved.provider.choices[0].message.content == saved.raw
    result = next(row for row in saved.events if row.get('phase') == 'aux_result')
    assert result['response'] == saved.raw and result['response_sha256'] == sha(saved.raw)
    assert result['response_truncated'] is False and result['structured_response_usable'] is True


def test_control_is_in_global_wire_enum_but_outside_selected_comparison(saved):
    row, *remaining = json.loads(saved.raw)['comparison_reviews']
    assert all(item['next_hypothesis'] is None for item in remaining)
    catalog = saved.catalogs[row['comparison_id']]
    assert catalog['branch'] == 'on_off'
    pair = {catalog['comparator_source_sha256'], catalog['challenger_source_sha256']}
    target = row['next_hypothesis']
    wire = saved.plan['response_format']['json_schema']['schema']['properties'][
        'comparison_reviews']['items']['properties']['next_hypothesis']['properties']
    for field in ('base_source_sha256', 'comparator_source_sha256'):
        assert target[field] in wire[field]['enum']
    assert target['base_source_sha256'] == catalog['comparator_source_sha256']
    assert target['comparator_source_sha256'] == saved.component['root_parent']['source_sha256']
    assert target['comparator_source_sha256'] not in pair
    terminal = saved.events[-1]
    assert terminal['status'] == 'parsed' and terminal['error_diagnostic'] is None
    assert terminal['error_detail'] == 'comparison target: target references an unavailable source'
    assert terminal['comparison_review_ids'] == []


def test_semantic_rejection_preserves_settled_cost_and_exact_unrepaired_raw(saved):
    summary = saved.ledger.snapshot()
    assert summary['calls'] == 1 and Decimal(summary['known_cost_usd']) == Decimal('.01')
    assert Decimal(summary['reserved_or_unknown_usd']) == 0 and summary['halted'] is None
    assert saved.client.chat.completions.create.call_count == 1
    terminal = saved.events[-1]
    for key in ('comparison_review_ids', 'new_lesson_ids', 'retired_lesson_ids'):
        assert terminal[key] == []
    assert terminal['new_ordinary_outcomes'] == terminal['new_search_attempts'] == 0
    assert saved.response['text'] == saved.raw and sha(saved.raw) == saved.raw_sha
