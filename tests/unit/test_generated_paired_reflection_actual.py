"""Authored rejection preserves complete raw text, named facts and settled cost."""
from decimal import Decimal
import json

import pytest

from sciel.engine.external_reflection import ExternalReflectionRejected
from sciel.hypothesis.parsing import parse_reflection
from paired_reflection_rejection_fixture import build, sha


@pytest.fixture(scope='module')
def saved(tmp_path_factory):
    return build(tmp_path_factory.mktemp('overlong-reflection'), 'overlong')


def test_exact_raw_response_rejects_overlong_reason_without_repair(saved):
    before = saved.raw.encode()
    result = next(row for row in saved.events if row.get('phase') == 'aux_result')
    assert saved.provider.choices[0].message.content.encode() == before
    assert result['response'].encode() == before and result['response_sha256'] == saved.raw_sha
    assert result['response_truncated'] is False and result['structured_response_usable'] is True
    payload = json.loads(saved.raw)
    assert [len(row['next_hypothesis_reason']) for row in payload['comparison_reviews']] == [826, 905, 1027]
    assert payload['lessons'] == payload['retire_ids'] == []
    assert all(row['next_hypothesis'] is None for row in payload['comparison_reviews'])
    assert parse_reflection(saved.raw, comparison_reviews=True, comparison_review_version=3) is None
    assert saved.response['text'].encode() == before and sha(before) == saved.raw_sha
    assert saved.events[-1]['error_diagnostic'] == dict(code='text_too_long',
        path='comparison_reviews[2].next_hypothesis_reason', actual=1027, maximum=1024)


def test_delivered_catalog_separates_metadata_claims_from_matching_named_facts(saved):
    assert sha(saved.plan['prompt']) == saved.plan['prompt_sha256']
    request = saved.client.chat.completions.create.call_args.kwargs
    assert request['messages'] == [dict(role='user', content=saved.plan['prompt'])]
    unsupported, matched = [], []
    for row in json.loads(saved.raw)['comparison_reviews']:
        catalog = saved.catalogs[row['comparison_id']]
        assert row['comparison_sha256'] == catalog['comparison_sha256']
        assert len(row['claims']) == len({claim['fact_id'] for claim in row['claims']}) == 11
        for claim in row['claims']:
            key = claim['fact_id']
            if key not in catalog['facts']:
                assert key in ('comparison_id', 'comparison_sha256')
                assert claim['value'] == row[key] == catalog[key]
                unsupported.append(key)
            else:
                assert claim['value'] == catalog['facts'][key]
                matched.append(key)
    assert len(unsupported) == 6 and len(matched) == 27


def test_rejection_keeps_one_settled_call_without_admission_or_followup(saved):
    summary = saved.ledger.snapshot()
    assert summary['calls'] == 1 and Decimal(summary['known_cost_usd']) == Decimal('.01')
    assert Decimal(summary['reserved_or_unknown_usd']) == 0 and summary['halted'] is None
    assert saved.client.chat.completions.create.call_count == 1
    assert [row.get('phase') for row in saved.events] == [None, 'generated_paired_reflection_start',
        'generated_paired_reflection_context', 'aux_start', 'aux_result', 'generated_paired_reflection']
    assert saved.events[:4] == [json.loads(line) for line in saved.plan['journal_prefix'].splitlines()]
    terminal = saved.events[-1]
    assert terminal['status'] == 'invalid'
    for key in ('comparison_review_ids', 'new_lesson_ids', 'retired_lesson_ids'):
        assert terminal[key] == []
    assert terminal['new_ordinary_outcomes'] == terminal['new_search_attempts'] == 0
    # Admission replay is pure: it neither sends a request nor alters transport accounting.
    with pytest.raises(ExternalReflectionRejected):
        saved.job.admit(saved.plan, saved.response, manifest_sha256='a' * 64,
                        journal_prefix=saved.plan['journal_prefix'])
    assert saved.ledger.snapshot() == summary
    assert saved.client.chat.completions.create.call_count == 1
