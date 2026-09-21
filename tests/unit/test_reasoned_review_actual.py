"""Reasoned review decisions over authored receipts, claims and forecast evidence."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from review_evidence_fixture import GROUP

from sciel.hypothesis import comparison_review, grounded_review, reasoned_review
from sciel.knowledge_base.base import _snapshot
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha

ROOT = Path(__file__).resolve().parents[2]
REASON = ('I cannot yet specify a bounded edit that distinguishes these untested '
          'explanations using the available measurements.')
WRITER = dict(phase='reflection', reference=2, raw_response_sha256='d' * 64)


@pytest.fixture(scope='module')
def actual():
    from review_evidence_fixture import reasoned_example
    return reasoned_example()

def proposal(actual, *, target=False):
    result = copy.deepcopy(actual.proposal)
    result['next_hypothesis_reason'] = REASON
    if target:
        result['next_hypothesis'] = dict(schema='sciel-comparison-target/1',
            base_source_sha256=actual.group['challenger_source_sha256'],
            comparator_source_sha256=actual.group['challenger_source_sha256'],
            change='Change only the declared grid resolution under the same scaffold.',
            quality_gate='paired_raw_exact_zero_loss_strict_v1', diagnostics=[])
        result['next_hypothesis_reason'] = (
            'The mixed evidence motivates testing one bounded grid change; '
            'this does not assert that it will pass the unchanged quality gate.')
    return result


def build(actual, supplied):
    return reasoned_review.build_card(supplied, actual.group, WRITER,
        records=actual.records, seen=actual.seen)


@pytest.fixture(scope='module')
def card(actual):
    return build(actual, proposal(actual))


def test_saved_null_response_stays_exactly_v2_and_is_not_silently_upgraded(actual):
    old = grounded_review.build_card(actual.proposal, actual.group, actual.old_card['writer'],
        records=actual.records, seen=actual.seen)
    assert canonical_json(old) == canonical_json(actual.old_card)
    assert len(actual.proposal['claims']) == 10
    with pytest.raises(ValueError, match='invalid proposal fields'):
        reasoned_review.validate_proposals([actual.proposal])
    with pytest.raises(ValueError, match='invalid proposal fields'):
        comparison_review.build_cards([actual.proposal], [actual.group], WRITER,
            version=3, records=actual.records, seen=actual.seen)
    with pytest.raises(ValueError, match='invalid proposal fields'):
        grounded_review.validate_proposals([proposal(actual)])


def test_v2_source_contract_files_remain_byte_identical():
    for path, expected in {
        'src/sciel/hypothesis/grounded_review.py': '6fba86f759c42211d2c064ac0ac84fb516b2e13e926bb7ae5cadf73e4e44d7f2',
        'src/sciel/hypothesis/comparison_targets.py': '98eb9a0ef1f09d9d2abb089ca88c3015e6349501e687345ea295b32b6538c3b1',
    }.items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected


@pytest.mark.parametrize('target', [False, True])
def test_explicit_decision_is_detached_inert_and_does_not_change_facts(actual, target):
    supplied = proposal(actual, target=target)
    before = canonical_json([supplied, actual.group, actual.records, actual.seen])
    cards = comparison_review.build_cards([supplied], [actual.group], WRITER,
        version=3, records=actual.records, seen=actual.seen)
    result = cards[actual.group['id']]
    assert result['schema'] == reasoned_review.CARD_SCHEMA
    assert result['host_facts'] == actual.old_card['host_facts']
    assert result['origin_snapshot'] == actual.old_card['origin_snapshot']
    assert result['unverified_interpretation']['next_hypothesis'] == supplied['next_hypothesis']
    assert result['unverified_interpretation']['next_hypothesis_reason'] == supplied['next_hypothesis_reason']
    assert comparison_review.validate_cards(cards, [actual.group], seen=actual.seen) == cards
    assert set(result) == set(actual.old_card)
    assert not {'generation_allowed', 'quality_confirmed', 'causal_proof', 'semantic_gate_passed'} & set(result)
    assert 'authorizes generation' in reasoned_review.RULES
    result['unverified_interpretation']['next_hypothesis_reason'] = 'changed detached result'
    assert canonical_json([supplied, actual.group, actual.records, actual.seen]) == before


@pytest.mark.parametrize('reason', [None, True, 1, 1.2, float('nan'), [], {}, '', ' \n\t ', 'x' * 1025])
def test_reason_is_required_bounded_nonblank_text(actual, reason):
    supplied = proposal(actual)
    supplied['next_hypothesis_reason'] = reason
    with pytest.raises(ValueError, match='next hypothesis reason'):
        reasoned_review.validate_proposals([supplied])


def test_reason_limit_counts_characters_and_validation_detaches(actual):
    supplied = proposal(actual)
    supplied['next_hypothesis_reason'] = '\u0436' * 1024
    result = reasoned_review.validate_proposals([supplied])
    assert result == [supplied]
    result[0]['claims'][0]['value'] = 'changed'
    assert supplied['claims'][0]['value'] == 'true'


def test_whole_list_budget_counts_reasons_in_addition_to_valid_v2_projection(actual):
    rows = []
    for index in range(4):
        row = proposal(actual)
        row.update(comparison_id=f'comparison:synthetic{index}', conjecture='c' * 1024,
                   limitations='l' * 1950, claims=[dict(fact_id='quality:all_valid', value='true')],
                   next_hypothesis_reason='r' * 1024)
        rows.append(row)
    base = [{k: v for k, v in row.items() if k != 'next_hypothesis_reason'} for row in rows]
    assert len(canonical_json(base)) <= reasoned_review.MAX_PROPOSAL_CHARS < len(canonical_json(rows))
    assert grounded_review.validate_proposals(base) == base
    with pytest.raises(ValueError, match='character limit'):
        reasoned_review.validate_proposals(rows)


@pytest.mark.parametrize('damage', ['wrong_claim', 'duplicate_claim', 'foreign_base', 'foreign_comparator',
    'exact_only', 'epsilon', 'unsupported_diagnostic', 'false_group_hash', 'extra_field'])
def test_reason_cannot_bypass_unchanged_fact_source_and_target_gates(actual, damage):
    supplied = proposal(actual, target=True)
    if damage == 'wrong_claim':
        supplied['claims'][0]['value'] = 'false'
    elif damage == 'duplicate_claim':
        supplied['claims'].append(copy.deepcopy(supplied['claims'][0]))
    elif damage.startswith('foreign_'):
        supplied['next_hypothesis'][damage.removeprefix('foreign_') + '_source_sha256'] = '0' * 64
    elif damage == 'exact_only':
        supplied['next_hypothesis']['quality_gate'] = 'exact_only'
    elif damage == 'epsilon':
        supplied['next_hypothesis']['epsilon'] = 1e-6
    elif damage == 'unsupported_diagnostic':
        supplied['next_hypothesis']['diagnostics'] = [dict(observable='helper_elapsed', direction='decrease')]
    elif damage == 'false_group_hash':
        supplied['comparison_sha256'] = '0' * 64
    else:
        supplied['generation_allowed'] = True
    with pytest.raises(ValueError):
        build(actual, supplied)


@pytest.mark.parametrize('damage', ['missing', 'blank', 'oversized', 'schema_downgrade',
    'host_fact', 'origin', 'writer'])
def test_retained_card_damages_are_rejected(actual, card, damage):
    bad = copy.deepcopy(card)
    if damage == 'missing':
        bad['unverified_interpretation'].pop('next_hypothesis_reason')
    elif damage in ('blank', 'oversized'):
        bad['unverified_interpretation']['next_hypothesis_reason'] = '' if damage == 'blank' else 'r' * 1025
    elif damage == 'schema_downgrade':
        bad['schema'] = grounded_review.CARD_SCHEMA
    elif damage == 'host_fact':
        bad['host_facts']['facts']['quality:all_valid'] = 'false'
    elif damage == 'origin':
        bad['origin_snapshot']['hypothesis']['restart_predictions']['expected']['result_changed'] = True
    else:
        bad['writer']['raw_response_sha256'] = False
    with pytest.raises(ValueError):
        comparison_review.validate_cards({actual.group['id']: bad}, [actual.group], seen=actual.seen)


def test_reason_is_unverified_and_card_validation_is_not_raw_writer_authentication(actual, card):
    bad_reason = 'This proves a universal causal optimizer improvement.'
    syntactically_valid = copy.deepcopy(card)
    syntactically_valid['unverified_interpretation']['next_hypothesis_reason'] = bad_reason
    result = reasoned_review.validate_card(syntactically_valid, actual.group, seen=actual.seen)
    assert result['unverified_interpretation']['next_hypothesis_reason'] == bad_reason
    assert 'next_hypothesis_reason is also unverified model prose' in reasoned_review.presentation_card(result)['rules']
    # Caller-owned raw journal authentication must reject an altered valid reason;
    # local validation has no journal from which to establish prose authorship.
    assert result['writer'] == card['writer']


def test_reason_is_wholly_withheld_without_reallocating_its_reserve(actual, card):
    cards = {actual.group['id']: card}
    before = canonical_json(cards)
    views = [comparison_review.context(cards, [actual.group], 'p1', 10000,
        text_enabled=flag, seen=actual.seen) for flag in (True, False)]
    on, off = views
    assert on['presented_count'] == off['presented_count'] == 1
    assert on['reserved_chars'] == off['reserved_chars'] == on['rendered_chars'] + 1
    assert on['rendered_chars'] > off['rendered_chars']
    assert on['cards'][0]['unverified_interpretation']['next_hypothesis_reason'] == REASON
    assert off['cards'][0]['unverified_interpretation'] is None
    assert REASON not in canonical_json(off)
    common = []
    for flag, view in zip((True, False), views):
        assert view['rendered_chars'] == len(canonical_json(view))
        assert comparison_review.validate_context(view, cards, [actual.group], 'p1',
            text_enabled=flag, seen=actual.seen) == view
        item = copy.deepcopy(view)
        item.pop('text_enabled')
        item.pop('rendered_chars')
        for row in item['cards']:
            row.pop('unverified_interpretation')
            row.pop('interpretation_omission')
        common.append(item)
    assert common[0] == common[1]
    assert canonical_json(cards) == before
    bad = copy.deepcopy(on)
    bad['cards'][0]['unverified_interpretation']['next_hypothesis_reason'] = 'substituted plausible reason'
    with pytest.raises(ValueError, match='differs from authoritative cards'):
        comparison_review.validate_context(bad, cards, [actual.group], 'p1',
            text_enabled=True, seen=actual.seen)


def test_distinct_presentation_excludes_private_origin_and_is_detached(actual, card):
    view = reasoned_review.presentation_card(card)
    assert view['schema'] == 'sciel-comparison-review-grounded-presentation/2'
    assert 'origin_snapshot' not in view
    assert view['host_facts']['origin_sha256'] == actual.seen['r000002a1']
    view['unverified_interpretation']['next_hypothesis_reason'] = 'mutated'
    assert card['unverified_interpretation']['next_hypothesis_reason'] == REASON


def test_fact_catalog_is_unchanged_and_reason_rules_add_no_observations(actual):
    kwargs = dict(records=actual.records, seen=actual.seen)
    assert reasoned_review.source_catalog(actual.group, **kwargs) == grounded_review.source_catalog(actual.group, **kwargs)
    old = grounded_review.source_block([actual.group], **kwargs)
    new = reasoned_review.source_block([actual.group], **kwargs)
    assert new[len(reasoned_review.RULES):] == old[len(grounded_review.RULES):]


def test_new_schema_keeps_nullable_target_exact_and_does_not_mutate_v2_schema():
    old = grounded_review.json_schema()
    before = canonical_json(old)
    new = reasoned_review.json_schema()
    assert new['items']['properties']['next_hypothesis'] == old['items']['properties']['next_hypothesis']
    assert new['items']['properties']['next_hypothesis']['type'] == ['object', 'null']
    assert set(new['items']['required']) == set(old['items']['required']) | {'next_hypothesis_reason'}
    assert new['items']['additionalProperties'] is False
    new['items']['properties']['claims']['description'] = 'mutated'
    assert canonical_json(old) == canonical_json(grounded_review.json_schema()) == before
    identity = reasoned_review.identity()
    assert identity['grounded_policy'] == grounded_review.identity()
    assert identity['source_sha256'] == hashlib.sha256(Path(reasoned_review.__file__).read_bytes()).hexdigest()


def test_all_three_retained_versions_rebuild_without_upgrade(actual, card):
    others = [g for g in actual.groups if g['id'] != actual.group['id']]
    assert len(others) == 2
    legacy = dict(comparison_id=others[0]['id'], comparison_sha256=canonical_sha(others[0]),
        interpretation='This is a retained legacy interpretation.', limitations='No causal proof.', next_hypothesis=None)
    v2 = dict(comparison_id=others[1]['id'], comparison_sha256=canonical_sha(others[1]),
        claims=[dict(fact_id='quality:outcome', value=others[1]['derived']['outcome'])],
        conjecture='An untested possibility.', limitations='No causal proof.', next_hypothesis=None)
    cards = {**comparison_review.build_cards([legacy], actual.groups, WRITER),
             **comparison_review.build_cards([v2], actual.groups, WRITER, version=2,
                 records=actual.records, seen=actual.seen), actual.group['id']: copy.deepcopy(card)}
    before = canonical_json(cards)
    assert canonical_json(comparison_review.validate_cards(cards, actual.groups, seen=actual.seen)) == before
    assert {c['schema'] for c in cards.values()} == {
        comparison_review.CARD_SCHEMA, grounded_review.CARD_SCHEMA, reasoned_review.CARD_SCHEMA}
    assert 'next_hypothesis_reason' not in cards[others[0]['id']]['unverified_interpretation']
    assert 'next_hypothesis_reason' not in cards[others[1]['id']]['unverified_interpretation']
