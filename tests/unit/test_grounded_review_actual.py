"""Grounded review claims over authored mixed-sign receipts and forecast evidence."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from review_evidence_fixture import GROUP

from sciel.hypothesis import grounded_review
from sciel.knowledge_base.base import _snapshot
from sciel.knowledge_base.catalogs import canonical_sha

ROOT = Path(__file__).resolve().parents[2]
SEEDS = (541090050, 3513246259, 4186712104)
NAMES = ('effective_locations_changed', 'result_changed', 'accepted_interval_gain')
EXPECTED_VERDICTS = {
    541090050: ('CONFIRMED', 'REFUTED', 'REFUTED'),
    3513246259: ('CONFIRMED', 'INCONCLUSIVE', 'CONFIRMED'),
    4186712104: ('CONFIRMED', 'REFUTED', 'CONFIRMED'),
}


@pytest.fixture(scope='module')
def actual():
    from review_evidence_fixture import grounded_example
    return grounded_example()

@pytest.fixture(scope='module')
def catalog(actual):
    return grounded_review.source_catalog(actual.group, records=actual.records, seen=actual.seen)


def proposed(actual, facts):
    return dict(comparison_id=actual.group['id'], comparison_sha256=canonical_sha(actual.group),
        claims=[dict(fact_id=key, value=value) for key, value in facts.items()],
        conjecture='A bounded grid-resolution change is an untested possible next experiment.',
        limitations='One candidate and three seeds do not isolate a causal mechanism.',
        next_hypothesis=dict(schema='sciel-comparison-target/1',
            base_source_sha256=actual.group['challenger_source_sha256'],
            comparator_source_sha256=actual.group['challenger_source_sha256'],
            change='Change the grid resolution while retaining the same dual-order helper.',
            quality_gate='paired_raw_exact_zero_loss_strict_v1', diagnostics=[]))


def build(actual, proposal, *, records=None, seen=None):
    return grounded_review.build_card(proposal, actual.group, actual.writer,
        records=actual.records if records is None else records,
        seen=actual.seen if seen is None else seen)


@pytest.fixture(scope='module')
def card(actual, catalog):
    return build(actual, proposed(actual, catalog['facts']))


def test_actual_named_catalog_preserves_both_metric_axes_and_all_nine_verdicts(actual, catalog):
    facts = catalog['facts']
    assert facts['quality:outcome'] == 'mixed'
    assert facts['quality:all_valid'] == 'true'
    assert facts['quality:all_no_loss'] == 'false'
    assert facts['metric:raw_object'] == 'candidate_raw_radii'
    assert facts['metric:certified_object'] == 'radius_decreased_certified_packing'
    for seed, raw, certified in zip(SEEDS, ('negative', 'positive', 'negative'),
                                     ('positive', 'negative', 'negative')):
        assert facts[f'pair:{seed}:raw_direction'] == raw
        assert facts[f'pair:{seed}:certified_direction'] == certified
        assert facts[f'pair:{seed}:measured_no_loss'] == 'false'
        for name, verdict in zip(NAMES, EXPECTED_VERDICTS[seed]):
            assert facts[f'restart:{seed}:{name}:verdict'] == verdict
    assert catalog['origin_sha256'] == actual.seen['r000002a1']
    assert len(facts) == 23 and all(type(v) is str for v in facts.values())


@pytest.mark.parametrize('seed,name,wrong', [
    (3513246259, 'effective_locations_changed', 'REFUTED'),
    (4186712104, 'effective_locations_changed', 'REFUTED'),
    (541090050, 'result_changed', 'CONFIRMED'),
    (4186712104, 'result_changed', 'CONFIRMED'),
])
def test_each_of_four_actual_axis_errors_is_rejected(actual, catalog, seed, name, wrong):
    proposal = proposed(actual, catalog['facts'])
    ident = f'restart:{seed}:{name}:verdict'
    next(claim for claim in proposal['claims'] if claim['fact_id'] == ident)['value'] = wrong
    with pytest.raises(ValueError, match='claim differs from host fact'):
        build(actual, proposal)


def test_actual_all_three_sign_disagreements_cannot_be_encoded_as_true_claims(actual, catalog):
    proposal = proposed(actual, catalog['facts'])
    # The first two seeds do disagree. The third has two negative metrics;
    # declaring its certified direction positive is the concrete false cell.
    claim = next(c for c in proposal['claims'] if c['fact_id'] == 'pair:4186712104:certified_direction')
    assert claim['value'] == 'negative'
    claim['value'] = 'positive'
    with pytest.raises(ValueError, match='claim differs from host fact'):
        build(actual, proposal)


def test_same_packing_representation_claim_rejected(actual, catalog):
    proposal = proposed(actual, catalog['facts'])
    next(c for c in proposal['claims'] if c['fact_id'] == 'metric:certified_object')['value'] = 'candidate_raw_radii'
    with pytest.raises(ValueError, match='claim differs from host fact'):
        build(actual, proposal)


def test_missing_origin_is_explicit_unknown_not_false_or_reused_feedback(actual):
    missing = grounded_review.source_catalog(actual.group, records={}, seen=actual.seen)
    assert missing['origin_sha256'] is None
    assert missing['restart_feedback']['unavailable_reason'] == 'original_hypothesis_unavailable'
    assert missing['facts']['quality:outcome'] == 'mixed'
    for seed in SEEDS:
        for name in NAMES:
            assert missing['facts'][f'restart:{seed}:{name}:verdict'] == 'INCONCLUSIVE'
    proposal = proposed(actual, missing['facts'])
    unknown_card = build(actual, proposal, records={})
    assert unknown_card['origin_snapshot'] is None
    assert grounded_review.validate_card(unknown_card, actual.group, seen=actual.seen) == unknown_card
    next(c for c in proposal['claims'] if c['fact_id'] ==
         'restart:541090050:effective_locations_changed:verdict')['value'] = 'CONFIRMED'
    with pytest.raises(ValueError, match='claim differs from host fact'):
        build(actual, proposal, records={})


@pytest.mark.parametrize('mode', ['none', 'missing_origin', 'wrong_digest'])
def test_retained_actual_origin_requires_lifetime_seen_authentication(actual, card, mode):
    seen = copy.deepcopy(actual.seen)
    if mode == 'none':
        seen = None
    elif mode == 'missing_origin':
        seen.pop('r000002a1')
    else:
        seen['r000002a1'] = '0' * 64
    with pytest.raises(ValueError, match='lifetime observation digest'):
        grounded_review.validate_card(copy.deepcopy(card), actual.group, seen=seen)


@pytest.mark.parametrize('mode', ['origin_forecast', 'host_verdict', 'host_origin_hash', 'missing_origin_snapshot'])
def test_actual_saved_card_tamper_rejected(actual, card, mode):
    damaged = copy.deepcopy(card)
    if mode == 'origin_forecast':
        expected = damaged['origin_snapshot']['hypothesis']['restart_predictions']['expected']
        expected['result_changed'] = not expected['result_changed']
    elif mode == 'host_verdict':
        damaged['host_facts']['facts']['restart:541090050:result_changed:verdict'] = 'CONFIRMED'
    elif mode == 'host_origin_hash':
        damaged['host_facts']['origin_sha256'] = '0' * 64
    else:
        damaged.pop('origin_snapshot')
    with pytest.raises(ValueError):
        grounded_review.validate_card(damaged, actual.group, seen=actual.seen)


def test_actual_origin_persists_without_recent_record_and_is_not_presented_as_extra_evidence(actual, card):
    assert card['origin_snapshot'] == actual.records['r000002a1']
    assert grounded_review.validate_card(card, actual.group, seen=actual.seen) == card
    view = grounded_review.presentation_card(card)
    assert 'origin_snapshot' not in view and view['host_facts']['origin_sha256'] == actual.seen['r000002a1']
    assert view['unverified_interpretation'] == card['unverified_interpretation']
    assert card['origin_snapshot'] == actual.records['r000002a1']


@pytest.mark.parametrize('field', ['base_source_sha256', 'comparator_source_sha256'])
def test_prospective_contrast_must_reference_this_actual_group_source(actual, catalog, field):
    proposal = proposed(actual, catalog['facts'])
    proposal['next_hypothesis'][field] = '0' * 64
    with pytest.raises(ValueError):
        build(actual, proposal)


@pytest.mark.parametrize('mode', ['exact_only_gate', 'near_zero', 'old_text_falsifier'])
def test_actual_bad_quality_criterion_cannot_replace_host_gate(actual, catalog, mode):
    proposal = proposed(actual, catalog['facts'])
    target = proposal['next_hypothesis']
    if mode == 'exact_only_gate':
        target['quality_gate'] = 'at_least_two_exact_positive_and_one_near_zero'
    elif mode == 'near_zero':
        target['epsilon'] = 1e-6
    else:
        proposal['next_hypothesis'] = copy.deepcopy(actual.old_proposal['next_hypothesis'])
    with pytest.raises(ValueError):
        grounded_review.validate_proposals([proposal])


def test_checked_actual_claims_do_not_certify_false_free_prose(actual, catalog):
    proposal = proposed(actual, catalog['facts'])
    false_sentence = 'The signs differ between raw and exact measurements for all three seeds.'
    assert false_sentence in actual.old_proposal['interpretation']
    proposal['conjecture'] = false_sentence
    syntactically_accepted = build(actual, proposal)
    assert syntactically_accepted['unverified_interpretation']['conjecture'] == false_sentence
    assert syntactically_accepted['host_facts']['facts']['pair:4186712104:raw_direction'] == 'negative'
    assert syntactically_accepted['host_facts']['facts']['pair:4186712104:certified_direction'] == 'negative'
    assert 'semantic_gate_passed' not in syntactically_accepted
    assert 'unverified' in grounded_review.presentation_card(syntactically_accepted)['rules']


def test_old_paid_v1_reply_is_never_silently_upgraded(actual):
    with pytest.raises(ValueError, match='invalid proposal fields'):
        grounded_review.validate_proposals([actual.old_proposal])
