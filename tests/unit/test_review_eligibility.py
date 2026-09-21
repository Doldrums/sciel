"""Exact host decisions, without models, evaluation, or semantic-truth claims."""
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from sciel.hypothesis import review_eligibility as eligibility
from sciel.hypothesis.models import Reflection
from sciel.knowledge_base.catalogs import canonical_sha
from tests.unit.test_comparison_review import cards_for, group


POLICY_SHA = 'a' * 64


@pytest.fixture(scope='module')
def card():
    paired = group()
    return cards_for([paired])[paired['id']]


def decision(card, **changes):
    result = dict(schema=eligibility.DECISION_SCHEMA, scope=eligibility.SCOPE,
        decision='accepted', card_sha256=canonical_sha(card),
        raw_response_sha256=card['writer']['raw_response_sha256'],
        **{key: card[key] for key in ('comparison_id', 'comparison_sha256',
                                     'comparator_source_sha256', 'challenger_source_sha256')},
        policy_sha256=POLICY_SHA, audit_sha256='b' * 64, criteria_sha256='c' * 64,
        issuer='synthetic unit-test host')
    result.update(changes)
    return result


def policy(*decisions):
    return dict(schema=eligibility.POLICY_SCHEMA, decisions=list(decisions))


def resolve(card, registry, **kwargs):
    return eligibility.decision_for(card, registry, policy_sha256=kwargs.get('policy_sha256', POLICY_SHA))


def test_empty_host_policy_is_valid_and_pending(card):
    assert eligibility.validate_policy(policy()) == policy()
    assert resolve(card, policy()) == {'status': 'pending', 'decision_sha256': None}


@pytest.mark.parametrize('status', ['accepted', 'rejected'])
def test_exact_decision_detached_and_side_effect_free(card, status):
    row = decision(card, decision=status)
    registry = policy(row)
    before = copy.deepcopy((card, registry))
    assert resolve(card, registry) == {'status': status, 'decision_sha256': canonical_sha(row)}
    checked = eligibility.validate_policy(registry)
    checked['decisions'][0]['issuer'] = 'changed'
    metadata = resolve(card, registry)
    metadata['status'] = 'changed'
    assert (card, registry) == before
    assert resolve(card, registry)['status'] == status


@pytest.mark.parametrize('status', ['accepted', 'rejected'])
def test_matched_old_policy_is_stale_with_traceable_decision(card, status):
    row = decision(card, decision=status, policy_sha256='f' * 64)
    assert resolve(card, policy(row)) == {'status': 'stale', 'decision_sha256': canonical_sha(row)}


@pytest.mark.parametrize('change', ['reason', 'writer', 'source', 'group', 'extra_field'])
def test_approval_never_transfers_to_revised_artifact(card, change):
    original = copy.deepcopy(card)
    changed = copy.deepcopy(card)
    if change == 'reason':
        changed['unverified_interpretation']['limitations'] += ' More uncertainty.'
    elif change == 'writer':
        changed['writer']['reference'] += 1
    elif change == 'source':
        changed['challenger_source_sha256'] = 'f' * 64
    elif change == 'group':
        changed['comparison_sha256'] = 'f' * 64
    else:
        changed['model_asserted_approval'] = True
    assert resolve(changed, policy(decision(original))) == {'status': 'pending', 'decision_sha256': None}
    assert card == original


@pytest.mark.parametrize('field', ['raw_response_sha256', 'comparison_id', 'comparison_sha256',
                                  'comparator_source_sha256', 'challenger_source_sha256'])
@pytest.mark.parametrize('status', ['accepted', 'rejected', 'stale'])
def test_matched_card_redundant_binding_mismatch_is_invalid(card, field, status):
    changes = {field: 'another-group' if field == 'comparison_id' else 'f' * 64}
    changes.update(decision='rejected' if status == 'rejected' else 'accepted')
    if status == 'stale':
        changes['policy_sha256'] = 'e' * 64
    with pytest.raises(ValueError, match='binding'):
        resolve(card, policy(decision(card, **changes)))


@pytest.mark.parametrize('value', [None, False, [], (), {}, {'schema': 'sciel-review-text-policy/2', 'decisions': []},
                                  {'schema': eligibility.POLICY_SCHEMA, 'decisions': [], 'approval': True}])
def test_invalid_policy_envelope(value):
    with pytest.raises(ValueError):
        eligibility.validate_policy(value)


@pytest.mark.parametrize('value', [None, False, {}, (), 'accepted'])
def test_decisions_must_be_a_list(value):
    with pytest.raises(ValueError):
        eligibility.validate_policy(dict(schema=eligibility.POLICY_SCHEMA, decisions=value))


@pytest.mark.parametrize('field', ['card_sha256', 'raw_response_sha256', 'comparison_sha256',
    'comparator_source_sha256', 'challenger_source_sha256', 'policy_sha256', 'audit_sha256', 'criteria_sha256'])
@pytest.mark.parametrize('value', [False, None, float('nan'), 'A' * 64, 'a' * 63, 'g' * 64])
def test_strict_digest_fields(card, field, value):
    with pytest.raises(ValueError):
        eligibility.validate_policy(policy(decision(card, **{field: value})))


@pytest.mark.parametrize('field,value', [
    ('schema', True), ('schema', 'sciel-review-text-decision/2'),
    ('scope', False), ('scope', 'generation'), ('decision', True), ('decision', 'pending'),
    ('decision', 'ACCEPTED'), ('issuer', False), ('issuer', ''), ('issuer', ' \t\n'),
    ('issuer', 'x' * 129), ('comparison_id', False), ('comparison_id', ''),
    ('comparison_id', 'x' * 129), ('comparison_id', 'two words'), ('comparison_id', '\x00id')])
def test_strict_non_digest_fields(card, field, value):
    with pytest.raises(ValueError):
        eligibility.validate_policy(policy(decision(card, **{field: value})))


def test_exact_decision_fields_and_object_type(card):
    row = decision(card)
    for changed in (None, [], False, dict(row, accepted=True), {k: v for k, v in row.items() if k != 'issuer'}):
        with pytest.raises(ValueError):
            eligibility.validate_policy(policy(changed))


def test_capacity_and_duplicate_card_decisions(card):
    rows = [decision(card, card_sha256=f'{i:064x}') for i in range(64)]
    assert len(eligibility.validate_policy(policy(*rows))['decisions']) == 64
    with pytest.raises(ValueError, match='decision list'):
        eligibility.validate_policy(policy(*rows, decision(card, card_sha256='f' * 64)))
    for second in (copy.deepcopy(rows[0]), dict(rows[0], decision='rejected'), dict(rows[0], policy_sha256='e' * 64)):
        with pytest.raises(ValueError, match='duplicate'):
            eligibility.validate_policy(policy(rows[0], second))


@pytest.mark.parametrize('value', [False, None, float('nan'), 'A' * 64, 'a'])
def test_current_policy_digest_required_even_without_matching_decision(card, value):
    with pytest.raises(ValueError):
        resolve(card, policy(), policy_sha256=value)


@pytest.mark.parametrize('field,value', [
    ('schema', 'sciel-comparison-review-grounded-presentation/2'), ('schema', False),
    ('comparison_id', False), ('comparison_sha256', False),
    ('comparator_source_sha256', 'broken'), ('challenger_source_sha256', None),
    ('writer', None), ('writer', {'phase': 'reflection', 'reference': True, 'raw_response_sha256': 'a' * 64}),
    ('writer', {'phase': 'conclusion', 'reference': 1, 'raw_response_sha256': 'a' * 64}),
    ('writer', {'phase': 'reflection', 'reference': 0, 'raw_response_sha256': 'a' * 64}),
    ('writer', {'phase': 'reflection', 'reference': 1, 'raw_response_sha256': False})])
def test_binding_shape_validated_even_for_pending(card, field, value):
    bad = copy.deepcopy(card)
    bad[field] = value
    with pytest.raises(ValueError):
        resolve(bad, policy())


def test_nonfinite_or_non_json_content_cannot_get_a_card_digest(card):
    for value in (float('nan'), float('inf'), (), {1: 'non-string key'}):
        bad = copy.deepcopy(card)
        bad['unverified_interpretation']['next_hypothesis'] = value
        with pytest.raises(ValueError):
            resolve(bad, policy())


def test_review_only_accepts_no_effects_without_mutation():
    result = Reflection(lessons=[], retire_ids=[], comparison_reviews=[])
    before = copy.deepcopy(result)
    assert eligibility.require_review_only(result) is None
    assert result == before


@pytest.mark.parametrize('lessons,retired', [([{'rule': 'new'}], []), ([], ['l1']),
    ([{'rule': 'new'}], ['l1']), (None, []), ([], None), ((), []), ([], ()), (False, []), ([], False)])
def test_review_only_rejects_whole_effectful_or_malformed_result(lessons, retired):
    result = SimpleNamespace(lessons=lessons, retire_ids=retired)
    before = copy.deepcopy(result)
    with pytest.raises(ValueError, match='empty lessons and retire_ids'):
        eligibility.require_review_only(result)
    assert result == before


def test_review_only_rejects_missing_parsed_fields():
    with pytest.raises(ValueError):
        eligibility.require_review_only(SimpleNamespace())


def test_identity_pins_source_and_schema_without_mutable_shared_state():
    identity = eligibility.identity()
    assert identity['source_sha256'] == hashlib.sha256(Path(eligibility.__file__).read_bytes()).hexdigest()
    assert identity['schema'] == eligibility.POLICY_SCHEMA
    assert identity['decision_schema'] == eligibility.DECISION_SCHEMA
    assert identity['max_decisions'] == 64
    before = copy.deepcopy(identity)
    identity['scope'] = 'changed'
    assert eligibility.identity() == before
