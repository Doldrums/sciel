"""Exact host decisions gate synthetic review text, not facts or search authority.

All positive decisions refer to separately synthetic cards. No semantic approval
is attributed to either rejected real response. Inert fixture replies prepare
records only; no provider, evaluator, candidate or solver runs.
"""
import copy

import pytest

from sciel.engine.external_research_v2 import policy_identity
from sciel.hypothesis import comparison_review
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
from test_comparison_review_flow import restored
from test_comparison_review_text_flow import common_review
from test_reasoned_review_flow import reasoned_fixture
from test_research_measurement_state import begin_fixture
from test_research_paired_comparisons import decoded_context


POLICY_KEY = 'comparison_review_text_policy'


def decision(card, result='accepted', *, policy_sha256=None):
    return dict(schema='sciel-review-text-decision/1', scope='comparison_review_text',
        decision=result, card_sha256=canonical_sha(card),
        raw_response_sha256=card['writer']['raw_response_sha256'],
        comparison_id=card['comparison_id'], comparison_sha256=card['comparison_sha256'],
        comparator_source_sha256=card['comparator_source_sha256'],
        challenger_source_sha256=card['challenger_source_sha256'],
        policy_sha256=policy_sha256 or canonical_sha(policy_identity()),
        audit_sha256='a' * 64, criteria_sha256='b' * 64, issuer='inert fixture auditor')


def policy(*decisions):
    return dict(schema='sciel-review-text-policy/1', decisions=list(decisions))


@pytest.fixture(scope='module')
def synthetic_card():
    f = reasoned_fixture(abstain=True)
    f.c.complete_step()
    return copy.deepcopy(f.c._comparison_reviews[f.group['id']])


def staged(synthetic_card, *, result='accepted', text=True):
    decisions = [] if result == 'pending' else [decision(synthetic_card, result)]
    f = reasoned_fixture(abstain=True, comparison_review_text=text,
                         **{POLICY_KEY: policy(*decisions)})
    f.c.complete_step()
    assert f.c._comparison_reviews[f.group['id']] == synthetic_card
    return f


@pytest.mark.parametrize('status', ['pending', 'rejected', 'accepted'])
def test_exact_decision_masks_whole_model_block_and_preserves_matched_context(synthetic_card, status):
    arms = [staged(synthetic_card, result=status, text=text) for text in (True, False)]
    contexts = [f.c.context(f.db, f.parent, 0, 2) for f in arms]
    a, b = copy.deepcopy(contexts)
    on, off = a.pop('comparison_reviews'), b.pop('comparison_reviews')
    assert on['presented_count'] == off['presented_count'] == 1
    assert on['reserved_chars'] == off['reserved_chars']
    assert common_review(on) == common_review(off)
    assert canonical_json(a) == canonical_json(b)
    for index, (f, ctx) in enumerate(zip(arms, contexts)):
        shown = ctx['comparison_reviews']['cards'][0]
        assert shown['text_eligibility']['status'] == status
        assert shown['host_facts'] == synthetic_card['host_facts']
        assert shown['writer'] == synthetic_card['writer']
        assert (shown['unverified_interpretation'] is not None) is (status == 'accepted' and index == 0)
        _, prompt = begin_fixture(f.c, ctx)
        assert decoded_context(prompt)['comparison_reviews'] == ctx['comparison_reviews']
        assert f.c._comparison_reviews[f.group['id']] == synthetic_card
        assert len(f.calls) == 2  # Inert fixture C/R only; begin is not dispatch.


def test_reason_or_writer_replacement_cannot_inherit_exact_card_decision(synthetic_card):
    f = staged(synthetic_card)
    group = f.group
    records, seen = f.c.knowledge._evidence(), f.c.knowledge._seen
    original = canonical_json(synthetic_card)
    for damage in ('reason', 'writer'):
        # Presentation-only replacement, validated structurally; no invented
        # admission or authority is attached to this synthetic variant.
        replacement = copy.deepcopy(synthetic_card)
        if damage == 'reason':
            replacement['unverified_interpretation']['next_hypothesis_reason'] = 'A different synthetic rationale.'
        else:
            replacement['writer']['raw_response_sha256'] = 'c' * 64
        comparison_review.validate_cards({group['id']: replacement}, [group], seen=seen)
        block = comparison_review.context({group['id']: replacement}, [group], group['branch'], comparison_review.MAX_CONTEXT_CHARS,
            text_enabled=True, seen=seen, text_policy=policy(decision(synthetic_card)),
            policy_sha256=canonical_sha(policy_identity()))
        shown, = block['cards']
        assert shown['text_eligibility']['status'] == 'pending'
        assert shown['unverified_interpretation'] is None
        assert shown['host_facts'] == synthetic_card['host_facts']
    assert canonical_json(synthetic_card) == original and records


def test_approval_from_another_source_policy_is_stale_and_withheld(synthetic_card):
    f = staged(synthetic_card)
    registry = policy(decision(synthetic_card, policy_sha256='f' * 64))
    views = [comparison_review.context({f.group['id']: synthetic_card}, [f.group], f.group['branch'],
        comparison_review.MAX_CONTEXT_CHARS, text_enabled=mode, seen=f.c.knowledge._seen, text_policy=registry,
        policy_sha256=canonical_sha(policy_identity())) for mode in (True, False)]
    assert views[0]['reserved_chars'] == views[1]['reserved_chars']
    assert common_review(views[0]) == common_review(views[1])
    for view in views:
        shown, = view['cards']
        assert shown['text_eligibility']['status'] == 'stale'
        assert shown['unverified_interpretation'] is None
        assert shown['host_facts'] == synthetic_card['host_facts']


def test_saved_context_cannot_deliver_old_approved_text_after_card_replacement(synthetic_card):
    f = staged(synthetic_card)
    context = f.c.context(f.db, f.parent, 0, 2)
    replacement = copy.deepcopy(synthetic_card)
    replacement['writer']['raw_response_sha256'] = 'c' * 64
    replacement['unverified_interpretation']['next_hypothesis_reason'] = 'A later synthetic unapproved decision.'
    comparison_review.validate_cards({f.group['id']: replacement}, [f.group], seen=f.c.knowledge._seen)
    f.c._comparison_reviews[f.group['id']] = replacement
    before, events = copy.deepcopy(f.c.state()), copy.deepcopy(f.events)
    with pytest.raises(ValueError, match='comparison review|eligibility|policy'):
        begin_fixture(f.c, context)
    assert f.c.state() == before and f.events == events
    assert len(f.calls) == 2
    current = f.c.context(f.db, f.parent, 0, 2)['comparison_reviews']['cards'][0]
    assert current['text_eligibility']['status'] == 'pending'
    assert current['unverified_interpretation'] is None


@pytest.mark.parametrize('operation', ['context', 'begin', 'load'])
def test_registry_mutation_cannot_grant_permission_in_a_running_controller(synthetic_card, operation):
    f = staged(synthetic_card, result='pending')
    context = f.c.context(f.db, f.parent, 0, 2)
    saved, events = copy.deepcopy(f.c.state()), copy.deepcopy(f.events)
    f.c.config[POLICY_KEY] = policy(decision(synthetic_card))
    with pytest.raises(ValueError, match='policy|eligibility'):
        if operation == 'context':
            f.c.context(f.db, f.parent, 0, 2)
        elif operation == 'begin':
            begin_fixture(f.c, context)
        else:
            f.c.load_state(saved)
    assert f.events == events and len(f.calls) == 2


@pytest.mark.parametrize('operation', ['context', 'begin'])
def test_frozen_off_arm_cannot_switch_text_on_in_place(synthetic_card, operation):
    f = staged(synthetic_card, text=False)
    context = f.c.context(f.db, f.parent, 0, 2)
    saved, events = copy.deepcopy(f.c.state()), copy.deepcopy(f.events)
    f.c.config['comparison_review_text'] = True
    with pytest.raises(ValueError, match='policy|eligibility|text'):
        if operation == 'context':
            f.c.context(f.db, f.parent, 0, 2)
        else:
            begin_fixture(f.c, context)
    assert f.events == events and len(f.calls) == 2
    f.c.config['comparison_review_text'] = False
    assert f.c.state() == saved


def test_checkpoint_roundtrip_preserves_registry_and_cannot_self_approve(synthetic_card):
    f = staged(synthetic_card, result='rejected')
    clone = restored(f)
    assert clone.state() == f.c.state()
    assert clone.context(f.db, f.parent, 0, 2) == f.c.context(f.db, f.parent, 0, 2)
    forged = f.c.state()
    forged['config'][POLICY_KEY] = policy(decision(synthetic_card))
    before = clone.state()
    with pytest.raises(ValueError, match='configuration|policy|eligibility'):
        clone.load_state(forged)
    assert clone.state() == before


@pytest.mark.parametrize('damage', ['missing', 'registry_digest'])
def test_checkpoint_cannot_drop_or_substitute_frozen_policy_identity(synthetic_card, damage):
    f = staged(synthetic_card, result='rejected')
    clone = restored(f)
    forged, before = f.c.state(), clone.state()
    assert forged['review_text_policy']['decisions_sha256'] == canonical_sha(f.c.config[POLICY_KEY])
    if damage == 'missing':
        forged.pop('review_text_policy')
    else:
        forged['review_text_policy']['decisions_sha256'] = canonical_sha(policy(decision(synthetic_card)))
    with pytest.raises(ValueError, match='policy|eligibility'):
        clone.load_state(forged)
    assert clone.state() == before
