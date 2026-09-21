"""Gate complete R mutations and the exact new checkpoint identity."""
import copy

import pytest

from sciel.engine import external_research
from sciel.knowledge_base.catalogs import canonical_json
from test_comparison_review_flow import reflection_event, restored
from test_reasoned_review_flow import reasoned_fixture
from test_research_paired_comparisons import proposal as lesson_proposal
from test_review_eligibility_flow import POLICY_KEY, policy
from test_review_eligibility_history import configured


@pytest.mark.parametrize('replace', [False, True])
def test_controller_rejects_whole_review_with_otherwise_valid_lesson_effects(replace):
    f = reasoned_fixture(comparison_review_text=True, **{POLICY_KEY: policy()})
    proposed = lesson_proposal(f.record['id'])
    if replace:
        old, = f.c.knowledge.add_lessons([proposed], allowed_evidence_ids=[f.record['id']])
        f.reply['retire_ids'] = [old]
        proposed = {**proposed, 'rule': 'A different synthetic lesson over the same observation.'}
    f.reply['lessons'] = [proposed]
    # Prove this is an admissible lesson transaction, not a malformed proposal
    # that an older validator would already reject.
    alternative = copy.deepcopy(f.c.knowledge)
    alternative.reflect(f.reply['lessons'], f.reply['retire_ids'],
        allowed_evidence_ids=[f.record['id']], writer_context=dict(phase='reflection',
        reference=f.c.outcomes, raw_response_sha256='a' * 64))
    before = canonical_json(f.c.knowledge.state())
    f.c.complete_step()
    event = reflection_event(f)
    assert event['status'] == 'parsed' and event['error'] == 'ValueError'
    assert 'review-only' in event['error_detail']
    assert event['applied'] is None and event['comparison_review_ids'] == []
    assert canonical_json(f.c.knowledge.state()) == before
    assert f.c._comparison_reviews == {} and f.c.reflections == 0
    assert len(f.calls) == 2  # Inert C/R; no retry or next generation.


def test_checkpoint_delivery_identity_rejects_integer_boolean_alias():
    f = reasoned_fixture(comparison_review_text=True, **{POLICY_KEY: policy()})
    f.c.complete_step()
    clone = restored(f)
    before = canonical_json(clone.state())
    damaged = f.c.state()
    damaged['review_text_policy']['text_enabled'] = 1
    with pytest.raises(ValueError, match='review text policy'):
        clone.load_state(damaged)
    assert canonical_json(clone.state()) == before


def test_gated_public_legacy_import_is_explicitly_unsupported():
    with pytest.raises(ValueError, match='requires explicit v2 history'):
        external_research._validate({'schema': external_research.SCHEMA},
            expected_sha256='a' * 64, config=configured(), metric={})
