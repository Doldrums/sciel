"""An authored semantically false v3 card stays inert while factual memory survives.

The current-policy fixture tests import, replay and text withholding.
No provider, archived Python, candidate, evaluator or solver executes."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from sciel.engine.external_research_v2 import policy_identity
from sciel.hypothesis import comparison_review
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
from test_comparison_review_text_flow import common_review
from test_external_history_v2_actual import construct, import_history
from test_review_eligibility_flow import POLICY_KEY, decision, policy


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def actual():
    from external_review_fixture import reasoned_history
    authored = reasoned_history()
    capsule, config, terminal, response = (authored.capsule, authored.config,
        authored.terminal, authored.response)
    card, = terminal['comparison_reviews'].values()
    assert card['schema'] == 'sciel-comparison-review-card/3'
    assert card['writer']['raw_response_sha256'] == hashlib.sha256(response['text'].encode()).hexdigest()
    audit = dict(classification='semantic_failure',
        evidence=dict(raw_response_sha256=card['writer']['raw_response_sha256']))
    criteria = dict(rule='Interpretation must agree with the separately measured directions.')
    return NS(capsule=capsule, config=config, terminal=terminal, card=card,
        group=capsule['comparison_catalog'][card['comparison_id']],
        seen=terminal['knowledge']['seen_experiments'], response=response,
        audit_sha=canonical_sha(audit), criteria_sha=canonical_sha(criteria))


def rejection(actual):
    record = decision(actual.card, 'rejected')
    record.update(audit_sha256=actual.audit_sha, criteria_sha256=actual.criteria_sha,
                  issuer='inert import fixture referring to the authored semantic audit')
    return record


@pytest.mark.parametrize('status', ['pending', 'rejected'])
def test_actual_false_prose_is_withheld_wholly_with_all_host_facts_and_full_reserve(actual, status):
    registry = policy() if status == 'pending' else policy(rejection(actual))
    before = canonical_json(actual.card)
    views = [comparison_review.context({actual.card['comparison_id']: actual.card}, [actual.group],
        actual.group['branch'], comparison_review.MAX_CONTEXT_CHARS, text_enabled=mode, seen=actual.seen,
        text_policy=registry, policy_sha256=canonical_sha(policy_identity())) for mode in (True, False)]
    on, off = views
    assert on['presented_count'] == off['presented_count'] == 1
    assert on['reserved_chars'] == off['reserved_chars']
    assert common_review(on) == common_review(off)
    for view in views:
        shown, = view['cards']
        assert shown['unverified_interpretation'] is None
        assert shown['text_eligibility']['status'] == status
        assert shown['host_facts'] == actual.card['host_facts']
        assert len(shown['host_facts']['facts']) == 23
        assert shown['writer'] == actual.card['writer']
        assert 'origin_snapshot' not in shown
        rendered = canonical_json(view)
        for field in ('conjecture', 'limitations', 'next_hypothesis_reason'):
            assert actual.card['unverified_interpretation'][field] not in rendered
    assert canonical_json(actual.card) == before


def test_actual_rejected_capsule_import_and_resume_never_grants_text_or_replays_work(actual, tmp_path):
    capsule = copy.deepcopy(actual.capsule)
    capsule['policy'] = policy_identity()
    assert {k: v for k, v in capsule.items() if k != 'policy'} == {
        k: v for k, v in actual.capsule.items() if k != 'policy'}
    config = {**actual.config, POLICY_KEY: policy(rejection(actual)),
              'comparison_review_text': True, 'batch_only_lessons': True,
              'external_history_sha256': canonical_sha(capsule)}
    final = dict(config=config, metric=capsule['metric'], **copy.deepcopy(actual.terminal))
    f = construct(capsule, final, tmp_path / 'inert-import')
    import_history(f, capsule)
    assert f.c._comparison_reviews == actual.terminal['comparison_reviews']
    assert f.c.knowledge.state() == actual.terminal['knowledge']
    assert f.c.conclusions == actual.terminal['conclusions']
    context = f.c.context(NS(_all={'p1': f.parent}), f.parent, 0, 1)
    shown, = context['comparison_reviews']['cards']
    assert shown['text_eligibility']['status'] == 'rejected'
    assert shown['unverified_interpretation'] is None
    assert shown['host_facts'] == actual.card['host_facts']
    saved = f.c.state()
    clone = construct(capsule, final, tmp_path / 'inert-resume')
    clone.c.load_state(saved)
    clone.worker.load_state(f.worker.state())
    clone.worker.assert_journal(f.log.path, f.log.anchor())
    assert clone.c.state() == saved
    assert clone.c.context(NS(_all={'p1': f.parent}), f.parent, 0, 1) == context
    for subject in (f, clone):
        assert subject.model.calls == subject.charges == []
        assert subject.worker.calls_started == subject.worker.calls_returned == subject.worker.evaluations == 0
        assert all(getattr(subject.c, key) == 0 for key in
                   ('attempts', 'outcomes', 'aux_calls', 'aux_cost', 'reflections', 'reflected_through'))
    # No forged override or promotion of this failed semantic record occurs.
    assert all(row['decision'] == 'rejected' for row in config[POLICY_KEY]['decisions'])
