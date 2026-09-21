"""One pure gated R append over the immutable accepted v2 saved history.

Only JSON is read from the archive. No SDK, candidate, evaluator, solver or
archived Python executes. The empty synthetic response proves that structural
admission cannot grant review text eligibility or downgrade the request mode.
"""
import hashlib
import json
from pathlib import Path

import pytest

from sciel.engine.external_reflection import ExternalReflectionJob
from sciel.engine.external_research_v2 import policy_identity, validate
from sciel.engine.reflection_context import REVIEW_ONLY_INSTRUCTION
from sciel.hypothesis import comparison_review
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha


ROOT = Path(__file__).resolve().parents[2]
POLICY_KEY = 'comparison_review_text_policy'


def test_gated_job_empty_return_appends_exact_history_and_remains_pending(monkeypatch):
    from sciel.engine.research import ResearchController

    def forbidden(*args, **kwargs):
        pytest.fail('pure external reflection attempted a controller or scheduler')

    from external_review_fixture import grounded_history
    authored = grounded_history()
    capsule, config, terminal = authored.capsule, authored.config, authored.terminal
    monkeypatch.setattr(ResearchController, '__init__', forbidden)
    historical = {key: value for key, value in capsule.items() if key != 'policy'}
    capsule['policy'] = policy_identity()
    policy = {'schema': 'sciel-review-text-policy/1', 'decisions': []}
    config.update(comparison_review_version=2, comparison_review_text=True,
        batch_only_lessons=True, comparison_review_text_policy=policy,
        external_history_sha256=canonical_sha(capsule))
    source_before, config_before = canonical_json(capsule), canonical_json(config)
    assert len(capsule['journals']) == 7 and len(capsule['operations']) == 10
    job = ExternalReflectionJob(capsule, expected_sha256=canonical_sha(capsule), config=config)
    plan = job.prepare(run_id='inert-host-gated-reflection', segment_id='inert-host-gated-r1')
    assert plan['prompt'].startswith(REVIEW_ONLY_INSTRUCTION)
    assert plan['config_sha256'] == canonical_sha(config)
    prefix = list(map(json.loads, plan['journal_prefix'].splitlines()))
    context, = [row for row in prefix if row.get('phase') == 'reflection_context']
    assert context['reflection_effects_policy'] == 'review_only/1'
    assert prefix[-1]['phase'] == 'aux_start' and prefix[-1]['call_index'] == 1
    assert prefix[-1]['prompt_sha256'] == plan['prompt_sha256']
    assert prefix[-1]['response_format'] == plan['response_format']
    assert plan['new_ordinary_outcomes'] == plan['new_search_attempts'] == 0

    response = dict(text=json.dumps(dict(lessons=[], retire_ids=[], comparison_reviews=[])),
        model=config['model'], cost=0., prompt_tokens=1, completion_tokens=1,
        finish_reason='stop', refusal=None)
    result = job.admit(plan, response, manifest_sha256='a' * 64,
        journal_prefix=plan['journal_prefix'])
    assert result['status'] == 'accepted'
    assert result['accepted_review_ids'] == result['new_lesson_ids'] == result['retired_lesson_ids'] == []
    assert result['provider_dispatch_verified'] is False
    assert result['recorded_provider_responses'] == 1 and result['known_cost'] == 0.
    assert result['new_ordinary_outcomes'] == result['new_search_attempts'] == 0
    assert result['terminal'] == terminal
    assert result['journal'].startswith(plan['journal_prefix'])
    rows = list(map(json.loads, result['journal'].splitlines()))
    assert rows[:-2] == prefix
    assert [row['phase'] for row in rows[-2:]] == ['aux_result', 'reflection']
    assert rows[-2]['response'] == response['text']
    extended = result['extended_capsule']
    assert len(extended['journals']) == len(capsule['journals']) + 1
    assert extended['journals'][:-1] == capsule['journals']
    assert extended['operations'][:-1] == capsule['operations']
    assert extended['journals'][-1]['journal'] == result['journal']
    assert canonical_sha(extended) == result['extended_capsule_sha256']
    replay = validate(extended, config=config, metric=extended['metric'])
    assert replay['knowledge'].state() == terminal['knowledge']
    assert replay['conclusions'] == terminal['conclusions']
    assert replay['comparison_reviews'] == terminal['comparison_reviews']

    card, = replay['comparison_reviews'].values()
    group = extended['comparison_catalog'][card['comparison_id']]
    view = comparison_review.context(replay['comparison_reviews'], [group], group['branch'],
        comparison_review.MAX_CONTEXT_CHARS, text_enabled=True,
        seen=replay['knowledge']._seen, text_policy=policy,
        policy_sha256=canonical_sha(policy_identity()))
    shown, = view['cards']
    assert shown['text_eligibility']['status'] == 'pending'
    assert shown['unverified_interpretation'] is None
    assert shown['host_facts'] == card['host_facts']

    ungated = {key: value for key, value in config.items() if key != POLICY_KEY}
    with pytest.raises(ValueError, match='review-only reflection history requires receiving review text policy'):
        validate(extended, config=ungated, metric=extended['metric'])
    assert canonical_json(capsule) == source_before
    assert canonical_json(config) == config_before
    assert {key: value for key, value in capsule.items() if key != 'policy'} == historical
