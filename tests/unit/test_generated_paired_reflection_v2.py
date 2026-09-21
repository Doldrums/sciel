"""Structured authoring failures and prospective tests over inert saved facts."""
import copy
import json

import pytest

from sciel.engine.external_reflection import ExternalReflectionRejected
from sciel.hypothesis.generated_paired_review import AUTHORING_PREFIX, authoring_context
from sciel.hypothesis.comparison_targets import instruction as target_instruction
from sciel.knowledge_base.catalogs import canonical_sha

from test_generated_paired_reflection import no_runtime, saved, response, admit, proposal


def test_wire_v2_request_contains_full_bound_authoring_context(saved):
    prompt = saved.plan['prompt']
    appendix, _ = json.JSONDecoder().raw_decode(prompt.split(AUTHORING_PREFIX, 1)[1])
    assert appendix == authoring_context(saved.component)
    events = list(map(json.loads, saved.plan['journal_prefix'].splitlines()))
    context = next(row for row in events if row.get('phase') == 'generated_paired_reflection_context')
    from sciel.hypothesis.generated_paired_review import authoring_block
    from sciel.engine.generated_paired_reflection import _digest
    assert context['authoring_block_sha256'] == _digest(authoring_block(saved.component))
    assert target_instruction() in prompt
    assert saved.plan['schema'] == 'sciel-generated-paired-reflection-plan/2'
    assert saved.plan['auxiliary_schema']['schema'] == 'sciel-generated-paired-auxiliary-schema/2'


def test_exact_reason_diagnostic_survives_rejected_job_journal_without_repair(saved):
    value = response(saved)
    payload = json.loads(value['text'])
    payload['comparison_reviews'][0]['next_hypothesis_reason'] = 'PRIVATE_DRAFT_' + 'x' * 1014
    assert len(payload['comparison_reviews'][0]['next_hypothesis_reason']) == 1028
    value['text'] = json.dumps(payload)
    before = copy.deepcopy(value)
    with pytest.raises(ExternalReflectionRejected) as caught:
        admit(saved, value)
    rows = list(map(json.loads, caught.value.journal.splitlines()))
    assert rows[-2]['response'] == value['text']
    terminal = rows[-1]
    assert terminal['status'] == 'invalid'
    assert terminal['error_diagnostic'] == dict(code='text_too_long',
        path='comparison_reviews[0].next_hypothesis_reason', actual=1028, maximum=1024)
    assert 'PRIVATE_DRAFT_' not in terminal['error_detail']
    assert 'text_too_long' in terminal['error_detail']
    assert terminal['comparison_review_ids'] == terminal['new_lesson_ids'] == []
    assert value == before


@pytest.mark.parametrize('prospective', [False, True])
def test_known_loss_in_incomplete_data_allows_proposal_or_abstention_without_execution(saved, prospective):
    group = saved.component['comparisons'][1]
    value = response(saved)
    payload = json.loads(value['text'])
    item = proposal(saved, prospective=prospective)
    item.update(comparison_id=group['id'], comparison_sha256=canonical_sha(group),
        claims=[dict(fact_id='quality:strict_verdict', value='REFUTED')])
    if prospective:
        item['next_hypothesis'].update(base_source_sha256=group['challenger_source_sha256'],
            comparator_source_sha256=group['comparator_source_sha256'])
    payload['comparison_reviews'] = [item]
    value['text'] = json.dumps(payload)
    result = admit(saved, value)
    card, = result['extended_capsule']['review_cards'].values()
    assert card['host_facts']['strict_quality']['verdict'] == 'REFUTED'
    assert card['host_facts']['stored_quality_outcome'] == 'incomplete'
    assert card['unverified_interpretation']['next_hypothesis'] == item['next_hypothesis']
    assert result['new_ordinary_outcomes'] == result['new_search_attempts'] == 0
    assert result['new_lesson_ids'] == result['retired_lesson_ids'] == []
    assert 'generation_allowed' not in result
