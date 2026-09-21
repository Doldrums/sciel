"""External R planning/admission uses saved facts, with no provider or solver."""
import copy
import json

import pytest

from sciel.engine.external_reflection import ExternalReflectionJob, ExternalReflectionRejected
from sciel.engine.external_research_v2 import validate
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha

from test_comparison_review_flow import review_proposal
from test_external_research_v2_reducer import actual


@pytest.fixture(scope='module')
def saved():
    capsule, final = actual.__wrapped__()
    config = {**final['config'], 'comparison_reviews': True, 'max_aux_calls': 1,
        'model': 'anthropic/claude-sonnet-4.5', 'external_history_sha256': canonical_sha(capsule)}
    return capsule, final, config


def job_and_plan(saved):
    capsule, _, config = saved
    job = ExternalReflectionJob(capsule, expected_sha256=canonical_sha(capsule), config=config)
    return job, job.prepare(run_id='new-external-reflection', segment_id='one-new-segment')


def receipt(saved, *, empty=False):
    capsule, _, _ = saved
    group = list(capsule['comparison_catalog'].values())[-1]
    return dict(text=json.dumps(dict(lessons=[], retire_ids=[],
                    comparison_reviews=[] if empty else [review_proposal(group)])),
                model='anthropic/claude-sonnet-4.5', cost=.002, prompt_tokens=100,
                completion_tokens=50, finish_reason='stop', refusal=None)


def admit(job, plan, response):
    return job.admit(plan, response, manifest_sha256='a' * 64,
                     journal_prefix=plan['journal_prefix'])


def test_prepare_is_pure_reproducible_and_labels_historical_watermark(saved, monkeypatch):
    from sciel.engine.research import ResearchController
    def forbidden(*args, **kwargs):
        pytest.fail('external R instantiated a controller or scheduler')
    monkeypatch.setattr(ResearchController, '__init__', forbidden)
    capsule, final, config = saved
    original = canonical_json(capsule), canonical_json(config)
    job, plan = job_and_plan(saved)
    assert job.prepare(run_id=plan['run_id'], segment_id=plan['segment_id']) == plan
    assert plan['historical_through'] == 2
    assert plan['new_ordinary_outcomes'] == plan['new_search_attempts'] == 0
    events = list(map(json.loads, plan['journal_prefix'].splitlines()))
    assert events[0]['resumed'] is False
    assert events[-1]['phase'] == 'aux_start' and events[-1]['call_index'] == 1
    assert events[-1]['reference'] == 2
    assert sum(e.get('phase') == 'aux_start' for e in events) == 1
    assert not any(e.get('phase') in ('attempt', 'outcome', 'aux_result') for e in events)
    context = next(e for e in events if e.get('phase') == 'reflection_context')
    assert context['experiment_ids'][0] == 'r000002a1'
    assert original == (canonical_json(capsule), canonical_json(config))


def test_one_returned_review_extends_real_capsule_and_replays_exactly(saved):
    capsule, final, config = saved
    original = canonical_json(capsule)
    job, plan = job_and_plan(saved)
    response = receipt(saved)
    result = admit(job, plan, response)
    assert result['status'] == 'accepted' and result['recorded_provider_responses'] == 1
    assert result['provider_dispatch_verified'] is False and 'new_provider_calls' not in result
    assert result['known_cost'] == .002 and result['new_ordinary_outcomes'] == 0
    assert result['terminal']['knowledge'] == final['knowledge']
    assert result['terminal']['conclusions'] == final['conclusions']
    assert result['new_lesson_ids'] == result['retired_lesson_ids'] == []
    group_id, = result['accepted_review_ids']
    card = result['terminal']['comparison_reviews'][group_id]
    assert card['writer']['reference'] == 2
    assert card['host_facts']['derived']['all_no_loss'] is False
    assert result['journal'].startswith(plan['journal_prefix'])
    events = list(map(json.loads, result['journal'].splitlines()))
    assert [e['phase'] for e in events if e.get('phase') in ('aux_start', 'aux_result', 'reflection')] == [
        'aux_start', 'aux_result', 'reflection']
    assert events[-1]['comparison_review_ids'] == [group_id]
    extended = result['extended_capsule']
    assert len(extended['journals']) == len(capsule['journals']) + 1
    assert extended['journals'][:-1] == capsule['journals']
    assert extended['operations'][:-1] == capsule['operations']
    assert extended['operations'][-1]['kind'] == 'reflection'
    replay = validate(extended, config=config, metric=extended['metric'])
    assert replay['knowledge'].state() == final['knowledge']
    assert replay['comparison_reviews'] == result['terminal']['comparison_reviews']
    assert canonical_sha(extended) == result['extended_capsule_sha256']
    assert canonical_json(capsule) == original


def test_empty_valid_review_is_recorded_without_inventing_a_treatment(saved):
    job, plan = job_and_plan(saved)
    result = admit(job, plan, receipt(saved, empty=True))
    assert result['status'] == 'accepted'
    assert result['accepted_review_ids'] == [] and result['terminal']['comparison_reviews'] == {}
    assert result['terminal']['knowledge'] == saved[1]['knowledge']


@pytest.mark.parametrize('damage', ['prompt', 'context', 'prefix', 'schema', 'interval', 'policy'])
def test_prepared_plan_or_persisted_prefix_cannot_be_substituted(saved, damage):
    job, plan = job_and_plan(saved)
    bad = copy.deepcopy(plan)
    prefix = plan['journal_prefix']
    if damage == 'prompt':
        bad['prompt'] += ' invented direction'
    elif damage == 'context':
        bad['historical_through'] = 1
    elif damage == 'prefix':
        prefix += '{}\n'
    elif damage == 'schema':
        bad['response_format']['json_schema']['strict'] = False
    elif damage == 'interval':
        bad['interval'] = True
    else:
        bad['job_policy']['profile'] = 'unknown'
    with pytest.raises(ValueError):
        job.admit(bad, receipt(saved), manifest_sha256='a' * 64, journal_prefix=prefix)


@pytest.mark.parametrize('damage', ['length', 'refusal', 'long_finish', 'long_refusal',
                                  'oversize', 'bad_json', 'group_digest', 'bad_retirement'])
def test_returned_bad_response_keeps_terminal_journal_without_capsule(saved, damage):
    job, plan = job_and_plan(saved)
    response = receipt(saved)
    if damage == 'length':
        response['finish_reason'] = 'length'
    elif damage == 'refusal':
        response['refusal'] = 'fixture refusal'
    elif damage == 'long_finish':
        response['finish_reason'] = 'x' * 129
    elif damage == 'long_refusal':
        response['refusal'] = 'x' * 2049
    elif damage == 'oversize':
        response['text'] = 'x' * 32769
    elif damage == 'bad_json':
        response['text'] = 'not JSON'
    else:
        value = json.loads(response['text'])
        if damage == 'group_digest':
            value['comparison_reviews'][0]['comparison_sha256'] = '0' * 64
        else:
            value['retire_ids'] = ['l999999']
        response['text'] = json.dumps(value)
    with pytest.raises(ExternalReflectionRejected) as failure:
        admit(job, plan, response)
    journal = failure.value.journal
    assert journal.startswith(plan['journal_prefix'])
    events = list(map(json.loads, journal.splitlines()))
    assert sum(e.get('phase') == 'aux_result' for e in events) == 1
    recorded = next(e for e in events if e.get('phase') == 'aux_result')
    if damage == 'long_finish':
        assert recorded['finish_reason'] is None
    if damage == 'long_refusal':
        assert len(recorded['refusal']) == 2048 and recorded['refusal_truncated'] is True
    assert events[-1]['phase'] in ('reflection', 'aux_stopped')
    assert not hasattr(failure.value, 'extended_capsule')
    # Rejection never consumes/mutates the pure job or its source history.
    assert job.prepare(run_id=plan['run_id'], segment_id=plan['segment_id']) == plan


@pytest.mark.parametrize('damage', ['bool_cost', 'unknown_cost', 'negative_cost', 'bool_tokens', 'model', 'missing_field'])
def test_malformed_receipt_is_not_presented_as_an_authenticated_result(saved, damage):
    job, plan = job_and_plan(saved)
    response = receipt(saved)
    if damage == 'bool_cost': response['cost'] = True
    elif damage == 'unknown_cost': response['cost'] = None
    elif damage == 'negative_cost': response['cost'] = -1
    elif damage == 'bool_tokens': response['completion_tokens'] = True
    elif damage == 'model': response['model'] = 'another-model'
    else: del response['finish_reason']
    with pytest.raises(ValueError):
        admit(job, plan, response)


@pytest.mark.parametrize('damage', ['digest', 'undeclared', 'aux_disabled', 'reviews_disabled', 'old_run_id'])
def test_job_requires_current_explicit_history_and_fresh_identity(saved, damage):
    capsule, _, config = saved
    config = copy.deepcopy(config)
    if damage == 'digest':
        config['external_history_sha256'] = '0' * 64
    elif damage == 'undeclared':
        del config['external_history_sha256']
    elif damage == 'aux_disabled':
        config['max_aux_calls'] = 0
    elif damage == 'reviews_disabled':
        config['comparison_reviews'] = False
    if damage != 'old_run_id':
        with pytest.raises(ValueError):
            ExternalReflectionJob(capsule, expected_sha256=canonical_sha(capsule), config=config)
    else:
        job = ExternalReflectionJob(capsule, expected_sha256=canonical_sha(capsule), config=config)
        old = json.loads(capsule['journals'][0]['journal'].splitlines()[0])
        with pytest.raises(ValueError, match='fresh journal identities'):
            job.prepare(run_id=old['run_id'], segment_id='new-segment')
