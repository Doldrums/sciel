"""Pure R over real archived G-only facts; no transport or numerical execution.

Archive members are inert data. Replies below are explicit synthetic fixtures,
never represented as actual provider responses or semantic host approvals.
"""
import copy
import hashlib
import json
from types import SimpleNamespace as NS

import pytest

from sciel.engine import generated_paired_reflection as reflection
from sciel.engine.external_reflection import ExternalReflectionRejected
from sciel.hypothesis import generated_paired_review as review
from sciel.hypothesis.comparison_targets import QUALITY_GATE, SCHEMA as TARGET_SCHEMA
from sciel.hypothesis.review_eligibility import decision_for
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
from sciel.knowledge_base.generated_paired_context import PREFIX, expand_context


@pytest.fixture(scope='module', autouse=True)
def no_runtime():
    def forbidden(*_args, **_kwargs):
        pytest.fail('Pure generated-paired reflection reached runtime or transport.')
    with pytest.MonkeyPatch.context() as patch:
        for name in ('sciel.eval.repeat_draw.draw', 'sciel.eval.repeat_draw.adapter_identity',
                     'sciel.engine.llm.client.EnsembleLLM.generate',
                     'sciel.engine.loop.Engine.__init__',
                     'sciel.engine.research.ResearchController.__init__',
                     'subprocess.Popen'):
            patch.setattr(name, forbidden)
        yield


def digest(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


@pytest.fixture(scope='module')
def saved(no_runtime):
    from test_generated_paired_integration import pilot_artifacts, component as actual_component
    component = actual_component.__wrapped__(pilot_artifacts.__wrapped__())
    config = dict(external_component_sha256=canonical_sha(component),
        restart_predictions=True, structured_auxiliary=True, comparison_reviews=True,
        comparison_review_version=3, inject_lessons=True, max_aux_calls=1,
        model='anthropic/claude-sonnet-4.5', batch_only_lessons=True,
        comparison_review_text=True,
        comparison_review_text_policy={'schema': 'sciel-review-text-policy/1', 'decisions': []})
    job = reflection.GeneratedPairedReflectionJob(component,
        expected_sha256=canonical_sha(component), config=config)
    plan = job.prepare(run_id='new-generated-paired-R-fixture', segment_id='new-R-segment')
    return NS(component=component, config=config, job=job, plan=plan)


def proposal(saved, *, prospective=False):
    group = saved.component['comparisons'][0]
    target = None
    if prospective:
        target = dict(schema=TARGET_SCHEMA, base_source_sha256=group['challenger_source_sha256'],
            comparator_source_sha256=group['comparator_source_sha256'],
            change='A bounded prospective geometry change; effect remains unmeasured.',
            quality_gate=QUALITY_GATE, diagnostics=[])
    return dict(comparison_id=group['id'], comparison_sha256=canonical_sha(group),
        claims=[dict(fact_id='restart:comparator:763606949:accepted_interval_gain:verdict', value='CONFIRMED'),
                dict(fact_id='restart:challenger:763606949:accepted_interval_gain:verdict', value='REFUTED')],
        conjecture='These retained prefixes do not demonstrate beneficial helper intervals.',
        limitations='Only one paired seed was dispatched; native forecasts remain unmeasured.',
        next_hypothesis=target,
        next_hypothesis_reason=('This proposed change needs its own frozen experiment.' if prospective
                                else 'This fixture abstains; it does not authorize another measurement.'))


def response(saved, *, empty=False, prospective=False):
    return dict(text=json.dumps(dict(lessons=[], retire_ids=[],
        comparison_reviews=[] if empty else [proposal(saved, prospective=prospective)])),
        model=saved.config['model'], cost=.002, prompt_tokens=100,
        completion_tokens=50, finish_reason='stop', refusal=None)


def admit(saved, value=None, **changes):
    options = dict(manifest_sha256='a' * 64, journal_prefix=saved.plan['journal_prefix'])
    options.update(changes)
    return saved.job.admit(saved.plan, response(saved) if value is None else value, **options)


@pytest.fixture(scope='module')
def accepted(saved):
    return admit(saved)


def test_prepare_contains_complete_study_both_roles_and_no_native_watermark(saved):
    plan = saved.plan
    assert saved.job.prepare(run_id=plan['run_id'], segment_id=plan['segment_id']) == plan
    assert plan['new_ordinary_outcomes'] == plan['new_search_attempts'] == 0
    assert plan['author_sequence'] == 1 and 'historical_through' not in plan
    assert plan['prompt_sha256'] == digest(plan['prompt'])
    assert plan['prompt_bytes'] == len(plan['prompt'].encode('utf-8'))
    context, _ = json.JSONDecoder().raw_decode(plan['prompt'].split(PREFIX, 1)[1])
    card, = expand_context(context)['cards']
    assert len(card['authored_proposals']) == 2 and len(card['comparisons']) == 3
    assert card['draw_coverage']['status_counts'] == {'valid': 3, 'not_dispatched': 6}
    assert [c['strict_quality']['verdict'] for c in card['comparisons']] == ['INCONCLUSIVE', 'REFUTED', 'REFUTED']
    catalogs = json.loads(plan['prompt'].split(review.PREFIX, 1)[1])
    assert [c['comparison_id'] for c in catalogs] == [g['id'] for g in saved.component['comparisons']]
    off, on = catalogs[0]['authored_forecasts']
    assert [off['role'], on['role']] == ['comparator', 'challenger']
    assert off['proposal_id'] != on['proposal_id']
    assert off['rows'][0]['result']['verdicts']['accepted_interval_gain'] == 'CONFIRMED'
    assert on['rows'][0]['result']['verdicts']['accepted_interval_gain'] == 'REFUTED'
    assert all(c['origin_experiment_id'] is None for c in catalogs)
    assert off['native_forecast_status'] == on['native_forecast_status'] == 'UNMEASURED'
    events = list(map(json.loads, plan['journal_prefix'].splitlines()))
    assert events[0]['resumed'] is False
    assert [e['event_seq'] for e in events] == list(range(1, len(events) + 1))
    assert sum(e.get('phase') == 'aux_start' for e in events) == 1
    assert events[-1]['reference'] == events[-1]['call_index'] == 1
    assert not any(e.get('phase') in ('attempt', 'outcome', 'hypothesis', 'aux_result') for e in events)
    assert plan['response_format']['json_schema']['name'] == 'sciel_generated_paired_reflection_v2'
    assert plan['response_format']['json_schema']['strict'] is True


def test_admitted_sidecar_replays_original_raw_writer_without_ordinary_effects(saved, accepted):
    before = canonical_json([saved.component, saved.config, saved.plan])
    result = accepted
    sidecar = result['extended_capsule']
    assert result['status'] == 'accepted' and result['recorded_provider_responses'] == 1
    assert result['provider_dispatch_verified'] is False and 'new_provider_calls' not in result
    assert result['known_cost'] == .002
    assert result['new_ordinary_outcomes'] == result['new_search_attempts'] == 0
    assert result['new_lesson_ids'] == result['retired_lesson_ids'] == []
    assert sidecar['response'] == response(saved)
    assert sidecar['source_component_sha256'] == canonical_sha(saved.component)
    assert result['journal'].startswith(saved.plan['journal_prefix'])
    assert result['journal_sha256'] == digest(result['journal'])
    events = list(map(json.loads, result['journal'].splitlines()))
    assert sum(e.get('phase') == 'aux_result' for e in events) == 1
    assert not any(e.get('phase') in ('attempt', 'outcome', 'hypothesis') for e in events)
    card, = sidecar['review_cards'].values()
    assert card['writer'] == dict(phase='reflection', reference=1,
                                  raw_response_sha256=digest(response(saved)['text']))
    assert card['unverified_interpretation']['next_hypothesis'] is None
    assert card['unverified_interpretation']['next_hypothesis_reason'] == proposal(saved)['next_hypothesis_reason']
    assert decision_for(card, saved.config['comparison_review_text_policy'],
        policy_sha256=canonical_sha(sidecar['policy']))['status'] == 'pending'
    restored = reflection.validate_sidecar(sidecar, component=saved.component,
                                           expected_sha256=canonical_sha(sidecar))
    assert restored == sidecar and restored is not sidecar
    assert before == canonical_json([saved.component, saved.config, saved.plan])


def test_prospective_target_is_bound_but_grants_no_generation_authority(saved):
    result = admit(saved, response(saved, prospective=True))
    card, = result['extended_capsule']['review_cards'].values()
    assert card['unverified_interpretation']['next_hypothesis'] == proposal(saved, prospective=True)['next_hypothesis']
    assert result['new_search_attempts'] == result['new_ordinary_outcomes'] == 0
    assert 'generation_allowed' not in result


def test_empty_response_is_structurally_valid_without_inventing_treatment_or_retry(saved):
    result = admit(saved, response(saved, empty=True))
    assert result['status'] == 'accepted' and result['accepted_review_ids'] == []
    assert result['extended_capsule']['review_cards'] == {}
    assert result['recorded_provider_responses'] == 1 and result['provider_dispatch_verified'] is False
    assert result['new_lesson_ids'] == result['retired_lesson_ids'] == []
    # Pure validation does not classify semantic usefulness or grant dispatch.
    assert 'generation_allowed' not in result and 'retry' not in result


@pytest.mark.parametrize('damage', ['prompt', 'wire', 'author_bool', 'policy', 'config', 'prefix'])
def test_prepared_inputs_cannot_be_replaced(saved, damage):
    plan = copy.deepcopy(saved.plan)
    prefix = plan['journal_prefix']
    if damage == 'prompt': plan['prompt'] += ' Added unsupported direction.'
    elif damage == 'wire': plan['response_format']['json_schema']['strict'] = False
    elif damage == 'author_bool': plan['author_sequence'] = True
    elif damage == 'policy': plan['job_policy']['schema'] = 'forged'
    elif damage == 'config': plan['config_sha256'] = '0' * 64
    else: prefix += '{}\n'
    with pytest.raises(ValueError):
        saved.job.admit(plan, response(saved), manifest_sha256='a' * 64, journal_prefix=prefix)


@pytest.mark.parametrize('damage', ['raw_text', 'card_reason', 'card_target', 'writer', 'group',
                                  'manifest', 'journal', 'config', 'policy', 'source'])
def test_self_rehashed_sidecar_tampering_does_not_authenticate_a_writer(saved, accepted, damage):
    sidecar = copy.deepcopy(accepted['extended_capsule'])
    card = next(iter(sidecar['review_cards'].values()))
    if damage == 'raw_text': sidecar['response']['text'] = response(saved, empty=True)['text']
    elif damage == 'card_reason': card['unverified_interpretation']['next_hypothesis_reason'] += ' Forged.'
    elif damage == 'card_target': card['unverified_interpretation']['next_hypothesis'] = proposal(saved, prospective=True)['next_hypothesis']
    elif damage == 'writer': card['writer']['raw_response_sha256'] = '0' * 64
    elif damage == 'group': card['host_facts']['facts']['quality:strict_verdict'] = 'CONFIRMED'
    elif damage == 'manifest': sidecar['manifest_sha256'] = False
    elif damage == 'journal': sidecar['journal'] += '{}\n'
    elif damage == 'config': sidecar['config']['model'] = 'different-model'
    elif damage == 'policy': sidecar['policy']['schema'] = 'forged'
    else: sidecar['source_component_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        reflection.validate_sidecar(sidecar, component=saved.component, expected_sha256=canonical_sha(sidecar))


def test_trusted_digest_rejects_changed_valid_manifest_identity(saved, accepted):
    sidecar = copy.deepcopy(accepted['extended_capsule'])
    sidecar['manifest_sha256'] = 'b' * 64
    # The job cannot prove external manifest contents; trust the configured
    # sidecar digest, not a self-issued digest of a changed manifest string.
    with pytest.raises(ValueError, match='digest'):
        reflection.validate_sidecar(sidecar, component=saved.component,
                                    expected_sha256=accepted['extended_capsule_sha256'])


@pytest.mark.parametrize('damage', ['refusal', 'length', 'long_finish', 'long_refusal', 'oversize',
                                  'marker', 'fence', 'duplicate_key', 'lesson', 'retirement',
                                  'missing_reason', 'false_fact', 'target_source', 'target_gate'])
def test_invalid_return_keeps_one_bounded_raw_receipt_without_partial_cards(saved, damage):
    value = response(saved)
    if damage == 'refusal': value['refusal'] = 'fixture refusal'
    elif damage == 'length': value['finish_reason'] = 'length'
    elif damage == 'long_finish': value['finish_reason'] = 'x' * 100000
    elif damage == 'long_refusal': value['refusal'] = 'x' * 100000
    elif damage == 'oversize': value['text'] = 'x' * 32769
    elif damage == 'marker': value['text'] = 'REFLECTION_JSON: ' + value['text']
    elif damage == 'fence': value['text'] = '```json\n' + value['text'] + '\n```'
    elif damage == 'duplicate_key': value['text'] = value['text'].replace('"lessons": []', '"lessons": [], "lessons": []')
    else:
        parsed = json.loads(value['text'])
        if damage == 'lesson': parsed['lessons'] = [dict(rule='Unsupported signed lesson.', polarity='positive', scope='branch', evidence_ids=[saved.component['comparisons'][0]['id']])]
        elif damage == 'retirement': parsed['retire_ids'] = ['l000001']
        elif damage == 'missing_reason': del parsed['comparison_reviews'][0]['next_hypothesis_reason']
        elif damage == 'false_fact': parsed['comparison_reviews'][0]['claims'][1]['value'] = 'CONFIRMED'
        else:
            parsed['comparison_reviews'][0] = proposal(saved, prospective=True)
            parsed['comparison_reviews'][0]['next_hypothesis'][
                'base_source_sha256' if damage == 'target_source' else 'quality_gate'] = (
                '0' * 64 if damage == 'target_source' else 'exact_only')
        value['text'] = json.dumps(parsed)
    original = canonical_json([saved.component, saved.config, saved.plan, value])
    with pytest.raises(ExternalReflectionRejected) as failure:
        admit(saved, value)
    journal = failure.value.journal
    assert journal.startswith(saved.plan['journal_prefix'])
    events = list(map(json.loads, journal.splitlines()))
    receipts = [e for e in events if e.get('phase') == 'aux_result']
    assert len(receipts) == 1
    receipt = receipts[0]
    assert receipt['response_sha256'] == digest(value['text'])
    assert receipt['response'] == value['text'][:32768]
    assert receipt['cost'] == .002
    if damage == 'long_finish': assert receipt['finish_reason'] is None
    if damage == 'long_refusal':
        assert receipt['refusal'] == 'x' * 2048 and receipt['refusal_truncated'] is True
    assert len(journal) - len(saved.plan['journal_prefix']) < 40000
    assert not hasattr(failure.value, 'extended_capsule')
    assert all(not e.get('comparison_review_ids') and not e.get('new_lesson_ids')
               and not e.get('retired_lesson_ids') for e in events)
    assert original == canonical_json([saved.component, saved.config, saved.plan, value])


@pytest.mark.parametrize('damage', ['cost_bool', 'cost_unknown', 'tokens_bool', 'model', 'refusal_type'])
def test_malformed_receipt_cannot_become_authenticated_result(saved, damage):
    value = response(saved)
    key, invalid = {'cost_bool': ('cost', True), 'cost_unknown': ('cost', None),
        'tokens_bool': ('completion_tokens', False), 'model': ('model', 'different-model'),
        'refusal_type': ('refusal', {})}[damage]
    value[key] = invalid
    with pytest.raises(ValueError):
        admit(saved, value)


def test_constructor_detaches_inputs_and_rechecks_current_production_policy(saved, monkeypatch):
    component, config = copy.deepcopy(saved.component), copy.deepcopy(saved.config)
    job = reflection.GeneratedPairedReflectionJob(component,
        expected_sha256=canonical_sha(component), config=config)
    component['proposals'].clear()
    config['model'] = 'changed caller mapping'
    assert job.prepare(run_id=saved.plan['run_id'], segment_id=saved.plan['segment_id']) == saved.plan
    policy = reflection.policy_identity()
    monkeypatch.setattr(reflection, 'policy_identity', lambda: {**policy, 'schema': 'changed-policy'})
    with pytest.raises(ValueError, match='policy changed'):
        job.prepare(run_id=saved.plan['run_id'], segment_id=saved.plan['segment_id'])


@pytest.mark.parametrize('damage', ['missing_policy', 'ordinary_history', 'prior_sidecar', 'digest',
                                  'disabled_reviews', 'old_version', 'no_aux', 'text_mode'])
def test_constructor_requires_explicit_current_scoped_configuration(saved, damage):
    config = copy.deepcopy(saved.config)
    if damage == 'missing_policy': del config['comparison_review_text_policy']
    elif damage == 'ordinary_history': config['external_history_sha256'] = 'f' * 64
    elif damage == 'prior_sidecar': config['external_component_review_sha256'] = 'f' * 64
    elif damage == 'digest': config['external_component_sha256'] = 'f' * 64
    elif damage == 'disabled_reviews': config['comparison_reviews'] = False
    elif damage == 'old_version': config['comparison_review_version'] = 2
    elif damage == 'no_aux': config['max_aux_calls'] = 0
    else: config['comparison_review_text'] = None
    with pytest.raises(ValueError):
        reflection.GeneratedPairedReflectionJob(saved.component,
            expected_sha256=canonical_sha(saved.component), config=config)


def test_fresh_run_and_non_native_single_author_sequence_are_required(saved):
    old_run = saved.component['proposals'][0]['provenance']['run_id']
    for kwargs in (dict(run_id=old_run, segment_id='fresh'),
                   dict(run_id='fresh', segment_id='fresh', interval=True),
                   dict(run_id='fresh', segment_id='fresh', interval=2)):
        with pytest.raises(ValueError):
            saved.job.prepare(**kwargs)
