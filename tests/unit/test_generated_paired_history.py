"""Saved literal proposals/geometry only: no provider, candidate or subprocess."""
import copy
import json
from types import SimpleNamespace

import pytest

from sciel.eval import packing_geometry as geometry
from sciel.hypothesis.measurement import metric_context
from sciel.knowledge_base import generated_paired_history as history
from sciel.knowledge_base.catalogs import canonical_sha
from sciel.knowledge_base.paired_comparisons import build_comparison

LIMITS = dict(candidate_wall_seconds=30., certification_wall_seconds=10.,
    candidate_cpu_seconds=35, certification_cpu_seconds=35,
    candidate_mem_bytes=2147483648, certification_mem_bytes=2147483648)
ADAPTER = {'schema': 'sciel-repeat-adapter/1', 'fixture': 'pure saved geometry; no runtime',
           'certificate_policy': history.certificate_policy(),
           'certificate_policy_sha256': canonical_sha(history.certificate_policy())}
ADAPTER['identity_sha256'] = canonical_sha(ADAPTER)
SEEDS = (11, 12, 13)


def proposal(key, body, metric, parent=None):
    parent = parent or history.builder.source_for()
    raw = json.dumps(dict(body=body, hypothesis=dict(hypothesis='A bounded prospective change.',
        predicted_low=-.01, predicted_high=.01, confidence=.5,
        reasoning='An unmeasured expectation, not an achieved outcome.',
        restart_predictions={'schema': 'sciel-restart-predictions/1', 'expected': {
            'effective_locations_changed': True, 'result_changed': False, 'accepted_interval_gain': True}})))
    response = dict(text=raw, model='fixture/model', cost=.01, prompt_tokens=200,
                    completion_tokens=50, finish_reason='stop', refusal=None)
    provenance = dict(manifest_sha256='a' * 64, run_id='synthetic-study', request_key=key,
                      research_id='r000003a1')
    built = history.builder.apply(parent, body)
    context = dict(parent_source_sha256=history._digest(parent), metric=metric,
                   parent_score=None, parent_cases=None, source_scope={'contract': 'circle_restart_body/1'})
    prompt = 'Original unmodified synthetic prompt.\nRESEARCH_CONTEXT_JSON:\n' + json.dumps(context)
    return dict(id=history.proposal_id(provenance, response), provenance=provenance,
        request=dict(model=response['model'], messages=[dict(role='user', content=prompt)],
            temperature=.2, max_tokens=8192, response_format=history.response_format()), response=response,
        provider_response=dict(id='synthetic-provider-' + key, model=response['model'], choices=[dict(index=0,
            finish_reason='stop', message=dict(role='assistant', content=raw, refusal=None))],
            usage={k: response[k] for k in ('cost', 'prompt_tokens', 'completion_tokens')}),
        parent_source=parent, source=built['code'], proof=built['proof'])


def receipt(source, seed, radius=.1):
    centers = [[0., 0.] for _ in range(26)]
    centers[0] = [.5, .5]
    capture = geometry.validate_packing(dict(centers=centers, radii=[radius] + [0.] * 25))
    certificate = geometry.certify_packing(capture)
    return dict(schema='sciel-repeat-draw/1', source_sha256=history._digest(source), seed=seed,
        adapter_identity=copy.deepcopy(ADAPTER), limits=copy.deepcopy(LIMITS),
        seed_applied=dict(python_random=seed, numpy_legacy=seed, python_hash_seed=str(seed),
                          before_candidate_import=True, determinism_proven=False),
        status='valid', reason=None, capture=capture, certificate=certificate,
        raw_sum=capture['sum_radii'], certified_sum=certificate['sum_exact'],
        started_at='2026-09-18T00:00:00+00:00', completed_at='2026-09-18T00:00:01+00:00',
        elapsed_seconds=1., candidate_elapsed_seconds=.8, certification_elapsed_seconds=.1,
        identity_elapsed_seconds=0.)


def group(capsule, name, left, right, *, left_radius=.1, right_radius=.11, incomplete=False):
    return build_comparison(namespace='synthetic-generated-paired-study', batch_id=name,
        branch='foreign-' + name, origin_experiment_id=None,
        comparator_source_sha256=history._digest(left), challenger_source_sha256=history._digest(right),
        adapter_identity=ADAPTER, limits=LIMITS,
        pairs=[dict(seed=seed, comparator=None if incomplete and i else receipt(left, seed, left_radius),
                    challenger=None if incomplete and i else receipt(right, seed, right_radius)) for i, seed in enumerate(SEEDS)],
        source_pair=dict(comparator=left, challenger=right), task_metric=capsule['metric'],
        provenance={'scope': 'Synthetic saved data, no execution or dispatch assertion.'})


@pytest.fixture
def capsule():
    metric = metric_context(SimpleNamespace(qualified_name='oe/circle_packing', task_hash='c' * 64,
        primary_metric='combined_score', research_metric=None))
    root = history.builder.source_for()
    value = dict(schema=history.SCHEMA, profile=history.PROFILE, metric=metric,
        root_parent=dict(id='p1', source_sha256=history._digest(root)), policy=history.identity(),
        proposals=[proposal('first-independent-call', 'return best_params.copy()\n', metric),
                   proposal('second-independent-call', 'return initial_params.copy()\n', metric)], comparisons=[])
    a, b = [p['source'] for p in value['proposals']]
    value['comparisons'] = [group(value, 'a-root', root, a, right_radius=.11),
        group(value, 'b-root', root, b, right_radius=.12),
        group(value, 'b-a', a, b, left_radius=.11, right_radius=.12)]
    return value


def validate(capsule, **kwargs):
    return history.validate_capsule(capsule, expected_sha256=canonical_sha(capsule),
                                    metric=capsule['metric'], **kwargs)


def test_whole_annex_detached_exact_provenance_and_no_fake_native_outcomes(capsule):
    saved = copy.deepcopy(capsule)
    result = validate(capsule, root=capsule['root_parent'])
    assert result == saved and result is not capsule
    result['proposals'][0]['provenance']['research_id'] = 'changed'
    assert capsule == saved
    records = history.proposal_records(capsule)
    assert len(records) == 2
    assert all(key.startswith('generated:') for key in records)
    assert {r['provenance']['research_id'] for r in records.values()} == {'r000003a1'}
    assert all(r['native_observation'] == dict(score=None, actual_delta=None, evaluation=None,
        forecast_verdict='INCONCLUSIVE', forecast_scored=False, reason='native target unmeasured') for r in records.values())
    assert all('record' not in r and 'evaluation' not in r for r in records.values())
    facts = history.derive_facts(capsule)
    assert facts['root_parent'] == capsule['root_parent']
    assert [x['group'] for x in facts['comparisons']] == capsule['comparisons']
    assert all(x['quality']['verdict'] == 'CONFIRMED' for x in facts['comparisons'])
    assert all(g['origin_experiment_id'] is None for g in capsule['comparisons'])
    assert all(g['branch'].startswith('foreign-') for g in capsule['comparisons'])
    assert capsule == saved


def test_request_names_are_generic_and_ids_include_raw_response_namespace(capsule):
    a, b = capsule['proposals']
    assert a['provenance']['research_id'] == b['provenance']['research_id']
    assert a['id'] != b['id']
    modified = copy.deepcopy(a['response']); modified['text'] += ' '
    assert history.proposal_id(a['provenance'], modified) != a['id']
    validate(capsule)


def test_provider_response_is_not_a_claim_of_dispatch_or_payment(capsule):
    assert 'proof of provider dispatch' in history.__doc__
    records = history.proposal_records(capsule)
    assert all('provider_dispatch_verified' not in r and 'paid_calls' not in r for r in records.values())
    assert 'No execution or promotion authority' in history.derive_facts(capsule)['scope']


@pytest.mark.parametrize('mutation', [
    lambda p: p['response'].__setitem__('text', p['response']['text'] + ' changed'),
    lambda p: p['provider_response']['choices'][0]['message'].__setitem__('content', 'different'),
    lambda p: p['provider_response']['usage'].__setitem__('cost', .02),
    lambda p: p['provider_response']['usage'].__setitem__('prompt_tokens', True),
    lambda p: p['response'].__setitem__('finish_reason', 'length'),
    lambda p: p['response'].__setitem__('refusal', 'refused'),
    lambda p: p['request'].__setitem__('model', 'different'),
    lambda p: p['request'].__setitem__('temperature', True),
    lambda p: p['request'].__setitem__('max_tokens', False),
    lambda p: p['request'].__setitem__('response_format', {'type': 'json_object'}),
    lambda p: p['proof']['before']['byte_span'].__setitem__(0, True),
    lambda p: p.__setitem__('source', p['source'] + '\n# altered outside typed body\n'),
    lambda p: p.__setitem__('parent_source', history.builder.source_for('return ordinary_params.copy()\n')),
    lambda p: p.__setitem__('id', 'generated:' + '0' * 64),
    lambda p: p['provenance'].__setitem__('manifest_sha256', 'x' * 64),
])
def test_raw_request_response_source_proof_tamper_rejected_even_with_rehashed_annex(capsule, mutation):
    mutation(capsule['proposals'][0])
    before = copy.deepcopy(capsule)
    with pytest.raises(ValueError):
        validate(capsule)
    assert capsule == before


def test_missing_restart_predictions_rejected_before_native_invention(capsule):
    p = capsule['proposals'][0]
    raw = json.loads(p['response']['text']); del raw['hypothesis']['restart_predictions']
    p['response']['text'] = json.dumps(raw)
    p['provider_response']['choices'][0]['message']['content'] = p['response']['text']
    p['id'] = history.proposal_id(p['provenance'], p['response'])
    with pytest.raises(ValueError, match='hypothesis/body'):
        validate(capsule)


def test_duplicate_dispatch_and_provider_identity_rejected(capsule):
    a, b = capsule['proposals']
    b['provenance'] = copy.deepcopy(a['provenance'])
    b['id'] = history.proposal_id(b['provenance'], b['response'])
    with pytest.raises(ValueError, match='duplicate'):
        validate(capsule)
    b['provenance']['request_key'] = 'other'; b['id'] = history.proposal_id(b['provenance'], b['response'])
    b['provider_response']['id'] = a['provider_response']['id']
    with pytest.raises(ValueError, match='duplicate'):
        validate(capsule)


def test_unbound_config_root_metric_and_policy_fail_closed(capsule):
    with pytest.raises(ValueError, match='configured capsule'):
        history.validate_capsule(capsule, expected_sha256='0' * 64, metric=capsule['metric'])
    with pytest.raises(ValueError, match='receiving root'):
        validate(capsule, root={**capsule['root_parent'], 'id': 'foreign'})
    capsule['policy']['production_python_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='policy'):
        validate(capsule)


def test_group_origin_cannot_claim_an_ordinary_experiment(capsule):
    old = capsule['comparisons'][0]
    fields = {k: old[k] for k in ('namespace','batch_id','branch','origin_experiment_id',
        'comparator_source_sha256','challenger_source_sha256','adapter_identity','limits','pairs','provenance','task_metric','source_pair')}
    fields['origin_experiment_id'] = 'r000003a1'
    capsule['comparisons'][0] = build_comparison(**fields)
    with pytest.raises(ValueError, match='origin'):
        validate(capsule)


def test_shared_source_seed_requires_same_receipt_not_another_replication(capsule):
    root = history.builder.source_for()
    b = capsule['proposals'][1]['source']
    capsule['comparisons'][1] = group(capsule, 'b-root', root, b, left_radius=.099, right_radius=.12)
    with pytest.raises(ValueError, match='shared draw receipt'):
        validate(capsule)


def test_missing_and_unknown_are_distinct_and_known_loss_wins_incomplete(capsule):
    root = history.builder.source_for()
    a, b = [p['source'] for p in capsule['proposals']]
    capsule['comparisons'] = [group(capsule, 'a-root', root, a, left_radius=.1, right_radius=.09, incomplete=True),
        group(capsule, 'b-root', root, b, left_radius=.1, right_radius=.11, incomplete=True)]
    facts = history.derive_facts(capsule)
    assert facts['comparisons'][0]['group']['derived']['outcome'] == 'incomplete'
    assert facts['comparisons'][0]['quality']['reason'] == 'known_paired_loss'
    assert facts['comparisons'][0]['quality']['verdict'] == 'REFUTED'
    assert facts['comparisons'][1]['quality']['verdict'] == 'INCONCLUSIVE'
    row = facts['comparisons'][0]['restart_predictions'][0]['rows'][1]
    assert row['unavailable_reason'] == 'receipt_not_dispatched' and row['result'] is None
    g = capsule['comparisons'][0]
    fields = {k: g[k] for k in ('namespace','batch_id','branch','origin_experiment_id',
        'comparator_source_sha256','challenger_source_sha256','adapter_identity','limits','pairs','provenance','task_metric','source_pair')}
    fields['pairs'][1]['challenger'] = copy.deepcopy(history.UNKNOWN_RECEIPT)
    capsule['comparisons'][0] = build_comparison(**fields)
    row = history.derive_facts(capsule)['comparisons'][0]['restart_predictions'][0]['rows'][1]
    assert row['unavailable_reason'] == 'dispatch_unknown' and row['receipt_sha256'] is not None


def test_restart_projection_passes_exact_source_seed_adapter_and_scaffold(capsule, monkeypatch):
    calls = []
    original = history.observations.build_observation
    def observe(receipt, **kwargs):
        calls.append(kwargs)
        return original(receipt, **kwargs)
    monkeypatch.setattr(history.observations, 'build_observation', observe)
    facts = history.derive_facts(capsule)
    records = history.proposal_records(capsule)
    assert len(calls) == 12  # four source-role appearances, each three seeds; not independent draws
    assert {x['source_sha256'] for x in calls} == {r['source_sha256'] for r in records.values()}
    assert all(x['adapter_identity'] == ADAPTER and x['seed'] in SEEDS for x in calls)
    assert {x['scaffold_sha256'] for x in calls} == {r['proof']['common_scaffold_sha256'] for r in records.values()}
    assert all(row['unavailable_reason'] for g in facts['comparisons'] for p in g['restart_predictions'] for row in p['rows'])


@pytest.mark.parametrize('bad', [None, [], {'x': float('nan')}, {'x': b'not-json'}])
def test_malformed_capsules_have_value_error(bad):
    with pytest.raises(ValueError):
        history.validate_capsule(bad, expected_sha256='0' * 64, metric={})


def test_proposal_capacity_and_orphan_proposal_fail_closed(capsule):
    capsule['proposals'] = capsule['proposals'][:1]
    with pytest.raises(ValueError, match='capacity'):
        validate(capsule)


def test_unused_proposal_not_silently_imported(capsule):
    capsule['comparisons'] = capsule['comparisons'][:1]
    with pytest.raises(ValueError, match='no planned paired comparison'):
        validate(capsule)


@pytest.mark.parametrize('field', ['parent', 'metric', 'source_scope', 'missing_context', 'duplicate_key'])
def test_raw_request_context_cannot_be_cross_wired(capsule, field):
    message = capsule['proposals'][0]['request']['messages'][0]
    prefix, encoded = message['content'].split('RESEARCH_CONTEXT_JSON:\n')
    context = json.loads(encoded)
    if field == 'parent':
        context['parent_source_sha256'] = capsule['proposals'][1]['proof']['child_source_sha256']
    elif field == 'metric':
        context['metric'] = metric_context(SimpleNamespace(qualified_name='oe/circle_packing', task_hash='d' * 64,
            primary_metric='combined_score', research_metric=None))
    elif field == 'source_scope':
        context['source_scope']['contract'] = 'different'
    message['content'] = prefix + 'RESEARCH_CONTEXT_JSON:\n' + json.dumps(context)
    if field == 'missing_context':
        message['content'] = 'An unrelated request without bound research context.'
    if field == 'duplicate_key':
        message['content'] = prefix + 'RESEARCH_CONTEXT_JSON:\n' + json.dumps(context)[:-1] + ',"parent_score":null}'
    with pytest.raises(ValueError):
        validate(capsule)


@pytest.mark.parametrize('missing', [True, False])
def test_certificate_policy_missing_or_relabelled_cannot_be_presented_as_fact(capsule, missing):
    # Rebuild all affected group/receipt hashes; the numerical geometry itself is valid.
    adapter = copy.deepcopy(ADAPTER)
    if missing:
        del adapter['certificate_policy']
    else:
        adapter['certificate_policy']['raw_tolerance'] = .5
        adapter['certificate_policy_sha256'] = canonical_sha(adapter['certificate_policy'])
    adapter['identity_sha256'] = canonical_sha({k: v for k, v in adapter.items() if k != 'identity_sha256'})
    for index, g in enumerate(capsule['comparisons']):
        fields = {k: copy.deepcopy(g[k]) for k in ('namespace','batch_id','branch','origin_experiment_id',
            'comparator_source_sha256','challenger_source_sha256','adapter_identity','limits','pairs','provenance','task_metric','source_pair')}
        fields['adapter_identity'] = copy.deepcopy(adapter)
        for pair in fields['pairs']:
            for role in ('comparator', 'challenger'):
                pair[role]['adapter_identity'] = copy.deepcopy(adapter)
        capsule['comparisons'][index] = build_comparison(**fields)
    with pytest.raises(ValueError, match='certificate policy'):
        validate(capsule)
