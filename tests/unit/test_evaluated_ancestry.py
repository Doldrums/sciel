"""Authenticated retained ancestors constrain reuse, not editable target scope.

All new R/G authors and numeric receipts are explicit synthetic fixtures over
saved sources. No archived code, provider, candidate or solver is executed.
"""
import copy
import json
from types import SimpleNamespace as NS

import pytest

from sciel.engine.db import normalized_hash
from sciel.engine.external_reflection import ExternalReflectionRejected
from sciel.hypothesis import circle_optimizer_traced_body as builder
from sciel.knowledge_base import evaluated_detached_study as core
from sciel.knowledge_base.catalogs import canonical_sha
from test_evaluated_target_study import artifact, prepared, generated, host, no_execution
from test_evaluated_detached_study import synthetic_g_bundle
from test_evaluated_target_generation import reflected, make_job, admit
from test_component_target_generation import (prepared as component_prepared, saved, component,
    pilot_artifacts, response_for, provider_for, TRANSPORT)
from test_component_generation_hypothesis import payload


@pytest.fixture(scope='module')
def next_generation(generated):
    r = reflected(generated, provider_id='synthetic-ancestry-R-provider', run_id='synthetic-ancestry-R')
    job, config = make_job(r)
    plan = job.prepare(run_id='synthetic-ancestry-G', segment_id='synthetic-ancestry-G',
                       transport_settings=TRANSPORT)
    base = plan['source_registry'][plan['target']['base_source_sha256']]['body']
    assert 'restart_index % 16' in base
    value = payload(base.replace('restart_index % 16', 'restart_index % 32', 1))
    value['target_sha256'] = plan['target_sha256']
    value['hypothesis']['paired_forecast']['target_sha256'] = plan['target_sha256']
    value['hypothesis']['text'] = 'SYNTHETIC future edit; not a measured claim.'
    return NS(r=r, job=job, config=config, plan=plan, value=value)


@pytest.fixture(scope='module')
def component_generated(component_prepared, host):
    return synthetic_g_bundle(component_prepared, host)


def ancestor_only_source(bundle):
    parent = bundle['author_dependencies']['source_study_bundle']['capsule']
    digest = parent['author']['base_source_sha256']
    assert digest not in bundle['capsule']['source_registry']
    return digest, parent['source_registry'][digest]


def test_HOST_history_has_sources_but_no_invented_model_IDs(host):
    result = core.retained_ancestry({key:host[key] for key in
        ('capsule', 'author_dependencies', 'author_authority')})
    assert result == dict(source_registry=host['capsule']['source_registry'], provider_response_ids=[])


def test_authenticated_ancestor_sources_and_R_G_IDs_are_detached(generated):
    before = canonical_sha(generated)
    result = core.retained_ancestry(generated)
    ancestor = generated['author_dependencies']['source_study_bundle']
    sources = dict(ancestor['capsule']['source_registry'], **generated['capsule']['source_registry'])
    deps = generated['author_dependencies']
    assert result == dict(source_registry=dict(sorted(sources.items())), provider_response_ids=sorted([
        deps['generation_result']['provider_response']['id'], deps['review_sidecar']['provider_response']['id']]))
    digest, source = ancestor_only_source(generated)
    assert result['source_registry'][digest] == source
    result['source_registry'][digest] = 'caller mutation'
    result['provider_response_ids'].clear()
    assert canonical_sha(generated) == before
    assert core._retained_ancestry(generated)['source_registry'][digest] == source


def test_terminal_component_history_keeps_all_available_G_IDs_without_inventing_R_ID(component_generated):
    result = core.retained_ancestry(component_generated)
    deps = component_generated['author_dependencies']
    expected = dict(component_generated['capsule']['source_registry'])
    ids = {deps['generation_result']['provider_response']['id']}
    for p in deps['source_component']['proposals']:
        for source in (p['parent_source'], p['source']):
            expected[core._digest(source)] = source
        ids.add(p['provider_response']['id'])
    assert 'provider_response' not in deps['review_sidecar']
    assert result == dict(source_registry=dict(sorted(expected.items())), provider_response_ids=sorted(ids))


@pytest.mark.parametrize('damage', ['nested_source', 'outer_authority'])
def test_public_history_authentication_cannot_be_bypassed_by_rehashed_or_changed_input(generated, damage):
    bad = copy.deepcopy(generated)
    if damage == 'nested_source':
        ancestor = bad['author_dependencies']['source_study_bundle']['capsule']
        key = ancestor['author']['base_source_sha256']
        ancestor['source_registry'][key] += '\n'
        bad['author_authority']['dependencies_sha256'] = canonical_sha(bad['author_dependencies'])
    else:
        bad['author_authority']['dependencies_sha256'] = 'f'*64
    with pytest.raises(ValueError):
        core.retained_ancestry(bad)


@pytest.mark.parametrize('normalized_variant', [False, True])
def test_actual_G_admit_rejects_older_source_outside_current_edit_registry(next_generation, normalized_variant):
    p = next_generation
    digest, source = ancestor_only_source(p.r.bundle)
    assert digest not in p.plan['source_registry']
    value = copy.deepcopy(p.value)
    value['body'] = builder.body_of(source)
    if normalized_variant:
        value['body'] += '\n\n'
        assert core._digest(builder.source_for(value['body'])) != digest
        assert normalized_hash(builder.source_for(value['body'])) == normalized_hash(source)
    result = admit(p, value=value, provider=provider_for(response_for(value, p.config['model']),
                                                       'synthetic-fresh-duplicate-G'))
    assert result['status'] == 'duplicate' and result['eligible'] is False
    assert result['error'] == dict(code='historical_duplicate', path='body')
    assert result['proof'] is result['code'] is None
    assert set(result['plan']['source_registry']) == set(p.r.bundle['capsule']['source_registry'])


@pytest.mark.parametrize('author', ['review_sidecar', 'generation_result'])
def test_actual_G_admit_rejects_ancestor_provider_response_relabel(next_generation, author):
    p = next_generation
    old_id = p.r.bundle['author_dependencies'][author]['provider_response']['id']
    assert old_id != p.r.sidecar['provider_response']['id']
    response = response_for(p.value, p.config['model'])
    result = admit(p, response=response, provider=provider_for(response, old_id))
    assert not result['eligible'] and result['code'] is result['proof'] is None
    assert result['error'] == dict(code='historical_provider_replay', path='provider_response.id')
    assert result['provider_response']['id'] == old_id and result['response'] == response


@pytest.mark.parametrize('author', ['review_sidecar', 'generation_result'])
def test_actual_R_admit_rejects_ancestor_ID_and_retains_exact_raw_journal(next_generation, author):
    r = next_generation.r
    old_id = r.bundle['author_dependencies'][author]['provider_response']['id']
    response = copy.deepcopy(r.sidecar['response'])
    with pytest.raises(ExternalReflectionRejected) as captured:
        r.job.admit(r.plan, response, manifest_sha256='e'*64, journal_prefix=r.plan['journal_prefix'],
            provider_request=r.plan['request'], provider_response=provider_for(response, old_id))
    rows = [json.loads(line) for line in captured.value.journal.splitlines()]
    assert rows[-2]['phase'] == 'returned' and rows[-2]['response'] == response['text']
    assert rows[-1]['status'] == 'invalid' and rows[-1]['review_card_sha256'] is None
    assert rows[-1]['error'] == dict(code='historical_provider_replay', path='provider_response.id')
    assert rows[-1]['new_lesson_ids'] == [] and rows[-1]['new_ordinary_outcomes'] == 0


def test_fresh_R_and_G_still_admit_without_expanding_target_scope(next_generation):
    p = next_generation
    assert p.r.sidecar['provider_response']['id'] == 'synthetic-ancestry-R-provider'
    response = response_for(p.value, p.config['model'])
    result = admit(p, response=response, provider=provider_for(response, 'synthetic-ancestry-G-provider'))
    assert result['eligible'] and result['proof']['parent_source_sha256'] == p.plan['target']['base_source_sha256']
    assert result['provider_dispatch_verified'] is result['execution_performed'] is False


def test_public_worker_consumes_ancestor_sources_even_when_memory_disabled(generated, tmp_path):
    from test_evaluated_study_integration import construct, import_study
    artifact = dict(generated, capsule_sha256=canonical_sha(generated['capsule']),
                    author_authority_sha256=canonical_sha(generated['author_authority']))
    receiver = construct(artifact, tmp_path, inject_lessons=False)
    try:
        assert import_study(receiver, artifact)
        digest, source = ancestor_only_source(generated)
        consumed, normalized = receiver.worker._source_history([], ())
        assert digest in consumed and normalized_hash(source) in normalized
        context = receiver.c.context(receiver.db, receiver.parent, 0, 1)
        assert context.get('evaluated_detached_study') is None
        assert receiver.worker.calls_started == receiver.c.attempts == 0
        assert receiver.model.calls == receiver.charges == []
    finally:
        receiver.log.close()
