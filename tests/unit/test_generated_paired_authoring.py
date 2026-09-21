"""Authored source identities and fixed observations in the R appendix.

Preparation/admission is pure; runtime and provider execution are forbidden.
"""
import copy
import hashlib
import json
from types import SimpleNamespace as NS

import pytest

from sciel.engine.external_reflection import ExternalReflectionRejected
from sciel.engine import generated_paired_reflection as reflection
from sciel.hypothesis import generated_paired_review as review
from sciel.hypothesis import circle_optimizer_traced_body as builder
from sciel.hypothesis.comparison_targets import QUALITY_GATE, SCHEMA as TARGET_SCHEMA
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
from sciel.knowledge_base.generated_paired_history import proposal_records
from test_generated_paired_reflection import no_runtime, saved


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


@pytest.fixture(scope='module')
def authors(saved):
    value = review.authoring_context(saved.component)
    originals = {row['provenance']['request_key']: row for row in saved.component['proposals']}
    catalogs = {g['id']: review.source_catalog(saved.component, g['id'])
                for g in saved.component['comparisons']}
    return NS(value=value, originals=originals,
        sources={row['source_sha256']: row for row in value['sources']},
        groups={row['branch']: row for row in value['groups']},
        records=proposal_records(saved.component), catalogs=catalogs)


def token(value):
    if value is None: return 'unknown'
    if type(value) is bool: return str(value).lower()
    return str(value)


def test_actual_registry_contains_exact_complete_bodies_and_raw_authors(saved, authors):
    assert authors.value['source_component_sha256'] == canonical_sha(saved.component)
    assert len(authors.sources) == 3
    assert len({v['source_ref'] for v in authors.sources.values()}) == 3
    for arm, original in authors.originals.items():
        record = authors.records[original['id']]
        source = authors.sources[sha(original['source'])]
        authored = json.loads(original['response']['text'])
        assert source['editable_body'] == dict(body=authored['body'], body_sha256=sha(authored['body']))
        assert builder.source_for(source['editable_body']['body']) == original['source']
        assert source['author'] == dict(proposal_id=original['id'],
            request_sha256=canonical_sha(original['request']),
            raw_response_sha256=sha(original['response']['text']),
            unverified_hypothesis=record['hypothesis'])
        assert record['hypothesis']['text'] == authored['hypothesis']['hypothesis']
    control_sha = saved.component['root_parent']['source_sha256']
    control = authors.sources[control_sha]
    assert control['author'] is None and control['editable_body']['body'].strip() == 'return None'
    assert builder.source_for(control['editable_body']['body']) == authors.originals['off']['parent_source']
    # Local research labels collide, but source-bound author identities do not.
    assert authors.originals['off']['provenance']['research_id'] == authors.originals['on']['provenance']['research_id']
    assert authors.originals['off']['id'] != authors.originals['on']['id']


def test_actual_role_changes_keep_source_alias_code_and_author_fixed(authors):
    off = authors.sources[sha(authors.originals['off']['source'])]
    on = authors.sources[sha(authors.originals['on']['source'])]
    for branch, role, source in [('on_off', 'comparator', off), ('on_off', 'challenger', on),
                                 ('on_control', 'challenger', on), ('off_control', 'challenger', off)]:
        assert authors.groups[branch]['roles'][role] == dict(source_ref=source['source_ref'],
            source_sha256=source['source_sha256'], proposal_id=source['author']['proposal_id'])
    controls = [authors.groups[branch]['roles']['comparator'] for branch in ('on_control', 'off_control')]
    assert controls[0] == controls[1] and controls[0]['proposal_id'] is None
    assert 'restart_index % 8' in off['editable_body']['body']
    assert 'np.copy(best_params)' in off['editable_body']['body']
    assert 'restart_index % 12' in on['editable_body']['body']
    assert 'np.copy(initial_params)' in on['editable_body']['body']
    # The host registry exposes exact code, not an automatically proven mechanism label.
    assert all('mechanism' not in source for source in authors.sources.values())


def test_every_forecast_triple_is_bound_to_same_catalog_role_seed_and_receipt(authors):
    count = 0
    for group in authors.groups.values():
        catalog = authors.catalogs[group['comparison_id']]
        assert group['comparison_sha256'] == catalog['comparison_sha256']
        forecasts = {f['role']: f for f in catalog['authored_forecasts']}
        for row in group['forecast_observations']:
            count += 1
            original = forecasts[row['role']]
            observed = next(x for x in original['rows'] if x['seed'] == row['seed'])
            source = authors.sources[row['source_sha256']]
            assert row['source_ref'] == source['source_ref']
            assert row['proposal_id'] == original['proposal_id'] == source['author']['proposal_id']
            assert row['receipt_sha256'] == observed['receipt_sha256']
            assert row['unavailable_reason'] == observed['unavailable_reason']
            result = observed['result']
            assert row['result_sha256'] == (result['result_sha256'] if result else None)
            assert row['coverage'] == (result['coverage'] if result else None)
            assert len(row['checks']) == 3
            for check in row['checks']:
                name = check['observable']
                assert check['expected'] is original['spec']['expected'][name]
                assert check['observed'] is (result['observed'][name] if result else None)
                assert check['forecast_verdict'] == (result['verdicts'][name] if result else 'INCONCLUSIVE')
                for field, data in [('expected', check['expected']), ('observed', check['observed']),
                                    ('verdict', check['forecast_verdict'])]:
                    assert catalog['facts'][check['fact_ids'][field]] == token(data)
    assert count == 12  # Shared source/seed observations appear in their two group routes.


def test_actual_unknown_and_refuted_forecasts_are_not_inverted(authors):
    first = [r for r in authors.groups['on_off']['forecast_observations'] if r['seed'] == 763606949]
    by_role = {r['role']: {c['observable']: c for c in r['checks']} for r in first}
    triple = lambda c: (c['expected'], c['observed'], c['forecast_verdict'])
    assert triple(by_role['comparator']['result_changed']) == (False, True, 'REFUTED')
    assert triple(by_role['challenger']['result_changed']) == (False, None, 'INCONCLUSIVE')
    assert triple(by_role['comparator']['accepted_interval_gain']) == (False, False, 'CONFIRMED')
    assert triple(by_role['challenger']['accepted_interval_gain']) == (True, False, 'REFUTED')
    for group in authors.groups.values():
        for row in group['forecast_observations']:
            if row['seed'] != 763606949:
                assert row['receipt_sha256'] is row['result_sha256'] is row['coverage'] is None
                assert row['unavailable_reason'] == 'receipt_not_dispatched'
                assert all(c['observed'] is None and c['forecast_verdict'] == 'INCONCLUSIVE' for c in row['checks'])


def test_authoring_appendix_in_literal_prompt_and_journal_is_complete(saved, authors):
    value, _ = json.JSONDecoder().raw_decode(saved.plan['prompt'].split(review.AUTHORING_PREFIX, 1)[1])
    assert value == authors.value
    block = review.authoring_block(saved.component)
    assert block in saved.plan['prompt'] and json.loads(block[len(review.AUTHORING_PREFIX):]) == authors.value
    events = list(map(json.loads, saved.plan['journal_prefix'].splitlines()))
    context, = [e for e in events if e.get('phase') == 'generated_paired_reflection_context']
    assert context['authoring_block_sha256'] == sha(block)
    assert saved.plan['schema'] == 'sciel-generated-paired-reflection-plan/2'
    assert saved.plan['response_format']['json_schema']['name'] == 'sciel_generated_paired_reflection_v2'
    assert saved.plan['new_ordinary_outcomes'] == saved.plan['new_search_attempts'] == 0
    assert 'Mixed, tied or refuted evidence may motivate' in saved.plan['prompt']
    assert 'return null with a concrete reason' in saved.plan['prompt']


def test_wire_contains_exact_union_but_excludes_metadata(saved, authors):
    vocabulary = review.fact_vocabulary(saved.component)
    catalogs = list(authors.catalogs.values())
    assert vocabulary['fact_ids'] == sorted({key for c in catalogs for key in c['facts']})
    assert not {'comparison_id', 'comparison_sha256', 'source_sha256'} & set(vocabulary['fact_ids'])
    wire = reflection.response_format(saved.component)
    proposal = wire['json_schema']['schema']['properties']['comparison_reviews']['items']['properties']
    assert proposal['comparison_id']['enum'] == sorted(authors.catalogs)
    assert proposal['comparison_sha256']['enum'] == sorted(c['comparison_sha256'] for c in catalogs)
    assert proposal['claims']['items']['properties']['fact_id']['enum'] == vocabulary['fact_ids']
    for key in ('base_source_sha256', 'comparator_source_sha256'):
        assert proposal['next_hypothesis']['properties'][key]['enum'] == sorted(authors.sources)
    assert wire == saved.plan['response_format']


@pytest.mark.parametrize('damage', ['foreign_fact', 'foreign_group_sha', 'foreign_target_source', 'metadata_fact'])
def test_union_membership_cannot_admit_cross_group_or_metadata_claims(saved, authors, damage):
    group = authors.groups['on_control']
    row = dict(comparison_id=group['comparison_id'], comparison_sha256=group['comparison_sha256'],
        claims=[dict(fact_id='quality:strict_verdict', value='REFUTED')],
        conjecture='Synthetic prospective interpretation; no causal assertion.',
        limitations='One dispatched paired seed and no native outcome.', next_hypothesis=None,
        next_hypothesis_reason='This test fixture abstains without requesting execution.')
    if damage == 'foreign_fact':
        fact = 'restart:comparator:763606949:result_changed:observed'
        assert fact in review.fact_vocabulary(saved.component)['fact_ids']
        assert fact not in authors.catalogs[group['comparison_id']]['facts']
        row['claims'] = [dict(fact_id=fact, value='true')]
    elif damage == 'foreign_group_sha':
        row['comparison_sha256'] = authors.groups['off_control']['comparison_sha256']
    elif damage == 'metadata_fact':
        row['claims'] = [dict(fact_id='comparison_id', value=group['comparison_id'])]
    else:
        off_sha = sha(authors.originals['off']['source'])
        assert off_sha in review.fact_vocabulary(saved.component)['source_sha256']
        row['next_hypothesis'] = dict(schema=TARGET_SCHEMA, base_source_sha256=off_sha,
            comparator_source_sha256=group['roles']['comparator']['source_sha256'],
            change='A bounded synthetic intervention on a source outside this group.',
            quality_gate=QUALITY_GATE, diagnostics=[])
    before = canonical_json([saved.component, saved.config, saved.plan])
    response = dict(text=json.dumps(dict(lessons=[], retire_ids=[], comparison_reviews=[row])),
        model=saved.config['model'], cost=.002, prompt_tokens=100, completion_tokens=50,
        finish_reason='stop', refusal=None)
    with pytest.raises(ExternalReflectionRejected) as failure:
        saved.job.admit(saved.plan, response, manifest_sha256='a' * 64,
                        journal_prefix=saved.plan['journal_prefix'])
    terminal = json.loads(failure.value.journal.splitlines()[-1])
    assert terminal['comparison_review_ids'] == [] and terminal['error']
    assert terminal['new_lesson_ids'] == terminal['retired_lesson_ids'] == []
    assert canonical_json([saved.component, saved.config, saved.plan]) == before


def test_generic_v1_wire_retains_its_public_compatibility_pin():
    current = reflection.response_format()
    assert current['json_schema']['name'] == 'sciel_generated_paired_reflection_v1'
    # response_format's AST is unchanged from the pre-refactor implementation.
    assert canonical_sha(current) == '9e3c04a0ca462b5327ea530184a0207317229c3efe9f89ac255da3acccc1e903'


def test_authoring_result_is_detached_and_does_not_enlarge_stored_host_catalog(saved, authors):
    before = copy.deepcopy(authors.catalogs)
    result = review.authoring_context(saved.component)
    result['sources'][0]['editable_body']['body'] = 'forged caller mutation'
    result['groups'][0]['roles'].clear()
    assert review.authoring_context(saved.component) == authors.value
    for ident, catalog in before.items():
        assert not {'sources', 'editable_body', 'authoring_context'} & set(catalog)
        assert review.source_catalog(saved.component, ident) == catalog
