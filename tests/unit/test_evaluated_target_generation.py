"""Synthetic accepted R/G over authored HOST facts; no transport or solver.

Acceptance is an explicit unit fixture. A prospective body and its forecasts
here are not measured evidence.
"""
import copy
import json
from types import SimpleNamespace as NS

import pytest

from sciel.engine import evaluated_target_generation as generation
from sciel.engine.evaluated_study_reflection import EvaluatedStudyReflectionJob
from sciel.engine.external_research_v2 import policy_identity
from sciel.hypothesis import circle_optimizer_traced_body as builder
from sciel.hypothesis import evaluated_study_review as review
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
from test_component_generation_hypothesis import payload
from test_component_target_generation import TRANSPORT, response_for, provider_for



@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Evaluated target fixture attempted provider or numerical execution.')
    for name in ('subprocess.Popen', 'sciel.eval.repeat_draw.draw', 'sciel.eval.repeat_draw.adapter_identity',
                 'sciel.engine.llm.client.EnsembleLLM.generate'):
        monkeypatch.setattr(name, forbidden)


@pytest.fixture(scope='module')
def artifact():
    from evaluated_study_fixture import build
    built = build()
    return {key:built[key] for key in ('capsule','author_dependencies','author_authority')}


def r_config(bundle):
    return dict(model='fixture-model', external_evaluated_study_sha256=canonical_sha(bundle['capsule']),
        external_evaluated_study_authority_sha256=canonical_sha(bundle['author_authority']),
        restart_predictions=True, structured_auxiliary=True, comparison_reviews=True, inject_lessons=True,
        batch_only_lessons=True, comparison_review_text=True, comparison_review_version=3,
        max_aux_calls=1, comparison_review_text_policy=dict(schema='sciel-review-text-policy/1',decisions=[]))


def reflected(bundle, *, null_target=False, provider_id='synthetic-R-provider', run_id='synthetic-evaluated-R'):
    config = r_config(bundle)
    job = EvaluatedStudyReflectionJob(bundle,
        expected_sha256=config['external_evaluated_study_sha256'],
        expected_authority_sha256=config['external_evaluated_study_authority_sha256'], config=config)
    plan = job.prepare(run_id=run_id, segment_id='synthetic-R', transport_settings=TRANSPORT)
    facts = review.source_catalog(bundle)['facts']
    fact = next(key for key,value in facts.items() if key.startswith('quality:') and value['value']=='REFUTED')
    target = None if null_target else dict(
        base_source_sha256=bundle['capsule']['author']['candidate_source_sha256'],
        comparator_source_sha256=bundle['capsule']['author']['comparator_source_sha256'],
        change='SYNTHETIC fixture: change the restart cadence from 8 to 16.', diagnostics=[],
        motivating_evidence=[fact])
    proposal = dict(claims=[dict(fact_id=fact,value=facts[fact]['value'])],
        conjecture='Synthetic admission example, not a verified causal explanation.',
        limitations='This fixture proves no numerical or memory benefit.',
        next_hypothesis_reason='Synthetic source-edit test; no execution is requested.',
        evidence_refs={key:[fact] for key in ('conjecture','limitations','next_hypothesis_reason')},
        next_hypothesis=target)
    response = response_for(proposal,config['model'])
    result = job.admit(plan,response,manifest_sha256='e'*64,journal_prefix=plan['journal_prefix'],
        provider_request=plan['request'],provider_response=provider_for(response,provider_id))
    return NS(bundle=bundle,config=config,job=job,plan=plan,proposal=proposal,
              sidecar=result['extended_capsule'])


def approval(card, *, status='accepted'):
    decision = dict(schema='sciel-review-text-decision/1',scope='comparison_review_text',decision=status,
        card_sha256=canonical_sha(card),raw_response_sha256=card['writer']['raw_response_sha256'],
        policy_sha256=canonical_sha(policy_identity()),audit_sha256='a'*64,criteria_sha256='b'*64,
        issuer='SYNTHETIC TEST AUTHORITY')
    for key in ('comparison_id','comparison_sha256','comparator_source_sha256','challenger_source_sha256'):
        decision[key]=card[key]
    return dict(schema='sciel-review-text-policy/1',decisions=[decision])


def make_job(r, *, config=None, reflection_config=None, sidecar=None, card_sha=None):
    sidecar = r.sidecar if sidecar is None else sidecar
    card = sidecar['review_card']
    cfg = dict(r.config,external_evaluated_review_sha256=canonical_sha(sidecar),
        source_reflection_config_sha256=canonical_sha(r.config),comparison_review_text_policy=approval(card))
    if config is not None:
        cfg.update(config)
    job = generation.EvaluatedTargetGenerationJob(r.bundle,sidecar,
        reflection_config=r.config if reflection_config is None else reflection_config,
        expected_sha256=canonical_sha(r.bundle['capsule']),
        expected_authority_sha256=canonical_sha(r.bundle['author_authority']),
        expected_sidecar_sha256=canonical_sha(sidecar),expected_card_sha256=card_sha or canonical_sha(card),
        config=cfg,task_prompt='Construct a packing of 26 circles in a unit square.')
    return job,cfg


@pytest.fixture(scope='module')
def prepared(artifact):
    r = reflected(artifact)
    job,cfg = make_job(r)
    plan = job.prepare(run_id='synthetic-evaluated-G',segment_id='synthetic-G',transport_settings=TRANSPORT)
    base = plan['source_registry'][plan['target']['base_source_sha256']]['body']
    assert 'restart_index % 8' in base
    body = base.replace('restart_index % 8','restart_index % 16',1)
    value = payload(body)
    value['target_sha256'] = plan['target_sha256']
    value['hypothesis']['paired_forecast']['target_sha256'] = plan['target_sha256']
    value['hypothesis']['text'] = 'SYNTHETIC fixture, no measured outcome.'
    return NS(r=r,job=job,config=cfg,plan=plan,value=value)


def admit(p, *, value=None, response=None, provider=None, request=None, plan=None):
    plan = p.plan if plan is None else plan
    response = response_for(p.value if value is None else value,plan['request']['model']) if response is None else response
    return p.job.admit(plan,response,manifest_sha256='d'*64,journal_prefix=plan['journal_prefix'],
        provider_request=plan['request'] if request is None else request,
        provider_response=provider_for(response) if provider is None else provider)


def test_real_HOST_facts_synthetic_R_to_G_exact_base_and_no_native_measurement(prepared):
    p=prepared;target=p.plan['target'];result=admit(p)
    assert result['schema']==generation.SCHEMA and result['eligible'] is True
    assert result['generation_id'].startswith('evaluated-generated:')
    assert target['base_source_sha256']!=target['comparator_source_sha256']
    assert result['proof']['parent_source_sha256']==target['base_source_sha256']
    assert builder.apply(p.plan['source_registry'][target['base_source_sha256']]['source'],result['body'])['code']==result['code']
    intent,_=json.JSONDecoder().raw_decode(p.plan['prompt'].split('EVALUATED_TARGET_CONTEXT_JSON:\n',1)[1])
    delivery=intent['selected_review'];card=p.r.sidecar['review_card']
    assert delivery['schema']=='sciel-evaluated-study-review-delivery/1'
    assert delivery['card_sha256']==canonical_sha(card)
    assert delivery['card_fields']=={key:value for key,value in card.items() if key!='host_facts'}
    catalog,_=json.JSONDecoder().raw_decode(p.plan['prompt'].split(review.PREFIX,1)[1])
    restored={key:value for key,value in catalog.items() if key not in ('fact_tables','catalog_sha256')}
    restored['facts']={row['prefix']+':'+column['name']:dict(kind=column['kind'],unit=column['unit'],value=value)
        for table in catalog['fact_tables'] for row in table['rows']
        for column,value in zip(table['columns'],row['values'],strict=True)}
    assert restored==card['host_facts']
    assert canonical_sha(restored)==delivery['host_facts_catalog_sha256']==catalog['catalog_sha256']
    assert p.plan['prompt_bytes']<=generation.MAX_PROMPT_BYTES
    for role,key in [('edit_base','base_source_sha256'),('paired_comparator','comparator_source_sha256')]:
        assert builder.source_for(intent['source_bodies'][role]['body'])==p.plan['source_registry'][target[key]]['source']
    shown,_=json.JSONDecoder().raw_decode(p.plan['prompt'].split('EVALUATED_DETACHED_STUDY_JSON:\n',1)[1])
    assert shown['presented_count']==1 and shown['cards'][0]['strict_quality']['verdict']=='REFUTED'
    assert shown['cards'][0]['coverage']['unissued']==2
    assert result['native_observation']['score'] is result['native_observation']['forecast'] is None
    assert result['paired_observation']['status']=='UNMEASURED'
    assert result['new_ordinary_outcomes']==result['new_search_attempts']==0
    assert not result['execution_performed'] and not result['provider_dispatch_verified']
    assert generation.validate_result(result,bundle=p.r.bundle,review_sidecar=p.r.sidecar,
        reflection_config=p.r.config,expected_sha256=canonical_sha(result),config=p.config)==result


@pytest.mark.parametrize('decision',['pending','rejected','stale'])
def test_current_exact_host_acceptance_required(prepared,decision):
    policy=approval(prepared.r.sidecar['review_card'],status='rejected' if decision=='rejected' else 'accepted')
    if decision=='pending':policy['decisions']=[]
    if decision=='stale':policy['decisions'][0]['policy_sha256']='f'*64
    with pytest.raises(ValueError,match='acceptance'):
        make_job(prepared.r,config=dict(comparison_review_text_policy=policy))


def test_descriptive_null_review_cannot_start_G(artifact):
    with pytest.raises(ValueError,match='no prospective target'):
        make_job(reflected(artifact,null_target=True))


@pytest.mark.parametrize('key',['external_evaluated_study_sha256','external_evaluated_study_authority_sha256',
    'external_evaluated_review_sha256','source_reflection_config_sha256'])
def test_independent_config_pins_cannot_be_replaced(prepared,key):
    with pytest.raises(ValueError):make_job(prepared.r,config={key:'f'*64})


def test_sidecar_config_and_card_cannot_authorize_themselves(prepared):
    r=prepared.r;bad=copy.deepcopy(r.config);bad['model']='foreign-model'
    with pytest.raises(ValueError):make_job(r,reflection_config=bad)
    with pytest.raises(ValueError):make_job(r,card_sha='f'*64)
    altered=copy.deepcopy(r.sidecar)
    altered['review_card']['resolved_target']['change']='Rehashed unauthored target.'
    with pytest.raises(ValueError):make_job(r,sidecar=altered)


@pytest.mark.parametrize('role',['base_source_sha256','candidate_source_sha256','comparator_source_sha256'])
def test_all_saved_study_bodies_remain_duplicates(prepared,role):
    value=copy.deepcopy(prepared.value)
    source=prepared.r.bundle['capsule']['source_registry'][prepared.r.bundle['capsule']['author'][role]]
    value['body']=builder.body_of(source)
    result=admit(prepared,value=value)
    assert result['status']=='duplicate' and result['eligible'] is False and result['proof'] is None


def test_selected_R_provider_response_id_is_not_a_fresh_G(prepared):
    response=response_for(prepared.value,prepared.plan['request']['model'])
    result=admit(prepared,response=response,provider=provider_for(response,'synthetic-R-provider'))
    assert result['error']['code']=='historical_provider_replay' and not result['eligible']


@pytest.mark.parametrize('damage',['request','returned_text','returned_cost','target','scaffold','fenced'])
def test_changed_transport_or_code_not_repaired(prepared,damage):
    response=response_for(prepared.value,prepared.plan['request']['model'])
    provider=provider_for(response);request=copy.deepcopy(prepared.plan['request'])
    value=copy.deepcopy(prepared.value)
    if damage=='request':request['extra_body']['provider']['allow_fallbacks']=True
    elif damage=='returned_text':provider['choices'][0]['message']['content']+=' '
    elif damage=='returned_cost':provider['usage']['cost']+=1
    else:
        if damage=='target':value['target_sha256']='f'*64
        elif damage=='scaffold':value['body']='import os\nreturn ordinary_params\n'
        else:value['body']='```python\nreturn ordinary_params\n```'
        response=response_for(value,prepared.plan['request']['model']);provider=provider_for(response)
    result=admit(prepared,response=response,provider=provider,request=request)
    assert not result['eligible'] and result['proof'] is result['code'] is None


def test_result_rehash_cannot_change_approved_source_or_outcome(prepared):
    result=admit(prepared);result['native_observation']['score']=1
    with pytest.raises(ValueError):
        generation.validate_result(result,bundle=prepared.r.bundle,review_sidecar=prepared.r.sidecar,
            reflection_config=prepared.r.config,expected_sha256=canonical_sha(result),config=prepared.config)


def test_no_final_LF_body_is_retained_exactly(prepared):
    value=copy.deepcopy(prepared.value);value['body']=value['body'].rstrip('\n')
    result=admit(prepared,value=value)
    assert result['eligible'] and result['body']==value['body'] and result['proof']['target_body']==value['body']
