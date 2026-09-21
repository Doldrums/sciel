"""Synthetic evaluated R/G authors over authored offline HOST facts.

The receipts below are literal synthetic geometry, not fresh candidate runs.
No old rejected model reply is repaired, reused as approved, or executed.
"""
import copy
import json
from types import SimpleNamespace as NS

import pytest

from sciel.knowledge_base import evaluated_detached_study as core
from sciel.knowledge_base.catalogs import canonical_sha
from sciel.knowledge_base.paired_comparisons import build_comparison
from sciel.hypothesis.comparison_targets import _quality
from sciel.hypothesis.evaluated_generation_acceptance import make_protocol
from sciel.eval import packing_geometry as geometry
from test_evaluated_detached_study import (digest, text, facts, validate, reseal_terminal,
    no_execution, host)
from test_evaluated_target_generation import (artifact, prepared, admit, make_job, r_config,
    EvaluatedStudyReflectionJob)
from test_component_target_generation import response_for, provider_for, TRANSPORT
from test_component_generation_hypothesis import payload
from evaluated_study_fixture import adapter_identity as authored_adapter_identity


def acceptance_for(result, protocol):
    return dict(schema='sciel-evaluated-G-numeric-decision/1', decision='accepted',
        generation_result_sha256=canonical_sha(result), generated_source_sha256=result['source_sha256'],
        source_proof_sha256=canonical_sha(result['proof']), target_sha256=result['plan']['target_sha256'],
        policy_sha256=result['plan']['policy_sha256'], audit_sha256='a'*64, criteria_sha256='b'*64,
        numeric_protocol_sha256=canonical_sha(protocol), issuer='SYNTHETIC numerical decision fixture')


def synthetic_evaluated_bundle(prepared, source_bundle, *, seeds=None):
    # Preserve a valid authored no-final-LF body, rather than normalizing its proof.
    value=copy.deepcopy(prepared.value);value['body']=value['body'].rstrip('\n')
    response=response_for(value,prepared.plan['request']['model'])
    result=admit(prepared,response=response,provider=provider_for(response,
        'synthetic-provider-'+prepared.plan['run_id']))
    assert result['eligible'] and not result['body'].endswith('\n')
    target=result['plan']['target'];config=prepared.config
    deps=dict(kind='evaluated_target_G', generation_result=result, source_study_bundle=prepared.r.bundle,
        reflection_config=prepared.r.config, review_sidecar=prepared.r.sidecar, generation_config=config,
        host_decision=copy.deepcopy(config['comparison_review_text_policy']['decisions'][0]))
    author=dict(kind='evaluated_target_G',artifact_sha256=canonical_sha(result),
        base_source_sha256=target['base_source_sha256'],candidate_source_sha256=result['source_sha256'],
        comparator_source_sha256=target['comparator_source_sha256'])
    role_map={'comparator':'comparator','challenger':'challenger'}
    seeds=[17,19,23] if seeds is None else seeds
    addendum=make_protocol('synthetic-measurement-'+prepared.plan['run_id'],seeds)
    schedule=addendum['schedule']
    prior=source_bundle['capsule'];limits=prior['comparison']['limits']
    # Author this new synthetic measurement under its accepted prospective
    # protocol. The ancestor's offline runtime is not this study's runtime.
    adapter_identity=authored_adapter_identity(runtime_config=addendum['runtime'])
    sources={digest_:result['plan']['source_registry'][digest_]['source'] for digest_ in
             {target['receiving_root']['source_sha256'],target['base_source_sha256'],target['comparator_source_sha256']}}
    sources[result['source_sha256']]=result['code']
    source_map={'comparator':author['comparator_source_sha256'],'challenger':author['candidate_source_sha256']}
    assert addendum['limits']==limits and addendum['runtime']==adapter_identity['runtime_config']
    deps['numeric_protocol']=copy.deepcopy(addendum)
    deps['generation_acceptance']=acceptance_for(result, addendum)
    manifest=dict(schema=core.EVALUATED_G_PRODUCER,source_sha256=source_map,adapter_identity=adapter_identity,
        numeric_addendum=addendum,runtime={'fixture':'synthetic host interpreter identity'},
        source_proof_sha256=canonical_sha(result['proof']),generation_result_sha256=canonical_sha(result),
        generation_config_sha256=canonical_sha(config),host_decision_sha256=result['plan']['host_decision_sha256'],
        files={'candidates/'+role+'.py':sha for role,sha in source_map.items()})
    manifest.update({field:canonical_sha(deps[key]) for field,key in
        [('source_study_bundle_sha256','source_study_bundle'), ('reflection_config_sha256','reflection_config'),
         ('review_sidecar_sha256','review_sidecar'), ('generation_acceptance_sha256','generation_acceptance'),
         ('numeric_protocol_sha256','numeric_protocol')]})
    manifest['payload_sha256']=canonical_sha(manifest);manifest_text=text(manifest);manifest_sha=digest(manifest_text)
    started_text=text(dict(manifest_sha256=manifest_sha,no_resume=True,timestamp=1))
    receipts={};slots=[];artifacts={'manifest.json':manifest_sha,'started.json':digest(started_text),
        **{'frozen/'+key:value for key,value in manifest['files'].items()}}
    for slot in schedule:
        retained={**slot,**dict.fromkeys(('intent_text','raw_receipt_text','validated_text','unknown_text'))}
        if slot['index']<2:
            intent=dict(**slot,schema=core.EVALUATED_G_PRODUCER,manifest_sha256=manifest_sha,source_sha256=source_map[slot['role']],
                adapter_identity_sha256=adapter_identity['identity_sha256'],limits=limits,runtime=addendum['runtime'],restart_trace=True)
            # Literal synthetic safe grid, deliberately worse candidate radii.
            capture=geometry.validate_packing(dict(centers=[[.1+.15*(i%6),.1+.15*(i//6)] for i in range(26)],
                radii=[.02 if slot['role']=='comparator' else .01]*26))
            cert=geometry.certify_packing(capture)
            receipt=dict(schema='sciel-repeat-draw/1',status='valid',source_sha256=source_map[slot['role']],seed=slot['seed'],
                adapter_identity=adapter_identity,limits=limits,started_at='2026-01-01T00:00:00+00:00',
                completed_at='2026-01-01T00:00:01+00:00',elapsed_seconds=1.,candidate_elapsed_seconds=.5,
                certification_elapsed_seconds=.5,identity_elapsed_seconds=0.,
                seed_applied=dict(python_random=slot['seed'],numpy_legacy=slot['seed'],python_hash_seed=str(slot['seed']),
                                  before_candidate_import=True,determinism_proven=False),
                raw_sum=capture['sum_radii'],certified_sum=cert['sum_exact'],capture=capture,certificate=cert)
            raw=text(receipt);retained.update(intent_text=text(intent),raw_receipt_text=raw,
                validated_text=text(dict(status='valid',raw_receipt_sha256=digest(raw))))
            receipts[(slot['seed'],slot['role'])]=receipt
        slots.append(retained)
    group=build_comparison(namespace='synthetic-target:'+manifest_sha,batch_id='synthetic-target-control',
        branch='synthetic-target-control',origin_experiment_id=None,comparator_source_sha256=source_map['comparator'],
        challenger_source_sha256=source_map['challenger'],adapter_identity=adapter_identity,limits=limits,
        pairs=[dict(seed=seed,comparator=receipts.get((seed,'comparator')),challenger=receipts.get((seed,'challenger'))) for seed in seeds],
        provenance=dict(schema=core.EVALUATED_G_PRODUCER,numeric_manifest_sha256=manifest_sha,schedule=schedule,detached=True,
            authorship='evaluated_target_G',numeric_addendum_sha256=canonical_sha(addendum),source_proof_sha256=canonical_sha(result['proof'])),
        task_metric=prior['metric'],source_pair=dict(comparator=sources[source_map['comparator']],challenger=sources[source_map['challenger']]))
    summary=dict(schema=core.EVALUATED_G_PRODUCER,manifest_sha256=manifest_sha,status='early_stopped_known_target_loss',errors=[],
        slots=[],artifacts=artifacts,planned_candidate_allocations=6,issued_draw_intents=0,validated_terminal_receipts=0,
        remaining_unissued_slots=schedule,quality=_quality(group),group_sha256=canonical_sha(group),quality_gate_passed=False,
        automatic_promotion=False,resume_allowed=False,evidence_integrity_verified=True,provider_calls=0,new_api_cost_usd='0')
    capsule=dict(schema=core.SCHEMA,profile=core.PROFILE,metric=prior['metric'],receiving_root=target['receiving_root'],
        validation_policy=core.identity(),author=author,source_registry=sources,comparison=group,
        study=dict(manifest_text=manifest_text,summary_text=text(summary),started_text=started_text,role_map=role_map,slots=slots))
    authority=dict(schema=core.AUTHORITY_SCHEMA,kind='evaluated_target_G',dependencies_sha256=canonical_sha(deps),
        producer_manifest_sha256=manifest_sha,producer_summary_sha256=digest(text(summary)),author_artifact_sha256=canonical_sha(result),
        generation_config_sha256=canonical_sha(config),host_decision_sha256=result['plan']['host_decision_sha256'],issuer='synthetic test only')
    return reseal_terminal(dict(capsule=capsule,author_dependencies=deps,author_authority=authority))


@pytest.fixture(scope='module')
def generated(prepared):
    return synthetic_evaluated_bundle(prepared, prepared.r.bundle)


def test_evaluated_R_G_authority_source_proof_and_measurements_are_distinct(generated):
    before=canonical_sha(generated); f=facts(generated)
    result=generated['author_dependencies']['generation_result']
    assert f['author']['kind']=='evaluated_target_G'
    assert f['author']['generation_id']==result['generation_id']
    assert f['author']['unverified_hypothesis']==result['hypothesis']
    assert f['author']['native_forecast'] is None
    assert f['native']==dict(observation_status='UNMEASURED',forecast_status='NOT_DECLARED',score=None,ordinary_origin=None)
    assert f['comparison']['origin_experiment_id'] is None
    assert f['quality']['verdict']=='REFUTED'
    assert f['paired_forecast']['expected'] is f['paired_forecast']['observed'] is False
    assert f['paired_forecast']['verdict']=='CONFIRMED'
    assert f['paired_forecast']['target_evaluation']['original_target_sha256']==result['plan']['target_sha256']
    assert not f['coverage']['producer_quality_gate_passed']
    assert f['coverage']['issued']==2 and f['coverage']['unissued']==4
    assert f['restart'][0]['forecast'] is None
    assert f['restart'][1]['forecast']==result['hypothesis']['restart_predictions']
    assert result['proof']['target_body']==result['body'] and not result['body'].endswith('\n')
    assert canonical_sha(generated)==before
    assert validate(generated)==generated['capsule']


@pytest.mark.parametrize('damage',['source_bundle','reflection_config','review_raw','generation_config',
    'R_decision','numeric_decision','numeric_protocol','body_proof','native_score','G_raw','metric','wrong_author'])
def test_external_authority_never_substitutes_for_raw_reconstruction(generated, damage):
    b=copy.deepcopy(generated);d=b['author_dependencies'];a=b['author_authority'];c=b['capsule']
    if damage=='source_bundle':d['source_study_bundle']['capsule']['author']['artifact_sha256']='f'*64
    elif damage=='reflection_config':d['reflection_config']['model']='forged-reflector'
    elif damage=='review_raw':d['review_sidecar']['response']['text']+=' '
    elif damage=='generation_config':d['generation_config']['model']='forged-generator'
    elif damage=='R_decision':d['host_decision']['decision']='rejected'
    elif damage=='numeric_decision':d['generation_acceptance']['target_sha256']='f'*64
    elif damage=='numeric_protocol':d['numeric_protocol']['max_candidate_calls']=8
    elif damage=='body_proof':d['generation_result']['proof']['parent_source_sha256']='f'*64
    elif damage=='native_score':d['generation_result']['native_observation']['score']=.5
    elif damage=='G_raw':d['generation_result']['response']['text']+=' '
    elif damage=='metric':c['metric']['primary_metric']='invented'
    else:c['author']['kind']='component_target_G'
    # Give the forged envelope a new explicit outer pin. This must not bypass
    # reconstruction of independent R/config/decision/raw/proof dependencies.
    a['dependencies_sha256']=canonical_sha(d)
    if damage in ('body_proof','native_score','G_raw'):
        c['author']['artifact_sha256']=a['author_artifact_sha256']=canonical_sha(d['generation_result'])
    with pytest.raises(ValueError):validate(b)


def test_separately_accepted_valid_protocol_must_be_the_executed_addendum(generated):
    b=copy.deepcopy(generated);d=b['author_dependencies']
    # Both protocols are individually valid, but approving a different study
    # identity cannot authorize the already recorded execution conditions.
    d['numeric_protocol']=make_protocol('another-synthetic-study',[17,19,23])
    d['generation_acceptance']=acceptance_for(d['generation_result'],d['numeric_protocol'])
    b['author_authority']['dependencies_sha256']=canonical_sha(d)
    with pytest.raises(ValueError,match='executed addendum'):validate(b)


def test_validated_results_are_copied_and_cache_does_not_survive_public_call(generated,monkeypatch):
    original=core._validate_uncached;calls=[]
    def counted(*args,**kwargs):
        calls.append(canonical_sha(args[0]));return original(*args,**kwargs)
    monkeypatch.setattr(core,'_validate_uncached',counted)
    first=facts(generated); first['author']['unverified_hypothesis']['text']='mutated caller projection'
    initial=len(calls); second=facts(generated)
    assert len(calls)==2*initial
    assert second['author']['unverified_hypothesis']['text']!='mutated caller projection'
    assert core._VALIDATION_SESSION.get() is None
    b=copy.deepcopy(generated);b['author_authority']['dependencies_sha256']='f'*64
    with pytest.raises(ValueError):validate(b)
    assert core._VALIDATION_SESSION.get() is None


def test_evaluated_ancestry_limit_precedes_any_generation_reconstruction(host,monkeypatch):
    b=copy.deepcopy(host);deps=dict(kind='HOST')
    for _ in range(core.MAX_EVALUATED_ANCESTRY+1):
        deps=dict(kind='evaluated_target_G',source_study_bundle=dict(capsule={},author_dependencies=deps,author_authority={}))
    b['author_dependencies']=deps
    def forbidden(*args,**kwargs):pytest.fail('overbound ancestry entered author reconstruction')
    monkeypatch.setattr(core,'_validate_uncached',forbidden)
    with pytest.raises(ValueError,match='ancestry limit'):validate(b)
    assert core._VALIDATION_SESSION.get() is None


def test_cyclic_ancestry_is_rejected_as_nonliteral_without_recursing(host):
    b=copy.deepcopy(host);b['author_dependencies']=dict(kind='evaluated_target_G',source_study_bundle=b)
    with pytest.raises(ValueError,match='acyclic'):validate(b)
    assert core._VALIDATION_SESSION.get() is None


def test_two_generated_ancestors_reconstruct_once_each_within_one_call(generated,monkeypatch):
    # Explicitly synthetic second R/G, with fresh request/provider identities.
    # It is not an acceptance of any archived paid R reply.
    config=r_config(generated)
    job=EvaluatedStudyReflectionJob(generated,
        expected_sha256=config['external_evaluated_study_sha256'],
        expected_authority_sha256=config['external_evaluated_study_authority_sha256'],config=config)
    plan=job.prepare(run_id='synthetic-R-depth2',segment_id='synthetic-R-depth2',transport_settings=TRANSPORT)
    proposal=dict(claims=[dict(fact_id='quality:verdict',value='REFUTED')],
        conjecture='SYNTHETIC second-generation fixture, no causal assertion.',
        limitations='SYNTHETIC receipts do not establish packing or memory benefit.',
        next_hypothesis_reason='SYNTHETIC bounded edit only; no candidate execution.',
        evidence_refs={field:['quality:verdict'] for field in
            ('conjecture','limitations','next_hypothesis_reason')},
        next_hypothesis=dict(base_source_sha256=generated['capsule']['author']['candidate_source_sha256'],
            comparator_source_sha256=generated['capsule']['author']['comparator_source_sha256'],
            change='SYNTHETIC: replace cadence16 with cadence32.',diagnostics=[],motivating_evidence=['quality:verdict']))
    response=response_for(proposal,config['model'])
    admitted=job.admit(plan,response,manifest_sha256='c'*64,journal_prefix=plan['journal_prefix'],
        provider_request=plan['request'],provider_response=provider_for(response,'synthetic-R-provider-depth2'))
    r=NS(bundle=generated,config=config,sidecar=admitted['extended_capsule'])
    g,cfg=make_job(r)
    gplan=g.prepare(run_id='synthetic-G-depth2',segment_id='synthetic-G-depth2',transport_settings=TRANSPORT)
    base=gplan['source_registry'][gplan['target']['base_source_sha256']]['body']
    assert 'restart_index % 16' in base
    value=payload(base.replace('restart_index % 16','restart_index % 32',1))
    value['target_sha256']=gplan['target_sha256']
    value['hypothesis']['paired_forecast']['target_sha256']=gplan['target_sha256']
    value['hypothesis']['text']='SYNTHETIC second-generation fixture, not a measured claim.'
    second=synthetic_evaluated_bundle(NS(r=r,job=g,config=cfg,plan=gplan,value=value),generated,seeds=[29,31,37])
    original=core._validate_uncached;counts={}
    def counted(*args,**kwargs):
        key=canonical_sha(args[0]);counts[key]=counts.get(key,0)+1
        return original(*args,**kwargs)
    monkeypatch.setattr(core,'_validate_uncached',counted)
    result=facts(second)
    assert result['author']['kind']=='evaluated_target_G'
    assert core._ancestry_bound(second['author_dependencies'])==2
    assert counts=={canonical_sha(second['capsule']):1,canonical_sha(generated['capsule']):1,
                   canonical_sha(generated['author_dependencies']['source_study_bundle']['capsule']):1}
    assert core._VALIDATION_SESSION.get() is None


def test_transaction_cache_copies_results_and_never_caches_failure(host,monkeypatch):
    c=host['capsule'];a=host['author_authority'];d=host['author_dependencies']
    args=(c,canonical_sha(c),c['metric'],None,d,a)
    original=core._validate_uncached;calls=[]
    def counted(*values,**kwargs):
        calls.append(canonical_sha(values[0]));return original(*values,**kwargs)
    monkeypatch.setattr(core,'_validate_uncached',counted)
    with core._validation_scope():
        first=core._validate(*args)
        first[0]['unverified_hypothesis']='caller mutation'
        second=core._validate(*args)
        assert second[0]['unverified_hypothesis']!='caller mutation' and len(calls)==1
        bad=list(args);bad[-1]=dict(a,dependencies_sha256='f'*64)
        for _ in range(2):
            with pytest.raises(ValueError):core._validate(*bad)
        assert len(calls)==3
    assert core._VALIDATION_SESSION.get() is None


@pytest.mark.parametrize('binding',['expected_sha256','metric','root','policy'])
def test_transaction_cache_does_not_reuse_other_receiving_bindings(host,monkeypatch,binding):
    c=host['capsule'];a=host['author_authority'];d=host['author_dependencies']
    args=[c,canonical_sha(c),c['metric'],None,d,a]
    with core._validation_scope():
        core._validate(*args)
        if binding=='expected_sha256':args[1]='f'*64
        elif binding=='metric':args[2]=dict(c['metric'],primary_metric='invented')
        elif binding=='root':args[3]=dict(c['receiving_root'],id='another-root')
        else:
            changed=dict(core.identity(),maximum_evaluated_ancestry=99)
            monkeypatch.setattr(core,'identity',lambda:changed)
        with pytest.raises(ValueError):core._validate(*args)
    assert core._VALIDATION_SESSION.get() is None
