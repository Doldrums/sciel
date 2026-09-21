"""Sealed authorship and recorded evidence only; never execute a candidate.

HOST and Target-G rows are explicitly authored fixtures: synthetic R/G replies
and packing receipts. They are not historical measurements or new results.
"""
import copy
import hashlib
import json
from types import SimpleNamespace as NS

import pytest

from sciel.knowledge_base import evaluated_detached_study as core
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
from sciel.knowledge_base.paired_comparisons import build_comparison, UNKNOWN_RECEIPT
from sciel.hypothesis.comparison_targets import _quality
from sciel.eval import packing_geometry as geometry
import evaluated_study_fixture as adapter
from test_component_target_generation import prepared, saved, component, pilot_artifacts, admit


def digest(text):return hashlib.sha256(text.encode()).hexdigest()
def text(value):return canonical_json(value)+'\n'


@pytest.fixture(scope='module',autouse=True)
def no_execution():
    def forbidden(*_args,**_kwargs):pytest.fail('Evidence admission attempted external/candidate execution.')
    with pytest.MonkeyPatch.context() as patch:
        for name in ('subprocess.Popen','tarfile.TarFile.extract','tarfile.TarFile.extractall',
            'sciel.eval.repeat_draw.draw','sciel.eval.repeat_draw.adapter_identity',
            'sciel.engine.llm.client.EnsembleLLM.generate','sciel.campaign.budget.BudgetedLLM.generate'):
            patch.setattr(name,forbidden)
        yield


@pytest.fixture(scope='module')
def host():return adapter.build()


def validate(bundle):
    capsule=bundle['capsule']
    return core.validate_capsule(capsule,expected_sha256=canonical_sha(capsule),metric=capsule['metric'],
        root=capsule['receiving_root'],author_dependencies=bundle['author_dependencies'],
        author_authority=bundle['author_authority'])


def facts(bundle):
    return core.derive_facts(bundle['capsule'],author_dependencies=bundle['author_dependencies'],
                             author_authority=bundle['author_authority'])


def rebuild_group(group):
    fields=('namespace','batch_id','branch','origin_experiment_id','comparator_source_sha256',
        'challenger_source_sha256','adapter_identity','limits','pairs','provenance','task_metric','source_pair')
    return build_comparison(**{key:group[key] for key in fields})


def reseal_terminal(bundle):
    """An explicitly reauthorized test terminal, not a repair of saved artifacts."""
    capsule=bundle['capsule'];study=capsule['study'];group=rebuild_group(capsule['comparison'])
    capsule['comparison']=group;summary=json.loads(study['summary_text'])
    schedule=json.loads(study['manifest_text'])['numeric_addendum']['schedule']
    summary['quality']=_quality(group);summary['group_sha256']=canonical_sha(group)
    summary['comparison_sha256']={group['batch_id']:canonical_sha(group)}
    summary['quality_gate_passed']=summary['evidence_integrity_verified'] and summary['quality']['verdict']=='CONFIRMED'
    summary['slots']=[];summary['artifacts']={k:v for k,v in summary['artifacts'].items() if not k.startswith('draws/')}
    for slot in study['slots']:
        if slot['intent_text'] is not None:
            role=next(k for k,v in study['role_map'].items() if v==slot['role'])
            receipt=group['pairs'][slot['index']//2][role]
            row={k:slot[k] for k in ('index','seed','role')};row.update(status=receipt['status'],intent_sha256=digest(slot['intent_text']))
            if slot['raw_receipt_text'] is not None:row['raw_receipt_sha256']=digest(slot['raw_receipt_text'])
            summary['slots'].append(row)
        prefix=f"draws/{slot['index']:02d}_{slot['seed']}_{slot['role']}/"
        for key,name in [('intent_text','intent.json'),('raw_receipt_text','raw_receipt.json'),
                         ('validated_text','validated.json'),('unknown_text','unknown.json')]:
            if slot[key] is not None:summary['artifacts'][prefix+name]=digest(slot[key])
    count=len(summary['slots']);summary['issued_draw_intents']=count
    summary['validated_terminal_receipts']=sum(row['status']!='unknown' for row in summary['slots'])
    summary['remaining_unissued_slots']=schedule[count:]
    study['summary_text']=text(summary)
    bundle['author_authority']['producer_summary_sha256']=digest(study['summary_text'])
    return bundle


def synthetic_g_bundle(prepared,host):
    # Preserve a valid authored no-final-LF body, rather than normalizing its proof.
    value=copy.deepcopy(prepared.value);value['body']=value['body'].rstrip('\n')
    result=admit(prepared,value=value)
    assert result['eligible'] and not result['body'].endswith('\n')
    target=result['plan']['target'];config=prepared.config
    deps=dict(kind='component_target_G',generation_result=result,source_component=prepared.saved.component,
        review_sidecar=prepared.saved.sidecar,generation_config=config)
    author=dict(kind='component_target_G',artifact_sha256=canonical_sha(result),
        base_source_sha256=target['base_source_sha256'],candidate_source_sha256=result['source_sha256'],
        comparator_source_sha256=target['comparator_source_sha256'])
    role_map={'comparator':'control','challenger':'candidate'};seeds=[17,19,23]
    schedule=[dict(index=2*i+j,seed=seed,role=role) for i,seed in enumerate(seeds)
              for j,role in enumerate(('control','candidate') if i%2==0 else ('candidate','control'))]
    prior=host['capsule'];adapter_identity=prior['comparison']['adapter_identity'];limits=prior['comparison']['limits']
    sources={digest_:result['plan']['source_registry'][digest_]['source'] for digest_ in
             {target['receiving_root']['source_sha256'],target['base_source_sha256'],target['comparator_source_sha256']}}
    sources[result['source_sha256']]=result['code']
    source_map={'control':author['comparator_source_sha256'],'candidate':author['candidate_source_sha256']}
    addendum=dict(schema=core.G_PRODUCER,schedule=schedule,limits=limits,runtime=adapter_identity['runtime_config'],
        execution_profile=core.PROFILE,restart_trace=True,max_candidate_calls=6,max_certifier_calls=6,
        provider_calls=0,no_resume=True,no_retry=True,quality_gate=core.QUALITY_GATE,automatic_promotion=False)
    manifest=dict(schema=core.G_PRODUCER,source_sha256=source_map,adapter_identity=adapter_identity,
        numeric_addendum=addendum,runtime={'fixture':'synthetic host interpreter identity'},
        source_proof_sha256=canonical_sha(result['proof']),generation_result_sha256=canonical_sha(result),
        generation_config_sha256=canonical_sha(config),host_decision_sha256=result['plan']['host_decision_sha256'],
        files={'candidates/'+role+'.py':sha for role,sha in source_map.items()})
    manifest['payload_sha256']=canonical_sha(manifest);manifest_text=text(manifest);manifest_sha=digest(manifest_text)
    started_text=text(dict(manifest_sha256=manifest_sha,no_resume=True,timestamp=1))
    receipts={};slots=[];artifacts={'manifest.json':manifest_sha,'started.json':digest(started_text),
        **{'frozen/'+key:value for key,value in manifest['files'].items()}}
    for slot in schedule:
        retained={**slot,**dict.fromkeys(('intent_text','raw_receipt_text','validated_text','unknown_text'))}
        if slot['index']<2:
            intent=dict(**slot,schema=core.G_PRODUCER,manifest_sha256=manifest_sha,source_sha256=source_map[slot['role']],
                adapter_identity_sha256=adapter_identity['identity_sha256'],limits=limits,runtime=addendum['runtime'],restart_trace=True)
            # Literal synthetic safe grid, deliberately worse candidate radii.
            capture=geometry.validate_packing(dict(centers=[[.1+.15*(i%6),.1+.15*(i//6)] for i in range(26)],
                radii=[.02 if slot['role']=='control' else .01]*26))
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
        branch='synthetic-target-control',origin_experiment_id=None,comparator_source_sha256=source_map['control'],
        challenger_source_sha256=source_map['candidate'],adapter_identity=adapter_identity,limits=limits,
        pairs=[dict(seed=seed,comparator=receipts.get((seed,'control')),challenger=receipts.get((seed,'candidate'))) for seed in seeds],
        provenance=dict(schema=core.G_PRODUCER,numeric_manifest_sha256=manifest_sha,schedule=schedule,detached=True,
            authorship='component_target_G',numeric_addendum_sha256=canonical_sha(addendum),source_proof_sha256=canonical_sha(result['proof'])),
        task_metric=prior['metric'],source_pair=dict(comparator=sources[source_map['control']],challenger=sources[source_map['candidate']]))
    summary=dict(schema=core.G_PRODUCER,manifest_sha256=manifest_sha,status='early_stopped_known_target_loss',errors=[],
        slots=[],artifacts=artifacts,planned_candidate_allocations=6,issued_draw_intents=0,validated_terminal_receipts=0,
        remaining_unissued_slots=schedule,quality=_quality(group),group_sha256=canonical_sha(group),quality_gate_passed=False,
        automatic_promotion=False,resume_allowed=False,evidence_integrity_verified=True,provider_calls=0,new_api_cost_usd='0')
    capsule=dict(schema=core.SCHEMA,profile=core.PROFILE,metric=prior['metric'],receiving_root=target['receiving_root'],
        validation_policy=core.identity(),author=author,source_registry=sources,comparison=group,
        study=dict(manifest_text=manifest_text,summary_text=text(summary),started_text=started_text,role_map=role_map,slots=slots))
    authority=dict(schema=core.AUTHORITY_SCHEMA,kind='component_target_G',dependencies_sha256=canonical_sha(deps),
        producer_manifest_sha256=manifest_sha,producer_summary_sha256=digest(text(summary)),author_artifact_sha256=canonical_sha(result),
        generation_config_sha256=canonical_sha(config),host_decision_sha256=result['plan']['host_decision_sha256'],issuer='synthetic test only')
    return reseal_terminal(dict(capsule=capsule,author_dependencies=deps,author_authority=authority))


@pytest.fixture(scope='module')
def generated(prepared,host):return synthetic_g_bundle(prepared,host)


def test_authored_HOST_full_authorship_four_measurements_and_two_unissued(host):
    before=canonical_sha(host);result=validate(host);f=facts(host)
    assert result==host['capsule'] and result is not host['capsule']
    assert f['author']['kind']=='HOST' and f['author']['generation_id'] is None
    assert f['native']==dict(observation_status='UNMEASURED',forecast_status='NOT_DECLARED',score=None,ordinary_origin=None)
    assert f['quality']['verdict']=='REFUTED' and f['quality']['reason']=='known_paired_loss'
    assert f['coverage']['issued']==4 and f['coverage']['unissued']==2 and f['coverage']['unknown']==0
    assert f['coverage']['producer_integrity_verified'] is True and not f['coverage']['execution_verified']
    assert f['paired_forecast'] is None and all(row['forecast'] is None for row in f['restart'])
    assert all(row['prediction'] is None for role in f['restart'] for row in role['rows'])
    assert f['restart'][1]['rows'][0]['observation']['status']=='available'
    assert f['restart'][1]['rows'][2]['observation'] is None
    assert canonical_sha(host)==before


@pytest.mark.parametrize('key,value',[('schema','wrong'),('kind','model'),('dependencies_sha256',None),
    ('producer_manifest_sha256',True),('producer_summary_sha256','F'*64),('author_artifact_sha256','x'),
    ('generation_config_sha256','a'*64),('host_decision_sha256','a'*64),('issuer',' ')])
def test_authority_is_strict_detached_shape_not_a_self_grant(host,key,value):
    auth=copy.deepcopy(host['author_authority']);auth[key]=value
    with pytest.raises(ValueError):core.validate_authority(auth)


@pytest.mark.parametrize('damage',['source','root','group_origin','metric','author_kind','base','extra_source',
    'producer_policy','body_proof','design','dependency_grant','declared_measurement','started','schedule','runtime','empty_slot'])
def test_rehashed_capsule_cannot_forge_sources_authorship_or_lifecycle(host,damage):
    b=copy.deepcopy(host);c=b['capsule'];d=b['author_dependencies'];s=c['study']
    if damage=='source':c['source_registry'][c['author']['candidate_source_sha256']]+='\n# forged'
    elif damage=='root':c['receiving_root']['source_sha256']=c['author']['base_source_sha256']
    elif damage=='group_origin':c['comparison']['origin_experiment_id']='fake-r000001a1'
    elif damage=='metric':c['metric']['primary_metric']='fake'
    elif damage=='author_kind':c['author']['kind']='component_target_G'
    elif damage=='base':c['author']['base_source_sha256']=c['author']['comparator_source_sha256']
    elif damage=='extra_source':c['source_registry']['a'*64]='return None'
    elif damage=='producer_policy':c['validation_policy']['production_policy_sha256']='a'*64
    elif damage=='body_proof':p=json.loads(d['proof_text']);p['parent_source_sha256']='a'*64;d['proof_text']=text(p)
    elif damage=='design':p=json.loads(d['design_text']);p['hypothesis']['mechanism']='forged';d['design_text']=text(p)
    elif damage=='dependency_grant':b['author_authority']['dependencies_sha256']='a'*64
    elif damage=='declared_measurement':c['author']['native_score']=2.6
    elif damage=='started':p=json.loads(s['started_text']);p['timestamp']+=1;s['started_text']=text(p)
    elif damage=='schedule':s['slots'][0]['seed']+=1
    elif damage=='runtime':p=json.loads(s['slots'][0]['intent_text']);p['runtime']={'backend':'native'};s['slots'][0]['intent_text']=text(p)
    else:s['slots'][0]['intent_text']=None
    with pytest.raises(ValueError):validate(b)


def test_invalid_raw_metric_cannot_be_hidden_by_new_terminal_hashes(host):
    b=copy.deepcopy(host);c=b['capsule'];slot=c['study']['slots'][0]
    raw=json.loads(slot['raw_receipt_text']);raw['raw_sum']+=1
    slot['raw_receipt_text']=text(raw);slot['validated_text']=text(dict(status='valid',raw_receipt_sha256=digest(slot['raw_receipt_text'])))
    c['comparison']['pairs'][0]['comparator']=raw
    with pytest.raises(ValueError):rebuild_group(c['comparison'])


def test_retrospective_receipt_failure_keeps_other_measurements_but_never_invents_dispatch(host):
    b=copy.deepcopy(host);c=b['capsule'];summary=json.loads(c['study']['summary_text'])
    summary.update(status='stopped_infrastructure',evidence_integrity_verified=False,
        errors=[dict(stage='terminal_receipt_integrity',slot=0,type='ValueError',message='SYNTHETIC recorded raw drift')])
    c['study']['summary_text']=text(summary);c['comparison']['pairs'][0]['comparator']=copy.deepcopy(UNKNOWN_RECEIPT)
    # The raw/validated files intentionally remain, as in the actual runner's downgrade path.
    reseal_terminal(b);f=facts(b)
    assert f['coverage']['issued']==4 and f['coverage']['unknown']==1 and f['coverage']['terminal']==3
    assert f['coverage']['producer_integrity_verified'] is False
    assert f['quality']['verdict']=='REFUTED'  # The other complete pair still has its known loss.
    assert f['restart'][0]['rows'][0]['observation'] is None


def test_immediate_unknown_with_original_marker_stops_and_never_becomes_unissued(host):
    b=copy.deepcopy(host);c=b['capsule'];summary=json.loads(c['study']['summary_text'])
    error=dict(stage='issued_draw',slot={k:c['study']['slots'][0][k] for k in ('index','seed','role')},type='TimeoutError',message='SYNTHETIC unknown')
    summary.update(status='stopped_unknown',evidence_integrity_verified=False,errors=[error])
    c['study']['summary_text']=text(summary)
    for i,slot in enumerate(c['study']['slots']):
        role=next(k for k,v in c['study']['role_map'].items() if v==slot['role'])
        c['comparison']['pairs'][i//2][role]=copy.deepcopy(UNKNOWN_RECEIPT) if i==0 else None
        for key in ('raw_receipt_text','validated_text','unknown_text'):slot[key]=None
        if i:slot['intent_text']=None
        else:slot['unknown_text']=text(dict(marker=UNKNOWN_RECEIPT,error=error,cleanup_confirmed=False))
    reseal_terminal(b);f=facts(b)
    assert f['coverage']['issued']==f['coverage']['unknown']==1 and f['coverage']['unissued']==5
    assert f['quality']['verdict']=='INCONCLUSIVE'
    assert all(row['observation'] is None for role in f['restart'] for row in role['rows'])


def test_no_silent_losses_or_unknown_markers_can_be_omitted(host):
    b=copy.deepcopy(host)
    b['capsule']['study']['slots'][4]['intent_text']=b['capsule']['study']['slots'][0]['intent_text']
    with pytest.raises(ValueError):validate(b)
    b=copy.deepcopy(host);b['capsule']['study']['slots'][0]['unknown_text']=text(dict(marker=UNKNOWN_RECEIPT,cleanup_confirmed=False))
    with pytest.raises(ValueError):validate(b)


def test_new_G_keeps_exact_external_authority_no_LF_proof_and_forecast_quality_separation(generated):
    f=facts(generated);result=generated['author_dependencies']['generation_result']
    assert f['author']['kind']=='component_target_G' and f['author']['generation_id']==result['generation_id']
    assert f['author']['base_source_sha256']==result['plan']['target']['base_source_sha256']
    assert f['native']['forecast_status']=='NOT_DECLARED' and result['proof']['target_body']==result['body']
    assert not result['body'].endswith('\n') and f['quality']['verdict']=='REFUTED'
    assert f['paired_forecast']['expected'] is f['paired_forecast']['observed'] is False
    assert f['paired_forecast']['verdict']=='CONFIRMED' and not f['coverage']['producer_quality_gate_passed']
    assert f['paired_forecast']['target_sha256']==result['plan']['target_sha256']
    assert f['paired_forecast']['target_evaluation']['original_target_sha256']==result['plan']['target_sha256']
    assert f['restart'][0]['forecast'] is None
    assert f['restart'][1]['forecast']==result['hypothesis']['restart_predictions']
    assert all(row['prediction'] is None or set(row['prediction']['verdicts'].values())=={'INCONCLUSIVE'} for row in f['restart'][1]['rows'])


@pytest.mark.parametrize('damage',['grant','selfgrant','raw','proof','declared_comparator','authorship'])
def test_new_G_cannot_self_approve_rewrite_raw_or_change_evaluated_contrast(generated,damage):
    b=copy.deepcopy(generated);deps=b['author_dependencies'];result=deps['generation_result']
    if damage=='grant':b['author_authority']['host_decision_sha256']='a'*64
    elif damage=='selfgrant':deps['generation_config']['comparison_review_text_policy']['decisions']=[]
    elif damage=='raw':result['response']['text']+='\nforged'
    elif damage=='proof':result['proof']['parent_source_sha256']='a'*64
    elif damage=='declared_comparator':b['capsule']['author']['comparator_source_sha256']=b['capsule']['author']['base_source_sha256']
    else:b['capsule']['author']['kind']='HOST'
    # Even a newly pinned dependency digest is not an independent approval/raw writer.
    b['author_authority']['dependencies_sha256']=canonical_sha(deps)
    with pytest.raises(ValueError):validate(b)


def test_positive_saved_geometry_cannot_confirm_target_after_producer_integrity_failure(generated):
    b=copy.deepcopy(generated);c=b['capsule'];study=c['study'];group=c['comparison']
    manifest=json.loads(study['manifest_text']);summary=json.loads(study['summary_text'])
    templates={slot['role']:copy.deepcopy(slot) for slot in study['slots'][:2]}
    for slot in study['slots']:
        template=templates[slot['role']];intent=json.loads(template['intent_text'])
        intent.update({key:slot[key] for key in ('index','seed','role')})
        receipt=json.loads(template['raw_receipt_text']);receipt['seed']=slot['seed']
        receipt['seed_applied'].update(python_random=slot['seed'],numpy_legacy=slot['seed'],python_hash_seed=str(slot['seed']))
        capture=geometry.validate_packing(dict(centers=receipt['capture']['centers'],
            radii=[.01 if slot['role']=='control' else .02]*26))
        cert=geometry.certify_packing(capture)
        receipt.update(capture=capture,certificate=cert,raw_sum=capture['sum_radii'],certified_sum=cert['sum_exact'])
        slot.update(intent_text=text(intent),raw_receipt_text=text(receipt),unknown_text=None)
        slot['validated_text']=text(dict(status='valid',raw_receipt_sha256=digest(slot['raw_receipt_text'])))
        role=next(key for key,value in study['role_map'].items() if value==slot['role'])
        group['pairs'][slot['index']//2][role]=receipt
    summary.update(status='stopped_infrastructure',evidence_integrity_verified=False,
        errors=[dict(stage='terminal_output',type='OSError',message='SYNTHETIC terminal bookkeeping failure')])
    study['summary_text']=text(summary);reseal_terminal(b);f=facts(b)
    assert f['coverage']['issued']==6 and f['coverage']['producer_integrity_verified'] is False
    assert f['quality']['verdict']=='CONFIRMED' and not f['coverage']['producer_quality_gate_passed']
    assert f['paired_forecast']['observed'] is None and f['paired_forecast']['verdict']=='INCONCLUSIVE'
    evaluation=f['paired_forecast']['target_evaluation']
    assert evaluation['saved_data']['combined_verdict']=='CONFIRMED'
    assert evaluation['verdict']=='INCONCLUSIVE' and evaluation['producer_integrity_verified'] is False
    assert evaluation['projected_legacy_target_sha256']==evaluation['saved_data']['target_sha256']


def test_prediction_failure_keeps_bound_observation_without_forecast_credit(generated,monkeypatch):
    def reject(*_args,**_kwargs):raise ValueError('synthetic prediction decoder failure')
    # Do not alter the shared decoder used to authenticate the earlier R/G
    # journal. Only this adapter's subsequent observation evaluation fails.
    monkeypatch.setattr(core,'predictions',NS(evaluate=reject))
    f=facts(generated);row=f['restart'][1]['rows'][0]
    assert row['observation'] is not None and row['prediction'] is None
    assert row['unavailable_reason']=='diagnostic_prediction_invalid_or_unsupported'
    assert row['observation']['bindings']['source_sha256']==f['author']['candidate_source_sha256']
