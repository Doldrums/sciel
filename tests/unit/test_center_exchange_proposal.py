"""Pure/fake profile and proposal tests; no helper, optimizer, API or subprocess."""
from copy import deepcopy
from fractions import Fraction
from functools import lru_cache

import pytest
from sciel.hypothesis import circle_initializer_witness as core
from sciel.hypothesis import circle_restart_body as builder
from sciel.hypothesis import initializer_evidence as evidence

ROLES={'candidate':'candidate','parent':'parent'}
PAIRS={'u05':(0,2),'u06':(0,2),'u07':(0,2),'u08':(1,0),
       'u09':(25,2),'u10':(0,2),'u11':(1,0),'u13':(0,3)}


def codes():
    # Source labels for fake child observations; no helper is executed here.
    body = "x = best_params.copy()\nx[:2] = best_params[4:6]\nx[4:6] = best_params[:2]\nreturn x\n"
    return {'control': builder.source_for(), 'single': builder.source_for(body)}


def process(plan,label,mode='swap'):
    events=[dict(kind='header',plan_sha256=plan['plan_sha256'],label=label,identity=plan['identity'])]
    for case in plan['fixtures']['cases']:
        events.append(dict(kind='intent',case_id=case['id']))
        if mode=='incomplete' and label=='candidate':break
        pair=PAIRS.get(case['id']) if label=='candidate' else None
        if mode=='none':pair=None
        output=core._decode_array(case['inputs']['best_params'] if pair else case['inputs']['ordinary_params'])
        if pair:
            i,j=pair;a=output[2*i:2*i+2].copy();output[2*i:2*i+2]=output[2*j:2*j+2];output[2*j:2*j+2]=a
            if mode=='full_label':output[52+i],output[52+j]=output[52+j],output[52+i]
            if mode=='wrong_tie' and case['id']=='u13':
                output=core._decode_array(case['inputs']['best_params']);i,j=0,4
                a=output[:2].copy();output[:2]=output[8:10];output[8:10]=a
        events.append(dict(kind='observation',case_id=case['id'],status='observed',
            raw_return=dict(kind='value' if pair else 'none',array=core._array(output) if pair else None,
                python_type='ndarray' if pair else 'NoneType',shares_input_storage=[False]*4 if pair else None),
            post_inputs=deepcopy(case['inputs']),effective_output=core._array(output),error_code=None,input_storage_detached=True,
            handoff_counts=dict(proposal_attempts=1,proposal_accepted=int(pair is not None),proposal_none=int(pair is None),proposal_errors=0)))
    partial=mode=='incomplete' and label=='candidate'
    if not partial:events.append(dict(kind='terminal',completed=True))
    return dict(returncode=-9 if partial else 0,failure='timeout' if partial else None,elapsed_seconds=0.,
        stdout=b''.join(core._canonical(x).encode()+b'\n' for x in events),stderr=b'')


@lru_cache(maxsize=8)
def receipt(profile='unequal_radii_v1',mode='swap'):
    values=codes();sources={'candidate':values['single'],'parent':values['control']}
    pins={k:core._sha(v.encode()) for k,v in sources.items()}
    plan=core.prepare_witness(sources,core.make_fixture_bundle(profile),expected_source_sha256=pins)
    results=[{**pin,**core._consume(process(plan,pin['label'],mode),plan,pin['label'])} for pin in plan['sources']]
    compact={s['label']:{**s,'observations':[{**o,'effective_output':evidence._compact(o['effective_output'])
        if o['status']=='observed' else None} for o in s['observations']]} for s in results}
    comparison=evidence._compare_sources(compact['candidate'],compact['parent'],[c['id'] for c in plan['fixtures']['cases']])
    value=dict(schema=core.SCHEMA,plan=plan,sources=results,comparisons=[dict(left='candidate',right='parent',**comparison)],
        status='observed' if all(s['status']=='observed' for s in results) else 'incomplete',scope=core.SCOPE,
        candidate_rejection_authorized=False,semantic_equivalence_proven=False,new_model_calls=0,new_solver_calls=0,new_native_evaluations=0)
    return {**value,'receipt_sha256':core._hash(value)}


def test_default_recipe_hash_is_byte_identical_and_named_profiles_are_detached():
    old=core.make_fixture_bundle()
    assert old==core.make_fixture_bundle('legacy_v1')
    assert old['sha256']=='ed291e3bbe8632b09303a06ceb4f9a4415cd311775cee6da448f35339f938d4f'
    new=core.make_fixture_bundle('unequal_radii_v1');assert new['sha256']!=old['sha256']
    assert len(new['cases'])==16 and [c['id'] for c in new['cases']]==[f'u{i:02d}' for i in range(16)]
    new['cases'].clear();assert len(core.make_fixture_bundle('unequal_radii_v1')['cases'])==16


@pytest.mark.parametrize('profile',['unknown',None,True])
def test_unknown_profile_rejected(profile):
    with pytest.raises(ValueError,match='fixture_profile'):core.make_fixture_bundle(profile)


def test_unregistered_rehashed_inputs_rejected_before_launch(monkeypatch):
    monkeypatch.setattr(core,'_launch',lambda *a:pytest.fail('no launch'))
    fixed=core.make_fixture_bundle('unequal_radii_v1');fixed['cases'][5]['restart_index']=17
    fixed['sha256']=core._hash({k:v for k,v in fixed.items() if k!='sha256'})
    src={'x':builder.source_for()}
    with pytest.raises(ValueError,match='unsupported fixed fixture'):
        core.prepare_witness(src,fixed,expected_source_sha256={'x':core._sha(src['x'].encode())})


def test_profile_incumbents_meet_pinned_input_constraints_without_trajectory_claim():
    bundle=core.make_fixture_bundle('unequal_radii_v1')
    for case in bundle['cases']:
        v=[Fraction.from_float(float(x)) for x in core._decode_array(case['inputs']['best_params'])]
        tol=Fraction.from_float(1e-6)
        assert all(v[52+i]>=0 and all(-tol<=v[2*i+d]<=1+tol and
            v[52+i]-tol<=v[2*i+d]<=1-v[52+i]+tol for d in (0,1)) for i in range(26))
        assert all((v[2*i]-v[2*j])**2+(v[2*i+1]-v[2*j+1])**2>=max(0,v[52+i]+v[52+j]-tol)**2
            for i in range(26) for j in range(i))
        assert sum(v[52:])>=Fraction(1,4)
        assert case['domain'].endswith('reachability_not_proven')
        if case['restart_index']==0:
            assert case['inputs']['best_params']==case['inputs']['initial_params']
    assert bundle['cases'][13]['fixture_kind']=='wall_filtered_tie'
    assert bundle['cases'][14]['fixture_kind']=='wall_no_partner'


def test_body_compiles_only_to_expected_source_and_scaffold():
    values = codes()
    body = builder.body_of(values['single'])
    result = builder.apply(values['control'], body)
    assert result['code'] == values['single'] != values['control']
    assert builder.body_of(result['code']) == body
    assert result['proof']['unchanged_outside_helper'] is True
    assert result['proof']['list_policy'] == builder.list_policy_identity()


def test_new_profile_evidence_and_whole_view_roundtrip_without_current_execution(monkeypatch):
    raw=deepcopy(receipt())
    for name in ('identity','prepare_witness','_launch','witness_sources'):
        monkeypatch.setattr(core,name,lambda *a,**k:pytest.fail('pure historical read'))
    value=evidence.build_evidence(raw,roles=ROLES)
    assert evidence.validate_evidence(value)==value
    view=evidence.evidence_view(value,max_chars=20000)
    assert evidence.validate_evidence_view(view)==view
    assert view['case_ids'][0]=='u00'
    omitted=evidence.evidence_view(value,max_chars=700)
    assert omitted['status']=='omitted' and 'candidate_cases' not in omitted
    assert evidence.validate_evidence_view(omitted)==omitted


def test_cross_profile_comparison_is_incompatible_even_at_same_runtime():
    old=evidence.build_evidence(receipt('legacy_v1'),roles=ROLES)
    new=evidence.build_evidence(receipt(),roles=ROLES)
    assert old['identity']==new['identity']
    result=evidence.compare_evidence(old,new)
    assert result['status']=='incompatible' and result['cases']==[] and result['fixture_sha256'] is None
    assert evidence.validate_evidence_comparison(result)==result
    same=evidence.compare_evidence(new,new)
    assert same['status']=='same_on_witnesses' and same['fixture_sha256']==new['fixture_sha256']
    assert evidence.validate_evidence_comparison(same)==same


def test_legacy_comparison_transport_still_readable_new_profile_cannot_masquerade():
    old=evidence.build_evidence(receipt('legacy_v1'),roles=ROLES)
    value=evidence.compare_evidence(old,old);value['schema']=evidence.LEGACY_COMPARISON_SCHEMA
    value.pop('fixture_sha256');value['comparison_sha256']=core._hash({k:v for k,v in value.items() if k!='comparison_sha256'})
    assert evidence.validate_evidence_comparison(value)==value
    value['cases'][0]['case_id']='u00';value['comparison_sha256']=core._hash({k:v for k,v in value.items() if k!='comparison_sha256'})
    with pytest.raises(ValueError,match='case order'):evidence.validate_evidence_comparison(value)


def test_rehashed_profile_substitution_cannot_change_retained_inputs():
    value=evidence.build_evidence(receipt(),roles=ROLES)
    value['fixture_sha256']=evidence.FIXTURE_SHA256
    value['evidence_sha256']=core._hash({k:v for k,v in value.items() if k!='evidence_sha256'})
    with pytest.raises(ValueError):evidence.validate_evidence(value)


def test_real_witness_orchestrator_with_fake_child_receipts_only(monkeypatch):
    values=codes();sources={'candidate':values['single'],'parent':values['control']};calls=[]
    def fake(request,python,timeout):
        calls.append(request['label']);return process(request['plan'],request['label'])
    monkeypatch.setattr(core,'_launch',fake)
    result=core.witness_sources(sources,core.make_fixture_bundle('unequal_radii_v1'),
        expected_source_sha256={k:core._sha(v.encode()) for k,v in sources.items()})
    assert calls==['candidate','parent'] and result['status']=='observed'
    measured = evidence.build_evidence(result, roles=ROLES)
    assert evidence.validate_evidence(measured) == measured
    assert measured['fixture_sha256'] == core.make_fixture_bundle('unequal_radii_v1')['sha256']


@pytest.mark.parametrize('value',[None,[],3])
def test_malformed_comparison_is_value_error(value):
    with pytest.raises(ValueError):evidence.validate_evidence_comparison(value)
