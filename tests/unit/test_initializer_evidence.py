"""Saved witness and synthetic-array tests. Never execute a helper or solver."""
import base64
from copy import deepcopy
from functools import lru_cache
import hashlib
import itertools
import json

import pytest

from sciel.hypothesis import initializer_evidence as evidence
from sciel.hypothesis import circle_initializer_witness as core


def synthetic_receipt(*, child_shift=0., failed=False):
    """Pure fixture for other registry tests; hashes identify synthetic data only."""
    fixtures=core.make_fixture_bundle()
    identity={k:'d'*64 for k in evidence._IDENTITY if k.endswith('_sha256')}
    identity.update(builder_files={k:'e'*64 for k in ('circle_interventions.py','circle_proposals.py',
        'circle_restart_body.py','circle_restart_scales.py')},builder_contract='circle_restart_body/1',
        python='synthetic Python',platform='synthetic platform',numpy_version='synthetic version')
    source_pins=[{'label':label,'source_sha256':digest*64,'body_sha256':'b'*64,'helper_sha256':'c'*64}
                 for label,digest in [('candidate','1'),('parent','2')]]
    plan={'schema':core.SCHEMA,'identity':identity,'fixtures':fixtures,'sources':source_pins,
        'limits':core._limits(5.),'scope':core.SCOPE}
    plan['plan_sha256']=core._hash(plan)
    sources=[]
    for pin in source_pins:
        label=pin['label'];events=[{'kind':'header','plan_sha256':plan['plan_sha256'],'label':label,'identity':identity}]
        for case in fixtures['cases']:
            events.append({'kind':'intent','case_id':case['id']})
            output=core._decode_array(case['inputs']['ordinary_params'])
            if label=='candidate': output[0]+=child_shift
            broken=failed and label=='candidate'
            events.append({'kind':'observation','case_id':case['id'],'status':'failed' if broken else 'observed',
                'raw_return':None if broken else {'kind':'value','array':core._array(output),
                    'python_type':'ndarray','shares_input_storage':[False]*4},
                'post_inputs':deepcopy(case['inputs']),'effective_output':None if broken else core._array(output),
                'error_code':'ZeroDivisionError' if broken else None,'input_storage_detached':True,
                'handoff_counts':dict(proposal_attempts=1,proposal_accepted=int(not broken),proposal_none=0,proposal_errors=int(broken))})
        events.append({'kind':'terminal','completed':True})
        process={'returncode':0,'failure':None,'elapsed_seconds':0.,'stdout':b''.join(core._canonical(x).encode()+b'\n' for x in events),'stderr':b''}
        sources.append({**pin,**core._consume(process,plan,label)})
    comparisons=[]
    for left,right in itertools.combinations(sources,2):
        rows=[]
        for a,b in zip(left['observations'],right['observations']):
            comparable=a['status']==b['status']=='observed'
            equal=(a['effective_output']['bytes_sha256']==b['effective_output']['bytes_sha256']) if comparable else None
            rows.append({'case_id':a['case_id'],'equal_bits':equal,'first_differing_coordinate':0 if equal is False else None})
        comparisons.append({'left':left['label'],'right':right['label'],
            'status':'incomplete' if failed else 'different_on_witness' if child_shift else 'same_on_witnesses','cases':rows})
    receipt={'schema':core.SCHEMA,'plan':plan,'sources':sources,'comparisons':comparisons,
        'status':'incomplete' if failed else 'observed','scope':core.SCOPE,
        'candidate_rejection_authorized':False,'semantic_equivalence_proven':False,
        'new_model_calls':0,'new_solver_calls':0,'new_native_evaluations':0}
    return {**receipt,'receipt_sha256':core._hash(receipt)}


ROLES={'candidate':'candidate','parent':'parent'}


@lru_cache(maxsize=1)
def _baseline():
    return evidence.build_evidence(synthetic_receipt(),roles=ROLES)


@pytest.fixture
def value(): return deepcopy(_baseline())


def rehash(value):
    value['evidence_sha256']=core._hash({k:v for k,v in value.items() if k!='evidence_sha256'})


def test_synthetic_receipt_compacts_without_calls_and_returns_detached(value,monkeypatch):
    for name in ('identity','prepare_witness','_launch','witness_sources'):
        monkeypatch.setattr(core,name,lambda *a,**k:pytest.fail('historical validation must be pure'))
    validated=evidence.validate_evidence(value,source_sha256='1'*64,parent_source_sha256='2'*64)
    assert validated==value and len(core._canonical(value))<100*1024
    assert 'raw_stdout_base64' not in core._canonical(value)
    validated['sources']['candidate']['observations'].clear()
    assert len(value['sources']['candidate']['observations'])==16


@lru_cache(maxsize=1)
def authored_receipt():
    """Three saved primitive streams: two equal baselines and one translation.

    Only input data and synthetic events are authored here. No source is run,
    and these fixture hashes never claim physical execution provenance.
    """
    template = synthetic_receipt()
    plan = deepcopy(template['plan'])
    labels = ('baseline', 'equivalent', 'translated')
    plan['sources'] = [dict(label=label, source_sha256=str(index + 3) * 64,
                            body_sha256='b' * 64, helper_sha256='c' * 64)
                       for index, label in enumerate(labels)]
    plan['plan_sha256'] = core._hash({k: v for k, v in plan.items() if k != 'plan_sha256'})
    sources = []
    for pin in plan['sources']:
        events = [dict(kind='header', plan_sha256=plan['plan_sha256'],
                       label=pin['label'], identity=plan['identity'])]
        for case in plan['fixtures']['cases']:
            output = core._decode_array(case['inputs']['best_params'])
            if pin['label'] == 'translated' and case['id'] == 'w04':
                # Translate exactly circles 0 and 1 by the same dyadic offset.
                # Their separation/radii stay unchanged; other pair relations
                # need not. The checker must derive this from the saved bytes.
                output[:4] += 1 / 64
            events.extend([
                dict(kind='intent', case_id=case['id']),
                dict(kind='observation', case_id=case['id'], status='observed',
                     raw_return=dict(kind='value', array=core._array(output),
                                     python_type='ndarray', shares_input_storage=[False] * 4),
                     post_inputs=deepcopy(case['inputs']), effective_output=core._array(output),
                     error_code=None, input_storage_detached=True,
                     handoff_counts=dict(proposal_attempts=1, proposal_accepted=1,
                                         proposal_none=0, proposal_errors=0)),
            ])
        events.append(dict(kind='terminal', completed=True))
        process = dict(returncode=0, failure=None, elapsed_seconds=0., stderr=b'',
                       stdout=b''.join(core._canonical(event).encode() + b'\n' for event in events))
        sources.append({**pin, **core._consume(process, plan, pin['label'])})
    comparisons = []
    for left, right in itertools.combinations(labels, 2):
        equal = 'translated' not in (left, right)
        comparisons.append(dict(left=left, right=right,
            status='same_on_witnesses' if equal else 'different_on_witness',
            cases=[dict(case_id=case['id'], equal_bits=equal or case['id'] != 'w04',
                        first_differing_coordinate=0 if not equal and case['id'] == 'w04' else None)
                   for case in plan['fixtures']['cases']]))
    receipt = {**template, 'plan': plan, 'sources': sources, 'comparisons': comparisons}
    receipt['receipt_sha256'] = core._hash({k: v for k, v in receipt.items() if k != 'receipt_sha256'})
    return receipt


@lru_cache(maxsize=2)
def authored_evidence(candidate='equivalent'):
    roles = ({'candidate': candidate, 'parent': 'translated', 'prior': 'baseline'}
             if candidate == 'equivalent' else
             {'candidate': candidate, 'parent': 'equivalent', 'prior': 'baseline'})
    return evidence.build_evidence(authored_receipt(), roles=roles)


def test_saved_receipt_has_finite_equality_not_universal_equivalence(monkeypatch):
    for name in ('identity','prepare_witness','_launch','witness_sources'):
        monkeypatch.setattr(core,name,lambda *a,**k:pytest.fail('no historical execution'))
    value=authored_evidence()
    comparisons={(x['left'],x['right']):x for x in value['derived']['comparisons']}
    assert comparisons['candidate','prior']['status']=='same_on_witnesses'
    assert comparisons['candidate','parent']['status']=='different_on_witness'
    assert len(comparisons['candidate','prior']['cases'])==16
    assert len(core._canonical(value))<100*1024
    assert 'Not universal equivalence' in value['scope']
    assert value['receipt_sha256']==authored_receipt()['receipt_sha256']


def test_generic_two_circle_relation_recomputed_from_saved_outputs():
    value=authored_evidence('translated')
    relations=[r['facts']['two_moved_circles'] for r in value['derived']['candidate_cases'] if r['facts']['two_moved_circles']]
    assert len(relations) == 1
    assert all(r['ids'] == [0, 1] and r['common_translation_exact'] for r in relations)
    for row in relations:
        assert row['squared_distance_delta']=={'numerator':0,'denominator':1}
        assert row['translation'] == [{'numerator':1,'denominator':64}] * 2
    card=evidence.evidence_view(value)
    assert card['status']=='presented' and len(core._canonical(card))<=8192
    assert evidence.validate_evidence_view(card)==card


def test_whole_view_exact_budget_boundary_and_explicit_tiny_omission(value):
    card=evidence.evidence_view(value,max_chars=100000)
    size=len(core._canonical(card))
    assert evidence.evidence_view(value,max_chars=size)==card
    omitted=evidence.evidence_view(value,max_chars=size-1)
    assert omitted['status']=='omitted' and omitted['card_chars']==size
    assert 'candidate_cases' not in omitted and omitted['fits_budget']
    assert evidence.validate_evidence_view(omitted)==omitted
    tiny=evidence.evidence_view(value,max_chars=0)
    assert tiny['status']=='omitted' and tiny['fits_budget'] is False
    assert evidence.validate_evidence_view(tiny)==tiny


def test_cross_receipt_tiny_change_is_difference_with_no_epsilon(value):
    changed=evidence.build_evidence(synthetic_receipt(child_shift=2**-40),roles=ROLES)
    result=evidence.compare_evidence(value,changed)
    assert result['status']=='different_on_witness'
    assert all(r['equal_bits'] is False and r['first_differing_coordinate']==0 for r in result['cases'])
    assert evidence.validate_evidence_comparison(result)==result
    equal=evidence.compare_evidence(value,deepcopy(value))
    assert equal['status']=='same_on_witnesses'


@pytest.mark.parametrize('field',['checker_sha256','proposal_runtime_sha256','handoff_sha256','numpy_core_sha256','python'])
def test_cross_receipt_identity_mismatch_is_not_equality(value,field):
    other=deepcopy(value)
    other['identity'][field]='0'*64 if field.endswith('_sha256') else 'another Python'
    other['plan_sha256']=core._hash(evidence._plan(other,evidence._fixtures()))
    rehash(other)
    result=evidence.compare_evidence(value,other)
    assert result['status']=='incompatible' and result['cases']==[]
    assert evidence.validate_evidence_comparison(result)==result


def test_failures_remain_missing_not_fallback_or_equality(value):
    failed=evidence.build_evidence(synthetic_receipt(failed=True),roles=ROLES)
    rows=failed['sources']['candidate']['observations']
    assert all(x['status']=='failed' and x['effective_output'] is None for x in rows)
    assert all(x['facts'] is None for x in failed['derived']['candidate_cases'])
    result=evidence.compare_evidence(value,failed)
    assert result['status']=='incomplete' and all(r['equal_bits'] is None for r in result['cases'])
    rows[0]['effective_output']=value['sources']['candidate']['observations'][0]['effective_output']
    rehash(failed)
    with pytest.raises(ValueError,match='missing output'):
        evidence.validate_evidence(failed)


@pytest.mark.parametrize('mutation',[
    lambda v:v['sources']['candidate']['observations'].pop(),
    lambda v:v['sources']['candidate']['observations'].__setitem__(1,deepcopy(v['sources']['candidate']['observations'][0])),
    lambda v:v['sources']['candidate']['observations'][0]['handoff_counts'].__setitem__('proposal_attempts',True),
    lambda v:v['derived']['candidate_cases'][0]['facts'].__setitem__('changed_radii',25),
    lambda v:v['derived']['comparisons'][0]['cases'][0].__setitem__('equal_bits',False),
    lambda v:v['roles'].__setitem__('parent','candidate'),
    lambda v:v['sources']['candidate'].__setitem__('error_code','fabricated'),
])
def test_restamped_retained_tampering_rejected(value,mutation):
    mutation(value);rehash(value)
    with pytest.raises(ValueError): evidence.validate_evidence(value)


def test_candidate_parent_ordinary_source_binding(value):
    with pytest.raises(ValueError,match='ordinary candidate'):
        evidence.validate_evidence(value,source_sha256='9'*64)
    with pytest.raises(ValueError,match='ordinary parent'):
        evidence.validate_evidence(value,parent_source_sha256='9'*64)


def test_outer_digest_does_not_replace_fact_rederivation():
    receipt=deepcopy(authored_receipt())
    receipt['sources'][0]['observations'][0]['facts']['changed_centers']=999
    receipt['receipt_sha256']=core._hash({k:v for k,v in receipt.items() if k!='receipt_sha256'})
    with pytest.raises(ValueError,match='raw streams/facts'):
        evidence.build_evidence(receipt,roles={'candidate':'equivalent','parent':'translated','prior':'baseline'})


def test_restamped_fraction_changes_rejected():
    value=deepcopy(authored_evidence('translated'))
    relation=next(r['facts']['two_moved_circles'] for r in value['derived']['candidate_cases'] if r['facts']['two_moved_circles'])
    relation['squared_distance_delta']['denominator']*=2
    rehash(value)
    with pytest.raises(ValueError,match='derived'):
        evidence.validate_evidence(value)


def test_output_hash_and_finite_shape_constraints(value):
    output=value['sources']['candidate']['observations'][0]['effective_output']
    output['data_base64']=base64.b64encode(b'\x00'*616).decode()
    output['bytes_sha256']=hashlib.sha256(b'\x00'*616).hexdigest()
    rehash(value)
    with pytest.raises(ValueError,match='encoding'):
        evidence.validate_evidence(value)


def test_cached_facts_never_cache_validation_or_share_mutable_output(value,monkeypatch):
    evidence._candidate_derivation.cache_clear()
    first=evidence.validate_evidence(value)
    assert evidence._candidate_derivation.cache_info().misses==1
    monkeypatch.setattr(evidence,'_candidate_facts',lambda *a:pytest.fail('identical content should use bounded pure cache'))
    assert evidence.validate_evidence(value)==first
    value['derived']['candidate_cases'][0]['facts']['changed_radii']=26
    rehash(value)
    with pytest.raises(ValueError,match='derived'):
        evidence.validate_evidence(value)
    assert evidence._candidate_derivation.cache_info().maxsize==256
    evidence._candidate_derivation.cache_clear()


def test_view_and_comparison_transport_reject_restamped_false_shapes(value):
    card=evidence.evidence_view(value)
    card['candidate_cases'][0][1]['distance']=[0,324,0]
    card['view_sha256']=core._hash({k:v for k,v in card.items() if k!='view_sha256'})
    with pytest.raises(ValueError,match='pair total'):
        evidence.validate_evidence_view(card)
    result=evidence.compare_evidence(value,value)
    result['cases'][0]['equal_bits']=None
    result['comparison_sha256']=core._hash({k:v for k,v in result.items() if k!='comparison_sha256'})
    with pytest.raises(ValueError,match='equality'):
        evidence.validate_evidence_comparison(result)


def replace_candidate_stream(receipt, events, *, failure=None, returncode=0):
    """Pure recapture of edited synthetic event bytes, never execute their source."""
    pin=receipt['plan']['sources'][0]
    process={'returncode':returncode,'failure':failure,'elapsed_seconds':0.,
        'stdout':b''.join(core._canonical(x).encode()+b'\n' for x in events),'stderr':b''}
    receipt['sources'][0]={**pin,**core._consume(process,receipt['plan'],pin['label'])}
    receipt['receipt_sha256']=core._hash({k:v for k,v in receipt.items() if k!='receipt_sha256'})
    return receipt


def test_none_uses_saved_ordinary_even_after_argument_mutation():
    receipt=synthetic_receipt()
    events=[json.loads(line) for line in base64.b64decode(receipt['sources'][0]['raw_stdout_base64']).splitlines()]
    for event in events:
        if event['kind']!='observation': continue
        event['raw_return']={'kind':'none','array':None,'python_type':'NoneType','shares_input_storage':None}
        event['handoff_counts']=dict(proposal_attempts=1,proposal_accepted=0,proposal_none=1,proposal_errors=0)
        changed=core._decode_array(event['post_inputs']['ordinary_params']);changed[:]=0
        event['post_inputs']['ordinary_params']=core._array(changed)
    value=evidence.build_evidence(replace_candidate_stream(receipt,events),roles=ROLES)
    assert all(r['effective_origin']=='saved_ordinary' and r['raw_return_kind']=='none'
        for r in value['sources']['candidate']['observations'])
    assert value['derived']['comparisons'][0]['status']=='same_on_witnesses'
    obs=value['sources']['candidate']['observations'][0]
    obs['effective_output']=evidence._compact(core._array(changed))
    rehash(value)
    with pytest.raises(ValueError,match='saved fallback'):
        evidence.validate_evidence(value)


def test_timeout_retains_known_prefix_and_cannot_rescue_equality(value):
    receipt=synthetic_receipt()
    events=[json.loads(line) for line in base64.b64decode(receipt['sources'][0]['raw_stdout_base64']).splitlines()][:4]
    receipt['status']='incomplete';receipt['comparisons'][0]['status']='incomplete'
    for row in receipt['comparisons'][0]['cases'][1:]: row['equal_bits']=None
    partial=evidence.build_evidence(replace_candidate_stream(receipt,events,failure='wall_timeout',returncode=-9),roles=ROLES)
    rows=partial['sources']['candidate']['observations']
    assert [r['status'] for r in rows]==['observed','unknown']+['not_dispatched']*14
    result=evidence.compare_evidence(value,partial)
    assert result['status']=='incomplete'
    assert [r['equal_bits'] for r in result['cases']]==[True]+[None]*15
    assert all(r['facts'] is None for r in partial['derived']['candidate_cases'][1:])


def test_no_terminal_receipt_prevents_same_even_with_all_sixteen_outputs(value):
    receipt=synthetic_receipt()
    events=[json.loads(line) for line in base64.b64decode(receipt['sources'][0]['raw_stdout_base64']).splitlines()][:-1]
    receipt['status']='incomplete';receipt['comparisons'][0]['status']='incomplete'
    partial=evidence.build_evidence(replace_candidate_stream(receipt,events),roles=ROLES)
    result=evidence.compare_evidence(value,partial)
    assert result['status']=='incomplete' and all(r['equal_bits'] is True for r in result['cases'])
    assert evidence.validate_evidence_comparison(result)==result


def test_retained_source_pin_must_match_recorded_plan(value):
    value['sources']['candidate']['source_sha256']='9'*64
    rehash(value)
    with pytest.raises(ValueError,match='retained plan binding'):
        evidence.validate_evidence(value)


def test_negative_radius_overlap_is_undefined_for_affected_pairs(value):
    import struct
    output=value['sources']['candidate']['observations'][0]['effective_output']
    numbers=list(struct.unpack('<78d',base64.b64decode(output['data_base64'])))
    numbers[52]=-.5
    raw=struct.pack('<78d',*numbers)
    output.update(data_base64=base64.b64encode(raw).decode(),bytes_sha256=hashlib.sha256(raw).hexdigest())
    value['derived']=evidence._derive(value,evidence._fixtures());rehash(value)
    checked=evidence.validate_evidence(value)
    facts=checked['derived']['candidate_cases'][0]['facts']
    assert facts['changed_radii']==2  # Edited radius0 plus ordinary floor of initial radius25.
    assert facts['overlaps']['after_undefined']==25
    assert sum(facts['pair_radius_sum'].values())==325
