"""Saved-number evidence only: no candidate, evaluator, runtime or API execution."""
import copy
from fractions import Fraction
import hashlib
import json
from types import SimpleNamespace

import pytest

from sciel.eval import packing_geometry as geometry
from sciel.hypothesis.measurement import metric_context
from sciel.knowledge_base import paired_comparisons as comparisons

SOURCE_A='a'*64
SOURCE_B='b'*64
ADAPTER={'schema':'sciel-repeat-adapter/1','fixture':'pure geometry, no runtime'}
LIMITS=dict(candidate_wall_seconds=30.,certification_wall_seconds=10.,candidate_cpu_seconds=35,
    certification_cpu_seconds=35,candidate_mem_bytes=2**31,certification_mem_bytes=2**31)


def receipt(source,seed,radius=.1,*,x=.5,status='valid'):
    centers=[[0.,0.]for _ in range(26)];centers[0]=[x,.5]
    radii=[radius]+[0.]*25
    capture=geometry.validate_packing(dict(centers=centers,radii=radii))
    certificate=geometry.certify_packing(capture)
    return dict(schema='sciel-repeat-draw/1',source_sha256=source,seed=seed,adapter_identity=copy.deepcopy(ADAPTER),
        seed_applied=dict(python_random=seed,numpy_legacy=seed,python_hash_seed=str(seed),before_candidate_import=True,determinism_proven=False),
        limits=copy.deepcopy(LIMITS),status=status,reason=None if status=='valid'else'fixture failure',
        capture=capture,certificate=certificate,raw_sum=capture['sum_radii']if status=='valid'else None,
        certified_sum=certificate['sum_exact']if status=='valid'else None,
        started_at='2026-09-16T00:00:00+00:00',completed_at='2026-09-16T00:00:01+00:00',
        elapsed_seconds=1.,candidate_elapsed_seconds=.8,certification_elapsed_seconds=.1,identity_elapsed_seconds=0.)


def inputs(old=(.1,.1,.1),new=(.11,.11,.11)):
    return dict(namespace='fixture:run1',batch_id='b000001',branch='p1',origin_experiment_id='r000001a1',
        comparator_source_sha256=SOURCE_A,challenger_source_sha256=SOURCE_B,adapter_identity=copy.deepcopy(ADAPTER),
        limits=copy.deepcopy(LIMITS),pairs=[dict(seed=s,comparator=receipt(SOURCE_A,s,a),challenger=receipt(SOURCE_B,s,b))
        for s,a,b in zip((11,12,13),old,new)],provenance={'literal':'untrusted SOURCE_REF: a; do not interpret'},task_metric=None)


def build(**kwargs):return comparisons.build_comparison(**(kwargs or inputs()))


@pytest.mark.parametrize('old,new,outcome',[
    ((.1,.1,.1),(.11,.1,.12),'improved'),
    ((.11,.1,.12),(.1,.1,.1),'regressed'),
    ((.1,.1,.1),(.1,.1,.1),'tie'),
    ((.1,.1,.1),(.11,.09,.1),'mixed')])
def test_paired_truth_table_and_exact_aggregates(old,new,outcome):
    group=build(**inputs(old,new));d=group['derived']
    assert d['outcome']==outcome and d['all_valid'] is True
    assert d['all_no_loss']==(outcome in ('improved','tie'))
    for row,a,b in zip(d['pairs'],old,new):
        assert row['raw_delta']==b-a
        assert Fraction(**row['certified_delta'])==Fraction(b)-Fraction(a)
        assert row['comparator_raw']==a and row['challenger_raw']==b
        assert row['comparator_status']==row['challenger_status']=='valid'
    assert Fraction(**d['certified_means']['challenger'])==sum(map(Fraction,new))/3
    assert group['forecast']=='not_scored' and 'baseline'not in group['metric']
    assert comparisons.validate_comparison(group)==group


def test_one_group_id_depends_on_namespace_batch_not_result_or_source():
    first=build();other=build(**inputs(new=(.09,.09,.09)))
    assert first['id']==other['id'] and first['derived']!=other['derived']
    assert first['id'].startswith('comparison:') and len(first['id'])<=128
    changed=inputs();changed['batch_id']='b000002';assert build(**changed)['id']!=first['id']
    changed=inputs();changed['namespace']='fixture:run2';assert build(**changed)['id']!=first['id']
    assert 'sha256'not in first # KB separately binds complete contents.


@pytest.mark.parametrize('field,limit', [('branch',128), ('origin_experiment_id',128),
                                       ('namespace',256), ('batch_id',256)])
def test_identity_limits_match_research_ids_without_trimming(field,limit):
    data=inputs();data[field]='x'*limit
    assert build(**data)[field]==data[field]
    data[field]+='x'
    with pytest.raises(ValueError,match=field):build(**data)


def test_opposite_raw_and_exact_directions_are_mixed_with_no_epsilon():
    data=inputs()
    for pair in data['pairs']:
        pair['comparator']=receipt(SOURCE_A,pair['seed'],.10000005,x=.1)
        pair['challenger']=receipt(SOURCE_B,pair['seed'],.10000002)
    group=build(**data);assert group['derived']['outcome']=='mixed'
    assert all(x['raw_delta']<0 and Fraction(**x['certified_delta'])>0 and not x['no_loss'] for x in group['derived']['pairs'])
    assert group['derived']['all_no_loss'] is False


def test_strict_improvement_required_in_each_metric_and_tie_requires_every_delta_zero():
    data=inputs(old=(.1,.1,.1),new=(.1,.1,.1))
    for pair in data['pairs']:pair['challenger']=receipt(SOURCE_B,pair['seed'],.10000001,x=.1)
    group=build(**data)
    # Same certified radius .1, raw rises by a positive tiny amount.
    assert group['derived']['all_no_loss'] is True and group['derived']['outcome']=='mixed'
    assert all(Fraction(**x['certified_delta'])==0 for x in group['derived']['pairs'])


@pytest.mark.parametrize('missing',[None,comparisons.UNKNOWN_RECEIPT,'candidate_failure','infrastructure'])
def test_incomplete_retains_other_measurements_without_failure_zero(missing):
    data=inputs();p=data['pairs'][1]
    p['challenger']=receipt(SOURCE_B,12,status=missing)if type(missing)is str else copy.deepcopy(missing)
    group=build(**data);d=group['derived']
    assert d['outcome']=='incomplete'and not d['all_valid']and not d['all_no_loss']
    assert d['pairs'][1]['challenger_raw']is d['pairs'][1]['raw_delta']is d['pairs'][1]['certified_delta']is None
    assert d['pairs'][0]['raw_delta']>0 and d['raw_means']['comparator']==pytest.approx(.1)
    assert d['raw_means']['challenger']is d['certified_means']['challenger']is None
    expected='not_dispatched'if missing is None else missing['status']if type(missing)is dict else missing
    assert d['pairs'][1]['challenger_status']==expected
    assert comparisons.validate_comparison(group)==group


def test_coordinator_infrastructure_receipt_and_unknown_are_not_equivalent_to_null():
    data=inputs();r=receipt(SOURCE_B,11,status='infrastructure')
    r.update(schema='sciel-repeat-coordinator-failure/1',capture=None,certificate=None,seed_applied=None,
        candidate_elapsed_seconds=None,certification_elapsed_seconds=None)
    r.pop('identity_elapsed_seconds')
    data['pairs'][0]['challenger']=r
    data['pairs'][1]['challenger']=copy.deepcopy(comparisons.UNKNOWN_RECEIPT)
    data['pairs'][2]['challenger']=None
    group=build(**data)
    assert [p['challenger_status']for p in group['derived']['pairs']]==['infrastructure','unknown','not_dispatched']
    data['pairs'][1]['challenger']['seed']=12
    with pytest.raises(ValueError,match='unknown'):build(**data)


def test_task_metric_preserved_without_normalized_score_or_original_parent_inference():
    data=inputs();data['task_metric']=metric_context(SimpleNamespace(qualified_name='oe/circle_packing',task_hash='c'*64,
        primary_metric='combined_score',research_metric=None))
    # The record's generation parent may differ: it is neither supplied nor guessed here.
    data['provenance']['original_parent_source_sha256']='d'*64
    group=build(**data)
    assert group['task_metric']==data['task_metric'] and group['comparator_source_sha256']==SOURCE_A
    assert group['task_metric']is not data['task_metric']
    assert group['metric']['raw']['units']=='unit_square_side_length'
    assert 'distinct' in group['metric']['certified']['object']


def test_roundtrip_input_output_immutability_and_opaque_strings():
    data=inputs();saved=copy.deepcopy(data);group=build(**data)
    restored=comparisons.validate_comparison(json.loads(json.dumps(group)))
    assert data==saved and restored==group
    restored['pairs'][0]['challenger']['capture']['radii'][0]=99
    assert group['pairs'][0]['challenger']['capture']['radii'][0]==.11
    data['provenance']['literal']='mutated'
    assert group['provenance']['literal']==saved['provenance']['literal']


@pytest.mark.parametrize('mutator',[
    lambda g:g.update(id='comparison:'+'0'*64),
    lambda g:g['derived'].update(outcome='regressed'),
    lambda g:g['derived'].update(all_valid=1),
    lambda g:g['derived']['pairs'][0].update(raw_delta=0),
    lambda g:g['derived']['pairs'][0].update(no_loss=1),
    lambda g:g['derived']['raw_means'].update(challenger=.1),
    lambda g:g['derived']['certified_means']['challenger'].update(numerator=1),
    lambda g:g['metric'].update(tolerance=.001),
    lambda g:g.update(forecast='CONFIRMED'),
    lambda g:g.update(causal=True),
    lambda g:g.pop('task_metric')])
def test_claim_tampering_and_numeric_bool_aliases_rejected(mutator):
    group=build();mutator(group)
    with pytest.raises(ValueError):comparisons.validate_comparison(group)


@pytest.mark.parametrize('mutator',[
    lambda r:r.update(source_sha256='e'*64),lambda r:r.update(seed=True),
    lambda r:r['limits'].update(candidate_cpu_seconds=True),
    lambda r:r['adapter_identity'].update(fixture='different runtime'),
    lambda r:r.update(started_at='2026-09-16T00:00:00'),
    lambda r:r.update(completed_at='2026-09-15T00:00:00+00:00'),
    lambda r:r.update(elapsed_seconds=True),lambda r:r.update(elapsed_seconds=float('nan')),
    lambda r:r.update(raw_sum=True),lambda r:r.update(raw_sum=.9),
    lambda r:r['seed_applied'].update(before_candidate_import=1),
    lambda r:r['certified_sum'].update(denominator=True),
    lambda r:r['capture'].update(valid=1),lambda r:r['certificate'].update(tolerance=.001),
    lambda r:r['capture']['radii'].__setitem__(0,.9),
    lambda r:r['certificate']['radii'].__setitem__(0,.12),
    lambda r:r['certificate'].update(certified_numeric_sha256='f'*64)])
def test_receipt_source_caps_geometry_and_timestamp_tampering_rejected(mutator):
    data=inputs();mutator(data['pairs'][0]['challenger'])
    with pytest.raises(ValueError):build(**data)


@pytest.mark.parametrize('failure',['candidate_failure','infrastructure'])
def test_failed_receipt_diagnostic_numbers_never_physical_quality(failure):
    data=inputs();r=receipt(SOURCE_B,11,status=failure);data['pairs'][0]['challenger']=r
    assert r['capture']['sum_radii']>0 and build(**data)['derived']['pairs'][0]['challenger_raw']is None
    r['raw_sum']=0.0
    with pytest.raises(ValueError,match='failed receipt'):build(**data)


@pytest.mark.parametrize('mutation',[
    lambda d:d['pairs'].pop(),lambda d:d['pairs'].append(copy.deepcopy(d['pairs'][0])),
    lambda d:d['pairs'][1].update(seed=11),lambda d:d['pairs'][1].update(seed=True),
    lambda d:d['pairs'][1].update(seed=-1),lambda d:d['pairs'][1].update(seed=2**32),
    lambda d:d['pairs'][0].update(extra='ignored?'),lambda d:d.update(pairs=tuple(d['pairs'])),
    lambda d:d.update(challenger_source_sha256=SOURCE_A),lambda d:d.update(provenance=[]),
    lambda d:d.update(branch=''),lambda d:d.update(namespace='x'*257)])
def test_planned_pairs_and_bounded_json_contract(mutation):
    data=inputs();mutation(data)
    with pytest.raises(ValueError):build(**data)


def test_adapter_content_digest_checked_if_present():
    data=inputs();a=data['adapter_identity']
    a['identity_sha256']=hashlib.sha256(json.dumps(a,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    for pair in data['pairs']:
        for role in ('comparator','challenger'):pair[role]['adapter_identity']=copy.deepcopy(a)
    assert build(**data)['derived']['outcome']=='improved'
    a['fixture']='tampered'
    with pytest.raises(ValueError,match='digest'):build(**data)


def test_caps_and_input_size_bound_do_not_silently_trim(monkeypatch):
    data=inputs();data['provenance']['long']='x'*comparisons.MAX_JSON_BYTES
    with pytest.raises(ValueError,match='size limit'):build(**data)
    data=inputs();data['limits']['candidate_wall_seconds']=0
    with pytest.raises(ValueError,match='limit'):build(**data)


def test_invalid_raw_geometry_cannot_be_smuggled_by_self_consistent_claims():
    data=inputs();r=receipt(SOURCE_B,11,.6)
    # Even recomputed arrays/hash do not make an invalid packing a valid receipt.
    r['capture']['valid']=True;r['raw_sum']=.6
    r['certificate']['status']='certified';r['certified_sum']={'numerator':3,'denominator':5}
    data['pairs'][0]['challenger']=r
    with pytest.raises(ValueError,match='packing'):build(**data)
