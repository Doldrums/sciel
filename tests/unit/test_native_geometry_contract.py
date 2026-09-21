"""Synthetic saved arrays only: no evaluator, candidate, API or runtime probe."""
from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import math
from types import SimpleNamespace

import numpy as np
import pytest

from sciel.eval.native_geometry import (SCHEMA, VIEW_SCHEMA, CERTIFICATE_SCHEMA,
    MAX_ARTIFACT_BYTES, parse_artifact, validate_geometry, validate_binding,
    view_geometry, validate_view, certify_artifact)
from sciel.hypothesis.measurement import metric_context
from sciel.engine.repeat_support import BODY_TASK_HASH, CANONICAL_TASK_HASH, EVALUATOR_SHA, HISTORICAL_CIRCLE_TASK_HASHES


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def fixture(dtype='float64', *, validity=True, numeric=None, confirmation=None):
    if numeric is None:
        numeric = {'centers': [[(i % 6 + .5)/6, (i//6 + .5)/5] for i in range(26)],
                   'radii': [.01]*26}
    c = np.array(numeric['centers'], dtype=dtype)
    r = np.array(numeric['radii'], dtype=dtype)
    numeric = {'centers':[[float(v) for v in row] for row in c], 'radii':[float(v) for v in r]}
    total = float(np.sum(r)) if validity else 0.
    return {'schema':SCHEMA, 'source_sha256':'1'*64, 'evaluator_sha256':EVALUATOR_SHA,
            'invocation_id':'a'*32, 'stage':'full', 'confirmation':confirmation,
            'numeric':numeric, 'numeric_sha256':digest(numeric), 'centers_dtype':str(c.dtype),
            'radii_dtype':str(r.dtype), 'native_validity':validity,
            'native_sum_radii':total, 'native_score':total/2.635}


def observation(g, *, reached=None, ok=True, confirmed=False):
    cases = {'sum_radii':g['native_sum_radii'], 'validity':float(g['native_validity']),
             'combined_score':g['native_score'], 'target_ratio':g['native_score']}
    if g['confirmation']:
        cases['n_trials'] = float(g['confirmation']['trials'])
    return {'ok':ok, 'score':g['native_score'] if ok else 0., 'cases':cases,
            'reached':reached or g['stage'], 'confirmed':confirmed, 'raw_first_score':.99}


def metric():
    semantics={'kind':'validity_gated_ratio', 'score_units':'dimensionless',
        'numerator_case':'sum_radii', 'numerator_units':'unit_square_side_length',
        'denominator':2.635, 'validity_case':'validity', 'valid_value':1.,
        'invalid_score':0., 'invalid_numerator_value':0.,
        'invalid_numerator_semantics':'penalty_sentinel_not_physical_measurement'}
    return metric_context(SimpleNamespace(research_metric=semantics,
        qualified_name='oe/circle_packing', task_hash=BODY_TASK_HASH, primary_metric='combined_score'))


def test_roundtrip_detached_complete_native_observation():
    g=fixture(); original=deepcopy(g)
    parsed=parse_artifact(canonical(g))
    assert parsed==g and parsed is not g and parsed['numeric'] is not g['numeric']
    assert validate_binding(g,observation(g),source_sha256='1'*64,metric=metric())==g
    parsed['numeric']['radii'][0]=999.
    assert g==original


@pytest.mark.parametrize('dtype',['float16','float32','float64'])
def test_original_float_dtype_exactly_reconstructed(dtype):
    g=fixture(dtype)
    assert validate_geometry(g)==g
    assert parse_artifact(canonical(g))==g


@pytest.mark.parametrize('dtype',['int8','int16','int32','int64','uint8','uint16','uint32','uint64'])
def test_integer_dtype_metadata_rejected_even_for_zero_geometry(dtype):
    g=fixture(dtype,numeric={'centers':[[0.,0.]]*26,'radii':[0.]*26})
    with pytest.raises(ValueError,match='unsupported original numeric dtype'):validate_geometry(g)


def test_float32_sum_must_not_be_replaced_by_binary64_reduction():
    g=fixture('float32');r=np.array(g['numeric']['radii'],dtype='float32')
    f64=float(np.sum(r.astype('float64')))
    assert g['native_sum_radii']!=f64
    tampered=deepcopy(g);tampered.update(native_sum_radii=f64,native_score=f64/2.635)
    with pytest.raises(ValueError,match='dtype score'):validate_geometry(tampered)
    assert validate_geometry(g)==g


@pytest.mark.parametrize('change',[
    {'schema':VIEW_SCHEMA}, {'extra':'unexpected'}, {'source_sha256':'A'*64},
    {'evaluator_sha256':'2'*63}, {'invocation_id':'x'*32}, {'stage':'confirm'},
    {'native_validity':1}, {'native_sum_radii':True}, {'native_score':0},
    {'native_score':float('inf')}, {'native_sum_radii':float('nan')},
    {'numeric_sha256':'0'*64}, {'centers_dtype':'double'}, {'radii_dtype':'>f8'},
    {'radii_dtype':'complex64'}, {'radii_dtype':'bool'}, {'radii_dtype':'float128'},
])
def test_schema_metadata_and_scalar_tampering_rejected(change):
    g=fixture();g.update(change)
    with pytest.raises(ValueError):validate_geometry(g)


@pytest.mark.parametrize('damage',['row','radii','tuple','bool','int','nan','inf','negative_zero_cast'])
def test_numeric_shape_numbers_and_exact_dtype_representation(damage):
    g=fixture()
    if damage=='row':g['numeric']['centers'][0].append(0.)
    elif damage=='radii':g['numeric']['radii'].pop()
    elif damage=='tuple':g['numeric']['centers']=tuple(g['numeric']['centers'])
    elif damage=='bool':g['numeric']['radii'][0]=True
    elif damage=='int':g['numeric']['radii'][0]=0
    elif damage=='nan':g['numeric']['centers'][0][0]=float('nan')
    elif damage=='inf':g['numeric']['radii'][0]=float('inf')
    elif damage=='negative_zero_cast':
        g=fixture('float32');g['numeric']['radii'][0]=.0100000000000001
    if damage not in ('nan','inf'):g['numeric_sha256']=digest(g['numeric'])
    with pytest.raises(ValueError):validate_geometry(g)


def test_wrapped_uint8_geometry_cannot_claim_native_success():
    # uint8 wall arithmetic can wrap 128+128 to zero. The revised evaluator
    # casts exact integer inputs to binary64 before validation/artifact capture.
    g=fixture('uint8',numeric={'centers':[[128.,128.]]*26,'radii':[128.]*26})
    with pytest.raises(ValueError,match='unsupported original numeric dtype'):validate_geometry(g)


@pytest.mark.parametrize('invalid',['overlap','outside','negative_radius'])
def test_finite_invalid_raw_geometry_kept_separate_from_penalty(invalid):
    g=fixture();n=deepcopy(g['numeric'])
    if invalid=='overlap':n['centers'][1]=list(n['centers'][0])
    elif invalid=='outside':n['centers'][0]=[-.1,.5]
    else:n['radii'][0]=-.01
    g=fixture(numeric=n,validity=False)
    assert validate_geometry(g)==g
    assert g['native_sum_radii']==g['native_score']==0.
    assert g['numeric']['radii']!=[0.]*26
    assert validate_binding(g,observation(g))==g
    g['native_validity']=True
    with pytest.raises(ValueError):validate_geometry(g)


def test_native_tolerance_sample_and_separate_exact_certificate():
    n={'centers':[[.1,.5]]+[[0.,0.]]*25,'radii':[.1000001]+[0.]*25}
    g=fixture(numeric=n);saved=deepcopy(g)
    assert validate_geometry(g)==g
    result=certify_artifact(g);cert=result['certificate']
    assert result['schema']==CERTIFICATE_SCHEMA and result['native_artifact_sha256']==digest(g)
    assert result['source_sha256']==g['source_sha256'] and result['invocation_id']==g['invocation_id']
    assert cert['status']=='certified' and cert['original_exact_feasible'] is False
    assert cert['input_numeric_sha256']==g['numeric_sha256']
    assert cert['radii'][0]<g['numeric']['radii'][0] and cert['centers']==g['numeric']['centers']
    assert Fraction(**cert['sum_exact'])<sum(map(Fraction.from_float,g['numeric']['radii']))
    assert g==saved


def test_invalid_native_geometry_certificate_rejection_is_preserved():
    g=fixture(numeric={'centers':[[-.1,.5]]+[[0.,0.]]*25,'radii':[.1]+[0.]*25},validity=False)
    r=certify_artifact(g)
    assert r['certificate']['status']=='rejected' and r['native_artifact_sha256']==digest(g)
    assert g['numeric']['centers'][0]==[-.1,.5]


@pytest.mark.parametrize('raw',['{}','[]','null','{"schema":1,"schema":2}',
    '{"schema":NaN}', '{"schema":1e309}', '['*80+']'*80, '\ud800'])
def test_malformed_canonical_transport_rejected(raw):
    with pytest.raises(ValueError):parse_artifact(raw)


def test_parser_rejects_noncanonical_string_nontext_and_size_without_rewriting():
    g=fixture()
    for raw in (json.dumps(g,indent=2),canonical(g)+'\n',' '*MAX_ARTIFACT_BYTES+'{}',g):
        with pytest.raises(ValueError):parse_artifact(raw)


@pytest.mark.parametrize('confirmation',[
    {'trials':True,'selected_trial':0,'selection':'first_failure_else_lowest_score'},
    {'trials':0,'selected_trial':0,'selection':'first_failure_else_lowest_score'},
    {'trials':3,'selected_trial':True,'selection':'first_failure_else_lowest_score'},
    {'trials':3,'selected_trial':3,'selection':'first_failure_else_lowest_score'},
    {'trials':3,'selected_trial':-1,'selection':'first_failure_else_lowest_score'},
    {'trials':3,'selected_trial':1,'selection':'last'}, {'trials':3},
])
def test_strict_selected_confirmation_metadata(confirmation):
    with pytest.raises(ValueError):validate_geometry(fixture(confirmation=confirmation))


def test_selected_confirm_geometry_binds_cases_not_earlier_raw_first_score():
    g=fixture(confirmation={'trials':3,'selected_trial':1,'selection':'first_failure_else_lowest_score'})
    o=observation(g,reached='confirm',confirmed=True)
    assert o['raw_first_score']!=g['native_score']
    assert validate_binding(g,o)==g
    failed=deepcopy(o);failed.update(ok=False,score=0.,confirmed=False)
    assert validate_binding(g,failed)==g
    failed['cases']['sum_radii']=0.
    with pytest.raises(ValueError,match='case differs'):validate_binding(g,failed)


@pytest.mark.parametrize('change',[{'reached':'smoke'},{'reached':'confirm'},{'confirmed':True},
    {'ok':1},{'score':float('nan')},{'score':.9}])
def test_binding_stage_status_or_score_tampering(change):
    g=fixture();o=observation(g);o.update(change)
    with pytest.raises(ValueError):validate_binding(g,o)


@pytest.mark.parametrize('field',['score','sum_radii','validity','combined_score','target_ratio'])
def test_huge_observation_integer_is_value_error(field):
    g=fixture();o=observation(g)
    if field=='score':o['score']=10**1000
    else:o['cases'][field]=10**1000
    with pytest.raises(ValueError):validate_binding(g,o)


def test_source_and_metric_binding_without_source_authentication_claim():
    g=fixture();o=observation(g)
    with pytest.raises(ValueError,match='source differs'):validate_binding(g,o,source_sha256='f'*64)
    m=metric();m['semantics']['denominator']=3.;m['sha256']=hashlib.sha256(json.dumps({k:v for k,v in m.items() if k!='sha256'},sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
    with pytest.raises(ValueError,match='metric semantics'):validate_binding(g,o,metric=m)
    assert validate_binding(g,o,metric=metric())==g


def test_view_is_bounded_metadata_not_full_evidence_and_projection_is_detached():
    g=fixture();v=view_geometry(g)
    assert v['schema']==VIEW_SCHEMA and v['artifact_sha256']==digest(g)
    assert 'numeric' not in v and v['arrays_omitted'] is True and v['certification']=='not_performed'
    assert validate_view(v)==v
    with pytest.raises(ValueError):validate_geometry(v)
    with pytest.raises(ValueError):certify_artifact(v)
    with pytest.raises(ValueError):view_geometry(v)
    v['confirmation']={'trials':1,'selected_trial':0,'selection':'first_failure_else_lowest_score'}
    assert g['confirmation'] is None
    v['numeric']=g['numeric']
    with pytest.raises(ValueError):validate_view(v)


@pytest.mark.parametrize('task_hash',[*HISTORICAL_CIRCLE_TASK_HASHES,'a'*64])
def test_historical_task_keeps_exact_registered_evaluator_binding(task_hash):
    from sciel.engine.repeat_support import NATIVE_GEOMETRY_TASK_EVALUATORS
    g=fixture();m=metric();m['task_hash']=task_hash
    m['sha256']=digest({k:v for k,v in m.items() if k!='sha256'})
    if NATIVE_GEOMETRY_TASK_EVALUATORS.get(task_hash) == g['evaluator_sha256']:
        # A public-prompt-only task change preserves the evaluator's exact bytes.
        assert validate_binding(g,observation(g),metric=m) == g
    else:
        with pytest.raises(ValueError,match='unregistered native'):
            validate_binding(g,observation(g),metric=m)


def test_registered_evaluator_hash_must_match_task_exactly():
    g=fixture();g['evaluator_sha256']='0'*64
    with pytest.raises(ValueError,match='unregistered native'):validate_binding(g,observation(g),metric=metric())
    m=metric();m['task_hash']=CANONICAL_TASK_HASH;m['sha256']=digest({k:v for k,v in m.items() if k!='sha256'})
    g['evaluator_sha256']=EVALUATOR_SHA
    assert validate_binding(g,observation(g),metric=m)==g
