"""Inert synthetic helper witnesses only; no archived candidate/solver/API runs."""
import copy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from sciel.hypothesis import circle_initializer_witness as witness
from sciel.hypothesis import circle_restart_body as builder


def source(body):
    return builder.source_for(body)


def sources(*bodies):
    return {f's{i}':source(body) for i,body in enumerate(bodies)}


def pins(values):
    return {key:hashlib.sha256(value.encode()).hexdigest() for key,value in values.items()}


def run(*bodies, **options):
    values=sources(*bodies)
    return witness.witness_sources(values,expected_source_sha256=pins(values),**options)


def observations(receipt, index=0):
    return receipt['sources'][index]['observations']


def assert_terminal(receipt):
    assert receipt['status']=='observed', [(r['status'],r['error_code']) for r in receipt['sources']]
    assert all(r['complete'] for r in receipt['sources'])
    assert receipt['semantic_equivalence_proven'] is False
    assert receipt['candidate_rejection_authorized'] is False
    assert all(receipt[key]==0 for key in ('new_model_calls','new_solver_calls','new_native_evaluations'))
    unsigned={k:v for k,v in receipt.items() if k!='receipt_sha256'}
    assert receipt['receipt_sha256']==witness._hash(unsigned)


def test_fixed_cases_are_detached_deterministic_and_in_interface_domain():
    bundle=witness.make_fixture_bundle()
    assert len(bundle['cases'])==16
    assert bundle==witness.make_fixture_bundle()
    for case in bundle['cases']:
        arrays={key:witness._decode_array(value) for key,value in case['inputs'].items()}
        best,initial,noise,ordinary=(arrays[k] for k in witness.ALIASES)
        assert all(x.dtype==np.float64 and x.shape==(78,) for x in arrays.values())
        assert case['restart_index']%4==0 and case['out_of_runtime_domain'] is False
        assert initial[52:].sum()==pytest.approx(.25)
        assert np.all(abs(noise)<=.06)
        if case['restart_index']==0: assert np.array_equal(best,initial)
        else: assert not np.array_equal(best,initial) and best[52:].sum()>=initial[52:].sum()
        expected=best+noise;expected[:52]=np.clip(expected[:52],.001,.999);expected[52:]=np.maximum(expected[52:],.001)
        assert expected.tobytes()==ordinary.tobytes()
        centers=best[:52].reshape(26,2);radii=best[52:]
        assert min(radii)>=0
        for i in range(26):
            assert min(*centers[i],*(1-centers[i]))-radii[i]>=-1e-6
            for j in range(i):
                assert np.hypot(*(centers[i]-centers[j]))-radii[i]-radii[j]>=-1e-6
    bundle['cases'].clear()
    assert len(witness.make_fixture_bundle()['cases'])==16


@pytest.mark.parametrize('change',['source','fixture','plan','count','timeout','label'])
def test_bad_preparation_is_rejected_before_any_child(monkeypatch,change):
    values=sources('return None\n');expected=pins(values);bundle=witness.make_fixture_bundle();options={}
    if change=='source': expected['s0']='0'*64
    if change=='fixture': bundle['cases'][0]['restart_index']=4
    if change=='plan': options['expected_plan_sha256']='0'*64
    if change=='count': values={str(i):value for i,value in enumerate([values['s0']]*4)};expected=pins(values)
    if change=='timeout': options['timeout_seconds']=True
    if change=='label': values={'bad label':values['s0']};expected=pins(values)
    monkeypatch.setattr(witness,'_launch',lambda *a:pytest.fail('must reject before dispatch'))
    with pytest.raises(ValueError): witness.witness_sources(values,bundle,expected_source_sha256=expected,**options)


def test_no_solver_import_or_source_rewriting_in_function_extraction():
    text=source("x = 'a\u2028b'\nreturn best_params\n")
    helper=witness._function_text(text,'propose_restart')
    assert "x = 'a\u2028b'\n" in helper
    assert helper.endswith('    return best_params\n')
    assert 'N_CIRCLES =' not in helper and 'scipy' not in helper
    assert helper in text


@pytest.fixture(scope='module')
def baseline_receipt():
    return run('return None\n','return ordinary_params\n',
        'x = best_params + uniform_noise\nx[:52] = np.clip(x[:52], 0.001, 0.999)\nx[52:] = np.maximum(x[52:], 0.001)\nreturn x\n')


def test_three_effective_baselines_match_without_equivalence_or_rejection(baseline_receipt):
    result=baseline_receipt;assert_terminal(result)
    assert all(x['status']=='same_on_witnesses' for x in result['comparisons'])
    for index,row in enumerate(result['sources']):
        assert len(row['observations'])==16
        for obs in row['observations']:
            assert obs['handoff_counts']['proposal_attempts']==1
            assert obs['input_storage_detached'] is True
            assert not any(obs['input_mutated'].values())
            assert obs['effective_origin']==('saved_ordinary' if index==0 else 'return')
    assert observations(result,1)[0]['raw_return']['shares_input_storage']==[False,False,False,True]


def test_none_uses_saved_fallback_even_after_all_arguments_mutate():
    body='best_params.fill(0)\ninitial_params.fill(1)\nuniform_noise.fill(0)\nordinary_params.fill(0)\nreturn None\n'
    result=run(body,'return ordinary_params\n');assert_terminal(result)
    assert result['comparisons'][0]['status']=='same_on_witnesses'
    assert any(observations(result)[0]['input_mutated'].values())
    assert all(x['raw_return']['kind']=='none' for x in observations(result))


def test_index_zero_equal_values_have_independent_storage():
    result=run('best_params.fill(0)\nreturn initial_params\n','return initial_params\n');assert_terminal(result)
    assert result['comparisons'][0]['status']=='same_on_witnesses'
    assert observations(result)[0]['input_mutated']['best_params'] is True
    assert observations(result)[0]['input_mutated']['initial_params'] is False


def test_valid_list_integer_and_noncontiguous_view_follow_native_conversion():
    result=run('return [0 for i in range(78)]\n','return np.zeros(78, dtype=np.int64)\n',
               'x = np.zeros(156)\nreturn x[::2]\n');assert_terminal(result)
    assert all(x['status']=='same_on_witnesses' for x in result['comparisons'])
    assert observations(result)[0]['raw_return']['python_type']=='list'
    assert observations(result)[0]['raw_return']['array']['dtype'].endswith('i8')
    assert observations(result)[0]['effective_output']['dtype'].endswith('f8')


def test_tuple_return_and_unicode_crlf_body_are_not_rewritten_to_none():
    result=run("x = 'a\u2028b'\nreturn tuple(best_params)\n",
               "x = 'a\u2029b'\r\nreturn best_params\r\n",'return None\n');assert_terminal(result)
    assert result['comparisons'][0]['status']=='same_on_witnesses'
    assert result['comparisons'][1]['status']=='different_on_witness'
    assert observations(result)[0]['raw_return']['python_type']=='tuple'


def test_one_small_difference_is_not_epsilon_merged():
    result=run('x=ordinary_params.copy()\nx[0] += 2**-40\nreturn x\n','return None\n');assert_terminal(result)
    comparison=result['comparisons'][0]
    assert comparison['status']=='different_on_witness'
    assert all(x['equal_bits'] is False and x['first_differing_coordinate']==0 for x in comparison['cases'])


def test_same_on_finite_cases_does_not_cover_untested_index():
    result=run('if restart_index == 4000000:\n    return best_params\nreturn None\n','return None\n');assert_terminal(result)
    assert result['comparisons'][0]['status']=='same_on_witnesses'
    assert not result['semantic_equivalence_proven'] and not result['candidate_rejection_authorized']


def test_generic_exact_pair_facts_report_translation_without_an_oracle():
    # Deliberately tiny local translation, not the historical contact algorithm.
    result=run('x=best_params.copy()\nx[0] += 0.015625\nx[2] += 0.015625\nreturn x\n');assert_terminal(result)
    obs=observations(result)[4] # Positive dyadic incumbent.
    facts=obs['facts']['relative_to_best']
    pair=next(x for x in facts['pair_relations'] if (x['i'],x['j'])==(0,1))
    assert pair['squared_distance_delta']=={'numerator':0,'denominator':1}
    assert pair['radius_sum_delta']=={'numerator':0,'denominator':1}
    assert facts['changed_center_coordinates']==[0,2] and facts['changed_radius_indices']==[]
    assert any(x['squared_distance_delta']['numerator']!=0 for x in facts['pair_relations'])


def test_negative_radii_are_not_rejected_but_overlap_is_undefined():
    result=run('x=ordinary_params.copy()\nx[52:] = -1\nreturn x\n');assert_terminal(result)
    assert all(x['after_overlap'] is None for x in observations(result)[0]['facts']['relative_to_best']['pair_relations'])


@pytest.mark.parametrize('body',[ 'return np.zeros(77)\n', 'return np.full(78, np.nan)\n',
    "return np.zeros(78, dtype='complex128')\n", "return ['bad' for i in range(78)]\n",
    "return np.array([(1,)], dtype=[('a','f8')])\n", "return np.zeros(78, dtype='V0')\n",
    'x = 1 / 0\nreturn None\n'])
def test_invalid_returns_and_exceptions_are_failures_never_none_fallback(body):
    result=run(body,'return None\n')
    assert result['status']=='incomplete'
    assert all(x['status']=='failed' and x['effective_output'] is None and x['facts'] is None for x in observations(result))
    assert all(x['handoff_counts']['proposal_errors']==1 for x in observations(result))
    assert result['comparisons'][0]['status']=='incomplete'
    assert all(x['equal_bits'] is None for x in result['comparisons'][0]['cases'])


def test_timeout_has_missing_cases_and_no_equality_rescue():
    result=run('for i in range(1000000000000):\n    pass\nreturn None\n',timeout_seconds=.7)
    assert result['status']=='incomplete'
    row=result['sources'][0]
    assert row['process']['failure']=='timeout'
    assert all(x['status'] in ('unknown','not_dispatched') and x['effective_output'] is None for x in row['observations'])


def test_output_is_capped_during_read_not_after_child_finishes(tmp_path):
    # Controlled process fixture, not an accepted model helper or solver.
    executable=tmp_path/'noisy-python'
    executable.write_text('#!'+sys.executable+'\nimport os\nwhile True: os.write(1,b"x"*65536)\n')
    executable.chmod(0o700)
    result=witness._launch({},executable,2.)
    assert result['failure']=='output_limit'
    assert len(result['stdout'])==witness.MAX_STREAM_BYTES
    assert result['returncode']!=0


def test_tampered_success_bytes_cannot_produce_equality(baseline_receipt):
    import base64
    receipt=baseline_receipt;plan=receipt['plan'];row=receipt['sources'][0]
    events=[json.loads(x) for x in base64.b64decode(row['raw_stdout_base64']).splitlines()]
    observation=next(x for x in events if x['kind']=='observation')
    observation['effective_output']['bytes_sha256']='0'*64
    process={'stdout':b'\n'.join(witness._canonical(x).encode() for x in events)+b'\n',
             'stderr':b'','returncode':0,'failure':None,'elapsed_seconds':0.}
    changed=witness._consume(process,plan,'s0')
    assert changed['status']=='incomplete' and changed['error_code']=='invalid_child_receipt'
    assert all(x['effective_output'] is None for x in changed['observations'])


def test_wrong_interpreter_identity_fails_before_helper(monkeypatch):
    original=witness.identity
    def wrong():
        result=original();result['numpy_version']='different';return result
    monkeypatch.setattr(witness,'identity',wrong)
    result=run('return None\n')
    assert result['status']=='incomplete'
    assert all(x['status']=='not_dispatched' for x in observations(result))


def test_loaded_checker_drift_stops_before_dispatch(monkeypatch):
    monkeypatch.setattr(witness,'_LOADED_CHECKER','0'*64)
    with pytest.raises(ValueError,match='loaded_checker_drift'): run('return None\n')
