"""Compact exact measured facts; no provider, candidate or evaluator execution."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from sciel.engine.repeat_support import BODY_TASK_HASH, EVALUATOR_SHA
from sciel.eval.native_geometry import SCHEMA as GEOMETRY_SCHEMA
from sciel.hypothesis.measurement import metric_context
from sciel.knowledge_base import KnowledgeBase


SOURCE='1'*64
PARENT=.9988377503677311
CHILD=.998471926481528
HYPOTHESIS={'text':'Change the initializer','predicted_low':.0001,'predicted_high':.0005,
            'reasoning':'An uncertain fixture forecast','confidence':.6}


def metric(validity_case='validity', valid_value=1.0, *, native=False):
    return metric_context(SimpleNamespace(qualified_name='oe/circle_packing' if native else 'fixture/packing',
        task_hash=BODY_TASK_HASH if native else '2'*64,primary_metric='combined_score',research_metric={
            'kind':'validity_gated_ratio','score_units':'dimensionless','numerator_case':'sum_radii',
            'numerator_units':'unit_square_side_length','denominator':2.635,
            'validity_case':validity_case,'valid_value':valid_value,'invalid_score':0.,
            'invalid_numerator_value':0.,'invalid_numerator_semantics':'penalty_sentinel_not_physical_measurement'}))


def record(*, ident='r1', score=CHILD, cases=None, declaration=None, status='evaluated'):
    value={'id':ident,'branch':'p1','parent_score':PARENT,'status':status,
           'source_sha256':SOURCE,'proposed_sha256':SOURCE,'hypothesis':deepcopy(HYPOTHESIS),
           'score':score if status=='evaluated' else None,
           'actual_delta':score-PARENT if status=='evaluated' else None,
           'metric':metric() if declaration is None else declaration,'parent_cases':None,
           'evaluation':{'ok':status=='evaluated','score':score,'raw_first_score':score,'reached':'full',
                         'confirmed':False,'cases':{'validity':1.,'unrelated_case':987654321.} if cases is None else cases}}
    if status!='evaluated':value['failure']='fixture confirmation failed'
    return value


def facts(text):
    return json.JSONDecoder().raw_decode(text.split('; measured_facts=',1)[1])[0]


def feedback(value):
    kb=KnowledgeBase();kb.record(value)
    before=kb.state();text,=kb.failed_hypotheses('p1')
    assert kb.state()==before
    restored=KnowledgeBase();restored.load_state(deepcopy(before))
    assert restored.failed_hypotheses('p1')==[text] and restored.state()==before
    return text,kb


def test_evaluated_valid_result_has_exact_scores_delta_and_separate_validity():
    text,_=feedback(record())
    assert facts(text)=={'status':'evaluated','evaluator_ok':True,
        'task_validity':{'case':'validity','value':1.,'valid_value':1.},
        'parent_score':PARENT,'child_score':CHILD,'actual_delta':CHILD-PARENT}
    assert repr(CHILD-PARENT) in text and '-0.000365824' not in text
    assert 'Forecast refuted; measured regression' in text and SOURCE in text
    assert 'unrelated_case' not in text and '987654321' not in text


def test_evaluator_success_with_invalid_task_output_stays_explicitly_invalid():
    text,_=feedback(record(score=0.,cases={'validity':0.,'sum_radii':0.}))
    view=facts(text)
    assert view['evaluator_ok'] is True and view['task_validity']['value']==0.
    assert view['child_score']==0. and view['actual_delta']==-PARENT
    assert 'evaluator_ok is not task validity' in text


def test_missing_declared_case_is_unknown_not_zero_or_evaluator_success():
    text,_=feedback(record(cases={'other':1.}))
    assert facts(text)['task_validity']=={'case':'validity','value':None,'valid_value':1.}
    assert 'null=unknown' in text


def test_case_spelling_does_not_invent_undeclared_validity():
    text,_=feedback(record(declaration=metric_context(),cases={'validity':1.}))
    assert facts(text)['evaluator_ok'] is True
    assert facts(text)['task_validity']=={'case':None,'value':None,'valid_value':None}


def test_custom_declared_validity_name_and_value_are_preserved_without_boolean_inference():
    text,_=feedback(record(declaration=metric('admissible',7),cases={'admissible':7,'validity':0.}))
    assert facts(text)['task_validity']=={'case':'admissible','value':7,'valid_value':7}


@pytest.mark.parametrize('legacy',[True,False])
def test_missing_returned_evaluation_never_becomes_ok_true(legacy):
    value=record();value['evaluation']=None
    if legacy:
        for key in ('metric','parent_cases','evaluation'):del value[key]
    text,_=feedback(value)
    view=facts(text)
    assert view['evaluator_ok'] is None and view['task_validity']['value'] is None
    assert view['parent_score']==PARENT and view['child_score']==CHILD


def test_positive_eval_failed_diagnostic_is_not_presented_as_measured_quality():
    value=record(score=123.456789,cases={'validity':1.,'sum_radii':325.308641015},status='eval_failed')
    text,kb=feedback(value)
    assert 'Unmeasured (eval_failed)' in text and 'quality not measured' in text
    assert 'measured_facts=' not in text and '123.456789' not in text and '325.308641015' not in text
    assert kb.experiments[0]['evaluation']['score']==123.456789
    assert kb.experiments[0]['score'] is kb.experiments[0]['actual_delta'] is None


@pytest.mark.parametrize('score',[PARENT,PARENT+.0000000001])
def test_zero_and_positive_forecast_miss_keep_exact_observed_difference(score):
    text,_=feedback(record(score=score))
    assert facts(text)['actual_delta']==score-PARENT and facts(text)['child_score']==score
    assert ('Improved despite forecast miss' if score>PARENT else 'No measured improvement; forecast miss') in text


def test_full_native_geometry_remains_durable_but_only_declared_scalar_facts_are_shown():
    value=record(score=0.,declaration=metric(native=True),cases={'validity':1.,'sum_radii':0.,'combined_score':0.,'target_ratio':0.})
    numeric={'centers':[[0.,0.] for _ in range(26)],'radii':[0.]*26}
    numeric_sha=hashlib.sha256(json.dumps(numeric,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    geometry={'schema':GEOMETRY_SCHEMA,'source_sha256':SOURCE,'evaluator_sha256':EVALUATOR_SHA,
        'invocation_id':'a'*32,'stage':'full','confirmation':None,'numeric':numeric,'numeric_sha256':numeric_sha,
        'centers_dtype':'float64','radii_dtype':'float64','native_validity':True,'native_sum_radii':0.,'native_score':0.}
    value['evaluation']['native_geometry']=geometry
    text,kb=feedback(value)
    assert kb.experiments[0]['evaluation']['native_geometry']==geometry
    assert facts(text)['task_validity']['value']==1. and facts(text)['child_score']==0.
    assert all(word not in text for word in ('numeric','centers','radii','invocation_id','certified'))
