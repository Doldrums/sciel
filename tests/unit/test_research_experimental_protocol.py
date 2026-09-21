"""Protocol validation and trusted-vs-reported limits, using fixed JSON fixtures."""
from copy import deepcopy
import json
import pytest

from sciel.research_tools.experiment_protocol import parse_protocol,validate_dispatch,scope_key,result_assessment
from sciel.research_tools.controller import parse_action,response_format
from test_sequential_research_episode import build,decision


def protocol(**changes):
    return dict(kind='diagnostic',question_key='local_budget_headroom',prior_receipt_ids=[],new_uncertainty='',
        budget_action='initial',budget_reason='Initial bounded measurement',arms=[],matched_seeds=[],cpu_seconds_per_arm=None,
        conditions_json='{}',saved_state_plan='',**changes)


def comparison():
    value=protocol();value.update(kind='controlled_comparison',arms=[{'id':'a','method':'existing method'},{'id':'b','method':'new method'}],
        matched_seeds=[11,17],cpu_seconds_per_arm=2,conditions_json='{"inputs":"same"}',
        saved_state_plan='Save best result, complete continuation state, seed and conditions for every arm.')
    return value


def test_structured_protocol_opt_in_and_declared_equal_budget(tmp_path):
    tools,_,_,_=build(tmp_path,[],experimental_protocol=True,shared_observations=True)
    act=decision(experimental_protocol=comparison())
    parsed=parse_action(__import__('json').dumps(act),tools.config)
    audit=validate_dispatch(parsed,{},'f'*64,default_limits={'cpu_seconds':20})
    assert audit['declared_cpu_seconds']==8 and audit['adherence']=='declared_not_verified'
    assert 'experimental_protocol' in response_format(sequential=True,experimental_protocol=True)['json_schema']['schema']['required']
    assert 'experimental_protocol' not in response_format(sequential=True)['json_schema']['schema']['properties']
    with pytest.raises(ValueError,match='experimental_protocol'):
        parse_action(__import__('json').dumps(decision()),tools.config)


@pytest.mark.parametrize('field,value', [('arms',[{'id':'a','method':'one'}]),('matched_seeds',[1]),('cpu_seconds_per_arm',0),('conditions_json','{}'),('saved_state_plan','')])
def test_incomplete_controlled_comparison_rejected(field,value):
    data=comparison();data[field]=value
    with pytest.raises(ValueError):parse_protocol(data)


def test_budget_cannot_claim_more_equal_cpu_than_actual_request():
    action=decision(experimental_protocol=comparison())
    with pytest.raises(ValueError,match='smaller than arms'):
        validate_dispatch(action,{},'f'*64,default_limits={'cpu_seconds':7})


def test_renamed_tool_repeat_requires_new_uncertainty_and_receipt():
    action=decision(name='A completely different tool name',experimental_protocol=protocol())
    entry=dict(receipt_id='a'*64,scope_keys=[scope_key('f'*64,action['question'])],conclusion=None)
    plan={'shared_observations':{'check_index':[entry]}}
    with pytest.raises(ValueError,match='known-scope repeated'):
        validate_dispatch(action,plan,'f'*64,default_limits={'cpu_seconds':20})
    action['experimental_protocol'].update(prior_receipt_ids=['a'*64],new_uncertainty='Does the same cap help a different saved starting state?')
    audit=validate_dispatch(action,plan,'f'*64,default_limits={'cpu_seconds':20})
    assert audit['repeated_receipt_ids']==['a'*64]
    assert validate_dispatch(action,plan,'b'*64,default_limits={'cpu_seconds':20})['repeated_receipt_ids']==[]


def test_refuted_scope_must_change_allocation_not_just_rename_question():
    action=decision(question='Paraphrased surface wording',experimental_protocol=protocol())
    action['experimental_protocol'].update(prior_receipt_ids=['a'*64],new_uncertainty='Try another initial state',budget_action='maintain')
    entry=dict(receipt_id='a'*64,scope_keys=[scope_key('f'*64,'local_budget_headroom')],conclusion='refuted')
    plan={'shared_observations':{'check_index':[entry]}}
    with pytest.raises(ValueError,match='cannot retain its allocation'):
        validate_dispatch(action,plan,'f'*64,default_limits={'cpu_seconds':20})
    action['experimental_protocol']['budget_action']='reallocate'
    assert validate_dispatch(action,plan,'f'*64,default_limits={'cpu_seconds':20})['negative_allocation_change_declared'] is True


def test_reported_arm_resources_remain_unverified_even_with_complete_shape():
    design=comparison()
    rows=[dict(arm_id=arm['id'],seed=seed,cpu_seconds=1.5,best_result={'score':7},saved_state={'seed':seed},conditions={'inputs':'same'})
          for arm in design['arms'] for seed in design['matched_seeds']]
    assessment=result_assessment({'comparison_results':rows},design)
    assert assessment['report_shape']=='complete' and assessment['adherence']=='tool_reported_unverified'
    rows[0]['cpu_seconds']=9
    bad=result_assessment({'comparison_results':rows},design)
    assert bad['report_shape']=='invalid' and 'above declared cap' in ' '.join(bad['problems'])


@pytest.mark.parametrize('declared,reported,complete', [
    ({'enabled':True}, {'enabled':1}, False),
    ({'enabled':0}, {'enabled':False}, False),
    ({'nested':[{'enabled':False}, [1, True]]},
     {'nested':[{'enabled':0}, [1, True]]}, False),
    ({'nested':[{'enabled':True}, [1, False]]},
     {'nested':[{'enabled':True}, [True, False]]}, False),
    ({'nested':[{'count':1}, [0, 2.5]]},
     {'nested':[{'count':1.0}, [-0.0, 2.5]]}, True),
    ({'a':None,'b':[True,'1',1]}, {'b':[True,'1',1.0],'a':None}, True),
    ({'a':[1,2]}, {'a':[2,1]}, False),
    ({'a':[1]}, {'a':(1,)}, False),
    ({'a':1}, {'a':'1'}, False),
    ({'a':1}, {'a':1,'extra':None}, False),
    ({'a':1,'extra':None}, {'a':1}, False),
    ({'a':9007199254740993}, {'a':float(9007199254740993)}, False),
    ({'a':9007199254740992}, {'a':float(9007199254740992)}, True),
    ({'a':1}, {'a':float('inf')}, False),
    ({'a':1}, {'a':float('nan')}, False),
])
def test_comparison_conditions_use_recursive_json_types(declared,reported,complete):
    design=comparison();design['conditions_json']=json.dumps(declared,allow_nan=False)
    design=parse_protocol(design)
    rows=[dict(arm_id=arm['id'],seed=seed,cpu_seconds=1,
        best_result={'score':7},saved_state={'seed':seed},conditions=deepcopy(declared))
        for arm in design['arms'] for seed in design['matched_seeds']]
    rows[-1]['conditions']=reported
    before=json.dumps([design,rows],sort_keys=True)
    assessment=result_assessment({'comparison_results':rows},design)
    assert assessment['report_shape']==('complete' if complete else 'invalid')
    assert assessment['problems']==([] if complete else ['reported conditions differ'])
    assert assessment['adherence']=='tool_reported_unverified'
    assert json.dumps([design,rows],sort_keys=True)==before


def test_background_comparison_keeps_bound_design_and_unverified_report(tmp_path):
    from test_research_tool_controller import IO,LLM,prepare,snapshot
    from test_research_tool_result_flow import result,collect
    tools,backend,events,costs=build(tmp_path,[result({'comparison_results':[
        dict(arm_id=arm,seed=seed,cpu_seconds=.1,best_result={'value':1},saved_state={'seed':seed},conditions={'inputs':'same'})
        for arm in ('a','b') for seed in (11,17)]})],background=True,experimental_protocol=True,shared_observations=True)
    action=decision(experimental_protocol=comparison(),execution='background',cpu_seconds=8,wall_seconds=20)
    current=snapshot();episode=tools.execute(prepare(tools,current),current,LLM([action]),io=IO())
    tools.commit(episode)
    job=episode['submitted_jobs'][0]['job_id']
    assert tools.jobs.spec(job)['request']['owner']['experimental_protocol']==comparison()
    collect(tools,job)  # Fake executor only, never execute its source.
    done=tools.state()['collected_jobs'][job]
    assert done['comparison_report']['report_shape']=='complete'
    assert done['comparison_report']['adherence']=='tool_reported_unverified'
    assert done['resources']['cpu_seconds']==.01
    shared=tools.lifecycle.shared_observations(tools.library,1,tools.task_identity)
    assert shared['entries'][0]['comparison_report']==done['comparison_report']
    assert shared['entries'][0]['question_key']=='local_budget_headroom'
    tools.load_state(deepcopy(tools.state()))
    assert len(backend.calls)==1 and costs==[.03]


def test_same_opportunity_repeat_is_charged_rejected_then_new_uncertainty_accepted(tmp_path):
    from test_research_tool_controller import IO,LLM,prepare,snapshot
    from test_research_tool_result_flow import result
    from test_sequential_research_episode import current_receipt,finish
    tools,backend,events,costs=build(tmp_path,[result({'gain':0}),result({'gain':1})],experimental_protocol=True,shared_observations=True)
    first=decision(experimental_protocol=protocol())
    renamed=decision(name='renamed duplicate',experimental_protocol=protocol())
    def new_check(prompt):
        fixed=protocol();fixed.update(prior_receipt_ids=[current_receipt(prompt)],new_uncertainty='Does a different saved starting state help?',budget_action='reallocate')
        return decision(experimental_protocol=fixed,arguments_json='{"starting_state":2}')
    final=finish(experimental_protocol={**protocol(), 'kind':'closure', 'budget_action':'none'},conclusion='inconclusive')
    current=snapshot();episode=tools.execute(prepare(tools,current),current,LLM([first,renamed,new_check,final]),io=IO())
    assert len(backend.calls)==2 and costs==[.03]*4
    rejected=[step for step in episode['research_state']['steps'] if step['status']=='proposal_invalid']
    assert len(rejected)==1 and 'known-scope repeated' in rejected[0]['error']
    tools.commit(episode);tools.load_state(deepcopy(tools.state()))
    shared=tools.lifecycle.shared_observations(tools.library,1,tools.task_identity)
    assert shared['total_checks']==2 and all(row['question_key']=='local_budget_headroom' for row in shared['entries'])


@pytest.mark.parametrize('encoded',['{"x":1e999}','{"x":1,"x":2}','{"nested":{"x":1,"x":2}}'])
def test_conditions_are_finite_and_unambiguous(encoded):
    data=comparison();data['conditions_json']=encoded
    with pytest.raises(ValueError,match='finite object'):
        parse_protocol(data)


def test_failed_same_scope_can_be_repaired_without_fabricating_new_uncertainty():
    action=decision(experimental_protocol=protocol())
    plan={'shared_observations':{'check_index':[dict(receipt_id='a'*64,
        scope_keys=[scope_key('f'*64,action['question'])],status='tool_error',conclusion='inconclusive')]}}
    assert validate_dispatch(action,plan,'f'*64,default_limits={'cpu_seconds':20})['repeated_receipt_ids']==[]


def test_repeat_repair_error_contains_real_receipt_when_prompt_omitted_it():
    action=decision(experimental_protocol=protocol())
    plan={'shared_observations':{'check_index':[dict(receipt_id='a'*64,
        scope_keys=[scope_key('f'*64,action['question'])],status='success',conclusion='refuted')]}}
    with pytest.raises(ValueError) as error:
        validate_dispatch(action,plan,'f'*64,default_limits={'cpu_seconds':20})
    assert 'a'*64 in str(error.value) and 'refuted' in str(error.value)


@pytest.mark.parametrize('field',['experimental_protocol','comparison_report_assessment'])
def test_resume_rejects_changed_design_or_assessment(tmp_path,field):
    from test_research_tool_controller import IO,LLM,prepare,snapshot
    from test_research_tool_result_flow import result
    tools,_,_,_=build(tmp_path,[result({'comparison_results':[]})],experimental_protocol=True,shared_observations=True,yield_after_success=True)
    current=snapshot();episode=tools.execute(prepare(tools,current),current,LLM([decision(experimental_protocol=comparison())]),io=IO());tools.commit(episode)
    state=deepcopy(tools.state())
    step=next(iter(state['lifecycle']['research_state'].values()))['steps'][0]
    if field=='experimental_protocol':step[field]['question_key']='invented_question'
    else:step[field]['adherence']='host_verified'
    with pytest.raises(ValueError,match='differs from its saved'):
        tools.load_state(state)


def test_cancelled_undispatched_comparison_keeps_provenance_without_shared_measurement(tmp_path):
    from test_research_tool_controller import IO,LLM,prepare,snapshot
    from test_research_tool_result_flow import collect
    tools,backend,_,costs=build(tmp_path,[],background=True,experimental_protocol=True,shared_observations=True)
    current=snapshot()
    episode=tools.execute(prepare(tools,current),current,LLM([decision(experimental_protocol=comparison(),
        execution='background',cpu_seconds=8,wall_seconds=20)]),io=IO());tools.commit(episode)
    job=episode['submitted_jobs'][0]['job_id']
    tools.jobs.cancel(job);collect(tools,job)
    tools.load_state(deepcopy(tools.state()))
    shared=tools.lifecycle.shared_observations(tools.library,1,tools.task_identity)
    assert shared['total_checks']==0 and shared['check_index']==[]
    assert backend.calls==[] and costs==[.03]


def test_refuted_question_b_does_not_relabel_earlier_question_a_same_cell(tmp_path):
    from test_research_tool_controller import IO,LLM,prepare,snapshot
    from test_research_tool_result_flow import result
    from test_sequential_research_episode import finish
    tools,_,_,_=build(tmp_path,[result({'gain':1}),result({'gain':0})],experimental_protocol=True,shared_observations=True)
    a=protocol();a['question_key']='question_a'
    b=protocol();b['question_key']='question_b'
    current=snapshot();episode=tools.execute(prepare(tools,current),current,LLM([
        decision(question='Does method A help?',experimental_protocol=a),
        decision(question='Does method B help?',experimental_protocol=b,code=decision()['code']+'# method B\n'),
        finish(question='Does method B help?',experimental_protocol={**b, 'kind':'closure', 'budget_action':'none'},reason='B shows no gain; A had a positive measured gain.')]),io=IO())
    tools.commit(episode);tools.load_state(deepcopy(tools.state()))
    shared=tools.lifecycle.shared_observations(tools.library,1,tools.task_identity)
    arow=next(row for row in shared['entries'] if row['question_key']=='question_a')
    brow=next(row for row in shared['entries'] if row['question_key']=='question_b')
    indexed={row['receipt_id']:row for row in shared['check_index']}
    assert arow['decision']['conclusion']=='refuted'  # Preserved later interpretation as context.
    assert arow['decision_scope_matches_check'] is False
    assert indexed[arow['receipt_id']]['conclusion'] is None
    assert brow['decision_scope_matches_check'] is True
    assert indexed[brow['receipt_id']]['conclusion']=='refuted'
    repeat=deepcopy(a);repeat.update(prior_receipt_ids=[arow['receipt_id']],new_uncertainty='Does the same benefit hold for another input?',budget_action='maintain')
    audit=validate_dispatch(decision(question='Does method A help?',experimental_protocol=repeat),
        {'shared_observations':shared},arow['target_source_sha256'],default_limits={'cpu_seconds':20})
    assert audit['negative_allocation_change_declared'] is False
