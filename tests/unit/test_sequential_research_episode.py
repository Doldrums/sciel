"""Scientific stage continuation, exact state transfer and canonical submission.

Only saved fake receipts and model strings are used. No generated source runs.
"""
from copy import deepcopy
import json

import pytest

from research_target_fixtures import bind_snapshot
from sciel.campaign.budget import CampaignStopped
from sciel.engine.validation import source_sha256
from sciel.engine.research_cells_context import compact_context
from sciel.research_tools.controller import parse_action, response_format
from sciel.research_tools.research_steps import new_state
from test_research_tool_controller import IO, LLM, prepare, snapshot
from test_research_tool_result_flow import action, controller, result, collect
from test_tool_prior_results_prompt import prior_card


def decision(**changes):
    return {**action(), 'question':'Does a second local-search segment improve the saved incumbent?',
        'decision_if_supported':'Continue from the saved incumbent with the same seed.',
        'decision_if_refuted':'Test a different starting state using the same budget.',
        'experiment':'Compare one additional segment with the saved incumbent under equal conditions.',
        'next_decision':'Read the observed score difference before allocating another segment.',
        'conclusion':'','input_receipt_id':'','candidate_receipt_id':'', **changes}


def finish(**changes):
    return decision(**{'action':'finish','conclusion':'refuted',
        'next_decision':'No improvement under this budget; test a different initial state.', **changes})


def current_receipt(prompt):
    marker = 'PERSISTENT RESEARCH DECISION JSON:\n'
    state = json.JSONDecoder().raw_decode(prompt.split(marker,1)[1])[0]
    return state['available_input_receipt_ids'][-1]


def build(tmp_path,outputs,**changes):
    return controller(tmp_path,outputs,max_scientific_steps=3,**changes)


def lineage(current, *, ancestors=(), complete=True):
    current.research_search_evidence = {'selected_lineage':{
        'program_ids':[current.parent_id]+[row[0] for row in ancestors],
        'source_sha256s':[source_sha256(current.parent_code)]+[row[1] for row in ancestors],
        'complete':complete}}
    return current


def test_success_is_interpreted_then_revised_with_exact_data_and_negative_close(tmp_path):
    saved = {'incumbent':[0.0,-1,1.23456789012345], 'seed':149, 'rng':[1,2,3]}
    tools,backend,events,costs = build(tmp_path,[result({'continuation_state':saved,'gain':0}),result({'gain':0})])
    first = snapshot()
    def refine(prompt):
        assert '"gain":0' in prompt and 'continuation_state' in prompt
        return decision(code=action()['code']+'# second stage\n',input_receipt_id=current_receipt(prompt))
    llm = LLM([decision(),refine,finish()])
    episode = tools.execute(prepare(tools,first),first,llm,io=IO())
    assert len(backend.calls) == 2 and len(llm.prompts) == 3
    assert backend.calls[1]['inputs']['continuation_state'] == saved
    stage = backend.calls[1]['inputs']['research_stage']
    assert stage['observations'] == {'continuation_state':saved,'gain':0}
    assert 'research_stage' not in stage
    state = episode['research_state']
    assert state['status'] == 'closed' and state['conclusion'] == 'refuted'
    assert state['scientific_steps'] == 2 and state['known_llm_cost'] == pytest.approx(.09)
    assert state['resources'] == {'cpu_seconds':.02,'wall_seconds':.04,'unknown_cpu_invocations':0}
    assert costs == [.03]*3
    tools.commit(episode)
    saved_state = json.loads(json.dumps(tools.state()))
    tools.load_state(saved_state)
    assert tools.state() == saved_state
    assert any(event['phase']=='research_state' and event['research_state']['question']==decision()['question'] for event in events)


def test_documented_typed_access_works_before_and_after_exact_stage_handoff(tmp_path):
    tools, backend, _, _ = build(tmp_path, [result({'state':[1,2,3]}), result({'gain':0})])
    current = lineage(snapshot())
    llm = LLM([decision(), lambda prompt: decision(input_receipt_id=current_receipt(prompt)), finish()])
    tools.execute(prepare(tools,current),current,llm,io=IO())
    for prompt, call in zip(llm.prompts, backend.calls):
        assert 'Persistent decision and scheduling JSON are planner-only' in prompt
        assert 'CPU search, comparisons and candidate transforms are allowed with existing source/results' in prompt
        assert 'compute for new evidence' in prompt
        # Execute only our documented accessor, never model-authored code.
        accessor = prompt.split('Guarded Python input example:\n',1)[1].split('Only json.loads',1)[0]
        scope = {'payload':call['inputs']}
        exec(compile(accessor,'<typed-input-documentation>','exec'),{'__builtins__':{}},scope)
        assert scope['evidence'] == current.research_search_evidence
        assert isinstance(scope['evidence'],dict)
        assert scope['artifacts'] is None  # Fixture explicitly has no saved evaluator record.
    assert 'research_stage' not in backend.calls[0]['inputs']
    assert scope['prior'] == {'state':[1,2,3]}


def test_repairs_reset_after_success_and_keep_all_costs(tmp_path):
    tools,backend,_,costs = build(tmp_path,[result({'gain':1}),result({'gain':0})])
    current = snapshot()
    episode = tools.execute(prepare(tools,current),current,LLM([
        decision(),'not JSON',decision(code=action()['code']+'# changed\n'),finish()]),io=IO())
    assert len(backend.calls)==2 and len(costs)==4
    assert episode['research_state']['known_llm_cost']==pytest.approx(.12)
    assert [step['status'] for step in episode['research_state']['steps']] == ['success','proposal_invalid','success','closed']


def test_trial_cap_still_allows_reading_last_measurement_but_no_fourth_execution(tmp_path):
    tools,backend,_,costs = build(tmp_path,[result({'gain':1})]*3)
    current = snapshot()
    llm = LLM([decision()]*4)
    episode = tools.execute(prepare(tools,current),current,llm,io=IO())
    assert len(backend.calls)==3 and len(costs)==4
    assert episode['research_state']['status']=='step_limit'
    assert episode['research_state']['scientific_steps']==3


def test_candidate_source_is_submitted_exactly_once_and_method_trials_another_branch(tmp_path):
    code = 'def run_search():\n    return [1.23456789012345, 0]\n'
    tools,backend,_,costs = build(tmp_path,[result({'candidate_source':code,'continuation_state':{'seed':4}}),result({'gain':0})])
    current = lineage(snapshot(),ancestors=[('root','a'*64)])
    calls = []
    def submit(prompt):
        return decision(action='submit_candidate',candidate_receipt_id=current_receipt(prompt))
    def evaluate(source,provenance):
        calls.append((source,provenance))
        return {'status':'evaluated','score':.7,'timing_seconds':.025,'efficacy':'unknown'}
    llm = LLM([decision(),submit,finish()])
    episode = tools.execute(prepare(tools,current),current,llm,io=IO(),candidate_evaluator=evaluate)
    assert calls[0][0] == code and len(calls)==1
    assert calls[0][1]['discovery_resources']['cpu_seconds']==.01
    assert 'canonical_evaluation' in llm.prompts[-1] and '0.7' in llm.prompts[-1]
    assert episode['candidate_submission']['source']==code
    assert episode['research_state']['pending_method_trial']['source_program_id']==current.parent_id
    tools.commit(episode)
    tools.load_state(json.loads(json.dumps(tools.state())))
    another = snapshot(iteration=1)
    another.parent_id = 'p_other_branch'
    another.parent_code = 'def run_search():\n    return 3\n'
    bind_snapshot(another)
    lineage(another,ancestors=[('root','a'*64)])
    plan = prepare(tools,another)
    assert plan['cell_id']==episode['plan']['cell_id']
    transfer_llm = LLM([])
    followup = tools.execute(plan,another,transfer_llm,io=IO(),candidate_evaluator=evaluate)
    assert len(calls)==1 and len(backend.calls)==2
    assert backend.calls[-1]['inputs']['parent_source']==another.parent_code
    assert 'research_stage' not in backend.calls[-1]['inputs'] and 'continuation_state' not in backend.calls[-1]['inputs']
    assert followup['research_state']['pending_method_trial'] is None
    assert followup['research_state']['steps'][-1]['action']=='branch_trial'
    assert followup['research_state']['steps'][-1]['benefit']=='not_established'
    assert followup['research_state']['transfer_review_pending'] and not transfer_llm.prompts
    tools.commit(followup)
    tools.load_state(json.loads(json.dumps(tools.state())))


def test_candidate_evaluation_exception_is_not_repaired_or_retried(tmp_path):
    tools,_,_,costs = build(tmp_path,[result({'candidate_source':'def solve(): return 1\n'})])
    current = snapshot()
    def fail(*args):
        raise ValueError('receipt persistence failed after execution')
    with pytest.raises(CampaignStopped,match='could not be reconciled'):
        tools.execute(prepare(tools,current),current,LLM([decision(),lambda prompt:
            decision(action='submit_candidate',candidate_receipt_id=current_receipt(prompt))]),io=IO(),candidate_evaluator=fail)
    assert len(costs)==2


def test_foreign_or_unknown_stage_receipt_is_refused_before_execution(tmp_path):
    tools,backend,_,costs = build(tmp_path,[])
    current = snapshot()
    episode = tools.execute(prepare(tools,current),current,LLM([
        decision(input_receipt_id='f'*64),finish(conclusion='inconclusive')]),io=IO())
    assert not backend.calls and costs==[.03,.03]
    assert episode['research_state']['steps'][0]['status']=='proposal_invalid'


def test_successful_background_result_restores_question_and_exact_stage_access(tmp_path):
    tools,backend,_,costs = build(tmp_path,[result({'continuation_state':{'seed':3,'incumbent':[0,2]}}),result({'gain':0})],background=True)
    current = snapshot()
    first = tools.execute(prepare(tools,current),current,LLM([decision(execution='background',cpu_seconds=2,wall_seconds=5)]),io=IO())
    tools.commit(first)
    assert first['research_state']['status']=='waiting'
    job = first['submitted_jobs'][0]['job_id']
    collect(tools,job)
    tools.load_state(json.loads(json.dumps(tools.state())))
    next_snapshot = snapshot(iteration=1)
    def next_stage(prompt):
        return decision(execution='foreground',input_receipt_id=current_receipt(prompt))
    second = tools.execute(prepare(tools,next_snapshot),next_snapshot,LLM([next_stage,finish()]),io=IO())
    assert backend.calls[-1]['inputs']['continuation_state']=={'seed':3,'incumbent':[0,2]}
    assert second['research_state']['resources']['cpu_seconds']==.02
    assert second['research_state']['known_llm_cost']==pytest.approx(.09)
    tools.commit(second)
    tools.load_state(json.loads(json.dumps(tools.state())))


def test_success_then_background_submission_keeps_prior_committed_evidence(tmp_path):
    tools,backend,_,_ = build(tmp_path,[result({'gain':1}),result({'gain':2})],background=True)
    current = snapshot()
    episode = tools.execute(prepare(tools,current),current,LLM([
        decision(execution='foreground'),decision(execution='background',code=action()['code']+'# next\n')]),io=IO())
    tools.commit(episode)
    tools.load_state(json.loads(json.dumps(tools.state())))
    assert len(tools.lifecycle._state['methods'])==1
    collect(tools,episode['submitted_jobs'][0]['job_id'])
    tools.load_state(json.loads(json.dumps(tools.state())))
    assert tools.lifecycle._state['research_state'][episode['plan']['cell_id']]['resources']['cpu_seconds']==.02


def test_sequential_schema_and_parser_require_explicit_decision_branches(tmp_path):
    tools,_,_,_ = build(tmp_path,[])
    schema = response_format(sequential=True)['json_schema']['schema']
    assert {'question','decision_if_supported','decision_if_refuted','input_receipt_id','candidate_receipt_id'} <= set(schema['required'])
    with pytest.raises(ValueError,match='question'):
        parse_action(json.dumps(action()),tools.config)
    invalid = decision(decision_if_refuted='')
    with pytest.raises(ValueError,match='decision_if_refuted'):
        parse_action(json.dumps(invalid),tools.config)


def test_novelty_refusal_allows_new_saved_candidate_but_only_one_measured_result(tmp_path):
    codes = ['def solve(): return 1\n','def solve(): return 2\n']
    tools,backend,_,costs = build(tmp_path,[result({'candidate_source':source}) for source in codes])
    current = snapshot()
    evaluated = []
    def submit(prompt):
        return decision(action='submit_candidate',candidate_receipt_id=current_receipt(prompt))
    def evaluate(source,provenance):
        evaluated.append(source)
        return {'status':'rejected' if len(evaluated)==1 else 'evaluated','matched_id':'p_existing','score':.4}
    llm = LLM([decision(),submit,decision(code=action()['code']+'# revised candidate\n'),submit,finish()])
    episode = tools.execute(prepare(tools,current),current,llm,io=IO(),candidate_evaluator=evaluate)
    assert evaluated==codes and len(backend.calls)==2 and costs==[.03]*5
    assert [row['evaluation']['status'] for row in episode['candidate_submissions']]==['rejected','evaluated']
    assert episode['candidate_submission']['source']==codes[1]
    assert 'p_existing' in llm.prompts[2]


def test_only_three_pre_evaluation_novelty_submissions_are_allowed(tmp_path):
    tools,backend,_,costs = build(tmp_path,[result({'candidate_source':'def solve(): return 1\n'})])
    current = snapshot()
    submitted = []
    def submit(prompt):
        return decision(action='submit_candidate',candidate_receipt_id=current_receipt(prompt))
    def refuse(source,provenance):
        submitted.append(source)
        return {'status':'rejected','matched_id':'old','reason':'exact source repeat'}
    episode = tools.execute(prepare(tools,current),current,
        LLM([decision(),submit,submit,submit,submit,finish()]),io=IO(),candidate_evaluator=refuse)
    assert len(submitted)==3 and len(costs)==6 and len(backend.calls)==1
    assert episode['research_state']['pending_method_trial'] is None


def test_immutable_receipt_state_survives_resume_into_a_new_stage(tmp_path):
    tools,backend,_,_ = build(tmp_path,[result({'continuation_state':{'seed':31,'incumbent':[0]}}),result({'gain':0})])
    current = snapshot()
    first = tools.execute(prepare(tools,current),current,LLM([decision(),finish()]),io=IO())
    tools.commit(first)
    tools.load_state(json.loads(json.dumps(tools.state())))
    next_snapshot = snapshot(iteration=1)
    next_plan = prepare(tools,next_snapshot)
    assert next_plan['question']==decision()['question']
    second = tools.execute(next_plan,next_snapshot,LLM([
        lambda prompt:decision(input_receipt_id=current_receipt(prompt)),finish()]),io=IO())
    assert backend.calls[-1]['inputs']['continuation_state']=={'seed':31,'incumbent':[0]}
    assert second['research_state']['scientific_steps']==2


def test_resume_refuses_changed_recorded_resources(tmp_path):
    tools,_,_,_ = build(tmp_path,[result({'gain':0})])
    current = snapshot()
    episode = tools.execute(prepare(tools,current),current,LLM([decision(),finish()]),io=IO())
    tools.commit(episode)
    state = tools.state()
    state['lifecycle']['research_state'][episode['plan']['cell_id']]['steps'][0]['resources']['cpu_seconds']=999
    with pytest.raises(ValueError,match='differs from its saved invocation'):
        tools.load_state(state)


@pytest.mark.parametrize('budget',[8000,64000])
def test_sequential_prompt_fits_supported_budget_with_successful_measurement(tmp_path,budget):
    tools,_,_,_ = build(tmp_path,[result({'gain':0,'continuation_state':{'x':0}})],max_prompt_chars=budget)
    current = snapshot()
    llm = LLM([decision(),finish()])
    tools.execute(prepare(tools,current),current,llm,io=IO())
    assert all(len(prompt)<=budget for prompt in llm.prompts)
    assert ('"gain":0' in llm.prompts[-1]
            or '"path":"/output/observations/gain","value":0' in llm.prompts[-1])


def test_multi_stage_method_trial_rebuilds_data_on_independent_branch(tmp_path):
    origin_state = {'seed':12,'incumbent':[1,2]}
    branch_state = {'seed':12,'incumbent':[7,8]}
    tools,backend,_,_ = build(tmp_path,[result({'continuation_state':origin_state}),
        result({'candidate_source':'def solve(): return 3\n'}),
        result({'continuation_state':branch_state}),result({'candidate_source':'def solve(): return 15\n'})])
    current = lineage(snapshot(),ancestors=[('root','a'*64)])
    def second(prompt):
        return decision(input_receipt_id=current_receipt(prompt),code=action()['code']+'# polish\n')
    def submit(prompt):
        return decision(action='submit_candidate',candidate_receipt_id=current_receipt(prompt))
    first = tools.execute(prepare(tools,current),current,LLM([decision(),second,submit,finish()]),
        io=IO(),candidate_evaluator=lambda *_:{'status':'evaluated','score':3})
    tools.commit(first)
    sibling = snapshot(iteration=1)
    sibling.parent_id = 'sibling'
    sibling.parent_code = 'def run_search(): return 7\n'
    bind_snapshot(sibling); lineage(sibling,ancestors=[('root','a'*64)])
    episode = tools.execute(prepare(tools,sibling),sibling,LLM([finish()]),io=IO())
    assert len(backend.calls)==4
    assert 'continuation_state' not in backend.calls[2]['inputs']
    assert backend.calls[3]['inputs']['continuation_state']==branch_state
    assert backend.calls[3]['inputs']['research_stage']['prior_receipt_id']!=first['receipt_ids'][0]
    steps = [step for step in episode['research_state']['steps'] if step['action']=='branch_trial']
    assert [step['whole_method_completed'] for step in steps]==[False,True]
    assert steps[-1]['branch_independence']=='complete_lineages_no_ancestor_relation'
    tools.commit(episode); tools.load_state(json.loads(json.dumps(tools.state())))


@pytest.mark.parametrize('kind',['descendant','ancestor','missing_lineage'])
def test_method_trial_never_claims_transfer_to_ancestor_or_descendant(tmp_path,kind):
    tools,backend,_,_ = build(tmp_path,[result({'candidate_source':'def solve(): return 2\n'})])
    current = lineage(snapshot(),ancestors=[('root','a'*64)])
    first = tools.execute(prepare(tools,current),current,LLM([decision(),lambda prompt:
        decision(action='submit_candidate',candidate_receipt_id=current_receipt(prompt)),finish()]),
        io=IO(),candidate_evaluator=lambda *_:{'status':'evaluated','score':2})
    tools.commit(first)
    other = snapshot(iteration=1); other.parent_id='other'; other.parent_code='def run_search(): return 4\n'
    bind_snapshot(other)
    if kind=='descendant':
        lineage(other,ancestors=[(current.parent_id,source_sha256(current.parent_code)),('root','a'*64)])
    elif kind=='ancestor':
        other.parent_id='root'; bind_snapshot(other); lineage(other)
    pending = first['research_state']['pending_method_trial']
    assert not tools._independent_branch(other,pending)


def test_unfinished_question_resumes_without_new_stagnation_trigger(tmp_path):
    tools,_,_,_ = build(tmp_path,[result({'gain':1})]*3)
    current = snapshot()
    first = tools.execute(prepare(tools,current),current,LLM([decision()]*4),io=IO())
    tools.commit(first)
    next_snapshot = snapshot(iteration=1)
    next_plan = tools.prepare(next_snapshot,signals={'schema':'sciel-stagnation-signals/1',
        'population_stall':{'threshold_reached':False,'last_record_B_evals':0}})
    assert next_plan is not None and next_plan['cell_id']==first['plan']['cell_id']
    assert next_plan['question']==decision()['question']


def test_long_decision_text_is_explicit_excerpt_without_discarding_experiment(tmp_path):
    tools,backend,_,_ = build(tmp_path,[result({'gain':0})])
    current = snapshot()
    long_text = 'Compare the measured incumbent under fixed conditions. '*200
    episode = tools.execute(prepare(tools,current),current,LLM([
        decision(question=long_text),finish()]),io=IO())
    first = episode['research_state']['steps'][0]
    assert len(backend.calls)==1 and len(first['question'])==2000
    assert first['question'].endswith('[Excerpt; complete field is in the saved proposal.]')
    proposal = tools._read_artifact(first['proposal_id'],'proposal')
    assert json.loads(proposal['response']['text'])['question']==long_text


def test_next_opportunity_receives_prior_canonical_measurement(tmp_path):
    tools,_,_,_ = build(tmp_path,[result({'candidate_source':'def solve(): return 2\n'})])
    current = snapshot()
    first = tools.execute(prepare(tools,current),current,LLM([decision(),lambda prompt:
        decision(action='submit_candidate',candidate_receipt_id=current_receipt(prompt)),finish()]),io=IO(),
        candidate_evaluator=lambda *_:{'status':'evaluated','score':.7123456,'cases':{'native':2},'efficacy':'unknown'})
    tools.commit(first)
    next_snapshot = snapshot(iteration=1)
    llm = LLM([finish()])
    tools.execute(prepare(tools,next_snapshot),next_snapshot,llm,io=IO())
    assert '"score":0.7123456' in llm.prompts[0] and '"native":2' in llm.prompts[0]


def test_oversized_method_pipeline_defers_without_partial_transfer_claim(tmp_path):
    outputs = [result({'continuation_state':{'segment':i}}) for i in range(3)]
    outputs.append(result({'candidate_source':'def solve(): return 4\n'}))
    tools,backend,_,_ = build(tmp_path,outputs)
    current = lineage(snapshot(),ancestors=[('root','a'*64)])
    stage = lambda prompt:decision(input_receipt_id=current_receipt(prompt))
    first = tools.execute(prepare(tools,current),current,LLM([decision(),stage,stage,finish()]),io=IO())
    tools.commit(first)
    later = lineage(snapshot(iteration=1),ancestors=[('root','a'*64)])
    second = tools.execute(prepare(tools,later),later,LLM([stage,lambda prompt:
        decision(action='submit_candidate',candidate_receipt_id=current_receipt(prompt)),finish()]),io=IO(),
        candidate_evaluator=lambda *_:{'status':'evaluated','score':4})
    tools.commit(second)
    sibling = snapshot(iteration=2); sibling.parent_id='sibling'; sibling.parent_code='def run_search(): return 9\n'
    bind_snapshot(sibling); lineage(sibling,ancestors=[('root','a'*64)])
    episode = tools.execute(prepare(tools,sibling),sibling,LLM([finish()]),io=IO())
    assert len(backend.calls)==4 and episode['research_state']['pending_method_trial'] is not None
    trial = [step for step in episode['research_state']['steps'] if step['action']=='branch_trial'][-1]
    assert trial['status']=='deferred' and 'no partial transfer is claimed' in trial['error']


def test_branch_eligibility_accounts_for_upstream_stage_source(tmp_path):
    tools,_,_,_ = build(tmp_path,[result({'continuation_state':{'seed':4}}),result({'candidate_source':'def solve(): return 4\n'})])
    a = lineage(snapshot(),ancestors=[('root','a'*64)])
    first = tools.execute(prepare(tools,a),a,LLM([decision(),finish()]),io=IO()); tools.commit(first)
    b = snapshot(iteration=1); b.parent_id='branch_b'; b.parent_code='def run_search(): return 6\n'
    bind_snapshot(b); lineage(b,ancestors=[('root','a'*64)])
    second = tools.execute(prepare(tools,b),b,LLM([
        lambda prompt:decision(input_receipt_id=current_receipt(prompt)),
        lambda prompt:decision(action='submit_candidate',candidate_receipt_id=current_receipt(prompt)),finish()]),io=IO(),
        candidate_evaluator=lambda *_:{'status':'evaluated','score':4})
    tools.commit(second)
    descendant_a = snapshot(iteration=2); descendant_a.parent_id='child_a'; descendant_a.parent_code='def run_search(): return 12\n'
    bind_snapshot(descendant_a)
    lineage(descendant_a,ancestors=[(a.parent_id,source_sha256(a.parent_code)),('root','a'*64)])
    assert not tools._independent_branch(descendant_a,second['research_state']['pending_method_trial'])


@pytest.mark.parametrize('background',[False,True])
@pytest.mark.parametrize('saved_outputs',[False,True])
def test_saturated_small_prompt_keeps_latest_measured_observation(tmp_path,monkeypatch,background,saved_outputs):
    tools,_,_,_ = build(tmp_path,[],max_prompt_chars=8000,background=background)
    current = snapshot()
    synthetic_prior = prior_card()
    current.research_cells_context = compact_context([synthetic_prior])
    # This is a direct 8k formatter stress case with invented scheduling state,
    # not a provenance test. The sole synthetic prior card is an explicit
    # presentation boundary; every other card retains real authentication.
    authenticate = tools.lifecycle._delivery_measurement
    def render_prior(card, episode=None):
        return card if card == synthetic_prior else authenticate(card, episode)
    monkeypatch.setattr(tools.lifecycle, '_delivery_measurement', render_prior)
    current.research_search_evidence = {'schema':'sciel-search-evidence/1',
        'selected_runtime':{'observed_wall_seconds_per_trial':1.5,'per_run_wall_limit_seconds':28},
        'exact_recorded_output_repeats':17,
        'recent_admissions':[{'program_id':'program_'+str(index),'score':.9876543,
                             'source_sha256':str(index).zfill(64),'observed_wall_seconds_per_trial':1.5}
                             for index in range(12)],
        'selected_lineage':{'program_ids':[str(i) for i in range(256)],
                            'source_sha256s':[str(i).zfill(64) for i in range(256)],'complete':True}}
    plan = prepare(tools,current)
    if saved_outputs:
        from types import SimpleNamespace
        from sciel.engine.research_artifacts import capture_evaluation, saved_evaluation
        from sciel.eval.cascade import CascadeOutcome
        from sciel.eval.facade import EvalResult
        from sciel.methods.models import content_hash
        targets = current.research_targets
        engine = SimpleNamespace(run_dir=tmp_path, task=SimpleNamespace(task_hash=targets['task_hash']),
            cfg={}, log=SimpleNamespace(run_id=targets['run_id']))
        capture_evaluation(engine,current.parent_code,
            CascadeOutcome(EvalResult(True,.7,{'native':.7},artifacts={'saved_state':[0,1,2]}),
                           raw_first_score=.7,reached='full'),'saved-evaluation')
        saved_record = saved_evaluation(engine,current.parent_code,'saved-evaluation')
        for role in ('selected','population_best'):
            targets[role]['saved_evaluation'] = deepcopy(saved_record)
        targets['sha256'] = content_hash({key:value for key,value in targets.items() if key!='sha256'})
    state = new_state()
    for key in ('question','decision_if_supported','decision_if_refuted','experiment','next_decision'):
        state[key] = key+' '+('x'*(1999-len(key)))
    state['current_experiment'] = state['experiment']
    state['pending_method_trial'] = {'tool_id':'a'*64,'source_program_id':'p0','source_sha256':'b'*64,
                                    'submission_receipt_id':'c'*64}
    saved = tools._artifact('candidate_evaluation',{'evaluation':{'status':'evaluated','score':.7123456,
        'evaluation_id':'e0','cases':{'native':2.6},'timing_seconds':1.5}})
    state['steps'] = [{'action':'use','status':'success','receipt_id':format(i+1,'064x'),
                       'tool_id':'a'*64,'canonical_evaluation_id':saved} for i in range(32)]
    feedback = [{'tool_id':'a'*64,'receipt_id':'b'*64,'invocation_id':'c'*64,
        'result':{'status':'success','host_status':'ok'},'output':result({'latest_gain':0})}]
    prompt = tools._prompt(plan,current,[],feedback,31,research_state=state,scientific_step=2)
    assert len(prompt)<=8000
    assert ('"latest_gain":0' in prompt or '"path":"/output/observations/latest_gain","value":0' in prompt)
    assert 'older_receipts_omitted' in prompt and 'canonical_evaluations_omitted' in prompt
    assert '"observed_wall_seconds_per_trial":1.5' in prompt


def test_background_continuations_link_each_immediate_saved_segment(tmp_path):
    tools,backend,_,_ = build(tmp_path,[result({'continuation_state':{'segment':1,'seed':4}}),
        result({'continuation_state':{'segment':2,'seed':4}}),
        result({'continuation_state':{'segment':3,'seed':4},'candidate_source':'def solve(): return 3\n'})],background=True)
    current = lineage(snapshot(),ancestors=[('root','a'*64)])
    first = tools.execute(prepare(tools,current),current,LLM([decision(execution='background')]),io=IO())
    tools.commit(first); first_job=first['submitted_jobs'][0]['job_id']; collect(tools,first_job)
    first_receipt=tools._state['collected_jobs'][first_job]['invocation_artifact_id']
    second_snapshot=lineage(snapshot(iteration=1),ancestors=[('root','a'*64)])
    second=tools.execute(prepare(tools,second_snapshot),second_snapshot,LLM([
        decision(action='continue',job_id=first_job,arguments_json='{"segment_budget":2}')]),io=IO())
    tools.commit(second); second_job=second['submitted_jobs'][0]['job_id']; collect(tools,second_job)
    second_receipt=tools._state['collected_jobs'][second_job]['invocation_artifact_id']
    third_snapshot=lineage(snapshot(iteration=2),ancestors=[('root','a'*64)])
    third=tools.execute(prepare(tools,third_snapshot),third_snapshot,LLM([
        decision(action='continue',job_id=second_job)]),io=IO())
    tools.commit(third); third_job=third['submitted_jobs'][0]['job_id']; collect(tools,third_job)
    third_receipt=tools._state['collected_jobs'][third_job]['invocation_artifact_id']
    assert backend.calls[1]['inputs']['research_stage']['prior_receipt_id']==first_receipt
    assert backend.calls[2]['inputs']['research_stage']['prior_receipt_id']==second_receipt
    assert backend.calls[2]['inputs']['continuation_state']=={'segment':2,'seed':4}
    assert backend.calls[1]['inputs']['research_stage']['arguments']=={}
    assert backend.calls[1]['inputs']['arguments']=={'segment_budget':2}
    pending={'submission_receipt_id':third_receipt}
    chain=tools._method_pipeline(pending,first['plan']['cell_id'])
    assert [receipt for receipt,_ in chain]==[first_receipt,second_receipt,third_receipt]
    tools.load_state(json.loads(json.dumps(tools.state())))
