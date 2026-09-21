"""Source identity through tool creation, replay, adoption, continuation and resume."""
from copy import deepcopy
from types import SimpleNamespace
import json

import pytest

from sciel.engine.research_targets import freeze_targets, failed_provenance, input_binding
from sciel.methods.models import content_hash
from test_research_tool_controller import snapshot, prepare, IO, LLM
from test_research_tool_result_flow import controller, execute_create, collect, action
from test_research_tool_result_contract import result


def targeted(island=0, iteration=0, *, failed=False, champion='p_champion'):
    current = snapshot(island, iteration)
    parent = SimpleNamespace(id=current.parent_id, code=current.parent_code, score=current.parent_score)
    best = SimpleNamespace(id=champion, code='def solve(): return "' + champion + '"\n', score=.95)
    engine = SimpleNamespace(log=SimpleNamespace(run_id='fixture-run'), task=SimpleNamespace(task_hash='fixture'),
        iteration_no=iteration, B_evals=7, db=SimpleNamespace(pop_best_id=best.id, get=lambda _:best),
        observed_changes=None)
    failure = None
    if failed:
        current.parent_code = 'def solve(): raise RuntimeError("failed child")\n'
        current.parent_score = 0.0  # Legacy generator sentinel must not become a tool measurement.
        attempt = SimpleNamespace(child_code=current.parent_code, meta={'evaluation_id':'evaluation-7'},
            reached='smoke', confirmed=False, artifacts={'error':'failed child'})
        failure = failed_provenance(attempt, iteration=iteration, B_evals=7)
    current.research_targets = freeze_targets(engine, parent, current.parent_code, failed=failed, failure=failure)
    return current


def test_failed_source_and_champion_are_available_without_fabricating_measurement(tmp_path):
    tools, backend, events, _ = controller(tmp_path, [result({'finding':'auxiliary'})])
    current = targeted(failed=True)
    llm = LLM([action()])
    episode = execute_create(tools, current, llm=llm)
    inputs = backend.calls[0]['inputs']
    assert inputs['parent_program_id'] is inputs['parent_score'] is None
    assert inputs['ancestry_parent_id'] == current.parent_id
    assert inputs['research_targets']['selected']['failure']['artifacts'] == {'error':'failed child'}
    assert json.dumps(inputs['research_targets']['population_best']['source']) in llm.prompts[0]
    assert 'selection_score' in llm.prompts[0] and 'not proof of actual tool use' in llm.prompts[0]
    card = episode['cards'][0]
    assert card['source']['candidate_id'] is None and card['source']['target_kind']=='failed_candidate'
    assert card['source']['population_best_id'] == 'p_champion'
    artifact = tools._read_artifact(card['source']['receipt_id'], 'invocation')
    assert artifact['input_program_id'] is None
    assert artifact['target_binding'] == input_binding(inputs)
    assert next(e for e in events if e['phase']=='invocation_result')['target_binding'] == artifact['target_binding']
    tools.commit(episode); saved=tools.state(); tools.load_state(saved); assert tools.state()==saved


@pytest.mark.parametrize('background', [False, True])
def test_replay_keeps_original_bundle_adoption_gets_recipient_current_reference(tmp_path, background):
    tools, backend, _, _ = controller(tmp_path, [result({'finding':1})]*3, background=background)
    current = targeted(failed=True)
    local = execute_create(tools, current, background=background); tools.commit(local)
    if background: collect(tools, local['submitted_jobs'][0]['job_id'])
    original = deepcopy(backend.calls[0]['inputs'])
    recipient = targeted(1, 12, champion='p_new_champion')
    plan=prepare(tools, recipient); assert plan['stage']=='independent_replay'
    replay=tools.execute(plan,recipient,LLM([]),io=IO());tools.commit(replay)
    if background: collect(tools,replay['submitted_jobs'][0]['job_id'])
    assert backend.calls[1]['inputs']==original
    plan=prepare(tools, recipient)
    if background:
        assert plan['stage']=='research'  # Review a newly collected job before another assignment.
        tools.commit(tools.execute(plan,recipient,LLM([{'action':'skip'}]),io=IO()))
        plan=prepare(tools,recipient)
    assert plan['stage']=='adopt_trial'
    adoption=tools.execute(plan,recipient,LLM([]),io=IO());tools.commit(adoption)
    if background: collect(tools,adoption['submitted_jobs'][0]['job_id'])
    admitted = backend.calls[2]['inputs']
    assert admitted['parent_program_id']==recipient.parent_id
    assert admitted['research_targets']==recipient.research_targets
    assert admitted['research_targets']['population_best']['program_id']=='p_new_champion'
    assert original['research_targets']['population_best']['program_id']=='p_champion'
    saved=tools.state();tools.load_state(saved);assert tools.state()==saved


def test_background_continue_keeps_original_target_even_after_champion_changes(tmp_path):
    output=result({'continuation_state':{'incumbent':[1,2,3]},'finding':'search state'})
    tools, backend, _, _ = controller(tmp_path,[output,output],background=True)
    current=targeted(failed=True)
    local=execute_create(tools,current,background=True);tools.commit(local)
    first=local['submitted_jobs'][0]['job_id'];collect(tools,first)
    original=deepcopy(backend.calls[0]['inputs'])
    next_step=targeted(0,20,champion='p_latest')
    plan=prepare(tools,next_step);assert plan['stage']=='research'
    follow=tools.execute(plan,next_step,LLM([dict(action='continue',job_id=first,
        arguments_json='{}',cpu_seconds=2,wall_seconds=5)]),io=IO());tools.commit(follow)
    second=follow['submitted_jobs'][0]['job_id'];collect(tools,second)
    continued=backend.calls[1]['inputs']
    assert continued['research_targets']==original['research_targets']
    assert continued['parent_program_id'] is None
    assert continued['continuation_state']=={'incumbent':[1,2,3]}
    saved=tools.state();tools.load_state(saved);assert tools.state()==saved


def test_frozen_plan_rejects_rehashed_target_substitution_before_provider_or_execution(tmp_path):
    tools, backend, _, _ = controller(tmp_path,[])
    current=targeted();plan=prepare(tools,current)
    current.research_targets['population_best']['source']+='\n# substituted'
    from sciel.engine.research_targets import source_sha256
    current.research_targets['population_best']['source_sha256']=source_sha256(
        current.research_targets['population_best']['source'])
    current.research_targets['sha256']=content_hash({k:v for k,v in current.research_targets.items() if k!='sha256'})
    llm=LLM([])
    with pytest.raises(ValueError,match='frozen plan'):
        tools.execute(plan,current,llm,io=IO())
    assert backend.calls==llm.prompts==[]


def test_small_planner_budget_keeps_both_roles_and_explicit_source_omission(tmp_path):
    tools, _, _, _=controller(tmp_path,[result({'finding':1})],max_prompt_chars=8000)
    current=targeted()
    targets=current.research_targets
    best=targets['population_best'];best['source']+='\n# 中🙂'*20000
    from sciel.engine.research_targets import source_sha256
    best['source_sha256']=source_sha256(best['source'])
    targets['sha256']=content_hash({k:v for k,v in targets.items() if k!='sha256'})
    current.task_prompt='TASK DETAIL '*10000
    llm=LLM([action()]);execute_create(tools,current,llm=llm)
    prompt=llm.prompts[0]
    assert len(prompt)<=8000
    view = json.JSONDecoder().raw_decode(prompt.split('FROZEN_INPUT_VIEW_JSON:\n', 1)[1])[0]
    references = {row['role']: row for row in view['references']}
    assert set(references) == {'selected', 'population_best'}
    assert references['population_best']['source_sha256'] == best['source_sha256']
    excerpt = json.JSONDecoder().raw_decode(prompt.split('POPULATION_BEST SOURCE: ', 1)[1])[0]
    assert excerpt['source_excerpt_omitted'] is True
    assert excerpt['full_source_path'] == "payload['research_targets']['population_best']['source']"
