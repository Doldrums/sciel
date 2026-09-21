"""Real controller + disk job service: no model provider or Docker required."""
from copy import deepcopy
import json

import pytest

from sciel.research_tools.config import validate_config, library_limits
from sciel.research_tools.controller import ResearchTools, parse_action
from sciel.research_tools.library import ToolLibrary
from sciel.research_tools.jobs import BackgroundJobs
from sciel.research_tools.job_worker import run_job
from test_research_tool_controller import IMAGE, Executor, LLM, IO, snapshot, create, prepare


def setup(tmp_path, **changes):
    config = validate_config({**dict(enabled=True, image_id=IMAGE, interval=1,
        max_scientific_steps=1,  # This fixture covers one legacy repair/job episode.
        jobs={'enabled':True}), **changes})
    executor = Executor(output={'finding':'measured', 'continuation_state':{'array':list(range(8000)), 'rng':'saved'}})
    library = ToolLibrary(tmp_path/'run', {'image_id':IMAGE}, library_limits(config), executor=executor)
    jobs = BackgroundJobs(tmp_path/'run', library, config['jobs'], launch=lambda:None)
    events, costs = [], []
    controller = ResearchTools(config,tmp_path/'run',task_identity={'task':'generic'},
        emit=lambda **kw:events.append(kw),charge=costs.append,library=library,jobs=jobs)
    return controller, executor, events, costs


def launch(tools, source=None, action=None):
    source = source or snapshot()
    episode = tools.execute(prepare(tools,source),source,
        LLM([action or create(execution='background',cpu_seconds=120,wall_seconds=240)]),io=IO())
    tools.commit(episode,child_program_id='p_later')
    return episode['submitted_jobs'][0]['job_id']


def finish(tools, ident):
    run_job(tools.lifecycle.run_dir,ident,tools.library)
    tools.collect_background(IO())


def test_background_is_nonblocking_collects_once_and_resumes_full_state(tmp_path):
    tools, executor, events, costs = setup(tmp_path)
    ident = launch(tools)
    assert executor.calls == []
    assert not tools.state()['lifecycle']['methods']
    assert not tools.state()['published']
    assert not [e for e in events if e['phase']=='child_link']
    tools.load_state(tools.state())
    tools.collect_background(IO())
    assert tools.state()['job_observations'][ident]['status']=='queued'
    finish(tools,ident)
    assert len(executor.calls)==1
    assert executor.calls[0][2]['limits']['cpu_seconds']==120
    state=tools.state(); tools.load_state(state)
    tools.collect_background(IO())
    assert tools.state()['uses']==state['uses']
    assert len(executor.calls)==1
    current=snapshot(iteration=2)
    current.research_cells_context=tools.context(current)
    plan=tools.prepare(current,signals={})
    llm=LLM([dict(action='continue',job_id=ident,cpu_seconds=360,wall_seconds=480,arguments_json='{}')])
    episode=tools.execute(plan,current,llm,io=IO())
    tools.commit(episode)
    assert 'measured' in llm.prompts[0]
    child=episode['submitted_jobs'][0]['job_id']
    assert child != ident
    request=tools.jobs.spec(child)['request']
    assert request['parent_job_id']==ident
    assert request['inputs']['continuation_state']['array']==list(range(8000))
    assert request['inputs']['parent_source']==executor.calls[0][1]['parent_source']
    finish(tools,child)
    tools.load_state(tools.state())
    assert len(executor.calls)==2 and costs==[.03,.03]


def test_collector_ignores_origin_island_selection_and_poll_replay_is_frozen(tmp_path):
    tools, executor, _, _=setup(tmp_path,interval=1000)
    ident=launch(tools)
    baseline=tools.state()
    io=IO();tools.collect_background(io)
    assert tools.state()['job_observations'][ident]['status']=='queued'
    run_job(tools.lifecycle.run_dir,ident,tools.library)
    tools.load_state(baseline);tools.collect_background(io)
    assert ident not in tools.state()['collected_jobs']
    tools.collect_background(IO())
    assert ident in tools.state()['collected_jobs']
    # Fresh result bypasses the 1000-opportunity cadence without a stagnation trigger.
    assert tools.prepare(snapshot(iteration=1),signals={}) is not None


def test_queued_cancel_has_honest_cost_and_no_publication(tmp_path):
    tools, executor, events, _=setup(tmp_path)
    ident=launch(tools)
    current=snapshot(iteration=1)
    episode=tools.execute(tools.prepare(current,signals={}),current,
        LLM([dict(action='cancel',job_id=ident)]),io=IO())
    tools.commit(episode)
    finish(tools,ident)
    assert executor.calls==[]
    done=tools.state()['collected_jobs'][ident]
    assert done['status']=='cancelled' and done['resources']['cpu_seconds']==0
    assert not tools.state()['published']
    tools.load_state(tools.state())


def test_foreground_model_budget_is_bound_on_resume(tmp_path):
    tools, executor, _, _=setup(tmp_path)
    source=snapshot()
    episode=tools.execute(prepare(tools,source),source,
        LLM([create(execution='foreground',cpu_seconds=75,wall_seconds=100)]),io=IO())
    tools.commit(episode)
    assert executor.calls[0][2]['limits']['cpu_seconds']==75
    assert tools.state()['jobs']=={}
    tools.load_state(tools.state())


@pytest.mark.parametrize('value',[True,-1,0,3601,float('inf')])
def test_invalid_model_budget_is_rejected_without_silent_clipping(tmp_path,value):
    tools,executor,_,_=setup(tmp_path)
    with pytest.raises(ValueError):
        parse_action(json.dumps(create(cpu_seconds=value)),tools.config)
    assert executor.calls==[]


def test_background_replay_only_releases_after_actual_completion(tmp_path):
    tools,executor,_,_=setup(tmp_path)
    ident=launch(tools);finish(tools,ident)
    source=snapshot(island=1)
    episode=tools.execute(prepare(tools,source),source,LLM([]),io=IO())
    assert episode['plan']['stage']=='independent_replay'
    tools.commit(episode)
    assert not tools.state()['published']
    replay=episode['submitted_jobs'][0]['job_id']
    finish(tools,replay)
    assert tools.state()['published']==[tools.jobs.spec(ident)['request']['tool_id']]
    tools.load_state(tools.state())


def test_full_queue_is_a_charged_known_refusal_not_an_unknown_submission(tmp_path):
    tools,executor,events,costs=setup(tmp_path,jobs={'enabled':True,'max_pending':1},max_requests=1)
    launch(tools)
    current=snapshot(island=1)
    episode=tools.execute(prepare(tools,current),current,
        LLM([create(execution='background')]),io=IO())
    tools.commit(episode)
    assert len(tools.state()['jobs'])==1 and len(costs)==2
    assert 'pending capacity' in episode['context']
    assert executor.calls==[]
    tools.load_state(tools.state())


def test_accepted_job_spec_failure_is_not_repaired_or_silently_dropped(tmp_path,monkeypatch):
    from sciel.campaign.budget import CampaignStopped
    tools,executor,_,costs=setup(tmp_path)
    def broken(ident):
        raise ValueError('accepted spec changed')
    monkeypatch.setattr(tools.jobs,'spec',broken)
    source=snapshot()
    llm=LLM([create(execution='background'),create(execution='background')])
    with pytest.raises(CampaignStopped,match='accepted background'):
        tools.execute(prepare(tools,source),source,llm,io=IO())
    assert len(llm.prompts)==1 and costs==[.03]
    assert len(list(tools.jobs.root.glob('*/spec.json')))==1
