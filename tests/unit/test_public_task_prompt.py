"""Public contracts reach actual fake-leaf requests without reading hidden data."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from sciel.eval.registry import load_task
from sciel.eval.task_prompt import (MAX_TASK_PROMPT_CHARS, START, END,
    compose_task_prompt, split_task_prompt, validate_task_prompt)
from sciel.engine import checkpoint
from sciel.engine.idea import card_prompt, code_prompt
from sciel.engine.prompt.build import build as generation_prompt
from test_research_tool_controller import IO, LLM, prepare, snapshot
from test_research_canonical_citations import designed
from test_research_tool_result_flow import result
from test_sequential_research_episode import build, decision, finish, current_receipt

ROOT = Path(__file__).resolve().parents[2]
TASKS = ('circle_packing', 'heilbronn_triangle', 'function_minimization_oracle')


@pytest.mark.parametrize('value', ['', ' '*4, 12, 'x'*(MAX_TASK_PROMPT_CHARS+1), 'bad\x00text', END])
def test_public_contract_validation(value):
    with pytest.raises(ValueError): validate_task_prompt(value)


def test_historical_fallback_and_additional_instructions_do_not_replace_contract():
    legacy = SimpleNamespace(qualified_name='fixture/legacy', task_prompt=None)
    assert compose_task_prompt(legacy) == 'fixture/legacy'
    assert compose_task_prompt(legacy, 'existing system message') == 'existing system message'
    task = SimpleNamespace(qualified_name='fixture/public', task_prompt='Objective 雪🙂; return solve().')
    composed = compose_task_prompt(task, 'Prefer deterministic restarts.')
    public, extra = split_task_prompt(composed)
    assert public == START + task.task_prompt + END and extra == 'Prefer deterministic restarts.'


@pytest.mark.parametrize('name', TASKS)
@pytest.mark.parametrize('budget', [8000, 64000])
def test_complete_task_contract_survives_actual_planner_and_saved_followup(tmp_path,name,budget):
    task = load_task(str(ROOT/'tasks/oe'/name))
    assert task.task_prompt and len(task.task_prompt) <= MAX_TASK_PROMPT_CHARS
    tools, backend, _, _ = build(tmp_path, [result({'measurement': 7.0})],
        background=True, experimental_protocol=True, shared_observations=True,
        max_prompt_chars=budget)
    current = snapshot()
    current.task_prompt = compose_task_prompt(task, 'Optional extra run instruction. '*300)
    current.kb_lessons = ['Unrelated lesson '*3000]
    public, _ = split_task_prompt(current.task_prompt)
    def close(prompt):
        assert len(prompt) <= budget and public in prompt
        assert current_receipt(prompt)
        return designed(finish(conclusion='inconclusive'),kind='closure',budget_action='none')
    llm = LLM([designed(decision()), close])
    episode = tools.execute(prepare(tools,current),current,llm,io=IO())
    assert len(llm.prompts) == 2
    assert all(public in prompt and len(prompt)<=budget for prompt in llm.prompts)
    assert all('reallocate/escalate' in prompt and 'kind=closure' in prompt
               and 'budget_action=none' in prompt for prompt in llm.prompts)
    assert backend.calls[0]['inputs']['task_prompt'] == current.task_prompt
    tools.commit(episode); saved=deepcopy(tools.state()); tools.load_state(saved)
    assert tools.state() == saved


def test_real_engine_generation_bootstrap_idea_and_checkpoint_task_identity(tmp_path,monkeypatch):
    from test_parallel_engine import task_fixture, evaluator, Provider, config
    from sciel.engine.factory import build_engine
    tasks = tmp_path/'tasks'; name=task_fixture(tasks)
    path = tasks/name/'task.yaml'; metadata=yaml.safe_load(path.read_text())
    public = load_task(str(ROOT/'tasks/oe/function_minimization_oracle')).task_prompt
    metadata['task_prompt']=public; path.write_text(yaml.safe_dump(metadata))
    evaluator(monkeypatch); provider=Provider('run_search')
    cfg=config(name,tmp_path/'capacity',workers=1,num_islands=1,max_iterations=1,
        llm={'mock':True,'system_message':'Use reproducible algorithm seeds.'})
    engine=build_engine(cfg,str(tasks),str(tmp_path/'run'),llm=provider)
    engine.seed()
    before=len(provider.calls)
    engine._paradigm_child((tasks/name/'initial_program.py').read_text(),
                           'A different generic initialization',0,'portfolio')
    assert len(provider.calls)>before and public in provider.calls[-1]
    engine.run()
    assert provider.calls and all(public in prompt for prompt in provider.calls)
    current=snapshot(); current.task_prompt=engine.task_prompt
    card={'family':'search','strategy':'restart','mechanism':'budgeted proposals'}
    for prompt in (generation_prompt('full_rewrite',current), card_prompt(current),code_prompt(current,card)):
        assert public in prompt and 'Use reproducible algorithm seeds.' in prompt
    restored=build_engine(cfg,str(tasks),str(tmp_path/'run'),llm=provider)
    checkpoint.load(restored,str(tmp_path/'run/checkpoint.json'))
    assert restored.task_prompt == engine.task_prompt
    restored.close()
    original_hash=load_task(str(tasks/name)).task_hash
    metadata['task_prompt']=public+' Public metadata changed.'; path.write_text(yaml.safe_dump(metadata))
    assert load_task(str(tasks/name)).task_hash != original_hash
    changed=build_engine(cfg,str(tasks),str(tmp_path/'run'),llm=provider)
    with pytest.raises(checkpoint.IdentityMismatch):
        checkpoint.load(changed,str(tmp_path/'run/checkpoint.json'))
