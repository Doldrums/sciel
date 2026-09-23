"""Real renderer and worker budgets with inert provider/evaluator leaves."""
from copy import deepcopy
import hashlib
import json
import subprocess
import sys
from types import ModuleType
from types import SimpleNamespace

import pytest

from sciel.config import apply_defaults
from sciel.engine.generation_budget import (contract, select, verify_selection,
    GenerationBudgetExceeded, snapshot_value, snapshot_from)
from sciel.engine.prompt.build import build
from sciel.engine.research import ResearchController
from sciel.engine.types import NoChild, Result
from sciel.engine.worker import run_worker
from sciel.engine.db import Database
from sciel.engine.llm.client import LLMResponse
from sciel.hypothesis import hypothesis_instruction
from sciel.hypothesis.rendering import serialize_context
from upstream_native_fixture import provider
from test_research_retry_feedback import CHILD, PARENT, text
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult

IDENTITY = dict(schema='fixture-tokenizer/1', revision='fixed', chat_template_sha256='a'*64)
BLOCKS = dict.fromkeys(('study_text', 'study_review_text', 'component_text',
                       'component_review_text', 'restart_text', 'prediction_text', 'shown_rules'), '')

def count(prompt):
    return len(prompt.encode('utf-8')) + 7


def configured(cap=100000):
    return apply_defaults({'llm': {'backend': 'local_openai'}, 'op_fixed': 'full_rewrite', 'idea_island': False,
        'research': {'enabled': True, 'max_aux_calls': 0, 'reflection_interval': 0,
                     'generation_context_tokens': cap}})['research']


def setup(provider, monkeypatch, cap=100000):
    llm, sdk, _ = provider
    monkeypatch.setattr(llm, '_prompt_counter', lambda: SimpleNamespace(identity=IDENTITY, count=count))
    sdk.chat.completions.create.side_effect = lambda **kw: SimpleNamespace(model=kw['model'],
        choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(content=text(), refusal=None))],
        usage=SimpleNamespace(prompt_tokens=50, completion_tokens=20))
    events=[]
    research=ResearchController(configured(cap),llm,
        SimpleNamespace(research=lambda **e:events.append(deepcopy(e))),lambda _:None)
    db=Database(1);parent=db.add(PARENT,.5,{},0,'seed')
    from sciel.engine.types import Snapshot
    snap=Snapshot(0,'full_rewrite',llm.base_alias,0,parent.id,PARENT,.5,
        task_prompt='Exact task 🌍',directive='Exact required instruction',
        required_entrypoints=['run_search'],research_context=research.context(db,parent,0,1))
    return llm,sdk,research,snap,events


def evaluate(code):
    return CascadeOutcome(EvalResult(True,.6,{}),.6,'full')


def test_whole_source_selection_preserves_original_and_exact_visibility(provider,monkeypatch):
    llm,sdk,c,snap,events=setup(provider,monkeypatch)
    snap.inspirations=[dict(program_id='p2',code='OPTIONAL="'+'多'*2000+'"\n',score=.4),
                       dict(program_id='p3',code='OTHER="'+'x'*2000+'"\n',score=.3)]
    before=deepcopy(snap)
    # Measure the complete historical renderer, then shave a single token.
    _,wide=c.begin(snap,0,'full_rewrite',build('full_rewrite',snap),generation_budget=True)[:2]
    exact=count(wide)+llm.max_tokens
    # A new controller has no pending work; no provider was called by preparation.
    c=ResearchController(configured(exact-1),llm,SimpleNamespace(research=lambda **e:events.append(deepcopy(e))),lambda _:None)
    result=run_worker('full_rewrite',snap,llm,evaluate,dup_cap=1,research=c)
    assert isinstance(result,Result) and result.ok and sdk.chat.completions.create.call_count==1
    attempt=[e for e in events if e['phase']=='attempt'][-1]
    proof=attempt['generation_token_budget'];actual=result.prompt
    assert proof['steps']==[dict(kind='inspiration',index=1,
        removed_sha256=hashlib.sha256(json.dumps(before.inspirations[1],sort_keys=True,separators=(',',':'),ensure_ascii=True).encode()).hexdigest())]
    assert count(actual)+llm.max_tokens<=exact-1
    selected,context=verify_selection(proof,prompt=actual,contract=c._generation_token_budget,count=count)
    assert selected.parent_code==snap.parent_code and selected.task_prompt==snap.task_prompt
    assert snap.parent_code==before.parent_code and snap.inspirations==before.inspirations
    assert snap.research_context==before.research_context  # freeze_inference adds route fields only.
    refs=result.meta['validation_trace']['attempts'][0]['adaptation_visibility']['delivered_refs']
    assert [r['program_id'] for r in refs if r['role']=='inspiration']==['p2']
    assert 'OTHER=' not in actual and 'OPTIONAL=' in actual


def test_irreducible_core_is_known_skip_and_next_worker_can_generate(provider,monkeypatch):
    llm,sdk,c,snap,events=setup(provider,monkeypatch,6000)
    snap.task_prompt='REQUIRED-TASK-'*1000
    result=run_worker('full_rewrite',snap,llm,evaluate,dup_cap=1,research=c)
    assert isinstance(result,NoChild) and result.reason=='generation_context_limit'
    assert result.cost==0 and result.meta['generation_dispatch']=='not_dispatched'
    assert result.meta['current_call_cost_unknown'] is False
    assert sdk.chat.completions.create.call_count==0 and not c.pending and c.attempts==0
    assert c.stopped_reason is None
    assert events[-1]['phase']=='generation_budget_rejected'
    snap.task_prompt='small exact task'
    following=run_worker('full_rewrite',snap,llm,evaluate,dup_cap=1,research=c)
    assert isinstance(following,Result) and sdk.chat.completions.create.call_count==1


def test_optional_memory_is_removed_atomically_with_ids_and_bound_changes(provider,monkeypatch):
    llm,_,c,snap,_=setup(provider,monkeypatch,6000)
    lesson='Optional large memory '*1000
    snap.kb_lessons=[lesson];snap.research_context.update(injected_ids=['l1'],lesson_texts=[lesson])
    block='\nBOUND OPTIONAL HISTORY '+ 'h'*7000
    snap.directive+=block
    snap.adaptation_context=dict(observed_changes=dict(chars=len(block),text_sha256=hashlib.sha256(block.encode()).hexdigest()))
    original=deepcopy(snap)
    _,prompt,selected=c.begin(snap,0,'full_rewrite',build('full_rewrite',snap),generation_budget=True)
    assert selected.kb_lessons==[] and selected.research_context['injected_ids']==[]
    assert selected.directive=='Exact required instruction' and selected.task_prompt==original.task_prompt
    assert snap==original and count(prompt)+llm.max_tokens<=6000


@pytest.mark.parametrize('bad',[None,True,0,-1,8192.0,'8192'])
def test_config_rejects_bad_caps(bad):
    with pytest.raises(ValueError,match='generation_context_tokens'):
        configured(bad)


def test_unsupported_specialized_path_fails_configuration():
    with pytest.raises(ValueError,match='ordinary fixed'):
        apply_defaults({'llm':{'backend':'local_openai'},'op_fixed':'idea',
            'research':{'enabled':True,'generation_context_tokens':8192}})


def test_contract_checkpoint_and_selection_tamper_rejected(provider,monkeypatch):
    llm,_,c,snap,events=setup(provider,monkeypatch)
    ident,prompt,_=c.begin(snap,0,'full_rewrite',build('full_rewrite',snap),generation_budget=True)
    proof=events[-1]['generation_token_budget']
    for mutate in (lambda p:p['source']['snapshot'].__setitem__('parent_code','wrong'),
                   lambda p:p['measurements'][0].__setitem__('prompt_tokens',1),
                   lambda p:p.__setitem__('selected_sha256','0'*64)):
        bad=deepcopy(proof);mutate(bad)
        with pytest.raises(ValueError):verify_selection(bad,prompt=prompt,contract=c._generation_token_budget,count=count)
    c.pending.clear();state=c.state();state['generation_token_budget']['context_tokens']-=1
    with pytest.raises(ValueError,match='generation token budget'):
        c.load_state(state)
    llm.max_tokens+=1
    with pytest.raises(ValueError,match='identity changed'):
        c.begin(snap,1,'full_rewrite',build('full_rewrite',snap),generation_budget=True)






def test_invalid_counter_is_not_misreported_as_a_safe_budget_skip(provider,monkeypatch):
    llm,sdk,c,snap,_=setup(provider,monkeypatch)
    monkeypatch.setattr(llm,'count_prompt_tokens',lambda _:False)
    with pytest.raises(ValueError,match='invalid exact generation token count'):
        run_worker('full_rewrite',snap,llm,evaluate,dup_cap=1,research=c)
    assert sdk.chat.completions.create.call_count==0 and not c.pending


def test_selected_retry_keeps_known_earlier_cost_and_exact_previous_response(provider,monkeypatch):
    llm,sdk,c,snap,events=setup(provider,monkeypatch)
    returned=[text(PARENT),text()]
    sdk.chat.completions.create.side_effect=lambda **kw:SimpleNamespace(model=kw['model'],
        choices=[SimpleNamespace(finish_reason='stop',message=SimpleNamespace(content=returned.pop(0),refusal=None))],
        usage=SimpleNamespace(prompt_tokens=50,completion_tokens=20))
    snap.code_hashes={__import__('sciel.engine.db',fromlist=['normalized_hash']).normalized_hash(PARENT)}
    result=run_worker('full_rewrite',snap,llm,evaluate,dup_cap=2,research=c)
    assert isinstance(result,Result) and sdk.chat.completions.create.call_count==2
    assert c.attempts==c.outcomes==2 and not c.pending
    attempts=[e for e in events if e['phase']=='attempt']
    assert all('generation_token_budget' in row for row in attempts)
    assert attempts[1]['generation_token_budget']['source']['retry_feedback']['total_rejections']==1
