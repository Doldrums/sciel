"""Lesson-card ablation only: fake responses/scores, no evaluator or candidate execution."""
import copy
import json
import re
import sys
from types import SimpleNamespace

import pytest

from sciel.hypothesis.rendering import decode_evidence
from sciel.engine.types import Snapshot
from sciel.engine.prompt.build import build
from sciel.knowledge_base.presentation import expand_reflection_payload

from sciel.config import apply_defaults
from sciel.engine import checkpoint
from sciel.engine.llm.client import LLMResponse
from sciel.engine.loop import Engine
from sciel.engine.research import ResearchController, digest
from sciel.engine.types import Program
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.eval.registry import load_task
from sciel.knowledge_base import KnowledgeBase


def begin_fixture(controller, context):
    snap = Snapshot(context['island'], 'diff', 'fake', context['iteration'] - 1,
        context['parent_id'], context['parent_code'], context['parent_score'],
        research_context=context, kb_lessons=context.get('lesson_texts', []))
    return controller.begin(snap, 0, 'diff', build('diff', snap))


RULE = 'UNIQUE_LESSON_ADVICE: the fixture showed a positive measured change.'
REPLACEMENT = 'UNIQUE_REPLACEMENT_ADVICE: retain the same observed source.'


def controller(kb=None, **settings):
    cfg = apply_defaults({'research': {'enabled': True, **settings}})['research']
    events = []
    instance = ResearchController(cfg, SimpleNamespace(models=['fake'], max_retries=1),
        SimpleNamespace(research=lambda **kw: events.append(kw)), lambda cost: None)
    if kb is not None:
        instance.knowledge = kb
        instance.attempts = instance.outcomes = len(kb.experiments)
    parent = Program('p1', 'def run_search():\n    return 0.5\n', .5, {}, 0, 'seed')
    return instance, SimpleNamespace(_all={parent.id: parent}), parent, events


def populated():
    kb = KnowledgeBase()
    for number, score in [(1,.6),(2,.4)]:
        code = f'def run_search():\n    return {score}\n'
        kb.record({'id':f'e{number}', 'branch':'p1', 'parent_id':'p1', 'parent_score':.5,
            'status':'evaluated', 'score':score, 'actual_delta':score-.5,
            'source_sha256':digest(code), 'proposed_sha256':digest(code), 'injected_ids':[],
            'code_fragment':code if score>.5 else '',
            'hypothesis':{'text':'fixture', 'predicted_low':0., 'predicted_high':.2,
                          'reasoning':'fixture', 'confidence':.5}})
    for number in range(4):
        kb.add_lessons([{'rule':f'{RULE} {number}', 'polarity':'positive',
            'scope':'branch', 'evidence_ids':['e1']}], allowed_evidence_ids=['e1'])
    return kb


@pytest.mark.parametrize('value', [0,1,None,'false',[],{}])
def test_injection_flag_is_a_strict_boolean(value):
    with pytest.raises(ValueError, match='inject_lessons must be boolean'):
        apply_defaults({'research':{'inject_lessons':value}})


def test_default_does_not_enable_research_and_old_research_config_injects():
    assert 'research' not in apply_defaults({})
    cfg = apply_defaults({'research':{'enabled':True}})
    assert cfg['research']['inject_lessons'] is True
    assert apply_defaults(cfg) == cfg
    assert apply_defaults({'research':{}})['research']['enabled'] is False


def test_missing_direct_controller_key_matches_explicit_default_exactly():
    a,db,parent,_=controller(populated())
    b,_,_,_=controller(populated(), inject_lessons=True)
    del a.config['inject_lessons']
    assert a.context(db,parent,0,3) == b.context(db,parent,0,3)
    assert len(a.context(db,parent,0,3)['injected_ids']) == 3


@pytest.mark.parametrize('retrieve_k',[0,1,3])
def test_off_omits_all_ranked_cards_without_mutation_or_other_channel_loss(retrieve_k):
    off,db,parent,_=controller(populated(), inject_lessons=False, retrieve_k=retrieve_k)
    on,_,_,_=controller(populated(), inject_lessons=True, retrieve_k=retrieve_k)
    before=off.knowledge.state()
    absent=off.context(db,parent,0,3); visible=on.context(db,parent,0,3)
    assert absent['lessons'] == absent['lesson_texts'] == absent['injected_ids'] == []
    presentation=absent['lesson_presentation']
    assert presentation['presented']==[] and presentation['rendered_chars']==0
    assert presentation['ranked_count']==presentation['unshown_count']==4
    ranked=off.knowledge.retrieve(branch='p1',context=parent.card+'\n'+parent.code,limit=off.knowledge.max_lessons)
    assert presentation['omitted']==[{'id':l['id'],'reason':'ablation','chars':len(off.lesson_text(l))} for l in ranked]
    assert all(set(x)=={'id','reason','chars'} for x in presentation['omitted'])
    assert absent['failed_hypotheses'] == visible['failed_hypotheses'] and absent['failed_hypotheses']
    assert absent['fragments'] == visible['fragments'] and absent['fragments']
    assert off.knowledge.state()==before
    assert RULE not in json.dumps(absent)


@pytest.mark.parametrize('inject',[False,True])
def test_only_actually_shown_cards_receive_exposure(inject):
    instance,db,parent,events=controller(populated(), inject_lessons=inject)
    ctx=instance.context(db,parent,0,3)
    ident,prompt=begin_fixture(instance, ctx)
    instance.resolve(ident,status='generation_stopped',failure='inert fixture')
    assert (RULE in prompt) is inject
    assert events[0]['evidence']['injected_ids']==ctx['injected_ids']
    assert events[0]['lesson_presentation']==ctx['lesson_presentation']
    for lesson in instance.knowledge.lessons:
        assert lesson['exposure_ids']==([ident] if lesson['id'] in ctx['injected_ids'] else [])
        assert lesson['support_ids']==lesson['refute_ids']==[]  # unmeasured failure is exposure only


def test_controller_checkpoint_cannot_switch_injection_mode():
    off,_,_,_=controller(inject_lessons=False)
    on,_,_,_=controller(inject_lessons=True)
    with pytest.raises(ValueError,match='checkpoint contract/configuration mismatch'):
        on.load_state(off.state())
    restored,_,_,_=controller(inject_lessons=False)
    restored.load_state(json.loads(json.dumps(off.state())))
    assert restored.state()==off.state()


def decode(prompt,marker):
    value = decode_evidence(json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1].lstrip())[0])
    return expand_reflection_payload(value) if value.get("schema") == "sciel-reflection-presentation/2" else value


class FakeLLM:
    models=['fake']; weights=[1.0]; max_retries=1
    def __init__(self): self.calls=[]
    def choose_model(self,rng): return 'fake'
    def generate(self,prompt,model,temperature=None):
        if 'Review this one experiment.' in prompt:
            from _mechanism_fixtures import unknown_mechanism
            kind='conclusion'; record=decode(prompt,'EVIDENCE_JSON:')
            text='CONCLUSION_JSON: '+json.dumps({'verdict':record['forecast_verdict'],
                'mechanism_review': unknown_mechanism(record),
                'explanation':'Synthetic observed delta.', 'lessons':[{'rule':RULE,
                'polarity':'positive','scope':'branch','evidence_ids':[record['id']]}]})
        elif 'Review existing experimental evidence and lessons.' in prompt:
            kind='reflection'; data=decode(prompt,'EVIDENCE_JSON:')
            old=data['lessons'][0]; support=old.get('source_support',old)
            text='REFLECTION_JSON: '+json.dumps({'lessons':[{'rule':REPLACEMENT+' '+old['id'],
                'polarity':support['polarity'],'scope':old['scope'],'evidence_ids':support['evidence_ids']}],
                'retire_ids':[old['id']]})
        else:
            kind='mutation'; context=decode(prompt,'RESEARCH_CONTEXT_JSON:')
            score=context['parent_score']+.1
            hypothesis={'hypothesis':'Synthetic numeric change.', 'predicted_low':0.,'predicted_high':.2,
                'reasoning':'Fixture only.', 'confidence':.5}
            text='HYPOTHESIS_JSON: '+json.dumps(hypothesis)+f'\n```python\ndef run_search():\n    return {score!r}\n```'
        self.calls.append({'kind':kind,'prompt':prompt,'text':text,'cost':.01})
        return LLMResponse(text=text,model=model,cost=.01,prompt_tokens=len(prompt)//4,completion_tokens=len(text)//4)


@pytest.fixture
def fake_task(tmp_path):
    folder=tmp_path/'task'; folder.mkdir()
    (folder/'task.yaml').write_text('suite: test\nname: injection\norigin: math\nprimary_metric: score\nstochastic: false\nsmoke_timeout: 1\nfull_timeout: 1\n')
    (folder/'initial_program.py').write_text('def run_search():\n    return 0.1\n')
    (folder/'evaluator.py').write_text('raise AssertionError("this evaluator must never be imported")\n')
    return load_task(str(folder))


def fake_engine(task,directory,inject):
    cfg={'task':'test/injection','budget':4,'max_iterations':4,'dup_cap':1,'random_seed':21,
        'num_islands':1,'schedule_policy':'round_robin','milestone_fracs':[],
        'op_fixed':'full_rewrite','policy':{'enabled':False,'plateau_shift':False},'llm':{'mock':True},
        'portfolio_seed':False,'idea_island':False,'dynamic_islands':False,'crossover':False,
        'embed_grid':False,'embed_niche':False,'three_head':False,'confirm_records':False,
        'research':{'enabled':True,'inject_lessons':inject,'reflection_interval':2,'max_aux_calls':8}}
    llm=FakeLLM(); instance=Engine(cfg,task,str(directory),llm,python=sys.executable)
    def fixed_score(code):
        # Parse inert source text; do not import, exec or evaluate it.
        score=float(re.search(r'return\s+([0-9.]+)',code).group(1))
        return CascadeOutcome(EvalResult(ok=True,score=score,cases={'value':score},timing=.01),score,'full',False)
    instance._evaluate=fixed_score
    return instance,llm


def events(directory):
    return [json.loads(x) for x in (directory/'sciel_events.jsonl').read_text().splitlines()]


def test_fake_engine_off_keeps_paid_auxiliary_schedule_and_consolidation(fake_task,tmp_path):
    results=[]
    for inject in (False,True):
        folder=tmp_path/str(inject); instance,llm=fake_engine(fake_task,folder,inject)
        result=instance.run(); results.append((result,llm))
        assert result['B_evals']==4 and instance.research.reflections==2
        assert instance.research.reflection_changes==2 and instance.research.knowledge.lessons
        mutations=[c['prompt'] for c in llm.calls if c['kind']=='mutation']
        assert any(RULE in p or REPLACEMENT in p for p in mutations) is inject
        assert any('RESEARCH_LESSON_JSON:' in p for p in mutations) is inject
        attempts=[e for e in events(folder) if e.get('phase')=='attempt']
        if not inject:
            assert all(e['evidence']['injected_ids']==[] for e in attempts)
            assert any(e['lesson_presentation']['omitted'] for e in attempts)
            assert all(not l['exposure_ids'] and not l['inherited_exposures'] for l in instance.research.knowledge.lessons)
        saved=json.loads((folder/'checkpoint.json').read_text())
        assert saved['research']['config']['inject_lessons'] is inject
        assert saved['identity']['research']['config']['inject_lessons'] is inject
    assert [c['kind'] for c in results[0][1].calls]==[c['kind'] for c in results[1][1].calls]
    assert len(results[0][1].calls)==10  # 4 mutation, 4 conclusion, 2 reflection
    assert results[0][0]['cost_total']==results[1][0]['cost_total']==pytest.approx(.1)


@pytest.mark.parametrize('inject',[False,True])
def test_fake_engine_resume_preserves_prompts_costs_rng_and_injection_mode(fake_task,tmp_path,inject):
    whole,whole_llm=fake_engine(fake_task,tmp_path/'whole',inject); whole_result=whole.run()
    first,first_llm=fake_engine(fake_task,tmp_path/'split',inject); first._stop_at=2
    assert first.run()['reason']=='paused'
    resumed,rest_llm=fake_engine(fake_task,tmp_path/'split',inject)
    checkpoint.load(resumed,str(tmp_path/'split/checkpoint.json'))
    assert resumed.research.state()==first.research.state()
    result=resumed.run()
    assert result==whole_result
    assert resumed.research.state()==whole.research.state()
    assert resumed.db.state()==whole.db.state() and resumed.rng.state()==whole.rng.state()
    assert first_llm.calls+rest_llm.calls==whole_llm.calls


def test_engine_resume_rejects_off_to_on_before_any_calls(fake_task,tmp_path):
    folder=tmp_path/'split'; first,_=fake_engine(fake_task,folder,False); first._stop_at=2; first.run()
    changed,llm=fake_engine(fake_task,folder,True)
    with pytest.raises(checkpoint.IdentityMismatch): checkpoint.load(changed,str(folder/'checkpoint.json'))
    assert llm.calls==[]
    changed.log.close()
