"""Auxiliary schemas traverse real Research and budgeted SDK stubs; no I/O execution."""
import copy
import json
from types import SimpleNamespace as NS

import pytest

from sciel.campaign.budget import BudgetedLLM, CampaignStopped, CostLedger, request_cost_bound
from sciel.config import apply_defaults
from sciel.engine.db import Database
from sciel.engine.research import ResearchController
from sciel.engine.types import Snapshot
from sciel.hypothesis import circle_restart_body as body
from sciel.hypothesis.auxiliary_schema import identity, response_format
from sciel.hypothesis.prompts import conclusion_prompt, reflection_prompt
from test_research_circle_body import response, BODY


def construct(tmp_path, *, structured=True, sdk_error=False, finish='stop', refusal=None, bad_evidence=False,
              mechanism_entry=None):
    events, charges, calls = [], [], []
    cfg=apply_defaults({'research':{'enabled':True,'reflection_interval':1,'max_aux_calls':2,
        'batch_only_lessons':True, **({'structured_auxiliary':structured} if structured is not None else {})}})['research']
    llm_cfg=dict(models=['fake'],weights=[1],max_tokens=256,api_base='https://openrouter.ai/api/v1')
    pricing={'models':[{'id':'fake','pricing':{'prompt':'0.000001','completion':'0.000002'}}]}
    book=CostLedger(tmp_path/'cost.json',1)
    def create(**kw):
        calls.append(copy.deepcopy(kw))
        if sdk_error:raise RuntimeError('fixture provider failure')
        if len(calls)==1:
            payload=dict(verdict='CONFIRMED',explanation='The measured score increased within the interval.',lessons=[],
                mechanism_review=[dict(claim_quote='Redistribute the initial radii.',source_quotes=[],assessment='unknown',
                    direct_effect='The fixture does not verify a causal mechanism.',untested_downstream='Replication is unknown.')])
            if structured is True:
                payload['mechanism_review'][0].update(evidence_kind='unavailable', forecast_ref=None)
            if mechanism_entry is not None:
                payload['mechanism_review'] = [copy.deepcopy(mechanism_entry)]
        else:
            lessons=[] if not bad_evidence else [dict(rule='Invented support.',polarity='positive',scope='branch',branch='p1',evidence_ids=['missing'])]
            payload=dict(lessons=lessons,retire_ids=[])
        return NS(id='fixture'+str(len(calls)),model='fake',usage=NS(cost=.001,prompt_tokens=10,completion_tokens=10),
            choices=[NS(finish_reason=finish,message=NS(content=json.dumps(payload),refusal=refusal))])
    model=BudgetedLLM(llm_cfg,pricing,book,'fixture','fixture-key',client=NS(chat=NS(completions=NS(create=create))))
    research=ResearchController(cfg,model,NS(research=lambda **row:events.append(copy.deepcopy(row))),charges.append)
    db=Database(1);source=body.source_for();parent=db.add(source,.8,{},0,'seed')
    snap=Snapshot(0,'circle_generate','fake',0,parent.id,source,.8)
    snap.research_context=research.context(db,parent,0,1)
    ident,_=research.begin(snap,0,'circle_generate','Inert fixture.',response_instruction=body.response_instruction(),context_scope=body.OPERATION)
    resp=response();research.record_response(ident,resp,response_format=body.OPERATION)
    built=body.apply(source,BODY)
    assert research.proposal(ident,resp,built['code'],NS(code=built['code']),response_format=body.OPERATION,controlled_change=built['proof'])
    research.resolve(ident,status='evaluated',score=.9,source=built['code'])
    return NS(research=research,book=book,events=events,charges=charges,calls=calls,model=model,ident=ident)


def test_real_controller_dispatches_both_schemas_and_retains_authoritative_evidence(tmp_path):
    f=construct(tmp_path);before=copy.deepcopy(f.research.knowledge.experiments)
    f.research.complete_step()
    assert f.research.knowledge.experiments==before
    assert f.research.conclusions[f.ident]['status']=='parsed' and f.research.reflections==1
    assert f.research.knowledge.lessons==[] and f.research.aux_cost==.002
    starts=[r for r in f.events if r['phase']=='aux_start']
    state=json.loads(f.book.path.read_text());payments=list(state['calls'].values())
    assert len(f.calls)==len(starts)==len(payments)==2
    for index,kind in enumerate(('conclusion','reflection')):
        wire=f.calls[index];prompt=wire['messages'][0]['content']
        assert wire['response_format']==starts[index]['response_format']==response_format(kind)
        assert starts[index]['auxiliary_schema']==identity()
        assert 'Return one bare JSON object' in prompt and ('Return exactly '+kind.upper()+'_JSON:') not in prompt
        assert 'Every lesson must include branch:' in prompt
        assert 'optional branch field may be omitted' not in prompt
        assert 'five required fields' in prompt
        assert wire['extra_body']['provider']['allow_fallbacks'] is False
        bound=request_cost_bound(prompt,256,f.model.prices['fake'],response_format=response_format(kind))
        assert f.book.totals(state)==(f.book.totals(state)[0],0)
        assert payments[index]['reservation_usd']==str(bound)
        assert bound>request_cost_bound(prompt,256,f.model.prices['fake'])
    restored=ResearchController(f.research.config,f.model,NS(research=lambda **_:pytest.fail('replay')),lambda _:pytest.fail('charge'))
    restored.load_state(f.research.state())
    assert restored.state()==f.research.state()


@pytest.mark.parametrize('structured',[False,None])
def test_legacy_requests_keep_no_schema_and_legacy_output_instruction(tmp_path,structured):
    f=construct(tmp_path,structured=structured);f.research.complete_step()
    assert all('response_format' not in call for call in f.calls)
    assert all('auxiliary_schema' not in row for row in f.events)
    assert 'auxiliary_schema' not in f.research.state()
    assert 'Return exactly CONCLUSION_JSON:' in f.calls[0]['messages'][0]['content']


@pytest.mark.parametrize('finish,refusal',[('length',None),('content_filter',None),(None,None),('stop','Refused'),('stop',{'bad':'metadata'})])
def test_structured_incomplete_or_refused_reply_is_paid_preserved_and_never_learned_or_retried(tmp_path,finish,refusal):
    f=construct(tmp_path,finish=finish,refusal=refusal)
    with pytest.raises(CampaignStopped):f.research.complete_step()
    assert len(f.calls)==1 and f.research.aux_cost==.001 and f.research.aux_calls==1
    result=[r for r in f.events if r['phase']=='aux_result'][0]
    assert result['response'].startswith('{') and not result['structured_response_usable']
    assert f.research.knowledge.lessons==[] and f.research.conclusions=={}
    with pytest.raises(CampaignStopped):f.research._aux('reflection','inert',1)
    assert len(f.calls)==1 and f.book.totals(json.loads(f.book.path.read_text()))[1]==0


def test_unknown_provider_outcome_retains_schema_reservation_and_stops_without_fallback(tmp_path):
    f=construct(tmp_path,sdk_error=True)
    with pytest.raises(CampaignStopped):f.research.complete_step()
    state=json.loads(f.book.path.read_text());assert len(state['calls'])==1
    assert state['halted'] and f.book.totals(state)[1]>0 and f.research.aux_cost==0
    with pytest.raises(CampaignStopped):f.research._aux('conclusion','inert',f.ident)
    assert len(f.calls)==1


def test_schema_conformity_cannot_authorize_missing_lesson_evidence(tmp_path):
    f=construct(tmp_path,bad_evidence=True);f.research.complete_step()
    assert f.research.conclusions[f.ident]['status']=='parsed'
    assert not f.research.knowledge.lessons and f.research.reflections==0
    event=[r for r in f.events if r['phase']=='reflection'][0]
    assert event['status']=='parsed' and event['error'] is not None


@pytest.mark.parametrize('mutation',['missing','changed','unexpected_disabled'])
def test_checkpoint_schema_identity_is_required_only_for_enabled_mode(tmp_path,mutation):
    f=construct(tmp_path,structured=mutation!='unexpected_disabled');f.research.complete_step()
    state=f.research.state()
    if mutation=='missing':state.pop('auxiliary_schema')
    elif mutation=='changed':state['auxiliary_schema']['version']='changed'
    else:state['auxiliary_schema']=identity()
    with pytest.raises(ValueError,match='auxiliary schema mismatch'):f.research.load_state(state)


def test_schema_drift_fails_before_intent_reservation_or_dispatch(tmp_path,monkeypatch):
    f=construct(tmp_path)
    monkeypatch.setattr('sciel.hypothesis.auxiliary_schema.identity',lambda **_: {'version':'changed'})
    with pytest.raises(ValueError,match='schema changed'):f.research.complete_step()
    assert not f.calls and f.research.aux_calls==0 and not json.loads(f.book.path.read_text())['calls']


@pytest.mark.parametrize('value',[1,0,None,'true',{},[]])
def test_config_and_direct_controller_reject_nonboolean_mode(value):
    with pytest.raises(ValueError,match='structured_auxiliary'):apply_defaults({'research':{'enabled':True,'structured_auxiliary':value}})
    cfg=apply_defaults({'research':{'enabled':True}})['research'];cfg['structured_auxiliary']=value
    with pytest.raises(ValueError,match='structured_auxiliary'):
        ResearchController(cfg,NS(models=['fake']),NS(),lambda _:None)


@pytest.mark.parametrize('prompt,args',[(conclusion_prompt,({},)),(reflection_prompt,([],[]))])
def test_prompt_mode_rejects_nonboolean_configuration(prompt,args):
    with pytest.raises(ValueError,match='boolean'):prompt(*args,structured=1)
