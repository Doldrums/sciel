"""Native structured forecasts and source commitments, without provider/solver I/O."""
import copy
import json
from types import SimpleNamespace as NS

import pytest

from sciel.config import apply_defaults
from sciel.engine.db import Database
from sciel.engine.research import ResearchController, digest
from sciel.engine.types import Snapshot
from sciel.hypothesis import circle_restart_body as body
from sciel.hypothesis.controlled_changes import validate_controlled_change

BODY = 'proposal = best_params.copy()\nproposal[52:] *= 0.99\nreturn proposal'


def response(code=BODY, **overrides):
    value = {'hypothesis': {'hypothesis': 'Redistribute the initial radii.',
        'predicted_low': -.1, 'predicted_high': .2, 'reasoning': 'A different start may change local search.',
        'confidence': .5}, 'body': code}
    fields = dict(text=json.dumps(value, ensure_ascii=False), cost=.003, model='fake',
                  prompt_tokens=31, completion_tokens=17, finish_reason='stop', refusal=None)
    fields.update(overrides)
    return NS(**fields)


def fixture():
    events=[]; model=NS(models=['fake'], max_retries=1)
    log=NS(research=lambda **event: events.append(copy.deepcopy(event)))
    cfg=apply_defaults({'research':{'enabled':True,'max_aux_calls':0,'reflection_interval':0}})['research']
    research=ResearchController(cfg,model,log,lambda _:None)
    db=Database(1);source=body.source_for();parent=db.add(source,.8,{'sum_radii':2.108},0,'seed')
    snap=Snapshot(0,'circle_generate','fake',0,parent.id,source,.8)
    snap.research_context=research.context(db,parent,0,1)
    ident,prompt=research.begin(snap,0,'circle_generate','Write one new body.',
        response_instruction=body.response_instruction(),context_scope=body.OPERATION)
    return NS(research=research,events=events,snapshot=snap,ident=ident,prompt=prompt)


def commit(f, resp):
    proposal=body.parse_response(resp.text);built=body.apply(f.snapshot.parent_code,proposal.body)
    assert f.research.proposal(f.ident,resp,built['code'],NS(code=built['code']),
        response_format=body.OPERATION,controlled_change=built['proof'])
    return built


def test_raw_reply_and_known_cost_precede_parsing_without_fabricating_program(monkeypatch):
    f=fixture();r=response();original=body.decode_response
    def inspect(raw):
        assert f.events[-1]['phase']=='structured_response'
        assert f.events[-1]['response']==raw==r.text
        pending=f.research.pending[f.ident]
        assert pending['generation_cost']==r.cost and pending['generation_dispatch']=='returned'
        assert pending['source_sha256'] is pending['proposed_sha256'] is None
        return original(raw)
    monkeypatch.setattr(body,'decode_response',inspect)
    decoded=f.research.record_response(f.ident,r,response_format=body.OPERATION)
    assert decoded['proposal'].body==BODY and decoded['provider_error'] is None
    assert f.research.valid_hypotheses==1
    assert all(e.get('response',r.text)==r.text for e in f.events if e['phase']=='structured_response')


def test_valid_native_proposal_preserves_raw_forecast_and_exact_compiler_source():
    f=fixture();r=response();f.research.record_response(f.ident,r,response_format=body.OPERATION)
    built=commit(f,r)
    assert f.research.valid_hypotheses==1
    event=f.events[-1]
    assert event['phase']=='hypothesis' and event['response']==r.text
    assert event['response_sha256']==digest(r.text) and event['response_format']==body.OPERATION
    result=f.research.resolve(f.ident,status='evaluated',score=.9,source=built['code'])
    assert result['parent_score']==.8 and result['actual_delta']==.9-.8
    assert result['forecast_verdict']=='CONFIRMED' and not result['source_repaired']
    assert result['source_sha256']==result['proposed_sha256']==digest(built['code'])
    assert not f.research._structured_responses
    f.research.complete_step();state=f.research.state()
    restored=ResearchController(f.research.config,NS(models=['fake'],max_retries=1),
        NS(research=lambda **_:None),lambda _:None)
    restored.load_state(state)
    assert restored.state()==state


@pytest.mark.parametrize('code',['return incumbent._params','return time.monotonic()',
                                  'return _relocation_vector(best_params)'])
def test_valid_forecast_invalid_interface_is_retained_without_source_or_quality(code):
    f=fixture();r=response(code)
    value=f.research.record_response(f.ident,r,response_format=body.OPERATION)
    assert value['hypothesis'] is not None and value['proposal'] is None and value['error']
    record=f.research.resolve(f.ident,status='no_child',failure=value['error'])
    assert record['hypothesis'] is not None and record['generation_cost']==r.cost
    assert all(record[k] is None for k in ('source_sha256','proposed_sha256','score','actual_delta'))
    assert record['forecast_verdict']=='INCONCLUSIVE' and f.research.knowledge.lessons==[]


@pytest.mark.parametrize('text',['```json\n{}\n```','{"hypothesis":{},"body":"return None","extra":0}',
                                '{"hypothesis":{},"hypothesis":{},"body":"return None"}'])
def test_invalid_transport_never_invents_forecast_or_legacy_marker(text):
    f=fixture();r=response(text=text)
    value=f.research.record_response(f.ident,r,response_format=body.OPERATION)
    assert value['hypothesis'] is None and value['proposal'] is None
    record=f.research.resolve(f.ident,status='hypothesis_invalid',failure=value['error'])
    assert record['hypothesis'] is None and record['source_sha256'] is None
    assert f.events[-3]['response']==text and f.events[-3]['response_sha256']==digest(text)


@pytest.mark.parametrize('field,value',[('finish_reason','length'),('finish_reason',None),
                                      ('refusal','refused'),('refusal',False)])
def test_incomplete_or_refused_reply_cannot_commit_even_if_json_valid(field,value):
    f=fixture();r=response(**{field:value})
    decoded=f.research.record_response(f.ident,r,response_format=body.OPERATION)
    assert decoded['provider_error']
    with pytest.raises(ValueError,match='durable raw response'):commit(f,r)
    assert f.research.pending[f.ident]['source_sha256'] is None


def test_response_and_code_binding_rejects_rewrite_and_another_body():
    f=fixture();r=response();f.research.record_response(f.ident,r,response_format=body.OPERATION)
    before=copy.deepcopy(f.research.state())
    for altered in (response('return ordinary_params'),response(text=r.text+' '),response(cost=.004)):
        with pytest.raises(ValueError):commit(f,altered)
        assert f.research.state()==before
    foreign=body.apply(f.snapshot.parent_code,'return initial_params')
    with pytest.raises(ValueError,match='body or forecast'):
        f.research.proposal(f.ident,r,foreign['code'],NS(code=foreign['code']),
            response_format=body.OPERATION,controlled_change=foreign['proof'])
    with pytest.raises(ValueError,match='explicit transport'):
        built=body.apply(f.snapshot.parent_code,BODY)
        f.research.proposal(f.ident,r,built['code'],NS(code=built['code']))
    assert f.research.state()==before


def test_premarked_returned_cost_is_accepted_once_but_cannot_change():
    f=fixture();r=response();f.research.pending[f.ident].update(generation_dispatch='returned',generation_cost=r.cost)
    f.research.record_response(f.ident,r,response_format=body.OPERATION)
    with pytest.raises(ValueError,match='already recorded'):
        f.research.record_response(f.ident,r,response_format=body.OPERATION)
    assert f.research.valid_hypotheses==1
    other=fixture();other.research.pending[other.ident].update(generation_dispatch='returned',generation_cost=.5)
    with pytest.raises(ValueError,match='recorded charge'):
        other.research.record_response(other.ident,r,response_format=body.OPERATION)
    assert other.research.pending[other.ident]['generation_cost']==.5


def test_parser_failure_keeps_durable_raw_and_charge(monkeypatch):
    f=fixture();r=response()
    def broken(_):raise RuntimeError('internal codec failure')
    monkeypatch.setattr(body,'decode_response',broken)
    with pytest.raises(RuntimeError):f.research.record_response(f.ident,r,response_format=body.OPERATION)
    assert f.events[-1]['response']==r.text
    assert f.research.pending[f.ident]['generation_cost']==r.cost
    f.research.abort(f.ident,'internal stop')
    record=f.research.knowledge.experiments[0]
    assert record['generation_cost']==r.cost and record['generation_dispatch']=='returned'
    assert record['score'] is None and not f.research._structured_responses


def test_body_scope_omits_executable_fragments_with_explicit_hashes():
    f=fixture();ctx=f.snapshot.research_context
    ctx['fragments']=[{'id':'f:old','source_sha256':digest('foreign_private_helpers'),'code':'foreign_private_helpers'}]
    ident,prompt=f.research.begin(f.snapshot,1,'circle_generate','Interface only.',
        response_instruction=body.response_instruction(),context_scope=body.OPERATION)
    assert 'foreign_private_helpers' not in prompt
    assert digest('foreign_private_helpers') in prompt and 'operator_interface' in prompt
    assert 'f:old' in prompt and ctx['fragments'][0]['code']=='foreign_private_helpers'
    assert f.research.pending[ident]['parent_source_sha256']==digest(f.snapshot.parent_code)


def test_body_proof_is_rederived_including_raw_body_and_outside_bytes():
    parent=body.source_for();built=body.apply(parent,BODY)
    args=dict(parent_source_sha256=digest(parent),proposed_sha256=digest(built['code']),
        source_sha256=digest(built['code']),source_repaired=False)
    assert validate_controlled_change(built['proof'],**args)==built['proof']
    for key,value in [('target_body','return initial_params'),('parent_body','return initial_params'),
                      ('parent_source_sha256','0'*64),('extra',True)]:
        damaged=copy.deepcopy(built['proof']);damaged[key]=value
        with pytest.raises(ValueError):validate_controlled_change(damaged,**args)
    with pytest.raises(ValueError):validate_controlled_change(built['proof'],**{**args,'source_repaired':True})
