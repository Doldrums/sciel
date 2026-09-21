"""Auxiliary/worker failure receipts with local stubs; no API or evaluation."""
from __future__ import annotations

import copy
import json
from types import SimpleNamespace

import pytest

from sciel.campaign.budget import CampaignStopped
from sciel.config import apply_defaults
from sciel.engine.db import Database, normalized_hash
from sciel.engine.llm.client import LLMResponse
from sciel.engine.research import ResearchController
from sciel.engine.types import Snapshot
from sciel.engine.worker import run_worker


class Model:
    models = ['fake']
    max_retries = 1

    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def generate(self, prompt, model, temperature=None):
        self.calls.append(prompt)
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def setup(responses, *, fail_journal=()):
    model, rows, charged = Model(responses), [], []
    def emit(**row):
        if row['phase'] in fail_journal:
            raise OSError('sensitive-provider-or-filesystem-fixture')
        rows.append(copy.deepcopy(row))
    cfg = apply_defaults({'research':{'enabled':True,'max_aux_calls':10,'reflection_interval':0}})['research']
    research = ResearchController(cfg,model,SimpleNamespace(research=emit),charged.append)
    return research,model,rows,charged


def response(**updates):
    return SimpleNamespace(**({'cost':.03,'text':'{}','model':'fake',
                              'prompt_tokens':10,'completion_tokens':5} | updates))


@pytest.mark.parametrize('updates',[
    {'text':None},{'text':['sensitive-response-fixture']},{'model':None},{'model':''},
    {'prompt_tokens':None},{'prompt_tokens':True},{'completion_tokens':-1},
    {'completion_tokens':2.5},{'completion_tokens':10**9+1},
])
def test_malformed_returned_auxiliary_is_charged_once_then_stopped(updates):
    research,model,rows,charged = setup([response(**updates)])
    with pytest.raises(CampaignStopped,match='auxiliary response stopped: ValueError') as error:
        research._aux('conclusion','fixture prompt','r1')
    assert charged == [.03] and research.aux_cost == .03 and research.aux_calls == 1
    assert len(model.calls) == 1
    assert [r['phase'] for r in rows] == ['aux_start','aux_stopped']
    assert rows[-1]['generation_returned'] and rows[-1]['cost'] == .03
    assert rows[-1]['current_call_cost_unknown'] is False
    assert 'sensitive' not in str(error.value) and 'sensitive' not in json.dumps(rows)
    with pytest.raises(CampaignStopped):
        research._aux('conclusion','same or new prompt','r1')
    assert len(model.calls) == 1 and charged == [.03]


@pytest.mark.parametrize('cost',[None,True,-.01,float('nan'),float('inf'),'0.03'])
def test_unknown_auxiliary_cost_does_not_fabricate_a_charge(cost):
    research,model,rows,charged = setup([response(cost=cost)])
    with pytest.raises(CampaignStopped):
        research._aux('conclusion','fixture','r1')
    assert research.aux_calls == 1 and research.aux_cost == 0 and charged == []
    assert rows[-1]['phase'] == 'aux_stopped' and rows[-1]['cost'] is None
    assert rows[-1]['current_call_cost_unknown'] is True
    assert len(model.calls) == 1


@pytest.mark.parametrize('failures',[('aux_result',),('aux_result','aux_stopped')])
def test_auxiliary_result_persistence_failure_stops_with_known_charge(failures):
    research,model,_rows,charged = setup([response()],fail_journal=failures)
    with pytest.raises(CampaignStopped,match='auxiliary response stopped: OSError') as error:
        research._aux('reflection','fixture',4)
    assert 'sensitive' not in str(error.value)
    assert ('stop persistence failed' in str(error.value)) == ('aux_stopped' in failures)
    assert charged == [.03] and research.aux_cost == .03 and research.aux_calls == 1
    with pytest.raises(CampaignStopped):
        research._aux('reflection','fixture',4)
    assert len(model.calls) == 1 and charged == [.03]


def test_auxiliary_missing_metadata_preserves_known_charge():
    item = response()
    del item.text
    research,_,rows,charged = setup([item])
    with pytest.raises(CampaignStopped,match='AttributeError'):
        research._aux('conclusion','fixture','r1')
    assert charged == [.03] and rows[-1]['cost'] == .03


def test_auxiliary_zero_cost_and_empty_text_remain_valid_transport():
    research,model,rows,charged = setup([response(cost=0,text='')])
    assert research._aux('conclusion','fixture','r1') == ''
    assert charged == [0.] and research.aux_cost == 0 and research.stopped_reason is None
    assert len(model.calls) == 1 and rows[-1]['phase'] == 'aux_result'


def test_auxiliary_charge_callback_is_never_retried_on_its_failure():
    research,model,_,_charged = setup([response()])
    attempts = []
    def broken_charge(value):
        attempts.append(value)
        raise RuntimeError('sensitive-accounting-fixture')
    research.charge = broken_charge
    with pytest.raises(CampaignStopped,match='auxiliary response stopped: RuntimeError'):
        research._aux('conclusion','fixture','r1')
    assert attempts == [.03] and research.aux_cost == .03
    with pytest.raises(CampaignStopped):
        research._aux('conclusion','fixture','r1')
    assert len(model.calls) == 1 and attempts == [.03]


def worker_setup(*, research_on):
    parent = 'def run_search():\n    return .5\n'
    hyp = {'hypothesis':'Repeat fixture source.','predicted_low':0.,'predicted_high':.1,
           'reasoning':'Local fixture.','confidence':.5}
    text = 'HYPOTHESIS_JSON: '+json.dumps(hyp)+'\n```python\n'+parent+'```\n'
    research,model,rows,charged = setup([LLMResponse(text,'fake',.01,10,10),
                                       RuntimeError('sensitive-raw-provider-fixture')])
    db = Database(1)
    program = db.add(parent,.5,{},0,'seed')
    snapshot = Snapshot(0,'diff','fake',0,program.id,parent,.5,
        code_hashes={normalized_hash(parent)},required_entrypoints=['run_search'])
    if research_on:
        snapshot.research_context = research.context(db,program,0,1)
    return research,model,rows,charged,snapshot


@pytest.mark.parametrize('kind', ['conclusion', 'reflection'])
def test_explicit_skip_policy_charges_truncated_auxiliary_once_without_retry(kind):
    research, model, rows, charged = setup([])
    research.config.update(structured_auxiliary=True, incomplete_auxiliary='skip')
    from sciel.hypothesis.auxiliary_schema import identity
    research._auxiliary_schema_identity = identity()
    def generate(*args, **kwargs):
        model.calls.append(args)
        return response(finish_reason='length', refusal=None)
    model.generate = generate
    assert research._aux(kind, 'fixture', 'r1') is None
    assert charged == [.03] and research.aux_calls == 1 and len(model.calls) == 1
    assert research.stopped_reason is None
    assert [row['phase'] for row in rows] == ['aux_start', 'aux_result', 'aux_skipped']
    assert rows[-1]['reason'] == 'response_length'
    assert rows[-1]['current_call_cost_unknown'] is False


@pytest.mark.parametrize('updates', [dict(cost=None, finish_reason='length'),
    dict(finish_reason='length', refusal='blocked'), dict(finish_reason='length', refusal=True),
    dict(finish_reason=None)])
def test_skip_policy_still_stops_unknown_cost_refusal_and_missing_metadata(updates):
    research, model, rows, charged = setup([])
    research.config.update(structured_auxiliary=True, incomplete_auxiliary='skip')
    from sciel.hypothesis.auxiliary_schema import identity
    research._auxiliary_schema_identity = identity()
    model.generate = lambda *a, **k: response(**updates)
    with pytest.raises(CampaignStopped):
        research._aux('conclusion', 'fixture', 'r1')
    assert research.stopped_reason and rows[-1]['phase'] == 'aux_stopped'


def test_worker_raw_provider_exception_preserves_prior_known_cost_and_stops():
    research,model,rows,charged,snapshot = worker_setup(research_on=True)
    with pytest.raises(CampaignStopped,match='generation outcome unknown: RuntimeError') as error:
        run_worker('diff',snapshot,model,lambda _:pytest.fail('must not evaluate'),dup_cap=4,research=research)
    assert len(model.calls) == 2 and charged == [.01]
    assert research.attempts == research.outcomes == 2 and not research.pending
    record = research.knowledge.experiments[-1]
    assert record['status'] == 'generation_stopped' and record['generation_dispatch'] == 'unknown'
    assert record['score'] is record['actual_delta'] is record['generation_cost'] is None
    assert 'sensitive' not in str(error.value) and 'sensitive' not in json.dumps(rows)


def test_worker_raw_provider_exception_marks_unknown_offpath_charge():
    _research,model,_rows,charged,snapshot = worker_setup(research_on=False)
    with pytest.raises(CampaignStopped, match='outcome and cost are unknown') as stopped:
        run_worker('diff',snapshot,model,lambda _:pytest.fail('must not evaluate'),dup_cap=4)
    assert len(model.calls) == 2 and charged == []
    assert stopped.value.known_cost == .01
    assert stopped.value.current_call_cost_unknown is True
    assert 'sensitive' not in str(stopped.value)
