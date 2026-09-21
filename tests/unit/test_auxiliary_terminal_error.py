"""Known terminal provider errors can be skipped by policy, never retried."""
from copy import deepcopy
import json
import sys

import pytest

from sciel.campaign.budget import CampaignStopped
from sciel.engine import checkpoint
from sciel.engine.factory import build_engine
from sciel.engine.llm.client import LLMResponse
from sciel.engine.parallel_store import WaveTransaction
from test_conclusion_format_repair import setup
from test_parallel_engine import ResearchProvider, config, evaluator, events, task_fixture


@pytest.mark.parametrize('cost', [0., .03])
@pytest.mark.parametrize('malformed', [False, True])
def test_serial_error_skips_conclusion_without_repair_or_lesson(cost, malformed):
    controller, model, rows, charges = setup([], incomplete_auxiliary='skip')
    before=deepcopy(controller.knowledge.experiments)
    calls=[]
    def generate(*args, **kwargs):
        calls.append(True)
        return LLMResponse('' if malformed else 'unusable returned text', 'fixture', cost,
            None if malformed else 0, None if malformed else 0,
            finish_reason=None if malformed else 'error',
            response_error=({'stage':'response','reason':'empty_choices','cause_type':'OtherError'}
                            if malformed else None))
    model.generate=generate
    controller.complete_step()
    assert calls == [True] and charges == [cost] and controller.aux_cost == cost
    assert controller.conclusions['r000001a1']['status']=='skipped'
    assert controller.conclusions['r000001a1']['llm'] is None
    assert controller.knowledge.lessons==[] and controller.knowledge.experiments==before
    assert controller.stopped_reason is None and controller.queue==[]
    assert not any(row['phase'] in ('conclusion_repair','conclusion_rejected') for row in rows)
    skipped=[row for row in rows if row['phase']=='aux_skipped']
    assert len(skipped)==1 and skipped[0]['reason']=='response_error'
    assert skipped[0]['cost']==cost and skipped[0]['current_call_cost_unknown'] is False


@pytest.mark.parametrize('policy,cost,marker', [
    ('stop', .03, None), ('skip', None, None), ('skip', .03, {'reason':'bogus'}),
])
def test_unknown_cost_explicit_stop_and_invalid_error_receipt_remain_fail_closed(policy,cost,marker):
    controller, model, rows, charges=setup([],incomplete_auxiliary=policy)
    model.generate=lambda *a,**k:LLMResponse('', 'fixture', cost, 0, 0,
                                            finish_reason='error',response_error=marker)
    with pytest.raises(CampaignStopped):controller.complete_step()
    assert controller.stopped_reason and not controller.conclusions and not controller.knowledge.lessons
    assert charges == ([] if cost is None else [.03])
    assert rows[-1]['phase']=='aux_stopped'


def test_returned_error_receipt_cannot_be_charged_or_consumed_twice():
    c,_,rows,charges=setup([],incomplete_auxiliary='skip')
    job=c.prepare_conclusions()[0]
    request=c.prepare_aux('conclusion',job['prompt'],job['record']['id'],
        conclusion_evidence=job['evidence'])
    response=LLMResponse('terminal error','fixture',.03,0,0,finish_reason='error')
    assert c.consume_aux(request,response=response) is None
    with pytest.raises(ValueError,match='already consumed'):
        c.consume_aux(request,response=response)
    assert charges==[.03]


@pytest.mark.parametrize('replay', [False, True])
@pytest.mark.parametrize('conclusion_tokens', [None, 4096])
def test_parallel_wave_commits_skipped_conclusions_and_replays_paid_receipts_once(tmp_path,monkeypatch,replay,conclusion_tokens):
    class Provider(ResearchProvider):
        def generate(self,prompt,model,temperature=None,**options):
            response=super().generate(prompt,model,temperature,**options)
            if 'Review this one experiment.' in prompt[:160]:
                assert options.get('max_tokens') == conclusion_tokens
                response.text='terminal provider failure'
                response.finish_reason='error'
            return response
    tasks=tmp_path/'tasks'
    task=task_fixture(tasks)
    cfg=config(task,tmp_path/'capacity',workers=4,max_iterations=4,
        research={'enabled':True,'structured_auxiliary':True,'incomplete_auxiliary':'skip',
                  'reflection_interval':0,'max_aux_calls':None,'conclusion_repair_attempts':1,
                  'conclusion_source_refs':True,'conclusion_max_tokens':conclusion_tokens})
    provider=Provider()
    evaluations,_=evaluator(monkeypatch)
    directory=tmp_path/'run'
    commit=WaveTransaction.commit
    crashed=False
    def crash(self):
        nonlocal crashed
        if replay and self.wave_id=='1' and not crashed:
            crashed=True
            assert self.engine.research.aux_calls==4
            assert all(row['status']=='skipped' for row in self.engine.research.conclusions.values())
            raise RuntimeError('inert crash after all returned auxiliary receipts')
        return commit(self)
    monkeypatch.setattr(WaveTransaction,'commit',crash)
    engine=build_engine(cfg,str(tasks),str(directory),python=sys.executable,llm=provider)
    if replay:
        with pytest.raises(CampaignStopped):engine.run()
        assert len(provider.calls)==8 and len(evaluations)==5
        engine=build_engine(cfg,str(tasks),str(directory),python=sys.executable,llm=provider)
        checkpoint.load(engine,str(directory/'checkpoint.json'))
    result=engine.run()
    assert result['iteration_no']==result['B_evals']==4
    assert result['cost_total']==pytest.approx(.16) and engine.research.aux_cost==pytest.approx(.04)
    assert len(provider.calls)==8 and provider.roles.count('C')==4 and len(evaluations)==5
    assert len(engine.research.conclusions)==4 and not engine.research.knowledge.lessons
    assert all(row['status']=='skipped' and row['llm'] is None for row in engine.research.conclusions.values())
    rows=events(directory)
    assert len([row for row in rows if row.get('phase')=='aux_skipped'])==4
    assert not any(row.get('phase')=='conclusion_repair' for row in rows)
    saved=json.loads((directory/'checkpoint.json').read_text())
    assert saved['research']==engine.research.state() and saved['cost_total']==pytest.approx(.16)
