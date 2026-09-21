"""Continuous orchestration keeps finite calls, durable waiting and explicit renewal."""
from contextlib import nullcontext

import pytest

from sciel.engine.capacity import CapacityCancelled, CapacityTimeout
from sciel.methods.budget import BudgetUnknown, ResourceLedger
from sciel.methods.coordinator import Coordinator, Refusal, StopRequested, TrialQuotaExceeded
from sciel.methods.registry import Registry
from sciel.methods.tasks import SortTask
from test_methods_coordinator import (BUDGET, RESOURCES, FixedProposer, ScriptedExecutor,
    experiment, lab, method, register)


def setup_continuous(tmp_path, *, proposer=None, capacity=None, should_stop=None, **options):
    registry=Registry(tmp_path/'registry')
    ledger=ResourceLedger(tmp_path/'budget.jsonl',{k:None for k in BUDGET})
    executor=ScriptedExecutor()
    proposer=proposer or FixedProposer()
    config=dict(max_steps=None,calendar_seconds=None,storage_bytes=None,capacity_wait_seconds=None)
    config.update(options)
    c=Coordinator(registry,ledger,executor,proposer,SortTask(),resources=RESOURCES,
        capacity=capacity,should_stop=should_stop,**config)
    lab(c,'author')
    return c,registry,ledger,executor,proposer


def test_null_total_caps_allow_steps_beyond_old_limit_without_unbounded_calls(tmp_path,monkeypatch):
    c,r,ledger,_,proposer=setup_continuous(tmp_path,proposer=FixedProposer([{}]*10))
    monkeypatch.setattr(c,'_size',lambda: (_ for _ in ()).throw(AssertionError('unlimited headroom must not scan disk')))
    c._headroom()
    monkeypatch.undo()
    steps=[c.step('author') for _ in range(10)]
    assert [s['index'] for s in steps]==list(range(10)) and proposer.calls==10
    state=ledger.snapshot()
    assert state['limits']['steps'] is None and state['remaining']['wall_seconds'] is None
    assert state['spent']['steps']==10 and not state['halted']
    assert all(v is not None for row in state['reservations'] for v in row['bounds'].values())
    assert c.identity['max_steps'] is None and c.identity['calendar_seconds'] is None


@pytest.mark.parametrize('axis,value',[('max_steps',0),('max_steps',True),('calendar_seconds',float('inf')),
    ('calendar_seconds',False),('storage_bytes',1.5),('capacity_wait_seconds',-1)])
def test_null_configuration_does_not_relax_invalid_finite_limits(tmp_path,axis,value):
    with pytest.raises(ValueError):setup_continuous(tmp_path,**{axis:value})


def test_unlimited_queue_retries_admission_only_and_accounts_wait_once(tmp_path,monkeypatch):
    now=[100.]
    class Capacity:
        calls=0
        def acquire(self,kind,**kwargs):
            assert kind=='evaluator' and kwargs['timeout']==30
            self.calls+=1
            if self.calls<=2:
                now[0]+=30
                raise CapacityTimeout(30)
            now[0]+=1
            return nullcontext()
    capacity=Capacity()
    c,r,ledger,executor,_=setup_continuous(tmp_path,capacity=capacity)
    ident=register(r,method())
    plan=c.plan('author','one-dispatch-after-wait',experiment([ident]))
    monkeypatch.setattr('sciel.methods.coordinator.time.monotonic',lambda:now[0])
    result=c.execute(plan)
    assert r.get_artifact('experiment',result)['status']=='completed'
    assert capacity.calls==3 and len(executor.calls)==1
    state=ledger.snapshot()
    waits=[x for x in state['reservations'] if x['category']=='capacity_wait']
    assert len(waits)==3 and sum(float(x['actual']['wall_seconds']) for x in waits)==61
    invocation=next(x for x in state['reservations'] if x['category']=='execution')
    assert float(invocation['actual']['wall_seconds'])==0
    assert all(float(x['bounds']['wall_seconds'])==32 for x in waits)
    assert not state['halted'] and not state['unknown_reservations']
    before=state['spent']
    assert c.execute(plan)==result
    assert ledger.snapshot()['spent']==before and capacity.calls==3 and len(executor.calls)==1


def test_unexpected_queue_failure_does_not_charge_known_wait_again(tmp_path,monkeypatch):
    now=[100.]
    class Capacity:
        calls=0
        def acquire(self,kind,**kwargs):
            self.calls+=1
            if self.calls==1:
                now[0]+=30
                raise CapacityTimeout(30)
            now[0]+=1
            raise OSError('synthetic local capacity observer failure')
    c,r,ledger,executor,_=setup_continuous(tmp_path,capacity=Capacity())
    ident=register(r,method());plan=c.plan('author','queue-infrastructure-error',experiment([ident]))
    before=float(ledger.snapshot()['spent']['wall_seconds'])
    monkeypatch.setattr('sciel.methods.coordinator.time.monotonic',lambda:now[0])
    with pytest.raises(BudgetUnknown):c.execute(plan)
    state=ledger.snapshot()
    assert not executor.calls
    assert float(state['spent']['wall_seconds'])-before==pytest.approx(31)
    execution=next(x for x in state['reservations'] if x['category']=='execution')
    assert execution['status']=='unknown' and float(execution['actual']['wall_seconds'])==0
    assert len(state['unknown_reservations'])==2


def test_provider_unknown_after_queue_keeps_money_unknown_without_double_wait(tmp_path,monkeypatch):
    now=[100.]
    class Capacity:
        calls=0
        def acquire(self,kind,**kwargs):
            self.calls+=1
            now[0]+=30 if self.calls==1 else 1
            if self.calls==1:raise CapacityTimeout(30)
            return nullcontext()
    class UnknownProposer(FixedProposer):
        def generate(self,prompt):
            self.calls+=1
            now[0]+=.5
            raise OSError('synthetic disconnected provider')
    c,_,ledger,_,proposer=setup_continuous(tmp_path,capacity=Capacity(),proposer=UnknownProposer())
    monkeypatch.setattr('sciel.methods.coordinator.time.monotonic',lambda:now[0])
    with pytest.raises(BudgetUnknown):c.step('author')
    state=ledger.snapshot()
    assert proposer.calls==1 and float(state['spent']['wall_seconds'])==31.5
    generation=next(x for x in state['reservations'] if x['category']=='generation')
    assert float(generation['actual']['wall_seconds'])==.5 and 'dollars' not in generation['actual']
    assert float(state['reserved']['dollars'])==1


def test_operator_stop_during_provider_queue_is_known_no_dispatch_then_can_resume(tmp_path):
    stopped=[False]
    class Capacity:
        def acquire(self,kind,**kwargs):
            assert kind=='llm' and callable(kwargs['cancel'])
            stopped[0]=True
            raise CapacityCancelled(0)
    c,r,ledger,_,proposer=setup_continuous(tmp_path,capacity=Capacity(),should_stop=lambda:stopped[0],
        proposer=FixedProposer([dict(kind='stop',reason='fixture finished')]))
    with pytest.raises(StopRequested):c.step('author')
    state=ledger.snapshot()
    assert proposer.calls==0 and not state['unknown_reservations'] and not state['pending_reservations']
    assert float(state['spent']['dollars'])==0
    stopped[0]=False
    c.capacity=None
    # Complete the previous known no-dispatch attempt; do not reissue it.
    first=c.step('author')
    receipt=r.get_artifact('proposal_receipt',first['proposal_id'])
    assert receipt['error']=='proposal_not_dispatched' and receipt['diagnostic']['failure_scope']=='operator'
    assert proposer.calls==0
    assert c.step('author')['index']==1 and proposer.calls==1


def test_operator_stop_during_evaluator_queue_has_no_unknown_or_candidate_failure(tmp_path):
    stopped=[False]
    class Capacity:
        def acquire(self,kind,**kwargs):
            stopped[0]=True
            raise CapacityCancelled(0)
    c,r,ledger,executor,_=setup_continuous(tmp_path,capacity=Capacity(),should_stop=lambda:stopped[0])
    ident=register(r,method());plan=c.plan('author','operator-stop',experiment([ident]))
    with pytest.raises(StopRequested):c.execute(plan)
    assert not executor.calls and not ledger.snapshot()['unknown_reservations']
    stopped[0]=False
    result=r.get_artifact('experiment',c.execute(plan))
    assert result['diagnostics'][0]['dispatch_status']=='not_dispatched'
    assert result['diagnostics'][0]['failure_scope']=='operator'
    assert not executor.calls


def test_finite_capacity_refusal_has_explicit_infrastructure_diagnostic(tmp_path):
    class Capacity:
        def acquire(self,*args,**kwargs):raise CapacityTimeout(0)
    c,r,ledger,executor,_=setup_continuous(tmp_path,capacity=Capacity(),capacity_wait_seconds=1)
    ident=register(r,method());plan=c.plan('author','finite-refusal',experiment([ident]))
    result=r.get_artifact('experiment',c.execute(plan))
    assert result['diagnostics'][0]['dispatch_status']=='not_dispatched'
    assert result['diagnostics'][0]['failure_scope']=='infrastructure'
    assert not executor.calls and not ledger.snapshot()['unknown_reservations']


def test_explicit_reopening_keeps_history_global_index_and_account(tmp_path):
    c,r,ledger,_,proposer=setup_continuous(tmp_path,proposer=FixedProposer([
        dict(kind='stop',reason='First question exhausted'),dict(kind='stop',reason='Second question exhausted')]))
    first=c.step('author');before=ledger.snapshot()['spent']['steps']
    assert c.step('author')==first and proposer.calls==1
    account=c._account('author')
    grant=c.reopen_episode('author',question='Evaluate another bounded direction.',evidence_refs=[first['proposal_id']])
    assert grant['next_index']==1 and c._account('author')==account
    assert ledger.snapshot()['spent']['steps']==before
    assert c.reopen_episode('author',question='Evaluate another bounded direction.',evidence_refs=[first['proposal_id']])==grant
    second=c.step('author')
    assert second['index']==1 and proposer.calls==2
    assert c._events('episode_step')[0]==first and len(c._events('episode_reopened'))==1
    assert r.get_artifact('funding_grant',grant['grant_id'])['evidence_refs']==[first['proposal_id']]
    c.archive_lab('author',reason='fixture archive')
    with pytest.raises(Refusal,match='archived'):
        c.reopen_episode('author',question='No archive escape',evidence_refs=[second['proposal_id']])


def test_old_parent_quota_refusal_is_local_and_evidence_bound_renewal_works(tmp_path):
    c,r,ledger,executor,proposer=setup_continuous(tmp_path)
    bundle=method();ident=c.register_draft('author','own-draft',bundle['files'],bundle['manifest'])
    evidence=r.put_artifact('fixture_evidence',dict(technical_only=True))
    policy=c._funding()
    for index in range(8):
        policy.record_block('lab:author','fixture-block-'+str(index),informative=False,evidence_refs=[evidence])
    c._save_funding(policy)
    with pytest.raises(TrialQuotaExceeded):c.step('author')
    c.grant_direction('author','A separately authorized direction',[evidence])
    proposer.proposals=[dict(kind='experiment',experiment=experiment([ident]))]
    outcome=c.step('author')
    refusal=r.get_artifact('proposal_receipt',outcome['refusal_id'])
    assert outcome['status']=='continued' and not executor.calls
    assert refusal['funding_lineage']=='lab:author' and refusal['failure_scope']=='trial_policy'
    assert refusal['plan_id']==outcome['plan_id']
    with pytest.raises(Refusal,match='latest recorded'):
        c.renew_trial('author',1,question='No matching evidence',evidence_refs=[evidence],lineage_id='lab:author')
    c.renew_trial('author',1,question='One explicit repair trial',evidence_refs=[outcome['refusal_id']],lineage_id='lab:author')
    assert r.get_artifact('experiment',c.execute(outcome['plan_id']))['contract_passed']
    assert len(executor.calls)==1 and not ledger.snapshot()['halted']
    with pytest.raises(Refusal,match='existing lineage'):
        c.renew_trial('author',1,question='No foreign authority',evidence_refs=[evidence],lineage_id='lab:someone-else')


def test_prompt_revision_view_is_bounded_but_immutable_history_is_unchanged(tmp_path):
    c,r,_,_,_=setup_continuous(tmp_path)
    refs=[r.put_artifact('fixture_evidence',dict(index=i)) for i in range(25)]
    previous=r.lab('author')['active_revision']
    current=r.revise_lab('author',previous,dict(evidence_refs=refs),'fixture accumulated history')
    full=r.revision(current)
    context=c._context('author',current,full)
    assert context['revision']['evidence_refs']==refs[-8:]
    assert context['revision_delivery']['omitted_counts']['evidence_refs']==17
    assert r.revision(current)['evidence_refs']==refs
    assert context['revision_delivery']['immutable_revision_id']==current


def test_runtime_stop_callback_does_not_change_frozen_identity(tmp_path):
    c,r,ledger,executor,proposer=setup_continuous(tmp_path)
    restored=Coordinator(r,ledger,executor,proposer,SortTask(),resources=RESOURCES,max_steps=None,
        calendar_seconds=None,storage_bytes=None,capacity_wait_seconds=None,should_stop=lambda:True)
    assert restored.identity==c.identity
    with pytest.raises(StopRequested):restored.step('author')
    assert not ledger.snapshot()['reservations']
