"""Full coordinator/ledger/registry evidence flow with an explicit fake executor.

Only external method execution is scripted; plans, task measurements, resource
receipts, cohort assignment and authorization use production implementations.
No generated source, provider call, or live campaign is executed by this suite.
"""
import pytest

from sciel.methods.coordinator import Coordinator, Refusal
from sciel.methods.tasks import SortTask
from test_methods_coordinator import (
    BUDGET, RESOURCES, FixedProposer, ScriptedExecutor, experiment, lab, method, setup,
)
from sciel.methods.budget import ResourceLedger
from sciel.methods.registry import Registry


def assigned_campaign(tmp_path, *, transfer_policy='methods'):
    if transfer_policy=='methods':
        c,r,ledger,executor,proposer=setup(tmp_path)
    else:
        r=Registry(tmp_path/'registry')
        ledger=ResourceLedger(tmp_path/'budget.jsonl',BUDGET)
        executor,proposer=ScriptedExecutor(),FixedProposer()
        c=Coordinator(r,ledger,executor,proposer,SortTask(),resources=RESOURCES,
                      max_steps=8,transfer_policy=transfer_policy)
    for name,mode in [('author','B'),('review1','B'),('review2','C'),('replay','B')]:
        lab(c,name,mode)
    bundle=method()
    ident=c.register_draft('author','draft-before-assignments',bundle['files'],bundle['manifest'])
    exp=experiment([ident])
    protocol_id=c.freeze_holdouts(ident,[
        dict(reviewer_lab='review1',cases=[dict(id='heldout-1',input=[6,4,5])]),
        dict(reviewer_lab='review2',cases=[dict(id='heldout-2',input=[9,8,7])])],
        treatment=[ident],control=[],scope=exp['scope'],criterion=exp['criterion'])
    original=c.execute(c.plan('author','frozen-author-comparison',exp))
    return c,r,ledger,executor,ident,protocol_id,original


def run_heldouts(c,r,method_id,protocol_id):
    heldout=[]
    for assignment in r.get_artifact('holdout_protocol',protocol_id)['assignments']:
        plan_id=c.plan(assignment['reviewer_lab'],'assigned-'+assignment['id'],
            experiment([method_id],inputs=[case['input'] for case in assignment['cases']]),
            purpose='heldout',selected=[method_id],heldout_protocol_id=protocol_id,
            heldout_assignment_id=assignment['id'])
        heldout.append(c.execute(plan_id))
    return heldout


def test_real_coordinator_receipts_measurements_and_budget_establish_complete_cohort(tmp_path):
    c,r,ledger,executor,m,p,original=assigned_campaign(tmp_path)
    heldout=run_heldouts(c,r,m,p)
    evidence_id=c.assess_benefit(m,original,heldout,protocol_id=p)
    evidence=r.get_artifact('method_evidence',evidence_id)
    assert evidence['level']=='benefit_replicated'
    assert evidence['supporting']==[original]+heldout
    assert [row['mean_effect'] for row in evidence['heldout_comparisons']]==[1,1]
    assert len(executor.calls)==3
    assert len({invocation for _,invocation in executor.calls})==3
    receipts=[r.get_artifact('call_receipt',event['data']['id']) for event in r.events()
              if event['kind']=='artifact_registered' and event['data']['kind']=='call_receipt']
    actual=[receipt for receipt in receipts if receipt['category']=='execution']
    measured=[receipt for receipt in receipts if receipt['category']=='measurement']
    assert len(actual)==3 and len(measured)==6
    assert all(receipt['result']['dispatch_status']=='invoked' for receipt in actual)
    assert all(receipt['status']=='completed' for receipt in actual+measured)
    for name in ('author','review1','review2'):
        account=ledger.snapshot('account-'+name)
        assert float(account['spent']['cpu_seconds'])>0 and not account['halted']
        assert float(account['spent']['dollars'])==0
    spent=ledger.snapshot()['spent']
    assert c.assess_benefit(m,original,heldout,protocol_id=p)==evidence_id
    assert ledger.snapshot()['spent']==spent and len(executor.calls)==3
    # Known cache reads do not become another held-out observation.
    with pytest.raises(Refusal,match='duplicate'):
        c.assess_benefit(m,original,[heldout[0],heldout[0]],protocol_id=p)
    assert not ledger.snapshot('account-author')['halted']


def test_assigned_verification_does_not_authorize_foreign_persistent_adoption(tmp_path):
    c,r,ledger,executor,m,p,original=assigned_campaign(tmp_path)
    heldout=run_heldouts(c,r,m,p)
    for name,trial,diagnostic in [('review1',heldout[0],'does not authorize'),
                                  ('review2',heldout[1],'outside this revision')]:
        previous=r.lab(name)['active_revision']
        with pytest.raises(Refusal,match=diagnostic):
            c.adopt(name,m,trial,expected_revision=previous)
        assert r.lab(name)['active_revision']==previous
        assert m not in r.revision(previous)['active_methods']
        assert not ledger.snapshot('account-'+name)['halted']
    # Release permits C adoption, but independent B remains isolated.
    replay=c.replay(original,'replay',assignment_id='assigned-reproducibility-check')
    c.release(m,original,replay)
    previous=r.lab('review1')['active_revision']
    with pytest.raises(Refusal,match='does not authorize'):
        c.adopt('review1',m,heldout[0],expected_revision=previous)
    previous=r.lab('review2')['active_revision']
    # Refused intents stay terminal/idempotent. A new recipient trial is a new
    # explicit action after authorization changes; it is not a retried charge.
    with pytest.raises(Refusal,match='outside this revision'):
        c.adopt('review2',m,heldout[1],expected_revision=previous)
    recipient_trial=c.execute(c.plan('review2','new-authorized-recipient-trial',
        experiment([m],inputs=[[40,20,30]]),purpose='transfer',selected=[m]))
    adopted=c.adopt('review2',m,recipient_trial,expected_revision=previous)
    current=r.get_artifact('adoption',adopted)['new_revision']
    assert current!=previous and m in r.revision(current)['active_methods']


def test_solution_only_cannot_adopt_foreign_method_even_after_assigned_use_and_release(tmp_path):
    c,r,ledger,executor,m,p,original=assigned_campaign(tmp_path,transfer_policy='solution_only')
    heldout=run_heldouts(c,r,m,p)
    replay=c.replay(original,'replay',assignment_id='assigned-reproducibility-check')
    c.release(m,original,replay)
    previous=r.lab('review2')['active_revision']
    with pytest.raises(Refusal,match='does not authorize'):
        c.adopt('review2',m,heldout[1],expected_revision=previous)
    assert r.lab('review2')['active_revision']==previous


def test_free_origin_label_or_altered_assignment_refused_without_method_dispatch(tmp_path):
    c,r,ledger,executor,m,p,original=assigned_campaign(tmp_path)
    before=len(executor.calls)
    with pytest.raises(Refusal,match='preregistered complete cohort'):
        c.plan('review1','free-label',experiment([m],inputs=[[6,4,5]]),purpose='heldout',
               selected=[m],input_origin='independent_heldout')
    assignment=r.get_artifact('holdout_protocol',p)['assignments'][0]
    with pytest.raises(Refusal,match='changed assigned inputs'):
        c.plan('review1','changed-case',experiment([m],inputs=[[16,4,5]]),purpose='heldout',
            selected=[m],heldout_protocol_id=p,heldout_assignment_id=assignment['id'])
    assert len(executor.calls)==before
    assert not ledger.snapshot('account-review1')['halted']
    heldout=run_heldouts(c,r,m,p)
    with pytest.raises(Refusal,match='all assigned reviewers'):
        c.assess_benefit(m,original,heldout[:1],protocol_id=p)
    assert r.get_artifact('method_evidence',c.assess_benefit(m,original,heldout,protocol_id=p))['level']=='benefit_replicated'


@pytest.mark.parametrize('location,field', [
    ('experiment','treatment'), ('experiment','control'), ('experiment','criterion'),
    ('experiment','inputs'), ('method','dependencies'), ('method','parents'),
    ('method','resources'), ('solution','inputs'),
])
def test_malformed_generated_contract_is_known_refusal_and_next_step_can_repair(tmp_path,location,field):
    if location=='method':
        bundle=method()
        bundle['manifest'][field]=None
        malformed=dict(kind='method',files=bundle['files'],manifest=bundle['manifest'])
    else:
        value=experiment(['$new'])
        value[field]=None
        malformed=dict(kind=location,experiment=value)
        if location=='solution':
            malformed['code']='def solve(values):\n    return sorted(values)\n'
    proposer=FixedProposer([malformed,dict(kind='stop',reason='Technical malformed-input repair fixture complete')])
    c,r,ledger,executor,_=setup(tmp_path,proposer=proposer)
    lab(c,'author')
    first=c.step('author')
    assert first['status']=='continued'
    assert proposer.calls==1 and executor.calls==[]
    assert ledger.snapshot('account-author')['unknown_reservations']==[]
    second=c.step('author')
    assert second['status']=='stopped' and second['index']==1 and proposer.calls==2
    assert float(ledger.snapshot('account-author')['spent']['dollars'])==.02
    assert len(c._events('episode_step'))==2
