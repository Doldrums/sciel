"""Lifecycle/funding/delivery regressions with explicit nonexecuting fixtures."""
from copy import deepcopy

import pytest

from sciel.methods.budget import BudgetExceeded, FundingPolicy
from sciel.methods.coordinator import Refusal
from sciel.methods.models import content_hash
from sciel.methods.tasks import SortTask
from test_methods_coordinator import setup, lab, method, experiment
from test_methods_solution_integration import setup as solution_setup, proposal


def register_draft(coordinator, bundle, label):
    return coordinator.register_draft('author',label,bundle['files'],bundle['manifest'])


def test_archived_lab_reads_completed_plan_but_cannot_dispatch_preplanned_work(tmp_path):
    c,r,ledger,executor,_=setup(tmp_path)
    lab(c,'author')
    m=register_draft(c,method(),'draft')
    completed_plan=c.plan('author','completed',experiment([m]))
    completed=c.execute(completed_plan)
    pending_plan=c.plan('author','not-yet-executed',experiment([m],inputs=[[7,5,6]]))
    c.archive_lab('author',reason='Explicit operator archive fixture')
    with pytest.raises(BudgetExceeded):
        c.step('author')
    before=ledger.snapshot()['spent']
    assert c.execute(completed_plan)==completed
    assert ledger.snapshot()['spent']==before and len(executor.calls)==1
    with pytest.raises((Refusal,BudgetExceeded),match='archiv'):
        c.execute(pending_plan)
    assert len(executor.calls)==1
    assert not ledger.snapshot('account-author')['unknown_reservations']


def test_funding_blocks_shrink_persist_and_do_not_reset_for_replication_or_new_version(tmp_path):
    c,r,ledger,executor,_=setup(tmp_path)
    lab(c,'author'); lab(c,'reviewer')
    # Host policy injection is explicit and durable, not monkeypatching outcomes.
    c._save_funding(FundingPolicy(initial_quota=2))
    first=register_draft(c,method(),'first-version')
    completed=[]
    for index in range(4):
        pid=c.plan('author','repeat-'+str(index),experiment([first]))
        completed.append(c.execute(pid))
    policy=c._funding().snapshot('lab:author')
    assert policy['next_quota']==2 and policy['no_information_streak']==1
    before=c._funding().state()
    replay=c.replay(completed[0],'reviewer',assignment_id='same-observation-independent-replay')
    assert r.get_artifact('experiment',replay)['contract_passed']
    assert c._funding().state()==before
    assert not [row for row in c._events('trial_reserved') if row['lab_id']=='reviewer']
    updated=method(name='new-source-version',code='def run(values):\n    return sorted(values) # new immutable bytes\n')
    updated['manifest']['parents']=[first]
    newer=register_draft(c,updated,'next-version-same-funding')
    assert newer!=first and c._funding().snapshot('lab:author')==policy
    for index in range(4,6):
        c.execute(c.plan('author','repeat-'+str(index),experiment([newer])))
    assert c._funding().snapshot('lab:author')['next_quota']==1
    assert [row['informative'] for row in c._events('trial_block')]==[True,False,False]
    restored,rr,ll,restarted_executor,_=setup(tmp_path)
    assert restored._funding().snapshot('lab:author')['next_quota']==1
    assert restored._funding().snapshot('lab:author')['no_information_streak']==0
    for index in range(6,8):
        restored.execute(restored.plan('author','repeat-'+str(index),experiment([newer])))
    assert restored._funding().snapshot('lab:author')['next_quota']==0
    assert restored.lab_state('author')['status']=='dormant'
    another=method(name='fresh-root-hash-cannot-mint-quota')
    last=register_draft(restored,another,'no-parent-still-same-lab')
    assert last!=newer and restored._funding().snapshot('lab:author')['next_quota']==0
    pending=restored.plan('author','dormant-plan',experiment([last]))
    with pytest.raises(BudgetExceeded):
        restored.execute(pending)
    assert len(restarted_executor.calls)==2
    assert not ll.snapshot('account-author')['unknown_reservations']


def test_duplicate_feedback_delivers_bound_prior_measurement_outside_recent_windows(tmp_path):
    task=SortTask()
    codes=['def solve(values):\n    marker = '+str(i)+'\n    return sorted(values)\n' for i in range(6)]
    responses=[proposal(task,code) for code in codes]
    responses.append(proposal(task,codes[0]+'# repeated first source\n'))
    c,r,ledger,executor,proposer=solution_setup(tmp_path,task,responses)
    accepted=[c.step('author') for _ in range(6)]
    first=r.get_artifact('solution',accepted[0]['solution_id'])
    repeated=c.step('author')
    assert repeated['novelty']['match_kind']=='ast'
    assert repeated['novelty']['matched_call_receipt_id']==first['call_receipt_id']
    current=r.lab('author')['active_revision']
    context=c._context('author',current,r.revision(current))
    assert accepted[0]['solution_id'] not in {row['id'] for row in context['previous_solutions']}
    assert accepted[0]['experiment_id'] not in {row['id'] for row in context['evidence']}
    diagnostics=[row['body']['novelty'] for row in context['evidence']
                 if isinstance(row['body'].get('novelty'),dict)]
    assert len(diagnostics)==1
    observation=diagnostics[0].get('matched_observation')
    assert observation is not None, 'A receipt ID alone does not deliver the prior evaluator result.'
    assert observation['call_receipt_id']==first['call_receipt_id']
    assert observation['score']==first['evaluation']['score']
    assert observation['cases']==first['evaluation']['cases']
    assert observation['status']=='ok'
    assert len(executor.calls)==6 and len(proposer.calls)==7
    assert not ledger.snapshot('account')['unknown_reservations']
