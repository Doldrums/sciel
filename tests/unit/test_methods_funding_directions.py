"""Explicit direction grants share money but retain separate trial streaks."""
import pytest

from sciel.methods.budget import BudgetExceeded, FundingPolicy
from sciel.methods.coordinator import Refusal
from test_methods_coordinator import setup, lab, method, experiment


def draft(c,name,*,parents=(),code=None):
    bundle=method(name=name,code=code)
    bundle['manifest']['parents']=list(parents)
    return c.register_draft('author','draft-'+name,bundle['files'],bundle['manifest'])


def initialize(tmp_path):
    c,r,ledger,executor,proposer=setup(tmp_path)
    lab(c,'author')
    c._save_funding(FundingPolicy(initial_quota=2,no_information_blocks=1))
    evidence=r.put_artifact('technical_evidence',{'scope':'synthetic direction-grant fixture'})
    return c,r,ledger,executor,evidence


def grant(c,r,evidence,question='A separately funded research question'):
    ident=c.grant_direction('author',question,[evidence])
    return ident,r.get_artifact('funding_grant',ident)['lineage_id']


def run(c,method_id,key):
    return c.execute(c.plan('author',key,experiment([method_id])))


def test_two_directions_keep_separate_streaks_old_plans_and_shared_budget(tmp_path):
    c,r,ledger,executor,evidence=initialize(tmp_path)
    a=draft(c,'initial-direction')
    old_plan=c.plan('author','old-plan',experiment([a]))
    old_revision=r.lab('author')['active_revision']
    assert r.get_artifact('experiment_plan',old_plan)['funding_lineage']=='lab:author'
    account_before=ledger.snapshot('account-author')['limits']
    ident,b_lineage=grant(c,r,evidence)
    active_revision=r.lab('author')['active_revision']
    assert active_revision!=old_revision
    assert r.revision(active_revision)['capabilities']['active_funding_lineage']==b_lineage
    body=r.get_artifact('funding_grant',ident)
    assert body['budget_account']=='account-author' and body['initial_quota']==2
    assert ledger.snapshot('account-author')['limits']==account_before
    b=draft(c,'independent-direction')
    assert c.lineage_for_version(a)=='lab:author'
    assert c.lineage_for_version(b)==b_lineage
    c.execute(old_plan)
    run(c,a,'a-1')  # First A block is informative.
    run(c,b,'b-0'); run(c,b,'b-1')
    b_before=c._funding().snapshot(b_lineage)
    run(c,a,'a-2'); run(c,a,'a-3')
    assert c._funding().snapshot('lab:author')['next_quota']==1
    assert c._funding().snapshot(b_lineage)==b_before
    # Identical observations shrink only A; its old pinned plans cannot charge B.
    run(c,a,'a-4')
    assert c._funding().snapshot('lab:author')['next_quota']==0
    assert c.lab_state('author')['status']=='active'
    assert c._funding().snapshot(b_lineage)['next_quota']==2
    assert {row['lineage_id'] for row in c._events('trial_reserved')}=={'lab:author',b_lineage}
    assert all(row['lineage_id'] in ('lab:author',b_lineage) for row in c._events('trial_block'))
    with pytest.raises(BudgetExceeded):
        run(c,a,'a-dormant')
    calls=len(executor.calls)
    restored,rr,ll,restarted_executor,_=setup(tmp_path)
    assert restored._active_funding_lineage('author')==b_lineage
    assert restored.lineage_for_version(a)=='lab:author'
    assert restored._funding().snapshot('lab:author')['next_quota']==0
    assert restored.execute(old_plan)==c.execute(old_plan)
    assert len(executor.calls)==calls and restarted_executor.calls==[]
    run(restored,b,'b-2'); run(restored,b,'b-3')
    run(restored,b,'b-4')
    assert restored.lab_state('author')['status']=='dormant'
    restored.renew_trial('author',1,question='Explicit bounded retry',evidence_refs=[evidence])
    assert restored._funding().snapshot(b_lineage)['next_quota']==1
    assert restored._funding().snapshot('lab:author')['next_quota']==0
    assert restored.lab_state('author')['status']=='trial'


def test_parent_lineage_inheritance_and_idempotent_grant_cannot_reset_quota(tmp_path):
    c,r,ledger,executor,evidence=initialize(tmp_path)
    a=draft(c,'initial')
    old_revision=r.lab('author')['active_revision']
    first,b=grant(c,r,evidence)
    first_revision=r.lab('author')['active_revision']
    child=draft(c,'child-of-original',parents=[a],code='def run(values):\n    return sorted(values) # child\n')
    root=draft(c,'new-root')
    assert c.lineage_for_version(child)=='lab:author' and c.lineage_for_version(root)==b
    assert r.get_artifact('experiment_plan',c.plan('author','child-plan',experiment([child])))['funding_lineage']=='lab:author'
    # Existing bytes/manifests keep their original binding after activation changes.
    same=method(name='initial')
    assert c.register_draft('author','same-version-again',same['files'],same['manifest'])==a
    assert c.lineage_for_version(a)=='lab:author'
    second,c_lineage=grant(c,r,evidence,'A third explicitly authorized question')
    second_revision=r.lab('author')['active_revision']
    spent=ledger.snapshot()['spent']
    assert c.grant_direction('author','A separately funded research question',[evidence])==first
    assert ledger.snapshot()['spent']==spent
    assert r.lab('author')['active_revision']==second_revision
    assert c._active_funding_lineage('author')==c_lineage
    assert len(c._events('funding_direction_granted'))==2
    assert old_revision!=first_revision!=second_revision
    assert c._funding().snapshot(b)['next_quota']==2


def test_mixed_parent_composition_inherits_first_parent_and_pipeline_pins_active_grant(tmp_path):
    c,r,ledger,executor,evidence=initialize(tmp_path)
    a=draft(c,'one')
    _,b=grant(c,r,evidence)
    second=draft(c,'two')
    composed=draft(c,'composed-child',parents=[a,second])
    reversed_parents=draft(c,'other-composition-order',parents=[second,a])
    assert c.lineage_for_version(composed)=='lab:author'
    assert c.lineage_for_version(reversed_parents)==b
    pipeline=c.plan('author','mixed-pipeline',experiment([a,second]))
    assert r.get_artifact('experiment_plan',pipeline)['funding_lineage']==b
    _,third=grant(c,r,evidence,'Another active question after composition planning')
    assert third!=b
    c.execute(pipeline)
    assert c._events('trial_reserved')[-1]['lineage_id']==b
    assert len(executor.calls)==2 and ledger.snapshot('account-author')['unknown_reservations']==[]


def test_unrelated_new_hash_uses_existing_active_grant_and_fork_cannot_inherit_it(tmp_path):
    c,r,ledger,executor,evidence=initialize(tmp_path)
    _,lineage=grant(c,r,evidence)
    first=draft(c,'one')
    second=draft(c,'renamed')
    assert first!=second
    assert c.lineage_for_version(first)==c.lineage_for_version(second)==lineage
    before=c._funding().state()
    revision=r.lab('author')['active_revision']
    c.create_lab('fork','Explicit fork','account-fork',{},mode='B',parent_revision=revision)
    child_revision=r.revision(r.lab('fork')['active_revision'])
    assert 'active_funding_lineage' not in child_revision['capabilities']
    assert c._active_funding_lineage('fork')=='lab:fork'
    assert c._funding().state()==before
    assert r.lab('fork')['budget_account']!='account-author'


@pytest.mark.parametrize('question,refs', [('',None),('why',[]),('why',['a'*64]),('why',['a'*64,'a'*64])])
def test_grants_require_explicit_question_and_registered_evidence(tmp_path,question,refs):
    c,r,ledger,executor,evidence=initialize(tmp_path)
    previous=r.lab('author')['active_revision']
    with pytest.raises(Refusal):
        c.grant_direction('author',question,refs)
    assert r.lab('author')['active_revision']==previous
    assert not c._events('funding_direction_granted')
    assert not ledger.snapshot('account-author')['unknown_reservations']
