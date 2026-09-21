"""Explicit foreign dependencies authorize a bounded C trial, never adoption.

The external executor below returns fixed data; no method source is run here.
"""
from copy import deepcopy

import pytest

from sciel.methods.coordinator import Coordinator, Refusal
from sciel.methods.models import content_hash
from sciel.methods.tasks import SortTask
from test_methods_coordinator import (BUDGET, RESOURCES, FixedProposer, ScriptedExecutor,
    experiment, lab, method, register, release_fixture, setup)


class DependencyRecordingExecutor(ScriptedExecutor):
    def __init__(self):
        super().__init__()
        self.dependency_calls=[]
    def execute(self,bundle,value,**kwargs):
        self.dependency_calls.append([b['id'] for b in kwargs.get('dependency_bundles',())])
        return super().execute(bundle,value,**kwargs)


def wrapper(dependency,*,kind='adapter'):
    value=method(author='recipient',name='explicit-selected-dependency',deps=[dependency],
                 code='import m_'+dependency+'\ndef run(values):\n    return m_'+dependency+'.invoke(values)\n')
    value['manifest']['kind']=kind
    value['id']=content_hash(dict(schema='sciel-method/1',manifest=value['manifest'],files=value['files']))
    return value


@pytest.mark.parametrize('kind',['tool','adapter'])
def test_c_can_trial_new_wrapper_with_explicit_released_foreign_dependency_without_adoption(tmp_path,kind):
    foreign=method(author='foreign')
    own=wrapper(foreign['id'],kind=kind)
    proposer=FixedProposer([dict(kind='method',files=own['files'],manifest=own['manifest'],
                               experiment=experiment(['$new']))])
    executor=DependencyRecordingExecutor()
    c,r,ledger,_,_=setup(tmp_path,executor=executor,proposer=proposer)
    lab(c,'recipient','C')
    register(r,foreign); release_fixture(r,foreign['id'])
    before=r.revision(r.lab('recipient')['active_revision'])
    result=c.step('recipient')
    assert 'experiment_id' in result,result
    observed=r.get_artifact('experiment',result['experiment_id'])
    assert observed['contract_passed'] and observed['invoked_methods']==[own['id']]
    assert executor.dependency_calls==[[foreign['id']]]
    assert [ident for ident,_ in executor.calls]==[own['id']]
    assert foreign['id'] not in observed['invoked_methods']  # nested inclusion is not independent credit
    after=r.revision(r.lab('recipient')['active_revision'])
    assert after['active_methods']==before['active_methods']==[]
    assert after['active_adapters']==before['active_adapters']==[]
    assert r.method_status(own['id'])['status']=='draft'
    assert float(ledger.snapshot('account-recipient')['spent']['dollars'])==.01
    assert not c._events('adoption')


@pytest.mark.parametrize('mode',['A','B'])
def test_a_b_cannot_extend_toolbox_through_declared_foreign_dependency(tmp_path,mode):
    c,r,_,executor,_=setup(tmp_path)
    lab(c,'recipient',mode)
    foreign=register(r,method(author='foreign'))
    release_fixture(r,foreign)
    own=wrapper(foreign)
    with pytest.raises(Refusal):
        c.register_draft('recipient','explicit-foreign',own['files'],own['manifest'])
    assert executor.calls==[]
    assert not [e for e in r.events() if e['kind']=='method_registered' and e['data']['id']==own['id']]


@pytest.mark.parametrize('status',['draft','quarantined','superseded'])
def test_c_cannot_trial_unpublished_or_disabled_foreign_dependency(tmp_path,status):
    c,r,_,executor,_=setup(tmp_path)
    lab(c,'recipient','C')
    foreign=register(r,method(author='foreign'))
    if status!='draft':
        release_fixture(r,foreign)
        r.status_method(foreign,status)
    own=wrapper(foreign)
    with pytest.raises(Refusal):
        c.register_draft('recipient','bad-dependency',own['files'],own['manifest'])
    assert executor.calls==[]


@pytest.mark.parametrize('violation',['runtime','resources','transitive_resources'])
def test_explicit_dependency_does_not_bypass_runtime_or_resource_checks(tmp_path,violation):
    c,r,_,executor,_=setup(tmp_path)
    lab(c,'recipient','C')
    changes={}
    if violation=='runtime':
        changes['image']='sha256:'+'b'*64
    else:
        changes['resources']={**RESOURCES,'cpu_seconds':RESOURCES['cpu_seconds']+1}
    foreign=register(r,method(author='foreign',**changes))
    release_fixture(r,foreign)
    if violation=='transitive_resources':
        foreign=register(r,method(author='foreign',deps=[foreign],name='foreign-wrapper'))
        release_fixture(r,foreign)
    own=wrapper(foreign)
    with pytest.raises(Refusal):
        ident=c.register_draft('recipient','invalid-dependency',own['files'],own['manifest'])
        c.plan('recipient','invalid-trial',experiment([ident]))
    assert executor.calls==[]
    revision=r.revision(r.lab('recipient')['active_revision'])
    assert not revision['active_methods'] and not revision['active_adapters']


def test_undeclared_foreign_root_is_not_authorized_by_some_other_wrapper(tmp_path):
    c,r,_,executor,_=setup(tmp_path)
    lab(c,'recipient','C')
    selected=register(r,method(author='foreign',name='selected'))
    other=register(r,method(author='foreign',name='not-selected'))
    for ident in (selected,other): release_fixture(r,ident)
    own=wrapper(selected)
    ident=c.register_draft('recipient','wrapper',own['files'],own['manifest'])
    with pytest.raises(Refusal):
        c.plan('recipient','smuggled-root',experiment([ident,other]))
    assert executor.calls==[]


def test_solution_only_control_cannot_receive_foreign_method_via_dependency(tmp_path):
    c,r,ledger,executor,proposer=setup(tmp_path)
    # Constructing a separate fresh coordinator is necessary: transfer policy is identity-bound.
    from sciel.methods.registry import Registry
    from sciel.methods.budget import ResourceLedger
    r=Registry(tmp_path/'solution-registry')
    ledger=ResourceLedger(tmp_path/'solution-budget.jsonl',BUDGET)
    c=Coordinator(r,ledger,executor,proposer,SortTask(),resources=RESOURCES,transfer_policy='solution_only')
    lab(c,'recipient','C')
    foreign=register(r,method(author='foreign')); release_fixture(r,foreign)
    own=wrapper(foreign)
    with pytest.raises(Refusal):
        c.register_draft('recipient','control-method-leak',own['files'],own['manifest'])
    assert executor.calls==[]
