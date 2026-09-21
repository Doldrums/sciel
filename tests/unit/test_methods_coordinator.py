"""Coordinator authorization/accounting fixtures; scripted executor is explicit.

The opt-in final test uses the real Docker executor. No generated source ever
runs in this test process, and the proposer is always a labelled fixed fixture.
"""
from copy import deepcopy
import json
import os
from types import SimpleNamespace

import pytest

from sciel.methods.budget import BudgetUnknown, ResourceLedger
from sciel.methods.coordinator import Coordinator, Refusal
from sciel.methods.executor import DockerExecutor, ExecutionRefused
from sciel.methods.models import content_hash
from sciel.methods.registry import Registry
from sciel.methods.tasks import SortTask


IMAGE = 'sha256:' + 'a' * 64
RESOURCES = dict(wall_seconds=4, cpu_seconds=1, memory_bytes=128*1024*1024,
                 pids=32, max_output_bytes=8192, max_stderr_bytes=2048)
BUDGET = dict(dollars=100, cpu_seconds=10000, gpu_seconds=0, wall_seconds=100000,
              storage_bytes=100_000_000, steps=100)


def method(author='author', *, code=None, deps=(), image=IMAGE, resources=None, name='fixture'):
    task = SortTask()
    manifest = dict(name=name, kind='tool', author_lab=author, entrypoint='tool.py:run',
        input_schema=task.input_schema, output_schema=task.output_schema,
        protocol='Technical fixture; no scientific claim.', runtime_lock={'image_id':image},
        dependencies=list(deps), parents=[], resources=deepcopy(resources or RESOURCES),
        scope='Assigned sorting inputs only.', examples=[])
    files = {'tool.py': code or 'def run(values):\n    return sorted(values)\n'}
    return dict(id=content_hash(dict(schema='sciel-method/1', manifest=manifest, files=files)),
                manifest=manifest, files=files)


class ScriptedExecutor:
    """Returns prescribed outcomes; does not execute or interpret source."""
    identity = {'schema':'scripted-executor-fixture/1','image_id':IMAGE}
    overhead_wall_bound = 12

    def __init__(self):
        self.calls = []
        self.behavior = {}

    def execute(self, bundle, value, *, invocation_id, plan_id, reservation_id, limits, dependency_bundles=()):
        if limits['pids'] < 2:
            raise ExecutionRefused('unsupported process allocation')
        self.calls.append((bundle['id'], invocation_id))
        behavior = self.behavior.get(bundle['id'], 'ok')
        status = 'failed' if behavior == 'failed' else 'unknown' if behavior == 'unknown' else 'ok'
        output = list(value) if behavior == 'identity' else sorted(value) if status == 'ok' else None
        return dict(schema='scripted-receipt-fixture/1', status=status,
            reason='scripted_'+status, dispatch_status='invoked',
            method_id=bundle['id'], invocation_id=invocation_id, plan_id=plan_id,
            reservation_id=reservation_id, input_sha256=content_hash(value),
            source_sha256=content_hash(bundle['files']), runtime_sha256=content_hash(self.identity),
            output=output, output_sha256=content_hash(output) if output is not None else None,
            resources={'cpu_seconds':None if status=='unknown' else .01, 'wall_seconds':.02}, stderr='fixture diagnostic')


class FixedProposer:
    identity = {'schema':'fixed-proposer-fixture/1', 'label':'technical repair fixture, not model generation'}

    def __init__(self, proposals=()):
        self.proposals = list(proposals)
        self.calls = 0

    def bounds(self, prompt):
        return dict(dollars=1, cpu_seconds=1, gpu_seconds=0, wall_seconds=1, storage_bytes=1_000_000, steps=1)

    def generate(self, prompt):
        proposal = self.proposals[self.calls]
        self.calls += 1
        return SimpleNamespace(text=json.dumps(proposal), model='fixed-fixture', cost=.01,
                               prompt_tokens=1, completion_tokens=1)


def setup(tmp_path, *, executor=None, proposer=None, common_methods=(), capacity=None):
    registry = Registry(tmp_path/'registry')
    ledger = ResourceLedger(tmp_path/'budget.jsonl', BUDGET)
    execution = executor or ScriptedExecutor()
    proposer = proposer or FixedProposer()
    coordinator = Coordinator(registry, ledger, execution, proposer, SortTask(), resources=RESOURCES,
                              common_methods=common_methods, max_steps=8, capacity=capacity)
    return coordinator, registry, ledger, execution, proposer


def lab(coordinator, name, mode='B'):
    return coordinator.create_lab(name, 'Technical sorting experiment.', 'account-'+name, {}, mode=mode)


def register(registry, value):
    return registry.register(value['files'], value['manifest'])


def experiment(treatment, *, control=(), inputs=None):
    return dict(question='Does the explicit method improve this assigned case?',
        treatment=list(treatment), control=list(control), inputs=inputs or [[3,1,2]],
        criterion=dict(metric='correct', direction='max', minimum_effect=0), scope='Assigned fixture inputs only.')


def release_fixture(registry, ident):
    proof = registry.put_artifact('method_evidence', dict(method_id=ident, label='authorization fixture only'))
    registry.status_method(ident, 'released', [proof])


def test_known_execution_receipt_reused_after_resume_without_second_call_or_charge(tmp_path):
    c, registry, ledger, execution, proposer = setup(tmp_path)
    lab(c, 'author')
    ident = register(registry, method())
    plan = c.plan('author', 'proposal-1', experiment([ident]))
    result = c.execute(plan)
    spent = deepcopy(ledger.snapshot()['spent'])
    assert len(execution.calls) == 1
    registry.status_method(ident,'quarantined')  # Historical receipts remain readable.
    resumed = Coordinator(Registry(registry.root), ResourceLedger(ledger.path, BUDGET), execution, proposer,
                          SortTask(), resources=RESOURCES, max_steps=8)
    assert resumed.execute(plan) == result
    assert len(execution.calls) == 1
    assert ledger.snapshot()['spent'] == spent
    evidence = registry.get_artifact('experiment', result)
    assert evidence['invoked_methods'] == [ident]
    assert evidence['mean_effect'] == 1


@pytest.mark.parametrize('mode', ['A','B','C'])
def test_foreign_unselected_method_is_not_local_authority(tmp_path, mode):
    c, registry, _, execution, _ = setup(tmp_path)
    lab(c, 'recipient', mode)
    ident = register(registry, method(author='foreign'))
    release_fixture(registry, ident)
    with pytest.raises(Refusal, match='outside'):
        c.plan('recipient', 'proposal-foreign', experiment([ident]))
    assert execution.calls == []


def test_a_cannot_register_permanent_method_and_bad_input_is_repairable_refusal(tmp_path):
    c, registry, ledger, execution, _ = setup(tmp_path)
    lab(c, 'author', 'A')
    value = method()
    with pytest.raises(Refusal): c.register_draft('author','p1',value['files'],value['manifest'])
    assert execution.calls == []
    # Another authorized lab may continue after this known validation failure.
    lab(c, 'other', 'B')
    ident = register(registry, method(author='other'))
    with pytest.raises(Refusal): c.plan('other','p2',experiment([ident],inputs=[True]))
    ledger.reserve('account-other','known-refusal-does-not-block',{'steps':1},'fixture')


def test_known_predispatch_refusal_and_quarantine_do_not_consume_invocation_credit(tmp_path):
    c, registry, ledger, execution, _ = setup(tmp_path)
    lab(c, 'author')
    value = method(resources={**RESOURCES,'pids':1})
    ident = register(registry,value)
    plan = c.plan('author','small-allocation',experiment([ident]))
    result = registry.get_artifact('experiment',c.execute(plan))
    assert not result['contract_passed'] and result['invoked_methods'] == []
    assert execution.calls == []
    valid = register(registry,method(name='valid'))
    plan2 = c.plan('author','later-quarantine',experiment([valid]))
    registry.status_method(valid,'quarantined')
    result2 = registry.get_artifact('experiment',c.execute(plan2))
    assert not result2['contract_passed'] and result2['invoked_methods'] == []
    assert execution.calls == []
    # Unknown holds would block this fresh reservation.
    ledger.reserve('account-author','proof-account-still-open',{'steps':1},'fixture')


def test_common_tool_cannot_expand_frozen_plan_resource_contract(tmp_path):
    registry=Registry(tmp_path/'registry')
    ident=register(registry,method(author='common',resources={**RESOURCES,'cpu_seconds':100}))
    c,_,_,execution,_=setup(tmp_path,common_methods=[ident])
    lab(c,'author','A')
    with pytest.raises(Refusal,match='resource'):
        c.plan('author','oversized-common-tool',experiment([ident]))
    assert execution.calls==[]


def test_real_host_capacity_kind_and_known_capacity_refusal(tmp_path):
    from sciel.engine.capacity import HostCapacity, CapacityTimeout
    capacity=HostCapacity(tmp_path/'private-capacity',llm_slots=1,evaluator_slots=1)
    c,registry,ledger,execution,_=setup(tmp_path,capacity=capacity)
    lab(c,'author')
    ident=register(registry,method())
    result=c.execute(c.plan('author','available-capacity',experiment([ident])))
    assert registry.get_artifact('experiment',result)['contract_passed']
    assert len(execution.calls)==1
    class TimedOutCapacity:
        def acquire(self, kind, **kwargs):
            assert kind=='evaluator'
            raise CapacityTimeout(0)
    c.capacity=TimedOutCapacity()
    result=c.execute(c.plan('author','capacity-timeout',experiment([ident])))
    assert not registry.get_artifact('experiment',result)['contract_passed']
    assert len(execution.calls)==1
    ledger.reserve('account-author','capacity-failure-does-not-block',{'steps':1},'fixture')


def test_generation_capacity_refusal_is_known_and_next_step_can_proceed(tmp_path):
    from sciel.engine.capacity import CapacityTimeout
    proposer=FixedProposer([dict(kind='stop',reason='technical fixture complete')])
    c,registry,ledger,_,_=setup(tmp_path,proposer=proposer)
    lab(c,'author')
    class TimedOutCapacity:
        def acquire(self, kind, **kwargs):
            assert kind=='llm'
            raise CapacityTimeout(0)
    c.capacity=TimedOutCapacity()
    first=c.step('author')
    assert first['status']!='stopped' and proposer.calls==0
    assert float(ledger.snapshot('account-author')['spent']['dollars'])==0
    c.capacity=None
    second=c.step('author')
    assert second['status']=='stopped' and proposer.calls==1
    assert second['index']==first['index']+1


def test_malformed_action_feedback_reaches_next_paid_step_without_hidden_retry(tmp_path):
    class RecordingProposer(FixedProposer):
        def generate(self, prompt):
            self.prompts = getattr(self, 'prompts', []) + [prompt]
            return super().generate(prompt)

    proposer = RecordingProposer([
        {'method': {'kind': 'tool', 'files': {}, 'manifest': {}}},
        {'kind': 'stop', 'reason': 'Format correction fixture complete'},
    ])
    c, registry, ledger, execution, _ = setup(tmp_path, proposer=proposer)
    lab(c, 'author')
    first = c.step('author')
    receipt = registry.get_artifact('proposal_receipt', first['proposal_id'])
    assert receipt['error'] == 'invalid_proposal_format'
    assert receipt['diagnostic']['code'] == 'missing_action_kind'
    assert proposer.calls == 1 and execution.calls == []
    assert ledger.snapshot()['spent']['steps'] == 1
    assert float(ledger.snapshot()['spent']['dollars']) == pytest.approx(.01)
    second = c.step('author')
    context = json.loads(proposer.prompts[-1].split('\nCONTEXT_JSON\n', 1)[1])
    assert context['protocol_contract']['max_cases'] == c.max_cases
    assert context['protocol_contract']['solution']['input_schema'] == c.solution_task.input_schema
    feedback = next(item for item in context['evidence'] if item['id'] == first['proposal_id'])
    assert feedback['body']['diagnostic'] == receipt['diagnostic']
    assert second['status'] == 'stopped' and proposer.calls == 2
    assert ledger.snapshot()['spent']['steps'] == 2
    assert float(ledger.snapshot()['spent']['dollars']) == pytest.approx(.02)


def test_unknown_execution_is_not_replayed_and_holds_resource_reservation(tmp_path):
    c, registry, ledger, execution, _ = setup(tmp_path)
    lab(c,'author')
    ident = register(registry,method())
    execution.behavior[ident]='unknown'
    plan = c.plan('author','unknown',experiment([ident]))
    with pytest.raises(BudgetUnknown): c.execute(plan)
    with pytest.raises(BudgetUnknown): c.execute(plan)
    assert len(execution.calls)==1
    assert float(ledger.snapshot('account-author')['reserved']['cpu_seconds'])>0
    with pytest.raises(BudgetUnknown): ledger.reserve('account-author','new-work',{'steps':1},'fixture')


def test_independent_replay_release_transfer_and_subsequent_actual_use(tmp_path):
    c, registry, _, execution, _ = setup(tmp_path)
    for name,mode in [('author','B'),('reviewer','B'),('recipient','C')]: lab(c,name,mode)
    ident = register(registry,method())
    original = c.execute(c.plan('author','original',experiment([ident])))
    with pytest.raises(Refusal): c.replay(original,'author',assignment_id='not-independent')
    replay = c.replay(original,'reviewer',assignment_id='independent-assignment')
    assert len(execution.calls)==2 and execution.calls[0][1]!=execution.calls[1][1]
    c.release(ident,original,replay)
    assert registry.method_status(ident)['status']=='released'
    previous = registry.lab('recipient')['active_revision']
    trial = c.execute(c.plan('recipient','recipient-trial',experiment([ident],inputs=[[5,2,4]]),
                            purpose='transfer',selected=[ident],input_origin='independent_fixture_case'))
    adoption = c.adopt('recipient',ident,trial,expected_revision=previous)
    changed = registry.get_artifact('adoption',adoption)['new_revision']
    assert changed!=previous and ident in registry.revision(changed)['active_methods']
    followup = c.execute(c.plan('recipient','after-adoption',experiment([ident],inputs=[[9,1,3]])))
    assert registry.get_artifact('experiment',followup)['invoked_methods']==[ident]
    assert len(execution.calls)==4


def test_adoption_cas_and_indirect_control_method_exclusion(tmp_path):
    c, registry, _, execution, _ = setup(tmp_path)
    lab(c,'recipient','C')
    selected = register(registry,method(author='foreign',name='selected'))
    control = register(registry,method(author='foreign',name='wrapper',deps=[selected]))
    for ident in (selected,control): release_fixture(registry,ident)
    execution.behavior[control]='identity'
    previous = registry.lab('recipient')['active_revision']
    trial = c.execute(c.plan('recipient','indirect-control',experiment([selected],control=[control]),
                            purpose='transfer',selected=[selected,control]))
    assert registry.get_artifact('experiment',trial)['rule_passed']
    with pytest.raises(Refusal,match='control'):
        c.adopt('recipient',selected,trial,expected_revision=previous)
    # A proper contrast still cannot replace a concurrently activated revision.
    other = c.execute(c.plan('recipient','clean-control',experiment([selected]),purpose='transfer',selected=[selected]))
    evidence=registry.put_artifact('method_evidence',{'fixture':'new revision'})
    registry.revise_lab('recipient',previous,{'evidence_refs':[evidence]},'concurrent-update')
    with pytest.raises(Refusal): c.adopt('recipient',selected,other,expected_revision=previous)


def test_fixed_proposer_repair_loop_records_failed_and_successful_experiments(tmp_path):
    bad = method(code='def run(values):\n    raise ValueError("technical fixture defect")\n')
    fixed = method(name='repaired')
    fixed['manifest']['parents']=[bad['id']]
    fixed['id']=content_hash(dict(schema='sciel-method/1',manifest=fixed['manifest'],files=fixed['files']))
    proposals=[dict(kind='method',files=v['files'],manifest=v['manifest'],experiment=experiment(['$new'])) for v in [bad,fixed]]
    proposer=FixedProposer(proposals)
    c,registry,ledger,execution,_=setup(tmp_path,proposer=proposer)
    lab(c,'author')
    execution.behavior[bad['id']]='failed'
    first,second=c.step('author'),c.step('author')
    assert not registry.get_artifact('experiment',first['experiment_id'])['contract_passed']
    assert registry.get_artifact('experiment',second['experiment_id'])['contract_passed']
    assert first['method_id']!=second['method_id']
    assert proposer.calls==2 and len(execution.calls)==2
    assert float(ledger.snapshot('account-author')['spent']['dollars'])==.02
    assert second['revision_id']!=first['revision_id']


@pytest.mark.skipif(not os.environ.get('SCIEL_TEST_METHOD_IMAGE'),reason='explicit immutable local Docker image required')
def test_real_docker_coordinator_known_defect_repair(tmp_path):
    image=os.environ['SCIEL_TEST_METHOD_IMAGE']
    bad=method(code='def run(values):\n    raise ValueError("technical fixture defect")\n',image=image)
    fixed=method(name='repaired',image=image)
    fixed['manifest']['parents']=[bad['id']]
    proposals=[dict(kind='method',files=v['files'],manifest=v['manifest'],experiment=experiment(['$new'])) for v in [bad,fixed]]
    c,registry,ledger,_,proposer=setup(tmp_path,executor=DockerExecutor(image),proposer=FixedProposer(proposals))
    lab(c,'author')
    first,second=c.step('author'),c.step('author')
    assert not registry.get_artifact('experiment',first['experiment_id'])['contract_passed']
    assert registry.get_artifact('experiment',second['experiment_id'])['contract_passed']
    assert proposer.calls==2 and float(ledger.snapshot('account-author')['spent']['cpu_seconds'])>0
