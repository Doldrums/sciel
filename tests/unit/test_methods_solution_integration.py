"""End-to-end coordination with explicitly fake external execution, no host candidate execution."""
from dataclasses import asdict
import hashlib

import pytest

from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.methods.budget import BudgetUnknown, ResourceLedger
from sciel.methods.coordinator import Coordinator
from sciel.methods.engine_bridge import EngineCascadeBridge, LabSolutionHistory
from sciel.methods.models import content_hash
from sciel.methods.proposals import ScriptedProposer
from sciel.methods.registry import Registry
from sciel.methods.tasks import Circle26Task, SortTask

IMAGE='sha256:'+'a'*64
RESOURCES=dict(wall_seconds=10,cpu_seconds=5,memory_bytes=268435456,pids=32,
               max_output_bytes=131072,max_stderr_bytes=4096)
BUDGET=dict(dollars=1,cpu_seconds=1000,gpu_seconds=0,wall_seconds=10000,storage_bytes=100_000_000,steps=50)
CODE='def run_packing():\n    return 1\n'


class FixtureExecutor:
    identity={'schema':'nonexecuting-cascade-fixture/1','image_id':IMAGE}
    overhead_wall_bound=12
    def __init__(self,task):
        self.adapter=EngineCascadeBridge(task.task) if isinstance(task,Circle26Task) else task
        self.calls=[]
        self.status='ok'
        self.valid=True
    def execute(self,bundle,value,*,invocation_id,plan_id,reservation_id,limits,dependency_bundles=()):
        self.calls.append(invocation_id)
        if isinstance(self.adapter,EngineCascadeBridge):
            result=EvalResult(ok=self.valid,score=.5 if self.valid else 0.,cases={'validity':1.,'sum_radii':1.3175},
                              artifacts={'fixture':'not an actual evaluator measurement'})
            output=dict(schema='sciel-method-cascade-result/1',
                source_sha256=hashlib.sha256(bundle['files']['candidate.py'].encode()).hexdigest(),
                task_identity_sha256=content_hash(self.adapter.identity),
                outcome=asdict(CascadeOutcome(result,.4,'full',False)))
        else:
            output=sorted(value) if self.valid else list(value)
        return dict(status=self.status,method_id=bundle['id'],plan_id=plan_id,invocation_id=invocation_id,
            reservation_id=reservation_id,input_sha256=content_hash(value),source_sha256=content_hash(bundle['files']),
            runtime_sha256=content_hash(self.identity),dispatch_status='invoked',reason='fixture',stderr='',
            output=output if self.status=='ok' else None,resources={'cpu_seconds':None if self.status=='unknown' else .01,'wall_seconds':.02})


def proposal(task,code=CODE,inputs=None):
    return dict(kind='solution',code=code,experiment=dict(question='An explicitly scripted test.',
        treatment=['$new'],control=[],inputs=inputs or ([{}] if isinstance(task,Circle26Task) else [[3,1,2]]),
        criterion={'metric':task.metric,'direction':'max','minimum_effect':0},scope='Fixture only.'))


def setup(tmp_path,task,responses):
    registry=Registry(tmp_path/'registry')
    ledger=ResourceLedger(tmp_path/'budget.jsonl',BUDGET)
    executor=FixtureExecutor(task)
    proposer=ScriptedProposer(responses)
    c=Coordinator(registry,ledger,executor,proposer,task,resources=RESOURCES,max_steps=10)
    c.create_lab('author','Fixture','account',{},mode='B')
    return c,registry,ledger,executor,proposer


def test_canonical_solution_uses_bound_cascade_and_retains_real_result_fields(tmp_path):
    task=Circle26Task()
    c,r,_,executor,_=setup(tmp_path,task,[proposal(task)])
    result=c.step('author')
    assert 'solution_id' in result,result
    saved=r.get_artifact('solution',result['solution_id'])
    plan=r.get_artifact('experiment_plan',result['plan_id'])
    assert plan['measurement_protocols']['treatment']==c.solution_task.identity
    assert plan['measurement_protocols']['control']==task.identity
    assert saved['evaluation']['score']==.5 and saved['evaluation']['cases']['sum_radii']==1.3175
    assert saved['cascade']['raw_first_score']==.4 and saved['cascade']['reached']=='full'
    assert saved['program']['code']==CODE and saved['program']['score']==.5
    assert len(executor.calls)==1
    receipt=r.get_artifact('proposal_receipt',result['proposal_id'])
    assert r.get_artifact('proposal_context',receipt['context_id'])['lab_id']=='author'


@pytest.mark.parametrize('task',[Circle26Task(),SortTask()])
def test_solution_duplicate_chain_is_task_general_bounded_and_charges_steps(tmp_path,task):
    code=CODE if isinstance(task,Circle26Task) else 'def solve(values):\n    return sorted(values)\n'
    p=proposal(task,code)
    duplicate=proposal(task,code+'# only a comment\n')
    c,r,ledger,executor,proposer=setup(tmp_path,task,[p,duplicate,duplicate,duplicate])
    first=c.step('author')
    assert 'solution_id' in first
    for count in range(1,4):
        result=c.step('author')
        assert result['novelty_retry_count']==count
        assert result['novelty']['match_kind']=='ast'
        assert result['novelty']['matched_program_id']
        assert result['novelty']['matched_call_receipt_id']
        assert 'experiment_id' not in result and 'solution_id' not in result
    assert result['status']=='stopped' and result['reason']=='solution_novelty_resamples_exhausted'
    assert len(executor.calls)==1 and len(proposer.calls)==4
    assert c.step('author')==result
    assert float(ledger.snapshot()['spent']['steps'])==4


def test_malformed_repair_consumes_duplicate_attempt_and_does_not_reset_it(tmp_path):
    task=SortTask(); code='def solve(values):\n    return sorted(values)\n'
    p=proposal(task,code)
    c,r,_,executor,_=setup(tmp_path,task,[p,p,'invalid JSON',p])
    c.step('author'); c.step('author')
    assert c.step('author')['novelty_retry_count']==2
    result=c.step('author')
    assert result['status']=='stopped' and len(executor.calls)==1


def test_real_constant_edit_ends_retry_chain_without_semantic_claim(tmp_path):
    task=Circle26Task()
    p=proposal(task)
    c,r,_,executor,_=setup(tmp_path,task,[p,p,proposal(task,CODE.replace('return 1','return 2'))])
    c.step('author'); assert c.step('author')['novelty_retry_count']==1
    result=c.step('author')
    assert 'solution_id' in result and 'novelty_retry_count' not in result
    assert len(executor.calls)==2


@pytest.mark.parametrize('task',[Circle26Task(),SortTask()])
def test_known_failed_measurement_allows_remeasurement_without_fake_program(tmp_path,task):
    code=CODE if isinstance(task,Circle26Task) else 'def solve(values):\n    return sorted(values)\n'
    p=proposal(task,code)
    c,r,_,executor,_=setup(tmp_path,task,[p,p])
    executor.valid=False
    first=c.step('author')
    assert 'solution_id' not in first and 'solution_evaluation_id' in first
    history=LabSolutionHistory(r,'author',c.solution_task)
    assert history.inspect(code)['remeasurement'] is True
    executor.valid=True
    assert 'solution_id' in c.step('author')
    assert len(executor.calls)==2


def test_unknown_execution_remains_pending_and_resume_never_dispatches(tmp_path):
    task=Circle26Task()
    c,r,_,executor,proposer=setup(tmp_path,task,[proposal(task)])
    executor.status='unknown'
    with pytest.raises(BudgetUnknown): c.step('author')
    executor.status='ok'
    with pytest.raises(BudgetUnknown): c.step('author')
    assert len(executor.calls)==1 and len(proposer.calls)==1
    assert not LabSolutionHistory(r,'author',c.solution_task).inspect(CODE)['accept']


def test_crash_after_known_result_recovers_without_generation_or_evaluation_replay(tmp_path,monkeypatch):
    task=Circle26Task()
    c,r,_,executor,proposer=setup(tmp_path,task,[proposal(task)])
    original=LabSolutionHistory.finalize
    def interrupted(*args,**kwargs):
        raise RuntimeError('fixture crash before durable finalization')
    monkeypatch.setattr(LabSolutionHistory,'finalize',interrupted)
    with pytest.raises(RuntimeError): c.step('author')
    monkeypatch.setattr(LabSolutionHistory,'finalize',original)
    assert 'solution_id' in c.step('author')
    assert len(executor.calls)==len(proposer.calls)==1


def test_real_docker_circle_coordinator_canonical_seed(tmp_path):
    import os
    from sciel.methods.executor import DockerExecutor
    image=os.environ.get('SCIEL_TEST_METHOD_IMAGE')
    if not image:
        pytest.skip('explicit immutable local Docker image required')
    task=Circle26Task()
    executor=DockerExecutor(image)
    resources={**RESOURCES,'wall_seconds':30,'cpu_seconds':10,'memory_bytes':536870912,
               'pids':64,'max_output_bytes':262144,'max_stderr_bytes':8192}
    registry=Registry(tmp_path/'registry')
    ledger=ResourceLedger(tmp_path/'budget.jsonl',BUDGET)
    proposer=ScriptedProposer([proposal(task,task.canonical_seed_source)])
    c=Coordinator(registry,ledger,executor,proposer,task,resources=resources,max_steps=2)
    c.create_lab('author','CPU-only canonical seed smoke','account',{},mode='A')
    result=c.step('author')
    assert 'solution_id' in result,result
    saved=registry.get_artifact('solution',result['solution_id'])
    assert saved['evaluation_protocol']=='cascade'
    assert saved['cascade']['reached']=='full' and saved['evaluation']['ok']
    assert saved['evaluation']['cases']['validity']==1.
    assert saved['program']['score']==saved['evaluation']['score']
    assert saved['program']['code']==task.canonical_seed_source
    receipt=registry.get_artifact('call_receipt',saved['call_receipt_id'])
    assert receipt['status']=='completed' and receipt['result']['dispatch_status']=='invoked'
    assert float(ledger.snapshot()['spent']['cpu_seconds'])>0
    assert float(ledger.snapshot()['spent']['dollars'])==0
    assert len(proposer.calls)==1
    assert not LabSolutionHistory(registry,'author',c.solution_task).inspect(task.canonical_seed_source)['accept']


def test_audited_successful_solution_resume_consumes_recovered_receipt_without_new_calls(tmp_path):
    from copy import deepcopy
    task=SortTask(); code='def solve(values):\n    return sorted(values)\n'
    c,r,ledger,executor,proposer=setup(tmp_path,task,[proposal(task,code)])
    executor.status='unknown'
    with pytest.raises(BudgetUnknown) as caught:
        c.step('author')
    outer=r.get_artifact('call_receipt',caught.value.record['call_receipt_id'])
    known=deepcopy(outer['result'])
    known.update(status='ok',output=[1,2,3],output_sha256=content_hash([1,2,3]),reason=None,
                 resources={'cpu_seconds':.01,'wall_seconds':.02})
    audit=r.put_artifact('execution_audit',{'fixture':'externally recovered successful execution'})
    c.dispatch.resolve_unknown(outer['key'],outer['account'],result=known,status='completed',
        actual={'dollars':0,'cpu_seconds':1,'wall_seconds':1,'gpu_seconds':0,'storage_bytes':10000,'steps':0},
        reason='recovered method result and actual resources',evidence_refs=[audit])
    resumed=c.step('author')
    assert 'solution_id' in resumed,resumed
    assert len(executor.calls)==len(proposer.calls)==1
    assert not ledger.snapshot()['unknown_reservations']
    saved=r.get_artifact('solution',resumed['solution_id'])
    assert saved['program']['score']==1 and saved['evaluation']['ok']
