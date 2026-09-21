"""Durable job semantics without model, Docker or task-evaluator dispatch."""
from copy import deepcopy
import json
import os
import signal
import socket
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest

from sciel.methods.executor import ExecutionRefused
from sciel.engine.capacity import HostCapacity
from sciel.engine.runtime import process_identity, process_matches
from sciel.methods.models import content_hash
from sciel.research_tools.library import ToolLibrary
from sciel.research_tools.jobs import BackgroundJobs, read_job, read_json, write_json
from sciel.research_tools.job_worker import run_job, run_service
from test_research_tool_result_contract import result as tool_result

IMAGE = 'sha256:'+'a'*64
LIMITS = dict(cpu_seconds=3600,wall_seconds=7200,memory_bytes=128*1024*1024,
              pids=16,max_output_bytes=65536,max_stderr_bytes=2048)
ACTUAL = dict(LIMITS,cpu_seconds=3,wall_seconds=7)
CONFIG = dict(enabled=True,max_active=2,max_pending=3,max_cpu_seconds=3600,max_wall_seconds=7200)


class FakeExecutor:
    def __init__(self, run_dir=None):
        self.identity = {'image_id':IMAGE,'schema':'test-fake/1'}
        self.calls = []
        self.run_dir = Path(run_dir) if run_dir else None
        self.cancel_mode = False
        self.started = threading.Event()

    def execute(self, bundle, inputs, **kw):
        self.calls.append(deepcopy({k:v for k,v in kw.items() if k!='cancel_requested'}))
        self.started.set()
        if self.run_dir is not None:
            with (self.run_dir/'fake-invocations.log').open('a') as stream:
                stream.write(kw['invocation_id']+'\n')
            time.sleep(.15)
        if self.cancel_mode:
            deadline=time.monotonic()+3
            while not kw['cancel_requested']() and time.monotonic()<deadline:
                time.sleep(.005)
        output=tool_result({'continuation_state':{'incumbent':inputs.get('continuation_state',{}).get('incumbent',0)+1},'observations':['measured']})
        receipt=dict(schema='sciel-method-execution/1',method_id=bundle['id'],
            invocation_id=kw['invocation_id'],plan_id=kw['plan_id'],reservation_id=kw['reservation_id'],
            source_sha256=content_hash(bundle['files']),runtime_sha256=content_hash(self.identity),
            input_sha256=content_hash(inputs),request_sha256='b'*64,
            dependency_ids=sorted(value['id'] for value in kw.get('dependency_bundles',[])),
            status='unknown' if self.cancel_mode else 'ok',dispatch_status='unknown' if self.cancel_mode else 'invoked',
            reason='cancel_requested_without_terminal_receipt' if self.cancel_mode else None,
            output=None if self.cancel_mode else output,output_sha256=None if self.cancel_mode else content_hash(output),
            limits=kw['limits'],resources={'cpu_seconds':None if self.cancel_mode else .12,'wall_seconds':.2},stderr='')
        receipt['receipt_sha256']=content_hash(receipt)
        return receipt


def fake_library(run_dir,runtime_lock,limits):
    return ToolLibrary(run_dir,runtime_lock,limits,executor=FakeExecutor(run_dir))


class JobTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.run=Path(self.temp.name)/'run';self.run.mkdir()
        self.executor=FakeExecutor()
        self.library=ToolLibrary(self.run,{'image_id':IMAGE},LIMITS,executor=self.executor)
        self.bundle=self.library.register('diagnostic','Bounded diagnostic.','def run(payload): return payload',0)
        self.jobs=BackgroundJobs(self.run,self.library,CONFIG,launch=lambda:None)
        self.owner={'action_id':'c'*64,'cell_id':'d'*64,'island_id':0,'parent_program_id':'p1'}

    def tearDown(self):
        self.temp.cleanup()

    def submit(self,key='one',inputs=None,**kw):
        return self.jobs.submit(key,self.bundle,inputs or {'arguments':{}},ACTUAL,self.owner,**kw)

    def test_submit_ack_is_identical_before_and_after_completion_and_budget_bound(self):
        ack=self.submit();ident=ack['job_id']
        run_job(self.run,ident,self.library)
        self.assertEqual(self.submit(),ack)
        self.assertEqual(self.jobs.poll(ident)['status'],'completed')
        self.assertEqual(self.executor.calls[0]['limits'],ACTUAL)
        self.assertEqual(self.executor.calls[0]['plan_id'],self.owner['action_id'])
        self.assertEqual(self.executor.calls[0]['invocation_id'],ident)
        previous=self.jobs.poll(ident)
        run_job(self.run,ident,self.library)
        self.assertEqual(self.jobs.poll(ident),previous)
        self.assertEqual(len(self.executor.calls),1)
        with self.assertRaises(ValueError):self.submit(inputs={'different':True})
        with self.assertRaises(ValueError):
            self.jobs.submit('too-large',self.bundle,{},dict(ACTUAL,cpu_seconds=3601),self.owner)

    def test_crash_dispatch_marker_never_redispatches_and_cost_is_unknown(self):
        ident=self.submit()['job_id'];spec=self.jobs.spec(ident)
        write_json(self.jobs.root/ident/'dispatch.json',{'job_id':ident,'request_hash':spec['request_hash']},immutable=True)
        run_job(self.run,ident,self.library)
        result=self.jobs.poll(ident)
        self.assertEqual(result['status'],'unknown')
        self.assertIsNone(result['resources']['cpu_seconds'])
        self.assertEqual(self.executor.calls,[])
        run_job(self.run,ident,self.library)
        self.assertEqual(self.executor.calls,[])

    def test_cancel_queued_is_known_not_dispatched_with_stable_ack(self):
        ident=self.submit()['job_id'];ack=self.jobs.cancel(ident)
        run_job(self.run,ident,self.library)
        self.assertEqual(self.jobs.cancel(ident),ack)
        result=self.jobs.poll(ident)
        self.assertEqual((result['status'],result['dispatch_status']),('cancelled','not_dispatched'))
        self.assertEqual(result['resources']['cpu_seconds'],0)
        self.assertEqual(self.executor.calls,[])

    def test_cancel_running_does_not_turn_unknown_cpu_into_zero(self):
        self.executor.cancel_mode=True
        ident=self.submit()['job_id']
        thread=threading.Thread(target=run_job,args=(self.run,ident,self.library));thread.start()
        self.assertTrue(self.executor.started.wait(2))
        self.jobs.cancel(ident);thread.join(3)
        self.assertFalse(thread.is_alive())
        result=self.jobs.poll(ident)
        self.assertTrue(result['cancel_requested'])
        self.assertEqual(result['status'],'unknown')
        self.assertIsNone(result['resources']['cpu_seconds'])

    def test_continuation_uses_full_verified_output_and_new_intent(self):
        first=self.submit()['job_id'];run_job(self.run,first,self.library)
        continuation=self.jobs.continuation_inputs(first)
        self.assertEqual(continuation['continuation_state'],{'incumbent':1})
        inputs=continuation['inputs'];inputs['continuation_state']=continuation['continuation_state']
        second=self.submit('two',inputs,parent_job_id=first)['job_id'];run_job(self.run,second,self.library)
        self.assertNotEqual(first,second)
        self.assertEqual(self.jobs.result(second)['output']['observations']['continuation_state'],{'incumbent':2})
        self.assertEqual(self.jobs.spec(second)['request']['parent_job_id'],first)
        detail=read_job(self.run,second,include_receipt=True)
        self.assertEqual(detail['request']['inputs'],inputs)

    def test_pending_cap_and_source_or_receipt_tampering_fail_closed(self):
        ids=[self.submit(str(n))['job_id'] for n in range(3)]
        with self.assertRaises(ExecutionRefused):self.submit('four')
        run_job(self.run,ids[0],self.library)
        self.submit('four')
        path=self.jobs.root/ids[0]/'receipt.json';receipt=json.loads(path.read_text())
        receipt['resources']['cpu_seconds']=0;path.write_text(json.dumps(receipt))
        with self.assertRaises(ValueError):self.jobs.result(ids[0])
        with self.assertRaises(ValueError):read_job(self.run,'../outside')

    def test_pure_read_does_not_start_worker_or_mutate_status(self):
        ident=self.submit()['job_id'];path=self.jobs.root/ident/'status.json';before=path.read_bytes()
        self.assertEqual(read_job(self.run,ident)['status'],'queued')
        self.assertEqual(path.read_bytes(),before)
        self.assertEqual(self.executor.calls,[])

    def test_dispatcher_start_failure_does_not_change_durable_acceptance(self):
        def fail():raise OSError('fixture')
        self.jobs._launch=fail
        ack=self.submit()
        self.assertEqual(self.submit(),ack)
        self.assertEqual(self.jobs.poll(ack['job_id'])['status'],'queued')
        self.assertEqual(read_json(self.jobs.root/'service.json')['status'],'launch_failed')

    def test_service_recovers_terminal_receipt_before_status_publish(self):
        ident=self.submit()['job_id'];run_job(self.run,ident,self.library)
        state=read_json(self.jobs.root/ident/'status.json');state.update(status='running',receipt_id=None)
        write_json(self.jobs.root/ident/'status.json',state)
        write_json(self.jobs.root/'queue.json',[ident])
        run_service(self.run,library_factory=lambda *args:self.library,stop_when_idle=True,poll_seconds=.001)
        self.assertEqual(self.jobs.poll(ident)['status'],'completed')
        self.assertEqual(len(self.executor.calls),1)

    def test_service_honors_concurrency_cap(self):
        self.executor.run_dir=self.run
        guard=threading.Lock();seen={'active':0,'maximum':0}
        original=self.executor.execute
        def execute(*args,**kwargs):
            with guard:
                seen['active']+=1;seen['maximum']=max(seen['maximum'],seen['active'])
            try:return original(*args,**kwargs)
            finally:
                with guard:seen['active']-=1
        self.executor.execute=execute
        ids=[self.submit(str(n))['job_id'] for n in range(3)]
        run_service(self.run,library_factory=lambda *args:self.library,stop_when_idle=True,poll_seconds=.001)
        self.assertEqual(seen['maximum'],2)
        self.assertTrue(all(read_job(self.run,ident)['status']=='completed' for ident in ids))

    def test_cancel_while_shared_evaluator_capacity_busy_never_dispatches(self):
        capacity=HostCapacity(str(self.run/'capacity'),llm_slots=1,evaluator_slots=1,provider_limits={})
        ident=self.submit()['job_id']
        with capacity.acquire('evaluator',timeout=1):
            thread=threading.Thread(target=run_job,args=(self.run,ident,self.library,capacity));thread.start()
            deadline=time.monotonic()+2
            while read_job(self.run,ident)['status']=='queued' and time.monotonic()<deadline:time.sleep(.005)
            self.jobs.cancel(ident);thread.join(2)
            self.assertFalse(thread.is_alive())
        result=read_job(self.run,ident)
        self.assertEqual(result['status'],'cancelled')
        self.assertEqual(self.executor.calls,[])

    def test_orphan_dispatch_blocks_other_work_until_cancelled_without_redispatch(self):
        lost=self.submit('lost')['job_id'];queued=self.submit('queued')['job_id']
        write_json(self.jobs.root/lost/'dispatch.json',{'intent':'crashed'},immutable=True)
        thread=threading.Thread(target=run_service,args=(self.run,),kwargs={
            'library_factory':lambda *args:self.library,'stop_when_idle':True,'poll_seconds':.005})
        thread.start();deadline=time.monotonic()+2
        while read_job(self.run,lost)['status']!='unknown' and time.monotonic()<deadline:time.sleep(.005)
        self.assertEqual(read_job(self.run,queued)['status'],'queued')
        self.assertEqual(self.executor.calls,[])
        self.jobs.cancel(queued);thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(read_job(self.run,queued)['status'],'cancelled')

    def test_background_service_survives_submitting_process_exit_fake_executor(self):
        ident=self.submit()['job_id']
        code=('import subprocess,sys,os; '
            'stderr=open(os.path.join(sys.argv[1],"fake-worker.stderr"),"ab"); '
            'p=subprocess.Popen([sys.executable,"-c",'
            '"from sciel.research_tools.job_worker import run_service; '
            'from test_research_tool_jobs import fake_library; '
            'run_service(sys.argv[1],library_factory=fake_library,stop_when_idle=True)",sys.argv[1]],'
            'stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=stderr,'
            'start_new_session=True); print(p.pid)')
        # Explicit import in child snippet; no production option can select this fake.
        code=code.replace('"from sciel.research_tools.job_worker', '"import sys; from sciel.research_tools.job_worker')
        env=dict(os.environ,PYTHONPATH=os.pathsep.join([str(Path(__file__).resolve().parents[2]/'src'),str(Path(__file__).parent)]),PYTHONDONTWRITEBYTECODE='1')
        launched=subprocess.run([sys.executable,'-c',code,str(self.run)],capture_output=True,text=True,env=env,timeout=10)
        self.assertEqual(launched.returncode,0,launched.stderr)
        pid=int(launched.stdout.strip())
        worker={'pid':pid,'hostname':socket.gethostname(),'process_identity':process_identity(pid)}
        deadline=time.monotonic()+20
        diagnostic=lambda:(self.run/'fake-worker.stderr').read_text()[-5000:]
        try:
            while read_job(self.run,ident)['status'] not in ('completed','failed','unknown') and time.monotonic()<deadline:
                time.sleep(.02)
            self.assertEqual(read_job(self.run,ident)['status'],'completed',diagnostic())
            self.assertEqual((self.run/'fake-invocations.log').read_text().splitlines(),[ident])
            while read_json(self.jobs.root/'service.json')['status']!='stopped' and time.monotonic()<deadline:
                time.sleep(.02)
            self.assertEqual(read_json(self.jobs.root/'service.json')['status'],'stopped',diagnostic())
        finally:
            # A stopped idle service has already joined its work and is exiting.
            # Let it reap naturally before signalling its process group: on
            # macOS the group can disappear between identity lookup and killpg.
            finish=time.monotonic()+2
            while process_matches(worker) is True and time.monotonic()<finish:time.sleep(.02)
            if process_matches(worker) is True:
                try:
                    os.killpg(pid,signal.SIGTERM)
                except ProcessLookupError:
                    pass  # The idle worker may exit after the identity check.
                finish=time.monotonic()+2
                while process_matches(worker) is True and time.monotonic()<finish:time.sleep(.02)


if __name__=='__main__':unittest.main()
