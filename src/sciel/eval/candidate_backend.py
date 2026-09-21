"""Run an isolated candidate cascade within one durable Engine operation."""
from copy import deepcopy
import math
from pathlib import Path
import platform
import sys
import time

import numpy as np

from ..methods.executor import DockerExecutor, ExecutionRefused
from ..methods.models import content_hash
from .candidate_config import validate_task
from .candidate_registry import PINS
from .candidate_evidence import SCHEMA, FAILURE_SCHEMA, ROOT, sha, _write
from .candidate_evidence import runtime_sources as runtime_sources
from .candidate_receipts import EvaluationBackendError as EvaluationBackendError
from .candidate_receipts import validate_execution as validate_execution
from .candidate_receipts import verify_execution as verify_execution
from .candidate_receipts import _validate_candidate_receipt
from .cascade import CascadeOutcome
from .facade import EvalResult, EvaluationInfrastructureError
from .trusted_scoring.adapter import HostBinding, aggregate_completed, _failure
from .trusted_scoring.transport import MAX_BYTES

MAX_WRAPPED_BYTES = 6 * MAX_BYTES + 2


class CandidateBackend:
    def __init__(self,cfg,task,*,executor=None):
        self.kind=validate_task(cfg,task)
        if self.kind is None: raise ValueError('candidate backend must be explicitly selected')
        self.policy=deepcopy(cfg['evaluation'])
        self.task=task
        self.executor=executor if executor is not None else DockerExecutor(self.policy['image_id'])
        self.runtime=self.executor.identity
        if self.runtime.get('image_id')!=self.policy['image_id']:
            raise ValueError('candidate executor image differs from selected immutable image')
        self.worker=(ROOT/'_candidate_worker.py').read_text()
        self.identity={'schema':'sciel-candidate-backend/1','kind':self.kind,
            'task':deepcopy(PINS[self.kind]),'effective_task':{key:getattr(task,key) for key in ('stochastic','confirm_n','smoke_timeout','full_timeout','confirm_timeout','per_run_timeout','cpu_seconds','mem_bytes')},'policy':self.policy,'executor':self.runtime,
            'source_files':runtime_sources(),'host_runtime':{'python':platform.python_version(),
                'numpy':np.__version__,'platform':sys.platform},
            'limits_scope':'stage CPU/wall shared across all candidate invocations and trusted scorer containers; cleanup grace up to10s per owned container',
            'confirmation':'all requested calls; first failure else first lowest score'}

    def assert_compatible(self,cfg,task):
        if validate_task(cfg,task)!=self.kind or cfg['evaluation']!=self.policy or runtime_sources()!=self.identity['source_files']:
            raise ValueError('candidate backend/task/runtime changed since preflight')

    def _prepared(self,source,stage,trial,request_hash,limits):
        invocation=sha((request_hash+':'+stage+':'+str(trial)).encode())[:32]
        public_stage='full' if stage=='confirm' else stage
        binding={'evaluation_request_sha256':request_hash,'source_sha256':sha(source),
            'identity_sha256':content_hash(self.identity),'stage':stage,'trial':trial,
            'invocation_id':invocation,'limits':limits}
        inputs={'kind':self.kind,'stage':public_stage,'binding_sha256':content_hash(binding)}
        manifest=dict(name='candidate-output-'+self.kind,kind='adapter',author_lab='trusted-host',
            entrypoint='worker.py:run',input_schema={'type':'object','properties':{
                'kind':{'type':'string','enum':[self.kind]},'stage':{'type':'string','enum':[public_stage]},
                'binding_sha256':{'type':'string','minLength':64,'maxLength':64}},
                'required':['kind','stage','binding_sha256'],'additionalProperties':False},
            output_schema={'type':'string','minLength':1,'maxLength':MAX_BYTES},
            protocol='Invoke public task constructor once; return encoded values, never an authoritative score.',
            runtime_lock={'image_id':self.policy['image_id']},dependencies=[],parents=[],resources=limits,
            scope='Candidate-only isolated evaluation; scorer/task root are not mounted.',examples=[])
        files={'candidate.py':source.decode('utf-8'),'worker.py':self.worker}
        bundle={'id':content_hash({'schema':'sciel-method/1','manifest':manifest,'files':files}),'manifest':manifest,'files':files}
        request=dict(schema='sciel-method-request/1',method_id=bundle['id'],invocation_id=invocation,
            plan_id='evaluation-'+request_hash,reservation_id='evaluation-'+invocation,input=inputs,
            input_sha256=content_hash(inputs),source_sha256=content_hash(files),runtime_sha256=content_hash(self.runtime),
            limits=limits,methods={bundle['id']:{'root':'/bundle','entrypoint':'worker.py:run'}},dependency_ids=[])
        request['request_sha256']=content_hash(request)
        self.executor.preflight(bundle,inputs,limits)
        return bundle,inputs,request

    def _method(self,receipt,expected,container):
        return _validate_candidate_receipt(receipt,expected,container,self.runtime,self.policy['image_id'])

    def evaluate(self,source,*,evaluation_id,confirm):
        from ..engine.parallel_calls import current_evaluation_operation
        operation=current_evaluation_operation()
        if operation is None: raise EvaluationInfrastructureError('candidate evaluation requires an active durable evaluation operation')
        directory,request_hash=operation
        refs=[]; stages=[]; cpu_values=[]; known_cpu=0.; cpu_complete=True; begun=time.monotonic()
        def evidence(status):
            return {'schema':SCHEMA,'request_sha256':request_hash,'evaluation_id':evaluation_id,
                'identity':deepcopy(self.identity),'status':status,'receipts':deepcopy(refs),'stages':deepcopy(stages),
                'resources':{'cpu_seconds':known_cpu if cpu_complete else None,'known_cpu_seconds':known_cpu,
                    'cpu_complete':cpu_complete,'wall_seconds':time.monotonic()-begun,
                    'scope':'candidate and trusted scorer cgroup CPU observations; excludes host orchestration and Docker daemon'},
                'stage_scope':'one stage deadline across transport/scoring/all confirmations; FIFO queue excluded'}
        def fatal(reason,status='unknown'):
            execution=evidence(status)
            raise EvaluationBackendError({'schema':FAILURE_SCHEMA,'reason':reason,'execution':execution,
                                          'execution_sha256':content_hash(execution)})
        def _stage(name):
            nonlocal known_cpu,cpu_complete
            deadline=time.monotonic()+getattr(self.task,name+'_timeout')
            stage_cpu=0.; first_cpu=len(cpu_values); results=[]
            count=self.task.confirm_n if name=='confirm' else 1
            for trial in range(count):
                remaining=deadline-time.monotonic()
                remaining_cpu=self.policy['cpu_seconds']-stage_cpu
                if remaining<=0 or remaining_cpu<=0:
                    return _failure(self.kind,'Evaluation stage resource budget exhausted'),False
                limits={'wall_seconds':remaining,'cpu_seconds':remaining_cpu,'memory_bytes':self.policy['memory_bytes'],
                        'pids':self.policy['pids'],'max_output_bytes':MAX_WRAPPED_BYTES,'max_stderr_bytes':4096}
                try: bundle,inputs,request=self._prepared(source,name,trial,request_hash,limits)
                except Exception: fatal('execution_refused','infrastructure_failed')
                container='sciel-eval-'+sha((request_hash+':'+name+':'+str(trial)).encode())
                intent={'schema':'sciel-candidate-physical-intent/1','request_sha256':request_hash,
                    'stage':name,'trial':trial,'invocation_id':request['invocation_id'],
                    'container_name':container,'method_request':request}
                try: intent_ref=_write(Path(directory)/f'{name}-{trial}.intent.json',intent)
                except Exception: cpu_complete=False; fatal('attachment_failure')
                method=None; scorer=None; scorer_intent=None; result=None; failure=None
                candidate_not_dispatched=False; scorer_not_dispatched=False
                try:
                    method=self.executor.execute(bundle,inputs,invocation_id=request['invocation_id'],
                        plan_id=request['plan_id'],reservation_id=request['reservation_id'],limits=limits,container_name=container)
                    status,cpu=self._method(method,request,container)
                    if cpu is not None:
                        cpu_values.append(float(cpu));known_cpu=math.fsum(cpu_values);stage_cpu=math.fsum(cpu_values[first_cpu:])
                    if status=='unknown': cpu_complete=False; failure='execution_unknown'
                    elif status=='not_dispatched': failure='execution_refused'
                    elif status=='failed': result=_failure(self.kind,'Candidate execution failed: '+str(method.get('reason'))[:120])
                    else:
                        left=deadline-time.monotonic(); cpu_left=self.policy['cpu_seconds']-stage_cpu
                        if left<=0 or cpu_left<=0:
                            result=_failure(self.kind,'Evaluation stage resource budget exhausted')
                        else:
                            from .scorer_container import prepare_scorer, score_in_container, ScorerInfrastructureError
                            scorer_id=sha((request_hash+':'+name+':'+str(trial)+':scorer').encode())
                            scorer_name='sciel-eval-'+scorer_id
                            scorer_limits={**limits,'wall_seconds':left,'cpu_seconds':cpu_left}
                            try:
                                raw_output=method['output'].encode('utf-8')
                            except UnicodeError:
                                result=_failure(self.kind,'Candidate output is not valid UTF-8')
                            else:
                                if not 0 < len(raw_output) <= MAX_BYTES:
                                    result=_failure(self.kind,'Candidate output exceeds bounded JSON transport')
                                else:
                                    prepared=prepare_scorer(self.executor,self.kind,raw_output,
                                        HostBinding(sha(source),PINS[self.kind]['evaluator_sha256'],request['invocation_id'],
                                            'full' if name=='confirm' else name,float(method['resources']['method_wall_seconds'])),
                                        limits=scorer_limits,invocation_id=scorer_id[:32],
                                        plan_id=request['plan_id'],reservation_id='evaluation-'+scorer_id[:32])
                                    scorer_intent=_write(Path(directory)/f'{name}-{trial}.scorer-intent.json',
                                        {'schema':'sciel-scorer-physical-intent/1','request_sha256':request_hash,
                                         'container_name':scorer_name,'method_request':prepared['request']})
                                    try:
                                        scored=score_in_container(self.executor,prepared,container_name=scorer_name)
                                        result,scorer=scored['result'],scored['receipt']
                                    except ScorerInfrastructureError as error:
                                        scorer=error.receipt; failure='scorer_failure'; scorer_not_dispatched=error.not_dispatched
                                    cpu=(scorer.get('resources') or {}).get('cpu_seconds') if type(scorer) is dict else None
                                    if cpu is None and not scorer_not_dispatched: cpu_complete=False
                                    elif cpu is not None:
                                        cpu_values.append(float(cpu));known_cpu=math.fsum(cpu_values);stage_cpu=math.fsum(cpu_values[first_cpu:])
                                    if type(scorer) is dict and (scorer.get('status')=='unknown' or scorer.get('container',{}).get('removed') is not True):
                                        cpu_complete=False
                except ExecutionRefused: failure='execution_refused'; candidate_not_dispatched=True
                except Exception: cpu_complete=False; failure='receipt_invalid'
                record={'schema':'sciel-candidate-physical-receipt/1','request_sha256':request_hash,
                    'intent_sha256':intent_ref['sha256'],'stage':name,'trial':trial,'method_receipt':method,
                    'scorer_receipt':scorer,'scorer_intent':scorer_intent,'result':result,'failure':failure,
                    'candidate_not_dispatched':candidate_not_dispatched,'scorer_not_dispatched':scorer_not_dispatched}
                try: refs.append(_write(Path(directory)/f'{name}-{trial}.receipt.json',record))
                except Exception: cpu_complete=False; fatal('attachment_failure')
                if failure: fatal(failure,'unknown' if not cpu_complete else 'infrastructure_failed')
                # Scorer output is trusted code, but validated before scientific use.
                try: EvalResult.from_json(result)
                except Exception: fatal('scorer_failure','infrastructure_failed')
                results.append(result)
                if time.monotonic()>=deadline or stage_cpu>=self.policy['cpu_seconds']:
                    return _failure(self.kind,'Evaluation stage resource budget exhausted'),False
            return (aggregate_completed(self.kind,results,count) if name=='confirm' else results[0]), True
        def stage(name):
            start=time.monotonic(); cpu_start=len(cpu_values)
            result,complete=_stage(name)
            elapsed=time.monotonic()-start; cpu=math.fsum(cpu_values[cpu_start:])
            exhausted=elapsed>=getattr(self.task,name+'_timeout') or cpu>=self.policy['cpu_seconds']
            if exhausted: result,complete=_failure(self.kind,'Evaluation stage resource budget exhausted'),False
            summary={'schema':'sciel-candidate-stage/1','request_sha256':request_hash,'stage':name,
                'timeout':getattr(self.task,name+'_timeout'),'elapsed_seconds':elapsed,'cpu_seconds':cpu,
                'resource_exhausted':exhausted,'complete':complete,'result':result}
            try: stages.append(_write(Path(directory)/(name+'.stage.json'),summary))
            except Exception: fatal('attachment_failure')
            return result,complete,elapsed
        smoke,_,timing=stage('smoke')
        result=smoke; reached='smoke'; raw=smoke['score']; confirmed=False
        if smoke['ok']:
            result,_,timing=stage('full'); reached='full'; raw=result['score']
            if result['ok'] and confirm and self.task.stochastic:
                result,complete,timing=stage('confirm'); reached='confirm'
                confirmed=bool(complete and result['ok'] and self.task.confirm_n>=2)
        value=EvalResult.from_json(result)
        value.confirmed_sampled=confirmed
        value.timing=timing
        execution=evidence('completed')
        validate_execution(execution)
        outcome=CascadeOutcome(value,raw,reached,confirmed)
        outcome.execution=execution
        return outcome

def create_backend(cfg,task):
    return CandidateBackend(cfg,task) if validate_task(cfg,task) is not None else None
