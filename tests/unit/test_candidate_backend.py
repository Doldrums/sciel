"""Real Engine/WAL and real receipt codec, fake physical containers only.

No candidate source is imported/executed. Fake scorer invokes trusted pure
arithmetic on fixed primitive arrays; Docker boundary has separate opt-in tests.
"""
import base64
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from sciel.campaign.budget import CampaignStopped
from sciel.config import apply_defaults
from sciel.engine import checkpoint, parallel_calls
from sciel.engine.factory import build_engine
from sciel.engine.loop import Engine
from sciel.eval.candidate_backend import CandidateBackend, EvaluationBackendError, verify_execution
from sciel.eval.facade import EvaluationInfrastructureError
from sciel.eval.registry import load_task
from sciel.eval.trusted_scoring.adapter import HostBinding, score_payload
from sciel.eval.trusted_scoring.transport import SCHEMA
from sciel.methods.executor import DockerExecutor
from sciel.methods.models import canonical_bytes, content_hash
from test_parallel_engine import Provider, config, events

ROOT=Path(__file__).resolve().parents[2]
IMAGE='sha256:ff59425be3cb209f0dfe6bb607d37ae8db1254f8e8a1173b1dde0e6c2e18cb29'
POLICY={'backend':'candidate_only','image_id':IMAGE,'cpu_seconds':5.,'memory_bytes':512*1024*1024,'pids':32}
TASKS={'circle':'oe/circle_packing','heil':'oe/heilbronn_triangle','beat':'math/beat_the_average'}
BOUND=('method_id','invocation_id','plan_id','reservation_id','source_sha256','runtime_sha256','input_sha256','request_sha256')


def payload(kind):
    if kind=='circle': values={'centers':[[((i%6)+.5)/6,((i//6)+.5)/5] for i in range(26)],
        'radii':[.03]*26,'centers_dtype':'float64','radii_dtype':'float64','reported_sum':'999.000000'}
    elif kind=='heil': values={'points':[[x,.03+.15*x*x] for x in [(.1+.08*i) for i in range(11)]]}
    else: values={'support':[0.,1.,2.],'weights':[1.,3.,7.]}
    return json.dumps({'schema':SCHEMA,'kind':kind,'values':values},separators=(',',':'))


class FakeDocker(DockerExecutor):
    def __init__(self,image=IMAGE):
        self._identity={'image_id':image,'schema':'sciel-method-executor/1','test_transport':True}
        source=ROOT/'src/sciel/methods'
        self._runner=(source/'_runner.py').read_bytes();self._loader=(source/'_loader.py').read_bytes()
        self.calls=[];self.change=None

    def _container(self,directory,request,container_name=None,**kwargs):
        candidate=(directory/'bundle/candidate.py').exists()
        kind='candidate' if candidate else 'scorer'
        self.calls.append((kind,deepcopy(request),container_name))
        inputs=request['input']
        if candidate:
            assert not (directory/'bundle/trusted_scoring').exists()
            output=payload(inputs['kind'])
        else:
            assert not (directory/'bundle/candidate.py').exists()
            output={'schema':'sciel-trusted-scorer-output/1', 'result':score_payload(inputs['kind'],base64.b64decode(inputs['payload_base64']),HostBinding(**inputs['binding'])), 'runtime':{'python':'3.11.11','numpy':'2.2.0'}}
        raw={'schema':'sciel-method-container-receipt/1',**{k:request[k] for k in BOUND},
            'status':'ok','reason':None,'invoked':True,'exit_code':0,'output':output,'output_sha256':content_hash(output),
            'limits':request['limits'],'resources':{'cpu_seconds':.125,'wall_seconds':.25,
                'memory_peak_bytes':12345,'output_bytes':len(canonical_bytes(output)),'stderr_bytes':0},
            'stderr':'','stderr_truncated':False}
        transport={'stdout':b'','stderr':'','exit_code':0,'timed_out':False,'overflow':False,
                   'dispatched':True,'removed':True,'name':container_name}
        if self.change:self.change(kind,request,raw,transport)
        raw['output_sha256']=content_hash(raw.get('output'))
        transport['stdout']=json.dumps(raw).encode()
        return transport


def setup_backend(tmp_path,monkeypatch,kind='circle',workers=1,**overrides):
    import sciel.eval.candidate_backend as backend_module
    executor=FakeDocker()
    monkeypatch.setattr(backend_module,'DockerExecutor',lambda _image:executor)
    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate',lambda *_a,**_k:pytest.fail('native evaluation forbidden'))
    cfg=config(TASKS[kind],tmp_path/'capacity',workers=workers,num_islands=2,max_iterations=2,
               evaluation=deepcopy(POLICY),**overrides)
    task=load_task(str(ROOT/'tasks'/TASKS[kind]))
    return cfg,task,executor


def evaluate_once(tmp_path,cfg,task,executor,source=b'raise RuntimeError("candidate must not execute on host")\n',confirm=False):
    from sciel.engine.capacity import HostCapacity
    backend=CandidateBackend(apply_defaults(cfg),task,executor=executor)
    ledger=parallel_calls.CallLedger(tmp_path/'wave',HostCapacity(str(tmp_path/'caps'),llm_slots=1,evaluator_slots=1,provider_limits={'fake':1}),'fake')
    evaluation_id='evaluation:test:wave:1:G:1:0'
    request={'source':source.decode(),'source_sha256':hashlib.sha256(source).hexdigest(),'task_hash':task.task_hash,
             'evaluation_id':evaluation_id,'evaluation_backend':backend.identity,'confirm_records':confirm,'output_resolution':None}
    directory,digest,cached=ledger.prepare('G:1:0','evaluation',request)
    def invoke():return backend.evaluate(source,evaluation_id=evaluation_id,confirm=confirm)
    receipt=cached or ledger.perform(directory,digest,'evaluation',invoke)
    return ledger,directory,request,receipt


@pytest.mark.parametrize('workers',[1,2])
def test_actual_engine_routes_all_evaluations_and_replays_once(tmp_path,monkeypatch,workers):
    cfg,task,executor=setup_backend(tmp_path,monkeypatch,workers=workers)
    provider=Provider('run_packing',concurrent=workers)
    run=tmp_path/'run';original=Engine._commit_step;crashed=False
    def stop_after_result(self,*args,**kwargs):
        nonlocal crashed
        if not crashed:crashed=True;raise RuntimeError('after returned safe result')
        return original(self,*args,**kwargs)
    monkeypatch.setattr(Engine,'_commit_step',stop_after_result)
    def build(resume=False):
        e=build_engine(cfg,str(ROOT/'tasks'),str(run),llm=provider)
        if resume:checkpoint.load(e,str(run/'checkpoint.json'))
        return e
    with pytest.raises(CampaignStopped,match="after returned safe result"):build().run()
    physical_before=len(executor.calls);llm_before=len(provider.calls)
    assert physical_before==4*(1+workers)  # seed and each G: candidate+scorer for smoke/full
    done=build(True).run()
    assert done['iteration_no']==2
    assert len(executor.calls)==12 and len(provider.calls)==2
    if workers==2:assert (len(executor.calls),len(provider.calls))==(physical_before,llm_before)
    rows=[row for row in events(run) if row.get('policy')=='evaluation' and row.get('action')=='result']
    assert len(rows)==3 and len({r['evaluation_id'] for r in rows})==3
    assert all(row['execution']['resources']['cpu_seconds']==.5 for row in rows)
    assert all(row['execution']['resources']['cpu_complete'] is True for row in rows)
    cp=json.loads((run/'checkpoint.json').read_text())
    assert cp['identity']['evaluation_backend']['policy']==POLICY
    assert cp['n_evaluation_results']==3


@pytest.mark.parametrize('kind,count,reached',[('circle',4,'full'),('heil',10,'confirm'),('beat',4,'full')])
def test_stage_contract_and_distinct_physical_receipts(tmp_path,monkeypatch,kind,count,reached):
    cfg,task,executor=setup_backend(tmp_path,monkeypatch,kind)
    ledger,directory,request,receipt=evaluate_once(tmp_path,cfg,task,executor,confirm=True)
    outcome=ledger.result('evaluation',receipt)
    assert outcome.reached==reached and outcome.confirmed==(kind=='heil')
    assert len(executor.calls)==count and len({x[2] for x in executor.calls})==count
    assert outcome.execution['resources']['cpu_seconds']==count*.125
    if kind=='heil':
        assert outcome.result.cases['n_trials']==3.
        assert json.loads(outcome.result.artifacts['confirmation_selection'])['selected_trial']==0
    verify_execution(outcome.execution,directory,request,receipt['request_sha256'],outcome=receipt['value'])
    before=len(executor.calls)
    _,_,cached=ledger.prepare('G:1:0','evaluation',request)
    assert ledger.result('evaluation',cached).execution==outcome.execution and len(executor.calls)==before


def test_known_failed_confirmation_still_invokes_all_three(tmp_path,monkeypatch):
    cfg,task,executor=setup_backend(tmp_path,monkeypatch,'heil')
    def fail(kind,request,raw,transport):
        if len(executor.calls)==5:raw.update(status='failed',reason='candidate_failed',exit_code=1,output=None)
    executor.change=fail
    ledger,_,_,receipt=evaluate_once(tmp_path,cfg,task,executor,confirm=True)
    outcome=ledger.result('evaluation',receipt)
    assert not outcome.result.ok and not outcome.confirmed
    assert outcome.result.cases['n_trials']==3.
    assert json.loads(outcome.result.artifacts['confirmation_selection'])['selected_trial']==0
    assert len(executor.calls)==9  # failed candidate has no scorer, remaining trials do run


@pytest.mark.parametrize('which',['candidate','scorer'])
def test_cleanup_unknown_stays_fatal_and_never_replays(tmp_path,monkeypatch,which):
    cfg,task,executor=setup_backend(tmp_path,monkeypatch)
    def unknown(kind,request,raw,transport):
        if kind==which:transport['removed']=False
    executor.change=unknown
    ledger,directory,request,receipt=evaluate_once(tmp_path,cfg,task,executor)
    assert receipt['status']=='error'
    with pytest.raises(EvaluationInfrastructureError):ledger.result('evaluation',receipt)
    e=receipt['error']['candidate_evaluation']['execution']
    assert e['status']=='unknown' and e['resources']['cpu_seconds'] is None
    assert e['resources']['known_cpu_seconds']>=.125
    before=len(executor.calls)
    _,_,cached=ledger.prepare('G:1:0','evaluation',request)
    with pytest.raises(EvaluationInfrastructureError):ledger.result('evaluation',cached)
    assert len(executor.calls)==before


@pytest.mark.parametrize('change',['score','hash','duplicate','shape'])
def test_candidate_cannot_supply_host_authority(tmp_path,monkeypatch,change):
    cfg,task,executor=setup_backend(tmp_path,monkeypatch)
    def mutate(kind,request,raw,transport):
        if kind!='candidate':return
        value=json.loads(raw['output'])
        if change=='score':value['score']=99999
        if change=='hash':value['source_sha256']='0'*64
        if change=='shape':value['values']['centers']=[]
        raw['output']=json.dumps(value)
        if change=='duplicate':raw['output']=raw['output'].replace('"schema":','"schema":"forged","schema":',1)
    executor.change=mutate
    ledger,_,_,receipt=evaluate_once(tmp_path,cfg,task,executor)
    outcome=ledger.result('evaluation',receipt)
    assert not outcome.result.ok and outcome.result.score==0 and outcome.reached=='smoke'
    assert len(executor.calls)==2 and outcome.execution['resources']['cpu_complete']


@pytest.mark.parametrize('field',['task_hash','stochastic','confirm_n','smoke_timeout','cpu_seconds'])
def test_effective_taskspec_cannot_bypass_pins(tmp_path,monkeypatch,field):
    cfg,task,_=setup_backend(tmp_path,monkeypatch)
    setattr(task,field, '0'*64 if field=='task_hash' else True if field=='stochastic' else 999)
    with pytest.raises(ValueError,match='TaskSpec'):CandidateBackend(apply_defaults(cfg),task,executor=FakeDocker())


def test_serial_enabled_false_fails_before_provider_or_run_directory(tmp_path,monkeypatch):
    cfg,_,_=setup_backend(tmp_path,monkeypatch)
    cfg['parallel_search']['enabled']=False
    monkeypatch.setattr('sciel.engine.factory.build_llm',lambda *_:pytest.fail('provider constructed'))
    with pytest.raises(ValueError,match='parallel_search.enabled'):build_engine(cfg,str(ROOT/'tasks'),str(tmp_path/'run'))
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('target',['receipt','intent','dispatch'])
def test_lost_evidence_is_typed_fatal_not_new_dispatch(tmp_path,monkeypatch,target):
    cfg,task,executor=setup_backend(tmp_path,monkeypatch)
    ledger,directory,request,receipt=evaluate_once(tmp_path,cfg,task,executor)
    name={'receipt':'smoke-0.receipt.json','intent':'smoke-0.intent.json','dispatch':'dispatched.json'}[target]
    (directory/name).unlink()
    with pytest.raises(EvaluationInfrastructureError):ledger.prepare('G:1:0','evaluation',request)
    assert len(executor.calls)==4


@pytest.mark.parametrize('phase',['smoke_final_scorer','confirm_final_scorer'])
def test_shared_deadline_rejects_late_final_scorer_return(tmp_path,monkeypatch,phase):
    from types import SimpleNamespace
    import sciel.eval.candidate_backend as module
    cfg,task,executor=setup_backend(tmp_path,monkeypatch,'heil')
    clock=SimpleNamespace(now=0.)
    monkeypatch.setattr(module,'time',SimpleNamespace(monotonic=lambda:clock.now))
    def delayed(kind,request,raw,transport):
        if kind=='scorer' and ((phase=='smoke_final_scorer' and len(executor.calls)==2)
                or (phase=='confirm_final_scorer' and len(executor.calls)==10)):
            # Simulates create/attach/cleanup consuming the final stage budget,
            # although the child itself returned a valid successful payload.
            clock.now+=task.smoke_timeout+1 if phase.startswith('smoke') else task.confirm_timeout+1
    executor.change=delayed
    ledger,_,request,receipt=evaluate_once(tmp_path,cfg,task,executor,confirm=True)
    outcome=ledger.result('evaluation',receipt)
    assert not outcome.result.ok and not outcome.confirmed
    assert len(executor.calls)==(2 if phase.startswith('smoke') else 10)
    assert outcome.execution['resources']['cpu_seconds']==len(executor.calls)*.125
    assert 'resource budget exhausted' in outcome.result.artifacts['error']
    _,_,cached=ledger.prepare('G:1:0','evaluation',request)
    assert not ledger.result('evaluation',cached).result.ok


def test_confirmation_cpu_is_one_stage_allowance_not_reset_per_call(tmp_path,monkeypatch):
    cfg,task,executor=setup_backend(tmp_path,monkeypatch,'heil')
    cfg['evaluation']['cpu_seconds']=.4
    ledger,_,request,receipt=evaluate_once(tmp_path,cfg,task,executor,confirm=True)
    outcome=ledger.result('evaluation',receipt)
    assert not outcome.result.ok and not outcome.confirmed and outcome.reached=='confirm'
    assert len(executor.calls)==8  # two confirmation trials use up the shared cap
    confirm_limits=[call[1]['limits']['cpu_seconds'] for call in executor.calls[4:]]
    assert confirm_limits==pytest.approx([.4,.275,.15,.025])
    assert outcome.execution['resources']['cpu_seconds']==1.
    _,_,cached=ledger.prepare('G:1:0','evaluation',request)
    assert not ledger.result('evaluation',cached).confirmed


def test_stop_during_candidate_drains_one_cascade_then_replays_without_reexecution(tmp_path,monkeypatch):
    cfg,task,executor=setup_backend(tmp_path,monkeypatch)
    cfg['max_iterations']=1
    provider=Provider('run_packing');run=tmp_path/'run';holder={}
    def stop(kind,request,raw,transport):
        if len(executor.calls)==1:holder['engine'].request_stop()
    executor.change=stop
    engine=build_engine(cfg,str(ROOT/'tasks'),str(run),llm=provider);holder['engine']=engine
    assert engine.run()['reason']=='paused'
    assert len(executor.calls)==4 and not provider.calls
    cp=json.loads((run/'checkpoint.json').read_text())
    assert cp['db']['pop_best_id'] is None
    restored=build_engine(cfg,str(ROOT/'tasks'),str(run),llm=provider)
    checkpoint.load(restored,str(run/'checkpoint.json'))
    result=restored.run()
    assert result['iteration_no']==1 and len(executor.calls)==8 and len(provider.calls)==1


@pytest.mark.parametrize('mutate',['scorer','image','cpu'])
def test_checkpoint_backend_identity_cannot_be_forced(tmp_path,monkeypatch,mutate):
    cfg,task,executor=setup_backend(tmp_path,monkeypatch)
    cfg['max_iterations']=1
    provider=Provider('run_packing');run=tmp_path/'run'
    build_engine(cfg,str(ROOT/'tasks'),str(run),llm=provider).run()
    engine=build_engine(cfg,str(ROOT/'tasks'),str(run),llm=provider)
    identity=engine.evaluation_backend.identity
    if mutate=='scorer':identity['source_files']['eval/trusted_scoring/circle.py']='0'*64
    elif mutate=='image':identity['executor']['image_id']='sha256:'+'0'*64
    else:identity['policy']['cpu_seconds']=4.
    before=len(executor.calls)
    try:
        with pytest.raises(checkpoint.IdentityMismatch,match='candidate-only'):
            checkpoint.load(engine,str(run/'checkpoint.json'),allow_identity_mismatch=True)
    finally:engine.close()
    assert len(executor.calls)==before


def test_context_invariant_failure_is_infrastructure_not_candidate_failure(tmp_path,monkeypatch):
    cfg,task,executor=setup_backend(tmp_path,monkeypatch)
    backend=CandidateBackend(apply_defaults(cfg),task,executor=executor)
    with pytest.raises(EvaluationInfrastructureError,match='durable'):
        backend.evaluate(b'never executed',evaluation_id='test',confirm=False)
    assert not executor.calls


@pytest.mark.parametrize('which',['candidate','scorer'])
def test_proven_pretransport_refusal_retains_known_accounting(tmp_path,monkeypatch,which):
    from sciel.methods.executor import ExecutionRefused
    cfg,task,executor=setup_backend(tmp_path,monkeypatch)
    execute=executor.execute
    def refuse(bundle,*args,**kwargs):
        target='candidate' if 'candidate.py' in bundle['files'] else 'scorer'
        if target==which:raise ExecutionRefused('inert proven pretransport fixture')
        return execute(bundle,*args,**kwargs)
    monkeypatch.setattr(executor,'execute',refuse)
    ledger,_,request,receipt=evaluate_once(tmp_path,cfg,task,executor)
    assert receipt['status']=='error'
    execution=receipt['error']['candidate_evaluation']['execution']
    assert execution['resources']['cpu_complete'] is True
    assert execution['resources']['cpu_seconds']==(0. if which=='candidate' else .125)
    assert len(executor.calls)==(0 if which=='candidate' else 1)
    _,_,cached=ledger.prepare('G:1:0','evaluation',request)
    with pytest.raises(EvaluationInfrastructureError):ledger.result('evaluation',cached)


@pytest.mark.parametrize('raw',['', 'x'*(128*1024+1), '\ud800'])
def test_invalid_raw_candidate_string_is_a_known_output_rejection(tmp_path,monkeypatch,raw):
    cfg,task,executor=setup_backend(tmp_path,monkeypatch)
    def invalid(kind,request,value,transport):
        if kind=='candidate':value['output']=raw
    executor.change=invalid
    ledger,_,request,receipt=evaluate_once(tmp_path,cfg,task,executor)
    result=ledger.result('evaluation',receipt)
    assert not result.result.ok and result.result.score==0 and len(executor.calls)==1
    _,_,cached=ledger.prepare('G:1:0','evaluation',request)
    assert not ledger.result('evaluation',cached).result.ok


def test_evaluator_lease_covers_every_candidate_and_scorer_until_cascade_returns(tmp_path,monkeypatch):
    from sciel.engine.capacity import HostCapacity,CapacityTimeout
    cfg,task,executor=setup_backend(tmp_path,monkeypatch,'heil')
    observer=HostCapacity(str(tmp_path/'caps'),llm_slots=1,evaluator_slots=1,provider_limits={'fake':1})
    def assert_held(kind,request,value,transport):
        with pytest.raises(CapacityTimeout):observer.acquire('evaluator',timeout=0.)
    executor.change=assert_held
    ledger,_,_,receipt=evaluate_once(tmp_path,cfg,task,executor,confirm=True)
    assert ledger.result('evaluation',receipt).confirmed and len(executor.calls)==10
    with observer.acquire('evaluator',timeout=0.) as lease:assert not lease.released


def test_post_generation_unknown_cleanup_is_not_quality_or_conclusion(tmp_path,monkeypatch):
    cfg,task,executor=setup_backend(tmp_path,monkeypatch)
    provider=Provider('run_packing');run=tmp_path/'run'
    def unknown(kind,request,value,transport):
        if len(executor.calls)==5:transport['removed']=False
    executor.change=unknown
    with pytest.raises(CampaignStopped):build_engine(cfg,str(ROOT/'tasks'),str(run),llm=provider).run()
    assert len(provider.calls)==1 and len(executor.calls)==5
    journal=events(run)
    assert not any(row.get('policy')=='evaluation' and row.get('action')=='result' and row.get('evaluation_id','').endswith('G:0:0') for row in journal)
    assert not any(row.get('event')=='step' for row in journal)
    cp=json.loads((run/'checkpoint.json').read_text())
    assert cp['iteration_no']==0 and cp['n_evaluation_results']==1
    # Existing paid G is retained, but the unresolved candidate cannot become a
    # measured failure, conclusion request or a new physical evaluation.
    again=build_engine(cfg,str(ROOT/'tasks'),str(run),llm=provider)
    checkpoint.load(again,str(run/'checkpoint.json'))
    with pytest.raises(CampaignStopped):again.run()
    assert len(provider.calls)==1 and len(executor.calls)==5


def test_direct_malformed_saved_execution_decode_is_typed_fatal():
    with pytest.raises(EvaluationInfrastructureError):
        parallel_calls.CallLedger.result('evaluation',{'status':'returned','value':{'execution':None}})
