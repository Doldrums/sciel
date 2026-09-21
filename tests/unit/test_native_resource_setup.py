"""Resource controls and all evaluation/provider leaves are inert mocks."""
import copy
from contextlib import nullcontext
import hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from sciel.campaign.budget import CampaignStopped
from sciel.eval import _runner,sandbox
from sciel.eval.facade import (EvalResult,EvaluationSetupError,SETUP_FAILURE_SCHEMA,
                              setup_failure)
from sciel.engine.parallel_calls import CallLedger,CooperativeIO
from sciel.engine.worker import run_worker


def fault():
    return EvaluationSetupError(setup_failure('setrlimit_failed','mem_bytes',2,1048576,.012))


@pytest.mark.parametrize('name,exception', [('cpu_seconds',OSError),('mem_bytes',ValueError),
                                           ('cpu_seconds',OverflowError)])
def test_runner_refused_limit_never_loads_evaluator(monkeypatch,tmp_path,name,exception):
    load=Mock(side_effect=AssertionError('must not import evaluator'))
    apply=Mock(side_effect=exception('private mock setup failure'))
    monkeypatch.setitem(sys.modules,'resource',NS(RLIMIT_CPU=0,RLIMIT_AS=9,setrlimit=apply))
    monkeypatch.setattr(_runner,'os',NS(name='posix'))
    monkeypatch.setattr(_runner,'_load',load)
    path=tmp_path/'out.json'
    option='--cpu-seconds' if name=='cpu_seconds' else '--mem-bytes'
    _runner.main(['runner','unused','full','unused',str(path),option,'2'])
    row=json.loads(path.read_text());diagnostic=EvaluationSetupError(row).diagnostic
    assert diagnostic['code']=='setrlimit_failed' and diagnostic['resource']==name
    assert diagnostic['execution']==diagnostic['resource_usage']=='unknown'
    assert diagnostic['reported_wall_seconds']>=0 and 'private' not in json.dumps(row)
    assert 'ok' not in row and 'score' not in row and 'timing' not in row
    load.assert_not_called();apply.assert_called_once()


@pytest.mark.parametrize('platform,resource', [('nt',None),('posix',NS()),('posix',None)])
def test_runner_unsupported_limits_never_import_candidate(monkeypatch,tmp_path,platform,resource):
    load=Mock(side_effect=AssertionError('must not import evaluator'))
    monkeypatch.setattr(_runner,'os',NS(name=platform))
    monkeypatch.setitem(sys.modules,'resource',resource)
    monkeypatch.setattr(_runner,'_load',load)
    path=tmp_path/'out.json'
    _runner.main(['runner','unused','full','unused',str(path),'--mem-bytes','1048576'])
    row=json.loads(path.read_text())
    assert row['code']==('unsupported_platform' if platform!='posix' else 'resource_unavailable')
    load.assert_not_called()


@pytest.mark.parametrize('platform',['posix','nt'])
def test_none_caps_do_not_require_resource_support(monkeypatch,tmp_path,platform):
    stage=Mock(return_value={'ok':True,'score':.5})
    monkeypatch.setattr(_runner,'os',NS(name=platform))
    monkeypatch.setitem(sys.modules,'resource',None)
    monkeypatch.setattr(_runner,'_load',Mock(return_value=NS(full=stage)))
    path=tmp_path/'out.json'
    _runner.main(['runner','unused','full','unused',str(path)])
    assert EvalResult.from_json(json.loads(path.read_text())).score==.5
    stage.assert_called_once_with('unused')


def test_setup_accepted_then_candidate_exception_is_still_candidate_failure(monkeypatch,tmp_path):
    stage=Mock(side_effect=ValueError('mock task failure'))
    apply=Mock()
    monkeypatch.setitem(sys.modules,'resource',NS(RLIMIT_CPU=0,RLIMIT_AS=9,setrlimit=apply))
    monkeypatch.setattr(_runner,'os',NS(name='posix'))
    monkeypatch.setattr(_runner,'_load',Mock(return_value=NS(full=stage)))
    path=tmp_path/'out.json'
    _runner.main(['runner','unused','full','unused',str(path),'--cpu-seconds','2'])
    result=EvalResult.from_json(json.loads(path.read_text()))
    assert result.ok is False and result.artifacts['error_type']=='ValueError'
    apply.assert_called_once_with(0,(2,3));stage.assert_called_once()


def test_parent_platform_preflight_rejects_before_transport(monkeypatch):
    # Patch only sandbox's os reference; never mutate the process os.name global.
    monkeypatch.setattr(sandbox,'os',NS(name='nt'))
    spawn=Mock(side_effect=AssertionError('must not dispatch'))
    allocate=Mock(side_effect=AssertionError('must not allocate'))
    monkeypatch.setattr(sandbox.subprocess,'Popen',spawn)
    monkeypatch.setattr(sandbox.tempfile,'mkstemp',allocate)
    with pytest.raises(EvaluationSetupError) as exc:
        sandbox.run_stage('unused','full','unused',1.,cpu_seconds=2)
    assert exc.value.diagnostic['execution']=='unknown'
    spawn.assert_not_called();allocate.assert_not_called()


def terminal(monkeypatch,payload,*,returncode=0):
    clean=[];paths=[]
    class Process:
        stderr=None
        def __init__(self):self.returncode=returncode
        def wait(self,timeout=None):return self.returncode
    def spawn(argv,**kwargs):
        paths.append(argv[5]);Path(argv[5]).write_text(json.dumps(payload));return Process()
    monkeypatch.setattr(sandbox.subprocess,'Popen',spawn)
    monkeypatch.setattr(sandbox,'_kill_group',lambda proc:clean.append(1))
    monkeypatch.setattr(sandbox,'_StderrTail',lambda _:NS(finish=lambda:''))
    with pytest.raises(EvaluationSetupError) as exc:
        sandbox.run_stage('unused','full','unused',1.,cpu_seconds=2,mem_bytes=1048576)
    assert clean and all(not Path(p).exists() for p in paths)
    return exc.value.diagnostic


@pytest.mark.parametrize('returncode',[0,1,78])
def test_setup_envelope_conservatively_stops_and_keeps_reported_time(monkeypatch,returncode):
    diag=terminal(monkeypatch,fault().diagnostic,returncode=returncode)
    assert diag==fault().diagnostic
    assert diag['reported_wall_seconds']==.012
    assert diag['execution']==diag['resource_usage']=='unknown'


@pytest.mark.parametrize('field,value',[('code','unknown'),('execution','not_started'),
    ('resource_usage',0),('requested_limits',{}),('reported_wall_seconds',-1)])
def test_malformed_setup_marker_never_becomes_bad_candidate(monkeypatch,field,value):
    payload=fault().diagnostic;payload[field]=value
    diag=terminal(monkeypatch,payload)
    assert diag['code'] in ('malformed_report','request_mismatch')
    assert diag['execution']==diag['resource_usage']=='unknown'


def test_report_cannot_rebind_requested_limits(monkeypatch):
    payload=fault().diagnostic;payload['requested_limits']['cpu_seconds']=3
    diag=terminal(monkeypatch,payload)
    assert diag['code']=='request_mismatch'
    assert diag['requested_limits']=={'cpu_seconds':2,'mem_bytes':1048576}
    assert diag['reported_wall_seconds']==.012


def test_error_type_alone_is_not_a_setup_marker(tmp_path):
    path=tmp_path/'result.json'
    path.write_text(json.dumps(EvalResult(False,0.,artifacts={'error_type':'EvaluationSetupError'}).to_json()))
    result=sandbox._read_result(path)
    assert result.ok is False and result.artifacts['error_type']=='EvaluationSetupError'


def assert_unmeasured(research):
    record,=research.knowledge.experiments
    assert record['status']=='generation_stopped'
    assert record['score'] is record['actual_delta'] is record['evaluation'] is None
    assert record['forecast_verdict']=='INCONCLUSIVE'
    assert not research.queue and not research.knowledge.lessons and not research.knowledge.fragments


def test_paid_ordinary_worker_has_no_conclusion_or_repair_quality():
    from test_research_retry_feedback import setup,text
    model,research,snap,_,_,_=setup([text()]);charged=[];research.charge=charged.append
    with pytest.raises(CampaignStopped) as exc:
        run_worker('diff',snap,model,lambda _:(_ for _ in ()).throw(fault()),dup_cap=1,research=research)
    assert isinstance(infrastructure_cause(exc.value),EvaluationSetupError)
    assert_unmeasured(research)
    assert len(model.calls)==1 and charged==[.01]
    assert research.knowledge.experiments[0]['generation_cost']==.01


def test_legacy_worker_preserves_known_paid_response_on_setup_stop():
    from test_research_retry_feedback import setup,text
    model,_,snap,_,_,_=setup([text()],research_on=False)
    with pytest.raises(EvaluationSetupError) as exc:
        run_worker('diff',snap,model,lambda _:(_ for _ in ()).throw(fault()),dup_cap=1)
    assert len(model.calls)==1 and exc.value.unbooked_known_cost==.01
    assert exc.value.current_call_cost_unknown is False and exc.value.generation_returned is True


def test_paid_idea_preserves_both_calls_without_conclusion():
    from test_two_stage_idea import setup,response,CARD,code_response
    research,idea,_,model,_,charged,_,run=setup([
        response(json.dumps(CARD),.03),response(code_response(),.07)])
    with pytest.raises(CampaignStopped) as exc:run(evaluator=lambda _:(_ for _ in ()).throw(fault()))
    assert isinstance(infrastructure_cause(exc.value),EvaluationSetupError)
    assert_unmeasured(research)
    assert len(model.calls)==2 and sum(charged)==pytest.approx(.10)
    assert idea.known_cost==pytest.approx(.10)


@pytest.mark.parametrize('route',['circle','intervention'])
def test_real_structured_engine_setup_stop_has_no_C_RL_or_admission(tmp_path,route):
    if route=='circle':
        from test_circle_generation_engine import build,response,COST
        engine=build(tmp_path/'run',[response()])
    else:
        from test_circle_intervention_engine import build_engine,plan,COST
        engine=build_engine(tmp_path/'run',[plan()])
    evaluate=engine._evaluate;count=0
    def isolated(source):
        nonlocal count
        count+=1
        if count==1:return evaluate(source)
        raise fault()
    engine._evaluate=isolated
    engine.reward.improvement_event=Mock(wraps=engine.reward.improvement_event)
    with pytest.raises(CampaignStopped) as exc:engine.run()
    assert isinstance(infrastructure_cause(exc.value),EvaluationSetupError)
    engine.reward.improvement_event.assert_not_called()
    assert count==2 and len(engine.llm.calls)==1 and engine.cost_total==pytest.approx(COST)
    assert engine.B_evals==engine.n_eval_fail==0 and engine.lineage_fail=={}
    assert len(engine.db._all)==1
    assert_unmeasured(engine.research)


@pytest.mark.parametrize('research_on',[False,True])
def test_initial_setup_failure_admits_no_seed_or_fake_zero(tmp_path,monkeypatch,research_on):
    from test_output_novelty_engine import setup
    cfg={'enabled':research_on,'max_aux_calls':0,'reflection_interval':0}
    build,provider,_,_=setup(tmp_path,monkeypatch,[1],{},parallel=False,research=cfg)
    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate',lambda *a,**k:(_ for _ in ()).throw(fault()))
    engine=build()
    with pytest.raises(EvaluationSetupError):engine.run()
    assert provider.calls==[] and len(engine.db._all)==0
    assert engine.B_evals==engine.n_eval_fail==engine.cost_total==0
    from test_parallel_engine import source
    assert_pending(engine.candidate_novelty,source('run_search',0))


def test_actual_WAL_keeps_typed_setup_failure_and_replay_never_dispatches(tmp_path):
    class Capacity:
        def acquire(self,*args,**kwargs):return nullcontext()
    calls=[]
    def run():
        io=CooperativeIO(CallLedger(tmp_path,Capacity(),'fixture'),1)
        def leaf():calls.append('once');raise fault()
        try:return io.run([('work',lambda:io.call('evaluation',{'task':'inert'},leaf))])
        finally:io.close()
    for _ in range(2):
        results,errors=run()
        assert results=={} and isinstance(errors['work'],EvaluationSetupError)
        assert errors['work'].diagnostic==fault().diagnostic
    assert calls==['once']
    receipt,=list((tmp_path/'calls').glob('*/receipt.json'))
    payload=json.loads(receipt.read_text())['payload']
    assert payload['status']=='error' and payload['not_dispatched'] is False
    assert payload['error']['evaluation_setup']==fault().diagnostic
    assert payload['elapsed_seconds']>=0


@pytest.mark.parametrize('damage',['code','unknown_execution','extra'])
def test_cold_WAL_diagnostic_tamper_stays_fatal_not_quality(damage):
    diag=fault().diagnostic
    if damage=='code':diag['code']='transient_retry'
    elif damage=='unknown_execution':diag['execution']='not_started'
    else:diag['extra']='unbounded-private-data'
    with pytest.raises(EvaluationSetupError) as exc:
        CallLedger.result('evaluation',{'status':'error','not_dispatched':False,
                                       'error':{'evaluation_setup':diag}})
    assert exc.value.diagnostic==setup_failure('malformed_report','unknown')


@pytest.mark.parametrize('parallel,novelty_on',[(False,False),(False,True),(True,True)])
@pytest.mark.parametrize('research_on',[False,True])
def test_real_ordinary_engine_paid_setup_stop_and_cold_replay(tmp_path,monkeypatch,parallel,research_on,novelty_on):
    import test_output_novelty_engine as fixture
    from test_output_novelty_engine import setup
    original_config=fixture.config
    def config(*args,**kwargs):
        if not novelty_on:
            kwargs['candidate_novelty']={**kwargs['candidate_novelty'],'enabled':False,'output_repeats':False}
        return original_config(*args,**kwargs)
    monkeypatch.setattr(fixture,'config',config)
    import sciel.engine.loop as loop
    cfg={'enabled':research_on,'max_aux_calls':0,'reflection_interval':0}
    build,provider,evaluations,directory=setup(tmp_path,monkeypatch,[1],{},parallel=parallel,research=cfg)
    original_generate=provider.generate
    def generate(*args,**kwargs):
        response=original_generate(*args,**kwargs)
        if research_on:
            response.text='HYPOTHESIS_JSON: '+json.dumps(dict(hypothesis='Change inert fixture source.',
                predicted_low=-1,predicted_high=1,reasoning='Unmeasured fixture.',confidence=.5))+'\n'+response.text
        return response
    provider.generate=generate
    evaluate=loop.cascade_evaluate;attempts=[]
    def fail_after_seed(task,path,**kwargs):
        code=Path(path).read_text();attempts.append(code)
        if len(attempts)==1:return evaluate(task,path,**kwargs)
        raise fault()
    monkeypatch.setattr(loop,'cascade_evaluate',fail_after_seed)
    engine=build()
    engine.reward.improvement_event=Mock(wraps=engine.reward.improvement_event)
    with pytest.raises((CampaignStopped,EvaluationSetupError)) as exc:engine.run()
    if not parallel:assert isinstance(infrastructure_cause(exc.value),EvaluationSetupError)
    engine.reward.improvement_event.assert_not_called()
    assert len(provider.calls)==1 and len(attempts)==2 and len(evaluations)==1
    assert engine.B_evals==engine.n_eval_fail==0 and engine.lineage_fail=={}
    assert len(engine.db._all)==1
    if not parallel:
        assert engine.cost_total==pytest.approx(.03)
        if novelty_on:assert_pending(engine.candidate_novelty,attempts[-1])
        if research_on:assert_unmeasured(engine.research)
    else:
        receipts=[json.loads(p.read_text())['payload'] for p in (directory/'.parallel').glob('wave-*/calls/*/receipt.json')]
        returned=[p for p in receipts if p['status']=='returned' and 'text' in p.get('value',{})]
        failed=[p for p in receipts if p['status']=='error' and 'evaluation_setup' in p.get('error',{})]
        assert len(returned)==len(failed)==1
        assert returned[0]['value']['cost']==.03 and failed[0]['error']['evaluation_setup']==fault().diagnostic
        before={str(p):p.read_bytes() for p in (directory/'.parallel').glob('wave-*/calls/*/receipt.json')}
        with pytest.raises(CampaignStopped):build(resume=True).run()
        assert len(provider.calls)==1 and len(attempts)==2
        assert {str(p):p.read_bytes() for p in (directory/'.parallel').glob('wave-*/calls/*/receipt.json')}==before
        # A failed uncommitted wave retains its paid response exactly once in
        # immutable WAL; do not fabricate an admitted/population cost record.
        assert not engine.research or not engine.research.queue


def infrastructure_cause(error):
    from sciel.eval.facade import EvaluationInfrastructureError
    seen=set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error,EvaluationInfrastructureError):return error
        error=error.__cause__
    return None


def assert_pending(registry,source):
    from sciel.engine.novelty import CandidateNoveltyRegistry
    sha=hashlib.sha256(source.encode()).hexdigest()
    rows=[r for r in registry.state()['records'] if r['source_sha256']==sha]
    assert len(rows)==1 and rows[0]['evaluation_ok'] is None
    restored=CandidateNoveltyRegistry(registry.neighbors);restored.load_state(registry.state())
    for current in [registry,restored]:
        for remeasurement in [False,True]:
            check=current.inspect(source,remeasurement=remeasurement)
            assert check['accept'] is False and check['reason']=='previous_evaluation_pending_or_unknown'
            assert check['neighbors'][0]['evaluation_ok'] is None


def test_unexpected_exception_during_setup_is_infrastructure(monkeypatch,tmp_path):
    load=Mock(side_effect=AssertionError('must not import evaluator'))
    monkeypatch.setattr(_runner,'_apply_limits',Mock(side_effect=RuntimeError('private failure')))
    monkeypatch.setattr(_runner,'_load',load)
    path=tmp_path/'out.json'
    _runner.main(['runner','unused','full','unused',str(path)])
    row=json.loads(path.read_text())
    assert row['code']=='setup_exception' and row['resource']=='unknown'
    assert 'private' not in json.dumps(row)
    load.assert_not_called()


def test_candidate_phase_cannot_relabel_exception_type_as_setup(monkeypatch,tmp_path):
    stage=Mock(side_effect=_runner._ResourceSetupError('setrlimit_failed','cpu_seconds'))
    monkeypatch.setattr(_runner,'_load',Mock(return_value=NS(full=stage)))
    path=tmp_path/'out.json'
    _runner.main(['runner','unused','full','unused',str(path)])
    result=EvalResult.from_json(json.loads(path.read_text()))
    assert result.ok is False and result.artifacts['error_type']=='_ResourceSetupError'
    stage.assert_called_once()


def test_tool_candidate_setup_stop_retains_prior_positive_discovery_resources():
    from test_research_candidate_submission import SOURCE,provenance,snapshot
    from sciel.engine.research_candidate import evaluate_submission
    saved=provenance();before=copy.deepcopy(saved);evaluate=Mock(side_effect=fault())
    with pytest.raises(EvaluationSetupError):evaluate_submission(SOURCE,saved,snapshot(),evaluate)
    evaluate.assert_called_once_with(SOURCE)
    assert saved==before and saved['discovery_resources']['cpu_seconds']==80.5


@pytest.mark.parametrize('status',['unknown','not_dispatched'])
def test_generic_infrastructure_base_keeps_unknown_reservation_and_known_no_dispatch_distinct(tmp_path,monkeypatch,status):
    from sciel.eval.facade import EvaluationInfrastructureError
    from test_output_novelty_engine import setup
    from test_parallel_engine import source,events
    class BackendFailure(EvaluationInfrastructureError):
        execution_status=status
    build,_,_,directory=setup(tmp_path,monkeypatch,[1],{},parallel=False)
    engine=build()
    failed=BackendFailure('fixed mock diagnostic')
    leaf=Mock(side_effect=failed)
    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate',leaf)
    code=source('run_search',7)
    with pytest.raises(BackendFailure):engine._evaluate(code)
    if status=='unknown':assert_pending(engine.candidate_novelty,code)
    else:
        row,=engine.candidate_novelty.state()['records']
        assert row['evaluation_ok'] is False
        assert engine.candidate_novelty.inspect(code)['accept'] is True
    actions=[r['action'] for r in events(directory) if r.get('policy')=='candidate_measurement']
    assert actions==['start','unknown' if status=='unknown' else 'failed']
    leaf.assert_called_once()


@pytest.mark.parametrize('damage',['invalid_enum','wrong_type','null','missing'])
def test_real_engine_malformed_native_WAL_stays_infrastructure_with_paid_replay(tmp_path,monkeypatch,damage):
    from test_output_novelty_engine import setup
    from sciel.engine.research import ResearchController
    import sciel.engine.parallel_calls as calls
    import sciel.engine.loop as loop
    cfg={'enabled':True,'max_aux_calls':0,'reflection_interval':0}
    build,provider,evaluations,directory=setup(tmp_path,monkeypatch,[1],{},parallel=True,research=cfg)
    original_generate=provider.generate
    def generate(*args,**kwargs):
        response=original_generate(*args,**kwargs)
        response.text='HYPOTHESIS_JSON: '+json.dumps(dict(hypothesis='Change inert fixture source.',
            predicted_low=-1,predicted_high=1,reasoning='Unmeasured fixture.',confidence=.5))+'\n'+response.text
        return response
    provider.generate=generate
    original_evaluate=loop.cascade_evaluate;attempts=[]
    def evaluate(task,path,**kwargs):
        attempts.append(Path(path).read_text())
        if len(attempts)==1:return original_evaluate(task,path,**kwargs)
        raise fault()
    monkeypatch.setattr(loop,'cascade_evaluate',evaluate)
    statuses=[];original_resolve=ResearchController.resolve
    def resolve(self,*args,**kwargs):
        statuses.append(kwargs.get('status'))
        return original_resolve(self,*args,**kwargs)
    monkeypatch.setattr(ResearchController,'resolve',resolve)
    original_write=calls.durable_json
    def fixture_write(path,value):
        # Simulate a structurally valid stored terminal diagnostic from a faulty
        # older writer, before its first seal. No committed receipt is rewritten.
        payload=value.get('payload',{})
        if path.name=='receipt.json' and payload.get('error',{}).get('cause_type')=='EvaluationSetupError':
            metadata=payload['error']
            if damage=='invalid_enum':metadata['evaluation_setup']['code']='not_supported'
            elif damage=='wrong_type':metadata['evaluation_setup']=[]
            elif damage=='null':metadata['evaluation_setup']=None
            else:del metadata['evaluation_setup']
            value['sha256']=calls.digest(payload)
        return original_write(path,value)
    monkeypatch.setattr(calls,'durable_json',fixture_write)
    engine=build();engine.reward.improvement_event=Mock(wraps=engine.reward.improvement_event)
    with pytest.raises(CampaignStopped):engine.run()
    assert statuses==['generation_stopped'] and not engine.research.queue
    assert len(provider.calls)==1 and len(attempts)==2 and len(evaluations)==1
    assert engine.B_evals==engine.n_eval_fail==0 and len(engine.db._all)==1
    engine.reward.improvement_event.assert_not_called()
    records=[json.loads(p.read_bytes())['payload'] for p in (directory/'.parallel').glob('wave-*/calls/*/receipt.json')]
    paid=[r for r in records if r['status']=='returned' and 'text' in r.get('value',{})]
    errors=[r for r in records if r['status']=='error']
    assert len(paid)==len(errors)==1 and paid[0]['value']['cost']==.03
    assert errors[0]['elapsed_seconds']>=0
    with pytest.raises(EvaluationSetupError) as exc:calls.CallLedger.result('evaluation',errors[0])
    assert exc.value.diagnostic==setup_failure('malformed_report','unknown')
    before={str(p):p.read_bytes() for p in (directory/'.parallel').glob('wave-*/calls/*/receipt.json')}
    with pytest.raises(CampaignStopped):build(resume=True).run()
    assert all(s=='generation_stopped' for s in statuses)
    assert len(provider.calls)==1 and len(attempts)==2
    assert {str(p):p.read_bytes() for p in (directory/'.parallel').glob('wave-*/calls/*/receipt.json')}==before


def test_legacy_untyped_other_error_is_not_reclassified():
    with pytest.raises(RuntimeError) as exc:
        CallLedger.result('evaluation',{'status':'error','not_dispatched':False,'error':{'cause_type':'OtherError'}})
    from sciel.eval.facade import EvaluationInfrastructureError
    assert not isinstance(exc.value,EvaluationInfrastructureError)


def test_real_tool_candidate_fatal_receipt_preserves_paid_discovery_and_cold_replay(tmp_path,monkeypatch):
    from test_research_tool_engine import FakeTools,setup_tools
    from test_research_candidate_submission import submit_fixture_candidate
    import sciel.engine.loop as loop
    original_execute=FakeTools.execute
    def execute(self,plan,snap,llm,*,io,candidate_evaluator):
        episode=original_execute(self,plan,snap,llm,io=io)
        source='VALUE = 10\ndef run_search():\n    return VALUE\n'
        submit_fixture_candidate(episode,source,candidate_evaluator)
        return episode
    monkeypatch.setattr(FakeTools,'execute',execute)
    build,provider,evaluations,directory=setup_tools(tmp_path,monkeypatch,workers=1,max_iterations=1,num_islands=1)
    original_evaluate=loop.cascade_evaluate;attempts=[]
    def evaluate(task,path,**kwargs):
        attempts.append(Path(path).read_text())
        if len(attempts)==1:return original_evaluate(task,path,**kwargs)
        raise fault()
    monkeypatch.setattr(loop,'cascade_evaluate',evaluate)
    engine=build();engine.reward.improvement_event=Mock(wraps=engine.reward.improvement_event)
    with pytest.raises(CampaignStopped):engine.run()
    assert engine.B_evals==engine.n_eval_fail==0 and len(engine.db._all)==1
    engine.reward.improvement_event.assert_not_called()
    assert engine.research_tools.committed==[] and len(FakeTools.invocations)==1
    records=[json.loads(p.read_bytes())['payload'] for p in (directory/'.parallel').glob('wave-*/calls/*/receipt.json')]
    paid=[r for r in records if r['status']=='returned' and 'text' in r.get('value',{})]
    tool=[r for r in records if r['status']=='returned' and 'cpu_seconds' in r.get('value',{})]
    failed=[r for r in records if r['status']=='error' and 'evaluation_setup' in r.get('error',{})]
    assert len(paid)==len(tool)==len(failed)==1
    assert paid[0]['value']['cost']==.03 and tool[0]['value']['cpu_seconds']==.1
    assert len(provider.calls)==1 and len(attempts)==2 and len(evaluations)==1
    before={str(p):p.read_bytes() for p in (directory/'.parallel').glob('wave-*/calls/*/receipt.json')}
    with pytest.raises(CampaignStopped):build(resume=True).run()
    assert len(provider.calls)==1 and len(attempts)==2 and len(FakeTools.invocations)==1
    assert {str(p):p.read_bytes() for p in (directory/'.parallel').glob('wave-*/calls/*/receipt.json')}==before
