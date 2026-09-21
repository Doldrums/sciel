"""Opt-in real Docker checks. Set SCIEL_ORACLE_IMAGE to a source-bound local ID.

Only the new task and deterministic adversarial fixtures run here. No LLM calls,
old task imports, image pulls or builds happen during these tests.
"""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap
import time

import pytest


TASK = Path(__file__).resolve().parents[2]/'tasks/oe/function_minimization_oracle'


@pytest.fixture(scope='module')
def evaluator():
    if not os.environ.get('SCIEL_ORACLE_IMAGE'):
        pytest.skip('requires explicitly selected local SCIEL_ORACLE_IMAGE')
    spec=importlib.util.spec_from_file_location('oracle_isolation_evaluator', TASK/'evaluator.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.runtime.verify_runtime_image(os.environ['SCIEL_ORACLE_IMAGE'])
    return module


@pytest.fixture
def runtime(evaluator):
    return evaluator.runtime


def run(runtime, code, **kwargs):
    return runtime.run_container_trial(textwrap.dedent(code).encode(),
        runtime.instance_from_seed(b't'*32), os.environ['SCIEL_ORACLE_IMAGE'], **kwargs)


def absent(name):
    result=subprocess.run(['docker','inspect',name],capture_output=True,timeout=3)
    return result.returncode != 0


def test_real_query_count_and_forged_return_are_independently_scored(runtime):
    result=run(runtime, '''
        def run_search(oracle):
            for i in range(50):
                oracle.query(0., 0.)
            return (-100, -100, 999999999)
    ''')
    expected=runtime.objective_for(runtime.instance_from_seed(b't'*32))(0.,0.)
    assert result['status']=='ok' and result['queries']==50
    assert result['score']==1/(1+expected) and result['best_point']==(0.,0.)
    assert absent(result['container_name'])


def test_client_monkeypatch_cannot_add_a_free_observation(runtime):
    result=run(runtime, '''
        def run_search(oracle):
            oracle.query_budget=500
            for i in range(51):
                oracle.query(0.,0.)
    ''')
    assert result['status']=='protocol_error' and result['queries']==50
    assert result['score']==0 and 'budget' in result['error']


def test_candidate_cannot_read_host_or_broker_signal_pid1_or_gain_privilege(runtime, tmp_path, monkeypatch):
    secret=tmp_path/'host-secret'
    secret.write_text('private-marker')
    monkeypatch.setenv('OPENROUTER_API_KEY','not-a-real-key')
    code='''
        import os, socket
        def run_search(oracle):
            assert os.getuid()==65534 and os.getgid()==65534
            assert 'OPENROUTER_API_KEY' not in os.environ
            for path in [HOST_SECRET, '/var/run/docker.sock', '/opt/oracle/oracle_runtime.py', '/proc/1/mem', '/proc/1/environ']:
                try:
                    open(path,'rb').read(1)
                except OSError:
                    pass
                else:
                    raise RuntimeError('read escaped '+path)
            for action in [lambda: os.kill(1,9), lambda: os.setuid(0),
                           lambda: open('/opt/oracle/container_supervisor.py','w')]:
                try:
                    action()
                except OSError:
                    pass
                else:
                    raise RuntimeError('privilege escaped')
            status=open('/proc/self/status').read()
            assert 'CapEff:\\t0000000000000000' in status
            assert 'NoNewPrivs:\\t1' in status
            assert open('/sys/fs/cgroup/pids.max').read().strip()=='64'
            assert open('/sys/fs/cgroup/memory.max').read().strip()=='268435456'
            assert open('/sys/fs/cgroup/memory.swap.max').read().strip()=='0'
            quota,period=map(int,open('/sys/fs/cgroup/cpu.max').read().split())
            assert quota==period
            s=socket.socket();s.settimeout(.2)
            try:
                s.connect(('1.1.1.1',443))
            except OSError:
                pass
            else:
                raise RuntimeError('network escaped')
            oracle.query(0,0)
    '''.replace('HOST_SECRET',repr(str(secret)))
    result=run(runtime,code)
    assert result['status']=='ok',result
    assert result['queries']==1


@pytest.mark.parametrize('method',['spawn','forkserver'])
def test_nested_processes_share_host_counter(runtime,method):
    result=run(runtime,f'''
        import multiprocessing as mp
        def child(oracle):
            oracle.query(.1,.2)
        def run_search(oracle):
            p=mp.get_context({method!r}).Process(target=child,args=(oracle,))
            p.start();p.join(2)
            assert p.exitcode==0
            oracle.query(0,0)
    ''')
    assert result['status']=='ok',result
    assert result['queries']==2


def test_wall_timeout_cleans_detached_descendant(runtime):
    result=run(runtime, '''
        import os,time
        def run_search(oracle):
            oracle.query(0,0)
            if os.fork()==0:
                os.setsid()
            time.sleep(60)
    ''',timeout=.5)
    assert result['status']=='timeout',result
    assert result['queries']==1 and result['score']==0
    assert absent(result['container_name'])


def test_actual_smoke_full_and_confirm_use_new_task_only(evaluator,tmp_path,monkeypatch):
    from sciel.eval.sandbox import run_stage
    monkeypatch.setenv('SCIEL_ORACLE_PRIVATE_DIR',str(tmp_path/'private'))
    monkeypatch.setenv('SCIEL_ORACLE_COMMITMENT_DIR',str(tmp_path/'public'))
    source=tmp_path/'candidate.py'
    source.write_text('def run_search(oracle):\n    oracle.query(0,0)\n')
    for stage,n in [('smoke',1),('full',5),('confirm',20)]:
        result=run_stage(str(TASK/'evaluator.py'),stage,str(source),timeout=40)
        assert result.ok,(stage,result)
        assert result.cases['n_trials']==n and result.cases['objective_queries']==n
        assert 0<result.score<=1
        trials=json.loads(result.artifacts['trials'])
        assert len(trials)==n and all(t['status']=='ok' for t in trials)
        assert all(absent(t['container_name']) for t in trials)
    assert len(list((tmp_path/'private').glob('*.json')))==3


def test_outer_stage_sigkill_does_not_leave_daemon_container(evaluator,tmp_path,monkeypatch):
    from sciel.eval.sandbox import run_stage
    monkeypatch.setenv('SCIEL_ORACLE_PRIVATE_DIR',str(tmp_path/'private'))
    monkeypatch.setenv('SCIEL_ORACLE_COMMITMENT_DIR',str(tmp_path/'public'))
    source=tmp_path/'candidate.py'
    source.write_text('import os,time\ndef run_search(oracle):\n    oracle.query(0,0)\n    if os.fork()==0:\n        os.setsid()\n    time.sleep(60)\n')
    image=os.environ['SCIEL_ORACLE_IMAGE']
    def names():
        return set(subprocess.check_output(['docker','ps','-a','--filter','ancestor='+image,
            '--format','{{.Names}}'],text=True,timeout=3).splitlines())
    before=names()
    result=run_stage(str(TASK/'evaluator.py'),'smoke',str(source),timeout=1.)
    assert not result.ok and result.artifacts['error_type']=='TimeoutError'
    # Confirm the required failure path occurred: the stage died while its daemon
    # container was still alive, without host-side finally cleanup.
    orphaned=names()-before
    assert orphaned and all(name.startswith('sciel-oracle-') for name in orphaned)
    deadline=time.monotonic()+7
    while names()-before and time.monotonic()<deadline:
        time.sleep(.1)
    assert not names()-before
