"""Real cheap fresh-process fixtures; no optimizer, evaluator, network or model."""
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
import subprocess

import numpy as np
import pytest

from sciel.eval import repeat_draw as adapter
from sciel.eval import packing_geometry as geometry

MEMORY = 4 * 1024**3
LIMITS = dict(seed=19, timeout=4.0, cpu_seconds=3, mem_bytes=MEMORY, certification_seconds=2.0)
VALID = """
def run_packing():
    centers = [[(i % 5 + .5)/5, (i // 5 + .5)/5] for i in range(25)] + [[0., 0.]]
    return centers, [.01]*25+[0.], 999999999999999.
"""


@pytest.fixture(scope='session')
def native_memory_limits_supported():
    probe = subprocess.run([sys.executable, '-c',
        'import resource; resource.setrlimit(resource.RLIMIT_AS,(4294967296,4294967296))'],
        capture_output=True)
    return probe.returncode == 0


@pytest.fixture
def native_limits():
    """Opt out of the explicitly mocked-memory transport tests below."""


@pytest.fixture(autouse=True)
def fixture_only_memory_shim(request, monkeypatch, tmp_path, native_memory_limits_supported):
    """On unsupported hosts test transport with mocked memory receipts only.

    Candidate fixtures remain real fresh processes and CPU/wall/group cleanup
    remain real. This does NOT test memory enforcement on Darwin; the separate
    native_limits test requires the unchanged production runner to fail closed.
    Linux/other hosts accepting RLIMIT_AS use the production runner unmodified.
    """
    if native_memory_limits_supported or 'native_limits' in request.fixturenames:
        return
    shim = tmp_path/'fixture_runner_mock_memory.py'
    shim.write_text('''import importlib.util, resource, sys
spec = importlib.util.spec_from_file_location('trusted_repeat_runner', ''' + repr(str(adapter._FILES['runner'])) + ''')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
def fixture_limits(cpu_seconds, mem_bytes):
    if mem_bytes < 1048576:
        raise ValueError('fixture unsupported memory cap')
    resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds,cpu_seconds))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1048576,1048576))
    return {'cpu_seconds':list(resource.getrlimit(resource.RLIMIT_CPU)),
            'mem_bytes':[mem_bytes,mem_bytes],
            'file_bytes':list(resource.getrlimit(resource.RLIMIT_FSIZE))}
runner._limits = fixture_limits
raise SystemExit(runner.main(sys.argv[1:]))
''')
    monkeypatch.setitem(adapter._FILES,'runner',shim)


def run(source=VALID, **kw):
    return adapter.draw(source, **(LIMITS | kw))


def test_identity_pins_sources_policy_and_actual_clean_runtime(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'test-credential-must-not-cross')
    value = adapter.adapter_identity()
    assert value['schema'] == 'sciel-repeat-adapter/1'
    for key, path in adapter._FILES.items():
        assert value['source_sha256'][key] == hashlib.sha256(path.read_bytes()).hexdigest()
    bare = {k:v for k,v in value.items() if k != 'identity_sha256'}
    assert value['identity_sha256'] == hashlib.sha256(adapter._encoded(bare)).hexdigest()
    assert value['runtime']['numpy'] == np.__version__
    assert value['runtime']['python_executable'] == sys.executable
    assert set(value['runtime']['thread_limits'].values()) == {'1'}
    assert value['certificate_policy']['geometry_sha256'] == value['source_sha256']['geometry']


def test_real_seed_before_import_hashseed_clean_env_and_limits(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'secret-fixture')
    monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'secret-fixture')
    monkeypatch.setenv('PYTHONPATH', '/not/a/search/path')
    source = '''
import os, random, sys, resource
import numpy as np
assert 'OPENROUTER_API_KEY' not in os.environ
assert 'AWS_SECRET_ACCESS_KEY' not in os.environ
assert 'PYTHONPATH' not in os.environ
assert '' not in sys.path and os.getcwd() not in sys.path
assert os.environ['OPENBLAS_NUM_THREADS'] == '1'
assert resource.getrlimit(resource.RLIMIT_CPU) == (3, 3)
if sys.platform != 'darwin':
    assert resource.getrlimit(resource.RLIMIT_AS) == (4294967296, 4294967296)
p = random.random()
n = np.random.random()
h = (hash('h15-seed-import-fixture') % 1000000) / 1000000

def run_packing():
    return [[p,n],[h,.5]]+[[.5,.5]]*24, [0.]*26, -1.
'''
    python_state, numpy_state = random.getstate(), np.random.get_state()
    first, same, other = run(source), run(source), run(source,seed=20)
    assert first['status'] == same['status'] == other['status'] == 'valid'
    assert first['capture'] == same['capture']
    assert first['capture']['centers'] != other['capture']['centers']
    assert first['capture']['centers'][0] == [random.Random(19).random(), np.random.RandomState(19).random_sample()]
    assert first['seed_applied'] == {'python_random':19,'numpy_legacy':19,'python_hash_seed':'19',
                                   'before_candidate_import':True,'determinism_proven':False}
    assert random.getstate() == python_state
    assert np.array_equal(np.random.get_state()[1],numpy_state[1])
    assert np.random.get_state()[2:] == numpy_state[2:]


def test_raw_sum_ignores_reported_sum_and_exact_certificate_is_separate():
    result = run()
    assert result['status'] == 'valid', result
    assert result['source_sha256'] == hashlib.sha256(VALID.encode()).hexdigest()
    assert result['raw_sum'] == .25
    exact = sum(map(Fraction.from_float,result['certificate']['radii']),Fraction())
    assert Fraction(**result['certified_sum']) == exact
    assert result['certificate']['centers'] == result['capture']['centers']
    assert result['certificate']['original_exact_feasible'] is True
    assert result['limits']['certification_cpu_seconds'] == 3
    assert result['limits']['certification_mem_bytes'] == MEMORY
    assert result['certification_process']['setup']['limits_applied']['cpu_seconds'] == [3,3]
    for key in ('candidate_elapsed_seconds','certification_elapsed_seconds','identity_elapsed_seconds'):
        assert 0 < result[key] <= result['elapsed_seconds']
    assert result['completed_at'] >= result['started_at']


def test_tolerance_geometry_has_explicit_shrink_and_distinct_metrics():
    source = '''
def run_packing():
    return [[.1,.5],[.3,.5]]+[[.9,.9]]*24, [.1000001,.1]+[0.]*24, 123.
'''
    result = run(source)
    assert result['status'] == 'valid',result
    assert result['capture']['valid'] and not result['capture']['strict_valid']
    assert not result['certificate']['original_exact_feasible']
    assert float(Fraction(**result['certified_sum'])) < result['raw_sum']
    assert result['capture']['radii'][:2] == [.1000001,.1]
    assert result['certificate']['centers'] == result['capture']['centers']


@pytest.mark.parametrize('source,reason',[
    ('raise RuntimeError("import fixture")','RuntimeError'),
    ('def run_packing():\n raise RuntimeError("call fixture")','RuntimeError'),
    ('def malformed syntax','SyntaxError'),
    ('def run_packing():\n return [], [], 0','expected centers shape'),
    ('def run_packing():\n return [[.5,.5]]*26, [.1]*26, 0','overlapping'),
    ('def run_packing():\n return [[.5,.5]]*26, [-.1]*26, 0','negative radius'),
    ('def run_packing():\n return [[.5,.5]]*26, [float("nan")]*26, 0','ValueError'),
    ('import os\nos._exit(70)','candidate exited 70'),
])
def test_candidate_failures_are_terminal_absent_quality(source,reason):
    result = run(source)
    assert result['status'] == 'candidate_failure',result
    assert reason in result['reason']
    assert result['raw_sum'] is result['certified_sum'] is None
    assert result['seed_applied']['before_candidate_import']
    assert result['certification_elapsed_seconds'] == 0


def test_certificate_rejection_is_failure_even_when_float_tolerance_valid():
    result = run('def run_packing():\n return [[-1e-7,.5]]+[[.5,.5]]*25, [0.]*26, 0')
    assert result['status'] == 'candidate_failure'
    assert result['capture']['valid']
    assert result['certificate']['status'] == 'rejected'
    assert result['raw_sum'] is result['certified_sum'] is None


def test_candidate_timeout_is_terminal_failure_after_seed_receipt():
    result = run('import time\ndef run_packing():\n time.sleep(30)',timeout=1.0)
    assert result['status'] == 'candidate_failure',result
    assert result['reason'] == 'candidate timeout'
    assert result['raw_sum'] is result['certified_sum'] is None
    assert result['candidate_process']['timed_out']


@pytest.mark.parametrize('sleep_candidate',[False,True])
def test_process_group_descendants_cleaned_on_return_and_timeout(tmp_path,sleep_candidate):
    heartbeat = tmp_path/'heartbeat'
    child = 'import time\nfrom pathlib import Path\np=Path('+repr(str(heartbeat))+')\nwhile True:\n p.write_text(str(time.monotonic()))\n time.sleep(.02)\n'
    source = 'import subprocess, sys, time\nfrom pathlib import Path\n'
    source += 'def run_packing():\n subprocess.Popen([sys.executable,"-c",'+repr(child)+'])\n'
    source += ' while not Path('+repr(str(heartbeat))+').exists(): time.sleep(.005)\n'
    if sleep_candidate:
        source += ' time.sleep(30)\n'
    source += ' return [[.5,.5]]*26,[0.]*26,0\n'
    result = run(source,timeout=1.0)
    assert result['status'] == ('candidate_failure' if sleep_candidate else 'valid'),result
    assert heartbeat.exists()
    last = heartbeat.read_bytes()
    time.sleep(.12)
    assert heartbeat.read_bytes() == last


def test_launch_and_parent_output_io_failures_are_infrastructure(monkeypatch):
    def fail(*args,**kwargs):
        raise OSError('fixture launch/output failure')
    monkeypatch.setattr(adapter.subprocess,'Popen',fail)
    result = run()
    assert result['status'] == 'infrastructure'
    assert result['raw_sum'] is result['certified_sum'] is None
    assert result['seed_applied'] is None


def test_candidate_output_io_failure_after_success_is_infrastructure(monkeypatch):
    original = adapter._read_json
    def fail(path):
        if Path(path).name == 'candidate_output.json':
            raise OSError('fixture output read failure')
        return original(path)
    monkeypatch.setattr(adapter,'_read_json',fail)
    result = run()
    assert result['status'] == 'infrastructure',result
    assert 'fixture output read failure' in result['reason']
    assert result['raw_sum'] is result['certified_sum'] is None
    assert result['candidate_elapsed_seconds'] > 0


def test_runtime_or_rlimit_setup_failure_is_infrastructure(native_limits):
    result = run(mem_bytes=1)
    assert result['status'] == 'infrastructure',result
    assert result['seed_applied'] is None
    assert result['raw_sum'] is result['certified_sum'] is None


def test_native_memory_support_is_never_silently_ignored(native_limits,native_memory_limits_supported):
    if native_memory_limits_supported:
        assert run()['status'] == 'valid'
    else:
        with pytest.raises(adapter.RepeatInfrastructureError,match='runtime probe failed'):
            adapter.adapter_identity()
        result = run()
        assert result['status'] == 'infrastructure'
        assert result['adapter_identity'] is result['seed_applied'] is None


def test_certificate_timeout_is_infrastructure_not_candidate_failure():
    result = run(certification_seconds=.000001)
    assert result['status'] == 'infrastructure',result
    assert result['capture']['valid']
    assert result['certificate'] is None
    assert result['certification_process']['timed_out']
    assert result['raw_sum'] is result['certified_sum'] is None


def test_source_drift_refuses_preflight_before_candidate(monkeypatch):
    monkeypatch.setitem(adapter._LOADED_SHA,'geometry','0'*64)
    with pytest.raises(adapter.RepeatInfrastructureError,match='differs'):
        adapter.adapter_identity()
    assert run()['status'] == 'infrastructure'


def test_expected_identity_has_no_extra_probe(monkeypatch):
    identity = adapter.adapter_identity()
    monkeypatch.setattr(adapter,'adapter_identity',lambda *_:pytest.fail('unexpected extra runtime probe'))
    result = run(expected_identity=identity)
    assert result['status'] == 'valid',result
    assert result['adapter_identity'] == identity
    assert result['identity_elapsed_seconds'] == 0


def test_runtime_drift_rejected_in_child_before_candidate_import(tmp_path):
    identity = adapter.adapter_identity()
    identity['runtime']['numpy'] = 'changed-fixture-version'
    identity['identity_sha256'] = hashlib.sha256(adapter._encoded(
        {k:v for k,v in identity.items() if k != 'identity_sha256'})).hexdigest()
    marker = tmp_path/'must-not-import'
    source = 'from pathlib import Path\nPath('+repr(str(marker))+').write_text("imported")\n'+VALID
    result = run(source,expected_identity=identity)
    assert result['status'] == 'infrastructure',result
    assert not marker.exists()
    assert result['seed_applied'] is None
    assert 'frozen adapter identity' in result['candidate_process']['stderr_tail']


def test_frozen_source_identity_drift_rejected_before_dispatch(monkeypatch):
    identity = adapter.adapter_identity()
    identity['source_sha256']['runner'] = '0'*64
    identity['identity_sha256'] = hashlib.sha256(adapter._encoded(
        {k:v for k,v in identity.items() if k != 'identity_sha256'})).hexdigest()
    monkeypatch.setattr(adapter,'_process',lambda *_args,**_kw:pytest.fail('unexpected dispatch'))
    result = run(expected_identity=identity)
    assert result['status'] == 'infrastructure'
    assert 'before dispatch' in result['reason']
    assert result['candidate_elapsed_seconds'] == 0


def test_certificate_helper_failure_cannot_be_scored(monkeypatch):
    original = adapter._process
    def broken(mode,*args,**kwargs):
        if mode == 'certificate':
            raise OSError('certificate output fixture')
        return original(mode,*args,**kwargs)
    monkeypatch.setattr(adapter,'_process',broken)
    result = run()
    assert result['status'] == 'infrastructure'
    assert result['capture']['valid']
    assert result['raw_sum'] is result['certified_sum'] is None
    assert result['certification_elapsed_seconds'] > 0


@pytest.mark.parametrize('payload',[
    {'centers':[[.5,.5]]*26,'radii':[True]*26},
    {'centers':[[.5,.5]]*26,'radii':[float('inf')]*26},
    {'centers':[[.5,.5]]*25,'radii':[0.]*26},
])
def test_geometry_strict_numeric_shape_policy(payload):
    assert not geometry.validate_packing(payload)['valid']
    assert geometry.certify_packing(payload)['status'] == 'rejected'


@pytest.mark.parametrize('kw',[{'seed':True},{'seed':-1},{'seed':2**32},{'timeout':0},
                              {'timeout':float('inf')},{'cpu_seconds':1.5},
                              {'mem_bytes':True},{'certification_seconds':0}])
def test_invalid_contract_rejected_before_runtime_probe(monkeypatch,kw):
    monkeypatch.setattr(adapter,'adapter_identity',lambda *_:pytest.fail('unexpected probe'))
    with pytest.raises(ValueError):
        run(**kw)
