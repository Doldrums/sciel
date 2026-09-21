"""Explicit opt-in real Docker fixtures; no optimizer, evaluator, model or API.

Set SCIEL_REPEAT_DOCKER_IMAGE to an immutable locally built image ID. Without it
the integration fixtures skip; pure configuration/dispatch guards still run.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from sciel.eval import repeat_draw as adapter
from sciel.eval import repeat_docker as docker

VALID = 'def run_packing():\n return [[.5,.5]]*26, [0.]*26, 999999\n'
MEMORY = 512 * 1024**2


@pytest.fixture(scope='module')
def runtime():
    image = os.environ.get('SCIEL_REPEAT_DOCKER_IMAGE')
    if not image:
        pytest.skip('explicit local Docker image ID required')
    return {'backend': 'docker', 'image': image}


@pytest.fixture(scope='module')
def identity(runtime):
    return adapter.adapter_identity(runtime=runtime)


def draw(identity, runtime, source=VALID, **kw):
    limits = dict(seed=19, timeout=5., cpu_seconds=4, mem_bytes=MEMORY,
                  certification_seconds=3., runtime=runtime, expected_identity=identity)
    return adapter.draw(source, **(limits | kw))


def absent(name):
    return subprocess.run(['docker', 'inspect', name], capture_output=True,
                          timeout=10).returncode != 0


@pytest.mark.parametrize('runtime', [{}, 'docker', {'backend': 'auto'},
    {'backend': 'native', 'image': 'unused'}, {'backend': 'docker', 'image': 'latest'},
    {'backend': 'docker', 'image': 'sha256:' + 'G'*64}])
def test_invalid_backend_never_dispatches(monkeypatch, runtime):
    monkeypatch.setattr(adapter, '_process', lambda *a, **k: pytest.fail('dispatch'))
    with pytest.raises(ValueError):
        adapter.draw(VALID, seed=0, timeout=1, cpu_seconds=1, mem_bytes=MEMORY,
                     certification_seconds=1, runtime=runtime)


def test_host_cli_environment_drops_provider_credentials(monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'never-forward')
    monkeypatch.setenv('AWS_SECRET_ACCESS_KEY', 'never-forward')
    assert not {'OPENROUTER_API_KEY', 'AWS_SECRET_ACCESS_KEY'} & docker._environment().keys()


def test_spawn_importable_candidate_and_readonly_loader(identity, runtime):
    """Real spawned workers import the pinned source under the Docker boundary."""
    source = '''
from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path
import sciel.candidate_modules as loader
assert Path(loader.__file__).resolve() == Path('/opt/sciel/sciel/candidate_modules/__init__.py')
try:
    Path(loader.__file__).write_text('changed')
except OSError:
    pass
else:
    raise AssertionError('trusted candidate loader is writable')
@dataclass
class Item:
    value: int
def worker(item):
    return item.value * item.value
def run_packing():
    with mp.get_context('spawn').Pool(2) as pool:
        assert pool.map(worker, [Item(2), Item(3)]) == [4, 9]
    return [[.5,.5]]*26, [0.]*26, 0.
'''
    assert identity['source_sha256']['candidate_loader']
    assert identity['source_sha256']['package_init']
    result = draw(identity, runtime, source, timeout=8., cpu_seconds=6)
    assert result['status'] == 'valid', result
    assert result['raw_sum'] == 0.


def test_identity_is_container_runtime_and_immutable_image(identity, runtime):
    assert identity['runtime_config'] == runtime
    assert identity['execution']['image'] == runtime['image']
    assert identity['runtime']['python_executable'] == docker.PYTHON
    assert identity['runtime']['platform'].startswith('Linux-')
    assert identity['runtime']['numpy'] == '2.2.6'
    assert identity['runtime']['scipy'] == '1.15.3'
    assert identity['source_sha256']['docker']
    assert identity['execution']['policy']['memory_scope'] == 'whole container; swap disabled'


def test_real_limits_clean_env_and_seed_before_import(identity, runtime, monkeypatch):
    monkeypatch.setenv('OPENROUTER_API_KEY', 'fixture-secret')
    source = '''
import os, random, resource
from pathlib import Path
import numpy as np
assert os.geteuid() == 10001
assert 'OPENROUTER_API_KEY' not in os.environ
assert 'PYTHONPATH' not in os.environ
assert resource.getrlimit(resource.RLIMIT_AS) == (536870912,536870912)
assert resource.getrlimit(resource.RLIMIT_CPU) == (4,4)
assert Path('/sys/fs/cgroup/memory.max').read_text().strip() == '536870912'
assert Path('/sys/fs/cgroup/memory.swap.max').read_text().strip() == '0'
assert Path('/sys/fs/cgroup/pids.max').read_text().strip() == '64'
assert Path('/sys/fs/cgroup/cpu.max').read_text().strip() == '100000 100000'
status=Path('/proc/self/status').read_text()
assert 'CapEff:\\t0000000000000000' in status
assert 'NoNewPrivs:\\t1' in status
try:
    os.kill(1,0)
except PermissionError:
    pass
else:
    raise AssertionError('candidate can signal trusted watchdog')
try:
    Path('/opt/sciel/_repeat_runner.py').write_text('changed')
except OSError:
    pass
else:
    raise AssertionError('helper writable')
assert not Path('/var/run/docker.sock').exists()
assert len(Path('/proc/net/route').read_text().splitlines()) == 1
p,n,h=random.random(),np.random.random(),(hash('repeat-docker')%1000000)/1000000
def run_packing():
    return [[p,n],[h,.5]]+[[.5,.5]]*24,[0.]*26,9999999
'''
    one, two, other = draw(identity, runtime, source), draw(identity, runtime, source), draw(identity, runtime, source, seed=20)
    assert one['status'] == two['status'] == other['status'] == 'valid', one
    assert one['capture'] == two['capture'] != other['capture']
    assert one['raw_sum'] == 0 and one['certified_sum'] == {'numerator': 0, 'denominator': 1}
    assert one['identity_elapsed_seconds'] == 0
    for phase in ('candidate_process', 'certification_process'):
        assert one[phase]['container']['removed']
        assert absent(one[phase]['container']['name'])


def test_memory_cap_rejects_oversized_allocation(identity, runtime):
    result = draw(identity, runtime, 'def run_packing():\n x=bytearray(1073741824)\n' + ' return [],[],0\n')
    assert result['status'] == 'candidate_failure', result
    assert 'MemoryError' in result['reason']
    assert result['raw_sum'] is result['certified_sum'] is None
    assert absent(result['candidate_process']['container']['name'])


@pytest.mark.parametrize('timeout', [False, True])
def test_namespace_cleans_detached_descendants_on_return_and_timeout(identity, runtime, timeout):
    source = 'import subprocess,sys,time\ndef run_packing():\n'
    source += " subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)'],start_new_session=True)\n"
    if timeout:
        source += ' time.sleep(60)\n'
    source += ' return [[.5,.5]]*26,[0.]*26,0\n'
    result = draw(identity, runtime, source, timeout=2.)
    assert result['status'] == ('candidate_failure' if timeout else 'valid'), result
    if timeout:
        assert result['candidate_process']['timed_out']
    assert absent(result['candidate_process']['container']['name'])


def test_watchdog_survives_host_kill_and_autoremoves_container(tmp_path, runtime):
    # A separate trusted host fixture dies after its candidate container starts.
    # No evaluator/optimizer is involved; PID1 must still end detached sleepers.
    existing = set(subprocess.check_output(['docker', 'ps', '-aq', '--filter', 'name=sciel-repeat-'], text=True).split())
    root = Path(__file__).resolve().parents[2]
    program = """import sys
sys.path.insert(0,sys.argv[1])
from sciel.eval.repeat_draw import adapter_identity,draw
runtime={'backend':'docker','image':sys.argv[2]}
identity=adapter_identity(runtime=runtime)
draw('import time\\ndef run_packing(): time.sleep(60)',seed=3,timeout=3,cpu_seconds=4,mem_bytes=536870912,certification_seconds=2,runtime=runtime,expected_identity=identity)
"""
    env = {k: v for k, v in os.environ.items() if k in ('PATH', 'HOME', 'DOCKER_HOST', 'DOCKER_CONTEXT', 'DOCKER_CONFIG')}
    env.update(TMPDIR=str(tmp_path), PYTHONDONTWRITEBYTECODE='1')
    proc = subprocess.Popen([sys.executable, '-I', '-B', '-c', program, str(root/'src'), runtime['image']],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    names = set()
    try:
        deadline = time.monotonic()+12
        while time.monotonic()<deadline:
            names = set(subprocess.check_output(['docker', 'ps', '-aq', '--filter', 'name=sciel-repeat-'], text=True).split())-existing
            # Identity probe comes first; wait until the candidate ready receipt.
            if names and list(tmp_path.rglob('candidate_setup.json')):
                break
            time.sleep(.05)
        assert names and list(tmp_path.rglob('candidate_setup.json'))
        proc.kill(); proc.wait(timeout=2)
        deadline=time.monotonic()+8
        while time.monotonic()<deadline and any(not absent(name) for name in names):
            time.sleep(.1)
        assert all(absent(name) for name in names)
    finally:
        if proc.poll() is None:
            proc.kill();proc.wait(timeout=2)
        proc.stderr.close()
        for name in names:
            if not absent(name):
                subprocess.run(['docker','rm','-f',name],capture_output=True,timeout=10)
