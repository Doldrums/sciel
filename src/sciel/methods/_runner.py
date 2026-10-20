"""Trusted container-only watchdog and unprivileged JSON method runner.

Never invoke this module to execute generated code on the host. The supervisor
requires Linux PID 1, UID 0, and a readable cgroup-v2 CPU counter.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import resource
import selectors
import subprocess
import sys
import time
import traceback

PYTHON = '/usr/local/bin/python'
_REQUEST = None
_INVOKE = None


def _plain(value):
    """NumPy scalars and arrays a tool returns become their Python values; anything else stays unserializable."""
    if hasattr(value, 'tolist'):
        return value.tolist()
    raise TypeError(f'Object of type {type(value).__name__} is not JSON serializable')


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True,
                      allow_nan=False, default=_plain).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _invoke(method_id, value):
    return _INVOKE(method_id, value)


def invoke_dependency(method_id, value):
    if method_id not in _REQUEST['dependency_ids']:
        raise ValueError('method is outside the pinned dependency closure')
    return _invoke(method_id, value)


def _view(target, parent):
    saved = target.get('saved_evaluation') or {}
    result = (saved.get('record') or {}).get('result') or {} if saved.get('status') == 'available' else {}
    witness = (result.get('artifacts') or {}).get('full_output')
    return {'program_id': target.get('program_id'), 'score': target.get('selection_score'), 'parent': parent,
            'cases': result.get('cases'), 'output': None if witness is None else json.loads(witness).get('output')}


def _candidate_environment(payload):
    """What a candidate imports while it is evaluated: the task's assets through sciel.task, and through
    sciel.population the run's best and the selected program from the tool's frozen input, best first."""
    if not Path('/runtime/task/evaluator.py').is_file():
        return
    sys.path.insert(0, '/runtime/lib')
    os.environ['SCIEL_TASK_ROOT'] = '/runtime/task'
    targets = payload.get('research_targets') if isinstance(payload, dict) else None
    if not isinstance(targets, dict):
        return
    selected, best = targets.get('selected') or {}, targets.get('population_best')
    programs = [_view(selected, True)]
    if best is not None and best.get('program_id') != selected.get('program_id'):
        programs.insert(0, _view(best, False))
    path = '/tmp/sciel_population.json'
    Path(path).write_text(json.dumps({'schema': 'sciel-population-view/1', 'programs': programs}))
    os.environ['SCIEL_POPULATION'] = path


def _call(request_path):
    global _REQUEST, _INVOKE
    if os.geteuid() != 10001 or os.getpid() == 1:
        raise RuntimeError('method calls require the unprivileged container child')
    _REQUEST = json.loads(Path(request_path).read_text())
    # Before the file-size limit, which bounds the tool's output and not the view written for it.
    _candidate_environment(_REQUEST['input'])
    limits = _REQUEST['limits']
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
    cpu = max(1, math.ceil(limits['cpu_seconds']))
    resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    resource.setrlimit(resource.RLIMIT_FSIZE, (limits['max_output_bytes'], limits['max_output_bytes']))
    loader_spec = importlib.util.spec_from_file_location('_trusted_method_loader', '/runtime/_loader.py')
    loader = importlib.util.module_from_spec(loader_spec)
    loader_spec.loader.exec_module(loader)
    _INVOKE = loader.make_method_loader(_REQUEST['methods'])
    sys.modules['sciel_method_runtime'] = sys.modules[__name__]
    # Ordinary print diagnostics go to stderr; raw writes to stdout are still
    # treated as untrusted output and checked by the separate supervisor.
    saved_stdout = sys.stdout
    sys.stdout = sys.stderr
    try:
        result = _invoke(_REQUEST['method_id'], _REQUEST['input'])
        raw = canonical(result)
        if len(raw) > limits['max_output_bytes']:
            raise ValueError('method output exceeds max_output_bytes')
        sys.stdout = saved_stdout
        os.write(1, raw)
        return 0
    except BaseException:
        traceback.print_exc(limit=12, file=sys.stderr)
        return 71


def _cpu_seconds():
    values = dict(line.split() for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())
    return int(values['usage_usec']) / 1_000_000


def _memory_peak():
    try:
        return int(Path('/sys/fs/cgroup/memory.peak').read_text())
    except (OSError, ValueError):
        return None


def _supervise(request_path):
    if os.getpid() != 1 or os.geteuid() != 0 or sys.platform != 'linux':
        raise RuntimeError('method watchdog requires root Linux container PID1')
    request = json.loads(Path(request_path).read_text())
    expected = request['request_sha256']
    if digest({k: v for k, v in request.items() if k != 'request_sha256'}) != expected:
        raise ValueError('request digest mismatch')
    limits = request['limits']
    cpu_before = _cpu_seconds()  # Unsupported accounting fails before invocation.
    begun = time.monotonic()
    env = dict(PATH='/usr/local/bin:/usr/bin:/bin', LANG='C.UTF-8', TMPDIR='/tmp',
               PYTHONDONTWRITEBYTECODE='1', PYTHONHASHSEED='0')
    for key in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS',
                'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'BLIS_NUM_THREADS'):
        env[key] = '1'
    child = subprocess.Popen([PYTHON, '-s', '-P', '-B', '/runtime/_runner.py', '--call', request_path],
        stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd='/tmp', user=10001, group=10001, extra_groups=[], env=env, start_new_session=True)
    selector = selectors.DefaultSelector()
    for pipe, kind in ((child.stdout, 'output'), (child.stderr, 'stderr')):
        os.set_blocking(pipe.fileno(), False)
        selector.register(pipe, selectors.EVENT_READ, kind)
    output, stderr = bytearray(), bytearray()
    stderr_bytes = 0
    reason = None
    while True:
        elapsed = time.monotonic() - begun
        if elapsed >= limits['wall_seconds']:
            reason = 'wall_limit'
            break
        if _cpu_seconds() - cpu_before >= limits['cpu_seconds']:
            reason = 'cpu_limit'
            break
        if not selector.get_map() and child.poll() is not None:
            break
        delay = min(.025, max(.001, limits['wall_seconds'] - elapsed))
        if not selector.get_map():
            # EOF does not prove process exit. Keep enforcing the cgroup CPU
            # cap even after a child closes both inherited output descriptors.
            time.sleep(delay)
            continue
        for key, _ in selector.select(delay):
            chunk = os.read(key.fileobj.fileno(), 65536)
            if not chunk:
                selector.unregister(key.fileobj)
                continue
            if key.data == 'output':
                output.extend(chunk[:limits['max_output_bytes'] + 1 - len(output)])
                if len(output) > limits['max_output_bytes']:
                    reason = 'output_limit'
                    break
            else:
                stderr_bytes += len(chunk)
                stderr.extend(chunk)
                del stderr[:-limits['max_stderr_bytes']]
        if reason:
            break
    exit_code = child.poll()
    cpu_after = _cpu_seconds()
    if reason is None and cpu_after - cpu_before >= limits['cpu_seconds']:
        reason = 'cpu_limit'
    value, output_sha = None, None
    if reason is None and exit_code == 0:
        try:
            value = json.loads(output, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite JSON')))
            output_sha = digest(value)
        except (ValueError, UnicodeError, RecursionError):
            reason = 'invalid_json_output'
    elif reason is None:
        reason = 'method_failed'
    receipt = dict(schema='sciel-method-container-receipt/1',
        **{key: request[key] for key in ('method_id', 'invocation_id', 'plan_id', 'reservation_id',
            'source_sha256', 'runtime_sha256', 'input_sha256', 'request_sha256')},
        status='failed' if reason else 'ok', reason=reason, invoked=True, exit_code=exit_code,
        output=value, output_sha256=output_sha,
        resources=dict(cpu_seconds=cpu_after,
            wall_seconds=time.monotonic() - begun, memory_peak_bytes=_memory_peak(),
            output_bytes=len(output), stderr_bytes=stderr_bytes),
        stderr=stderr.decode('utf-8', 'replace'), stderr_truncated=stderr_bytes > len(stderr),
        limits=limits)
    # The root PID1, not the method, writes this envelope. Exiting kills every
    # process in the private PID namespace, including detached grandchildren.
    os.write(1, canonical(receipt))
    return 0


if __name__ == '__main__':
    try:
        if len(sys.argv) != 3 or sys.argv[1] not in ('--supervise', '--call'):
            raise RuntimeError('private container entry point only')
        code = _supervise(sys.argv[2]) if sys.argv[1] == '--supervise' else _call(sys.argv[2])
    except BaseException:
        traceback.print_exc(limit=12, file=sys.stderr)
        code = 70
    os._exit(code)
