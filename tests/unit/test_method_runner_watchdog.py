"""Fake process/selector/clock: no child, Docker or submitted source is run."""
import json
from types import SimpleNamespace

import pytest

from sciel.methods import _runner as runner


def supervise(tmp_path, monkeypatch, *, cpu_after_eof=.05, exit_at=.10, wall=1,
              startup_cpu=3.0, final_cpu=None, open_pipes=False):
    state = {'time': 0., 'eof': False, 'exited': False, 'waits': [], 'writes': []}

    class Pipe:
        def __init__(self, fd): self.fd = fd
        def fileno(self): return self.fd

    class Child:
        stdout, stderr = Pipe(10), Pipe(11)
        def poll(self):
            state['exited'] = state['time'] >= exit_at
            return 0 if state['exited'] else None
        def wait(self, timeout):
            state['waits'].append(timeout)
            if state['time'] + timeout < exit_at:
                state['time'] += timeout
                raise runner.subprocess.TimeoutExpired('fake', timeout)
            state['time'] = exit_at
            return 0

    class Selector:
        def __init__(self): self.mapping = {}
        def register(self, pipe, events, kind):
            self.mapping[pipe.fileno()] = SimpleNamespace(fileobj=pipe, data=kind)
        def unregister(self, pipe): self.mapping.pop(pipe.fileno())
        def get_map(self): return self.mapping
        def select(self, timeout):
            state['time'] += timeout
            if open_pipes and state['time'] < exit_at:
                return []
            return [(key, 1) for key in list(self.mapping.values())]

    chunks = {10: [b'{"measured":1}', b''], 11: [b'']}
    def read(fd, size):
        value = chunks[fd].pop(0)
        if not chunks[10] and not chunks[11]: state['eof'] = True
        return value
    def cpu():
        elapsed_cpu = (final_cpu if final_cpu is not None and state['exited']
                       else cpu_after_eof if state['eof'] or open_pipes and state['time'] > 0 else 0)
        return startup_cpu + elapsed_cpu

    request = {key: 'a'*64 for key in ('method_id', 'invocation_id', 'plan_id', 'reservation_id',
        'source_sha256', 'runtime_sha256', 'input_sha256')}
    request['limits'] = dict(cpu_seconds=.1, wall_seconds=wall,
                            max_output_bytes=4096, max_stderr_bytes=4096)
    request['request_sha256'] = runner.digest(request)
    path = tmp_path/'request.json'; path.write_text(json.dumps(request))
    monkeypatch.setattr(runner.os, 'getpid', lambda: 1)
    monkeypatch.setattr(runner.os, 'geteuid', lambda: 0)
    monkeypatch.setattr(runner.sys, 'platform', 'linux')
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *a, **k: Child())
    monkeypatch.setattr(runner.selectors, 'DefaultSelector', Selector)
    monkeypatch.setattr(runner.os, 'set_blocking', lambda *a: None)
    monkeypatch.setattr(runner.os, 'read', read)
    monkeypatch.setattr(runner.os, 'write', lambda fd, raw: state['writes'].append(raw) or len(raw))
    monkeypatch.setattr(runner.time, 'monotonic', lambda: state['time'])
    monkeypatch.setattr(runner.time, 'sleep', lambda seconds: state.update(time=state['time']+seconds))
    monkeypatch.setattr(runner, '_cpu_seconds', cpu)
    monkeypatch.setattr(runner, '_memory_peak', lambda: 1024)
    runner._supervise(str(path))
    return json.loads(b''.join(state['writes'])), state


@pytest.mark.parametrize('open_pipes', [False, True])
def test_cpu_cap_remains_enforced_until_child_exit(tmp_path, monkeypatch, open_pipes):
    receipt, state = supervise(tmp_path, monkeypatch, cpu_after_eof=.25, open_pipes=open_pipes)
    assert receipt['status'] == 'failed' and receipt['reason'] == 'cpu_limit'
    assert receipt['output'] is None
    assert receipt['resources']['cpu_seconds'] == 3.25
    assert state['time'] < .10  # Never block until exit after losing the pipes.


def test_eof_under_cap_exits_successfully_and_keeps_total_cpu(tmp_path, monkeypatch):
    receipt, state = supervise(tmp_path, monkeypatch)
    assert receipt['status'] == 'ok' and receipt['output'] == {'measured': 1}
    assert receipt['resources']['cpu_seconds'] == 3.05
    assert state['time'] >= .10


def test_no_stream_wall_timeout_still_applies(tmp_path, monkeypatch):
    receipt, state = supervise(tmp_path, monkeypatch, wall=.12, exit_at=5)
    assert receipt['status'] == 'failed' and receipt['reason'] == 'wall_limit'
    assert state['time'] < .15


def test_final_cpu_crossing_cannot_be_accepted_after_exit(tmp_path, monkeypatch):
    receipt, _ = supervise(tmp_path, monkeypatch, final_cpu=.125)
    assert receipt['status'] == 'failed' and receipt['reason'] == 'cpu_limit'
    assert receipt['output'] is None
