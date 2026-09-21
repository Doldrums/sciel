"""Operator controls pause work without interrupting an in-flight model call."""
import json
import os
import signal
import socket
from types import SimpleNamespace

from sciel.cli import main as cli
from sciel.engine.runtime import process_identity


def test_signals_request_pause_and_restore_original_handlers():
    before = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    calls = []
    class Search:
        def request_stop(self):
            calls.append('pause')
        def run(self):
            signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
            assert calls == ['pause']
            calls.append('current iteration completed')
            return {'reason': 'paused'}
    assert cli._run_engine(Search()) == {'reason': 'paused'}
    assert calls[-1] == 'current iteration completed'
    assert all(signal.getsignal(sig) == old for sig, old in before.items())


def test_stop_command_requires_the_same_live_local_process(tmp_path):
    record = dict(status='running', hostname=socket.gethostname(), pid=os.getpid(),
                  process_identity=process_identity(os.getpid()))
    runtime = tmp_path / 'runtime.json'
    runtime.write_text(json.dumps(record))
    args = SimpleNamespace(run_dir=str(tmp_path))
    assert cli.cmd_stop(args) == 0
    marker = tmp_path / 'STOP'
    assert marker.exists()
    marker.unlink()
    record['status'] = 'completed'
    runtime.write_text(json.dumps(record))
    assert cli.cmd_stop(args) == 2
    assert not marker.exists()
    record.update(status='running', process_identity={'kind': 'wrong', 'value': 'reused pid'})
    runtime.write_text(json.dumps(record))
    assert cli.cmd_stop(args) == 2
    assert not marker.exists()
