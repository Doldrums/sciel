"""A cooldown publisher cannot overwrite a competing supervisor's telemetry."""
import importlib.util
import json
from pathlib import Path

import pytest

from sciel.engine.run_lock import RunDirectoryBusy, RunDirectoryLock


def supervisor():
    source = Path(__file__).resolve().parents[2] / 'scripts/continue_live_run.py'
    spec = importlib.util.spec_from_file_location('ownership_supervisor', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_competing_owner_keeps_its_runtime_unchanged(tmp_path):
    module = supervisor()
    runtime = tmp_path / 'runtime.json'
    runtime.write_text('{"status":"running","segment_id":"new-owner"}')
    original = runtime.read_bytes()
    lock = RunDirectoryLock(tmp_path)
    try:
        # No checkpoint is needed: a displaced publisher must not even read it.
        assert module.backoff(tmp_path, 0) is False
        assert runtime.read_bytes() == original
    finally:
        lock.close()


@pytest.mark.parametrize('stop', [False, True])
def test_cooldown_retains_ownership_through_terminal_publish(tmp_path, monkeypatch, stop):
    module = supervisor()
    (tmp_path / 'checkpoint.json').write_text(json.dumps({'provenance': {}}))
    if stop:
        (tmp_path / 'STOP').touch()
    original = module.RuntimeStatus.finish
    published = []

    def finish(runtime, status, **kwargs):
        with pytest.raises(RunDirectoryBusy):
            RunDirectoryLock(tmp_path)
        published.append(status)
        return original(runtime, status, **kwargs)

    monkeypatch.setattr(module.RuntimeStatus, 'finish', finish)
    assert module.backoff(tmp_path, 0) is (not stop)
    assert published == ['paused' if stop else 'running']
    RunDirectoryLock(tmp_path).close()


def test_bad_checkpoint_releases_cooldown_ownership(tmp_path):
    (tmp_path / 'checkpoint.json').write_text('invalid')
    with pytest.raises(ValueError):
        supervisor().backoff(tmp_path, 0)
    RunDirectoryLock(tmp_path).close()
