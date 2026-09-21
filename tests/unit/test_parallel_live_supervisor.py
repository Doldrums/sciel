"""The live supervisor must select the parallel abandonment protocol."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sciel.campaign.budget import CampaignStopped


@pytest.mark.parametrize('continue_after_backoff', [False, True])
def test_parallel_stop_authorizes_once_then_restores_without_serial_recovery(
        tmp_path, monkeypatch, continue_after_backoff):
    source = Path(__file__).resolve().parents[2] / 'scripts/continue_live_run.py'
    spec = importlib.util.spec_from_file_location('parallel_supervisor_fixture', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    directory = tmp_path / 'run'
    configuration = tmp_path / 'configuration.json'
    cfg = {'parallel_search': {'enabled': True}, 'llm': {'mock': True},
           'budget': None, 'max_iterations': None, 'warm_start': None}
    configuration.write_text(json.dumps(cfg))
    monkeypatch.setattr(module, 'validate', lambda value: value)
    monkeypatch.setattr(module, 'load_config', lambda _: cfg)
    monkeypatch.setattr(module.sys, 'argv', ['continue_live_run.py', str(directory),
                                          '--fresh-config', str(configuration)])
    actions, engines = [], []
    authorization = {'auto_recoverable': True, 'recovery_id': 'fixture'}

    def build(*args, **kwargs):
        engine = SimpleNamespace(iteration_no=0, _stop_requested=False,
                                 research=SimpleNamespace(stopped_reason=None))
        engine.close = lambda: actions.append(('close', len(engines)))
        engines.append(engine)
        return engine

    def run(engine):
        actions.append(('run', len(engines)))
        if len(engines) == 1:
            raise CampaignStopped('parallel provider error')
        return {'iteration_no': 4}

    monkeypatch.setattr(module, 'build_engine', build)
    monkeypatch.setattr(module, '_run_engine', run)
    monkeypatch.setattr(module.checkpoint, 'load', lambda *args: actions.append(('restore', len(engines))))
    monkeypatch.setattr(module, 'inspect_parallel_stop',
                        lambda _: actions.append(('inspect', len(engines))) or authorization)
    monkeypatch.setattr(module, 'authorize_parallel_recovery',
                        lambda engine, path, receipt: actions.append(('authorize', len(engines))))
    monkeypatch.setattr(module, 'inspect_provider_stop', lambda *a, **k: pytest.fail('serial inspector used'))
    monkeypatch.setattr(module, 'prepare_provider_recovery', lambda *a, **k: pytest.fail('serial recovery used'))
    monkeypatch.setattr(module, 'backoff',
                        lambda _, delay: actions.append(('backoff', delay)) or continue_after_backoff)
    assert module.main() == 0
    expected = [('run', 1), ('inspect', 1), ('authorize', 1), ('close', 1), ('backoff', 30)]
    if continue_after_backoff:
        expected += [('restore', 2), ('run', 2), ('close', 2)]
    assert actions == expected


def test_existing_parallel_failure_requires_explicit_abandonment(tmp_path, monkeypatch):
    source = Path(__file__).resolve().parents[2] / 'scripts/continue_live_run.py'
    spec = importlib.util.spec_from_file_location('parallel_existing_stop_fixture', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cfg = {'parallel_search': {'enabled': True}, 'llm': {'mock': True},
           'budget': None, 'max_iterations': None}
    (tmp_path / 'checkpoint.json').write_text(json.dumps({'config': cfg}))
    closed = []
    engine = SimpleNamespace(iteration_no=0, research=SimpleNamespace(stopped_reason=None),
                             close=lambda: closed.append(True))
    monkeypatch.setattr(module, 'validate', lambda value: value)
    monkeypatch.setattr(module.sys, 'argv', ['continue_live_run.py', str(tmp_path)])
    monkeypatch.setattr(module, 'build_engine', lambda *a, **k: engine)
    monkeypatch.setattr(module.checkpoint, 'load', lambda *a: None)
    monkeypatch.setattr(module, 'has_unabandoned_provider_failure', lambda _: True)
    monkeypatch.setattr(module, '_run_engine', lambda *a: pytest.fail('old failure cannot authorize itself'))
    monkeypatch.setattr(module, 'authorize_parallel_recovery', lambda *a: pytest.fail('explicit flag required'))
    with pytest.raises(SystemExit) as stopped:
        module.main()
    assert stopped.value.code == 2 and closed == [True]
