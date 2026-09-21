"""The cooldown heartbeat remains bound to the experiment identity."""
import importlib.util
import json
from pathlib import Path

from sciel.engine.runtime import process_matches


def test_provider_cooldown_heartbeat_matches_run_identity(tmp_path):
    path = Path(__file__).resolve().parents[2] / 'scripts/continue_live_run.py'
    spec = importlib.util.spec_from_file_location('live_supervisor_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    state = dict(provenance=dict(run_id='fixture-run', journal_anchor=dict(segment_id='fixture-segment')),
                 iteration_no=7, B_evals=5)
    (tmp_path / 'checkpoint.json').write_text(json.dumps(state))
    assert module.backoff(tmp_path, 0) is True
    runtime = json.loads((tmp_path / 'runtime.json').read_text())
    assert runtime['run_id'] == 'fixture-run'
    assert runtime['segment_id'] == 'fixture-segment'
    assert runtime['status'] == 'running' and process_matches(runtime) is True
    (tmp_path / 'STOP').write_text('pause')
    assert module.backoff(tmp_path, 0) is False
    runtime = json.loads((tmp_path / 'runtime.json').read_text())
    assert runtime['status'] == 'paused'
