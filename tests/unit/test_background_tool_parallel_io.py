"""Detached-job control WAL and wave collection; no providers or containers."""
from copy import deepcopy
import json
from pathlib import Path
import sys

import pytest

pytest.importorskip('greenlet')

from sciel.campaign.budget import CampaignStopped, RequestNotDispatched
from sciel.engine import parallel_calls
from sciel.engine.loop import Engine
from sciel.engine.parallel_calls import (CallLedger, CooperativeIO, JSON_CONTROL_KINDS,
                                         UnknownParallelCall)

from test_research_tool_engine import FakeTools, setup_tools


class NoCapacity:
    def acquire(self, *args, **kwargs):
        raise AssertionError('Job metadata must not acquire provider/evaluator capacity')


def run_control(directory, kind, function, *, key='control', request=None):
    io = CooperativeIO(CallLedger(directory, NoCapacity(), 'fixture'), 1)
    try:
        return io.run([(key, lambda: io.call(kind, request or {'job_id': 'job-1'}, function))])
    finally:
        io.close()


@pytest.mark.parametrize('kind', sorted(JSON_CONTROL_KINDS))
def test_control_roundtrips_json_and_replays_without_capacity_or_side_effect(tmp_path, kind):
    calls = []
    expected = {'job_id': 'job-1', 'accepted': True, 'detail': {'unicode': 'π', 'unknown_cpu': None}}

    def leaf():
        calls.append(kind)
        return deepcopy(expected)

    result, errors = run_control(tmp_path, kind, leaf)
    assert result == {'control': expected} and not errors
    result['control']['detail']['unicode'] = 'changed in caller'
    assert run_control(tmp_path, kind, leaf) == ({'control': expected}, {})
    assert calls == [kind]


@pytest.mark.parametrize('kind', sorted(JSON_CONTROL_KINDS))
@pytest.mark.parametrize('value', [None, [], {'cpu_seconds': float('nan')}])
def test_invalid_control_return_is_unresolved_and_never_reexecuted(tmp_path, kind, value):
    calls = []

    def leaf():
        calls.append('executed')
        return value

    for _ in range(2):
        result, errors = run_control(tmp_path, kind, leaf)
        assert not result and isinstance(errors['control'], UnknownParallelCall)
    assert calls == ['executed']


@pytest.mark.parametrize('kind', sorted(JSON_CONTROL_KINDS))
def test_control_exception_is_not_an_llm_error(tmp_path, kind):
    def leaf():
        raise RuntimeError('metadata operation has an unresolved outcome')

    result, errors = run_control(tmp_path, kind, leaf)
    assert not result and isinstance(errors['control'], UnknownParallelCall)


def test_proven_not_dispatched_control_remains_distinct(tmp_path):
    def leaf():
        raise RequestNotDispatched('no job accepted')

    result, errors = run_control(tmp_path, 'tool_job_submit', leaf)
    assert not result and isinstance(errors['control'], RequestNotDispatched)


def test_accepted_submit_without_wal_receipt_never_repeats_on_resume(tmp_path, monkeypatch):
    accepted = []
    original = parallel_calls.durable_json

    def lose_ack(path, value):
        if Path(path).name == 'receipt.json':
            raise OSError('crash after accepted submit before WAL acknowledgement')
        return original(path, value)

    def submit():
        accepted.append('job-1')
        return {'job_id': 'job-1', 'accepted': True}

    monkeypatch.setattr(parallel_calls, 'durable_json', lose_ack)
    result, errors = run_control(tmp_path, 'tool_job_submit', submit)
    assert not result and isinstance(errors['control'], OSError)
    monkeypatch.setattr(parallel_calls, 'durable_json', original)
    result, errors = run_control(tmp_path, 'tool_job_submit', submit)
    assert not result and isinstance(errors['control'], UnknownParallelCall)
    assert accepted == ['job-1']


def test_poll_is_frozen_on_wave_replay_but_refreshed_next_wave(tmp_path):
    current = {'status': 'running', 'cpu_seconds': None}
    polls = []

    def poll():
        polls.append('polled')
        return deepcopy(current)

    assert run_control(tmp_path / 'wave-0', 'tool_job_poll', poll)[0]['control'] == current
    current.update(status='completed', cpu_seconds=2.5)
    assert run_control(tmp_path / 'wave-0', 'tool_job_poll', poll)[0]['control'] == {
        'status': 'running', 'cpu_seconds': None}
    assert run_control(tmp_path / 'wave-1', 'tool_job_poll', poll)[0]['control'] == current
    assert polls == ['polled', 'polled']


def setup_collector(tmp_path, monkeypatch, *, bootstrap=False):
    build, provider, evaluations, directory = setup_tools(tmp_path, monkeypatch,
        workers=2, max_iterations=2, portfolio_seed=bootstrap)
    observed = {'status': 'completed', 'job_id': 'job-1', 'cpu_seconds': 2.5}
    polls, prepared = [], []

    class Collector(FakeTools):
        background_enabled = True

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.background_collected = []
            self.background_cpu = 0.0

        def collect_background(self, io):
            if 'job-1' in self.background_collected:
                return

            def poll():
                polls.append('job-1')
                return deepcopy(observed)

            result = io.call('tool_job_poll', {'job_id': 'job-1'}, poll)
            if result['status'] == 'completed':
                self.background_collected.append(result['job_id'])
                self.background_cpu += result['cpu_seconds']
                self.emit(phase='background_collected', job_id=result['job_id'])

        def prepare(self, snapshot, *, signals):
            # Normal search has no tool-episode opportunity in this fixture.
            prepared.append(list(self.background_collected))
            return None

        def state(self):
            return {**super().state(), 'background_collected': list(self.background_collected),
                    'background_cpu': self.background_cpu}

        def load_state(self, value, *, expected_run_id=None):
            value = deepcopy(value)
            self.background_collected = value.pop('background_collected')
            self.background_cpu = value.pop('background_cpu')
            super().load_state(value, expected_run_id=expected_run_id)

    monkeypatch.setattr(sys.modules['sciel.research_tools.controller'], 'ResearchTools', Collector)
    return build, provider, evaluations, directory, observed, polls, prepared


@pytest.mark.parametrize('bootstrap', [False, True])
def test_collection_precedes_generation_and_bootstrap_without_tool_opportunity(tmp_path, monkeypatch, bootstrap):
    build, _, _, directory, _, polls, prepared = setup_collector(tmp_path, monkeypatch, bootstrap=bootstrap)
    engine = build()
    engine.run()
    assert polls == ['job-1'] and prepared == [['job-1'], ['job-1']]
    checkpoint = json.loads((directory / 'checkpoint.json').read_text())
    assert checkpoint['research_tools']['background_cpu'] == 2.5
    first = json.loads((directory / '.parallel/wave-0/transaction.json').read_text())
    assert first['phase'] == 'committed'
    # Initial canonical seed does not collect; the first non-seed wave does,
    # including portfolio bootstrap before ordinary generation starts.
    assert not list((directory / '.parallel/wave-0/calls').glob('*/intent.json')) or all(
        json.loads(path.read_text())['kind'] != 'tool_job_poll'
        for path in (directory / '.parallel/wave-0/calls').glob('*/intent.json'))
    collection_wave = directory / '.parallel/wave-1/calls'
    assert any(json.loads(path.read_text())['kind'] == 'tool_job_poll'
               for path in collection_wave.glob('*/intent.json'))


def test_collection_rollback_replays_same_observation_and_accounts_once(tmp_path, monkeypatch):
    build, provider, evaluations, directory, observed, polls, prepared = setup_collector(tmp_path, monkeypatch)
    original = Engine._prepare_step
    crashed = False

    def crash_after_collection(self, *args, **kwargs):
        nonlocal crashed
        if not crashed:
            crashed = True
            assert self.research_tools.background_collected == ['job-1']
            raise RuntimeError('fixture crash after collection before generation')
        return original(self, *args, **kwargs)

    monkeypatch.setattr(Engine, '_prepare_step', crash_after_collection)
    with pytest.raises(CampaignStopped, match='fixture crash after collection'):
        build().run()
    checkpoint = json.loads((directory / 'checkpoint.json').read_text())
    assert checkpoint['research_tools']['background_collected'] == []
    assert checkpoint['research_tools']['background_cpu'] == 0
    assert polls == ['job-1'] and not provider.calls and len(evaluations) == 1
    observed['cpu_seconds'] = 99  # A mutable service view cannot alter replay.
    engine = build(resume=True)
    engine.run()
    assert polls == ['job-1'] and engine.research_tools.background_cpu == 2.5
    assert prepared == [['job-1'], ['job-1']]
    checkpoint = json.loads((directory / 'checkpoint.json').read_text())
    assert checkpoint['research_tools']['background_collected'] == ['job-1']
    assert checkpoint['research_tools']['background_cpu'] == 2.5


def test_collection_failure_prevents_new_generation_dispatch(tmp_path, monkeypatch):
    build, provider, evaluations, directory, _, _, prepared = setup_collector(tmp_path, monkeypatch)
    collector = sys.modules['sciel.research_tools.controller'].ResearchTools

    def broken(self, io):
        def poll():
            raise RuntimeError('fixture unavailable job outcome')
        return io.call('tool_job_poll', {'job_id': 'job-1'}, poll)

    monkeypatch.setattr(collector, 'collect_background', broken)
    for resume in (False, True):
        with pytest.raises(CampaignStopped, match='no usable outcome'):
            build(resume=resume).run()
    assert not provider.calls and not prepared and len(evaluations) == 1
    checkpoint = json.loads((directory / 'checkpoint.json').read_text())
    assert checkpoint['research_tools']['background_collected'] == []


@pytest.fixture
def cell_lab(tmp_path):
    from types import SimpleNamespace
    from sciel.research_tools.config import validate_config, library_limits
    from sciel.research_tools.library import ToolLibrary
    from sciel.research_tools.lifecycle import CellLifecycle
    from test_research_cell_lifecycle import IMAGE

    cfg = validate_config({'enabled': True, 'image_id': IMAGE, 'interval': 1,
        'max_active_cells': 2, 'jobs': {'enabled': True, 'max_cpu_seconds': 100.,
                                      'max_wall_seconds': 200.}})
    events = []
    return SimpleNamespace(config=cfg, run=tmp_path, events=events,
        library=ToolLibrary(tmp_path, {'image_id': IMAGE}, library_limits(cfg), executor_factory=lambda _: None),
        lifecycle=CellLifecycle(tmp_path, cfg, lambda **event: events.append(event)))


def test_maintenance_reuses_question_without_stall_or_assigned_replay(cell_lab):
    from test_research_cell_lifecycle import snapshot, signals, prepare, episode, local_success

    life = cell_lab.lifecycle
    assert life.prepare(snapshot(1), signals(active=False), maintenance=True) is None
    assert not life.cells.all_cells()
    recipient = prepare(cell_lab, 1)
    life.commit(episode(cell_lab, recipient))
    local_success(cell_lab, author=0)  # Otherwise eligible for an assigned replay.
    for signal in (signals(active=False), signals(active=True, anchor=99)):
        plan = life.prepare(snapshot(1), signal, maintenance=True)
        assert plan['cell_id'] == recipient['cell_id']
        assert plan['question'] == recipient['question']
        assert plan['stage'] == 'research' and plan['assigned_tool_id'] is None
        assert len(life.cells.all_cells()) == 2
        life.cells.deactivate(plan['cell_id'])


def test_pending_revision_is_not_success_and_terminal_collection_can_resume(cell_lab):
    from sciel.methods.models import content_hash
    from sciel.research_tools.lifecycle import CellLifecycle
    from test_research_cell_lifecycle import package, prepare, episode

    life = cell_lab.lifecycle
    bundle = package(cell_lab)
    plan = prepare(cell_lab)
    job_id = content_hash({'request': 'saved background request'})
    pending = {'plan': plan, 'submitted_jobs': [{'job_id': job_id, 'tool_id': bundle['id']}]}
    assert life.pending(pending) == []
    cell = life.cells.get(plan['cell_id'])
    assert cell['phase'] == 'background_submitted' and not cell['active']
    assert cell['tool_ids'] == [] and life.state()['methods'] == life.state()['cards'] == {}
    assert cell['revision']['previous_revision'] == plan['cell_revision_id']
    assert cell['revision']['evidence'] == [{'kind': 'tool_job', 'id': job_id, 'tool_id': bundle['id']}]
    restored = CellLifecycle(cell_lab.run, cell_lab.config, lambda **_: None)
    restored.load_state(life.state())
    # A completed job binds its original submission plan, not the current head.
    terminal = episode(cell_lab, plan, bundle)
    assert restored.commit(terminal) == []
    assert restored.method(bundle['id'])['stage'] == 'locally_tested'
    resumed = CellLifecycle(cell_lab.run, cell_lab.config, lambda **_: None)
    resumed.load_state(restored.state())
    assert resumed.cells.get(plan['cell_id'])['tool_ids'] == [bundle['id']]


def test_known_deferred_submission_releases_cell_without_execution_evidence(cell_lab):
    from test_research_cell_lifecycle import prepare

    life = cell_lab.lifecycle
    plan = prepare(cell_lab)
    with pytest.raises(ValueError, match='no submitted jobs'):
        life.pending({'plan': plan, 'submitted_jobs': []})
    assert life.cells.get(plan['cell_id'])['active'] is True
    assert life.pending({'plan': plan, 'submitted_jobs': [], 'deferred': True}) == []
    cell = life.cells.get(plan['cell_id'])
    assert cell['phase'] == 'background_deferred' and cell['revision']['evidence'] == []
    assert cell['active'] is False and life.state()['methods'] == {}


@pytest.mark.parametrize('changed_limit,accepted', [
    ({'cpu_seconds': 45., 'wall_seconds': 90.}, True),
    ({'cpu_seconds': 101.}, False),
    ({'wall_seconds': 201.}, False),
    ({'memory_bytes': 1024 * 1024 * 1024}, False),
    ({'pids': 63}, False),
])
def test_resume_binds_job_duration_but_not_changed_sandbox_limits(cell_lab, changed_limit, accepted):
    from sciel.methods.models import canonical_bytes, content_hash
    from sciel.research_tools.lifecycle import CellLifecycle
    from test_research_cell_lifecycle import package, prepare, episode

    bundle = package(cell_lab)
    plan = prepare(cell_lab)
    terminal = episode(cell_lab, plan, bundle)
    row = terminal['invocations'][0]
    directory = cell_lab.library.path / 'artifacts'
    artifact = json.loads((directory / (row['receipt_id'] + '.json')).read_text())
    receipt = artifact['payload']['receipt']
    receipt['limits'].update(changed_limit)
    receipt['receipt_sha256'] = content_hash({key: value for key, value in receipt.items()
                                            if key != 'receipt_sha256'})
    row['receipt_id'] = content_hash(artifact)
    (directory / (row['receipt_id'] + '.json')).write_bytes(canonical_bytes(artifact))
    cell_lab.lifecycle.commit(terminal)
    restored = CellLifecycle(cell_lab.run, cell_lab.config, lambda **_: None)
    if accepted:
        restored.load_state(cell_lab.lifecycle.state())
        assert restored.method(bundle['id'])['local_receipt_id'] == row['receipt_id']
    else:
        with pytest.raises(ValueError):
            restored.load_state(cell_lab.lifecycle.state())
        assert restored.state()['methods'] == {}
