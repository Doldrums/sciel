"""Continuous scheduling fixtures; no provider or candidate code is executed."""
from copy import deepcopy
import json

import pytest

from sciel.methods import __main__ as cli
from sciel.methods.budget import BudgetExceeded, BudgetUnknown
from sciel.methods.coordinator import TrialQuotaExceeded
from sciel.methods.technical import sort_bundle, experiment
from test_methods_cli import config, mocked_provider, IMAGE, RESOURCES


def unlimited_config():
    value = config()
    value.update(claim='continuous_run', max_steps=None, calendar_seconds=None,
                 storage_bytes=None, capacity_wait_seconds=None, checkpoint_interval=5,
                 upfront_costs=[])
    value['budget'] = {key: None for key in value['budget']}
    for lab in value['labs']:
        lab['limits'] = deepcopy(value['budget'])
    return value


def stops():
    return {name: [dict(kind='stop', reason=f'End local episode {i}') for i in range(30)]
            for name in ('author', 'reviewer', 'recipient')}


def test_null_caps_require_explicit_continuous_mode():
    value = unlimited_config()
    parsed = cli.validate_config(value)
    assert all(cap is None for cap in parsed['budget'].values())
    assert parsed['capacity_wait_seconds'] is None
    value['claim'] = 'local_episode'
    with pytest.raises(ValueError):
        cli.validate_config(value)


@pytest.mark.parametrize('change', [
    {'checkpoint_interval': 0}, {'checkpoint_interval': True}, {'max_cases': None},
    {'resources': {'wall_seconds': None}}, {'capacity_wait_seconds': -1},
])
def test_unlimited_does_not_remove_operation_contracts(change):
    value = unlimited_config(); value.update(change)
    with pytest.raises((ValueError, KeyError, TypeError)):
        cli.validate_config(value)


def test_continuous_exceeds_twelve_steps_checkpoints_stops_and_resumes_without_repeating(tmp_path, monkeypatch):
    value = unlimited_config()
    _, proposers = mocked_provider(monkeypatch, stops())
    real_step = cli.Coordinator.step
    target = [16]
    observed = []

    def step(self, lab):
        result = real_step(self, lab)
        observed.append((lab, result['index'], result['proposal_id']))
        if len(observed) >= target[0]:
            (tmp_path / 'stop.request').write_text('operator fixture')
        return result

    monkeypatch.setattr(cli.Coordinator, 'step', step)
    first = cli.run_config(value, tmp_path)
    assert first['status'] == 'stopped' and first['reason'] == 'operator_requested_stop'
    assert len(first['outcomes']) == 16 and first['steps_executed_this_invocation'] == 16
    assert (tmp_path / 'checkpoint.json').is_file()
    assert first['provider_cost_guard']['limit_usd'] is None
    assert first['resource_ledger']['spent']['steps'] == 16
    assert not first['resource_ledger']['unknown_reservations']
    assert not first['resource_ledger']['pending_reservations']
    prefix = list(observed)
    (tmp_path / 'stop.request').unlink()
    target[0] = 19
    second = cli.run_config(value, tmp_path)
    assert second['steps_executed_this_invocation'] == 3
    assert len(second['outcomes']) == 19
    assert observed[:16] == prefix and len(set(observed)) == 19
    assert sum(len(p.calls) for p in proposers) == 19
    assert second['resource_ledger']['spent']['steps'] == 19


def test_continuous_unknown_stops_instead_of_starting_another_lab(tmp_path, monkeypatch):
    value = unlimited_config()
    mocked_provider(monkeypatch, stops())
    calls = []
    def unknown(self, lab):
        calls.append(lab)
        raise BudgetUnknown('fixture unknown accounting')
    monkeypatch.setattr(cli.Coordinator, 'step', unknown)
    result = cli.run_config(value, tmp_path)
    assert result['status'] == 'accounting_stopped'
    assert calls == ['author'] and result['outcomes'] == []


@pytest.mark.parametrize('failure,expected_status,expected_steps', [
    (TrialQuotaExceeded, 'stopped', 3), (BudgetExceeded, 'budget_exhausted', 1),
])
def test_routing_local_quota_does_not_end_campaign_but_resource_overrun_does(
        tmp_path, monkeypatch, failure, expected_status, expected_steps):
    value = unlimited_config(); value['mode'] = 'C'
    common = sort_bundle(IMAGE, RESOURCES)
    new = sort_bundle(IMAGE, RESOURCES, author='author')
    value['routing'] = dict(author='author', reviewer='reviewer', recipient='recipient',
        control=[common['id']], transfer_inputs=[[3, 1, 2]], max_transfers=None)
    responses = stops()
    responses['author'][0] = dict(kind='method', files=new['files'], manifest=new['manifest'],
                                 experiment=experiment('$new', common['id'], [[2, 1, 2]]))
    mocked_provider(monkeypatch, responses)
    real_execute, real_step = cli.Coordinator.execute, cli.Coordinator.step
    count = []
    def execute(self, plan_id):
        if self.registry.get_artifact('experiment_plan', plan_id)['purpose'] == 'replay':
            raise failure('fixture: routing allocation exhausted')
        return real_execute(self, plan_id)
    def step(self, lab):
        result = real_step(self, lab); count.append(result)
        if len(count) == 3:
            (tmp_path / 'stop.request').write_text('operator fixture')
        return result
    monkeypatch.setattr(cli.Coordinator, 'execute', execute)
    monkeypatch.setattr(cli.Coordinator, 'step', step)
    result = cli.run_config(value, tmp_path)
    assert result['status'] == expected_status
    assert result['steps_executed_this_invocation'] == expected_steps
    if failure is TrialQuotaExceeded:
        assert result['routing'][0]['status'] == 'refused'


def test_stop_command_creates_only_explicit_stop_marker(tmp_path):
    (tmp_path / 'run-config.json').write_text('{}')
    assert cli.main(['stop', '--output', str(tmp_path)]) == 0
    assert json.loads((tmp_path / 'stop.request').read_text())['reason'] == 'operator_requested_stop'


def test_stop_scope_observes_marker_without_erasing_it(tmp_path):
    with cli._stop_scope(tmp_path) as stopping:
        assert not stopping()
        (tmp_path / 'stop.request').write_text('requested')
        assert stopping()
    assert (tmp_path / 'stop.request').exists()
