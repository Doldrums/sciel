"""Research scheduling reads lease counts without copying checkpoint history."""

import pytest

from test_research_cell_lifecycle import lab, prepare, snapshot, signals


def no_checkpoint_copy():
    raise AssertionError('scheduling must not copy the complete cell checkpoint')


@pytest.mark.parametrize('proactive', [False, True])
def test_cell_scheduling_preserves_creation_and_leases_without_full_copy(lab, monkeypatch, proactive):
    lifecycle = lab.lifecycle
    lifecycle.config['proactive_every_attempts'] = 8
    monkeypatch.setattr(lifecycle.cells, 'state', no_checkpoint_copy)
    active_signals = signals(active=not proactive)
    active_signals['committed_iteration'] = 0
    first = lifecycle.prepare(snapshot(), active_signals, proactive=proactive)
    assert first is not None
    assert lifecycle.cells.cell_count == 1
    assert lifecycle.cells.active_count == 1
    assert len([event for event in lab.events if event['phase'] == 'cell_created']) == 1

    # The same question reuses its cell after the wave releases the lease.
    lifecycle.cells.deactivate(first['cell_id'])
    assert lifecycle.cells.active_count == 0
    again = lifecycle.prepare(snapshot(), active_signals, proactive=proactive)
    assert again['cell_id'] == first['cell_id']
    assert lifecycle.cells.cell_count == 1
    assert lifecycle.cells.active_count == 1
    assert len([event for event in lab.events if event['phase'] == 'cell_created']) == 1


def test_proactive_capacity_refusal_does_not_create_a_dormant_cell(lab, monkeypatch):
    lifecycle = lab.lifecycle
    lifecycle.config['proactive_every_attempts'] = 8
    assert prepare(lab, island=0) is not None
    assert prepare(lab, island=1) is not None
    assert lifecycle.cells.active_count == lifecycle.cells.max_active
    monkeypatch.setattr(lifecycle.cells, 'state', no_checkpoint_copy)
    request = signals(active=False, committed_iteration=8)
    assert lifecycle.prepare(snapshot(island=2), request, proactive=True) is None
    assert lifecycle.cells.cell_count == 2
    assert lifecycle.cells.active_count == 2
    assert '2' not in lifecycle._state['affinity']
