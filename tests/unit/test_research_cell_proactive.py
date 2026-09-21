"""Reserved research beats sharing queues without inventing results or leases.

Receipts below are deterministic fixtures. No provider, generated tool or
candidate is executed; the global due cursor belongs to controller tests.
"""
from copy import deepcopy

import pytest

from sciel.research_tools.lifecycle import CellLifecycle
from test_research_cell_lifecycle import (
    lab, snapshot, signals, prepare, episode, package, local_success,
)


@pytest.fixture
def proactive_lab(lab):
    lab.config['proactive_every_attempts'] = 8
    lab.lifecycle.config['proactive_every_attempts'] = 8
    return lab


def reserved(lab, island=0, *, clock=0, anchor=0, active=False, maintenance=False, parent='p_original'):
    saved = signals(active=active, anchor=anchor)
    saved['committed_iteration'] = clock
    result = lab.lifecycle.prepare(snapshot(island, iteration=clock, parent=parent), saved,
                                    proactive=True, maintenance=maintenance)
    return dict(result, island_id=island, parent_program_id=parent) if result is not None else None


def files(lab):
    return sorted(str(path.relative_to(lab.run)) for path in lab.run.rglob('*.json'))


def test_default_healthy_behavior_is_unchanged(proactive_lab):
    lab = proactive_lab
    before, artifacts = lab.lifecycle.state(), files(lab)
    assert prepare(lab, active=False) is None
    assert lab.lifecycle.state() == before and files(lab) == artifacts


def test_research_starts_before_stagnation_and_continuous_records_do_not_fork_cell(proactive_lab):
    lab = proactive_lab
    first = None
    for clock in (0, 8, 16, 24):
        plan = reserved(lab, clock=clock, anchor=clock, parent=f'p_record_{clock}')
        assert plan['stage'] == 'research' and plan['proactive'] is True
        assert plan['trigger_sources'] == ['proactive']
        assert plan['assigned_tool_id'] is plan['reference_receipt_id'] is None
        assert plan['stagnation_signals']['population_stall']['threshold_reached'] is False
        assert plan['stagnation_signals']['committed_iteration'] == clock
        first = first or plan
        assert plan['cell_id'] == first['cell_id'] and plan['question'] == first['question']
        lab.lifecycle.commit(episode(lab, plan))  # explicit skipped proposal
    cell = lab.lifecycle.cells.get(first['cell_id'])
    assert cell['trigger']['kind'] == 'proactive'
    assert cell['trigger']['evidence']['committed_iteration'] == 0
    assert cell['trigger']['evidence']['every_attempts'] == 8
    assert 'no stagnation is asserted' in cell['trigger']['evidence']['scope']
    assert len(lab.lifecycle.state()['affinity']['0']) == 1
    assert len([e for e in lab.events if e['phase'] == 'cell_created']) == 1
    assert 'before a slowdown' in plan['question']
    assert len(plan['question']) < 600


def test_explicit_proactive_trigger_does_not_relabel_a_real_stall(proactive_lab):
    lab = proactive_lab
    plan = reserved(lab, clock=40, anchor=32, active=True)
    assert plan['stage'] == 'research'
    assert plan['trigger_sources'] == ['proactive', 'population_stall']
    assert plan['stagnation_signals']['population_stall']['threshold_reached'] is True
    assert lab.lifecycle.cells.get(plan['cell_id'])['trigger']['kind'] == 'proactive'


def test_reserved_research_is_not_replaced_by_foreign_replay_backlog(proactive_lab):
    lab = proactive_lab
    for index in range(4):
        bundle = package(lab, author=0, name=f'foreign_{index}')
        author = reserved(lab, clock=index * 8)
        lab.lifecycle.commit(episode(lab, author, bundle, ordinal=index + 1))
    before = deepcopy(lab.lifecycle.state()['methods'])
    receiver = reserved(lab, 1, clock=32)
    assert receiver['stage'] == 'research' and receiver['assigned_tool_id'] is None
    lab.lifecycle.commit(episode(lab, receiver))
    assert lab.lifecycle.state()['methods'] == before
    # Nonreserved work still performs the existing independent-replay gate.
    ordinary = prepare(lab, 1, active=False)
    assert ordinary['stage'] == 'independent_replay'
    assert ordinary['assigned_tool_id'] == bundle['id']


def test_reserved_research_precedes_recipient_trial_and_later_use_without_granting_them(proactive_lab):
    lab = proactive_lab
    bundle, _, _ = local_success(lab)
    replay = prepare(lab, 1)
    assert replay['stage'] == 'independent_replay'
    assert lab.lifecycle.commit(episode(lab, replay, bundle)) == [bundle['id']]
    receiver = reserved(lab, 2, clock=8)
    lab.lifecycle.commit(episode(lab, receiver))
    assert not lab.lifecycle.method(bundle['id'])['adoptions']
    trial = prepare(lab, 2, active=False)
    assert trial['stage'] == 'adopt_trial'
    lab.lifecycle.commit(episode(lab, trial, bundle))
    method = lab.lifecycle.method(bundle['id'])
    assert receiver['cell_id'] in method['adoptions'] and method['used_again'] == {}
    next_research = reserved(lab, 2, clock=16)
    assert next_research['stage'] == 'research' and next_research['assigned_tool_id'] is None
    lab.lifecycle.commit(episode(lab, next_research))
    assert lab.lifecycle.method(bundle['id'])['used_again'] == {}
    later = prepare(lab, 2, active=False)
    assert later['stage'] == 'use_again' and later['assigned_tool_id'] == bundle['id']


def test_cap_refusal_makes_no_new_cell_revision_affinity_or_artifact(proactive_lab):
    lab = proactive_lab
    lab.config['max_active_cells'] = 1
    lab.lifecycle = CellLifecycle(lab.run, lab.config, lambda **event: lab.events.append(event))
    first = reserved(lab)
    before, artifacts = lab.lifecycle.state(), files(lab)
    assert reserved(lab, 1, clock=8) is None
    assert lab.lifecycle.state() == before and files(lab) == artifacts
    assert '1' not in lab.lifecycle.state()['affinity']
    lab.lifecycle.commit(episode(lab, first))
    retry = reserved(lab, 1, clock=8)
    assert retry is not None and retry['stage'] == 'research'


def test_leased_existing_cell_refusal_has_no_mutation(proactive_lab):
    lab = proactive_lab
    first = reserved(lab)
    before, artifacts = lab.lifecycle.state(), files(lab)
    assert reserved(lab, clock=8, active=True, anchor=7) is None
    assert lab.lifecycle.state() == before and files(lab) == artifacts
    lab.lifecycle.commit(episode(lab, first))
    assert reserved(lab, clock=8)['cell_id'] == first['cell_id']


def test_private_unreleased_method_and_cell_lineage_survive_records_restore_and_reuse(proactive_lab):
    lab = proactive_lab
    own = package(lab, author=0, name='retained_alternative')
    first = reserved(lab)
    assert lab.lifecycle.commit(episode(lab, first, own)) == []
    prior = lab.lifecycle.cells.get(first['cell_id'])
    method = lab.lifecycle.method(own['id'])
    assert method['stage'] == 'locally_tested' and method['release_record'] is None
    assert own['id'] in prior['tool_ids']
    saved = lab.lifecycle.state()
    restored = CellLifecycle(lab.run, lab.config, lambda **_: None)
    restored.load_state(saved)
    lab.lifecycle = restored
    next_plan = reserved(lab, clock=48, anchor=47, parent='p_new_record')
    assert next_plan['cell_id'] == first['cell_id'] and next_plan['stage'] == 'research'
    assert lab.library.get(own['id'])['id'] == own['id']
    assert lab.lifecycle.commit(episode(lab, next_plan, own, ordinal=2)) == []
    after = lab.lifecycle.cells.get(first['cell_id'])
    assert after['revision']['previous_revision'] == prior['head_revision']
    assert own['id'] in after['tool_ids']
    assert lab.lifecycle.method(own['id'])['stage'] == 'locally_tested'
    assert not lab.lifecycle.method(own['id'])['adoptions']


def test_existing_stagnation_cell_is_reused_without_displacing_pending_lineage(proactive_lab):
    lab = proactive_lab
    initial = prepare(lab, anchor=3)
    lab.lifecycle.commit(episode(lab, initial))
    current = reserved(lab, clock=24, anchor=23, active=True, maintenance=True)
    assert current['cell_id'] == initial['cell_id']
    assert current['stage'] == 'research'
    assert lab.lifecycle.cells.get(current['cell_id'])['trigger']['kind'] == 'stagnation'
    assert current['trigger_sources'][0] == 'proactive'  # current reason, not changed genesis
    assert len(lab.lifecycle.state()['affinity']['0']) == 1


def test_plan_is_detached_and_lifecycle_restore_reproduces_same_reservation(proactive_lab):
    lab = proactive_lab
    first = reserved(lab)
    lab.lifecycle.commit(episode(lab, first))
    baseline = lab.lifecycle.state()
    raw = signals(active=False, anchor=8)
    raw['committed_iteration'] = 16
    current = snapshot(iteration=16)
    plan = lab.lifecycle.prepare(current, raw, proactive=True)
    original = deepcopy(plan)
    raw['committed_iteration'] = 999
    current.iteration_no = 999
    assert plan == original
    lab.lifecycle.load_state(baseline)
    assert reserved(lab, clock=16, anchor=8)['cell_id'] == original['cell_id']


@pytest.mark.parametrize('clock,every', [(None, 8), (True, 8), (-1, 8), (0, None), (0, True), (0, 0)])
def test_proactive_clock_and_enabled_cadence_are_required_without_state_changes(proactive_lab, clock, every):
    lab = proactive_lab
    lab.lifecycle.config['proactive_every_attempts'] = every
    before, artifacts = lab.lifecycle.state(), files(lab)
    invalid_signals = {**signals(active=False), 'committed_iteration': clock}
    with pytest.raises(ValueError, match='committed clock'):
        lab.lifecycle.prepare(snapshot(), invalid_signals, proactive=True)
    assert lab.lifecycle.state() == before and files(lab) == artifacts


def test_nonboolean_reservation_is_rejected(proactive_lab):
    with pytest.raises(ValueError, match='boolean'):
        proactive_lab.lifecycle.prepare(snapshot(), signals(active=False), proactive=1)
