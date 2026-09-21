"""Healthy cells must share methods without ordinary recipient reselection.

The Engine, cell lifecycle, immutable artifacts and cooperative WAL are real.
Provider/evaluator/executor leaves are deterministic fakes; saved code is never
executed. Scheduling ordinary work away from the two cells is a fixture, not a
replacement for the research scheduler under test.
"""
from copy import deepcopy
import hashlib
import json

import pytest

from sciel.research_tools.jobs import BackgroundJobs
from sciel.research_tools.job_worker import run_job
from test_independent_research_engine import designed, setup
from test_parallel_engine import events
from test_research_tool_result_flow import result
from test_sequential_research_episode import decision, finish


OBSERVATIONS = {"measurement": -0.125, "valid": True,
                "marker": "REPRODUCIBLE_NEGATIVE_SHARING_FIXTURE"}


def _receipts(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (directory / '.parallel').rglob('receipt.json')}


def _sharing_run(tmp_path, monkeypatch, *, pause_after_replay=False,
                 replay_output=None, research_slots=1, background=False,
                 iterations=10):
    captured = {'planner_targets': [], 'first_created': False, 'leases': []}

    def scientist(prompt):
        header = json.loads(prompt.splitlines()[1])
        view = json.JSONDecoder().raw_decode(
            prompt.split('FROZEN_INPUT_VIEW_JSON:\n', 1)[1])[0]
        selected = next(row for row in view['references'] if row['role'] == 'selected')
        captured['planner_targets'].append((header['island_id'], selected['program_id']))
        if header['island_id'] == 0 and not captured['first_created']:
            captured['first_created'] = True
            return designed(decision(name='negative_diagnostic'),
                question_key='reproducible_negative_diagnostic')
        return designed(finish(conclusion='inconclusive',
            reason='The available check is auxiliary evidence; no task benefit is established.',
            next_decision='Close this bounded question and retain its exact observations.'),
            kind='closure', budget_action='none')

    build_original, provider, backend, evaluations, directory = setup(
        tmp_path, monkeypatch, [scientist] * 20,
        [result(OBSERVATIONS), replay_output or result(OBSERVATIONS)]
        + [result(OBSERVATIONS)] * 6,
        iterations=iterations, tools_overrides={
            'proactive_every_attempts': 1, 'share_between_islands': True,
            'max_active_cells': 2, 'research_jobs_per_wave': research_slots,
            'jobs': {'enabled': background}})

    def build(*, resume=False):
        engine = build_original(resume=resume)
        prepare = engine.research_tools.prepare

        def exactly_two_healthy_cells(snapshot, *, signals=None):
            if snapshot.island in (0, 1) and snapshot.iteration_no < 2:
                assert not signals['population_stall']['threshold_reached']
                return prepare(snapshot, signals=signals)
            return None

        engine.research_tools.prepare = exactly_two_healthy_cells
        pick = engine.schedule.pick_island

        def other_ordinary_islands(islands, in_flight):
            if engine.iteration_no >= 2:
                islands = [island for island in islands if island not in (0, 1)]
            return pick(islands, in_flight)

        engine.schedule.pick_island = other_ordinary_islands
        reserve = engine.research_tools.reserve_continuations

        def unique_cell_leases(*args, **kwargs):
            jobs = reserve(*args, **kwargs)
            plans = [snapshot.tool_episode_plan for _, snapshot in jobs]
            cells = [plan['cell_id'] for plan in plans]
            assert len(cells) == len(set(cells))  # Question and sharing cannot co-own a cell.
            assert len(cells) <= research_slots
            captured['leases'].append(deepcopy(plans))
            return jobs

        engine.research_tools.reserve_continuations = unique_cell_leases
        return engine

    execute = backend.execute

    def stop_after_paid_replay(*args, **kwargs):
        receipt = execute(*args, **kwargs)
        if pause_after_replay and len(backend.calls) == 2:
            (directory / 'STOP').touch()
        return receipt

    monkeypatch.setattr(backend, 'execute', stop_after_paid_replay)
    return build, provider, backend, evaluations, directory, captured


@pytest.mark.parametrize('pause_after_replay', [False, True])
@pytest.mark.parametrize('research_slots', [1, 2])
def test_healthy_cells_replay_adopt_and_use_without_ordinary_revisit_or_double_charge(
        tmp_path, monkeypatch, pause_after_replay, research_slots):
    build, provider, backend, evaluations, directory, captured = _sharing_run(
        tmp_path, monkeypatch, pause_after_replay=pause_after_replay,
        research_slots=research_slots)
    engine = build()
    completed = engine.run()
    preserved = {}
    if pause_after_replay:
        assert completed['reason'] == 'paused'
        assert len(backend.calls) == 2
        preserved = _receipts(directory)
        physical = (deepcopy(backend.receipts), deepcopy(provider.planner_prompts))
        engine = build(resume=True)
        assert (backend.receipts, provider.planner_prompts) == physical
        completed = engine.run()

    assert completed['iteration_no'] == completed['B_evals'] == 10
    state = engine.research_tools.state()
    assert len(state['lifecycle']['cells']['cells']) == 2
    assert len(state['registered']) == 1
    tool_id = state['registered'][0]
    method = state['lifecycle']['methods'][tool_id]
    assert method['stage'] == 'released'
    assert state['published'] == [tool_id]
    assert len(method['replay_attempts']) == len(method['adoptions']) == len(method['used_again']) == 1
    recipient = next(iter(method['adoptions']))
    assert recipient != method['author_cell_id']
    assert recipient in method['replay_attempts'] and recipient in method['used_again']
    assert method['release_record']['measurement_reproduced'] is True
    assert len(backend.calls) == 4  # Local, exact-input replay, recipient trial, later use.
    local, replay, adoption, later = backend.calls
    assert replay['inputs'] == local['inputs']
    for call in backend.calls:
        assert call['bundle']['id'] == tool_id
        assert call['bundle']['files'] == local['bundle']['files']
    assert adoption['inputs']['island_id'] == later['inputs']['island_id'] == 1
    assert adoption['inputs']['parent_program_id'] != local['inputs']['parent_program_id']
    assert later['inputs']['parent_program_id'] == adoption['inputs']['parent_program_id']
    assert later['inputs']['parent_source'] == adoption['inputs']['parent_source']
    assert len({receipt['invocation_id'] for receipt in backend.receipts}) == 4
    assert sum(receipt['resources']['cpu_seconds'] for receipt in backend.receipts) == pytest.approx(.04)

    ordinary = [row for row in events(directory) if row.get('event') == 'iteration']
    assert [row['island'] for row in ordinary[:2]] == [0, 1]
    assert all(row['island'] not in (0, 1) for row in ordinary[2:])
    assert {island for island, _ in captured['planner_targets']} == {0, 1}
    # Author create/review plus the recipient's own unrelated question. Sharing
    # is an execution lifecycle; it must not fabricate another scientific turn.
    assert len(provider.planner_prompts) == 3
    assert sorted(s['scientific_steps'] for s in state['lifecycle']['research_state'].values()) == [0, 1]
    assert all(s['status'] == 'closed' for s in state['lifecycle']['research_state'].values())
    assert len(evaluations) == len(set(evaluations)) == 11
    assert completed['cost_total'] == pytest.approx(.39)
    assert len(state['charges']) == len(provider.planner_prompts)
    assert all(_receipts(directory).get(path) == digest for path, digest in preserved.items())
    assert not any(cell['active'] for cell in state['lifecycle']['cells']['cells'].values())
    assert not any(program.origin == 'research_tool' for program in engine.db._all.values())
    use = next(row for row in events(directory) if row.get('phase') == 'post_adoption_use')
    assert use['adoption_revision_id'] == method['adoptions'][recipient]
    assert use['receipt_id'] == method['used_again'][recipient]

    physical = (deepcopy(backend.receipts), deepcopy(provider.planner_prompts), list(evaluations))
    restored = build(resume=True)
    assert restored.research_tools.state() == state
    assert (backend.receipts, provider.planner_prompts, evaluations) == physical


@pytest.mark.parametrize('replay_output', [
    result({'measurement': 999, 'valid': True, 'marker': 'DIFFERENT_OBSERVATIONS'}),
    result(status='tool_error', error='fixed fixture failure, not a negative measurement'),
], ids=['exact-observation-mismatch', 'tool-error'])
def test_failed_independent_replay_is_attempted_once_per_existing_recipient(
        tmp_path, monkeypatch, replay_output):
    build, provider, backend, evaluations, directory, captured = _sharing_run(
        tmp_path, monkeypatch, replay_output=replay_output, research_slots=2)
    engine = build()
    completed = engine.run()
    state = engine.research_tools.state()
    tool_id = state['registered'][0]
    method = state['lifecycle']['methods'][tool_id]
    assert len(backend.calls) == 2  # One local result and one actual failed replay.
    assert backend.calls[1]['inputs'] == backend.calls[0]['inputs']
    assert method['stage'] == 'locally_tested' and state['published'] == []
    assert len(method['replay_attempts']) == 1
    assert method['adoption_attempts'] == method['adoptions'] == method['used_again'] == {}
    assert len(provider.planner_prompts) == 3 and completed['cost_total'] == pytest.approx(.39)
    assert all(s['status'] == 'closed' for s in state['lifecycle']['research_state'].values())
    assert not any(row.get('phase') in {'adopted', 'post_adoption_use'} for row in events(directory))
    before = (deepcopy(backend.receipts), deepcopy(provider.planner_prompts), list(evaluations))
    restored = build(resume=True)
    assert restored.research_tools.state() == state
    assert (backend.receipts, provider.planner_prompts, evaluations) == before


def test_background_sharing_submit_stop_and_cold_collect_preserve_question_and_source(
        tmp_path, monkeypatch):
    monkeypatch.setattr(BackgroundJobs, '_spawn', lambda self: None)
    build, provider, backend, evaluations, directory, captured = _sharing_run(
        tmp_path, monkeypatch, background=True, iterations=20)
    submits = []
    original_submit = BackgroundJobs.submit

    def stop_after_first_submit(self, *args, **kwargs):
        acknowledgement = original_submit(self, *args, **kwargs)
        submits.append(deepcopy(acknowledgement))
        if len(submits) == 1:
            (directory / 'STOP').touch()
        return acknowledgement

    monkeypatch.setattr(BackgroundJobs, 'submit', stop_after_first_submit)
    engine = build()
    paused = engine.run()
    assert paused['reason'] == 'paused'
    assert len(submits) == len(backend.calls) == 1
    # The accepted side effect is durable before the containing wave commits.
    preserved = _receipts(directory)
    first_job = submits[0]['job_id']
    engine = build(resume=True)
    engine._stop_at = paused['B_evals'] + 4
    waiting = engine.run()
    assert waiting['reason'] == 'paused'
    assert waiting['iteration_no'] > paused['iteration_no']
    assert len(submits) == len(backend.calls) == 1
    assert len(provider.planner_prompts) == 2  # Recipient waits for its cell's job lease.
    original_questions = deepcopy(engine.research_tools.state()['lifecycle']['research_state'])
    original_cards = deepcopy(engine.research_tools.state()['lifecycle']['cards'])
    assert all(row['status'] == 'closed' for row in original_questions.values())
    completed_job_receipts = {}

    for index, stage in enumerate(('independent_replay', 'adopt_trial', 'use_again')):
        assert len(submits) == index + 1
        job_id = submits[index]['job_id']
        if index == 0:
            assert job_id == first_job
        stored = engine.research_tools.state()['jobs'][job_id]
        plan = stored['plan']
        assert plan['work_kind'] == 'sharing' and plan['stage'] == stage
        entry = engine.research_tools.state()['sharing']['items'][plan['sharing_work_id']]
        assert entry['status'] == 'waiting' and entry['job_ids'] == [job_id]
        with pytest.raises(FileNotFoundError):
            engine.research_tools.jobs.result(job_id)
        # Only the inert SavedExecutor runs; no worker process or saved source executes.
        run_job(directory, job_id, engine.research_tools.library)
        completed_job_receipts[job_id] = deepcopy(engine.research_tools.jobs.result(job_id))
        old_evals = waiting['B_evals']
        engine = build(resume=True)
        engine._stop_at = old_evals + 4
        waiting = engine.run()
        assert waiting['iteration_no'] > old_evals
        assert len(backend.calls) == index + 2
        # The fair queue may start recipient adoption before its own question;
        # the next release must let that due question run before later use.
        assert len(provider.planner_prompts) == (2 if index == 0 else 3)
        questions = engine.research_tools.state()['lifecycle']['research_state']
        assert all(questions[cell] == row for cell, row in original_questions.items())
        assert all(row['status'] == 'closed' for row in questions.values())
        original_questions = deepcopy(questions)
        cards = engine.research_tools.state()['lifecycle']['cards']
        assert all(cards[cell] == card for cell, card in original_cards.items())
        original_cards = deepcopy(cards)
        for completed_id, receipt in completed_job_receipts.items():
            assert engine.research_tools.jobs.result(completed_id) == receipt

    state = engine.research_tools.state()
    method = next(iter(state['lifecycle']['methods'].values()))
    assert method['stage'] == 'released'
    assert len(method['adoptions']) == len(method['used_again']) == 1
    assert len(submits) == 3 and len(backend.calls) == 4
    visible = [json.JSONDecoder().raw_decode(part)[0]
               for part in provider.planner_prompts[-1].split('VISIBLE TOOL:\n')[1:]]
    assert len(visible) == 1
    assert visible[0]['tool_id'] == state['registered'][0]
    assert visible[0]['code'] == backend.calls[0]['bundle']['files']['tool.py']
    local, replay, adoption, later = backend.calls
    assert replay['inputs'] == local['inputs']
    assert adoption['inputs']['island_id'] == later['inputs']['island_id'] == 1
    assert adoption['inputs']['parent_program_id'] == later['inputs']['parent_program_id']
    assert adoption['inputs']['parent_source'] == later['inputs']['parent_source']
    assert adoption['inputs']['parent_program_id'] != local['inputs']['parent_program_id']
    assert all(row['status'] == 'completed' for row in state['sharing']['items'].values())
    assert all(row['reviewed'] for row in state['jobs'].values())
    assert all(_receipts(directory).get(path) == digest for path, digest in preserved.items())
    assert waiting['cost_total'] == pytest.approx(.03 * (len(provider.calls) + 3))
    assert len(evaluations) == len(set(evaluations)) == waiting['B_evals'] + 1
    assert sum(row['resources']['cpu_seconds'] for row in backend.receipts) == pytest.approx(.04)
    ordinary = [row for row in events(directory) if row.get('event') == 'iteration']
    assert all(row['island'] not in (0, 1) for row in ordinary[2:])
    assert not any(cell['active'] for cell in state['lifecycle']['cells']['cells'].values())
    physical = (deepcopy(backend.receipts), deepcopy(provider.planner_prompts), list(evaluations))
    restored = build(resume=True)
    assert restored.research_tools.state() == state
    assert (backend.receipts, provider.planner_prompts, evaluations) == physical
