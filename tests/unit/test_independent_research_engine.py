"""Real Engine/WAL scheduling with fixed fake provider/evaluator/tool leaves.

No authored candidate or tool source is executed. The observations below are
fixtures, not evidence that a numerical method improved a task.
"""
from copy import deepcopy
import json
import sys
import threading

import pytest

from sciel.engine import checkpoint
from sciel.engine.factory import build_engine
from sciel.research_tools.config import validate_config
from test_parallel_engine import config, evaluator, events, source, task_fixture
from test_research_experimental_protocol import protocol
from test_research_tool_controller import IMAGE
from test_research_tool_result_flow import SavedExecutor, result
from test_research_yield_engine_integration import YieldProvider, generation_prompts, OBSERVATIONS, SAVED_STATE
from test_research_terminal_delivery_engine import delivered_cards
from test_sequential_research_episode import current_receipt, decision, finish


def designed(action, **changes):
    spec = protocol()
    spec.update(changes)
    return dict(action, experimental_protocol=spec)


def setup(tmp_path, monkeypatch, actions, outputs, *, iterations=2, workers=1,
          budget=None, barrier=False, tools_overrides=None, interleaved=False):
    tasks = tmp_path / 'tasks'
    task = task_fixture(tasks)
    evaluations, _ = evaluator(monkeypatch)
    backend = SavedExecutor(outputs)
    monkeypatch.setattr('sciel.research_tools.library.DockerExecutor', lambda image: backend)
    provider = YieldProvider(actions)
    if barrier:
        rendezvous = threading.Barrier(2)
        original = provider.generate
        count = 0
        lock = threading.Lock()
        def generate(*args, **kwargs):
            nonlocal count
            with lock:
                count += 1
                first_pair = count <= 2
            if first_pair:
                rendezvous.wait(timeout=3)
            return original(*args, **kwargs)
        provider.generate = generate
    cfg = config(task, tmp_path / 'capacity', workers=workers, num_islands=4,
        budget=budget, max_iterations=iterations, pop_cap=4,
        research={'enabled': False},
        research_tools={'enabled': True, 'image_id': IMAGE, 'interval': 1000,
            'proactive_every_attempts': 1000, 'max_scientific_steps': 3,
            'yield_after_success': True, 'independent_research': True,
            'research_jobs_per_wave': 1, 'shared_observations': True,
            'experimental_protocol': True, 'max_requests': 2,
            'share_between_islands': False, **(tools_overrides or {})})
    if interleaved:
        cfg.update(portfolio_seed=True, pop_cap=1)
        cfg['parallel_search']['portfolio_schedule'] = 'interleaved'
    directory = tmp_path / 'run'
    def build(resume=False):
        engine = build_engine(cfg, str(tasks), str(directory), python=sys.executable, llm=provider)
        if resume:
            checkpoint.load(engine, str(directory / 'checkpoint.json'))
        return engine
    return build, provider, backend, evaluations, directory


def queue(engine):
    return engine.research_tools.state()['continuations']['items']


def test_queue_continues_without_island_revisit_and_does_not_occupy_ordinary_worker(tmp_path, monkeypatch):
    def close(prompt):
        saved = json.loads((directory / 'checkpoint.json').read_text())
        assert saved['iteration_no'] == 1
        assert 'preserved research observation' in prompt
        assert current_receipt(prompt)
        return designed(finish(), kind='closure', budget_action='none')

    build, provider, backend, evaluations, directory = setup(tmp_path, monkeypatch,
        [designed(decision()), close], [result(OBSERVATIONS)], barrier=True)
    engine = build()
    completed = engine.run()
    assert completed['iteration_no'] == completed['B_evals'] == 2
    assert len(evaluations) == 3 and len(backend.calls) == 1
    assert len(provider.planner_prompts) == 2
    assert completed['cost_total'] == pytest.approx(.12)
    ordinary = [row for row in events(directory) if row.get('event') == 'iteration']
    assert [row['island'] for row in ordinary] == [0, 1]
    states = engine.research_tools.state()['lifecycle']['research_state']
    assert len(states) == len(queue(engine)) == 1
    cell_id, state = next(iter(states.items()))
    assert state['status'] == 'closed' and state['known_llm_cost'] == pytest.approx(.06)
    assert queue(engine)[cell_id]['status'] == 'closed'
    assert queue(engine)[cell_id]['dispatch_count'] == 2
    assert queue(engine)[cell_id]['last_completed_iteration'] == 2
    prompts = generation_prompts(provider)
    assert 'RESEARCH_CHECK_RESULTS_JSON:' not in prompts[0]
    delivered = delivered_cards(prompts[1], False)
    assert len(delivered) == 1 and delivered[0]['cell_id'] == cell_id
    assert 'preserved research observation' in delivered[0]['observations']['output_excerpt']
    resumed = build(resume=True)
    assert resumed.research_tools.state() == engine.research_tools.state()


def test_all_optins_continue_exact_state_then_canonical_candidate_without_ordinary_credit(tmp_path, monkeypatch):
    receipts = []
    def follow_up(prompt):
        receipt = current_receipt(prompt)
        receipts.append(receipt)
        return designed(decision(input_receipt_id=receipt, code=decision()['code'] + '# next stage\n'),
            kind='follow_up', prior_receipt_ids=[receipt], new_uncertainty='Does another starting state help?',
            budget_action='reallocate', budget_reason='Test another start after the fixed initial segment.')
    def submit(prompt):
        receipt = current_receipt(prompt)
        receipts.append(receipt)
        return designed(decision(action='submit_candidate', candidate_receipt_id=receipt),
            kind='closure', budget_action='none')
    build, provider, backend, evaluations, directory = setup(tmp_path, monkeypatch,
        [designed(decision()), follow_up, submit,
         designed(finish(conclusion='inconclusive'), kind='closure', budget_action='none')],
        [result(OBSERVATIONS), result({'candidate_source': source('run_search', 19), 'gain': 0})], iterations=3)
    engine = build()
    completed = engine.run()
    assert completed['iteration_no'] == 3 and completed['B_evals'] == 4
    assert len(evaluations) == 5 and len(backend.calls) == 2
    assert completed['cost_total'] == pytest.approx(.21)
    assert backend.calls[1]['inputs']['continuation_state'] == SAVED_STATE
    assert backend.calls[1]['inputs']['research_stage']['prior_receipt_id'] == receipts[0]
    assert backend.calls[1]['inputs']['research_targets']['selected']['program_id'] == 'p1'
    assert backend.calls[1]['inputs']['search_evidence']['committed_iteration'] == 1
    children = [p for p in engine.db._all.values() if p.origin == 'research_tool']
    assert len(children) == 1 and children[0].code == source('run_search', 19)
    assert children[0].parent_id == 'p1' and children[0].island == 0
    ordinary = [row for row in events(directory) if row.get('event') == 'iteration'
                and not row.get('meta', {}).get('research_candidate')]
    assert [row['island'] for row in ordinary] == [0, 1, 2]
    state = next(iter(engine.research_tools.state()['lifecycle']['research_state'].values()))
    assert state['status'] == 'closed'
    assert state['pending_method_trial']  # Canonical admission alone is not transfer evidence.
    assert state['resources']['cpu_seconds'] == .02
    assert state['known_llm_cost'] == pytest.approx(.12)
    assert len([row for row in events(directory) if row.get('action') == 'research_candidate']) == 1
    build(resume=True)  # Authenticates source, queue, cell, canonical receipt and full stage chain.


def test_finite_admission_budget_reserves_ordinary_attempt_before_research(tmp_path, monkeypatch):
    build, provider, backend, evaluations, directory = setup(tmp_path, monkeypatch,
        [], [], iterations=4, budget=1)
    engine = build()
    completed = engine.run()
    assert completed['B_evals'] == completed['iteration_no'] == 1
    assert len(evaluations) == 2 and not provider.planner_prompts and not backend.calls
    assert next(iter(queue(engine).values()))['status'] == 'ready'


def test_pause_at_admission_boundary_also_reserves_ordinary_attempt(tmp_path, monkeypatch):
    build, provider, backend, evaluations, _ = setup(tmp_path, monkeypatch, [], [], iterations=4)
    engine = build()
    engine._stop_at = 1
    completed = engine.run()
    assert completed['reason'] == 'paused'
    assert completed['B_evals'] == completed['iteration_no'] == 1
    assert len(evaluations) == 2 and not provider.planner_prompts and not backend.calls
    assert next(iter(queue(engine).values()))['status'] == 'ready'


def test_independent_method_trial_selects_another_branch_but_keeps_author_island(tmp_path, monkeypatch):
    def submit(prompt):
        return designed(decision(action='submit_candidate', candidate_receipt_id=current_receipt(prompt)),
                        kind='closure', budget_action='none')
    build, provider, backend, evaluations, directory = setup(tmp_path, monkeypatch,
        [designed(decision()), submit,
         designed(finish(conclusion='inconclusive'), kind='closure', budget_action='none'),
         designed(finish(conclusion='inconclusive'), kind='closure', budget_action='none')],
        [result({'candidate_source': source('run_search', 19)}),
         result({'candidate_source': source('run_search', 21)})],
        iterations=3, interleaved=True, tools_overrides={'yield_after_success': False})
    engine = build()
    completed = engine.run()
    assert completed['iteration_no'] == 3 and completed['B_evals'] == 5
    assert len(backend.calls) == 2
    first, second = [call['inputs'] for call in backend.calls]
    origin = first['research_targets']['selected']['program_id']
    other = second['research_targets']['selected']['program_id']
    assert origin != other
    assert engine.db._all[origin].island == 0 and engine.db._all[other].island == 1
    assert first['island_id'] == second['island_id'] == 0
    assert 'continuation_state' not in second and 'research_stage' not in second
    assert origin not in second['search_evidence']['selected_lineage']['program_ids']
    assert other not in first['search_evidence']['selected_lineage']['program_ids']
    children = [program for program in engine.db._all.values() if program.origin == 'research_tool']
    assert len(children) == 2 and [p.parent_id for p in children] == [origin, other]
    assert all(p.island == 0 for p in children)
    state = next(iter(engine.research_tools.state()['lifecycle']['research_state'].values()))
    assert state['pending_method_trial'] is None and state['status'] == 'closed'
    assert next(iter(queue(engine).values()))['status'] == 'closed'
    assert len([s for s in state['steps'] if s.get('independent_branch_trial')]) == 1
    assert len(provider.planner_prompts) == 4  # The transfer itself makes no planner request.
    assert completed['cost_total'] == pytest.approx(.30)
    assert len(evaluations) == 9  # Seed + three portfolio + three ordinary + two method candidates.
    build(resume=True)


@pytest.mark.parametrize('follow_up_after_block', [False, True])
def test_oversized_immutable_transfer_closes_once_then_waits_across_waves_and_resume(
        tmp_path, monkeypatch, follow_up_after_block):
    def stage(index):
        def action(prompt):
            receipt = current_receipt(prompt)
            return designed(decision(code=decision()['code'] + f'# stage {index}\n', input_receipt_id=receipt),
                kind='follow_up', prior_receipt_ids=[receipt],
                new_uncertainty=f'Does bounded stage {index} change the stored state?',
                budget_action='reallocate', budget_reason='Test the next bounded stage with exact saved input.')
        return action
    def submit(prompt):
        return designed(decision(action='submit_candidate', candidate_receipt_id=current_receipt(prompt)),
                        kind='closure', budget_action='none')
    def acknowledge_block(prompt):
        assert 'complete saved method needs 4 stages' in prompt
        if follow_up_after_block:
            receipt = current_receipt(prompt)
            return designed(decision(code=decision()['code'] + '# a new small check\n'),
                kind='follow_up', prior_receipt_ids=[receipt], new_uncertainty='Can a separate one-stage method check the other source?',
                budget_action='reallocate', budget_reason='The old four-stage transfer is infeasible; test a bounded new check.')
        return designed(finish(conclusion='inconclusive', reason='The full saved chain exceeds this opportunity cap.'),
                        kind='closure', budget_action='none')
    actions = [designed(decision()), stage(2), stage(3), stage(4), submit,
               designed(finish(conclusion='inconclusive'), kind='closure', budget_action='none'),
               acknowledge_block]
    outputs = [result({'continuation_state': {'stage': index}}) for index in (1, 2, 3)]
    outputs.append(result({'candidate_source': source('run_search', 19)}))
    if follow_up_after_block:
        outputs.append(result({'gain': 0}))
        actions.append(designed(finish(conclusion='inconclusive'), kind='closure', budget_action='none'))
    build, provider, backend, evaluations, directory = setup(tmp_path, monkeypatch, actions, outputs,
        iterations=7, interleaved=True)
    first = build()
    first._stop_at = 7
    stopped = first.run()
    assert stopped['reason'] == 'paused' and stopped['iteration_no'] == 6
    assert len(provider.planner_prompts) == 7 + follow_up_after_block
    assert len(backend.calls) == 4 + follow_up_after_block
    cell, entry = next(iter(queue(first).items()))
    state = first.research_tools.state()['lifecycle']['research_state'][cell]
    assert state['status'] == 'closed'
    assert entry['status'] == 'waiting' and entry['reason'] == 'pipeline_exceeds_scientific_limit'
    pending = state['pending_method_trial']
    assert pending['pipeline_stages'] == 4 and pending['scientific_step_limit'] == 3
    assert len([s for s in state['steps'] if s.get('action') == 'branch_trial']) == 1
    before = (deepcopy(provider.planner_prompts), deepcopy(backend.receipts))
    resumed = build(resume=True)
    completed = resumed.run()
    assert completed['iteration_no'] == 7 and completed['B_evals'] == 8
    assert completed['cost_total'] == pytest.approx(.54 + .03 * follow_up_after_block)
    assert (provider.planner_prompts, backend.receipts) == before
    assert queue(resumed)[cell]['dispatch_count'] == entry['dispatch_count']
    assert len(evaluations) == 13
    tampered = deepcopy(resumed.research_tools.state())
    tampered['lifecycle']['research_state'][cell]['pending_method_trial']['pipeline_stages'] = 5
    with pytest.raises(ValueError, match='saved pipeline'):
        resumed.research_tools.load_state(tampered)


@pytest.mark.parametrize('workers', [1, 2])
def test_pause_after_research_receipt_replays_queue_without_duplicate_calls_or_cost(tmp_path, monkeypatch, workers):
    build, provider, backend, evaluations, directory = setup(tmp_path, monkeypatch,
        [designed(decision())], [result(OBSERVATIONS)], iterations=workers, workers=workers)
    execute = backend.execute
    def stop_after_tool(*args, **kwargs):
        receipt = execute(*args, **kwargs)
        (directory / 'STOP').touch()
        return receipt
    monkeypatch.setattr(backend, 'execute', stop_after_tool)
    first = build()
    paused = first.run()
    assert paused['reason'] == 'paused' and paused['iteration_no'] == 0
    assert paused['cost_total'] == 0
    assert not queue(first)
    assert len(provider.planner_prompts) == len(backend.calls) == 1
    original_prompts = deepcopy(provider.planner_prompts)
    original_receipts = deepcopy(backend.receipts)
    resumed = build(resume=True)
    completed = resumed.run()
    assert completed['iteration_no'] == completed['B_evals'] == workers
    assert completed['cost_total'] == pytest.approx(.03 * (workers + 1))
    assert provider.planner_prompts == original_prompts and backend.receipts == original_receipts
    assert len(evaluations) == len(set(evaluations)) == workers + 1
    assert next(iter(queue(resumed).values()))['dispatch_count'] == 1
    assert next(iter(queue(resumed).values()))['status'] == 'ready'
    states = resumed.research_tools.state()['lifecycle']['research_state']
    assert next(iter(states.values()))['resources']['cpu_seconds'] == .01
    for marker in (directory / '.parallel').rglob('dispatched.json'):
        assert marker.with_name('receipt.json').is_file()


@pytest.mark.parametrize('field,value', [('independent_research', 1), ('shared_observations', 'true'),
    ('experimental_protocol', None), ('research_jobs_per_wave', True), ('research_jobs_per_wave', 0)])
def test_new_config_values_are_strict(field, value):
    with pytest.raises(ValueError):
        validate_config({field: value})


@pytest.mark.parametrize('changes', [dict(shared_observations=False), dict(max_scientific_steps=1)])
def test_protocol_requires_sequential_and_shared_evidence(changes):
    with pytest.raises(ValueError, match='experimental_protocol'):
        validate_config(dict(experimental_protocol=True, shared_observations=True, **changes)
                        if 'shared_observations' not in changes else dict(experimental_protocol=True, **changes))
