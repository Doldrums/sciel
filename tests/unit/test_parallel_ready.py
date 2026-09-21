"""Readiness-driven coordinator safety and authenticated resume order; fake leaves only."""
from concurrent.futures import Future
from functools import partial
import json
import threading

import pytest

from sciel.engine.parallel_calls import CooperativeIO, CallLedger, ParallelPause, digest, durable_json
from sciel.engine.parallel_schedule import ResumeOrder
from test_parallel_calls import Capacity, reply
import test_parallel_calls as leaf_tests
import test_parallel_abort as abort_tests

IDENTITY = {'run_id': 'fixture-ready', 'wave': 0, 'engine_contract': 'fixture-contract'}


def make_io(path, **kwargs):
    return CooperativeIO(CallLedger(path, Capacity(), 'local'), 2,
                         schedule_identity=IDENTITY, **kwargs)


@pytest.mark.parametrize('case,kwargs', [
    ('test_known_receipts_replay_without_new_provider_calls_and_preserve_all_fields', {}),
    ('test_submitted_without_receipt_refuses_replay_and_never_invokes_provider', {}),
    ('test_capacity_failure_is_proven_not_dispatched', {'during_enter': False}),
    ('test_capacity_failure_is_proven_not_dispatched', {'during_enter': True}),
    ('test_submit_failure_cannot_start_leaf_even_if_executor_queued_it_first', {'enqueued_before_error': False}),
    ('test_submit_failure_cannot_start_leaf_even_if_executor_queued_it_first', {'enqueued_before_error': True}),
    ('test_unknown_provider_failure_is_sealed_without_replay_and_sibling_receipt_is_delivered', {}),
    ('test_evaluation_receipt_round_trips_exact_task_observations', {}),
    ('test_stop_drains_dispatched_calls_but_blocks_next_stage_and_resume_uses_known_receipts', {'next_phase': 'evaluation'}),
    ('test_stop_drains_dispatched_calls_but_blocks_next_stage_and_resume_uses_known_receipts', {'next_phase': 'retry'}),
    ('test_stop_drains_dispatched_calls_but_blocks_next_stage_and_resume_uses_known_receipts', {'next_phase': 'conclusion'}),
    ('test_stop_cancels_host_queued_evaluation_without_marker_and_keeps_paid_sibling', {}),
    ('test_stop_does_not_relabel_an_already_dispatched_unknown_failure', {}),
    ('test_cached_fatal_receipt_prevents_earlier_uncached_request_dispatch', {}),
    ('test_fatal_return_cancels_host_queued_sibling_without_dispatch', {}),
    ('test_preparation_error_blocks_all_uncached_calls', {}),
])
def test_existing_leaf_safety_on_scoped_path(tmp_path, monkeypatch, case, kwargs):
    module = leaf_tests if hasattr(leaf_tests, case) else abort_tests
    # A real scoped call declares its phase before physical dispatch. These
    # legacy fixtures intentionally construct cached/unknown receipts directly.
    # Give that fixture evidence the same explicit phase binding.
    preseeded = {
        'test_submitted_without_receipt_refuses_replay_and_never_invokes_provider': ['a'],
        'test_cached_fatal_receipt_prevents_earlier_uncached_request_dispatch': ['earlier', 'later'],
    }
    if case in preseeded:
        order = ResumeOrder(tmp_path, IDENTITY, 0, preseeded[case])
        original_prepare = CallLedger.prepare
        def bound_prepare(ledger, key, kind, request):
            value = original_prepare(ledger, key, kind, request)
            order.bind_call(value[0], key, value[1])
            return value
        monkeypatch.setattr(CallLedger, 'prepare', bound_prepare)
    monkeypatch.setattr(leaf_tests, 'CooperativeIO', partial(CooperativeIO, schedule_identity=IDENTITY))
    monkeypatch.setattr(abort_tests, 'CooperativeIO', partial(CooperativeIO, schedule_identity=IDENTITY))
    if 'submit_failure' in case:
        kwargs = {**kwargs, 'monkeypatch': monkeypatch}
    getattr(module, case)(tmp_path, **kwargs)
    assert list(tmp_path.rglob('header.json')), 'The new scoped scheduler must actually be exercised'


def test_ready_job_advances_to_evaluation_before_slow_sibling_response(tmp_path):
    ordinary_evaluated = threading.Event()
    coordinator = threading.get_ident()
    mutations = []
    io = make_io(tmp_path)
    def research_leaf():
        assert ordinary_evaluated.wait(3), 'An unrelated planner must not block the ordinary evaluator'
        return reply('research')
    def ordinary():
        io.call('llm', {'ordinary': 1}, lambda: reply('ordinary'))
        mutations.append(threading.get_ident())
        return io.call('tool_execution', {'evaluation': 1},
                       lambda: (ordinary_evaluated.set() or {'valid': True}))
    try:
        results, errors = io.run([
            ('research', lambda: io.call('llm', {'research': 1}, research_leaf)),
            ('ordinary', ordinary)])
    finally:
        io.close()
    assert not errors and results['ordinary'] == {'valid': True}
    assert mutations == [coordinator]
    turns = [json.loads(p.read_bytes())['payload'] for p in sorted(tmp_path.rglob('turn-*.json'))]
    assert turns[0]['call_key'] == 'ordinary:0'
    # The evaluator sets the event before its receipt is sealed. Research can
    # therefore return/consume next, but cannot prevent the evaluator dispatch.
    assert {row['call_key'] for row in turns} == {'ordinary:0', 'ordinary:1', 'research:0'}


def test_callback_failure_drains_dispatched_sibling_and_preserves_exact_cost(tmp_path):
    both = threading.Barrier(2)
    costs, callbacks = [], []
    def callback(operation, receipt):
        callbacks.append(operation['key'])
        raise OSError('fixture transcript failure')
    io = make_io(tmp_path, on_receipt=callback)
    def leaf(name):
        both.wait(3)
        return reply(name, .03)
    def job(name):
        value = io.call('llm', {'name': name}, lambda: leaf(name))
        costs.append(value.cost)
        return value.text
    try:
        results, errors = io.run([(name, lambda name=name: job(name)) for name in ('a', 'b')])
    finally:
        io.close()
    assert results == {'a': 'a', 'b': 'b'} and costs == [.03, .03]
    assert set(callbacks) == {'a:0', 'b:0'}
    assert set(errors) == {'coordinator:receipt:a:0', 'coordinator:receipt:b:0'}
    assert all(isinstance(error, OSError) for error in errors.values())


def test_completed_future_timeout_is_failure_not_wait_loop(tmp_path, monkeypatch):
    io = make_io(tmp_path)
    def failed_submit(function):
        future = Future()
        future.set_exception(TimeoutError('already completed transport future'))
        return future
    monkeypatch.setattr(io.executor, 'submit', failed_submit)
    try:
        results, errors = io.run([('a', lambda: io.call('llm', {}, lambda: pytest.fail('no call')))])
    finally:
        io.close()
    assert not results and isinstance(errors['a'], TimeoutError)
    assert io.aborted and not list(tmp_path.rglob('dispatched.json'))


def _run_two(path):
    io = make_io(path)
    try:
        return io.run([('a', lambda: io.call('llm', {'a': 1}, reply)),
                       ('b', lambda: io.call('llm', {'b': 1}, reply))])
    finally:
        io.close()


@pytest.mark.parametrize('damage', ['wrong_identity', 'missing_header', 'missing_head', 'truncate',
                                    'reordered', 'receipt', 'wrong_jobs', 'entire_phase', 'entire_schedule', 'phase_declaration', 'head_and_turns'])
def test_tampered_or_misattached_order_fails_before_new_dispatch(tmp_path, damage):
    assert not _run_two(tmp_path)[1]
    directory = next(tmp_path.rglob('header.json')).parent
    if damage == 'wrong_identity':
        identity = {**IDENTITY, 'run_id': 'other-run'}
    else:
        identity = IDENTITY
    if damage == 'missing_header':
        (directory / 'header.json').unlink()
    elif damage == 'missing_head':
        (directory / 'head.json').unlink()
    elif damage == 'truncate':
        (directory / 'turn-00000001.json').unlink()
    elif damage == 'reordered':
        a, b = [directory / f'turn-{i:08d}.json' for i in range(2)]
        av, bv = a.read_bytes(), b.read_bytes()
        a.write_bytes(bv); b.write_bytes(av)
    elif damage in ('entire_phase', 'entire_schedule'):
        import shutil
        shutil.rmtree(directory if damage == 'entire_phase' else directory.parent)
        if damage == 'entire_schedule':
            shutil.rmtree(tmp_path / 'resume-order-phases')
    elif damage == 'head_and_turns':
        (directory / 'head.json').unlink()
        for path in directory.glob('turn-*.json'):
            path.unlink()
    elif damage == 'phase_declaration':
        next((tmp_path / 'resume-order-phases').glob('*.json')).unlink()
    elif damage == 'receipt':
        path = next(tmp_path.rglob('receipt.json'))
        value = json.loads(path.read_bytes())
        value['payload']['value']['text'] = 'tampered but rehashed'
        value['sha256'] = digest(value['payload'])
        durable_json(path, value)
    io = CooperativeIO(CallLedger(tmp_path, Capacity(), 'local'), 2, schedule_identity=identity)
    before = len(list(tmp_path.rglob('dispatched.json')))
    jobs = [('a', lambda: io.call('llm', {'a': 1}, lambda: pytest.fail('repaid'))),
            ('b', lambda: io.call('llm', {'b': 1}, lambda: pytest.fail('repaid')))]
    if damage == 'wrong_jobs':
        jobs.reverse()
    try:
        with pytest.raises(ValueError, match='Resume-order'):
            io.run(jobs)
    finally:
        io.close()
    assert len(list(tmp_path.rglob('dispatched.json'))) == before


def test_empty_and_no_io_phases_preserve_run_index_and_call_ordinals(tmp_path):
    def one_run():
        io = make_io(tmp_path)
        try:
            assert io.run([]) == ({}, {})
            assert io.run([('preparation', lambda: 3)]) == ({'preparation': 3}, {})
            for i in range(2):
                assert not io.run([('same', lambda: io.call('llm', {'phase': i}, reply))])[1]
        finally:
            io.close()
    one_run(); one_run()
    assert len(list(tmp_path.rglob('header.json'))) == 4
    assert len(list(tmp_path.rglob('receipt.json'))) == 2


def test_schedule_write_failure_drains_paid_leaves_then_replays_once(tmp_path, monkeypatch):
    both = threading.Barrier(2)
    calls, costs = [], []
    original = ResumeOrder._save_head
    def failed_head(order):
        if order.turns:
            raise OSError('fixture durable head write failure')
        return original(order)
    monkeypatch.setattr(ResumeOrder, '_save_head', failed_head)
    io = make_io(tmp_path)
    def leaf(name):
        calls.append(name)
        both.wait(3)
        return reply(name, .04)
    def job(current, name):
        response = current.call('llm', {'name': name}, lambda: leaf(name))
        costs.append(response.cost)
        return response.text
    try:
        with pytest.raises(OSError, match='head write'):
            io.run([(name, lambda name=name: job(io, name)) for name in ('a', 'b')])
    finally:
        io.close()
    assert sorted(calls) == ['a', 'b'] and costs == []
    assert len(list(tmp_path.rglob('receipt.json'))) == 2
    monkeypatch.setattr(ResumeOrder, '_save_head', original)
    replay = make_io(tmp_path)
    try:
        results, errors = replay.run([(name, lambda name=name: job(replay, name)) for name in ('a', 'b')])
    finally:
        replay.close()
    assert not errors and results == {'a': 'a', 'b': 'b'}
    assert costs == [.04, .04] and sorted(calls) == ['a', 'b']


def test_known_fatal_after_historical_prefix_does_not_change_novelty_prompt(tmp_path):
    # One receipt was already consumed and influenced shared scientific state;
    # a later sibling physically fails before the coordinator restarts.
    ledger = CallLedger(tmp_path, Capacity(), 'local')
    order = ResumeOrder(tmp_path, IDENTITY, 0, ['winner', 'other'])
    d, h, _ = ledger.prepare('winner:0', 'llm', {'stage': 'first'})
    order.bind_call(d, 'winner:0', h)
    first = ledger.perform(d, h, 'llm', lambda: reply('candidate'))
    order.consume('winner', {'key': 'winner:0', 'kind': 'llm', 'request': {'stage': 'first'}}, first)
    d, h, _ = ledger.prepare('other:0', 'llm', {'stage': 'other'})
    order.bind_call(d, 'other:0', h)
    def failure():
        raise ConnectionError('later fatal')
    ledger.perform(d, h, 'llm', failure)
    state, requests = [], []
    io = make_io(tmp_path)
    def winner():
        io.call('llm', {'stage': 'first'}, lambda: pytest.fail('repaid'))
        state.append('candidate-reserved')
        requests.append({'sees': list(state)})
        return io.call('llm', requests[-1], lambda: pytest.fail('fresh paid work after fatal'))
    try:
        results, errors = io.run([('winner', winner),
            ('other', lambda: io.call('llm', {'stage': 'other'}, lambda: pytest.fail('repaid')))])
    finally:
        io.close()
    assert state == ['candidate-reserved'] and requests == [{'sees': ['candidate-reserved']}]
    assert not results and io.aborted
    assert len(list(tmp_path.rglob('dispatched.json'))) == 2


@pytest.mark.parametrize('damage', ['missing_intent', 'marker_request'])
def test_global_cached_receipt_preflight_refuses_orphan_or_wrong_dispatch(tmp_path, damage):
    ledger = CallLedger(tmp_path, Capacity(), 'local')
    order = ResumeOrder(tmp_path, IDENTITY, 0, ['new'])
    directory, request_hash, _ = ledger.prepare('old:0', 'llm', {})
    order.bind_call(directory, 'old:0', request_hash)
    ledger.perform(directory, request_hash, 'llm', reply)
    if damage == 'missing_intent':
        (directory / 'intent.json').unlink()
    else:
        durable_json(directory / 'dispatched.json', {'request_sha256': 'f' * 64})
    io = make_io(tmp_path)
    try:
        with pytest.raises(ValueError, match='history|marker'):
            io.run([('new', lambda: io.call('llm', {}, lambda: pytest.fail('new paid request')))])
    finally:
        io.close()
    assert len(list(tmp_path.rglob('dispatched.json'))) == 1


def test_successful_wave_cannot_skip_saved_later_phase(tmp_path):
    first = make_io(tmp_path)
    try:
        first.run([('a', lambda: first.call('llm', {'a': 1}, reply))])
        first.run([('b', lambda: first.call('llm', {'b': 1}, reply))])
        first.validate_resume_complete()
    finally:
        first.close()
    replay = make_io(tmp_path)
    try:
        replay.run([('a', lambda: replay.call('llm', {'a': 1}, lambda: pytest.fail('repaid')))])
        with pytest.raises(ValueError, match='unvisited'):
            replay.validate_resume_complete()
    finally:
        replay.close()
