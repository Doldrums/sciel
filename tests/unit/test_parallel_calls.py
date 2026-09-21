"""Durable cooperative I/O with local functions; no API/task execution."""
from contextlib import contextmanager, nullcontext
import json
import threading

import pytest

pytest.importorskip('greenlet')

from sciel.campaign.budget import RequestNotDispatched
from sciel.engine.llm.client import LLMError, LLMResponse, llm_error_metadata
from sciel.engine.parallel_calls import (CallLedger, CooperativeIO, UnknownParallelCall,
                                         ParallelPause, digest, durable_json)
from sciel.engine.capacity import HostCapacity
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult


class Capacity:
    def acquire(self, *args, **kwargs):
        return nullcontext()


def reply(text='local', cost=.02):
    return LLMResponse(text, 'local-model', cost, 12, 8, 'stop', None)


def make_io(tmp_path, **kwargs):
    return CooperativeIO(CallLedger(tmp_path, Capacity(), 'local'), 2, **kwargs)


def test_concurrent_leaf_io_reverse_completion_and_single_writer_coordinator(tmp_path):
    io = make_io(tmp_path)
    coordinator = threading.get_ident()
    both_started = threading.Barrier(2)
    second_done = threading.Event()
    completed, mutations, threads = [], [], set()

    def leaf(name):
        threads.add(threading.get_ident())
        assert threading.get_ident() != coordinator
        both_started.wait(timeout=3)
        if name == 'first':
            assert second_done.wait(timeout=3)
        completed.append(name)
        if name == 'second':
            second_done.set()
        return reply(name)

    def job(name):
        mutations.append(('before', name, threading.get_ident()))
        result = io.call('llm', {'prompt': name}, lambda: leaf(name))
        mutations.append(('after', name, threading.get_ident()))
        return result.text

    try:
        results, errors = io.run([(name, lambda name=name: job(name)) for name in ('first', 'second')])
    finally:
        io.close()
    assert results == {'first': 'first', 'second': 'second'} and not errors
    assert completed == ['second', 'first'] and len(threads) == 2
    assert [(stage, name) for stage, name, _ in mutations] == [
        ('before', 'first'), ('before', 'second'), ('after', 'first'), ('after', 'second')]
    assert all(tid == coordinator for _, _, tid in mutations) and io.current is None
    assert len(list(tmp_path.rglob('receipt.json'))) == 2


def test_callback_failure_does_not_replace_paid_response_or_skip_sibling_accounting(tmp_path):
    callbacks, costs = [], []
    def failing_callback(operation, payload):
        callbacks.append(operation['key'])
        raise OSError('local transcript fixture')
    io = make_io(tmp_path, on_receipt=failing_callback)
    def job(name):
        result = io.call('llm', {'prompt': name}, lambda: reply(name, .03))
        costs.append(result.cost)
        return result.text
    try:
        results, errors = io.run([(name, lambda name=name: job(name)) for name in ('a', 'b')])
    finally:
        io.close()
    assert results == {'a': 'a', 'b': 'b'} and costs == [.03, .03]
    assert callbacks == ['a:0', 'b:0']
    assert set(errors) == {'coordinator:receipt:a:0', 'coordinator:receipt:b:0'}
    assert all(isinstance(e, OSError) for e in errors.values()) and io.current is None


def test_known_receipts_replay_without_new_provider_calls_and_preserve_all_fields(tmp_path):
    calls = []
    expected = reply('exact response\nπ', .004)
    def one_run():
        io = make_io(tmp_path)
        def leaf():
            calls.append('called')
            return expected
        try:
            return io.run([('stable-job', lambda: io.call('llm', {'prompt': 'same'}, leaf))])
        finally:
            io.close()
    first, first_errors = one_run()
    replay, replay_errors = one_run()
    assert not first_errors and not replay_errors and calls == ['called']
    assert first == replay == {'stable-job': expected}


def test_submitted_without_receipt_refuses_replay_and_never_invokes_provider(tmp_path):
    io = make_io(tmp_path)
    directory, request_hash, _ = io.ledger.prepare('a:0', 'llm', {'prompt': 'same'})
    durable_json(directory / 'dispatched.json', {'request_sha256': request_hash,
                                                'pid': 1, 'timestamp': '2026-09-21T00:00:00Z'})
    calls = []
    try:
        results, errors = io.run([('a', lambda: io.call('llm', {'prompt': 'same'}, lambda: calls.append('unsafe')))])
    finally:
        io.close()
    assert not results and not calls and isinstance(errors['a'], UnknownParallelCall)
    assert io.current is None and not (directory / 'receipt.json').exists()


@pytest.mark.parametrize('damage', ['receipt_digest', 'receipt_request', 'intent_body', 'new_request'])
def test_changed_intents_or_receipt_checksums_fail_closed_without_dispatch(tmp_path, damage):
    ledger = CallLedger(tmp_path, Capacity(), 'local')
    request = {'prompt': 'fixed'}
    directory, request_hash, _ = ledger.prepare('a', 'llm', request)
    ledger.perform(directory, request_hash, 'llm', reply)
    if damage == 'intent_body':
        path = directory / 'intent.json'
        value = json.loads(path.read_text())
        value['request']['prompt'] = 'tampered'
        durable_json(path, value)
    elif damage == 'new_request':
        request = {'prompt': 'changed'}
    else:
        path = directory / 'receipt.json'
        value = json.loads(path.read_text())
        if damage == 'receipt_digest':
            value['payload']['value']['cost'] = 999
        else:
            value['payload']['request_sha256'] = '0' * 64
            value['sha256'] = digest(value['payload'])
        durable_json(path, value)
    with pytest.raises(ValueError, match='differs|checksum'):
        ledger.prepare('a', 'llm', request)


@pytest.mark.parametrize('during_enter', [False, True])
def test_capacity_failure_is_proven_not_dispatched(tmp_path, during_enter):
    class BrokenCapacity:
        def acquire(self, *args, **kwargs):
            if not during_enter:
                raise OSError('local unavailable semaphore')
            @contextmanager
            def broken():
                raise OSError('local semaphore enter')
                yield
            return broken()
    io = CooperativeIO(CallLedger(tmp_path, BrokenCapacity(), 'local'), 1)
    calls = []
    try:
        results, errors = io.run([('a', lambda: io.call('llm', {}, lambda: calls.append('unsafe')))])
    finally:
        io.close()
    assert not results and isinstance(errors['a'], RequestNotDispatched) and not calls
    assert not list(tmp_path.rglob('dispatched.json')) and io.current is None


@pytest.mark.parametrize('enqueued_before_error', [False, True])
def test_submit_failure_cannot_start_leaf_even_if_executor_queued_it_first(tmp_path, monkeypatch, enqueued_before_error):
    io = make_io(tmp_path)
    original = io.executor.submit
    def broken_submit(function):
        if enqueued_before_error:
            original(function)
        raise RuntimeError('fixture thread startup failure')
    monkeypatch.setattr(io.executor, 'submit', broken_submit)
    calls = []
    try:
        results, errors = io.run([('a', lambda: io.call('llm', {}, lambda: calls.append('unsafe')))])
    finally:
        io.close()
    assert not results and isinstance(errors['a'], RequestNotDispatched) and not calls
    assert not list(tmp_path.rglob('dispatched.json')) and io.current is None


def test_unknown_provider_failure_is_sealed_without_replay_and_sibling_receipt_is_delivered(tmp_path):
    calls, accounted = [], []
    def one_run():
        io = make_io(tmp_path)
        both_started = threading.Barrier(2)
        def failed():
            calls.append('failed')
            both_started.wait(timeout=3)
            raise ConnectionError('fixture provider error')
        def successful():
            calls.append('ok')
            both_started.wait(timeout=3)
            return reply(cost=.04)
        def good_job():
            result = io.call('llm', {'prompt': 'good'}, successful)
            accounted.append(result.cost)
            return result
        try:
            return io.run([('bad', lambda: io.call('llm', {'prompt': 'bad'}, failed)), ('good', good_job)])
        finally:
            io.close()
    results, errors = one_run()
    assert set(results) == {'good'} and isinstance(errors['bad'], LLMError)
    assert llm_error_metadata(errors['bad'])['cause_type'] == 'ConnectionError'
    assert accounted == [.04] and sorted(calls) == ['failed', 'ok']
    one_run()
    assert sorted(calls) == ['failed', 'ok'] and accounted == [.04, .04]


def test_evaluation_receipt_round_trips_exact_task_observations(tmp_path):
    io = make_io(tmp_path)
    expected = CascadeOutcome(EvalResult(True, .75, {'valid': 1.}, stderr='local note',
        timing=.1, artifacts={'probe': 'source-bound'}, confirmed_sampled=True), .74, 'confirm', True)
    try:
        results, errors = io.run([('eval', lambda: io.call('evaluation', {'source_sha256': 'a' * 64}, lambda: expected))])
    finally:
        io.close()
    assert not errors and results['eval'] == expected


def test_fatal_coordinator_setup_error_also_resets_current(tmp_path):
    io = make_io(tmp_path)
    def jobs():
        io.current = 'partially prepared'
        raise ValueError('fixture setup failed')
        yield
    try:
        with pytest.raises(ValueError, match='setup failed'):
            io.run(jobs())
        assert io.current is None
    finally:
        io.close()


def test_pending_telemetry_is_optional_counts_submitted_work_and_cannot_break_accounting(tmp_path):
    states = []
    def broken_telemetry(state):
        states.append(state)
        raise RuntimeError('telemetry is not a scientific dependency')
    io = make_io(tmp_path, on_state=broken_telemetry)
    try:
        results, errors = io.run([('a', lambda: io.call('llm', {}, reply))])
    finally:
        io.close()
    assert not errors and results['a'].cost == .02
    assert states == [dict(pending_calls=1, pending_llm_calls=1, pending_evaluations=0),
                      dict(pending_calls=0, pending_llm_calls=0, pending_evaluations=0)]


@pytest.mark.parametrize('next_phase', ['retry', 'evaluation', 'conclusion'])
def test_stop_drains_dispatched_calls_but_blocks_next_stage_and_resume_uses_known_receipts(tmp_path, next_phase):
    requested = threading.Event()
    first_round = threading.Barrier(2)
    calls, consumed = [], []
    io = make_io(tmp_path, stop_requested=requested.is_set)

    def first_leaf(name):
        calls.append((name, 'generation'))
        first_round.wait(timeout=3)
        requested.set()
        # Both calls were already dispatched. Waiting here proves the
        # coordinator polls STOP while requests remain in flight.
        assert io.stop_event.wait(timeout=3)
        return reply(name, .03)

    def next_leaf(name):
        calls.append((name, next_phase))
        return (CascadeOutcome(EvalResult(True, .5), .5, 'full')
                if next_phase == 'evaluation' else reply(next_phase, .01))

    def job(current_io, name, first_function):
        first = current_io.call('llm', {'prompt': name}, lambda: first_function(name))
        consumed.append(first.cost)
        return current_io.call('evaluation' if next_phase == 'evaluation' else 'llm',
            {'name': name, 'stage': next_phase}, lambda: next_leaf(name))

    try:
        results, errors = io.run([(name, lambda name=name: job(io, name, first_leaf)) for name in ('a', 'b')])
        assert io.paused and io.stop_event.is_set()
        assert not results and set(errors) == {'a', 'b'}
        assert all(isinstance(error, ParallelPause) for error in errors.values())
        assert consumed == [.03, .03] and sorted(calls) == [('a', 'generation'), ('b', 'generation')]
        assert len(list(tmp_path.rglob('receipt.json'))) == 2
        assert len(list(tmp_path.rglob('dispatched.json'))) == 2
    finally:
        io.close()
    # Fresh coordinator reconstructs from baseline, consumes two known replies,
    # and dispatches only the previously paused operations.
    resumed = make_io(tmp_path)
    consumed.clear()
    def forbidden(_):
        pytest.fail('known paid request must not be dispatched again')
    try:
        results, errors = resumed.run([(name, lambda name=name: job(resumed, name, forbidden)) for name in ('a', 'b')])
    finally:
        resumed.close()
    assert set(results) == {'a', 'b'} and not errors and not resumed.paused
    assert consumed == [.03, .03]
    assert sorted(calls) == sorted([(name, phase) for name in ('a', 'b') for phase in ('generation', next_phase)])
    assert len(list(tmp_path.rglob('receipt.json'))) == 4


def test_stop_cancels_host_queued_evaluation_without_marker_and_keeps_paid_sibling(tmp_path):
    capacity = HostCapacity(tmp_path/'capacity', llm_slots=1, evaluator_slots=1,
                            provider_limits={'local': 1}, poll_interval=.01)
    occupied = capacity.acquire('evaluator')
    requested = threading.Event()
    io = CooperativeIO(CallLedger(tmp_path/'wave', capacity, 'local'), 2,
                       stop_requested=requested.is_set)
    evaluated = []
    def paid():
        requested.set()
        assert io.stop_event.wait(timeout=3)
        return reply(cost=.07)
    try:
        results, errors = io.run([
            ('evaluation', lambda: io.call('evaluation', {'source': 'fixed'}, lambda: evaluated.append('unsafe'))),
            ('generation', lambda: io.call('llm', {'prompt': 'fixed'}, paid))])
    finally:
        occupied.release()
        io.close()
    assert not evaluated and isinstance(errors['evaluation'], ParallelPause)
    assert results['generation'].cost == .07 and io.paused
    assert len(list((tmp_path/'wave').rglob('receipt.json'))) == 1
    assert len(list((tmp_path/'wave').rglob('dispatched.json'))) == 1


def test_stop_between_capacity_acquisition_and_dispatch_never_invokes_function(tmp_path):
    cancel = threading.Event()
    class StopsAtLease:
        def acquire(self, *args, **kwargs):
            assert kwargs['cancel'] is cancel
            cancel.set()
            return nullcontext()
    ledger = CallLedger(tmp_path, StopsAtLease(), 'local')
    directory, request_hash, _ = ledger.prepare('a', 'llm', {})
    called = []
    with pytest.raises(ParallelPause):
        ledger.perform(directory, request_hash, 'llm', lambda: called.append('unsafe'), cancel=cancel)
    assert not called and not (directory/'dispatched.json').exists()
    assert not (directory/'receipt.json').exists()


def test_stop_does_not_relabel_an_already_dispatched_unknown_failure(tmp_path):
    requested = threading.Event()
    io = make_io(tmp_path, stop_requested=requested.is_set)
    def failed():
        requested.set()
        assert io.stop_event.wait(timeout=3)
        raise ConnectionError('fixture unknown provider result')
    try:
        results, errors = io.run([('a', lambda: io.call('llm', {}, failed))])
    finally:
        io.close()
    assert io.paused and not results and isinstance(errors['a'], LLMError)
    assert not isinstance(errors['a'], ParallelPause)
    value = json.loads(next(tmp_path.rglob('receipt.json')).read_text())['payload']
    assert value['status'] == 'error' and value['not_dispatched'] is False


def test_stop_already_set_still_delivers_cached_receipt_but_blocks_uncached_calls(tmp_path):
    first = make_io(tmp_path)
    try:
        first.run([('cached', lambda: first.call('llm', {}, reply))])
    finally:
        first.close()
    paused = make_io(tmp_path, stop_requested=lambda: True)
    called = []
    try:
        results, errors = paused.run([(name, lambda name=name: paused.call('llm', {}, lambda: called.append(name)))
                                     for name in ('cached', 'new')])
    finally:
        paused.close()
    assert results['cached'].cost == .02 and isinstance(errors['new'], ParallelPause)
    assert not called and paused.paused
    assert len(list(tmp_path.rglob('receipt.json'))) == 1
