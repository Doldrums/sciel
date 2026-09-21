"""Fatal wave errors block new work while preserving dispatched receipts."""
import threading

import pytest

from sciel.engine.capacity import HostCapacity
from sciel.engine.llm.client import LLMError
from sciel.engine.parallel_calls import CallLedger, CooperativeIO, ParallelAborted
from test_parallel_calls import Capacity, make_io, reply


def test_cached_fatal_receipt_prevents_earlier_uncached_request_dispatch(tmp_path):
    ledger = CallLedger(tmp_path, Capacity(), 'local')
    directory, request_hash, _ = ledger.prepare('later:0', 'llm', {'prompt': 'old'})
    def failed():
        raise ConnectionError('fixture unknown outcome')
    ledger.perform(directory, request_hash, 'llm', failed)
    io = make_io(tmp_path)
    called = []
    try:
        results, errors = io.run([
            ('earlier', lambda: io.call('llm', {'prompt': 'new'}, lambda: called.append('unsafe'))),
            ('later', lambda: io.call('llm', {'prompt': 'old'}, lambda: called.append('repaid'))),
        ])
    finally:
        io.close()
    assert not results and not called
    assert isinstance(errors['earlier'], ParallelAborted)
    assert isinstance(errors['later'], LLMError)
    assert io.aborted and not io.paused
    assert len(list(tmp_path.rglob('dispatched.json'))) == 1


def test_fatal_return_cancels_host_queued_sibling_without_dispatch(tmp_path):
    capacity = HostCapacity(tmp_path / 'capacity', llm_slots=1, evaluator_slots=1,
                            provider_limits={'local': 1}, poll_interval=.01)
    occupied = capacity.acquire('evaluator')
    safety_release = threading.Timer(3, occupied.release)
    safety_release.start()
    io = CooperativeIO(CallLedger(tmp_path / 'wave', capacity, 'local'), 2)
    called = []
    def failure():
        raise ConnectionError('fixture dispatched failure')
    try:
        results, errors = io.run([
            ('queued', lambda: io.call('evaluation', {'source': 'fixed'}, lambda: called.append('unsafe'))),
            ('failed', lambda: io.call('llm', {'prompt': 'fixed'}, failure)),
        ])
    finally:
        safety_release.cancel()
        occupied.release()
        io.close()
    assert not results and not called
    assert isinstance(errors['queued'], ParallelAborted)
    assert isinstance(errors['failed'], LLMError)
    assert io.aborted and not io.paused
    assert len(list((tmp_path / 'wave').rglob('dispatched.json'))) == 1
    assert len(list((tmp_path / 'wave').rglob('receipt.json'))) == 1


def test_processing_error_blocks_next_round_but_accounts_dispatched_sibling(tmp_path):
    io = make_io(tmp_path)
    started = threading.Barrier(2)
    calls, booked = [], []
    def leaf(name):
        calls.append(name)
        started.wait(timeout=3)
        return reply(name, .03)
    def job(name):
        response = io.call('llm', {'prompt': name}, lambda: leaf(name))
        booked.append(response.cost)
        if name == 'bad':
            raise ValueError('fixture post-response processing failure')
        return io.call('llm', {'prompt': 'retry'}, lambda: calls.append('unsafe retry'))
    try:
        results, errors = io.run([(name, lambda name=name: job(name)) for name in ('bad', 'good')])
    finally:
        io.close()
    assert not results and sorted(calls) == ['bad', 'good']
    assert booked == [.03, .03]
    assert isinstance(errors['bad'], ValueError)
    assert isinstance(errors['good'], ParallelAborted)
    assert io.aborted and not io.paused
    assert len(list(tmp_path.rglob('receipt.json'))) == 2


def test_preparation_error_blocks_all_uncached_calls(tmp_path):
    io = make_io(tmp_path)
    called = []
    def failure():
        raise ValueError('fixture preparation failed')
    try:
        results, errors = io.run([
            ('good', lambda: io.call('llm', {}, lambda: called.append('unsafe'))),
            ('bad', failure),
        ])
    finally:
        io.close()
    assert not results and not called
    assert isinstance(errors['good'], ParallelAborted)
    assert isinstance(errors['bad'], ValueError)
    assert io.aborted and not io.paused
    assert not list(tmp_path.rglob('dispatched.json'))
