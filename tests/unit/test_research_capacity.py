"""Long tool jobs cannot monopolize shared candidate-evaluation capacity."""
import multiprocessing
import threading

import pytest

from sciel.engine.capacity import (HostCapacity, CapacityCancelled,
                                   CapacityTimeout, CapacityConfigurationError)


def _manager(root):
    return HostCapacity(root, evaluator_slots=4, research_slots=2, poll_interval=.005)


def _hold_research(root, connection):
    with _manager(root).acquire('research_evaluator', timeout=5):
        connection.send('acquired')
        connection.recv()


def test_long_research_leaves_two_ordinary_slots_and_cancels_without_dispatch(tmp_path):
    manager = _manager(tmp_path)
    research = [manager.acquire('research_evaluator', timeout=1) for _ in range(2)]
    stopped = threading.Event()
    result = []
    def wait():
        try:
            with manager.acquire('research_evaluator', cancel=stopped, timeout=3):
                result.append('unexpectedly admitted')
        except CapacityCancelled:
            result.append('cancelled')
    worker = threading.Thread(target=wait)
    worker.start()
    try:
        with manager.acquire('evaluator', timeout=1), manager.acquire('evaluator', timeout=1):
            with pytest.raises(CapacityTimeout):
                manager.acquire('evaluator', timeout=0)
            assert not result
        stopped.set()
        worker.join(2)
        assert not worker.is_alive()
        assert result == ['cancelled']
        assert research[0].receipt()['pools'] == [
            {'pool': 'research_evaluator', 'limit': 2}, {'pool': 'evaluator', 'limit': 4}]
    finally:
        stopped.set()
        for lease in research:
            lease.release()
        worker.join(2)
    # Cancellation did not leak either lease or a live queue ticket.
    with manager.acquire('research_evaluator', timeout=1), manager.acquire('research_evaluator', timeout=1):
        pass


def test_cancel_while_waiting_for_total_releases_research_class(tmp_path):
    manager = _manager(tmp_path)
    held = [manager.acquire('evaluator', timeout=1) for _ in range(4)]
    try:
        with pytest.raises(CapacityTimeout):
            manager.acquire('research_evaluator', timeout=.03)
    finally:
        for lease in held:
            lease.release()
    with manager.acquire('research_evaluator', timeout=1), manager.acquire('research_evaluator', timeout=1):
        pass


def test_process_loss_releases_both_class_and_total_leases(tmp_path):
    ctx = multiprocessing.get_context('spawn')
    parent, child = ctx.Pipe()
    worker = ctx.Process(target=_hold_research, args=(str(tmp_path), child))
    worker.start()
    try:
        assert parent.poll(10)
        assert parent.recv() == 'acquired'
        worker.terminate()
        worker.join(5)
        assert not worker.is_alive()
        manager = _manager(tmp_path)
        with manager.acquire('research_evaluator', timeout=1), manager.acquire('research_evaluator', timeout=1):
            with manager.acquire('evaluator', timeout=1), manager.acquire('evaluator', timeout=1):
                pass
    finally:
        if worker.is_alive():
            worker.terminate()
            worker.join(5)
        parent.close()
        child.close()


def test_conflicting_research_limits_fail_closed(tmp_path):
    with _manager(tmp_path).acquire('research_evaluator', timeout=1):
        other = HostCapacity(tmp_path, evaluator_slots=4, research_slots=1)
        with pytest.raises(CapacityConfigurationError):
            other.acquire('research_evaluator', timeout=0)


@pytest.mark.parametrize('slots', [0, -1, True, 4, 5])
def test_class_limit_must_leave_ordinary_capacity(tmp_path, slots):
    with pytest.raises(ValueError):
        HostCapacity(tmp_path, evaluator_slots=4, research_slots=slots)


def test_unconfigured_class_preserves_single_slot_behavior(tmp_path):
    manager = HostCapacity(tmp_path, evaluator_slots=1)
    with manager.acquire('research_evaluator', timeout=1):
        with pytest.raises(CapacityTimeout):
            manager.acquire('evaluator', timeout=0)


def test_foreground_tool_uses_research_admission_and_evaluation_does_not(tmp_path):
    from sciel.engine.parallel_calls import CallLedger
    manager = _manager(tmp_path/'capacity')
    kinds = []
    original = manager.acquire
    def acquire(kind, **kwargs):
        kinds.append(kind)
        return original(kind, **kwargs)
    manager.acquire = acquire
    ledger = CallLedger(tmp_path/'wave', manager, 'test')
    directory, request_hash, _ = ledger.prepare('tool', 'tool_execution', {'test': True})
    result = ledger.perform(directory, request_hash, 'tool_execution', lambda: {'result': 'saved'})
    assert result['status'] == 'returned'
    assert kinds[0] == 'research_evaluator'
    assert 'evaluator' in kinds  # Class admission precedes total admission.


def test_background_capacity_config_accepts_class_limit_but_not_monopoly():
    from sciel.research_tools.jobs import _config
    capacity = dict(capacity_dir=None, host_llm_slots=12, host_evaluator_slots=4,
                    host_research_slots=2)
    assert _config({'capacity': capacity})['capacity'] == capacity
    for invalid in [4, 0, True]:
        with pytest.raises(ValueError):
            _config({'capacity': dict(capacity, host_research_slots=invalid)})


def test_fork_between_class_and_total_cannot_inherit_research_admission(tmp_path):
    import os
    manager = _manager(tmp_path)
    original = manager.acquire
    parent_pid = os.getpid()
    read_fd, write_fd = os.pipe()
    child_pid = None
    def acquire(kind, **kwargs):
        nonlocal child_pid
        if kind == 'evaluator' and child_pid is None:
            child_pid = os.fork()
        return original(kind, **kwargs)
    manager.acquire = acquire
    try:
        try:
            lease = original('research_evaluator', timeout=2)
        except CapacityConfigurationError:
            if os.getpid() != parent_pid:
                os.write(write_fd, b'refused')
                os._exit(0)
            raise
        if os.getpid() != parent_pid:
            lease.release()
            os.write(write_fd, b'bad inherited permission')
            os._exit(1)
        lease.release()
        _, status = os.waitpid(child_pid, 0)
        assert status == 0
        os.close(write_fd)
        write_fd = None
        assert os.read(read_fd, 100) == b'refused'
        with original('research_evaluator', timeout=1), original('research_evaluator', timeout=1):
            pass
    finally:
        os.close(read_fd)
        if write_fd is not None:
            os.close(write_fd)
