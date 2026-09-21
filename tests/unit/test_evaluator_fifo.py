"""FIFO admission with actual local processes and locks; no experimental leaves."""
import gc
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import time

import pytest

from sciel.engine import capacity as cap, evaluator_fifo as fifo


class _Flag:
    """Lock-free fixture signal: killing a worker cannot poison Event's mutex."""
    def __init__(self, ctx):
        self.value = ctx.Value('b', 0, lock=False)

    def set(self):
        self.value.value = 1

    def is_set(self):
        return bool(self.value.value)

    def wait(self, timeout):
        deadline = time.monotonic() + timeout
        while not self.is_set() and time.monotonic() < deadline:
            time.sleep(.005)
        return self.is_set()


def _worker(root, label, writer, release, cancel, gate=None, fault=None, descendant_done=None):
    manager = cap.HostCapacity(root, evaluator_slots=2, poll_interval=.005)
    original = fifo._register
    registered = False
    gated = False

    def register(directory):
        nonlocal registered
        if fault == 'partial_counter':
            (directory / '.counter.crashed.tmp').write_bytes(b'{"next_ticket":')
            writer.send(('partial', label))
            os._exit(0)
        if fault == 'partial_waiter':
            original_write = fifo.os.write
            def interrupted(fd, data):
                original_write(fd, data[:8])
                writer.send(('partial', label))
                os._exit(0)
            fifo.os.write = interrupted
        path, fd = original(directory)
        registered = True
        writer.send(('registered', label, path.name))
        if fault == 'fork_then_die':
            child = os.fork()
            if child == 0:
                # at-fork cleanup must close inherited waiter AND guard FDs.
                release.wait(8)
                descendant_done.set()
                os._exit(0)
            writer.send(('forked', label, child))
            os._exit(0)
        return path, fd

    fifo._register = register

    def stopped():
        nonlocal gated
        if gate is not None and registered and not gated:
            gated = True
            writer.send(('gated', label))
            assert gate.wait(8), 'fixture parent did not release acquisition gate'
        return cancel.is_set()

    try:
        with manager.acquire('evaluator', cancel=stopped, timeout=10) as lease:
            writer.send(('acquired', label, lease.receipt()))
            assert release.wait(8), 'fixture parent did not release granted slot'
        writer.send(('released', label))
    except cap.CapacityCancelled:
        writer.send(('cancelled', label))
    except BaseException as exc:
        writer.send(('error', label, type(exc).__name__, str(exc)))
        raise
    finally:
        writer.close()


def _start(ctx, root, label, *, gate=None, fault=None, descendant_done=None):
    reader, writer = ctx.Pipe(duplex=False)
    release, cancel = _Flag(ctx), _Flag(ctx)
    process = ctx.Process(target=_worker,
        args=(str(root), label, writer, release, cancel, gate, fault, descendant_done))
    process.start()
    writer.close()
    return process, reader, release, cancel


def _message(worker, expected):
    assert worker[1].poll(8), f'Missing {expected} from process {worker[0].pid}'
    result = worker[1].recv()
    assert result[0] == expected, result
    return result


def _cleanup(workers):
    for process, reader, release, cancel in workers:
        cancel.set(); release.set()
    for process, reader, release, cancel in workers:
        process.join(3)
        if process.is_alive():
            process.kill(); process.join(3)
        reader.close()


def test_registered_old_waiter_cannot_be_overtaken_by_later_processes(tmp_path):
    ctx = multiprocessing.get_context('spawn')
    manager = cap.HostCapacity(tmp_path, evaluator_slots=2, poll_interval=.005)
    first, second = manager.acquire('evaluator'), manager.acquire('evaluator')
    gate = _Flag(ctx)
    workers = []
    try:
        oldest = _start(ctx, tmp_path, 'oldest', gate=gate); workers.append(oldest)
        old_ticket = _message(oldest, 'registered')[2]
        _message(oldest, 'gated')
        later = []
        for index in range(3):
            worker = _start(ctx, tmp_path, f'later-{index}')
            workers.append(worker); later.append(worker)
            assert _message(worker, 'registered')[2] > old_ticket
        first.release()
        # Slot is free, but the registered oldest has not polled again. Fresh
        # arrivals cannot grab it by winning a faster 50ms polling race.
        assert not any(worker[1].poll(.05) for worker in later)
        gate.set()
        _message(oldest, 'acquired')
        assert not any(worker[1].poll(.02) for worker in later)
        oldest[2].set(); _message(oldest, 'released')
        for worker in later:
            _message(worker, 'acquired')
            worker[2].set(); _message(worker, 'released')
        assert all(worker[0].exitcode in (None, 0) for worker in workers)
    finally:
        gate.set(); first.release(); second.release(); _cleanup(workers)
    assert not list(tmp_path.rglob('*.waiter'))


@pytest.mark.parametrize('retire', ['cancel', 'process_death'])
def test_cancelled_or_dead_head_cannot_block_next_process(tmp_path, retire):
    ctx = multiprocessing.get_context('spawn')
    manager = cap.HostCapacity(tmp_path, evaluator_slots=2, poll_interval=.005)
    holders = [manager.acquire('evaluator') for _ in range(2)]
    workers = []
    try:
        head = _start(ctx, tmp_path, 'head'); workers.append(head)
        _message(head, 'registered')
        next_waiter = _start(ctx, tmp_path, 'next'); workers.append(next_waiter)
        _message(next_waiter, 'registered')
        if retire == 'cancel':
            head[3].set(); _message(head, 'cancelled')
        else:
            head[0].kill()
        head[0].join(3)
        assert not head[0].is_alive()
        holders[0].release()
        _message(next_waiter, 'acquired')
        next_waiter[2].set(); _message(next_waiter, 'released')
    finally:
        for holder in holders: holder.release()
        _cleanup(workers)
    assert not list(tmp_path.rglob('*.waiter'))


@pytest.mark.parametrize('fault', ['partial_counter', 'partial_waiter'])
def test_process_death_during_metadata_write_leaves_no_stale_owner(tmp_path, fault):
    ctx = multiprocessing.get_context('spawn')
    worker = _start(ctx, tmp_path, 'crashed', fault=fault)
    try:
        _message(worker, 'partial'); worker[0].join(3)
        assert worker[0].exitcode == 0
        manager = cap.HostCapacity(tmp_path, evaluator_slots=2)
        with manager.acquire('evaluator', timeout=1):
            pass
        assert not list(tmp_path.rglob('*.waiter'))
        assert not list(tmp_path.rglob('.counter.*.tmp'))
    finally:
        _cleanup([worker])


@pytest.mark.skipif(not hasattr(os, 'fork'), reason='POSIX liveness inheritance')
def test_forked_descendant_does_not_keep_dead_waiter_or_guard_alive(tmp_path):
    ctx = multiprocessing.get_context('spawn')
    manager = cap.HostCapacity(tmp_path, evaluator_slots=2)
    holders = [manager.acquire('evaluator') for _ in range(2)]
    done = _Flag(ctx)
    worker = _start(ctx, tmp_path, 'parent', fault='fork_then_die', descendant_done=done)
    try:
        _message(worker, 'registered'); _message(worker, 'forked')
        worker[0].join(3)
        assert worker[0].exitcode == 0 and not done.is_set()
        holders[0].release()
        # Descendant is still alive, but cannot extend its dead parent's ticket
        # or the pool guard's ownership through an untracked duplicated FD.
        with manager.acquire('evaluator', timeout=1):
            assert not done.is_set()
        assert not list(tmp_path.rglob('*.waiter'))
    finally:
        worker[2].set()
        assert done.wait(3)
        for holder in holders: holder.release()
        _cleanup([worker])


def test_death_after_grant_before_work_releases_active_slot(tmp_path):
    ctx = multiprocessing.get_context('spawn')
    worker = _start(ctx, tmp_path, 'granted')
    manager = cap.HostCapacity(tmp_path, evaluator_slots=2)
    try:
        _message(worker, 'registered'); _message(worker, 'acquired')
        held = manager.acquire('evaluator', timeout=1)
        worker[0].kill(); worker[0].join(3)
        try:
            with manager.acquire('evaluator', timeout=1):
                pass
        finally:
            held.release()
    finally:
        _cleanup([worker])


def test_policy_is_versioned_and_refuses_legacy_in_both_directions(tmp_path):
    key = hashlib.sha256(b'evaluator').hexdigest()
    old = dict(schema=cap.SCHEMA, pool='evaluator', limit=2)
    (tmp_path / (key + '.json')).write_text(json.dumps(old))
    manager = cap.HostCapacity(tmp_path, evaluator_slots=2)
    with pytest.raises(cap.CapacityConfigurationError, match='admission policy'):
        manager.acquire('evaluator', timeout=0)
    assert json.loads((tmp_path / (key + '.json')).read_text()) == old
    fresh = cap.HostCapacity(tmp_path / 'fresh', evaluator_slots=2)
    with fresh.acquire('evaluator', timeout=0):
        pass
    policy = json.loads((fresh.root / (key + '.json')).read_text())
    assert policy == {**old, 'evaluator_admission': cap.EVALUATOR_ADMISSION}
    # Legacy _try_pool requires exact dictionary equality with `old`, so it
    # rejects this policy instead of bypassing the new waiters.
    assert policy != old
    with pytest.raises(cap.CapacityConfigurationError, match='FIFO'):
        fresh._try_pool('evaluator', 2)
    with manager.acquire('llm', provider='a', timeout=0):
        pass  # LLM/provider compatibility is independent of evaluator policy.


def test_waiter_unlink_error_never_constructs_lease_or_leaks_descriptors(tmp_path, monkeypatch):
    manager = cap.HostCapacity(tmp_path, evaluator_slots=2)
    before = set(cap._OPEN_FDS)
    original_unlink, original_lease = Path.unlink, cap.CapacityLease
    calls = []
    def unlink(path, *args, **kwargs):
        if path.name.endswith('.waiter'):
            raise OSError('fixture waiter retirement failed')
        return original_unlink(path, *args, **kwargs)
    def lease(*args, **kwargs):
        calls.append(True)
        return original_lease(*args, **kwargs)
    monkeypatch.setattr(Path, 'unlink', unlink)
    monkeypatch.setattr(cap, 'CapacityLease', lease)
    with pytest.raises(OSError, match='retirement'):
        manager.acquire('evaluator', timeout=0)
    assert not calls and cap._OPEN_FDS == before
    monkeypatch.setattr(Path, 'unlink', original_unlink)
    with manager.acquire('evaluator', timeout=0) as active:
        gc.collect()
        with manager.acquire('evaluator', timeout=0):
            with pytest.raises(cap.CapacityTimeout):
                manager.acquire('evaluator', timeout=0)
        assert not active.released
    assert cap._OPEN_FDS == before


def test_cancellation_callback_error_retires_registered_waiter(tmp_path):
    manager = cap.HostCapacity(tmp_path, evaluator_slots=1, poll_interval=.005)
    with manager.acquire('evaluator'):
        before = set(cap._OPEN_FDS)
        calls = 0
        def cancelled():
            nonlocal calls
            calls += 1
            if calls == 2:
                raise RuntimeError('fixture cancel callback failed')
            return False
        with pytest.raises(RuntimeError, match='callback'):
            manager.acquire('evaluator', cancel=cancelled)
        assert cap._OPEN_FDS == before and not list(tmp_path.rglob('*.waiter'))


def test_live_malformed_ticket_fails_closed_but_dead_partial_ticket_is_reclaimed(tmp_path):
    manager = cap.HostCapacity(tmp_path, evaluator_slots=1)
    with manager.acquire('evaluator'):
        directory = tmp_path / (fifo._KEY + '.fifo')
        path = directory / ('00000000000000000000-' + 'a'*32 + '.waiter')
        fd = cap._open_lock(path, exclusive=True)
        assert cap._try_lock(fd)
        os.write(fd, b'{')
        try:
            with pytest.raises(cap.CapacityConfigurationError, match='queue metadata'):
                manager.acquire('evaluator', timeout=0)
        finally:
            cap._close(fd)
    with manager.acquire('evaluator', timeout=1):
        pass
    assert not list(tmp_path.rglob('*.waiter'))
