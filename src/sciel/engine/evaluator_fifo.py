"""FIFO evaluator admission; slot and waiter ownership are kernel file locks.

A ticket is published under the pool guard while its liveness FD is locked.
Only the oldest live waiter can acquire a slot. Stale tickets are reclaimed by
their released locks, never by PID reuse, wall-clock age or heartbeat guesses.
All clients must agree on the versioned evaluator policy; legacy pools are not
silently upgraded. Queue registration is the FIFO linearization point.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import time
import uuid

from . import capacity as cap

_KEY = hashlib.sha256(b'evaluator').hexdigest()
_TICKET = re.compile(r'([0-9]{20})-([a-f0-9]{32})\.waiter')


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _json(path):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > 4096:
        raise cap.CapacityConfigurationError('Invalid evaluator queue metadata')
    try:
        return json.loads(path.read_bytes())
    except (OSError, ValueError, UnicodeError) as exc:
        raise cap.CapacityConfigurationError('Unreadable evaluator queue metadata') from exc


def _write_counter(path, value):
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=path.parent,
                                         prefix='.counter.', suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(value, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _queue(manager):
    directory = manager.root / (_KEY + '.fifo')
    directory.mkdir(mode=0o700, exist_ok=True)
    info = directory.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
        raise cap.CapacityConfigurationError('Evaluator queue must be a private user-owned directory')
    # The pool guard proves that no live writer is using a counter temporary.
    for path in directory.glob('.counter.*.tmp'):
        path.unlink()
    return directory


def _register(directory):
    counter = directory / 'counter.json'
    if counter.exists() or counter.is_symlink():
        value = _json(counter)
        if (type(value) is not dict or set(value) != {'schema', 'next_ticket'}
                or value['schema'] != cap.EVALUATOR_ADMISSION
                or type(value['next_ticket']) is not int or not 0 <= value['next_ticket'] < 10**20):
            raise cap.CapacityConfigurationError('Invalid evaluator ticket counter')
        sequence = value['next_ticket']
        if any(int(match[1]) >= sequence for path in directory.glob('*.waiter')
               if (match := _TICKET.fullmatch(path.name))):
            raise cap.CapacityConfigurationError('Evaluator ticket counter precedes an existing waiter')
    else:
        if list(directory.glob('*.waiter')):
            raise cap.CapacityConfigurationError('Evaluator ticket counter is missing')
        sequence = 0
    _write_counter(counter, {'schema': cap.EVALUATOR_ADMISSION, 'next_ticket': sequence + 1})
    nonce = uuid.uuid4().hex
    path = directory / f'{sequence:020d}-{nonce}.waiter'
    fd = cap._open_lock(path, exclusive=True)
    try:
        if not cap._try_lock(fd):
            raise cap.CapacityConfigurationError('New evaluator waiter is already locked')
        value = {'schema': cap.EVALUATOR_ADMISSION, 'ticket': sequence, 'nonce': nonce}
        data = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
        # Do not duplicate the liveness FD: every open-file description must be
        # covered by capacity's at-fork cleanup, including interrupted writers.
        while data:
            written = os.write(fd, data)
            if written <= 0:
                raise OSError('Evaluator waiter metadata write did not progress')
            data = data[written:]
        os.fsync(fd)
        _sync_directory(directory)
        return path, fd
    except BaseException:
        cap._close(fd)
        path.unlink(missing_ok=True)
        raise


def _head(directory, own_path):
    """Prune only unlocked entries; corrupt *live* evidence fails closed."""
    first = None
    for path in sorted(directory.glob('*.waiter')):
        match = _TICKET.fullmatch(path.name)
        if match is None:
            raise cap.CapacityConfigurationError('Invalid evaluator waiter filename')
        if path != own_path:
            fd = cap._open_lock(path)
            try:
                if cap._try_lock(fd):
                    # Even an interrupted partial write is harmless once its
                    # kernel liveness lease is gone. Unlink under the guard.
                    path.unlink()
                    continue
            finally:
                cap._close(fd)
        expected = {'schema': cap.EVALUATOR_ADMISSION, 'ticket': int(match[1]), 'nonce': match[2]}
        value = _json(path)
        if type(value) is not dict or type(value.get('ticket')) is not int or value != expected:
            raise cap.CapacityConfigurationError('Evaluator waiter metadata differs')
        if first is None:
            first = path
    return first


def _retire(path, fd, manager):
    # Closing is enough for other waiters to reclaim us even if a guard is busy.
    cap._close(fd)
    try:
        guard = cap._open_lock(manager.root / (_KEY + '.guard'))
    except OSError:
        return  # Liveness is already released; a later waiter can prune it.
    try:
        if cap._try_lock(guard):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass  # Never replace cancellation/original failure with cleanup.
    finally:
        cap._close(guard)


def acquire(manager, cancelled, *, cancel, timeout):
    begun, owner = time.monotonic(), os.getpid()
    waiter_path = waiter_fd = None
    try:
        while True:
            if os.getpid() != owner:
                raise cap.CapacityConfigurationError('Restart evaluator acquisition after fork')
            if cancelled():
                raise cap.CapacityCancelled(time.monotonic() - begun)
            if os.getpid() != owner:
                raise cap.CapacityConfigurationError('Restart evaluator acquisition after fork')
            guard = cap._open_lock(manager.root / (_KEY + '.guard'))
            slot = None
            try:
                if cap._try_lock(guard):
                    manager._pool_policy('evaluator', manager.evaluator_slots)
                    directory = _queue(manager)
                    if waiter_path is None:
                        waiter_path, waiter_fd = _register(directory)
                    if _head(directory, waiter_path) == waiter_path:
                        slot = manager._try_slot(_KEY, manager.evaluator_slots)
                        if slot is not None:
                            waited = time.monotonic() - begun
                            if cancelled():
                                raise cap.CapacityCancelled(waited)
                            if os.getpid() != owner:
                                raise cap.CapacityConfigurationError('Restart evaluator acquisition after fork')
                            if timeout is not None and timeout > 0 and waited > timeout:
                                raise cap.CapacityTimeout(waited)
                            waiter_path.unlink()
                            cap._close(waiter_fd)
                            waiter_path = waiter_fd = None
                            lease = cap.CapacityLease([slot], kind='evaluator', provider=None,
                                wait_seconds=waited, pools=[('evaluator', manager.evaluator_slots)])
                            slot = None  # Ownership transferred to the returned lease.
                            return lease
            finally:
                if os.getpid() == owner:
                    if slot is not None:
                        cap._close(slot)
                    cap._close(guard)
            waited = time.monotonic() - begun
            if timeout is not None and waited >= timeout:
                raise cap.CapacityTimeout(waited)
            delay = manager.poll_interval if timeout is None else min(manager.poll_interval, max(0, timeout - waited))
            wait = getattr(cancel, 'wait', None)
            if callable(wait):
                wait(delay)
            else:
                time.sleep(delay)
    finally:
        if waiter_fd is not None and os.getpid() == owner:
            _retire(waiter_path, waiter_fd, manager)
