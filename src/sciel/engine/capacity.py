"""Cooperative, host-wide admission limits for external work (POSIX).

Every run must use the same root and pool limits. Permanent advisory-lock files
represent slots; a crashed process cannot leave a stale lease in a JSON ledger.
This limits concurrency, not requests/tokens per minute or spend. It changes no
evaluator environment, resource limit, numerical setting or scientific state.
"""
from __future__ import annotations

from datetime import datetime, timezone
import errno
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import stat
import tempfile
import threading
import time
import uuid


SCHEMA = "sciel-host-capacity/1"
EVALUATOR_ADMISSION = "sciel-evaluator-fifo/1"
MAX_PROVIDER_COOLDOWN_SECONDS = 3600.0
_FD_LOCK = threading.RLock()
_OPEN_FDS = set()


def _fork_child():
    global _FD_LOCK
    # Closing inherited descriptors does not unlock the parent's open-file
    # description. Explicit LOCK_UN here would incorrectly release its lease.
    for fd in tuple(_OPEN_FDS):
        try:
            os.close(fd)
        except OSError:
            pass
    _OPEN_FDS.clear()
    _FD_LOCK = threading.RLock()


def _fork_before():
    _FD_LOCK.acquire()


def _fork_parent():
    _FD_LOCK.release()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(before=_fork_before, after_in_parent=_fork_parent,
                        after_in_child=_fork_child)


def _open_lock(path, *, exclusive=False):
    with _FD_LOCK:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | (os.O_EXCL if exclusive else 0) | getattr(os, "O_CLOEXEC", 0)
                     | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("Capacity lock must be a regular file")
            os.set_inheritable(fd, False)
            _OPEN_FDS.add(fd)
            return fd
        except BaseException:
            os.close(fd)
            raise


def _close(fd):
    with _FD_LOCK:
        if fd in _OPEN_FDS:
            _OPEN_FDS.remove(fd)
            os.close(fd)


def _try_lock(fd):
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError as exc:
        if exc.errno in (errno.EACCES, errno.EAGAIN):
            return False
        raise


def _positive_slots(value):
    if type(value) is not int or not 1 <= value <= 4096:
        raise ValueError("Capacity must be an integer from 1 to 4096")
    return value


def _provider(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 256:
        raise ValueError("Provider capacity key must be a nonempty string of at most 256 characters")
    return value


class CapacityConfigurationError(ValueError):
    """One pool has different declared limits; no request was dispatched."""


class CapacityCancelled(Exception):
    def __init__(self, wait_seconds):
        super().__init__("Capacity wait cancelled before dispatch")
        self.wait_seconds = wait_seconds


class CapacityTimeout(TimeoutError):
    def __init__(self, wait_seconds):
        super().__init__("Capacity wait timed out before dispatch")
        self.wait_seconds = wait_seconds


class CapacityLease:
    """A process-owned permit. Release after work actually terminates, not on submit."""

    def __init__(self, descriptors, *, kind, provider, wait_seconds, pools):
        self._descriptors = descriptors
        self._pid = os.getpid()
        self._lock = threading.Lock()
        self._released = False
        self.id = uuid.uuid4().hex
        self.kind, self.provider = kind, provider
        self.wait_seconds = wait_seconds
        self.acquired_at = datetime.now(timezone.utc).isoformat()
        self.pools = tuple(pools)

    @property
    def released(self):
        return self._released or os.getpid() != self._pid

    def release(self):
        if os.getpid() != self._pid:
            # The at-fork hook already closed these FDs in the child. Their
            # numbers might now refer to unrelated files, so never close again.
            return
        with self._lock:
            if not self._released:
                self._released = True
                for fd in reversed(self._descriptors):
                    _close(fd)
                self._descriptors = []

    def receipt(self):
        return dict(schema=SCHEMA, lease_id=self.id, kind=self.kind,
                    provider=self.provider, wait_seconds=self.wait_seconds,
                    acquired_at=self.acquired_at, pid=self._pid,
                    pools=[dict(pool=name, limit=limit) for name, limit in self.pools])

    def __enter__(self):
        if self.released:
            raise RuntimeError("Capacity lease has already been released")
        return self

    def __exit__(self, *exc):
        self.release()

    def __del__(self):
        if hasattr(self, "_released"):
            self.release()


class ResearchCapacityLease(CapacityLease):
    """Hold the research-class permit and total evaluator permit together."""

    def __init__(self, research, total, research_slots):
        self._children = (research, total)
        super().__init__([], kind='research_evaluator', provider=None,
            wait_seconds=research.wait_seconds + total.wait_seconds,
            pools=[('research_evaluator', research_slots), *total.pools])

    def release(self):
        if os.getpid() != self._pid:
            return
        with self._lock:
            if not self._released:
                self._released = True
                for lease in reversed(self._children):
                    lease.release()


class HostCapacity:
    """Shared host slots with optional provider and research-class limits.

    LLM/provider acquisition is all-or-none. Research acquires its class FIFO
    before the total evaluator FIFO, preserving ordinary evaluation capacity.

    ``cancel`` is a callable or an object with ``is_set()``. It only cancels the
    queue wait: once the lease is returned, the caller owns stopping/draining its
    actual work. Evaluators are FIFO by registered waiter ticket, including
    foreground and background tools. LLM/provider acquisition remains all-or-none
    without FIFO. Neither policy promises a deadline or multi-host fairness.
    """

    def __init__(self, root=None, *, llm_slots=8, evaluator_slots=2,
                 provider_limits=None, poll_interval=.05, research_slots=None):
        self.llm_slots = _positive_slots(llm_slots)
        self.evaluator_slots = _positive_slots(evaluator_slots)
        self.research_slots = None if research_slots is None else _positive_slots(research_slots)
        if self.research_slots is not None and self.research_slots >= self.evaluator_slots:
            raise ValueError("Research capacity must leave at least one ordinary evaluator slot")
        if provider_limits is not None and type(provider_limits) is not dict:
            raise ValueError("provider_limits must be a mapping")
        self.provider_limits = {_provider(name): _positive_slots(limit)
                                for name, limit in (provider_limits or {}).items()}
        if type(poll_interval) not in (int, float) or not math.isfinite(poll_interval) or not 0 < poll_interval <= 1:
            raise ValueError("poll_interval must be finite, positive and at most one second")
        self.poll_interval = poll_interval
        if root is None:
            host = hashlib.sha256(socket.gethostname().encode()).hexdigest()[:16]
            root = Path.home() / ".cache" / "sciel" / "capacity" / host
        self.root = Path(root).expanduser().absolute()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        info = self.root.lstat()
        if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o022:
            raise ValueError("Capacity root must be a private, user-owned directory, not a symlink")

    def _cooldown(self, provider):
        key = hashlib.sha256(('provider:' + provider).encode()).hexdigest()
        path = self.root / (key + '.cooldown.json')
        if not path.exists() and not path.is_symlink():
            return None
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_size > 4096:
            raise CapacityConfigurationError('Invalid provider cooldown')
        try:
            value = json.loads(path.read_bytes())
        except (OSError, ValueError, UnicodeError) as exc:
            raise CapacityConfigurationError('Unreadable provider cooldown') from exc
        if (type(value) is not dict or value.get('schema') != SCHEMA
                or value.get('provider') != provider
                or any(type(value.get(name)) not in (int, float)
                       or not math.isfinite(value[name]) for name in ('recorded_at_unix', 'until_unix'))
                or not 0 <= value['until_unix'] - value['recorded_at_unix'] <= MAX_PROVIDER_COOLDOWN_SECONDS):
            raise CapacityConfigurationError('Invalid provider cooldown')
        return value

    def defer_provider(self, provider, seconds):
        """Publish a bounded provider wait; never retry or classify call costs.

        All processes sharing this root observe it. Already granted leases are
        not revoked. Publish only after preserving the original error receipt;
        failure of this optional scheduling hint must not erase that receipt.
        """
        provider = _provider(provider)
        if (type(seconds) not in (int, float) or not math.isfinite(seconds)
                or not 0 <= seconds <= MAX_PROVIDER_COOLDOWN_SECONDS):
            raise ValueError('Provider cooldown must be finite and between 0 and 3600 seconds')
        key = hashlib.sha256(('provider:' + provider).encode()).hexdigest()
        guard = _open_lock(self.root / (key + '.guard'))
        temporary = None
        try:
            while not _try_lock(guard):
                time.sleep(self.poll_interval)
            now = time.time()
            previous = self._cooldown(provider)
            until = now + seconds
            if previous is not None and previous['until_unix'] >= until:
                return dict(previous)
            value = dict(schema=SCHEMA, provider=provider, recorded_at_unix=now, until_unix=until)
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.root,
                    prefix=key + '.', suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(value, stream, sort_keys=True, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.root / (key + '.cooldown.json'))
            directory = os.open(self.root, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
            return dict(value)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            _close(guard)

    def _try_pool(self, name, limit):
        if name == 'evaluator':
            raise CapacityConfigurationError('Evaluator permits require FIFO admission')
        key = hashlib.sha256(name.encode()).hexdigest()
        guard = _open_lock(self.root / (key + ".guard"))
        try:
            if not _try_lock(guard):
                return None
            if name.startswith('provider:'):
                cooldown = self._cooldown(name.removeprefix('provider:'))
                if cooldown is not None and cooldown['until_unix'] > time.time():
                    return None
            self._pool_policy(name, limit)
            return self._try_slot(key, limit)
        finally:
            _close(guard)

    def _pool_policy(self, name, limit):
        """Validate under the pool guard; never migrate another admission policy."""
        key = hashlib.sha256(name.encode()).hexdigest()
        policy = self.root / (key + '.json')
        expected = dict(schema=SCHEMA, pool=name, limit=limit)
        if name == 'evaluator':
            expected['evaluator_admission'] = EVALUATOR_ADMISSION
        if policy.exists() or policy.is_symlink():
            info = policy.lstat()
            if not stat.S_ISREG(info.st_mode) or info.st_size > 4096:
                raise CapacityConfigurationError('Invalid capacity pool policy')
            try:
                actual = json.loads(policy.read_bytes())
            except (OSError, ValueError, UnicodeError) as exc:
                raise CapacityConfigurationError('Unreadable capacity pool policy') from exc
            if actual != expected:
                raise CapacityConfigurationError(
                    'Conflicting capacity limits/admission policy; use one fresh shared capacity root for every run')
            return
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.root,
                                             prefix=key + '.', suffix='.tmp', delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(expected, stream, sort_keys=True)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, policy)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    def _try_slot(self, key, limit):
        """Take a kernel-owned slot while its admission guard is held."""
        for slot in range(limit):
            fd = _open_lock(self.root / f'{key}.{slot}.slot')
            try:
                if _try_lock(fd):
                    return fd
            except BaseException:
                _close(fd)
                raise
            _close(fd)
        return None

    def acquire(self, kind, *, provider=None, cancel=None, timeout=None):
        if kind not in ("llm", "evaluator", "research_evaluator"):
            raise ValueError("Capacity kind must be llm, evaluator or research_evaluator")
        if provider is not None:
            provider = _provider(provider)
        if kind == "llm" and provider is None:
            raise ValueError("LLM capacity requires an explicit provider key")
        if kind in ("evaluator", "research_evaluator") and provider is not None:
            raise ValueError("Evaluator capacity has no provider dimension")
        if timeout is not None and (type(timeout) not in (int, float)
                or not math.isfinite(timeout) or timeout < 0):
            raise ValueError("Capacity timeout must be finite and nonnegative")
        cancelled = (lambda: False) if cancel is None else cancel if callable(cancel) else getattr(cancel, "is_set", None)
        if not callable(cancelled):
            raise TypeError("cancel must be a callable or an event")
        if kind == 'research_evaluator':
            if self.research_slots is None:
                return self.acquire('evaluator', cancel=cancel, timeout=timeout)
            # Queue for the class first: waiters must never occupy total slots
            # while waiting for research capacity. Both FIFO namespaces use
            # kernel leases, so cancellation, process exit and fork are safe.
            research = HostCapacity(self.root / 'research',
                evaluator_slots=self.research_slots, poll_interval=self.poll_interval)
            begun, owner = time.monotonic(), os.getpid()
            first = research.acquire('evaluator', cancel=cancel, timeout=timeout)
            try:
                if os.getpid() != owner or first.released:
                    raise CapacityConfigurationError('Restart research acquisition after fork')
                remaining = None if timeout is None else max(0., timeout - (time.monotonic() - begun))
                if timeout is not None and timeout > 0 and remaining == 0:
                    raise CapacityTimeout(time.monotonic() - begun)
                total = self.acquire('evaluator', cancel=cancel, timeout=remaining)
                if os.getpid() != owner or first.released:
                    total.release()
                    raise CapacityConfigurationError('Restart research acquisition after fork')
                return ResearchCapacityLease(first, total, self.research_slots)
            except BaseException as error:
                first.release()
                if isinstance(error, (CapacityCancelled, CapacityTimeout)):
                    error.wait_seconds = time.monotonic() - begun
                raise
        if kind == 'evaluator':
            from .evaluator_fifo import acquire
            return acquire(self, cancelled, cancel=cancel, timeout=timeout)
        pools = [(kind, self.llm_slots if kind == "llm" else self.evaluator_slots)]
        if kind == "llm":
            pools.append(("provider:" + provider, self.provider_limits.get(provider, self.llm_slots)))
        started = time.monotonic()
        while True:
            acquired = []
            try:
                if cancelled():
                    raise CapacityCancelled(time.monotonic() - started)
                for name, limit in pools:
                    fd = self._try_pool(name, limit)
                    if fd is None:
                        break
                    acquired.append(fd)
                if len(acquired) == len(pools):
                    waited = time.monotonic() - started
                    if cancelled():
                        raise CapacityCancelled(waited)
                    if timeout is not None and timeout > 0 and waited > timeout:
                        raise CapacityTimeout(waited)
                    lease = CapacityLease(acquired, kind=kind, provider=provider,
                                          wait_seconds=waited, pools=pools)
                    acquired = []
                    return lease
            finally:
                for fd in reversed(acquired):
                    _close(fd)
            waited = time.monotonic() - started
            if timeout is not None and waited >= timeout:
                raise CapacityTimeout(waited)
            delay = self.poll_interval if timeout is None else min(self.poll_interval, max(0, timeout - waited))
            wait = getattr(cancel, "wait", None)
            if callable(wait):
                wait(delay)
            else:
                time.sleep(delay)
