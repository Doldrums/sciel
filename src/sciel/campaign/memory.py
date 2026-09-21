"""Optional, sampled process memory telemetry; never a memory resource limit.

The observer lives outside the measured process. RSS sums may double count shared
pages; USS is reported only for samples where every observed process supports it.
"""
from __future__ import annotations

from collections import Counter
from contextlib import ExitStack
import errno
import math
import os
import platform
import signal
import subprocess
import time


def _psutil():
    try:
        import psutil
    except ImportError as exc:
        raise RuntimeError("Memory measurement requires optional dependency psutil") from exc
    return psutil


def _positive(value, name):
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return float(value)


class ProcessTreeSampler:
    """Call ``sample`` periodically, then ``report``; no process is killed here.

    Descendants already observed remain tracked if their parent exits. Identity is
    (PID, creation time), so PID reuse cannot turn another process into a descendant.
    This class is deliberately synchronous; its owner controls lifecycle/cadence.
    """

    def __init__(self, pid, *, interval_seconds=.02):
        self.interval_seconds = _positive(interval_seconds, "interval_seconds")
        self.backend = _psutil()
        self.started = time.monotonic()
        self.root_pid = pid
        self.processes = {}
        self.errors = Counter()
        self.counts = Counter()
        self.peaks = {f"sampled_peak_{scope}_{metric}_bytes": None
                      for scope in ("root", "observed_tree") for metric in ("rss", "uss")}
        self.last_sample_time = None
        self.max_sample_gap_seconds = 0.
        self.max_observed_processes = 0
        self.root_identity = None
        try:
            process = self.backend.Process(pid)
            self.root_identity = (pid, process.create_time())
            self.processes[self.root_identity] = process
        except self.backend.Error as exc:
            self._error("root_identity", exc)

    def _error(self, operation, exc):
        self.errors[f"{operation}:{type(exc).__name__}"] += 1

    def sample(self):
        now = time.monotonic()
        if self.last_sample_time is not None:
            self.max_sample_gap_seconds = max(self.max_sample_gap_seconds, now - self.last_sample_time)
        self.last_sample_time = now
        self.counts["polls"] += 1
        discovery_ok = True
        # Discover from every still-observed descendant, including reparented ones.
        for identity, process in list(self.processes.items()):
            try:
                if not process.is_running() or process.create_time() != identity[1]:
                    continue
                for child in process.children(recursive=True):
                    self.processes.setdefault((child.pid, child.create_time()), child)
            except self.backend.Error as exc:
                discovery_ok = False
                self._error("discover", exc)
        rss, uss, root_rss, root_uss = [], [], None, None
        live = 0
        for identity, process in list(self.processes.items()):
            try:
                if not process.is_running() or process.create_time() != identity[1]:
                    continue
                live += 1
                value = process.memory_info().rss
                rss.append(value)
                if identity == self.root_identity:
                    root_rss = value
                try:
                    value = getattr(process.memory_full_info(), "uss", None)
                    if value is None:
                        self.counts["uss_unsupported_observations"] += 1
                    else:
                        uss.append(value)
                    if identity == self.root_identity:
                        root_uss = value
                except self.backend.Error as exc:
                    self._error("uss", exc)
            except self.backend.Error as exc:
                self._error("rss", exc)
        self.max_observed_processes = max(self.max_observed_processes, live)
        values = {"root_rss": root_rss, "root_uss": root_uss,
                  "observed_tree_rss": sum(rss) if rss else None,
                  "observed_tree_uss": sum(uss) if live and len(uss) == live else None}
        for name, value in values.items():
            if value is not None:
                self.counts[f"{name}_samples"] += 1
                field = f"sampled_peak_{name}_bytes"
                self.peaks[field] = max(self.peaks[field] or 0, value)
        if not discovery_ok or len(rss) != live:
            self.counts["incomplete_observed_tree_samples"] += 1
        return {"elapsed_seconds": now - self.started, "observed_processes": live,
                "discovery_ok": discovery_ok, **{f"{k}_bytes": v for k, v in values.items()}}

    def report(self):
        return {"schema": "sciel-sampled-process-memory/1", "root_pid": self.root_pid,
                "platform": platform.platform(), "psutil_version": self.backend.__version__,
                "requested_interval_seconds": self.interval_seconds,
                "max_sample_gap_seconds": self.max_sample_gap_seconds,
                "elapsed_seconds": time.monotonic() - self.started,
                "scope": "root process and discovered descendants; observer excluded",
                **self.peaks, "sample_counts": dict(self.counts),
                "max_observed_processes": self.max_observed_processes,
                "observed_process_identities": [list(p) for p in self.processes],
                "errors": dict(self.errors), "complete_tree_coverage_guaranteed": False,
                "limitations": [
                    "Sampled maxima, not exact peaks; allocations between polls can be missed.",
                    "Short-lived or inaccessible descendants may be missed; process reads are not simultaneous.",
                    "Observed-tree RSS sums can double count shared pages; USS availability is platform/permission dependent.",
                    "Missing measurements are null, never inferred zero; telemetry is not a memory cap."]}


def _child_state(proc):
    """Observe our direct child without releasing its PID on supported POSIX."""
    if proc.returncode is not None:
        return "reaped"
    names = ("waitid", "P_PID", "WEXITED", "WNOHANG", "WNOWAIT")
    if os.name != "posix" or not all(hasattr(os, name) for name in names):
        return "unsupported"
    try:
        status = os.waitid(os.P_PID, proc.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
    except ChildProcessError:
        return "unowned"
    except OSError as exc:
        if exc.errno in {errno.ENOSYS, errno.EINVAL, errno.ENOTSUP}:
            return "unsupported"
        raise
    return "exited" if status is not None and status.si_pid else "running"


def _group_has_live_members(pgid):
    """Check a still-reserved PGID; do not treat inaccessible members as absent."""
    backend = _psutil()
    for process in backend.process_iter():
        try:
            if os.getpgid(process.pid) == pgid and process.status() != backend.STATUS_ZOMBIE:
                return True
        except (ProcessLookupError, backend.NoSuchProcess):
            continue
    return False


def _observed_executing(process, identity, backend):
    return (process.is_running() and process.create_time() == identity[1]
            and process.status() != backend.STATUS_ZOMBIE)


def _cleanup(proc, sampler, timeout):
    """Signal the owned group before reaping; otherwise use observed identities."""
    errors = []
    deadline = time.monotonic() + timeout
    group_killed = False
    try:
        state = _child_state(proc)
    except OSError as exc:
        state = "unknown"
        errors.append(f"ownership_probe:{type(exc).__name__}")
    # WNOWAIT keeps our exited child waitable, reserving its PID/PGID until the
    # group signal. Do not use a bare killpg(pid, 0) as an ownership check after
    # poll()/wait(): a new, unrelated group may already have reused that number.
    group_skip_reason = None
    if os.name == "posix" and state in {"running", "exited"}:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            group_killed = True
        except ProcessLookupError:
            pass
        except PermissionError:
            # macOS can return EPERM when only the retained zombie leader is in
            # the group. It is benign only after an explicit, successful scan.
            try:
                empty = state == "exited" and not _group_has_live_members(proc.pid)
            except (OSError, _psutil().Error):
                empty = False
            if empty:
                group_skip_reason = "no_live_group_members"
            else:
                errors.append("group_kill:PermissionError")
        except OSError as exc:
            errors.append(f"group_kill:{type(exc).__name__}")
    else:
        group_skip_reason = f"direct_child_{state}"
    for identity, process in (sampler.processes.items() if sampler else ()):
        try:
            if _observed_executing(process, identity, sampler.backend):
                process.kill()
        except sampler.backend.NoSuchProcess:
            pass
        except sampler.backend.Error as exc:
            errors.append(f"descendant_kill:{type(exc).__name__}")
    if proc.poll() is None:
        try:
            proc.kill()
        except OSError as exc:
            errors.append(f"leader_kill:{type(exc).__name__}")
    try:
        proc.wait(timeout=max(0., deadline - time.monotonic()))
    except subprocess.TimeoutExpired:
        errors.append("leader_reap:TimeoutExpired")
    # A successful kill call is not evidence that an escaped descendant stopped.
    # Verify observed identities become absent/nonexecuting within the same bound.
    pending = dict(sampler.processes) if sampler else {}
    unverified = []
    while pending:
        for identity, process in list(pending.items()):
            try:
                if _observed_executing(process, identity, sampler.backend):
                    continue
            except sampler.backend.NoSuchProcess:
                pass
            except sampler.backend.Error as exc:
                errors.append(f"observed_exit:{type(exc).__name__}")
                unverified.append(identity)
            pending.pop(identity)
        remaining = deadline - time.monotonic()
        if pending and remaining <= 0:
            errors.append("observed_exit:TimeoutExpired")
            break
        if pending:
            time.sleep(min(.01, remaining))
    unverified.extend(pending)
    return {"group_killed": group_killed, "group_skip_reason": group_skip_reason,
            "leader_reaped": proc.poll() is not None,
            "observed_termination_verified": sampler is not None and not unverified,
            "unverified_observed_identities": [list(identity) for identity in unverified],
            "errors": errors,
            "scope": "owned unreaped child's private group and bounded verification that observed identities stopped executing; zombies are nonexecuting, only the direct child is reaped, and unobserved children cannot be guaranteed"}


def run_measured(argv, *, timeout, interval_seconds=.02, cwd=None, env=None,
                 stdout_path=None, stderr_path=None, on_sample=None, cleanup_timeout=5.):
    """Run one fresh process, synchronously sample it, and always clean up.

    ``on_sample(sample_dict)`` is optional and runs in the observer after each poll.
    Exceptions/KeyboardInterrupt from it propagate after process cleanup. Output
    paths are opened exclusively; environment and command arguments are not logged.
    Without paths, output is discarded to avoid unbounded in-memory pipe buffers.
    This function owns/reaps its child; other SIGCHLD handlers must not reap it.
    POSIX WNOWAIT preserves ownership until group cleanup. Where unavailable,
    cleanup uses the direct child and observed identities, never an unbound PGID.
    """
    timeout = _positive(timeout, "timeout")
    interval_seconds = _positive(interval_seconds, "interval_seconds")
    cleanup_timeout = _positive(cleanup_timeout, "cleanup_timeout")
    _psutil()  # Fail before launching anything if optional telemetry is unavailable.
    with ExitStack() as stack:
        stdout = stack.enter_context(open(stdout_path, "xb")) if stdout_path else subprocess.DEVNULL
        stderr = stack.enter_context(open(stderr_path, "xb")) if stderr_path else subprocess.DEVNULL
        started = time.monotonic()
        proc = subprocess.Popen(argv, cwd=cwd, env=env, stdout=stdout, stderr=stderr,
                                start_new_session=(os.name == "posix"))
        sampler = None
        timed_out = False
        try:
            sampler = ProcessTreeSampler(proc.pid, interval_seconds=interval_seconds)
            while True:
                sample = sampler.sample()
                if on_sample is not None:
                    on_sample(sample)
                state = _child_state(proc)
                if state in {"exited", "reaped"} or (
                        state in {"unowned", "unsupported"} and proc.poll() is not None):
                    break
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    timed_out = True
                    break
                time.sleep(min(interval_seconds, remaining))
        finally:
            # Constructor failure follows the same ownership-safe cleanup path.
            cleanup = _cleanup(proc, sampler, cleanup_timeout)
        return {"returncode": proc.returncode, "timed_out": timed_out,
                "elapsed_seconds": time.monotonic() - started,
                "memory": sampler.report(), "cleanup": cleanup}
