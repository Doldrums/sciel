"""External live telemetry for one bounded methods CLI invocation.

This is an observer, not a budget guard. It never changes scientific artifacts,
retries a dispatch, stores environment/command contents, or counts old receipts
as new work on resume. Container and controller CPU are disjoint observations;
neither may be added to the resource ledger's overlapping CPU total.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import signal
import socket
import stat
import subprocess
import sys
import tempfile
import time
import uuid

from sciel.engine.runtime import process_identity, process_matches

SCHEMA = "sciel-method-runtime/1"


def _now():
    return datetime.now(timezone.utc).isoformat()


def _atomic(path, value):
    data = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False).encode() + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".observer-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _read(path):
    try:
        if path.is_symlink() or path.stat().st_size > 32 * 1024 * 1024:
            return None
        value = json.loads(path.read_bytes())
        return value if isinstance(value, dict) else None
    except (OSError, ValueError):
        return None


def _report_snapshot(path):
    """Read a bounded report and identify this particular file occurrence.

    Content alone cannot distinguish an old report from an identical report
    freshly written by the current worker. Read both through the same fd;
    changing/nonregular files cannot certify a terminal result.
    """
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, 'rb') as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode) or before.st_size > 32 * 1024 * 1024:
                return None, None
            raw = stream.read(32 * 1024 * 1024 + 1)
            after = os.fstat(stream.fileno())
        def version(info):
            return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)
        if len(raw) > 32 * 1024 * 1024 or version(before) != version(after):
            return None, None
        value = json.loads(raw)
        if not isinstance(value, dict):
            return None, None
        return value, (*version(after), hashlib.sha256(raw).hexdigest())
    except (OSError, ValueError, RecursionError):
        return None, None


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=True, allow_nan=False).encode()).hexdigest()


def _number(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def logical_size(root):
    """Current regular-file length, not allocated blocks or cumulative writes."""
    total, inaccessible = 0, 0
    for directory, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [name for name in dirs if not (Path(directory) / name).is_symlink()]
        for name in files:
            try:
                value = (Path(directory) / name).lstat()
                if stat.S_ISREG(value.st_mode):
                    total += value.st_size
            except OSError:
                inaccessible += 1
    return total, inaccessible


def _receipt_snapshot(output):
    """Read immutable dispatch receipts once per unique invocation, not artifacts."""
    calls, invalid = {}, 0
    for path in (output / "registry" / "dispatch").glob("*/receipt.json"):
        value = _read(path)
        if value is None or value.get("sha256") != _digest({k: v for k, v in value.items() if k != "sha256"}):
            invalid += 1
            continue
        if value.get("category") not in ("execution", "replication"):
            continue
        result = value.get("result", {})
        if not isinstance(result, dict):
            invalid += 1
            continue
        # Exact receipt key remains stable when an old result is reused.
        calls[path.parent.name] = value
    return calls, invalid


def _ledger_snapshot(output):
    """Latest cumulative settlement per reservation, with no ledger mutation."""
    rows, errors, previous = {}, 0, "0" * 64
    path = output / "resources.jsonl"
    try:
        with path.open() as stream:
            for line in stream:
                value = json.loads(line)
                digest = value.get("sha256")
                if (digest != _digest({k: v for k, v in value.items() if k != "sha256"})
                        or value.get("prev_sha256") != previous):
                    errors += 1
                    break
                previous = digest
                data = value["data"]
                if value["op"] == "reserve":
                    rows[data["id"]] = dict(category=data["category"], status="reserved", cpu=0.)
                elif value["op"] in ("settle", "resolve") and data["id"] in rows:
                    cpu = float(data.get("actual", {}).get("cpu_seconds", 0))
                    if not _number(cpu):
                        errors += 1
                        break
                    rows[data["id"]].update(status=data["status"], cpu=cpu)
    except FileNotFoundError:
        pass
    except (OSError, ValueError, KeyError, TypeError):
        errors += 1
    return rows, errors


def ledger_cpu(output, baseline):
    rows, errors = _ledger_snapshot(output)
    rows = [value for key, value in rows.items()
            if key not in baseline and value["category"] != "upfront_creation"]
    return dict(ledger_cpu_seconds_this_invocation=sum(value["cpu"] for value in rows),
                ledger_current_unresolved=sum(value["status"] not in ("completed", "failed") for value in rows),
                ledger_read_errors=errors, ledger_available=(output / "resources.jsonl").is_file())


def container_cpu(output, baseline):
    calls, invalid = _receipt_snapshot(output)
    current = {key: value for key, value in calls.items() if key not in baseline}
    known, invocations, unknown = 0.0, set(), 0
    for value in current.values():
        result = value["result"]
        status = result.get("dispatch_status")
        if status == "not_dispatched":
            continue
        cpu = result.get("resources", {}).get("cpu_seconds")
        identity = result.get("invocation_id")
        measurement = result.get("resource_measurement", {}).get("cpu")
        if status != "invoked" or not isinstance(identity, str) or not _number(cpu) or measurement != "trusted_cgroup_observation":
            unknown += 1
            continue
        if identity not in invocations:
            known += cpu
            invocations.add(identity)
    # A started call without receipt is unresolved, including while in flight.
    pending = 0
    for path in (output / "registry" / "dispatch").glob("*/intent.json"):
        if path.parent.name in baseline or path.parent.name in calls:
            continue
        intent = _read(path)
        if intent and intent.get("category") in ("execution", "replication"):
            pending += 1
    return dict(container_cpu_seconds=round(known, 9), container_invocations=len(invocations),
                container_unknown_receipts=unknown, container_pending_calls=pending,
                invalid_receipts=invalid, container_cpu_complete=not (unknown or pending or invalid))


def _registry_binding(output):
    path = output / "registry" / "journal.jsonl"
    try:
        with path.open() as stream:
            for line in stream:
                value = json.loads(line)
                if value.get("kind") == "coordinator_identity":
                    data = value["data"]
                    return dict(registry_started_at=data.get("started_at"),
                                registry_identity_sha256=_digest(data["identity"]))
    except (OSError, ValueError, KeyError):
        pass
    return {}


def _sample_cpu(record):
    if process_matches(record) is not True:
        return None
    try:
        result = subprocess.run(["ps", "-o", "cputime=", "-p", str(record["pid"])],
                                capture_output=True, text=True, timeout=1, check=False)
        token = result.stdout.strip()
        days = 0
        if "-" in token:
            day, token = token.split("-", 1)
            days = int(day)
        values = [float(part) for part in token.split(":")]
        if result.returncode or not 1 <= len(values) <= 3 or any(not _number(n) for n in values):
            return None
        value = days * 86400 + sum(part * 60 ** index for index, part in enumerate(reversed(values)))
        return value if process_matches(record) is True else None
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def _terminal_status(exit_code, worker, report):
    if exit_code < 0 or worker is None:
        return "unknown", "controller_terminated_without_worker_terminal_receipt"
    if report and report.get("status") == "budget_exhausted":
        return "stopped", "budget_exhausted"
    if report and report.get("status") == "accounting_stopped":
        return "unknown", "accounting_stopped"
    if report and report.get('status') == 'stopped' and exit_code == 0:
        return 'stopped', report.get('reason') or 'operator_requested_stop'
    if exit_code != 0:
        return "failed", (report or {}).get("status", "controller_nonzero_exit")
    if report and report.get("status") in ("completed", "passed"):
        return "completed", None
    return "unknown", "missing_or_inconsistent_run_report"


def observe(command, output, *, interval=1.0, resume=False, worker_terminal=None):
    """Launch exactly once. ``command`` stays in memory and is never persisted."""
    output = Path(output).absolute()
    if output.is_symlink() or not math.isfinite(interval) or not .05 <= interval <= 10:
        raise ValueError("invalid observer directory or interval")
    output.mkdir(parents=True, exist_ok=True)
    fd = os.open(output / ".observer.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        old = _read(output / "runtime.json")
        if old and process_matches(old) is not False:
            raise ValueError("previous controller is live or its identity cannot be disproved")
        if not resume and (old or (output / "run-config.json").exists() or (output / "registry").exists()):
            raise ValueError("existing run requires explicit --resume; no implicit replay")
        baseline, _ = _receipt_snapshot(output)
        baseline_ids = set(baseline)
        # Missing old receipts stay historical on resume; recovering an old
        # result is not a new container invocation in this observation session.
        for path in (output / "registry" / "dispatch").glob("*/intent.json"):
            value = _read(path)
            if value and value.get("category") in ("execution", "replication"):
                baseline_ids.add(path.parent.name)
        baseline_ledger, baseline_ledger_errors = _ledger_snapshot(output)
        size_before, _ = logical_size(output)
        _, report_before = _report_snapshot(output / "run-report.json")
        runtime_id = uuid.uuid4().hex
        observer_dir = output / "observation" / runtime_id
        observer_dir.mkdir(parents=True)
        worker_terminal = Path(worker_terminal) if worker_terminal else observer_dir / "worker-terminal.json"
        start = time.monotonic()
        state = dict(schema=SCHEMA, runtime_id=runtime_id, output_name=output.name,
                     hostname=socket.gethostname(), observer_pid=os.getpid(),
                     observer_process_identity=process_identity(os.getpid()), status="running", stage="starting",
                     started_at=_now(), heartbeat_at=None, finished_at=None, reason=None,
                     heartbeat_interval_seconds=interval, stale_after_seconds=max(10, interval * 4),
                     resumed=resume, prior_container_receipts=len(baseline_ids))
        launch = dict(schema="sciel-method-observer-launch/1", runtime_id=runtime_id,
                      started_at=state["started_at"], prior_container_receipt_ids=sorted(baseline_ids),
                      prior_reservation_ids=sorted(baseline_ledger),
                      prior_ledger_read_errors=baseline_ledger_errors,
                      output_logical_bytes_before=size_before, resumed=resume)
        _atomic(observer_dir / "launch.json", launch)
        highest_cpu, exit_code = None, None
        with (observer_dir / "controller.log").open("ab", buffering=0) as log:
            # Only our direct child receives the inherited credential-bearing env.
            # No command, env, or provider exception is copied into runtime.json.
            child = subprocess.Popen(command, stdout=log, stderr=log, start_new_session=True)
            state.update(pid=child.pid, process_identity=process_identity(child.pid), stage="research")
            stop_signal = [None]
            previous_handlers = {}

            def stop(signum, frame):
                stop_signal[0] = signum
                if child.returncode is None and process_matches(state) is True:
                    os.kill(child.pid, signum)

            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.signal(signum, stop)
            try:
                while True:
                    sampled = _sample_cpu(state)
                    if sampled is not None:
                        highest_cpu = max(highest_cpu or 0, sampled)
                    exit_code = child.poll()
                    worker = _read(worker_terminal) if exit_code is not None else None
                    if worker and (worker.get("schema") != "sciel-method-worker-terminal/1"
                                   or worker.get("pid") != child.pid or worker.get("exit_code") != exit_code
                                   or not _number(worker.get("controller_cpu_seconds"))):
                        worker = None
                    cpu_source = "live_ps_sample_lower_bound"
                    if worker:
                        highest_cpu = round(worker["controller_cpu_seconds"], 9)
                        cpu_source = "worker_rusage_terminal"
                    containers = container_cpu(output, baseline_ids)
                    ledger = ledger_cpu(output, baseline_ledger)
                    size, errors = logical_size(output)
                    state.update(_registry_binding(output))
                    state["heartbeat_at"] = _now()
                    state.update(stop_requested=stop_signal[0] is not None,
                                 stop_signal=stop_signal[0])
                    state["overhead"] = dict(
                        scope="This observer invocation only; resumed historical container receipts excluded.",
                        controller_cpu_seconds=highest_cpu, controller_cpu_measurement=cpu_source,
                        wall_seconds=round(time.monotonic() - start, 6),
                        output_logical_bytes=size, output_logical_bytes_before=size_before,
                        output_logical_bytes_delta=size - size_before, output_inaccessible_files=errors,
                        **containers,
                        **ledger,
                        combined_observed_cpu_seconds=(round(highest_cpu + containers["container_cpu_seconds"], 9)
                                                       if highest_cpu is not None else None),
                        combined_is_lower_bound=True,
                        budget_enforced=False, additive_to_ledger=False,
                        exclusions=["Docker daemon and VM CPU", "local subprocess CPU outside the controller",
                                    "observer CPU", "filesystem metadata and allocated blocks",
                                    "interpreter shutdown after worker terminal receipt"],
                        note="Controller CPU includes callback CPU already charged by the ledger. Never add these totals to ledger CPU. Output bytes are current footprint, not cumulative writes.")
                    # This subtraction is meaningful only at a clean terminal
                    # boundary with matching new reservation/invocation scope.
                    estimate = None
                    metered_host = None
                    if (exit_code is not None and worker and highest_cpu is not None
                            and containers["container_cpu_complete"]
                            and ledger["ledger_available"] and not ledger["ledger_current_unresolved"]
                            and not ledger["ledger_read_errors"] and not baseline_ledger_errors):
                        metered_host = ledger["ledger_cpu_seconds_this_invocation"] - containers["container_cpu_seconds"]
                        if metered_host >= -1e-6:
                            metered_host = max(0., metered_host)
                            estimate = highest_cpu - metered_host
                    state["overhead"].update(metered_controller_cpu_seconds=metered_host,
                        unmetered_controller_cpu_seconds_estimate=estimate,
                        overhead_estimate_note="Terminal controller self CPU minus matching new ledger CPU after subtracting cgroup CPU; upfront allocations and old reservations excluded. A negative estimate indicates measurement inconsistency, not a credit.")
                    if exit_code is not None:
                        report, report_version = _report_snapshot(output / "run-report.json")
                        # A current worker receipt proves its exit/CPU, not that
                        # it replaced the preceding invocation's run report.
                        if report_version == report_before:
                            report = None
                        status, reason = _terminal_status(exit_code, worker, report)
                        state.update(status=status, reason=reason, stage="finished", exit_code=exit_code,
                                     finished_at=_now(), controller_measurement_terminal=worker is not None)
                    _atomic(output / "runtime.json", state)
                    if exit_code is not None:
                        _atomic(observer_dir / "terminal.json", state)
                        return state
                    time.sleep(interval)
            finally:
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)
                # Unexpected observer failure must not rerun or silently kill an
                # in-flight paid request. A stale sidecar remains unverifiable.
    finally:
        os.close(fd)


def _worker(config, output, terminal):
    error_type, code = None, 2
    try:
        from sciel.methods.__main__ import main
        code = main(["run", "--config", str(config), "--output", str(output)])
    except BaseException as exc:
        error_type = type(exc).__name__
        # Do not print provider exception reprs or credential-bearing context.
        code = 130 if isinstance(exc, KeyboardInterrupt) else 2
    finally:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        _atomic(Path(terminal), dict(schema="sciel-method-worker-terminal/1", pid=os.getpid(),
                finished_at=_now(), controller_cpu_seconds=usage.ru_utime + usage.ru_stime,
                exit_code=code, error_type=error_type))
    return code


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--worker-terminal", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.worker_terminal:
        return _worker(args.config, args.output, args.worker_terminal)
    terminal = args.output.absolute() / "observation" / ("worker-" + uuid.uuid4().hex + ".json")
    command = [sys.executable, "-m", "sciel.diagnostics.methods_observer", "--config", str(args.config.absolute()),
               "--output", str(args.output.absolute()), "--worker-terminal", str(terminal)]
    try:
        state = observe(command, args.output, interval=args.interval, resume=args.resume, worker_terminal=terminal)
    except (ValueError, OSError) as exc:
        print(json.dumps(dict(status="refused", error_type=type(exc).__name__)))
        return 2
    print(json.dumps(dict(status=state["status"], runtime_id=state["runtime_id"])))
    return 0 if state["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
