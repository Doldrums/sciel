"""Best-effort process telemetry, separate from scientific journal/checkpoint state.

This module uses only the standard library. A sidecar is never evidence that its
writer is still alive: readers must check freshness, log identity and the process
start token, not merely an extant or recently modified file.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from functools import wraps
import json
import logging
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import threading
import uuid


SCHEMA = "sciel-runtime/1"
_log = logging.getLogger("sciel.runtime")


def _now():
    return datetime.now(timezone.utc).isoformat()


def process_identity(pid):
    """Return an OS process-start token, or None if it cannot be established."""
    if type(pid) is not int or pid <= 0:
        return None
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
        fields = stat.rsplit(")", 1)[1].split()
        if fields[0] in {"Z", "X"}:
            return None
        return {"kind": "proc_start_ticks", "value": fields[19]}
    except (OSError, IndexError):
        pass
    try:
        result = subprocess.run(["ps", "-o", "lstart=", "-o", "stat=", "-p", str(pid)],
                                capture_output=True, text=True, timeout=1, check=False)
        fields = result.stdout.split()
        if result.returncode == 0 and len(fields) >= 6 and not fields[-1].startswith(("Z", "X")):
            return {"kind": "ps_lstart", "value": " ".join(fields[:-1])}
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def process_matches(record):
    """True=the same live local process; False=dead/different; None=unverified."""
    if not isinstance(record, dict) or record.get("hostname") != socket.gethostname():
        return None
    pid, expected = record.get("pid"), record.get("process_identity")
    if type(pid) is not int or pid <= 0:
        return None
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return None
    except OSError:
        return None
    if not isinstance(expected, dict) or not expected.get("value"):
        return None
    actual = process_identity(pid)
    return actual == expected if actual is not None else None


class RuntimeStatus:
    """Atomic heartbeat publisher; telemetry failures never stop a search."""

    def __init__(self, directory, metadata, *, interval=1.0):
        self.path = Path(directory) / "runtime.json"
        self.metadata = metadata
        self.interval = interval
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread = None
        self._warned = False
        self._state = dict(schema=SCHEMA, runtime_id=str(uuid.uuid4()), pid=os.getpid(),
            hostname=socket.gethostname(), process_identity=process_identity(os.getpid()),
            status="running", stage="starting", started_at=_now(), heartbeat_at=None,
            finished_at=None, heartbeat_interval_seconds=interval, stale_after_seconds=10,
            reason=None, error_type=None)

    def _publish(self):
        temporary = None
        try:
            with self._lock:
                self._state["heartbeat_at"] = _now()
                payload = dict(self._state, **self.metadata())
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8",
                        prefix=".runtime-", suffix=".tmp", dir=self.path.parent,
                        delete=False) as stream:
                    temporary = stream.name
                    json.dump(payload, stream, allow_nan=False, separators=(",", ":"))
                    stream.write("\n")
                os.replace(temporary, self.path)
                temporary = None
        except Exception as exc:
            if not self._warned:
                _log.warning("Runtime telemetry unavailable (%s); search continues.", type(exc).__name__)
                self._warned = True
        finally:
            if temporary is not None:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass

    def _heartbeat(self):
        while not self._stop.wait(self.interval):
            self._publish()

    def start(self):
        self._publish()
        self._thread = threading.Thread(target=self._heartbeat,
                                        name="sciel-runtime-heartbeat", daemon=True)
        try:
            self._thread.start()
        except RuntimeError:
            self._thread = None
            _log.warning("Runtime heartbeat could not start; search continues.")

    @contextmanager
    def phase(self, stage):
        with self._lock:
            previous = self._state["stage"]
            self._state["stage"] = stage
        self._publish()
        try:
            yield
        finally:
            with self._lock:
                self._state["stage"] = previous
            self._publish()

    def finish(self, status, *, reason=None, error_type=None):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        with self._lock:
            self._state.update(status=status, stage="finished", finished_at=_now(),
                               reason=reason, error_type=error_type)
        self._publish()


def observed_stage(stage):
    """Observe a method only while Engine.run owns an active runtime publisher."""
    def decorate(method):
        @wraps(method)
        def wrapped(self, *args, **kwargs):
            runtime = getattr(self, "_runtime_status", None)
            if runtime is None:
                return method(self, *args, **kwargs)
            with runtime.phase(stage):
                return method(self, *args, **kwargs)
        return wrapped
    return decorate
