"""Native subprocess transport with deadlines, not a hostile-code sandbox.

The task evaluator may import the candidate in its own interpreter under the
host user's identity. Process cleanup and bounded JSON do not authenticate a
score or isolate filesystem/network access.
"""

from __future__ import annotations

import json
import math
import hashlib
import os
import re
import signal
import stat
import subprocess
import sys
import tempfile
import threading

from .facade import (EvalResult, MAX_RESULT_BYTES, MAX_STDERR_BYTES,
                     EvaluationSetupError, SETUP_FAILURE_SCHEMA, setup_failure)

_RUNNER = os.path.join(os.path.dirname(__file__), "_runner.py")
_STDERR_BYTES = 8192


class ResultTooLarge(ValueError):
    pass


def _read_result(path):
    """Strict bounded parent-side wire decoding, not score authentication."""
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    try:
        info=os.fstat(fd)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError('evaluator result is not a regular file')
        if info.st_size>MAX_RESULT_BYTES:
            raise ResultTooLarge(f'evaluator result exceeds {MAX_RESULT_BYTES} bytes')
        with os.fdopen(fd,'rb',closefd=False) as stream:
            # Reading the full cap allocates several MiB even for a tiny result.
            # One extra byte detects growth without reserving the entire cap.
            raw=stream.read(info.st_size+1)
        if len(raw) != info.st_size:
            raise ValueError('evaluator result size changed while reading')
        def pairs(items):
            result={}
            for key,value in items:
                if key in result:
                    raise ValueError('duplicate evaluator result key')
                result[key]=value
            return result
        payload=json.loads(raw,object_pairs_hook=pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(ValueError('nonfinite evaluator JSON')))
        if type(payload) is dict and payload.get("schema") == SETUP_FAILURE_SCHEMA:
            try:
                failure = EvaluationSetupError(payload)
            except (ValueError, TypeError, OverflowError):
                failure = EvaluationSetupError(setup_failure("malformed_report", "unknown"))
            raise failure
        return EvalResult.from_json(payload)
    finally:
        os.close(fd)


class _StderrTail:
    """Drain a pipe continuously, retaining only bounded repair diagnostics."""

    def __init__(self, stream):
        self.stream = stream
        self.buffer = bytearray()
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._drain, daemon=True,
                                       name="sciel-evaluator-stderr")
        self.thread.start()

    def _drain(self):
        try:
            while chunk := self.stream.read(_STDERR_BYTES):
                with self.lock:
                    self.buffer.extend(chunk)
                    del self.buffer[:-_STDERR_BYTES]
        except (OSError, ValueError):
            pass
        finally:
            self.stream.close()

    def finish(self):
        # The process group has already been killed/reaped. A detached process
        # that escaped that group must not keep the engine waiting on its pipe.
        self.thread.join(timeout=1)
        with self.lock:
            return bytes(self.buffer).decode("utf-8", "replace")[-2000:]


def _evaluation_env():
    """Remove known ambient credential channels and automatic execution hooks.

    This compatibility filter is NOT a secret-free allowlist or filesystem /
    network isolation. Unknown task variables, HOME, local dependency paths and
    Docker configuration remain available to legacy evaluators. A native
    candidate still has the same user's access to their underlying files.
    """
    sensitive = {"TOKEN", "TOKENS", "JWT", "PASSWORD", "PASSWD", "SECRET",
                 "CREDENTIAL", "CREDENTIALS", "AUTH", "AUTHORIZATION",
                 "AUTHFILE", "PRIVATEKEY"}
    withheld = {
        "SSH_AUTH_SOCK", "GOOGLE_APPLICATION_CREDENTIALS", "AWS_ACCESS_KEY_ID",
        "KUBECONFIG", "KRB5CCNAME", "NETRC", "GIT_ASKPASS", "SSH_ASKPASS",
        "PYTHONSTARTUP", "PYTHONINSPECT", "PYTHONBREAKPOINT",
        "LD_PRELOAD", "LD_AUDIT", "DYLD_INSERT_LIBRARIES",
    }
    def allowed(key):
        upper = key.upper()
        words = set(re.split(r"[^A-Z0-9]+", upper))
        return not (words & sensitive or upper.endswith(("API_KEY", "ACCESS_KEY"))
                    or {"PRIVATE", "KEY"} <= words or upper in withheld)
    return {key: value for key, value in os.environ.items() if allowed(key)}


def _kill_group(proc) -> None:
    """SIGKILL the whole group, then reap. Best effort: the group may already be gone."""
    if os.name != "posix":
        if proc.poll() is None:
            proc.kill()
        return
    try:
        # setsid() makes the group ID equal to this PID. The leader may already
        # have exited and been reaped while its descendants are still running.
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except OSError:
            pass


def _exit_failure(returncode, path, tail):
    """Keep a saved payload as bounded diagnostics, never successful evidence."""
    message = f"Evaluator process exited with nonzero status {returncode}."
    artifacts = {"error_type": "EvaluatorProcessExit", "error": message,
                 "returncode": str(returncode)}
    saved_stderr = ""
    timing = cost = 0.0
    try:
        saved = _read_result(path)
    except (ValueError, TypeError, OSError, RecursionError, OverflowError) as exc:
        artifacts["saved_result_error"] = str(exc)[:1000]
    else:
        # The strict reader has already bounded and validated this object.
        # Retain its timings/costs without retaining its score as an outcome.
        timing, cost, saved_stderr = saved.timing, saved.cost, saved.stderr
        raw = json.dumps(saved.to_json(), ensure_ascii=False, separators=(",", ":"),
                         allow_nan=False).encode("utf-8")
        artifacts["saved_result_sha256"] = hashlib.sha256(raw).hexdigest()
        artifacts["saved_result_bytes"] = str(len(raw))
        if len(raw) <= 8192:
            artifacts["saved_result_json"] = raw.decode("utf-8")
        else:
            artifacts["saved_result_excerpt"] = raw[:8192].decode("utf-8", "ignore")
            artifacts["saved_result_truncated"] = "true"
    diagnostic = "\n".join(part for part in (message, saved_stderr, tail) if part)
    diagnostic = diagnostic.encode("utf-8")[-MAX_STDERR_BYTES:].decode("utf-8", "ignore")
    return EvalResult(ok=False, score=0.0, timing=timing, cost=cost,
                      artifacts=artifacts, stderr=diagnostic)


def validate_stage_limits(timeout, n=None, cpu_seconds=None, mem_bytes=None):
    """Reject malformed limits before allocating files or starting a process."""
    try:
        valid = type(timeout) in (int, float) and math.isfinite(timeout) and timeout > 0
    except OverflowError:
        valid = False
    if not valid:
        raise ValueError("timeout must be a positive finite number")
    for name, value in (("n", n), ("cpu_seconds", cpu_seconds), ("mem_bytes", mem_bytes)):
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError(name + " must be a positive integer or None")
        if value is not None and name != "n":
            # Reserve the CPU hard-limit increment and exclude a positive
            # RLIM_INFINITY marker on signed-size platforms such as macOS.
            hard_limit = value + 1 if name == "cpu_seconds" else value
            if hard_limit >= sys.maxsize:
                raise ValueError(name + " exceeds the portable finite resource range")


def run_stage(
    evaluator_path: str,
    stage: str,
    program_path: str,
    timeout: float,
    n: int | None = None,
    cpu_seconds: int | None = None,
    mem_bytes: int | None = None,
    python: str | None = None,
    seed: int | None = None,
    population: str | None = None,
) -> EvalResult:
    validate_stage_limits(timeout, n, cpu_seconds, mem_bytes)
    if os.name != "posix" and (cpu_seconds is not None or mem_bytes is not None):
        raise EvaluationSetupError(setup_failure("unsupported_platform", "platform",
                                                cpu_seconds, mem_bytes))
    fd, out_path = tempfile.mkstemp(suffix=".json", prefix="sciel_eval_")
    os.close(fd)
    argv = [python or sys.executable, _RUNNER, evaluator_path, stage,
            program_path, out_path]
    if n is not None:
        argv.append(str(n))
    if cpu_seconds is not None:
        argv.extend(["--cpu-seconds", str(cpu_seconds)])
    if mem_bytes is not None:
        argv.extend(["--mem-bytes", str(mem_bytes)])
    env = _evaluation_env()
    if seed is not None:
        argv.extend(["--seed", str(seed)])
        env["PYTHONHASHSEED"] = str(seed % 2 ** 32)
    if population is not None:
        env["SCIEL_POPULATION"] = population
    # A program can spend its stage: smoke checks it quickly, full and each confirm repeat may search.
    env["SCIEL_STAGE_SECONDS"] = repr(timeout / (n or 1))
    env["SCIEL_TASK_ROOT"] = os.path.dirname(os.path.abspath(evaluator_path))
    proc = stderr = None
    try:
        proc = subprocess.Popen(
            argv,
            # Task inputs are explicit arguments/files, never the engine's
            # terminal/pipe or other open host descriptors.
            stdin=subprocess.DEVNULL,
            close_fds=True,
            # Scores travel through out_path; candidate stdout is not consumed.
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            bufsize=0,
            env=env,
            # No Python preexec_fn: run_stage is also called from thread pools.
            # The runner applies rlimits after exec, before importing task code.
            start_new_session=(os.name == "posix"),
        )
        stderr = _StderrTail(proc.stderr)
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            _kill_group(proc)                     # the group, not just the child
            proc.wait(timeout=5)
            raise
        _kill_group(proc)                         # stop descendants holding stderr
        tail = stderr.finish()
        if proc.returncode != 0:
            return _exit_failure(proc.returncode, out_path, tail)
        if not os.path.lexists(out_path):
            return EvalResult(ok=False, score=0.0,
                              stderr=f"no result (rc={proc.returncode}): {tail}",
                              artifacts={"stderr": tail})
        try:
            result = _read_result(out_path)
        except (ValueError, TypeError, OSError, RecursionError, OverflowError) as e:
            # The runner writes incrementally, so a crash mid-dump leaves a file that
            # is non-empty and unparseable. Unguarded, this killed the whole run.
            return EvalResult(ok=False, score=0.0,
                              stderr="\n".join(part for part in
                                  (f"unreadable result: {str(e)[:1000]}", tail) if part),
                              artifacts={"error_type": "ResultTooLarge" if isinstance(e,ResultTooLarge) else "ResultCorrupt",
                                         "error": f"The evaluator terminal result violates its bounded JSON contract: {str(e)[:1000]}",
                                         "stderr": tail})
        if tail:
            result.stderr = "\n".join(part for part in (result.stderr, tail) if part).encode('utf-8')[-MAX_STDERR_BYTES:].decode('utf-8','ignore')
        return result
    except EvaluationSetupError as exc:
        if exc.diagnostic["requested_limits"] != {"cpu_seconds": cpu_seconds, "mem_bytes": mem_bytes}:
            raise EvaluationSetupError(setup_failure("request_mismatch", "unknown",
                cpu_seconds, mem_bytes, exc.diagnostic["reported_wall_seconds"])) from None
        raise
    except subprocess.TimeoutExpired:
        # `error` is what reaches the error_repair prompt (snapshot.artifacts);
        # leaving it empty starved the LLM of the one signal that explains ~75%
        # of failed evals (trace audit 2026-08-14)
        return EvalResult(ok=False, score=0.0,
                          stderr=f"timeout after {timeout}s",
                          artifacts={"error_type": "TimeoutError",
                                     "error": (
                                         f"TimeoutError: the program exceeded the {timeout:.0f}s "
                                         "wall-clock limit and was killed before producing any "
                                         "result. Reduce iterations/restarts or wrap the search "
                                         "in an explicit deadline guard (track elapsed time and "
                                         "return the best result found before the limit).")})
    finally:
        if proc is not None:
            _kill_group(proc)                     # never leave the group running
            try:
                proc.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError, ValueError):
                pass
        if stderr is not None:
            stderr.finish()
        elif proc is not None and proc.stderr is not None:
            proc.stderr.close()
        try:
            os.unlink(out_path)
        except OSError:
            pass
