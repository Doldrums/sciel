"""Trusted host protocol for the opt-in hidden-oracle task.

The candidate and its client are untrusted. Only this process evaluates the
objective, owns the query count and selects the best observed point. Resource
isolation alone is insufficient: production trials require the container backend.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import selectors
import struct
import subprocess
import time


from .oracle_protocol import (
    PROTOCOL, BOOTSTRAP, QUERY_BUDGET, MAX_FRAME_BYTES, MAX_SOURCE_BYTES,
    STARTUP_TIMEOUT, TRIAL_TIMEOUT, BASE_IMAGE, ProtocolError, _unique_object,
    decode_frame, _coordinate, OracleSession, _canonical, instance_from_seed, objective_for,
)


def docker_argv(image, name, timeout=TRIAL_TIMEOUT):
    if not (image.startswith("sha256:") and len(image) == 71):
        raise ValueError("runtime requires an exact local image ID")
    if not name.startswith("sciel-oracle-"):
        raise ValueError("invalid owned container name")
    return [
        "docker",
        "run",
        "--rm",
        "-i",
        "--pull=never",
        "--name",
        name,
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--cap-add=SETUID",
        "--cap-add=SETGID",
        "--cap-add=KILL",
        "--security-opt=no-new-privileges",
        "--pids-limit=64",
        "--memory=256m",
        "--memory-swap=256m",
        "--cpus=1",
        "--tmpfs=/input:rw,nosuid,nodev,noexec,size=2m,mode=0755",
        "--tmpfs=/tmp:rw,nosuid,nodev,noexec,size=32m,mode=1777",
        image,
        str(timeout),
    ]


def _docker_env():
    # CLI needs its ordinary context, but the container inherits none of this.
    return {
        k: v
        for k, v in os.environ.items()
        if k
        in {
            "PATH",
            "HOME",
            "LANG",
            "LC_ALL",
            "TMPDIR",
            "DOCKER_CONFIG",
            "DOCKER_CONTEXT",
        }
    }


def runtime_assets_sha256():
    root = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for name in (
        "Dockerfile",
        "container_supervisor.py",
        "oracle_client.py",
        "candidate_runner.py",
    ):
        digest.update(name.encode() + b"\0" + root.joinpath(name).read_bytes() + b"\0")
    return digest.hexdigest()


def verify_runtime_image(image):
    docker_argv(image, "sciel-oracle-preflight")
    result = subprocess.run(
        ["docker", "image", "inspect", image],
        capture_output=True,
        timeout=STARTUP_TIMEOUT,
        env=_docker_env(),
        check=True,
    )
    record = json.loads(result.stdout)[0]
    if (
        record["Id"] != image
        or record["Config"].get("Labels", {}).get("sciel.oracle.assets.sha256")
        != runtime_assets_sha256()
        or record["Config"]["Entrypoint"]
        != ["python", "-I", "-B", "/opt/oracle/container_supervisor.py"]
    ):
        raise ValueError("oracle image does not match trusted task assets")
    return {
        "image_id": image,
        "assets_sha256": runtime_assets_sha256(),
        "base_image": BASE_IMAGE,
    }


def run_container_trial(
    source: bytes,
    instance,
    image: str,
    *,
    algorithm_seed,
    container_name,
    journal_path,
    timeout=TRIAL_TIMEOUT,
    startup_timeout=STARTUP_TIMEOUT,
):
    """Untrusted stdout is bounded JSON RPC, never an evaluator result.

    The container watchdog owns its deadline independently of this process. An
    outer stage SIGKILL can orphan the Docker client but cannot disable PID1's
    deadline. --rm removes the namespace after that trusted PID1 exits.
    """
    if not 0 < len(source) <= MAX_SOURCE_BYTES:
        raise ValueError("candidate size outside allowed range")
    if not 0 < timeout <= 30 or not 0 < startup_timeout <= 30:
        raise ValueError("invalid runtime deadline")
    session = OracleSession(objective_for(instance), algorithm_seed=algorithm_seed)
    session.hello()  # Validate the trusted seed before starting any process.
    name = container_name
    docker_argv(image, name, timeout)
    journal = os.fdopen(
        os.open(
            journal_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600
        ),
        "wb",
        buffering=0,
    )

    def emit(value):
        journal.write(_canonical(value) + b"\n")
        os.fsync(journal.fileno())

    emit(
        dict(
            event="trial_intent",
            container_name=name,
            image_id=image,
            source_sha256=hashlib.sha256(source).hexdigest(),
            algorithm_seed=algorithm_seed,
            bootstrap=BOOTSTRAP,
            timeout=timeout,
            startup_timeout=startup_timeout,
        )
    )
    started = time.monotonic()
    cpu_started = time.process_time()
    deadline = started + startup_timeout
    status, phase, stderr = "infrastructure_error", "startup", bytearray()
    proc, ready_at, output_bytes = None, None, 0
    selector = selectors.DefaultSelector()
    outgoing, incoming = bytearray(), bytearray()
    error = None
    cleanup = dict(
        remove_returncode=None,
        remove_error=None,
        docker_client_returncode=None,
        container_cpu_seconds=None,
        container_cpu_scope="not observed",
    )
    try:
        proc = subprocess.Popen(
            docker_argv(image, name, timeout),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=_docker_env(),
            bufsize=0,
        )
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            os.set_blocking(stream.fileno(), False)
        selector.register(proc.stdout, selectors.EVENT_READ, "out")
        selector.register(proc.stderr, selectors.EVENT_READ, "err")
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                status = "startup_timeout" if phase == "startup" else "timeout"
                break
            events = selector.select(min(remaining, 0.1))
            for key, _ in events:
                if key.data == "in":
                    try:
                        size = os.write(key.fd, outgoing)
                    except BrokenPipeError:
                        raise ProtocolError("candidate input closed")
                    del outgoing[:size]
                    if not outgoing:
                        selector.unregister(proc.stdin)
                    continue
                chunk = os.read(key.fd, 8192)
                if not chunk:
                    selector.unregister(key.fileobj)
                    continue
                output_bytes += len(chunk)
                if output_bytes > 1048576:
                    raise ProtocolError("candidate output limit exceeded")
                if key.data == "err":
                    stderr.extend(chunk[: max(0, 65536 - len(stderr))])
                    continue
                incoming.extend(chunk)
                while b"\n" in incoming:
                    frame, _, rest = incoming.partition(b"\n")
                    incoming[:] = rest
                    # Retain even an invalid or uncommitted call before handling.
                    emit(
                        dict(
                            event="received", phase=phase, frame_hex=bytes(frame).hex()
                        )
                    )
                    if phase == "startup":
                        if decode_frame(frame) != {
                            "status": "ready",
                            "protocol": PROTOCOL,
                        }:
                            raise ProtocolError("invalid trusted supervisor greeting")
                        ready_at = time.monotonic()
                        deadline, phase = ready_at + timeout, "trial"
                        outgoing.extend(
                            struct.pack("!I", len(source))
                            + source
                            + _canonical(session.hello())
                            + b"\n"
                        )
                    else:
                        response = session.handle(frame)
                        emit(dict(event="response", payload=response))
                        outgoing.extend(_canonical(response) + b"\n")
                    if outgoing and not any(
                        k.data == "in" for k in selector.get_map().values()
                    ):
                        selector.register(proc.stdin, selectors.EVENT_WRITE, "in")
                    if session.finished:
                        status = "ok"
                        break
                if len(incoming) > MAX_FRAME_BYTES:
                    raise ProtocolError("candidate frame limit exceeded")
                if session.finished:
                    break
            if session.finished:
                break
            if proc.poll() is not None and not events:
                status = (
                    "infrastructure_error" if phase == "startup" else "candidate_error"
                )
                if proc.returncode == 124 and phase != "startup":
                    status = "timeout"
                elif proc.returncode == 125:
                    status = "infrastructure_error"
                break
    except ProtocolError as exc:
        status = (
            "infrastructure_error"
            if phase == "startup" or session.objective_failed
            else "protocol_error"
        )
        error = str(exc)
    except (OSError, subprocess.SubprocessError) as exc:
        status, error = "infrastructure_error", type(exc).__name__
    finally:
        selector.close()
        if proc is not None:
            try:
                removed = subprocess.run(
                    ["docker", "rm", "-f", name],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                    env=_docker_env(),
                )
                cleanup["remove_returncode"] = removed.returncode
            except (OSError, subprocess.TimeoutExpired) as exc:
                cleanup["remove_error"] = type(exc).__name__
            if proc.poll() is None:
                proc.kill()
            try:
                proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            cleanup["docker_client_returncode"] = proc.returncode
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                stream.close()
        try:
            emit(dict(event="trial_terminal", status=status, cleanup=cleanup))
        finally:
            journal.close()
    return {
        "schema": "sciel-fixed-oracle-trial/1",
        "status": status,
        "score": 1 / (1 + session.best_value) if status == "ok" else 0.0,
        "queries": session.queries,
        "query_budget": QUERY_BUDGET,
        "best_point": list(session.best_point)
        if session.best_point is not None
        else None,
        "best_value": session.best_value,
        "source_sha256": hashlib.sha256(source).hexdigest(),
        "image_id": image,
        "container_name": name,
        "elapsed_seconds": time.monotonic() - started,
        "startup_seconds": None if ready_at is None else ready_at - started,
        "broker_cpu_seconds": time.process_time() - cpu_started,
        "algorithm_seed": algorithm_seed,
        "bootstrap": BOOTSTRAP,
        "bootstrap_initialized": session.initialized,
        "finished": session.finished,
        "objective_failed": session.objective_failed,
        "cleanup": cleanup,
        "journal_sha256": hashlib.sha256(Path(journal_path).read_bytes()).hexdigest(),
        "observations": session.transcript,
        "error": error,
        "stderr": stderr.decode("utf-8", "replace"),
    }
