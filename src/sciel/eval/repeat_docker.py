"""Explicit Docker transport for repeat draws; also the trusted container PID1.

The unprivileged candidate never runs in PID1. Its watchdog survives loss of the
host/CLI, and exiting PID1 kills the private PID namespace, including setsid
descendants. This is not a claim of protection against a kernel/container escape.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
import uuid

CONTRACT = "sciel-repeat-docker/1"
PYTHON = "/usr/local/bin/python"
POLICY = {"network": "none", "cpus": 1, "pids_limit": 64,
          "rootfs_read_only": True, "candidate_uid": 10001,
          "watchdog": "root PID1; namespace exit; auto-remove",
          "memory_scope": "whole container; swap disabled",
          "cpu_scope": "whole-container one-core quota plus child RLIMIT_CPU",
          "phase_wall_scope": "create and attach; excludes cleanup and bounded-size host file I/O",
          "cleanup_cli_reap_seconds": 2, "cleanup_remove_seconds": 10,
          "identity_preflight_scope": "outside draw allocations; two Docker queries up to 10s each plus runtime probe",
          "tmpfs_bytes": 16777216}


def _environment():
    return {k: v for k, v in os.environ.items()
            if k in ("PATH", "HOME", "DOCKER_HOST", "DOCKER_CONTEXT",
                     "DOCKER_CONFIG", "LANG", "LC_ALL")}


def _cli(args, *, timeout=10):
    return subprocess.run(["docker", *args], stdin=subprocess.DEVNULL,
                          capture_output=True, timeout=timeout, env=_environment())


def image_identity(image):
    result = _cli(["image", "inspect", image])
    if result.returncode:
        raise RuntimeError("required immutable Docker image is unavailable")
    rows = json.loads(result.stdout)
    if len(rows) != 1 or rows[0]["Id"] != image or rows[0]["Os"] != "linux":
        raise RuntimeError("Docker image identity/OS differs")
    server = _cli(["info", "--format", "{{json .ServerVersion}}"])
    if server.returncode:
        raise RuntimeError("Docker server unavailable")
    return {"contract": CONTRACT, "image": image, "os": rows[0]["Os"],
            "architecture": rows[0]["Architecture"],
            "docker_server": json.loads(server.stdout), "policy": dict(POLICY)}


def _remove(name):
    result = _cli(["rm", "--force", name])
    if result.returncode and b"No such container" not in result.stderr:
        raise RuntimeError("Docker container cleanup could not be confirmed")


def _attach(name, timeout):
    """Bound stderr in memory instead of allowing candidate output to fill disk."""
    proc = subprocess.Popen(["docker", "start", "--attach", name],
                            stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            stderr=subprocess.PIPE, start_new_session=True,
                            env=_environment())
    begun, tail, timed_out = time.monotonic(), bytearray(), False
    os.set_blocking(proc.stderr.fileno(), False)
    try:
        while True:
            remaining = timeout - (time.monotonic() - begun)
            if remaining <= 0:
                timed_out = True
                break
            ready, _, _ = select.select([proc.stderr], [], [], min(.05, remaining))
            if ready:
                chunk = os.read(proc.stderr.fileno(), 65536)
                tail.extend(chunk)
                del tail[:-2000]
                if not chunk and proc.poll() is not None:
                    break
            elif proc.poll() is not None:
                break
    finally:
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        proc.wait(timeout=2)
        proc.stderr.close()
    return proc.returncode, timed_out, tail.decode("utf-8", "replace")


def process(mode, request, directory, *, timeout, seed, image, files,
            write_json, read_json):
    begun = time.monotonic()
    name = "sciel-repeat-" + uuid.uuid4().hex
    directory = Path(directory).resolve()
    translated = dict(request)
    for key in ("source_path", "packing_path"):
        if key in translated:
            original = Path(translated[key]).resolve()
            if original.parent != directory:
                raise ValueError("Docker inputs must belong to this draw directory")
            translated[key] = "/work/" + original.name
    request_path = directory / (mode + "_request.json")
    setup_path, output_path = [directory / (mode + suffix) for suffix in ("_setup.json", "_output.json")]
    write_json(request_path, translated)
    args = ["create", "--pull=never", "--rm", "--name", name,
            "--network=none", "--read-only", "--cpus=1", "--pids-limit=64",
            "--memory=" + str(request["mem_bytes"]),
            "--memory-swap=" + str(request["mem_bytes"]),
            "--cap-drop=ALL", "--cap-add=SETUID", "--cap-add=SETGID",
            "--security-opt=no-new-privileges", "--log-driver=none",
            "--tmpfs=/tmp:rw,noexec,nosuid,nodev,size=16777216,mode=1777",
            "--workdir=/work", "--entrypoint=" + PYTHON,
            "--mount", f"type=bind,src={directory},dst=/work"]
    for key in ("runner", "geometry", "docker"):
        args += ["--mount", f"type=bind,src={Path(files[key]).resolve()},dst=/opt/sciel/{Path(files[key]).name},readonly"]
    for key, target in (("package_init", "sciel/__init__.py"),
                        ("candidate_loader", "sciel/candidate_modules/__init__.py")):
        args += ["--mount", f"type=bind,src={Path(files[key]).resolve()},dst=/opt/sciel/{target},readonly"]
    args += ["--env=PYTHONHASHSEED=" + str(seed), "--env=PYTHONDONTWRITEBYTECODE=1"]
    for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        args.append("--env=" + key + "=1")
    args += [image, "-s", "-P", "-B", "/opt/sciel/" + Path(files["docker"]).name,
             "--supervise", str(timeout), "/opt/sciel/" + Path(files["runner"]).name,
             mode, "/work/" + request_path.name, "/work/" + setup_path.name,
             "/work/" + output_path.name]
    try:
        created = _cli(args, timeout=min(10, timeout))
        if created.returncode:
            raise RuntimeError("Docker create failed: " + created.stderr[-1000:].decode("utf-8", "replace"))
        remaining = timeout - (time.monotonic() - begun)
        if remaining <= 0:
            raise RuntimeError("Docker startup exhausted process wall allowance")
        code, timed_out, tail = _attach(name, remaining)
        # PID1 124 is its own deadline, which survives disappearance of the host.
        timed_out = timed_out or code == 124
    finally:
        _remove(name)
    return {"exit_code": code, "timed_out": timed_out, "stderr_tail": tail,
            "elapsed_seconds": time.monotonic() - begun,
            "setup": read_json(setup_path) if setup_path.exists() else None,
            "output": read_json(output_path) if output_path.exists() else None,
            "container": {"name": name, "image": image, "removed": True,
                          "policy": dict(POLICY)}}


def _supervise(argv):
    if os.getpid() != 1 or os.geteuid() != 0 or len(argv) != 6:
        raise RuntimeError("repeat watchdog requires root PID1 and six arguments")
    seconds, runner, mode, request, setup, output = argv
    deadline = time.monotonic() + float(seconds)
    # Only this ephemeral draw directory is writable from the host filesystem.
    os.chmod("/work", 0o777)
    env = {k: v for k, v in os.environ.items() if k in (
        "PYTHONHASHSEED", "PYTHONDONTWRITEBYTECODE", "OPENBLAS_NUM_THREADS",
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")}
    env.update(PATH="/usr/local/bin:/usr/bin:/bin", LANG="C.UTF-8", TMPDIR="/tmp")
    child = subprocess.Popen([PYTHON, "-s", "-P", "-B", runner, mode, request, setup, output],
                             stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                             stderr=sys.stderr, cwd="/work", user=10001, group=10001,
                             extra_groups=[], start_new_session=True, env=env)
    try:
        code = child.wait(timeout=max(.001, deadline - time.monotonic()))
        return code if code >= 0 else 128 - code
    except subprocess.TimeoutExpired:
        return 124


if __name__ == "__main__":
    try:
        if sys.argv[1:2] != ["--supervise"]:
            raise RuntimeError("private watchdog entry point only")
        status = _supervise(sys.argv[2:])
    except BaseException as exc:
        os.write(2, ("repeat watchdog: " + type(exc).__name__ + ": " + str(exc) + "\n").encode())
        status = 70
    # This ends the whole PID namespace, even detached children/inherited pipes.
    os._exit(status if 0 <= status <= 255 else 70)
