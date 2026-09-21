"""Fresh seeded Circle draws for opt-in repeat promotion.

The caller must persist its draw intent before calling ``draw`` and bind the
returned identity to that intent. This adapter never retries or chooses seeds.
Python, NumPy's legacy RNG and Python hash seed are controlled; a candidate can
reseed or use time/default_rng/OS entropy, so deterministic output is not promised.
Resource limits/process-group cleanup are not a hostile-code security sandbox.
"""
from __future__ import annotations

from datetime import datetime, timezone
import ast
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import stat
import subprocess
import sys
import tempfile
import time
import uuid

from . import packing_geometry as geometry
from . import repeat_docker
from .repeat_io import (RepeatInfrastructureError as RepeatInfrastructureError,
                        _read_json, _write_json)
from ..hypothesis import restart_optimizer_trace as trace_codec
from ..hypothesis import restart_trace as legacy_trace_codec

_DIRECTORY = Path(__file__).resolve().parent
_FILES = {"adapter": _DIRECTORY / "repeat_draw.py", "runner": _DIRECTORY / "_repeat_runner.py",
          "transport": _DIRECTORY / "repeat_io.py",
          "geometry": _DIRECTORY / "packing_geometry.py", "docker": _DIRECTORY / "repeat_docker.py",
          "package_init": _DIRECTORY.parent / "__init__.py",
          "candidate_loader": _DIRECTORY.parent / "candidate_modules/__init__.py",
          "restart_trace": Path(legacy_trace_codec.__file__),
          "restart_optimizer_trace": Path(trace_codec.__file__)}
_LOADED_SHA = {key: hashlib.sha256(_FILES[key].read_bytes()).hexdigest()
               for key in ("adapter", "transport", "geometry", "docker", "restart_trace", "restart_optimizer_trace")}
_MAX_TRACE_BYTES = 96 * 1024
_TRACE_SCHEMA = "sciel-repeat-restart-trace/1"
_THREAD_LIMITS = {name: "1" for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")}
_ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "TMPDIR", "SystemRoot")
_PROBE_SECONDS = 10.0
_PROBE_MEMORY = 4 * 1024**3


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _now():
    return datetime.now(timezone.utc).isoformat()


def _python(python):
    chosen = os.fspath(python) if python is not None else sys.executable
    found = shutil.which(chosen)
    if found is None:
        raise RepeatInfrastructureError("Python executable not found")
    # Do not resolve venv symlinks to the base interpreter.
    return os.path.abspath(found)


def _environment(seed):
    return {**{name: os.environ[name] for name in _ENV_ALLOWLIST if name in os.environ},
            **_THREAD_LIMITS, "PYTHONHASHSEED": str(seed), "PYTHONDONTWRITEBYTECODE": "1"}


def _trace_scaffold(source):
    """Read the source's explicit trace identity without importing candidate code."""
    name = "SCIEL_RESTART_TRACE_SCAFFOLD_SHA256"
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.Assign)
             and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)]
    if (len(nodes) != 1 or len(nodes[0].targets) != 1 or
            not isinstance(nodes[0].value, ast.Constant)):
        raise ValueError("trace scaffold literal missing or ambiguous")
    value = nodes[0].value.value
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("trace scaffold literal malformed")
    return value


def _bounded_regular_bytes(path, maximum):
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > maximum:
        raise ValueError("trace input must be bounded and regular")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(descriptor, "rb") as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError("trace input changed type")
        raw = stream.read(maximum + 1)
    if len(raw) > maximum:
        raise ValueError("trace input exceeds transport limit")
    return raw


def _read_restart_trace(path, *, source, seed, identity, invocation_id, setup, cpu_seconds, mem_bytes):
    """Trace invalidity is diagnostic only, independent of numeric certification."""
    result = {"status": "missing", "reason": "sidecar_missing", "sidecar_sha256": None, "trace": None}
    try:
        raw = _bounded_regular_bytes(path, _MAX_TRACE_BYTES)
        result["sidecar_sha256"] = _sha(raw)
        def unique(items):
            value = {}
            for key, item in items:
                if key in value:
                    raise ValueError("duplicate trace key")
                value[key] = item
            return value
        def constant(_value):
            raise ValueError("nonfinite trace number")
        envelope = json.loads(raw, object_pairs_hook=unique, parse_constant=constant)
        binding = {"source_sha256": _sha(source.encode("utf-8")), "seed": seed,
                   "runtime_sha256": _sha(_encoded(identity["runtime"])),
                   "adapter_identity_sha256": identity["identity_sha256"], "invocation_id": invocation_id}
        if (type(envelope) is not dict or set(envelope) != {"schema", "status", "reason", "trace", *binding}
                or envelope["schema"] != _TRACE_SCHEMA or
                any(_encoded(envelope[key]) != _encoded(value) for key, value in binding.items())):
            raise ValueError("trace envelope binding differs")
        status = envelope["status"]
        if type(status) is not str or status not in ("captured", "missing", "invalid", "error"):
            raise ValueError("trace envelope status malformed")
        if status != "captured":
            if (envelope["trace"] is not None or type(envelope["reason"]) is not str
                    or not envelope["reason"] or len(envelope["reason"]) > 128):
                raise ValueError("trace absence malformed")
            return {**result, "status": status, "reason": envelope["reason"]}
        expected_seed = {"python_random": seed, "numpy_legacy": seed, "python_hash_seed": str(seed),
                         "before_candidate_import": True, "determinism_proven": False}
        expected_setup = {"status": "ready", "source_sha256": binding["source_sha256"],
                          "runtime": identity["runtime"], "seed_applied": expected_seed,
                          "limits_applied": {"cpu_seconds": [cpu_seconds, cpu_seconds],
                              "mem_bytes": [mem_bytes, mem_bytes], "file_bytes": [1048576, 1048576]}}
        if _encoded(setup) != _encoded(expected_setup) or envelope["reason"] is not None:
            raise ValueError("trace candidate setup not verified")
        source_path = path.with_name("candidate.py")
        if _sha(_bounded_regular_bytes(source_path, len(source.encode("utf-8")))) != binding["source_sha256"]:
            raise ValueError("trace candidate source changed")
        trace = trace_codec.validate_trace(envelope["trace"], source_sha256=binding["source_sha256"],
            seed=seed, runtime_sha256=binding["runtime_sha256"], scaffold_sha256=_trace_scaffold(source),
            invocation_id=invocation_id)
        return {**result, "status": "captured", "reason": None, "trace": trace}
    except FileNotFoundError:
        return result if result["sidecar_sha256"] is None else {
            **result, "status": "invalid", "reason": "trace_source_unavailable"}
    except Exception as exc:
        return {**result, "status": "invalid", "reason": "trace_validation_failed:" + type(exc).__name__}


def _kill_group(proc):
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except OSError as exc:
            raise RepeatInfrastructureError(f"could not clean process group: {exc}") from exc
    if proc.poll() is None:
        proc.kill()
    try:
        proc.wait(timeout=1)
    except subprocess.TimeoutExpired as exc:
        raise RepeatInfrastructureError("child could not be reaped after kill") from exc


def _join_trace_quality(record):
    diagnostic = record.get("restart_trace")
    if record["status"] != "valid" or not diagnostic or diagnostic["status"] != "captured":
        return
    trace, raw = diagnostic["trace"], record["raw_sum"]
    total = trace["final_quality"]["sum"]
    # Only a same-result join: np.sum in the solver versus math.fsum over 26
    # nonnegative captured radii. This never relaxes a paired quality delta.
    bound = 0.0 if raw == 0.0 else 32 * math.ulp(raw)
    agrees = abs(total - raw) <= bound
    terminal_agrees = trace["status"] != "error"
    diagnostic["quality_join"] = {"raw_sum": raw, "trace_sum": total,
        "maximum_roundoff": bound, "within_roundoff": agrees,
        "terminal_status_compatible": terminal_agrees}
    if not agrees or not terminal_agrees:
        diagnostic.update(status="invalid", reason="trace_result_mismatch", trace=None)


def _process(mode, request, directory, *, timeout, python, seed, runtime=None):
    """Execute only the trusted runner; it alone imports candidate source."""
    if runtime and runtime["backend"] == "docker":
        return repeat_docker.process(mode, request, directory, timeout=timeout, seed=seed,
            image=runtime["image"], files=_FILES, write_json=_write_json, read_json=_read_json)
    begun = time.monotonic()
    request_path = directory / (mode + "_request.json")
    output_path = directory / (mode + "_output.json")
    setup_path = directory / (mode + "_setup.json")
    _write_json(request_path, request)
    proc = None
    timed_out = False
    with tempfile.TemporaryFile() as stderr:
        try:
            proc = subprocess.Popen(
                [python, "-s", "-P", "-B", str(_FILES["runner"]), mode,
                 str(request_path), str(setup_path), str(output_path)],
                cwd=directory, env=_environment(seed), stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL, stderr=stderr, start_new_session=True)
            try:
                proc.wait(timeout=max(0.0, timeout - (time.monotonic() - begun)))
            except subprocess.TimeoutExpired:
                timed_out = True
        finally:
            if proc is not None:
                _kill_group(proc)  # Includes descendants of a successfully exited leader.
        stderr.seek(0, os.SEEK_END)
        size = stderr.tell()
        stderr.seek(max(0, size - 2000))
        tail = stderr.read().decode("utf-8", "replace")
    setup = _read_json(setup_path) if setup_path.exists() else None
    output = _read_json(output_path) if output_path.exists() else None
    return {"exit_code": proc.returncode, "timed_out": timed_out, "stderr_tail": tail,
            "elapsed_seconds": time.monotonic() - begun, "setup": setup, "output": output}


def _source_hashes():
    hashes = {key: _sha(path.read_bytes()) for key, path in _FILES.items()}
    if any(hashes[key] != expected for key, expected in _LOADED_SHA.items()):
        raise RepeatInfrastructureError("loaded adapter/geometry differs from on-disk source")
    return hashes


def _caps_valid(caps, cpu_seconds, mem_bytes):
    return caps == {"cpu_seconds": [cpu_seconds, cpu_seconds],
                    "mem_bytes": [mem_bytes, mem_bytes], "file_bytes": [1048576, 1048576]}


def _runtime_config(runtime):
    if runtime is not None and type(runtime) is not dict:
        raise ValueError("runtime must be an object")
    value = {"backend": "native"} if runtime is None else dict(runtime)
    if value == {"backend": "native"}:
        return value
    if (set(value) == {"backend", "image"} and value["backend"] == "docker"
            and isinstance(value["image"], str) and value["image"].startswith("sha256:")
            and len(value["image"]) == 71 and all(c in "0123456789abcdef" for c in value["image"][7:])):
        return value
    raise ValueError("runtime must be native or Docker with an immutable image ID")


def adapter_identity(python=None, runtime=None):
    """Pin helper bytes and probe the actual clean child runtime without a candidate.

    No cache: a changed runtime/helper must invalidate the caller's frozen ledger.
    Broken runtime/rlimit/transport raises RepeatInfrastructureError at preflight.
    """
    try:
        if os.name != "posix":
            raise RepeatInfrastructureError("repeat draw requires POSIX process groups and rlimits")
        configuration = _runtime_config(runtime)
        docker = configuration["backend"] == "docker"
        execution = repeat_docker.image_identity(configuration["image"]) if docker else None
        executable = repeat_docker.PYTHON if docker else _python(python)
        hashes = _source_hashes()
        with tempfile.TemporaryDirectory(prefix="sciel_repeat_identity_") as directory:
            probe = _process("identity", {"cpu_seconds": 10, "mem_bytes": _PROBE_MEMORY},
                             Path(directory), timeout=_PROBE_SECONDS, python=executable, seed=0,
                             runtime=configuration)
        if probe["timed_out"] or probe["exit_code"] != 0 or not probe["output"]:
            raise RepeatInfrastructureError("runtime probe failed: " + probe["stderr_tail"])
        output = probe["output"]
        if set(output) != {"runtime", "limits_applied"} or not _caps_valid(output["limits_applied"], 10, _PROBE_MEMORY):
            raise RepeatInfrastructureError("runtime probe receipt is malformed")
        runtime = output["runtime"]
        if (set(runtime) != {"python_version", "python_executable", "python_implementation", "platform",
                            "numpy", "scipy", "blas_config", "thread_limits"} or
                runtime["thread_limits"] != _THREAD_LIMITS or
                any(not isinstance(v, str) or not v for k, v in runtime.items() if k != "thread_limits")):
            raise RepeatInfrastructureError("runtime identity is malformed")
        if _source_hashes() != hashes:
            raise RepeatInfrastructureError("helper changed during runtime probe")
        policy = {"name": geometry.CERTIFICATE_POLICY, "geometry_sha256": hashes["geometry"],
                  "raw_tolerance": geometry.TOLERANCE, "raw_target": geometry.TARGET_VALUE,
                  "strict_method": "common absolute decrement; clip at zero; round each radius down; exact recheck"}
        result = {"schema": "sciel-repeat-adapter/1", "source_sha256": hashes,
                  "certificate_policy": policy, "certificate_policy_sha256": _sha(_encoded(policy)),
                  "runtime": runtime, "runtime_config": configuration, "execution": execution}
        return {**result, "identity_sha256": _sha(_encoded(result))}
    except RepeatInfrastructureError:
        raise
    except Exception as exc:
        raise RepeatInfrastructureError(f"adapter preflight failed: {type(exc).__name__}: {exc}") from exc


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(name + " must be positive and finite")


def draw(source: str, *, seed: int, timeout: float, cpu_seconds: int, mem_bytes: int,
         certification_seconds: float, python=None, runtime=None, expected_identity=None,
         restart_trace=False) -> dict:
    """Return one terminal receipt; failed/unknown quality is None, never zero.

    A normal candidate timeout/exception/invalid geometry is candidate_failure.
    Missing/broken setup, parent output I/O, rlimits or certificate helper failures
    are infrastructure. Certificate rejection is candidate_failure. Candidate and
    certificate use distinct fresh processes, identical CPU/memory caps, and their
    separately declared wall budgets. With expected_identity, source hashes are
    checked before dispatch and the child checks its runtime before candidate
    import. Without one, a fresh identity probe is separately timed overhead.
    Docker phase allowances cover create/attach; cleanup can additionally take
    two seconds to reap the CLI and ten seconds to force-remove the container.
    Total elapsed also includes bounded-size host file I/O and is not a hard
    adapter-wide wall cap. This scope is bound in the Docker identity policy.
    restart_trace=True captures a separate bounded diagnostic sidecar from that
    same invocation. Missing/invalid trace facts do not change numeric quality.
    """
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be nonempty Python text")
    if type(restart_trace) is not bool:
        raise ValueError("restart_trace must be a boolean")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    _positive(timeout, "timeout")
    _positive(certification_seconds, "certification_seconds")
    if type(cpu_seconds) is not int or cpu_seconds <= 0 or type(mem_bytes) is not int or mem_bytes <= 0:
        raise ValueError("cpu_seconds and mem_bytes must be positive integers")
    configuration = _runtime_config(runtime)
    started = time.monotonic()
    record = {"schema": "sciel-repeat-draw/1", "source_sha256": _sha(source.encode("utf-8")),
              "adapter_identity": None, "seed": seed, "seed_applied": None,
              "limits": {"candidate_wall_seconds": float(timeout),
                         "certification_wall_seconds": float(certification_seconds),
                         "candidate_cpu_seconds": cpu_seconds, "candidate_mem_bytes": mem_bytes,
                         "certification_cpu_seconds": cpu_seconds, "certification_mem_bytes": mem_bytes},
              "status": "infrastructure", "reason": None, "capture": None, "certificate": None,
              "raw_sum": None, "certified_sum": None, "candidate_elapsed_seconds": 0.0,
              "certification_elapsed_seconds": 0.0, "identity_elapsed_seconds": 0.0,
              "candidate_process": None, "certification_process": None,
              "started_at": _now(), "completed_at": None, "elapsed_seconds": None}
    if restart_trace:
        record["restart_trace"] = {"status": "missing", "reason": "candidate_not_dispatched",
                                   "sidecar_sha256": None, "trace": None}
    try:
        if expected_identity is None:
            identity_started = time.monotonic()
            try:
                identity = adapter_identity(python, runtime=configuration)
            finally:
                record["identity_elapsed_seconds"] = time.monotonic() - identity_started
        else:
            # Round-trip prevents mutation by a caller while this draw is running.
            identity = json.loads(_encoded(expected_identity))
            bare = {k: v for k, v in identity.items() if k != "identity_sha256"}
            if (identity.get("schema") != "sciel-repeat-adapter/1" or
                    identity.get("identity_sha256") != _sha(_encoded(bare)) or
                    identity.get("source_sha256") != _source_hashes() or
                    identity.get("runtime_config") != configuration or
                    (repeat_docker.PYTHON if configuration["backend"] == "docker" else _python(python))
                    != identity["runtime"]["python_executable"]):
                raise RepeatInfrastructureError("frozen adapter identity/source/interpreter differs before dispatch")
        record["adapter_identity"] = identity
        executable = identity["runtime"]["python_executable"]
        with tempfile.TemporaryDirectory(prefix="sciel_repeat_draw_") as temporary:
            directory = Path(temporary)
            source_path = directory / "candidate.py"
            source_path.write_bytes(source.encode("utf-8"))
            request = {"source_path": str(source_path), "source_sha256": record["source_sha256"],
                       "seed": seed, "cpu_seconds": cpu_seconds, "mem_bytes": mem_bytes,
                       "expected_runtime": identity["runtime"]}
            if restart_trace:
                request.update(restart_trace=True, adapter_identity_sha256=identity["identity_sha256"],
                               restart_trace_invocation_id=uuid.uuid4().hex)
            phase_started = time.monotonic()
            process = None
            try:
                process = _process("candidate", request, directory, timeout=timeout, python=executable, seed=seed,
                                   runtime=configuration)
            finally:
                record["candidate_elapsed_seconds"] = time.monotonic() - phase_started
                if restart_trace:
                    record["restart_trace"] = _read_restart_trace(directory / "candidate_trace.json",
                        source=source, seed=seed, identity=identity,
                        invocation_id=request["restart_trace_invocation_id"],
                        setup=process.get("setup") if process is not None else None,
                        cpu_seconds=cpu_seconds, mem_bytes=mem_bytes)
            record["candidate_process"] = {k: v for k, v in process.items() if k != "output"}
            setup = process["setup"]
            expected_seed = {"python_random": seed, "numpy_legacy": seed, "python_hash_seed": str(seed),
                             "before_candidate_import": True, "determinism_proven": False}
            if (not setup or set(setup) != {"status", "source_sha256", "runtime", "seed_applied", "limits_applied"} or
                    setup["status"] != "ready" or setup["source_sha256"] != record["source_sha256"] or
                    setup["runtime"] != identity["runtime"] or setup["seed_applied"] != expected_seed or
                    not _caps_valid(setup["limits_applied"], cpu_seconds, mem_bytes)):
                raise RepeatInfrastructureError("candidate setup/seed/runtime/limits receipt missing or changed")
            record["seed_applied"] = setup["seed_applied"]
            if process["exit_code"] == 70 and "repeat helper output:" in process["stderr_tail"]:
                raise RepeatInfrastructureError("candidate helper output failed: " + process["stderr_tail"])
            if process["timed_out"] or process["exit_code"] != 0:
                record.update(status="candidate_failure", reason="candidate timeout" if process["timed_out"]
                              else f"candidate exited {process['exit_code']}")
            else:
                output = process["output"]
                if not output or output.get("status") not in ("ok", "candidate_failure"):
                    raise RepeatInfrastructureError("candidate helper output missing or malformed")
                if output["status"] == "candidate_failure":
                    if set(output) != {"status", "reason"} or not isinstance(output["reason"], str):
                        raise RepeatInfrastructureError("candidate error receipt malformed")
                    record.update(status="candidate_failure", reason=output["reason"])
                else:
                    if set(output) != {"status", "packing"}:
                        raise RepeatInfrastructureError("candidate numeric receipt malformed")
                    capture = geometry.validate_packing(output["packing"])
                    record["capture"] = capture
                    if not capture["valid"]:
                        record.update(status="candidate_failure", reason=capture["error"])
                    else:
                        packing_path = directory / "packing.json"
                        _write_json(packing_path, {"centers": capture["centers"], "radii": capture["radii"]})
                        phase_started = time.monotonic()
                        try:
                            certificate_process = _process("certificate", {
                                "packing_path": str(packing_path), "cpu_seconds": cpu_seconds, "mem_bytes": mem_bytes},
                                directory, timeout=certification_seconds, python=executable, seed=seed,
                                runtime=configuration)
                        finally:
                            record["certification_elapsed_seconds"] = time.monotonic() - phase_started
                        record["certification_process"] = {k: v for k, v in certificate_process.items() if k != "output"}
                        cert_setup, cert_output = certificate_process["setup"], certificate_process["output"]
                        if (certificate_process["timed_out"] or certificate_process["exit_code"] != 0 or
                                not cert_setup or set(cert_setup) != {"status", "limits_applied"} or
                                cert_setup["status"] != "ready" or not _caps_valid(cert_setup["limits_applied"], cpu_seconds, mem_bytes) or
                                not cert_output or set(cert_output) != {"certificate"}):
                            raise RepeatInfrastructureError("certificate process/limits/output failed")
                        certificate = cert_output["certificate"]
                        if not isinstance(certificate, dict) or certificate.get("status") not in ("certified", "rejected"):
                            raise RepeatInfrastructureError("certificate helper returned malformed receipt")
                        record["certificate"] = certificate
                        if certificate["status"] != "certified":
                            record.update(status="candidate_failure", reason="certificate rejected: " + str(certificate["error"]))
                        else:
                            value = certificate["sum_exact"]
                            if (set(value) != {"numerator", "denominator"} or any(type(v) is not int for v in value.values()) or
                                    value["numerator"] < 0 or value["denominator"] <= 0):
                                raise RepeatInfrastructureError("certified exact sum is malformed")
                            record.update(status="valid", raw_sum=capture["sum_radii"], certified_sum=value)
        _join_trace_quality(record)
        try:
            if _source_hashes() != identity["source_sha256"]:
                raise RepeatInfrastructureError("helper source changed during draw")
        except RepeatInfrastructureError:
            if restart_trace:
                record["restart_trace"].update(status="invalid", reason="adapter_source_changed", trace=None)
            raise
    except Exception as exc:
        record.update(status="infrastructure", reason=f"{type(exc).__name__}: {exc}"[:2000],
                      raw_sum=None, certified_sum=None)
    finally:
        record["completed_at"] = _now()
        record["elapsed_seconds"] = time.monotonic() - started
    return record
