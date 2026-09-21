"""Private fresh-process H15 transport. Never imported by the engine.

Invoke by absolute script path with -s -P -B: -I would ignore PYTHONHASHSEED.
Only trusted package helpers are loaded explicitly by path. Candidate source is
imported exactly once, after resource limits and Python/NumPy seeds are applied.
This is bounded numerical execution, not a hostile-code security sandbox.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import random
import stat
import sys

MAX_JSON_BYTES = 65536
MAX_TRACE_BYTES = 96 * 1024
TRACE_SCHEMA = "sciel-repeat-restart-trace/1"
THREAD_NAMES = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS")


def _write(path, value):
    data = json.dumps(value, sort_keys=True, allow_nan=False).encode()
    if len(data) > MAX_JSON_BYTES:
        raise ValueError("helper JSON exceeds transport size limit")
    temporary = str(path) + ".tmp"
    with open(temporary, "xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _trace_bytes(value):
    """Bound serialization independently of the unchanged packing transport."""
    chunks, size = [], 0
    encoder = json.JSONEncoder(sort_keys=True, separators=(",", ":"), allow_nan=False)
    for chunk in encoder.iterencode(value):
        if len(chunk) > MAX_TRACE_BYTES - size:
            raise ValueError("trace_size")
        data = chunk.encode("utf-8")
        size += len(data)
        if size > MAX_TRACE_BYTES:
            raise ValueError("trace_size")
        chunks.append(data)
    return b"".join(chunks)


def _write_trace(output_path, request, binding, module, *, source_size):
    """Best-effort diagnostics after the one invocation, including exceptions.

    The host validates the collector payload. Failure here never changes the
    existing candidate numeric result or writes anything to its stderr marker.
    """
    envelope = {"schema": TRACE_SCHEMA, **binding, "status": "missing",
                "trace": None, "reason": "module_unavailable"}
    try:
        descriptor = os.open(request["source_path"], os.O_RDONLY |
                             getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0))
        with os.fdopen(descriptor, "rb") as stream:
            info = os.fstat(stream.fileno())
            source = stream.read(source_size + 1) if stat.S_ISREG(info.st_mode) and info.st_size == source_size else b""
        if hashlib.sha256(source).hexdigest() != binding["source_sha256"]:
            envelope.update(status="invalid", reason="source_changed")
        elif module is not None:
            trace = module.__dict__.get("SCIEL_RESTART_TRACE")
            if trace is None:
                envelope["reason"] = "trace_not_exported"
            elif type(trace) is not dict:
                envelope.update(status="invalid", reason="trace_not_object")
            else:
                envelope.update(status="captured", reason=None, trace=trace)
        try:
            raw = _trace_bytes(envelope)
        except (ValueError, TypeError, OverflowError, RecursionError, UnicodeError):
            envelope.update(status="invalid", reason="trace_not_bounded_json", trace=None)
            raw = _trace_bytes(envelope)
    except BaseException:
        envelope.update(status="error", reason="trace_capture_error", trace=None)
        raw = _trace_bytes(envelope)
    path = Path(output_path).with_name("candidate_trace.json")
    temporary = path.with_name(path.name + ".tmp")
    try:
        # Exclusive creation and atomic replacement never follow a candidate's
        # destination symlink. A failed write is reported as missing by the host.
        with temporary.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        pass


def _limits(cpu_seconds, mem_bytes):
    import resource
    # Failure is infrastructure, never a warning followed by an unlimited run.
    for name, value in (("RLIMIT_CPU", cpu_seconds), ("RLIMIT_AS", mem_bytes),
                        ("RLIMIT_FSIZE", 1048576)):
        try:
            resource.setrlimit(getattr(resource, name), (value, value))
        except (ValueError, OSError) as exc:
            raise RuntimeError(f"cannot apply {name}={value}: {exc}") from exc
    return {"cpu_seconds": list(resource.getrlimit(resource.RLIMIT_CPU)),
            "mem_bytes": list(resource.getrlimit(resource.RLIMIT_AS)),
            "file_bytes": list(resource.getrlimit(resource.RLIMIT_FSIZE))}


def _runtime():
    import numpy as np
    import scipy
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        np.__config__.show()
    return {"python_version": sys.version, "python_executable": sys.executable,
            "python_implementation": platform.python_implementation(),
            "platform": platform.platform(), "numpy": np.__version__,
            "scipy": scipy.__version__, "blas_config": stream.getvalue(),
            "thread_limits": {name: os.environ.get(name) for name in THREAD_NAMES}}


def _load(path, name):
    # Native source tree or the explicitly mounted, read-only Docker package.
    # Do not add the candidate directory/cwd or accept ambient PYTHONPATH.
    directory = Path(__file__).resolve().parent
    package_root = (directory if (directory / "sciel/candidate_modules/__init__.py").is_file()
                    else directory.parents[1])
    if not (package_root / "sciel/candidate_modules/__init__.py").is_file():
        raise RuntimeError("trusted candidate loader unavailable")
    if str(package_root) not in sys.path:
        sys.path.insert(0, str(package_root))
    from sciel.candidate_modules import load_candidate
    return load_candidate(path, name)


def main(argv):
    if len(argv) != 4 or argv[0] not in ("identity", "candidate", "certificate"):
        return 70
    mode, request_path, setup_path, output_path = argv
    try:
        request = json.loads(Path(request_path).read_text())
        caps = _limits(request["cpu_seconds"], request["mem_bytes"])
        if mode == "identity":
            _write(output_path, {"runtime": _runtime(), "limits_applied": caps})
            return 0
        if mode == "certificate":
            # No NumPy, SciPy or candidate imports in the certification process.
            geometry = _load(Path(__file__).with_name("packing_geometry.py"), "_repeat_geometry")
            payload = json.loads(Path(request["packing_path"]).read_text())
            _write(setup_path, {"status": "ready", "limits_applied": caps})
            certificate = geometry.certify_packing(payload)
            _write(output_path, {"certificate": certificate})
            return 0
        source = Path(request["source_path"]).read_bytes()
        if hashlib.sha256(source).hexdigest() != request["source_sha256"]:
            raise RuntimeError("candidate source identity changed before import")
        runtime = _runtime()
        if runtime != request["expected_runtime"]:
            raise RuntimeError("candidate runtime differs from the frozen adapter identity")
        seed = request["seed"]
        trace_enabled = request.get("restart_trace", False)
        if type(trace_enabled) is not bool:
            raise ValueError("restart_trace must be a boolean")
        trace_binding = None
        if trace_enabled:
            trace_binding = {"source_sha256": request["source_sha256"], "seed": seed,
                "runtime_sha256": hashlib.sha256(json.dumps(runtime, sort_keys=True,
                    separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
                "adapter_identity_sha256": request["adapter_identity_sha256"],
                "invocation_id": request["restart_trace_invocation_id"]}
            for key, length in (("adapter_identity_sha256", 64), ("invocation_id", 32)):
                value = trace_binding[key]
                if type(value) is not str or len(value) != length or any(c not in "0123456789abcdef" for c in value):
                    raise ValueError("invalid restart trace binding")
        import numpy as np
        random.seed(seed)
        np.random.seed(seed)
        if os.environ.get("PYTHONHASHSEED") != str(seed):
            raise RuntimeError("PYTHONHASHSEED was not applied at spawn")
        receipt = {"python_random": seed, "numpy_legacy": seed,
                   "python_hash_seed": str(seed), "before_candidate_import": True,
                   "determinism_proven": False}
        _write(setup_path, {"status": "ready", "source_sha256": request["source_sha256"],
                           "runtime": runtime, "seed_applied": receipt, "limits_applied": caps})
    except BaseException as exc:
        print(f"repeat helper setup: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 70
    module = None
    try:
        module = _load(request["source_path"], "packing_candidate")
        if trace_enabled:
            module.__dict__["SCIEL_RESTART_TRACE_BINDING"] = dict(trace_binding)
        result = module.run_packing()
        if not isinstance(result, (tuple, list)) or len(result) != 3:
            raise ValueError("run_packing must return (centers, radii, reported_sum)")
        centers, radii, _reported_sum = result
        centers = centers.tolist() if hasattr(centers, "tolist") else centers
        radii = radii.tolist() if hasattr(radii, "tolist") else radii
        payload = {"centers": [[float(value) for value in row] for row in centers],
                   "radii": [float(value) for value in radii]}
        data = json.dumps(payload, allow_nan=False).encode()
        if len(data) > MAX_JSON_BYTES:
            raise ValueError("packing JSON exceeds transport size limit")
        result = {"status": "ok", "packing": payload}
    except BaseException as exc:
        result = {"status": "candidate_failure", "reason": f"{type(exc).__name__}: {exc}"[:2000]}
    if trace_enabled:
        try:
            _write_trace(output_path, request, trace_binding, module, source_size=len(source))
        except BaseException:
            # The unchanged numeric output still has priority if diagnostics
            # cannot even construct their bounded error envelope.
            pass
    try:
        _write(output_path, result)
    except BaseException as exc:
        print(f"repeat helper output: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 70
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
