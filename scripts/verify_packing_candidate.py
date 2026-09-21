"""Independently validate fresh packing outputs without importing the candidate.

The candidate runs in a separate, bounded process. Only JSON coordinates and radii
cross into the parent, which uses ordinary Python arithmetic to verify geometry;
the candidate's reported sum and mutations of its NumPy module are never trusted.
These are new candidate executions, not rechecks of the harness's earlier draws.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

N_CIRCLES = 26
TOLERANCE = 1e-6
TARGET_VALUE = 2.635
_MAX_JSON_BYTES = 65536
_THREAD_LIMITS = {name: "1" for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
    "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")}
_ENV_ALLOWLIST = ("PATH", "LANG", "LC_ALL", "TMPDIR", "SystemRoot")


def _failure(error):
    return {"centers": None, "radii": None, "valid": False, "strict_valid": False,
            "score": 0.0, "strict_score": 0.0, "sum_radii": None,
            "min_pair_slack": None, "min_wall_slack": None, "min_radius": None,
            "error": str(error)}


def _number(value):
    if type(value) not in (int, float):
        raise ValueError("coordinates and radii must be JSON numbers")
    try:
        result = float(value)
    except (OverflowError, ValueError):
        raise ValueError("coordinates and radii must be finite") from None
    if not math.isfinite(result):
        raise ValueError("coordinates and radii must be finite")
    return result


def validate_packing(payload):
    """Validate numeric JSON data, with tolerance 1e-6 and strict-zero diagnostics.

    ``sum_radii`` records the actual numeric sum even for an overlapping packing;
    ``score`` is zero unless all geometric checks pass. Malformed data are rejected
    rather than coerced into a potentially valid packing.
    """
    if not isinstance(payload, dict):
        return _failure("packing payload must be a JSON object")
    centers, radii = payload.get("centers"), payload.get("radii")
    if (not isinstance(centers, list) or len(centers) != N_CIRCLES or
            any(not isinstance(row, list) or len(row) != 2 for row in centers) or
            not isinstance(radii, list) or len(radii) != N_CIRCLES):
        return _failure("expected centers shape (26, 2) and radii shape (26,)")
    try:
        centers = [[_number(x), _number(y)] for x, y in centers]
        radii = [_number(radius) for radius in radii]
    except ValueError as exc:
        return _failure(exc)
    minimum_radius = min(radii)
    wall = min(min(x - r, y - r, 1 - x - r, 1 - y - r)
               for (x, y), r in zip(centers, radii))
    pair = min(math.hypot(centers[i][0] - centers[j][0], centers[i][1] - centers[j][1])
               - radii[i] - radii[j] for i in range(N_CIRCLES) for j in range(i + 1, N_CIRCLES))
    try:
        total = math.fsum(radii)
    except OverflowError:
        return _failure("sum of radii is not finite")
    if not all(math.isfinite(v) for v in (wall, pair, total)):
        return _failure("packing geometry is outside the finite numeric range")
    errors = []
    if minimum_radius < 0:
        errors.append("negative radius")
    if wall < -TOLERANCE:
        errors.append("circle outside unit square")
    if pair < -TOLERANCE:
        errors.append("overlapping circles")
    valid = not errors
    strict = minimum_radius >= 0 and wall >= 0 and pair >= 0
    return {"centers": centers, "radii": radii, "valid": valid, "strict_valid": strict,
            "score": total / TARGET_VALUE if valid else 0.0,
            "strict_score": total / TARGET_VALUE if strict else 0.0,
            "sum_radii": total, "min_pair_slack": pair, "min_wall_slack": wall,
            "min_radius": minimum_radius, "error": "; ".join(errors) if errors else None}


def _extract(program_path, output_path):
    """Child-only entry point: transport arrays, never compute their score."""
    try:
        spec = importlib.util.spec_from_file_location("packing_candidate", program_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.run_packing()
        if not isinstance(result, (tuple, list)) or len(result) != 3:
            raise ValueError("run_packing must return (centers, radii, reported_sum)")
        centers, radii, _reported_sum = result
        # Do not call np.sum, np.asarray, or any geometry helper in this process.
        # The parent checks shapes/finite values again after parsing plain JSON.
        centers = centers.tolist() if hasattr(centers, "tolist") else centers
        radii = radii.tolist() if hasattr(radii, "tolist") else radii
        payload = {"centers": [[float(value) for value in row] for row in centers],
                   "radii": [float(value) for value in radii]}
        data = json.dumps(payload, allow_nan=False).encode()
        if len(data) > _MAX_JSON_BYTES:
            raise ValueError("packing JSON exceeds transport size limit")
        temporary = str(output_path) + ".tmp"
        with open(temporary, "wb") as stream:
            stream.write(data)
        os.replace(temporary, output_path)
        return 0
    except BaseException as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


def _kill_group(proc):
    """Same POSIX process-group cleanup pattern as sciel.eval.sandbox."""
    if os.name == "posix":
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=1)
    except (subprocess.TimeoutExpired, OSError):
        pass


def _sample(source, timeout, python):
    started = time.monotonic()
    proc = None
    with tempfile.TemporaryDirectory(prefix="sciel_packing_verify_") as directory:
        directory = Path(directory)
        program_path, output_path = directory / "candidate.py", directory / "packing.json"
        program_path.write_bytes(source)
        status, error, payload = "error", None, None
        with tempfile.TemporaryFile() as stderr:
            try:
                proc = subprocess.Popen(
                    [python, "-I", "-B", str(Path(__file__).resolve()), "--_extract",
                     str(program_path), str(output_path)],
                    cwd=directory, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                    stderr=stderr, start_new_session=(os.name == "posix"),
                    env={**{key: os.environ[key] for key in _ENV_ALLOWLIST if key in os.environ},
                         **_THREAD_LIMITS, "PYTHONDONTWRITEBYTECODE": "1"})
                proc.wait(timeout=max(0.0, timeout - (time.monotonic() - started)))
                if proc.returncode != 0:
                    error = f"candidate extraction failed (exit {proc.returncode})"
                elif not output_path.is_file():
                    error = "candidate produced no numeric packing JSON"
                elif output_path.stat().st_size > _MAX_JSON_BYTES:
                    error = "packing JSON exceeds transport size limit"
                else:
                    try:
                        payload = json.loads(output_path.read_text())
                        status = "ok"
                    except (ValueError, OSError) as exc:
                        error = f"invalid packing JSON: {exc}"
            except subprocess.TimeoutExpired:
                status, error = "timeout", f"candidate exceeded {timeout:g}s execution limit"
            except OSError as exc:
                error = f"candidate could not start: {exc}"
            finally:
                if proc is not None:
                    _kill_group(proc)  # Also kill descendants after successful return.
            stderr.seek(0, os.SEEK_END)
            size = stderr.tell()
            stderr.seek(max(0, size - 2000))
            error_tail = stderr.read().decode("utf-8", "replace")
        checked = validate_packing(payload) if status == "ok" else _failure(error)
        if status == "ok" and not checked["valid"]:
            status = "invalid"
        return {**checked, "status": status, "elapsed_seconds": time.monotonic() - started,
                "stderr_tail": error_tail, "exit_code": proc.returncode if proc is not None else None}


def verify_candidate(program_path, *, repeats=3, timeout=5.0, expected_sha256=None, python=None):
    """Extract and verify up to three fresh candidate executions, at most 5s each.

    Process groups cover normal spawned descendants; this is numerical verification
    and bounded execution, not a claim of a hostile-code operating-system sandbox.
    The caller can compare this additional sample set with prior harness repeats,
    but these are not the same captured packings and are not separate search runs.
    """
    if type(repeats) is not int or not 1 <= repeats <= 3:
        raise ValueError("repeats must be an integer from 1 to 3")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0 < timeout <= 5:
        raise ValueError("timeout must be positive, finite and at most 5 seconds")
    program_path = Path(program_path).resolve()
    source = program_path.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if expected_sha256 is not None and expected_sha256 != digest:
        raise ValueError("candidate source SHA-256 does not match expected artifact")
    samples = []
    for repeat in range(1, repeats + 1):
        if hashlib.sha256(program_path.read_bytes()).hexdigest() != digest:
            raise ValueError("candidate source changed before verification")
        sample = _sample(source, float(timeout), python or sys.executable)
        if hashlib.sha256(program_path.read_bytes()).hexdigest() != digest:
            raise ValueError("candidate source changed during verification")
        samples.append({"repeat": repeat, "source_sha256": digest, **sample})
    return {"schema": "sciel-independent-packing-validation/1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_path": str(program_path), "source_sha256": digest,
            "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "n_circles": N_CIRCLES, "tolerance": TOLERANCE, "strict_tolerance": 0.0,
            "target_value": TARGET_VALUE, "timeout_seconds": float(timeout), "repeats": repeats,
            "new_candidate_executions": True,
            "sample_scope": "Fresh candidate executions; not the harness's previously scored packings or independent search runs.",
            "samples": samples, "valid_repeats": sum(s["valid"] for s in samples),
            "all_valid": all(s["valid"] for s in samples),
            "strict_all_valid": all(s["strict_valid"] for s in samples),
            "mean_score": math.fsum(s["score"] for s in samples) / repeats,
            "paid_calls": 0}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("program", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=5.0)
    parser.add_argument("--expected-sha256")
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("verification output already exists; preserve earlier samples")
    result = verify_candidate(args.program, repeats=args.repeats, timeout=args.timeout,
                              expected_sha256=args.expected_sha256)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(args.output.resolve()), "all_valid": result["all_valid"],
                      "strict_all_valid": result["strict_all_valid"],
                      "valid_repeats": result["valid_repeats"], "mean_score": result["mean_score"]}))
    return 0 if result["all_valid"] else 2


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--_extract":
        raise SystemExit(_extract(sys.argv[2], sys.argv[3]))
    raise SystemExit(main())
