"""Evaluator for circle_packing (n=26); smoke/full/confirm on the sciel contract."""

import hashlib
from sciel.candidate_modules import load_candidate
import json
from pathlib import Path
import uuid
import time
import traceback

import numpy as np

TARGET_VALUE = 2.635
N_CIRCLES = 26


def _load(program_path, module_name="program"):
    return load_candidate(program_path, module_name)


def _validate_packing(centers, radii):
    # Check the represented coordinates in binary64. Arithmetic in float16/32
    # can round an overlap or wall violation away before comparing the tolerance.
    # Do not change the original arrays or their native score reduction dtype.
    centers = np.asarray(centers, dtype=np.float64)
    radii = np.asarray(radii, dtype=np.float64)
    n = centers.shape[0]
    if np.isnan(centers).any():
        return False
    if np.isnan(radii).any():
        return False
    for i in range(n):
        if radii[i] < 0:
            return False
        elif np.isnan(radii[i]):
            return False
    for i in range(n):
        x, y = centers[i]
        r = radii[i]
        if x - r < -1e-6 or x + r > 1 + 1e-6 or y - r < -1e-6 or y + r > 1 + 1e-6:
            return False
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
            if dist < radii[i] + radii[j] - 1e-6:
                return False
    return True


# --- behavior readout -------------------------------------------------------
# The score is the sum of radii. These describe the SHAPE of the packing instead --
# how uneven the circles are, how much of it leans on the walls, how tightly the
# circles touch. Two packings with the same sum can differ on all six, which is the
# whole point of a diversity map. Keys are the engine's task-agnostic slots.

_BEHAVIOR_ZERO = {"b_dispersion": 0.0, "b_boundary_frac": 0.0, "b_contact": 0.0,
                  "b_offset": 0.0, "b_concentration": 0.0, "b_structure": 0.0}


def _behavior(centers, radii, valid):
    """Six bounded shape descriptors. An invalid packing has no shape worth
    describing, so it reports zeros -- as does any degenerate input."""
    if not valid:
        return dict(_BEHAVIOR_ZERO)
    try:
        c = np.asarray(centers, dtype=float)
        r = np.asarray(radii, dtype=float)
        n = r.shape[0]
        if n == 0 or c.shape[0] != n or not np.isfinite(c).all() or not np.isfinite(r).all():
            return dict(_BEHAVIOR_ZERO)
        mean_r = float(r.mean())
        dispersion = float(r.std() / mean_r) if mean_r > 0 else 0.0

        wall = np.minimum.reduce([c[:, 0] - r, c[:, 1] - r, 1.0 - c[:, 0] - r,
                                  1.0 - c[:, 1] - r])
        boundary_frac = float((wall < 1e-6).mean())

        d = np.linalg.norm(c[:, None, :] - c[None, :, :], axis=-1)
        touch = d - (r[:, None] + r[None, :])
        np.fill_diagonal(touch, np.inf)
        contacts = (touch < 1e-6).sum(axis=1)
        # 6 is the hexagonal maximum for equal circles; normalise to [0,1]
        contact = float(np.clip(contacts.mean() / 6.0, 0.0, 1.0))

        total = float(r.sum())
        if total > 0:
            weighted = (c * r[:, None]).sum(axis=0) / total
        else:
            weighted = c.mean(axis=0)
        offset = float(np.clip(np.linalg.norm(weighted - 0.5) / 0.5, 0.0, 1.0))

        k = max(1, n // 4)
        big = np.sort(r)[-k:]
        concentration = float(big.sum() / total) if total > 0 else 0.0

        structure = float(len(np.unique(np.round(r, 4))) / n)

        out = {"b_dispersion": dispersion / (1.0 + dispersion), "b_boundary_frac": boundary_frac,
               "b_contact": contact, "b_offset": offset,
               "b_concentration": concentration, "b_structure": structure}
        return {k2: (float(v) if np.isfinite(v) else 0.0) for k2, v in out.items()}
    except (ValueError, FloatingPointError, ZeroDivisionError):
        return dict(_BEHAVIOR_ZERO)


def _as_text(value):
    """Format a program-supplied number without ever raising on it."""
    try:
        return f"{float(value):.6f}"
    except Exception:
        try:
            return repr(value)[:80]
        except Exception:
            return "<unprintable reported_sum>"


def _run(program_path, module_name):
    program = _load(program_path, module_name)
    if not hasattr(program, "run_packing"):
        raise AttributeError("missing run_packing")
    centers, radii, reported_sum = program.run_packing()
    if not isinstance(centers, np.ndarray):
        centers = np.array(centers)
    if not isinstance(radii, np.ndarray):
        radii = np.array(radii)
    return centers, radii, reported_sum


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _snapshot(centers, radii):
    """Detach the exact finite real arrays used for both scoring and capture.

    Preserve floating dtypes: converting float32 radii to float64 before np.sum
    would silently change the native metric. Reject unsupported/rounded payloads
    instead of reporting geometry different from the one actually measured.
    """
    values = []
    for name, value, shape in (("centers", centers, (N_CIRCLES, 2)),
                               ("radii", radii, (N_CIRCLES,))):
        array = np.array(value, copy=True)
        if array.shape != shape:
            raise ValueError(f"Invalid shapes: {name}={array.shape}, expected {shape}")
        if array.dtype.kind not in "iuf" or array.dtype.itemsize > 8:
            raise ValueError(f"{name} must contain supported real numbers")
        if not np.isfinite(array).all():
            raise ValueError(f"{name} must be finite")
        # Python float/int equality checks exact representability for integers.
        if any(x.item() != float(x) for x in array.flat):
            raise ValueError(f"{name} cannot be represented exactly as binary64 JSON")
        # Integer wall/contact arithmetic can wrap (uint8: 128+128 == 0),
        # admitting an impossible packing. Exactly represented integer inputs
        # use float64; floating inputs retain their original reduction dtype.
        if array.dtype.kind in "iu":
            array = array.astype(float)
        values.append(array)
    return values


def _evaluate(program_path, stage):
    start_time = time.time()
    try:
        source_sha = _source_hash(program_path)
        evaluator_sha = _source_hash(__file__)
        invocation_id = uuid.uuid4().hex
        centers, radii, reported_sum = _run(program_path, "program_" + stage)
        centers, radii = _snapshot(centers, radii)
        if source_sha != _source_hash(program_path) or evaluator_sha != _source_hash(__file__):
            raise ValueError("source or evaluator changed during native invocation")
        valid = _validate_packing(centers, radii)
        sum_radii = float(np.sum(radii)) if valid else 0.0
        target_ratio = float(sum_radii / TARGET_VALUE) if valid else 0.0
        cases = {
            "sum_radii": sum_radii,
            "target_ratio": target_ratio,
            "validity": 1.0 if valid else 0.0,
            "combined_score": target_ratio,
        }
        cases.update(_behavior(centers, radii, valid))
        numeric = {"centers": centers.astype(float).tolist(),
                   "radii": radii.astype(float).tolist()}
        geometry = {
            "schema": "sciel-native-circle-geometry/1",
            "source_sha256": source_sha, "evaluator_sha256": evaluator_sha,
            "invocation_id": invocation_id, "stage": stage, "confirmation": None,
            "numeric": numeric,
            "numeric_sha256": hashlib.sha256(_canonical(numeric).encode()).hexdigest(),
            "centers_dtype": centers.dtype.name, "radii_dtype": radii.dtype.name,
            "native_validity": bool(valid), "native_sum_radii": sum_radii,
            "native_score": target_ratio,
        }
        # EvalResult artifacts are strings. A dict would become Python repr in
        # the subprocess adapter and would no longer be a strict JSON artifact.
        artifacts = {"native_geometry": _canonical(geometry)}
        if stage == "full":
            artifacts.update(eval_time=f"{time.time() - start_time:.4f}",
                             reported_sum=_as_text(reported_sum))
        # Formatting an untrusted reported sum can invoke __float__/__repr__.
        # Keep the final byte check after it, not only after run_packing().
        if source_sha != _source_hash(program_path) or evaluator_sha != _source_hash(__file__):
            raise ValueError("source or evaluator changed during native invocation")
        return {"ok": True, "score": target_ratio, "cases": cases, "artifacts": artifacts}
    except Exception as exc:
        return _fail(str(exc))


def smoke(program_path):
    return _evaluate(program_path, "smoke")


def full(program_path):
    return _evaluate(program_path, "full")


def confirm(program_path, n=3):
    """Re-run the program n times and keep the WORST result.

    This used to be `return full(program_path)`: n was ignored, one extra
    draw was taken, and the engine still recorded the result as a confirmed
    multi-sample. Keeping the worst is the point of a confirmation -- a
    record should survive its own repetition."""
    n = max(1, int(n or 1))
    runs = [full(program_path) for _ in range(n)]
    failures = [r for r in runs if not r.get("ok")]
    # A single failed repetition disproves reliable execution. Do not certify
    # the surviving successes as if every requested trial had succeeded.
    selected = (next(i for i, result in enumerate(runs) if not result.get("ok"))
                if failures else min(range(n), key=lambda i: runs[i]["score"]))
    worst = dict(runs[selected])
    worst["cases"] = {**worst["cases"], "n_trials": float(n)}
    worst["artifacts"] = dict(worst.get("artifacts", {}))
    selection = {"trials": n, "selected_trial": selected,
                 "selection": "first_failure_else_lowest_score"}
    worst["artifacts"]["confirmation_selection"] = _canonical(selection)
    if "native_geometry" in worst["artifacts"]:
        geometry = json.loads(worst["artifacts"]["native_geometry"])
        geometry["confirmation"] = selection
        worst["artifacts"]["native_geometry"] = _canonical(geometry)
    return worst


def _fail(msg):
    return {
        "ok": False,
        "score": 0.0,
        "cases": {"combined_score": 0.0, "sum_radii": 0.0,
                  "target_ratio": 0.0, "validity": 0.0, **_BEHAVIOR_ZERO},
        "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]},
    }
