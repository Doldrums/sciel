"""Evaluator for circle_packing (n=26); smoke/full/confirm on the sciel contract."""

from sciel.candidate_modules import load_candidate
import time
import traceback

import numpy as np

TARGET_VALUE = 2.635
N_CIRCLES = 26


def _load(program_path, module_name="program"):
    return load_candidate(program_path, module_name)


def _validate_packing(centers, radii):
    # Validate represented values in binary64, independently of candidate dtype.
    # Low-precision arithmetic can otherwise round a real violation away.
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
    except (TypeError, ValueError):
        return repr(value)[:80]


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


def smoke(program_path):
    try:
        centers, radii, _reported = _run(program_path, "program_smoke")
    except Exception as e:
        return _fail(str(e))
    shape_valid = centers.shape == (N_CIRCLES, 2) and radii.shape == (N_CIRCLES,)
    if not shape_valid:
        return _fail(f"Invalid shapes: centers={centers.shape}, radii={radii.shape}")
    valid = _validate_packing(centers, radii)
    actual_sum = float(np.sum(radii)) if valid else 0.0
    target_ratio = float(actual_sum / TARGET_VALUE) if valid else 0.0
    combined = target_ratio if valid else 0.0
    cases = {
        "validity": 1.0 if valid else 0.0,
        "sum_radii": actual_sum,
        "target_ratio": target_ratio,
        "combined_score": float(combined),
    }
    cases.update(_behavior(centers, radii, valid))
    return {
        "ok": True,
        "score": float(combined),
        "cases": cases,
        "artifacts": {},
    }


def full(program_path):
    start_time = time.time()
    try:
        centers, radii, reported_sum = _run(program_path, "program_full")
    except Exception as e:
        return _fail(str(e))
    if np.isnan(centers).any() or np.isnan(radii).any():
        return {
            "ok": True,
            "score": 0.0,
            "cases": {"sum_radii": 0.0, "target_ratio": 0.0, "validity": 0.0,
                      "combined_score": 0.0, **_BEHAVIOR_ZERO},
            "artifacts": {"eval_time": f"{time.time() - start_time:.4f}"},
        }
    valid = _validate_packing(centers, radii)
    shape_valid = centers.shape == (N_CIRCLES, 2) and radii.shape == (N_CIRCLES,)
    if not shape_valid:
        valid = False
    sum_radii = float(np.sum(radii)) if valid else 0.0
    target_ratio = float(sum_radii / TARGET_VALUE) if valid else 0.0
    validity = 1.0 if valid else 0.0
    combined = float(target_ratio * validity)
    eval_time = time.time() - start_time
    cases = {
        "sum_radii": sum_radii,
        "target_ratio": target_ratio,
        "validity": validity,
        "combined_score": combined,
    }
    cases.update(_behavior(centers, radii, valid))
    return {
        "ok": True,
        "score": combined,
        "cases": cases,
        "artifacts": {"eval_time": f"{eval_time:.4f}",
                      # the program's own claim, recorded but never trusted. Coercing
                      # it outside the try made `return centers, radii, None` pass
                      # smoke and throw in full.
                      "reported_sum": _as_text(reported_sum)},
    }


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
    worst = failures[0] if failures else min(runs, key=lambda r: r["score"])
    worst = dict(worst)
    worst["cases"] = {**worst["cases"], "n_trials": float(n)}
    return worst


def _fail(msg):
    return {
        "ok": False,
        "score": 0.0,
        "cases": {"combined_score": 0.0, "sum_radii": 0.0,
                  "target_ratio": 0.0, "validity": 0.0, **_BEHAVIOR_ZERO},
        "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]},
    }
