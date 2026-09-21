"""Evaluator for matmul_tensor: rank-R decomposition of the 4x4 matrix-multiplication
tensor. The evolved run_search() returns factors (U, V, W), each (R, 16); the
score rewards low reconstruction residual at the target rank, with a smooth
landscape (needle 0/1 would kill evolution) plus a discreteness bonus toward the
half-integer lattice that exact fast-matmul schemes live on.

n=4: naive rank 64, Strassen^2 rank 49, AlphaEvolve rank 48 (over C). TARGET_RANK
sets the budget the search must fit within; residual is measured at that rank.
"""

from sciel.candidate_modules import load_candidate
import time
import traceback

import numpy as np

N = 4                      # matrices are N x N
DIM = N * N                # 16 entries per factor
TARGET_RANK = 49           # stage-1 pilot: reproduce Strassen^2; lower later
TOL = 1e-6


def _matmul_tensor(n):
    """T[p, q, r] = 1 iff A-entry p, B-entry q contribute to C-entry r under
    C[i,k] = sum_j A[i,j] B[j,k]. p=(i,j), q=(j,k), r=(i,k) flattened row-major."""
    T = np.zeros((n * n, n * n, n * n))
    for i in range(n):
        for j in range(n):
            for k in range(n):
                p = i * n + j
                q = j * n + k
                r = i * n + k
                T[p, q, r] = 1.0
    return T


_T = _matmul_tensor(N)


def _load(program_path, module_name="program"):
    return load_candidate(program_path, module_name)


def _reconstruct(U, V, W):
    # T_hat[p,q,r] = sum_s U[s,p] V[s,q] W[s,r]
    return np.einsum("sp,sq,sr->pqr", U, V, W)


# --- behavior readout -------------------------------------------------------
# The score is a smooth function of the residual. These describe the SHAPE of the
# decomposition: how evenly the rank-one terms carry the tensor, how sparse the
# factors are, how close the coefficients sit to the half-integer lattice that exact
# schemes live on. Keys are the engine's task-agnostic slots -- the named metrics
# below stay as they are, but the embedder only reads these.

_BEHAVIOR_ZERO = {"b_dispersion": 0.0, "b_boundary_frac": 0.0, "b_contact": 0.0,
                  "b_offset": 0.0, "b_concentration": 0.0, "b_structure": 0.0}


def _behavior(U, V, W):
    try:
        norms = np.linalg.norm(U, axis=1) * np.linalg.norm(V, axis=1) \
            * np.linalg.norm(W, axis=1)
        if norms.size == 0 or not np.isfinite(norms).all():
            return dict(_BEHAVIOR_ZERO)
        mean_n = float(norms.mean())
        dispersion = float(norms.std() / mean_n) if mean_n > 0 else 0.0

        coeffs = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
        if coeffs.size == 0 or not np.isfinite(coeffs).all():
            return dict(_BEHAVIOR_ZERO)
        mag = np.abs(coeffs)
        sparsity = float((mag < 1e-6).mean())
        saturated = float((mag >= 1.0).mean())
        contact = float(np.clip(mag.mean(), 0.0, 1.0))

        total = float(norms.sum())
        k = max(1, norms.size // 4)
        concentration = float(np.sort(norms)[-k:].sum() / total) if total > 0 else 0.0

        out = {"b_dispersion": dispersion / (1.0 + dispersion), "b_boundary_frac": saturated,
               "b_contact": contact, "b_offset": sparsity,
               "b_concentration": concentration,
               # NOT discreteness: that is a factor of `combined` and is already in
               # `cases` under its own name, so using it here would restate the score,
               # which is the failure this readout exists to remove. The number of
               # distinct coefficient magnitudes is structural and absent from the score.
               "b_structure": float(len(np.unique(np.round(mag, 3))) / mag.size)}
        return {key: (float(v) if np.isfinite(v) else 0.0) for key, v in out.items()}
    except (ValueError, FloatingPointError, ZeroDivisionError):
        return dict(_BEHAVIOR_ZERO)


def _score(program_path, module_name):
    program = _load(program_path, module_name)
    if not hasattr(program, "run_search"):
        raise AttributeError("missing run_search")
    start = time.time()
    U, V, W = program.run_search()
    eval_time = time.time() - start
    U, V, W = np.asarray(U, float), np.asarray(V, float), np.asarray(W, float)
    if U.ndim != 2 or U.shape[1] != DIM or V.shape != U.shape or W.shape != U.shape:
        raise ValueError(f"factors must be (R,{DIM}); got {U.shape},{V.shape},{W.shape}")
    if not (np.isfinite(U).all() and np.isfinite(V).all() and np.isfinite(W).all()):
        # isnan alone let inf through: einsum then produced nan, resid was nan, and
        # the run died with ok=True and score=nan inside db.add, where
        # int(score * grid_bins) cannot convert NaN. Every other evaluator in the
        # suite checks finiteness of its score; this one did not.
        raise ValueError("non-finite value in factors")
    rank = U.shape[0]
    resid = float(np.linalg.norm(_reconstruct(U, V, W) - _T))
    # rank validity: at most the target budget
    rank_ok = rank <= TARGET_RANK
    # smooth quality: 1 at residual 0, decaying; exact-at-rank is the goal
    quality = 1.0 / (1.0 + 10.0 * resid)
    # discreteness: coefficients near the half-integer lattice (0, +-0.5, +-1)
    coeffs = np.concatenate([U.ravel(), V.ravel(), W.ravel()])
    lattice_gap = float(np.mean(np.abs(coeffs - np.round(coeffs * 2) / 2)))
    discreteness = 1.0 / (1.0 + 5.0 * lattice_gap)
    exact = resid < TOL and rank_ok
    combined = float(quality * (1.0 if rank_ok else 0.5) * (0.9 + 0.1 * discreteness))
    if not np.isfinite(combined):
        combined = 0.0
    cases = {
        "combined_score": combined,
        "residual": resid,
        "rank": float(rank),
        "rank_ok": 1.0 if rank_ok else 0.0,
        "exact": 1.0 if exact else 0.0,
        "discreteness": discreteness,
    }
    cases.update(_behavior(U, V, W))
    return {
        "ok": True,
        "score": combined,
        "cases": cases,
        "artifacts": {"eval_time": f"{eval_time:.4f}"},
    }


def smoke(program_path):
    try:
        return _score(program_path, "program_smoke")
    except Exception as e:
        return _fail(str(e))


def full(program_path):
    try:
        return _score(program_path, "program_full")
    except Exception as e:
        return _fail(str(e))


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
        "ok": False, "score": 0.0,
        "cases": {"combined_score": 0.0, "residual": 999.0, "rank": 0.0,
                  "rank_ok": 0.0, "exact": 0.0, "discreteness": 0.0,
                  **_BEHAVIOR_ZERO},
        "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]},
    }
