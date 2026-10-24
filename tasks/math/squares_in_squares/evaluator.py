"""Evaluator for Kaggle's Squares in Squares on a slice of 40 n; smoke/full/confirm on the sciel contract.

For each n the candidate packs n unit squares (centre x, y and rotation angle in radians) with disjoint interiors.
The side L = max(max_x - min_x, max_y - min_y) over all corners is translation invariant and the excess is
L / sqrt(n) - 1, the competition's per-n score (lower is better; its total is the sum over n = 1..1000). The bar is
the per-n best of the 12 public submissions of 23 October 2026, which no neighbour transfer improves (total 22.3084;
the leaderboard's best is 22.2248). The slice is the 40 n = k^2 + j, 1 <= j <= k, k >= 10, whose excess lies furthest
above a smooth fit of L - k in (j / k, 1 / k), weighted by 1 / sqrt(n) as in the total.

Each n is checked by the competition's own metric, unchanged, on the decimal repr of each binary64 value, in a fresh
interpreter: a float separating-axis test, then interval arithmetic at 50 to 500 digits for pairs within 1e-8, where an
undecidable pair is rejected, and L measured at 60 digits. A missing or rejected n counts at the bar's excess, and so
does a packing whose L is within 1e-6 of the bar's: closing the gaps of a public packing gains up to a few 1e-9, which
is polish, not a new packing, and worth under 1e-6 on the total.
combined_score = (bar's excess sum) / (counted excess sum) over the slice, above 1 when some n beats the public best;
kaggle_total is the public total with the slice's counted excesses in place of the bar's.

The metric and the packings are located through SCIEL_SQUARES and checked against pinned digests.
"""

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

import ast
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import traceback
import uuid

import numpy as np

from sciel.candidate_modules import load_candidate

ASSETS = Path(__file__).resolve().with_name("assets")
SIS = load_candidate(str(ASSETS / "sis.py"), "sis_evaluator")
SLICE = SIS.SLICE
SEED_L = {n: float(SIS.SEED_L[n]) for n in SLICE}
SEED_EXCESS = {n: SIS.seed_excess(n) for n in SLICE}
# A side within this of the bar's is the bar's own packing, rounded or with its gaps closed.
MARGIN = 1e-6
MAX_COORD, MAX_ANGLE = 1e4, 100.0
WORKERS = 4
FORBIDDEN_NAMES = {"open", "exec", "eval", "compile", "__import__", "globals", "vars", "locals", "getattr", "setattr",
                   "delattr", "breakpoint", "input"}
FORBIDDEN_ATTRIBUTES = {"modules", "system", "popen", "fork", "kill", "_getframe", "f_back", "f_globals", "f_locals",
                        "gi_frame", "tb_frame", "tofile", "save", "savez", "savetxt", "memmap"}
FORBIDDEN_MODULES = {"ctypes", "gc", "inspect", "importlib", "builtins", "subprocess", "multiprocessing", "shutil",
                     "pathlib", "socket", "pickle", "marshal", "signal", "threading", "concurrent", "io", "tempfile"}
CASES = ("combined_score", "excess_sum", "kaggle_total", "n_improved", "n_invalid", "best_gain", "best_gain_n",
         "validity")
_BEHAVIOR_ZERO = {"b_dispersion": 0.0, "b_boundary_frac": 0.0, "b_contact": 0.0,
                  "b_offset": 0.0, "b_concentration": 0.0, "b_structure": 0.0}
# The certifier runs in a fresh isolated interpreter, so nothing a candidate did in this process reaches it.
WORKER = r"""
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("squares_metric", sys.argv[1])
metric = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metric)
out = []
for n, rows in json.load(sys.stdin):
    try:
        out.append(list(metric.evaluate_single_puzzle(n, rows)))
    except Exception as e:
        out.append([n, metric.infeasible_score(n), None, "error: " + str(e)[:200]])
json.dump(out, sys.stdout)
"""


def _load(program_path, module_name="program"):
    return load_candidate(program_path, module_name)


def _check_source(program_path):
    tree = ast.parse(Path(program_path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            bad = [m for m in names if m.split(".")[0] in FORBIDDEN_MODULES]
            if bad:
                raise ValueError(f"import of {bad[0]} is not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ValueError(f"use of {node.id} is not allowed")
        elif isinstance(node, ast.Attribute) and (node.attr in FORBIDDEN_ATTRIBUTES or node.attr.startswith("__")):
            raise ValueError(f"attribute {node.attr} is not allowed")


def _packing(n, value):
    """The candidate's packing for n as (n, 3) binary64, or the reason it is not one."""
    try:
        p = np.array(value)
    except (ValueError, TypeError) as e:
        return None, f"not an array: {str(e)[:80]}"
    if p.dtype.kind not in "fiu":
        return None, f"values must be real numbers, got dtype {p.dtype}"
    p = p.astype(np.float64)
    if p.shape != (n, 3):
        return None, f"expected shape ({n}, 3), got {p.shape}"
    if not np.isfinite(p).all():
        return None, "values must be finite"
    if np.abs(p[:, :2]).max() > MAX_COORD or np.abs(p[:, 2]).max() > MAX_ANGLE:
        return None, f"|x|, |y| must be at most {MAX_COORD:g} and |angle| at most {MAX_ANGLE:g}"
    return p, None


def _metric_path():
    path = SIS._path("metric.py")
    if hashlib.sha256(path.read_bytes()).hexdigest() != SIS.DIGESTS["metric.py"]:
        raise ValueError("metric.py does not match its pinned digest")
    return path


def _certify(packings):
    """{n: (metric score, L text, reason)} from the official metric in WORKERS fresh interpreters."""
    if not packings:
        return {}
    path = str(_metric_path())
    order = sorted(packings, key=lambda n: -n)
    chunks = [[] for _ in range(WORKERS)]
    load = [0] * WORKERS
    for n in order:
        k = load.index(min(load))
        chunks[k].append([n, SIS.rows(n, packings[n])])
        load[k] += n
    procs = [subprocess.Popen([sys.executable, "-I", "-c", WORKER, path], stdin=subprocess.PIPE,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for chunk in chunks if chunk]
    out = {}
    for proc, chunk in zip(procs, [c for c in chunks if c]):
        stdout, stderr = proc.communicate(json.dumps(chunk))
        if proc.returncode:
            raise RuntimeError(f"certifier failed: {stderr[-400:]}")
        for n, score, side_text, reason in json.loads(stdout):
            out[int(n)] = (float(score), side_text, reason)
    return out


def _behavior(gains, counted, invalid, edited):
    """Six bounded descriptors: share of n edited, share improved, mean share of tilted squares, share of the gain
    below n = 300, share of the gain in the best n, share of n rejected or missing. Never raises."""
    try:
        total = sum(g for g in gains.values() if g > 0)
        tilted = np.mean([float((np.abs((p[:, 2] + np.pi / 4) % (np.pi / 2) - np.pi / 4) > 1e-6).mean())
                          for p in counted.values()])
        out = {"b_dispersion": len(edited) / len(SLICE),
               "b_boundary_frac": sum(g > 0 for g in gains.values()) / len(SLICE),
               "b_contact": float(tilted),
               "b_offset": sum(g for n, g in gains.items() if g > 0 and n < 300) / total if total > 0 else 0.0,
               "b_concentration": max(gains.values()) / total if total > 0 else 0.0,
               "b_structure": len(invalid) / len(SLICE)}
        return {k: (min(max(float(v), 0.0), 1.0) if math.isfinite(v) else 0.0) for k, v in out.items()}
    except (ValueError, FloatingPointError, ZeroDivisionError):
        return dict(_BEHAVIOR_ZERO)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _evaluate(program_path, stage):
    source_sha = _source_hash(program_path)
    evaluator_sha = _source_hash(__file__)
    invocation_id = uuid.uuid4().hex
    _check_source(program_path)
    seeds = {n: SIS.seed(n) for n in SLICE}
    _metric_path()
    program = _load(program_path, "program_" + stage)
    if not hasattr(program, "packings"):
        raise AttributeError("missing packings")
    start_time = time.time()
    value = program.packings()
    eval_time = time.time() - start_time
    if not isinstance(value, dict):
        raise ValueError("packings() must return a dict {n: list of n [x, y, angle]}")
    keys = {}
    for key in value:
        if type(key) is int:
            keys[key] = key
        elif type(key) is str and key.isdigit() and type(keys.get(int(key))) is not int:
            keys[int(key)] = key
    candidate, invalid = {}, {}
    for n in SLICE:
        if n not in keys:
            invalid[n] = "missing"
            continue
        p, why = _packing(n, value[keys[n]])
        if p is None:
            invalid[n] = why
        else:
            candidate[n] = p
    checked = _certify(candidate)
    counted, sides, gains, edited, improved = {}, {}, {}, [], []
    for n in SLICE:
        if n in candidate and checked[n][2] != "ok":
            invalid[n] = checked[n][2][:120]
        side = float(checked[n][1]) if n in candidate and n not in invalid else None
        if side is None:
            counted[n], sides[n] = seeds[n], SEED_L[n]
        else:
            counted[n], sides[n] = candidate[n], side
            if not np.array_equal(candidate[n], seeds[n]):
                edited.append(n)
        excess = SEED_EXCESS[n] if abs(sides[n] - SEED_L[n]) < MARGIN else SIS.excess(n, sides[n])
        gains[n] = SEED_EXCESS[n] - excess
        if gains[n] > 0:
            improved.append(n)
    bar = sum(SEED_EXCESS.values())
    total = bar - sum(gains.values())
    best = max(SLICE, key=lambda n: gains[n])
    combined = bar / total
    cases = {"combined_score": combined, "excess_sum": total, "kaggle_total": SIS.MERGED_TOTAL - (bar - total),
             "n_improved": float(len(improved)), "n_invalid": float(len(invalid)), "best_gain": gains[best],
             "best_gain_n": float(best), "validity": 1.0}
    cases.update(_behavior(gains, counted, invalid, edited))
    cases.update({f"gain_{n}": gains[n] for n in SLICE})
    witness = {
        "schema": "sciel-full-output/1",
        "source_sha256": source_sha, "evaluator_sha256": evaluator_sha,
        "invocation_id": invocation_id, "stage": stage, "confirmation": None,
        "input_sha256": None, "seed_sha256": None,
        # Base64 of the little-endian binary64 (x, y, angle) of the packing each n counted, the bar's for a rejected
        # n, keeps the witness exact and under the 1 MiB artifact limit.
        "output": {"packings": {str(n): SIS.encode(counted[n]) for n in SLICE},
                   "side": {str(n): sides[n] for n in SLICE}, "improved": improved,
                   "rejected": {str(n): why for n, why in invalid.items()}},
    }
    artifacts = {"eval_time": f"{eval_time:.4f}", "full_output": _canonical(witness)}
    if invalid:
        artifacts["rejected"] = "; ".join(f"{n}: {why}" for n, why in sorted(invalid.items()))[:2000]
    if source_sha != _source_hash(program_path) or evaluator_sha != _source_hash(__file__):
        raise ValueError("source or evaluator changed during invocation")
    return {"ok": True, "score": combined, "cases": cases, "artifacts": artifacts}


def smoke(program_path):
    try:
        return _evaluate(program_path, "smoke")
    except Exception as e:
        return _fail(str(e))


def full(program_path):
    try:
        return _evaluate(program_path, "full")
    except Exception as e:
        return _fail(str(e))


def confirm(program_path, n=3):
    """Re-run the program n times and keep the worst result; a record should survive its own repetition."""
    n = max(1, int(n or 1))
    runs = [full(program_path) for _ in range(n)]
    failures = [r for r in runs if not r.get("ok")]
    selected = (next(i for i, result in enumerate(runs) if not result.get("ok"))
                if failures else min(range(n), key=lambda i: runs[i]["score"]))
    worst = dict(runs[selected])
    worst["cases"] = {**worst["cases"], "n_trials": float(n)}
    worst["artifacts"] = dict(worst.get("artifacts", {}))
    selection = {"trials": n, "selected_trial": selected, "selection": "first_failure_else_lowest_score"}
    worst["artifacts"]["confirmation_selection"] = _canonical(selection)
    if "full_output" in worst["artifacts"]:
        witness = json.loads(worst["artifacts"]["full_output"])
        witness["confirmation"] = selection
        worst["artifacts"]["full_output"] = _canonical(witness)
    return worst


def _fail(msg):
    return {
        "ok": False,
        "score": 0.0,
        "cases": {**{k: 0.0 for k in CASES}, **_BEHAVIOR_ZERO, **{f"gain_{n}": 0.0 for n in SLICE}},
        "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]},
    }
