"""Evaluator for tess_transits: a transit detector on real TESS light curves, scored by injection and recovery.

The curves are TESS-SPOC 200-s FFI photometry of sector 61 binned to 10 minutes, from stars with no known planet,
candidate or eclipsing binary, less the few whose curves already carry a strong periodic signal. The evaluator
injects, from a fixed seed per stage, transiting planets into 40% of them and eclipsing binaries (V-shaped, with a
secondary eclipse, or with unequal odd and even eclipses) into 20%, leaving 40% clean. The candidate sees one curve at a time as time, flux and flux_err, in a freshly loaded module,
and returns up to three (period, epoch, duration, score). The threshold is the score that 2% of the clean curves
exceed; a planet counts as recovered when a signal above it matches its period (or half or twice it) and epoch.
h = recall * (1 - 0.5 * share of binaries above the threshold), and combined_score = 0.01 + h. confirm also ranks
real confirmed planets against real false positives and binaries (real_auc) on curves the search never sees.

Data are located through SCIEL_TESS_TRANSITS and checked against pinned digests before the candidate is imported;
the variable is removed while candidate code runs, and the source may import only numeric modules and the task's
own lc asset.
"""

import ast
import hashlib
import json
import math
import os
from pathlib import Path
import time
import traceback
import uuid

import numpy as np

from sciel.candidate_modules import load_candidate

ENV = "SCIEL_TESS_TRANSITS"
DIGESTS = {
    "tier_smoke.npz": "5efab82b1e5571dbfee310284b6f7142ab9a38355aae14c84af0ec885ea9d538",
    "tier_full.npz": "af0b101aa009f557d9a859eee10179888a6732eb7ff99cc51e8ea9543da6835c",
    "tier_confirm.npz": "247ff65bcfd18d7fe7206386e0603c022fbb0115f34c7f7487654a6b1e5980bc",
    "tier_real.npz": "7944ae433d5ee648145f3b4a943d1b0c0ca7e535cdef4379b0f0be71f14a7077",
}
SEEDS = {"smoke": 11, "full": 12, "confirm": 13}
FALSE_ALARM = 0.02
# Stars whose curves already carry a periodic signal with box SNR above 20 under the seed's search (eclipsing
# binaries, pulsators or spotted stars missing from the catalogs); they would set the false-alarm threshold alone.
EXCLUDED = {
    891518, 25152787, 31006639, 38461950, 51143654, 54872226, 55384576, 55526173, 71152716, 90459374, 96320797,
    140033128, 140758953, 148128489, 149102762, 149251336, 149738606, 156712369, 167303776, 170205751,
    177285194, 231617548, 238039088, 238734686, 260659412, 262532832, 262578204, 269797536, 270447145,
    270578753, 270681595, 271888205, 271975437, 277100604, 287400929, 307210811, 308749877, 323711549,
    339770560, 339891469, 349572614, 382578445, 391746191, 391900414, 443994180, 452875979, 455132633,
    455139300, 456797221, 813529007,
}
ALLOWED_MODULES = {"numpy", "scipy", "math", "itertools", "functools", "collections", "heapq", "bisect", "time",
                   "warnings", "dataclasses", "typing", "sciel"}
FORBIDDEN_NAMES = {"open", "__import__", "eval", "exec", "compile", "globals", "vars", "locals", "getattr",
                   "setattr", "delattr", "breakpoint", "input", "memoryview", "load", "save", "savez", "fromfile",
                   "tofile", "memmap", "loadtxt", "genfromtxt", "savetxt", "DataSource", "ctypeslib", "population"}
CASES = ("combined_score", "recall", "eb_rate", "threshold", "real_auc", "seconds_per_curve", "validity")


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _tier(name):
    root = Path(os.environ.get(ENV, ""))
    path = root / f"tier_{name}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"set {ENV} to the tess_transits tier files")
    if _sha256(path) != DIGESTS[path.name]:
        raise ValueError(f"{path.name} does not match its pinned digest")
    z = np.load(path)
    off = z["offsets"]
    keep = [i for i, tic in enumerate(z["tic_id"]) if name == "real" or int(tic) not in EXCLUDED]
    curves = [(z["time"][off[i]:off[i + 1]].astype(np.float64), z["flux"][off[i]:off[i + 1]].astype(np.float64),
               z["flux_err"][off[i]:off[i + 1]].astype(np.float64)) for i in keep]
    return curves, z["label"].astype(int)[keep]


def _check_source(program_path):
    tree = ast.parse(Path(program_path).read_text())
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            bad = [n for n in names if n.split(".")[0] not in ALLOWED_MODULES]
            if bad or (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("sciel")
                       and node.module != "sciel.task"):
                raise ValueError(f"import of {(bad or [node.module])[0]} is not allowed; data arrive as arguments")
        elif isinstance(node, (ast.Global, ast.Nonlocal)):
            raise ValueError("global and nonlocal are not allowed")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            raise ValueError(f"use of {node.id} is not allowed")
        elif isinstance(node, ast.Attribute) and (node.attr in FORBIDDEN_NAMES or node.attr.startswith("__")):
            raise ValueError(f"attribute {node.attr} is not allowed")
        elif (isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", None)) == "asset"
              and not (len(node.args) == 1 and isinstance(node.args[0], ast.Constant) and node.args[0].value == "lc")):
            raise ValueError("asset() may load only the task's 'lc' module")


# --- injections -------------------------------------------------------------

def _trapezoid(phase, duration, ingress):
    """Fractional depth (0..1) of a trapezoid centred on phase 0, total width duration, ramps of length ingress."""
    x = np.abs(phase)
    return np.clip((duration / 2.0 - x) / max(ingress, 1e-9), 0.0, 1.0)


def _fold(t, period, epoch):
    return (t - epoch + period / 2.0) % period - period / 2.0


def _draw(rng, t, flux):
    """Period, epoch, duration and depth for a box signal with SNR uniform in 8..20 and at least two events."""
    sigma = 1.4826 * np.median(np.abs(flux - np.median(flux)))
    while True:
        period = float(np.exp(rng.uniform(np.log(0.7), np.log(13.0))))
        epoch = float(t[0] + rng.uniform(0.0, period))
        rho = float(np.exp(rng.uniform(np.log(0.3), np.log(3.0))))
        duration = float(13.0 / 24.0 * (period / 365.25) ** (1.0 / 3.0) * rho ** (-1.0 / 3.0))
        inside = int((np.abs(_fold(t, period, epoch)) < duration / 2.0).sum())
        events = len(np.unique(np.floor((t[np.abs(_fold(t, period, epoch)) < duration / 2.0] - epoch) / period + 0.5)))
        if inside >= 6 and events >= 2:
            break
    depth = float(rng.uniform(8.0, 20.0) * sigma / math.sqrt(inside))
    return period, epoch, duration, depth


def _inject(curves, seed):
    """Per curve: kind (1 planet, 2 binary, 0 clean), its parameters, and the flux the candidate sees."""
    rng = np.random.default_rng(seed)
    n = len(curves)
    kinds = np.zeros(n, dtype=int)
    order = rng.permutation(n)
    kinds[order[:int(0.4 * n)]] = 1
    kinds[order[int(0.4 * n):int(0.6 * n)]] = 2
    shown, truth = [], []
    for (t, f, e), kind in zip(curves, kinds):
        if kind == 0:
            shown.append((t, f.copy(), e)); truth.append(None)
            continue
        period, epoch, duration, depth = _draw(rng, t, f)
        if kind == 1:
            b = rng.uniform(0.0, 0.9)
            model = depth * _trapezoid(_fold(t, period, epoch), duration, duration * (0.1 + 0.4 * b * b))
        else:
            style = int(rng.integers(3))
            ramp = duration / 2.0 if style == 0 else duration * 0.2
            model = depth * _trapezoid(_fold(t, period, epoch), duration, ramp)
            if style in (0, 1):
                model += rng.uniform(0.15, 0.7) * depth * _trapezoid(_fold(t, period, epoch + period / 2.0),
                                                                    duration, ramp)
            if style == 2:
                odd = np.floor((t - epoch) / period + 0.5).astype(int) % 2 == 1
                model[odd] *= 1.0 - rng.uniform(0.2, 0.5)
        shown.append((t, f * (1.0 - model), e))
        truth.append((period, epoch, duration))
    return kinds, truth, shown


def _matches(signal, truth):
    period, epoch, duration = truth
    p, t0 = float(signal[0]), float(signal[1])
    if not any(abs(p / period / r - 1.0) < 0.005 for r in (1.0, 0.5, 2.0)):
        return False
    base = min(p, period)
    return abs((t0 - epoch + base / 2.0) % base - base / 2.0) < duration / 2.0 + 0.02


# --- running the candidate ----------------------------------------------------

def _signals(value):
    out = []
    for item in list(value)[:3]:
        p, t0, d, s = (float(v) for v in item)
        if not all(math.isfinite(v) for v in (p, t0, d, s)) or p <= 0 or d <= 0:
            raise ValueError(f"signal {item!r} must be finite with positive period and duration")
        out.append((p, t0, d, s))
    return out


def _run(program_path, shown, stage):
    environ = os.environ.pop(ENV, None)
    try:
        results, started = [], time.perf_counter()
        for i, (t, f, e) in enumerate(shown):
            program = load_candidate(program_path, f"program_{stage}_{i}")
            results.append(_signals(program.detect(t.copy(), f.copy(), e.copy())))
        return results, (time.perf_counter() - started) / max(len(shown), 1)
    finally:
        if environ is not None:
            os.environ[ENV] = environ


def _score(kinds, truth, results):
    top = np.array([max((s[3] for s in r), default=-np.inf) for r in results])
    clean = np.sort(top[kinds == 0])[::-1]
    k = max(1, math.ceil(FALSE_ALARM * clean.size))
    threshold = float(clean[k - 1]) if clean.size else -np.inf
    planets = np.flatnonzero(kinds == 1)
    hit = np.array([any(s[3] > threshold and _matches(s, truth[i]) for s in results[i]) for i in planets])
    recall = float(hit.mean()) if hit.size else 0.0
    eb_rate = float((top[kinds == 2] > threshold).mean()) if (kinds == 2).any() else 0.0
    periods = np.array([truth[i][0] for i in planets])
    return recall * (1.0 - 0.5 * eb_rate), recall, eb_rate, threshold, hit, periods


def _auc(scores, labels):
    pos, neg = scores[labels == 1], scores[labels == 0]
    if not pos.size or not neg.size:
        return 0.0
    greater = (pos[:, None] > neg[None, :]).mean()
    ties = (pos[:, None] == neg[None, :]).mean()
    return float(greater + 0.5 * ties)


def _behavior(hit, periods, eb_rate, results):
    """Six bounded descriptors: recall at short and long periods, binary rate, mean signals returned, share of
    curves with no signal, and the spread of recall between period halves."""
    try:
        short = hit[periods < 3.0].mean() if (periods < 3.0).any() else 0.0
        long = hit[periods >= 3.0].mean() if (periods >= 3.0).any() else 0.0
        counts = np.array([len(r) for r in results])
        return {"b_dispersion": float(abs(short - long)), "b_boundary_frac": float(short),
                "b_contact": float(long), "b_offset": float(eb_rate), "b_concentration": float(counts.mean() / 3.0),
                "b_structure": float((counts == 0).mean())}
    except (ValueError, FloatingPointError, ZeroDivisionError):
        return {k: 0.0 for k in ("b_dispersion", "b_boundary_frac", "b_contact", "b_offset", "b_concentration",
                                 "b_structure")}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def _evaluate(program_path, stage):
    source_sha = hashlib.sha256(Path(program_path).read_bytes()).hexdigest()
    evaluator_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    _check_source(program_path)
    tier = {"smoke": "smoke", "full": "full", "confirm": "confirm"}[stage]
    curves, _ = _tier(tier)
    real = _tier("real") if stage == "confirm" else None
    kinds, truth, shown = _inject(curves, SEEDS[stage])
    results, per_curve = _run(program_path, shown, stage)
    h, recall, eb_rate, threshold, hit, periods = _score(kinds, truth, results)
    real_auc = 0.0
    if real is not None:
        real_results, _ = _run(program_path, real[0], stage + "_real")
        real_top = np.array([max((s[3] for s in r), default=-np.inf) for r in real_results])
        real_auc = _auc(real_top, real[1])
    combined = 0.01 + h
    cases = {"combined_score": combined, "recall": recall, "eb_rate": eb_rate,
             "threshold": threshold if math.isfinite(threshold) else 0.0, "real_auc": real_auc,
             "seconds_per_curve": per_curve, "validity": 1.0}
    cases.update(_behavior(hit, periods, eb_rate, results))
    witness = {"schema": "sciel-full-output/1", "source_sha256": source_sha, "evaluator_sha256": evaluator_sha,
               "invocation_id": uuid.uuid4().hex, "stage": stage, "confirmation": None, "input_sha256": None,
               "seed_sha256": None, "output": {"recall": recall, "eb_rate": eb_rate, "real_auc": real_auc,
                                               "recalled_by_period": [[float(p), bool(x)] for p, x in zip(periods, hit)]}}
    return {"ok": True, "score": combined, "cases": cases, "artifacts": {"full_output": _canonical(witness)}}


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


def confirm(program_path, n=1):
    """The held-out curves with their own injections, and the real-object check; the detector is deterministic."""
    try:
        result = _evaluate(program_path, "confirm")
        result["cases"]["n_trials"] = 1.0
        return result
    except Exception as e:
        return _fail(str(e))


def _fail(msg):
    return {"ok": False, "score": 0.0,
            "cases": {**{k: 0.0 for k in CASES}, **{k: 0.0 for k in ("b_dispersion", "b_boundary_frac", "b_contact",
                                                                    "b_offset", "b_concentration", "b_structure")}},
            "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]}}
