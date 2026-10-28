"""Evaluator for maximal volume polyhedra: n points on the unit sphere S^2 whose convex hull has the largest volume,
for n = 4..130 of the table of Hardin, Sloane and Smith (assets/mv_data.py); smoke/full/confirm on the sciel contract.

For each n the candidate gives n nonzero points in R^3; only their directions count. The scored polytope is the hull
of the directions rounded inward: each unit vector is computed in binary64 and shrunk by 1 - 2^-52 until its exact
squared length is at most 1, so every vertex lies in the closed unit ball, and since a hull volume is convex in each
vertex no such polytope beats the best inscribed one. Its volume is computed exactly: the directions are integer
vectors over a power of two, Qhull's triangles are checked to bound every point (orientations in floating point with
Shewchuk's error bound, the uncertain ones as integers; otherwise the hull is rebuilt by exact beneath-beyond), and the
volume is the rational sum of a . (b x c) / 6 over them. An instance scores V / V_bar and combined_score is the mean,
so a score above 1 means some polytope beats the bar. The bar is the table's configuration after local search
(assets/mv_best.py): L-BFGS on the volume and basin hopping, about four minutes per instance, which found no basin
above any published configuration, so the bar is the table up to binary64 rounding. A further minute of basin hopping
gains at most 1.8e-16 of the volume, so n_improved counts V >= V_bar (1 + 1e-11), far above that creep and the
floating error; n_beats_published counts V >= the table's displayed volume + 1e-12, a gain the 12-decimal table
would show.
"""

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

from sciel.candidate_modules import load_candidate
import base64
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import time
import traceback
import uuid

import numpy as np
from scipy.spatial import ConvexHull, QhullError

ASSETS = Path(__file__).resolve().with_name("assets")
DATA = load_candidate(str(ASSETS / "mv_data.py"), "mv_data_evaluator")
BEST = load_candidate(str(ASSETS / "mv_best.py"), "mv_best_evaluator")
INSTANCES = tuple(DATA.INSTANCES)
SIZE = {k: int(k) for k in INSTANCES}
MARGIN = Fraction(1, 10 ** 11)
SITE_MARGIN = Fraction(1, 10 ** 12)
BAR = {k: Fraction(BEST.VOLUME[k]) for k in INSTANCES}
PUBLISHED = {k: Fraction(DATA.VOLUME[k]) for k in INSTANCES}


def _load(program_path, module_name="program"):
    return load_candidate(program_path, module_name)


def _directions(p):
    top = np.abs(p).max(axis=1, keepdims=True)
    if (top == 0).any():
        raise ValueError("a point is the zero vector")
    u = p / top
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    for i in np.nonzero((u * u).sum(axis=1) > 1 - 1e-15)[0]:
        while sum(Fraction(v) ** 2 for v in u[i].tolist()) > 1:
            u[i] *= 1 - 2.0 ** -52
    return u


def _integers(u):
    ratios = [[v.as_integer_ratio() for v in row] for row in u.tolist()]
    q = max(b for row in ratios for _, b in row)
    return [tuple(a * (q // b) for a, b in row) for row in ratios], q.bit_length() - 1


def _det(a, b, c, d):
    ax, ay, az = a[0] - d[0], a[1] - d[1], a[2] - d[2]
    bx, by, bz = b[0] - d[0], b[1] - d[1], b[2] - d[2]
    cx, cy, cz = c[0] - d[0], c[1] - d[1], c[2] - d[2]
    return ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx)


def _qhull_faces(u):
    hull = ConvexHull(u)
    f = hull.simplices.copy()
    a, b, c = u[f[:, 0]], u[f[:, 1]], u[f[:, 2]]
    flip = np.einsum("ij,ij->i", np.cross(b - a, c - a), hull.equations[:, :3]) < 0
    f[flip, 1], f[flip, 2] = f[flip, 2], f[flip, 1].copy()
    return f


def _certified(u, ints, f):
    """Whether the triangles f are exactly a closed surface with every point on their inner side or on them."""
    edges = {}
    for a, b, c in f.tolist():
        for e in ((a, b), (b, c), (c, a)):
            if e in edges:
                return False
            edges[e] = True
    if any((b, a) not in edges for a, b in edges):
        return False
    A, B, C = u[f[:, 0]][:, None], u[f[:, 1]][:, None], u[f[:, 2]][:, None]
    ad, bd, cd = A - u[None], B - u[None], C - u[None]
    det = np.einsum("fnk,fnk->fn", ad, np.cross(bd, cd))
    aa, ba, ca = np.abs(ad), np.abs(bd), np.abs(cd)
    perm = (aa[..., 0] * (ba[..., 1] * ca[..., 2] + ba[..., 2] * ca[..., 1])
            + aa[..., 1] * (ba[..., 2] * ca[..., 0] + ba[..., 0] * ca[..., 2])
            + aa[..., 2] * (ba[..., 0] * ca[..., 1] + ba[..., 1] * ca[..., 0]))
    # Shewchuk's bound: a floating orientation of binary64 points is off by less than 7.8e-16 of the permanent.
    if (det < -2e-15 * perm).any():
        return False
    for i, j in zip(*np.nonzero(np.abs(det) <= 2e-15 * perm)):
        a, b, c = f[i]
        if j not in (a, b, c) and _det(ints[a], ints[b], ints[c], ints[j]) < 0:
            return False
    return True


def _exact_faces(ints):
    p0 = ints[0]
    rest = range(1, len(ints))
    sub = lambda a, b: (a[0] - b[0], a[1] - b[1], a[2] - b[2])
    cross = lambda a, b: (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])
    i1 = next((i for i in rest if ints[i] != p0), None)
    i2 = None if i1 is None else next(
        (i for i in rest if cross(sub(ints[i1], p0), sub(ints[i], p0)) != (0, 0, 0)), None)
    i3 = None if i2 is None else next((i for i in rest if _det(p0, ints[i1], ints[i2], ints[i]) != 0), None)
    if i3 is None:
        raise ValueError("the hull is flat")
    a, b, c, d = 0, i1, i2, i3
    if _det(ints[a], ints[b], ints[c], ints[d]) < 0:
        b, c = c, b
    faces = {(a, b, c), (a, d, b), (b, d, c), (c, d, a)}
    for i in range(len(ints)):
        if i in (a, b, c, d):
            continue
        p = ints[i]
        seen = [t for t in faces if _det(ints[t[0]], ints[t[1]], ints[t[2]], p) < 0]
        if not seen:
            continue
        edges = {e for t in seen for e in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
        faces.difference_update(seen)
        faces.update((x, y, i) for x, y in edges if (y, x) not in edges)
    return np.array(sorted(faces))


def volume(p) -> Fraction:
    """The exact volume of the hull of the inward-rounded directions of the (n, 3) binary64 points. Raises
    ValueError on a zero point, two points with the same direction or a flat hull."""
    u = _directions(p)
    if len(np.unique(u, axis=0)) < len(u):
        raise ValueError("two points have the same direction")
    ints, s = _integers(u)
    try:
        f = _qhull_faces(u)
    except (QhullError, ValueError):
        f = None
    if f is None or not _certified(u, ints, f):
        f = _exact_faces(ints)
    total = sum(_det(ints[a], ints[b], ints[c], (0, 0, 0)) for a, b, c in f.tolist())
    if total <= 0:
        raise ValueError("the hull is flat")
    return Fraction(total, 6 << (3 * s))


def _configs(value):
    if not isinstance(value, dict):
        raise ValueError('polyhedra() must return a dict {"n": list of n points in R^3}')
    out = {}
    for k in INSTANCES:
        if k not in value:
            raise ValueError(f"missing {k}")
        p = np.array(value[k], dtype=np.float64)
        if p.shape != (SIZE[k], 3) or not np.isfinite(p).all():
            raise ValueError(f"{k}: expected {SIZE[k]} finite points in R^3, got shape {p.shape}")
        out[k] = p
    return out


# --- behavior readout -------------------------------------------------------
# The score is one mean. These describe WHERE the polytopes stand against the bar, so runs that win in different
# ranges of n land in different cells of the diversity map.

_BEHAVIOR_ZERO = {"b_dispersion": 0.0, "b_boundary_frac": 0.0, "b_contact": 0.0,
                  "b_offset": 0.0, "b_concentration": 0.0, "b_structure": 0.0}


def _behavior(ratios):
    """Six bounded descriptors of the per-instance ratios: their spread, share at or above the bar, mean gain in
    the first, middle and last third of the instances (ordered by n), and the share within 1e-12 of the bar.
    Never raises."""
    try:
        r = np.array([ratios[k] for k in INSTANCES])
        third = np.arange(len(r)) * 3 // len(r)
        squash = lambda x: float(0.5 + 0.5 * np.tanh(x * 1e6))
        part = lambda m: squash(r[m].mean() - 1) if m.any() else 0.5
        out = {"b_dispersion": float(min(r.std() * 1e6, 1.0)), "b_boundary_frac": float((r >= 1.0).mean()),
               "b_contact": part(third == 0), "b_offset": part(third == 1),
               "b_concentration": part(third == 2), "b_structure": float((np.abs(r - 1) < 1e-12).mean())}
        return {k: (min(max(v, 0.0), 1.0) if np.isfinite(v) else 0.0) for k, v in out.items()}
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
    program = _load(program_path, "program_" + stage)
    if not hasattr(program, "polyhedra"):
        raise AttributeError("missing polyhedra")
    start_time = time.time()
    configs = _configs(program.polyhedra())
    eval_time = time.time() - start_time
    vol, ratios, improved, beats_published = {}, {}, [], 0
    for k, p in configs.items():
        try:
            exact = volume(p)
        except Exception as e:
            raise ValueError(f"{k}: {e}") from None
        vol[k] = float(exact)
        ratios[k] = float(exact / BAR[k])
        if exact >= BAR[k] * (1 + MARGIN):
            improved.append(k)
        beats_published += exact >= PUBLISHED[k] + SITE_MARGIN
    best = max(INSTANCES, key=lambda k: ratios[k])
    combined = float(np.mean([ratios[k] for k in INSTANCES]))
    cases = {"combined_score": combined, "n_improved": float(len(improved)),
             "best_gain_volume": vol[best] - float(BAR[best]), "best_gain_n": float(SIZE[best]),
             "n_beats_published": float(beats_published), "validity": 1.0}
    cases.update(_behavior(ratios))
    witness = {
        "schema": "sciel-full-output/1",
        "source_sha256": source_sha, "evaluator_sha256": evaluator_sha,
        "invocation_id": invocation_id, "stage": stage, "confirmation": None,
        "input_sha256": None, "seed_sha256": None,
        # Base64 of the little-endian binary64 rows keeps the witness exact and under the 1 MiB artifact limit.
        "output": {"points": {k: base64.b64encode(configs[k].astype("<f8").tobytes()).decode() for k in INSTANCES},
                   "volume": vol, "improved": improved},
    }
    artifacts = {"eval_time": f"{eval_time:.4f}", "full_output": _canonical(witness)}
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
        "cases": {"combined_score": 0.0, "n_improved": 0.0, "best_gain_volume": 0.0, "best_gain_n": 0.0,
                  "n_beats_published": 0.0, "validity": 0.0, **_BEHAVIOR_ZERO},
        "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]},
    }
