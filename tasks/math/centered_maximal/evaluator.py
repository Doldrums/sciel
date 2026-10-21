"""Evaluator for the centred Hardy-Littlewood maximal constant over squares in the plane, from below; smoke/full/confirm
on the sciel contract.

c_2 is the least constant with alpha |{Mf >= alpha}| <= c_2 ||f||_1 for every f in L^1(R^2), where Mf(x) is the
supremum of the averages of |f| over closed axis-parallel squares centred at x (Tao's optimization constants,
problem 47a). The candidate gives a periodic measure: a period box [0, P_x) x [0, P_y) and finitely many atoms
(x, y, m) in it, repeated with periods P_x and P_y. A point z has M mu(z) >= 1 as soon as some finite set S of atoms
fits in a square of side L = sqrt(m(S)) centred at z, that is z lies in the rectangle
[max_S x - L/2, min_S x + L/2] x [max_S y - L/2, min_S y + L/2]. The evaluator takes S over the atoms of the 3 x 3
block of periods around the box, as the atoms inside an axis-parallel box (enlarging S to the atoms of its bounding
box only grows the rectangle), and measures the union of the rectangles inside the period box. Every rectangle is
rounded inward to the grid 2**-GRID (L/2 downward, through an integer square root), so the union is measured exactly
in integers and is a subset of {M mu >= 1}; score = area / mass of the box is then a lower bound on c_2. Truncating
the lattice to (2N + 1)^2 periods keeps every witness of the inner (2N - 1)^2 periods, and smearing the atoms over
small squares makes the measure integrable at a loss that vanishes with the smearing; this is Lin's limiting
argument (github.com/CoolRmal/centered-maximal-constant, docs/PROOF.md sections 5-6), which uses nothing about his
measure beyond witnesses drawn from a bounded window. The score is not the exact level-set area: witnesses from
outside the window and level sets met only at the supremum are not counted, which can only lower it.
"""

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

from sciel.candidate_modules import load_candidate
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import time
import traceback
import uuid

import numpy as np

# Lin's lattice (2026): Phi = (2hV - 4ab)/(1 + w) = 1.68550999335552518..., Lean-checked lower bound on c_2.
REFERENCE = 1.6855099933555252
MAX_ATOMS = 12
# Period sides within [1/64, 1024] keep the grid rounding below 1e-13 of the box and the integers within int64.
MIN_SIDE = Fraction(1, 64)
MAX_SIDE = Fraction(1024)
# Masses within a factor 2**40 of each other, and every number a rational with denominator at most 2**96.
MASS_RATIO = 2 ** 40
MAX_DENOMINATOR = 2 ** 96
GRID = 50


def _load(program_path, module_name="program"):
    return load_candidate(program_path, module_name)


def _number(value, name):
    """An exact rational from an int, a binary64 float, a Fraction or a string such as '3/7' or '0.125'."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a number, got a boolean")
    if isinstance(value, (int, np.integer)):
        q = Fraction(int(value))
    elif isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise ValueError(f"{name} must be finite")
        q = Fraction(float(value))
    elif isinstance(value, Fraction):
        q = value
    elif isinstance(value, str):
        try:
            q = Fraction(value)
        except (ValueError, ZeroDivisionError):
            raise ValueError(f"{name} is not a rational number: {value!r}") from None
    else:
        raise ValueError(f"{name} must be an int, float, Fraction or string, got {type(value).__name__}")
    if q.denominator > MAX_DENOMINATOR:
        raise ValueError(f"{name} has a denominator above 2**96")
    return q


def measure(result):
    """((P_x, P_y), [(x, y, m), ...]) as Fractions, after the validity checks."""
    if not isinstance(result, (tuple, list)) or len(result) != 2:
        raise ValueError("periodic_measure() must return a pair (period, atoms)")
    period, atoms = result
    if not isinstance(period, (tuple, list, np.ndarray)) or len(period) != 2:
        raise ValueError("period must be a pair (P_x, P_y)")
    px, py = (_number(v, "period") for v in period)
    for side in (px, py):
        if not MIN_SIDE <= side <= MAX_SIDE:
            raise ValueError("each period side must lie in [1/64, 1024]")
    if not isinstance(atoms, (tuple, list, np.ndarray)) or not 1 <= len(atoms) <= MAX_ATOMS:
        raise ValueError(f"atoms must be a list of 1..{MAX_ATOMS} triples (x, y, mass)")
    out = []
    for atom in atoms:
        if not isinstance(atom, (tuple, list, np.ndarray)) or len(atom) != 3:
            raise ValueError("every atom must be a triple (x, y, mass)")
        x, y, m = _number(atom[0], "x"), _number(atom[1], "y"), _number(atom[2], "mass")
        if not (0 <= x < px and 0 <= y < py):
            raise ValueError("every atom must lie in the period box [0, P_x) x [0, P_y)")
        if m <= 0:
            raise ValueError("every mass must be positive")
        out.append((x, y, m))
    if len({(x, y) for x, y, _ in out}) != len(out):
        raise ValueError("two atoms share a position; merge their masses")
    masses = [m for _, _, m in out]
    if max(masses) > MASS_RATIO * min(masses):
        raise ValueError("the masses must lie within a factor 2**40 of each other")
    return (px, py), out


def _window(period, atoms):
    px, py = period
    return [(x + i * px, y + j * py, m) for i in (-1, 0, 1) for j in (-1, 0, 1) for x, y, m in atoms]


def _ceil(q):
    return -((-q.numerator) // q.denominator)


def _floor(q):
    return q.numerator // q.denominator


def _maximal(r, slack=0.0):
    """Indices of the rows of r (x0, x1, y0, y1) not inside another row grown by slack; of equal rows the first is
    kept. Dropping a rectangle can only shrink the union, so a loose slack never overstates the bound."""
    order = np.argsort(-(r[:, 1] - r[:, 0]).astype(float) * (r[:, 3] - r[:, 2]).astype(float), kind="stable")
    kept = []
    for i in order:
        if kept:
            k = r[kept]
            if ((k[:, 0] - slack <= r[i, 0]) & (k[:, 1] + slack >= r[i, 1]) & (k[:, 2] - slack <= r[i, 2])
                    & (k[:, 3] + slack >= r[i, 3])).any():
                continue
        kept.append(i)
    return sorted(kept)


def _boxes(period, win, X, Y):
    """Atom boxes [X_i, X_j] x [Y_k, Y_l] with an atom on each edge whose witness square of side sqrt(mass) holds
    them and whose rectangle meets the period box, with that rectangle in floating point, the maximal ones only."""
    xi = {x: i for i, x in enumerate(X)}
    yi = {y: i for i, y in enumerate(Y)}
    ax = np.array([xi[x] for x, _, _ in win])
    ay = np.array([yi[y] for _, y, _ in win])
    am = np.array([float(m) for _, _, m in win])
    fx, fy = np.array([float(v) for v in X]), np.array([float(v) for v in Y])
    fpx, fpy = float(period[0]), float(period[1])
    total = float(am.sum())
    height = fy[None, :] - fy[:, None]
    n = len(Y)

    def cumulative(mask, weights=None):
        return np.concatenate([[0.0], np.cumsum(np.bincount(ay[mask], weights, minlength=n))])
    boxes, rects = [], []
    for i in range(len(X)):
        ci = cumulative(ax == i)
        for j in range(i, len(X)):
            width = fx[j] - fx[i]
            if width * width > total * (1 + 1e-9):
                break
            inside = (ax >= i) & (ax <= j)
            c = cumulative(inside, am[inside])
            rows = np.bincount(ay[inside], minlength=n) > 0
            cj = cumulative(ax == j)
            mass = c[None, 1:] - c[:-1, None]
            r = 0.5 * np.sqrt(np.maximum(mass, 0.0))
            ok = ((height >= 0) & rows[:, None] & rows[None, :]
                  & (ci[None, 1:] - ci[:-1, None] > 0) & (cj[None, 1:] - cj[:-1, None] > 0)
                  & (2 * r * (1 + 1e-9) >= np.maximum(width, height))
                  & (fx[j] - r < fpx * (1 + 1e-9)) & (fx[i] + r > -1e-9 * fpx)
                  & (fy[None, :] - r < fpy * (1 + 1e-9)) & (fy[:, None] + r > -1e-9 * fpy))
            for k, l in zip(*np.nonzero(ok)):
                rk = r[k, l]
                boxes.append((i, j, k, l))
                rects.append((max(fx[j] - rk, 0.0), min(fx[i] + rk, fpx), max(fy[l] - rk, 0.0), min(fy[k] + rk, fpy)))
    if not boxes:
        return []
    keep = _maximal(np.array(rects), 1e-12 * max(fpx, fpy))
    return [boxes[t] for t in keep]


def rectangles(period, atoms):
    """Witness rectangles (x0, x1, y0, y1) in integer units of 2**-GRID, clipped to the period box and rounded
    inward, each with the atom box (x_lo, x_hi, y_lo, y_hi) whose atoms it holds: [(rect, box), ...]."""
    px, py = period
    win = _window(period, atoms)
    X = sorted({x for x, _, _ in win})
    Y = sorted({y for _, y, _ in win})
    scale = 1 << GRID
    gx, gy = _floor(px * scale), _floor(py * scale)
    out = []
    for i, j, k, l in _boxes(period, win, X, Y):
        m = sum((w for x, y, w in win if X[i] <= x <= X[j] and Y[k] <= y <= Y[l]), Fraction(0))
        # floor(sqrt(m) 2**GRID); half of it, over 2**GRID, is at most L/2.
        root = math.isqrt(_floor(m * scale * scale))
        x0 = max(_ceil(Fraction(X[j] * 2 * scale - root, 2)), 0)
        x1 = min(_floor(Fraction(X[i] * 2 * scale + root, 2)), gx)
        y0 = max(_ceil(Fraction(Y[l] * 2 * scale - root, 2)), 0)
        y1 = min(_floor(Fraction(Y[k] * 2 * scale + root, 2)), gy)
        if x0 < x1 and y0 < y1:
            out.append(((x0, x1, y0, y1), (X[i], X[j], Y[k], Y[l])))
    return out


def _dominated(rects):
    """The exact rectangles that are not inside another."""
    if len(rects) < 2:
        return rects
    r = np.array(rects, dtype=np.int64)
    return [tuple(int(v) for v in r[t]) for t in _maximal(r)]


def _strips(rects, gy):
    """For each x-strip [a, b) of the union of rects: (a, b, covered length, y-gaps as [(y0, y1), ...])."""
    if not rects:
        return []
    r = np.array(rects, dtype=np.int64)
    xs = np.unique(np.concatenate([r[:, 0], r[:, 1]]))
    out = []
    for a, b in zip(xs[:-1], xs[1:]):
        live = r[(r[:, 0] <= a) & (r[:, 1] >= b)]
        if not len(live):
            out.append((int(a), int(b), 0, [(0, gy)]))
            continue
        live = live[np.argsort(live[:, 2], kind="stable")]
        gaps, covered, top = [], 0, 0
        for y0, y1 in zip(live[:, 2].tolist(), live[:, 3].tolist()):
            if y0 > top:
                gaps.append((top, y0))
            if y1 > top:
                covered += y1 - max(y0, top)
                top = y1
        if top < gy:
            gaps.append((top, gy))
        out.append((int(a), int(b), covered, gaps))
    return out


def cover(period, atoms):
    """(pruned witness rectangles, covered area in units of 4**-GRID, gaps as integer rectangles)."""
    px, py = period
    scale = 1 << GRID
    gx, gy = _floor(px * scale), _floor(py * scale)
    rects = _dominated([rect for rect, _ in rectangles(period, atoms)])
    strips = _strips(rects, gy)
    area = sum((b - a) * covered for a, b, covered, _ in strips)
    segments = [(a, b, g) for a, b, _, g in strips]
    first, last = (strips[0][0], strips[-1][1]) if strips else (gx, gx)
    segments = [(0, first, [(0, gy)])] * (first > 0) + segments + [(last, gx, [(0, gy)])] * (last < gx)
    # Merge the gaps that run unchanged across consecutive strips into one rectangle.
    gaps, running = [], {}
    for a, b, g in segments:
        current = {interval: running.pop(interval, a) for interval in g}
        gaps += [(start, a, y0, y1) for (y0, y1), start in running.items()]
        running = current
    gaps += [(start, gx, y0, y1) for (y0, y1), start in running.items()]
    return rects, area, sorted(gaps)


def bound(period, atoms) -> Fraction:
    """The certified lower bound: covered area of the period box over its mass."""
    _, area, _ = cover(period, atoms)
    return Fraction(area, 1 << (2 * GRID)) / sum(m for _, _, m in atoms)


# --- behavior readout -------------------------------------------------------
# The score is one number. These describe the SHAPE of the measure instead, so two
# measures with the same bound can land in different cells of the diversity map.

_BEHAVIOR_ZERO = {"b_dispersion": 0.0, "b_boundary_frac": 0.0, "b_contact": 0.0,
                  "b_offset": 0.0, "b_concentration": 0.0, "b_structure": 0.0}


def _behavior(period, atoms, rects, area):
    """Six bounded descriptors: mass spread, uncovered fraction of the box, witness count, box aspect, heaviest
    atom's share and atom count. Never raises."""
    try:
        m = np.array([float(w) for _, _, w in atoms])
        spread = float(m.std() / m.mean())
        box = float(period[0] * period[1]) * float(1 << (2 * GRID))
        aspect = abs(math.log(float(period[0] / period[1])))
        out = {"b_dispersion": spread / (1.0 + spread), "b_boundary_frac": 1.0 - float(area) / box,
               "b_contact": len(rects) / (len(rects) + 10.0), "b_offset": aspect / (1.0 + aspect),
               "b_concentration": float(m.max() / m.sum()), "b_structure": len(atoms) / MAX_ATOMS}
        return {k: (float(v) if np.isfinite(v) else 0.0) for k, v in out.items()}
    except (ValueError, FloatingPointError, ZeroDivisionError, OverflowError):
        return dict(_BEHAVIOR_ZERO)


def _text(q):
    return f"{q.numerator}/{q.denominator}"


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False)


def _source_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _evaluate(program_path, stage):
    source_sha = _source_hash(program_path)
    evaluator_sha = _source_hash(__file__)
    invocation_id = uuid.uuid4().hex
    program = _load(program_path, "program_" + stage)
    if not hasattr(program, "periodic_measure"):
        raise AttributeError("missing periodic_measure")
    start_time = time.time()
    period, atoms = measure(program.periodic_measure())
    eval_time = time.time() - start_time
    rects, area, gaps = cover(period, atoms)
    value = Fraction(area, 1 << (2 * GRID)) / sum(m for _, _, m in atoms)
    combined = float(value / Fraction(REFERENCE))
    # Slivers where witnesses abut, left by the inward rounding, are not counted as gaps.
    sizes = [float((b - a) * (d - c)) / float(1 << (2 * GRID)) for a, b, c, d in gaps]
    gap_area = sum(sizes)
    gaps = [g for g, size in zip(gaps, sizes) if size >= 1e-12 * float(period[0] * period[1])]
    cases = {"lower_bound": float(value), "atoms": float(len(atoms)), "witnesses": float(len(rects)),
             "gaps": float(len(gaps)), "gap_area": gap_area, "combined_score": combined, "validity": 1.0}
    cases.update(_behavior(period, atoms, rects, area))
    witness = {
        "schema": "sciel-full-output/1",
        "source_sha256": source_sha, "evaluator_sha256": evaluator_sha,
        "invocation_id": invocation_id, "stage": stage, "confirmation": None,
        "input_sha256": None, "seed_sha256": None,
        "output": {"period": [_text(v) for v in period], "atoms": [[_text(v) for v in a] for a in atoms],
                   "bound": [value.numerator, value.denominator]},
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
    """Re-run the program n times and keep the worst result; a record should survive its own
    repetition."""
    n = max(1, int(n or 1))
    runs = [full(program_path) for _ in range(n)]
    failures = [r for r in runs if not r.get("ok")]
    selected = (next(i for i, result in enumerate(runs) if not result.get("ok"))
                if failures else min(range(n), key=lambda i: runs[i]["score"]))
    worst = dict(runs[selected])
    worst["cases"] = {**worst["cases"], "n_trials": float(n)}
    worst["artifacts"] = dict(worst.get("artifacts", {}))
    selection = {"trials": n, "selected_trial": selected,
                 "selection": "first_failure_else_lowest_score"}
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
        "cases": {"combined_score": 0.0, "lower_bound": 0.0, "atoms": 0.0, "witnesses": 0.0, "gaps": 0.0,
                  "gap_area": 0.0, "validity": 0.0, **_BEHAVIOR_ZERO},
        "artifacts": {"error": msg, "traceback": traceback.format_exc()[:2000]},
    }
