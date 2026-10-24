"""Squares in squares: the task's slice, the merged public packings, the official check and fast float geometry.

A packing is an (n, 3) array of unit squares (centre x, centre y, rotation angle in radians). Its side is
L = max(max_x - min_x, max_y - min_y) over all corners and its excess is L / sqrt(n) - 1. The scored packing is the
decimal repr of each binary64 value, checked by the competition's own metric (float separating-axis test, then
interval arithmetic for pairs closer than 1e-8); an exactly touching pair of rotated squares is undecidable and
rejected, so rotated contacts need a small gap (1e-12 is plenty).
"""

import os

for _name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
              "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
    os.environ[_name] = "1"

import base64
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from sciel.candidate_modules import load_candidate

ENV = "SCIEL_SQUARES"
DIGESTS = {
    "metric.py": "7be1497ce904fb3ea6e63880343a66dcf2b07a09fd8d03ade526643d98777b59",
    "slice.json": "22df0ac688c727fe360dc50f7eb622f73479d26073b6f0c63cb8b9f06bc0b8f6",
    "best/best.json": "02595e874337a6e52abfbcd591aa83c2ca25dd52ca105f3076cff3ccd468b0dc",
}
# Sum over n = 1..1000 of the per-n best certified public packing (12 public submissions, 23 Oct 2026); neighbour
# transfer (dropping squares from a better m > n) and strip growth improve none of them.
MERGED_TOTAL = 22.30836914756724
# The 40 n with the largest excess over a smooth fit of L - floor(sqrt n) in (j / k, 1 / k), n = k^2 + j, 1 <= j <= k,
# k >= 10, ranked by the residual divided by sqrt(n) (its weight in the total).
SLICE = (103, 105, 106, 108, 109, 110, 124, 127, 147, 150, 151, 154, 175, 176, 201, 203, 204, 232, 234, 260, 293, 295,
         299, 330, 366, 367, 370, 405, 410, 412, 419, 446, 453, 462, 491, 535, 582, 587, 848, 963)
# The metric's L for the merged packing of each n in the slice (60-digit measurement, 50 decimals).
SEED_L = {
    103: "10.679232047370986", 105: "10.7893037837579468", 106: "10.82290804414267", 108: "10.904821012330261",
    109: "10.949747468315782", 110: "10.99678339664159", 124: "11.656854249503037", 127: "11.810878787599992",
    147: "12.656854249504037", 150: "12.778174593063801", 151: "12.82287565554412", 154: "12.9211009407736006",
    175: "13.764174317499361", 176: "13.795872318449204", 201: "14.707106781200253", 203: "14.778174593065801",
    204: "14.808642840605306", 232: "15.764174317501361", 234: "15.822875655547118", 260: "16.645753250535456",
    293: "17.634146341480047", 295: "17.704232790752733", 299: "17.822875655549117", 330: "18.692388156132942",
    366: "19.656854249511037", 367: "19.692388155443811",
    370: "19.76417431880910310157953023862136555279320952153806",
    405: "20.64575325317847449999999606175974999478970814925517",
    410: "20.77473513404402699999990436689749944573903330535135",
    412: "20.82299541548050930961566084581877985919185135539848", 419: "20.949984998021191",
    446: "21.645753251345763", 453: "21.808663235415606",
    462: "21.96619309917676899999963007713749528355416092292340", 491: "22.69238815544681",
    535: "23.656854249515037", 582: "24.64969334352762729464486645521921886803763377310940",
    587: "24.756097562799254", 848: "29.642135623759593", 963: "31.535533905963272",
}
_HALF_DIAGONAL = math.sqrt(0.5)
_cache = {}


def _path(name):
    root = os.environ.get(ENV)
    if not root:
        raise FileNotFoundError(f"set {ENV} to the squares_in_squares data directory")
    path = Path(root) / name
    if name not in _cache:
        h = hashlib.sha256(path.read_bytes()).hexdigest()
        if h != DIGESTS[name]:
            raise ValueError(f"{name} does not match its pinned digest")
        _cache[name] = path
    return path


def metric():
    """The competition's metric module, unchanged (evaluate_single_puzzle, verify_puzzle, measure_side_mp, ...)."""
    if "metric" not in _cache:
        _cache["metric"] = load_candidate(str(_path("metric.py")), "squares_metric")
    return _cache["metric"]


def seed_excess(n: int) -> float:
    return float(SEED_L[n]) / math.sqrt(n) - 1.0


def excess(n: int, side_length: float) -> float:
    return side_length / math.sqrt(n) - 1.0


def _decimal(name, n):
    key = f"rows_{name}"
    if key not in _cache:
        _cache[key] = json.loads(_path(name).read_text())
    return np.array(_cache[key][str(n)], dtype=np.float64)


def seed(n: int) -> np.ndarray:
    """The merged public packing for slice n as (n, 3) binary64 (it certifies unchanged after rounding)."""
    return _decimal("slice.json", n)


def best(m: int) -> np.ndarray:
    """The merged public packing for any m in 1..1000 as (m, 3) binary64, e.g. to drop squares from a larger m."""
    return _decimal("best/best.json", m)


def encode(p) -> str:
    return base64.b64encode(np.asarray(p, dtype="<f8").tobytes()).decode()


def decode(text: str) -> np.ndarray:
    """A packing from a scored output's packings entry (base64 of little-endian binary64 x y angle), as (n, 3)."""
    return np.frombuffer(base64.b64decode(text), dtype="<f8").reshape(-1, 3).astype(np.float64)


def rows(n: int, p) -> list:
    """The packing as the metric's rows: decimal repr of each binary64 value, as written to the submission CSV."""
    return [{"n": str(n), "square_id": str(i), "x": repr(float(x)), "y": repr(float(y)), "angle": repr(float(a))}
            for i, (x, y, a) in enumerate(np.asarray(p, dtype=np.float64))]


def certify(p) -> tuple:
    """(feasible, L, reason) from the official metric on the packing's decimal reprs: L is the metric's side as a
    float, or None when the packing is rejected."""
    p = np.asarray(p, dtype=np.float64)
    _, _, side_text, reason = metric().evaluate_single_puzzle(len(p), rows(len(p), p))
    return reason == "ok", None if side_text is None else float(side_text), reason


# --- fast float geometry ----------------------------------------------------------

def corners(p) -> np.ndarray:
    """(n, 4, 2) corners in the metric's order."""
    p = np.asarray(p, dtype=np.float64)
    c, s = np.cos(p[:, 2]), np.sin(p[:, 2])
    dx1, dy1 = 0.5 * (c - s), 0.5 * (s + c)
    dx2, dy2 = 0.5 * (-c - s), 0.5 * (-s + c)
    x, y = p[:, 0], p[:, 1]
    return np.stack([np.stack([x + dx1, y + dy1], -1), np.stack([x + dx2, y + dy2], -1),
                     np.stack([x - dx1, y - dy1], -1), np.stack([x - dx2, y - dy2], -1)], 1)


def side(p) -> float:
    """Float L = max(max_x - min_x, max_y - min_y) over all corners."""
    q = corners(p).reshape(-1, 2)
    return float(max(q[:, 0].max() - q[:, 0].min(), q[:, 1].max() - q[:, 1].min()))


def half_width(alpha):
    """Half the extent of a unit square rotated by alpha along a fixed axis: (|cos| + |sin|) / 2."""
    return 0.5 * (np.abs(np.cos(alpha)) + np.abs(np.sin(alpha)))


def pairs(p, margin: float = 0.0) -> tuple:
    """(i, j) arrays, i < j, of the pairs whose centres are within sqrt 2 + margin: every pair that can touch."""
    p = np.asarray(p, dtype=np.float64)
    found = cKDTree(p[:, :2]).query_pairs(2 * _HALF_DIAGONAL + margin, output_type="ndarray")
    if not len(found):
        return np.zeros(0, int), np.zeros(0, int)
    return found[:, 0], found[:, 1]


def axis_gaps(p, i, j) -> np.ndarray:
    """(m, 4) signed gaps of each pair along the four separating axes (the two axes of square i, then of j)."""
    p = np.asarray(p, dtype=np.float64)
    ti, tj = p[i, 2], p[j, 2]
    d = p[j, :2] - p[i, :2]
    out = []
    for phi, other in ((ti, tj), (ti + np.pi / 2, tj), (tj, ti), (tj + np.pi / 2, ti)):
        proj = np.cos(phi) * d[:, 0] + np.sin(phi) * d[:, 1]
        out.append(np.abs(proj) - 0.5 - half_width(other - phi))
    return np.stack(out, 1)


def gaps(p, i=None, j=None) -> tuple:
    """(i, j, gap): the separation of every close pair, the largest of its four axis gaps; negative means overlap."""
    if i is None:
        i, j = pairs(p)
    g = axis_gaps(p, i, j)
    return i, j, g.max(1) if len(i) else np.zeros(0)


def min_gap(p) -> float:
    """The smallest pair separation, inf for a packing whose squares are all far apart."""
    _, _, g = gaps(p)
    return float(g.min()) if len(g) else math.inf


def overlaps(p, tol: float = 0.0) -> tuple:
    """(i, j) of the pairs separated by less than tol in float arithmetic."""
    i, j, g = gaps(p)
    bad = g < tol
    return i[bad], j[bad]


def contacts(p, tol: float = 1e-9) -> tuple:
    """(i, j) of the pairs separated by less than tol: the contact graph that holds the packing rigid."""
    return overlaps(p, tol)


def rattlers(p, tol: float = 1e-9) -> np.ndarray:
    """Indices of the squares with fewer than two contacts within tol: free to move, so they sit in holes."""
    i, j = contacts(p, tol)
    degree = np.bincount(np.concatenate([i, j]), minlength=len(p))
    return np.flatnonzero(degree < 2)


# --- moves between n ------------------------------------------------------------------

def normalize(p) -> np.ndarray:
    """The same squares with angles reduced to [-pi/4, pi/4) and the corners' lower-left at the origin."""
    p = np.array(p, dtype=np.float64)
    p[:, 2] = (p[:, 2] + np.pi / 4) % (np.pi / 2) - np.pi / 4
    q = corners(p).reshape(-1, 2)
    p[:, 0] -= q[:, 0].min()
    p[:, 1] -= q[:, 1].min()
    return p


def remove(p, k: int) -> np.ndarray:
    """Drop k squares, each time the one whose removal shrinks the float bounding side most (a packing of m > n
    minus m - n squares is a packing of n)."""
    p = np.array(p, dtype=np.float64)
    for _ in range(k):
        q = corners(p)
        ext = [q[:, :, 0].min(1), q[:, :, 1].min(1), -q[:, :, 0].max(1), -q[:, :, 1].max(1)]
        without = []
        for v in ext:
            order = np.argsort(v)
            w = np.full(len(v), v[order[0]])
            w[order[0]] = v[order[1]]
            without.append(w)
        sides = np.maximum(-without[2] - without[0], -without[3] - without[1])
        p = np.delete(p, int(np.argmin(sides)), axis=0)
    return p


def grow(p, gap: float = 1e-11) -> np.ndarray:
    """Strip growth: a packing of n with side S becomes one of n + floor(S) + floor(S + 1) with side S + 1 + gap, by
    a column of floor(S + 1) axis-parallel squares on the right and a row of floor(S) on top."""
    p = normalize(p)
    s = side(p)
    right = max(s, math.floor(s) * (1 + gap)) + gap
    col = [(right + 0.5, 0.5 + k * (1 + gap), 0.0) for k in range(int(math.floor(s + 1)))]
    row = [(0.5 + k * (1 + gap), s + gap + 0.5, 0.0) for k in range(int(math.floor(s)))]
    return np.vstack([p, np.array(col + row)])
