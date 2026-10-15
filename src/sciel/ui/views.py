"""The four pictures of a run as plain figures: culture disk, behaviour landscape, code map, edits.

A figure lives in the unit square with y pointing up and holds lines, dots, shaded cells and labels. The web page
draws it as SVG and the terminal as braille, so both show the same geometry.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

ISLAND_COLORS = ("#E8A33D", "#3DAE7C", "#0A84FF", "#A66BE0", "#2BB3C0", "#D67AB1", "#D1B23F", "#4F7BE8")
CHORD = "#E0637F"
SHADE = "#0A84FF"
MUTED = "#98989D"


def island_color(island) -> str:
    return ISLAND_COLORS[int(island or 0) % len(ISLAND_COLORS)]


def _figure(**extra):
    return {"items": [], **extra}


def _geodesic(p, q, steps=18):
    """Points along the Poincaré-disk geodesic from p to q."""
    (x1, y1), (x2, y2) = p, q
    det = x1 * y2 - x2 * y1
    if abs(det) < 1e-9:
        return [[x1 + (x2 - x1) * t / steps, y1 + (y2 - y1) * t / steps] for t in range(steps + 1)]
    k1, k2 = (x1 * x1 + y1 * y1 + 1) / 2, (x2 * x2 + y2 * y2 + 1) / 2
    cx, cy = (k1 * y2 - k2 * y1) / det, (x1 * k2 - x2 * k1) / det
    radius = math.hypot(x1 - cx, y1 - cy)
    a1, a2 = math.atan2(y1 - cy, x1 - cx), math.atan2(y2 - cy, x2 - cx)
    delta = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
    return [[cx + radius * math.cos(a1 + delta * t / steps), cy + radius * math.sin(a1 + delta * t / steps)]
            for t in range(steps + 1)]


def _to_unit(points):
    return [[(x + 1) / 2, (y + 1) / 2] for x, y in points]


def _sizes(programs):
    scores = [p["score"] for p in programs if p.get("score") is not None]
    low, high = (min(scores), max(scores)) if scores else (0.0, 1.0)
    span = (high - low) or 1.0
    return lambda score: 0.006 + 0.012 * (((score if score is not None else low) - low) / span)


def disk(snap) -> dict:
    """Islands as sectors, lineages as hyperbolic trees from the seed, transferred findings as chords."""
    programs = {p["id"]: p for p in snap["programs"]}
    islands = sorted({p["island"] for p in programs.values() if p.get("island") is not None})
    fig = _figure(kind="disk", caption="sector = island · radius = generation · chord = finding passed between islands")
    if not islands:
        return fig
    span = 2 * math.pi / len(islands)
    children = defaultdict(list)
    roots = defaultdict(list)
    for p in programs.values():
        parent = programs.get(p.get("parent"))
        if parent is not None and parent.get("island") == p.get("island"):
            children[parent["id"]].append(p["id"])
        else:
            roots[p.get("island")].append(p["id"])
    leaves = {}

    def count(pid):
        leaves[pid] = max(1, sum(count(c) for c in children[pid]))
        return leaves[pid]

    position = {}

    def place(pid, start, width, depth):
        p = programs[pid]
        gen = p.get("gen") if p.get("gen") is not None else depth
        radius = math.tanh(0.42 * gen + 0.25)
        angle = start + width / 2
        position[pid] = [radius * math.cos(angle), radius * math.sin(angle)]
        offset = start
        for child in sorted(children[pid], key=lambda c: programs[c].get("B_evals") or 0):
            share = width * leaves[child] / leaves[pid]
            place(child, offset, share, depth + 1)
            offset += share

    for index, island in enumerate(islands):
        total = sum(count(r) for r in roots[island])
        start = math.pi / 2 - (index + 1) * span + span * 0.06
        width = span * 0.88
        for root in roots[island]:
            share = width * leaves[root] / total
            place(root, start, share, 0)
            start += share
        label_angle = math.pi / 2 - (index + 0.5) * span
        fig["items"].append({"t": "text", "x": (1 + 1.03 * math.cos(label_angle)) / 2,
                             "y": (1 + 1.03 * math.sin(label_angle)) / 2, "s": f"I{island}", "c": MUTED,
                             "anchor": "middle"})
    fig["items"].insert(0, {"t": "ring", "x": 0.5, "y": 0.5, "r": 0.5, "c": "#2C2C2E"})
    for pid, p in programs.items():
        parent = programs.get(p.get("parent"))
        source = position.get(parent["id"]) if parent is not None else [0.0, 0.0]
        if pid in position and source is not None:
            fig["items"].append({"t": "line", "pts": _to_unit(_geodesic(source, position[pid])),
                                 "c": island_color(p.get("island")), "o": 0.55, "w": 1})
    cells = {c.get("cell"): c for c in snap.get("cells", [])}
    findings = {f["id"]: f for f in snap.get("findings", [])}
    best_by_island = {}
    for p in programs.values():
        if p.get("score") is not None and (p["island"] not in best_by_island
                                           or p["score"] > programs[best_by_island[p["island"]]]["score"]):
            best_by_island[p["island"]] = p["id"]
    for transfer in snap.get("transfers", []):
        finding = findings.get(transfer["finding"], {})
        origin = cells.get(finding.get("cell"), {}).get("program") or best_by_island.get(transfer["from_island"])
        if origin in position and transfer["child"] in position:
            fig["items"].append({"t": "line", "pts": _to_unit(_geodesic(position[origin], position[transfer["child"]])),
                                 "c": CHORD, "w": 1.4, "dash": not transfer["rose"]})
    size = _sizes(programs.values())
    alive = {p["id"] for p in snap["programs"] if p.get("alive")}
    for pid, (x, y) in position.items():
        p = programs[pid]
        fig["items"].append({"t": "dot", "x": (x + 1) / 2, "y": (y + 1) / 2, "r": size(p.get("score")),
                             "c": island_color(p.get("island")), "fill": pid in alive, "id": pid})
    fig["items"].append({"t": "dot", "x": 0.5, "y": 0.5, "r": 0.005, "c": "#E8E8EA", "fill": True})
    worked = sum(t["rose"] for t in snap.get("transfers", []))
    fig["stats"] = {"transfers": len(snap.get("transfers", [])), "worked": worked}
    return fig


def _metric_axes(programs):
    """The two behaviour metrics that track the score most closely and are not copies of each other."""
    keys = defaultdict(dict)
    for p in programs:
        for key, value in (p.get("cases") or {}).items():
            keys[key][p["id"]] = value
    scored = {p["id"]: p["score"] for p in programs}
    ranked = []
    for key, values in keys.items():
        ids = list(values)
        if len(ids) < 4 or np.var(list(values.values())) == 0:
            continue
        x, y = np.array([values[i] for i in ids]), np.array([scored[i] for i in ids])
        if np.var(y) == 0:
            continue
        r = abs(float(np.corrcoef(x, y)[0, 1]))
        if r < 0.98:
            ranked.append((r, key))
    chosen = []
    for _, key in sorted(ranked, reverse=True):
        shared = [i for i in keys[key] if chosen and i in keys[chosen[0]]]
        if chosen and len(shared) >= 4 and abs(np.corrcoef([keys[key][i] for i in shared],
                                                           [keys[chosen[0]][i] for i in shared])[0, 1]) > 0.97:
            continue
        chosen.append(key)
        if len(chosen) == 2:
            break
    return chosen


def _shade(fig, points, cells=(40, 26)):
    """Inverse-distance score field over the plot, drawn as translucent cells."""
    if len(points) < 2:
        return
    xy = np.array([[x, y] for x, y, _ in points])
    s = np.array([v for _, _, v in points])
    low, high = s.min(), s.max()
    if high <= low:
        return
    nx, ny = cells
    for i in range(nx):
        for j in range(ny):
            cx, cy = (i + 0.5) / nx, (j + 0.5) / ny
            w = 1 / (np.hypot(xy[:, 0] - cx, xy[:, 1] - cy) ** 3 + 1e-5)
            value = float((w * s).sum() / w.sum())
            fig["items"].append({"t": "cell", "x": i / nx, "y": j / ny, "w": 1 / nx, "h": 1 / ny, "c": SHADE,
                                 "o": round(0.03 + 0.42 * ((value - low) / (high - low)) ** 1.5, 3)})


def _blobs(fig, points, radius=0.06):
    """A soft disc of score around each program; dense good regions read as solid shapes."""
    s = [v for _, _, v in points]
    low, high = min(s), max(s)
    span = (high - low) or 1.0
    for x, y, v in points:
        fig["items"].append({"t": "blob", "x": x, "y": y, "r": radius, "c": SHADE,
                             "o": round(0.03 + 0.09 * ((v - low) / span), 3)})


def _scale(values):
    low, high = min(values), max(values)
    span = (high - low) or 1.0
    return lambda v: 0.04 + 0.92 * (v - low) / span, (low, high)


def _scatter(fig, programs, coords):
    by_id = {p["id"]: p for p in programs}
    for pid, (x, y) in coords.items():
        parent = by_id[pid].get("parent")
        if parent in coords:
            fig["items"].append({"t": "line", "pts": [coords[parent], [x, y]], "c": MUTED, "o": 0.7, "w": 0.8,
                                 "arrow": True})
    for pid, (x, y) in coords.items():
        fig["items"].append({"t": "dot", "x": x, "y": y, "r": 0.008, "c": island_color(by_id[pid].get("island")),
                             "fill": True, "id": pid})


def landscape(snap) -> dict:
    """Programs placed by their two most varying behaviour metrics; shading is the score."""
    programs = [p for p in snap["programs"] if p.get("score") is not None]
    axes = _metric_axes(programs)
    if len(axes) < 2:
        fig = _figure(kind="landscape", caption="the task reports fewer than two behaviour metrics; showing score by evaluation",
                      axes=["evaluation", "score"])
        placed = [p for p in programs if p.get("B_evals") is not None]
        if not placed:
            return fig
        fx, rx = _scale([p["B_evals"] for p in placed])
        fy, ry = _scale([p["score"] for p in placed])
        coords = {p["id"]: [fx(p["B_evals"]), fy(p["score"])] for p in placed}
        fig["ranges"] = [rx, ry]
    else:
        fig = _figure(kind="landscape", caption="shading is score · each arrow is one edit", axes=axes)
        placed = [p for p in programs if all(a in p["cases"] for a in axes)]
        fx, rx = _scale([p["cases"][axes[0]] for p in placed])
        fy, ry = _scale([p["cases"][axes[1]] for p in placed])
        coords = {p["id"]: [fx(p["cases"][axes[0]]), fy(p["cases"][axes[1]])] for p in placed}
        fig["ranges"] = [rx, ry]
    by_id = {p["id"]: p for p in placed}
    _shade(fig, [(x, y, by_id[pid]["score"]) for pid, (x, y) in coords.items()])
    _scatter(fig, placed, coords)
    return fig


def code_map(snap, code) -> dict:
    """Programs placed by the first two principal components of their identifier counts."""
    programs = [p for p in snap["programs"] if p["id"] in code and p.get("score") is not None]
    fig = _figure(kind="code", axes=["code, component 1", "code, component 2"],
                  caption="same programs by code; shading is score near each program")
    if len(programs) < 3:
        fig["caption"] = "too few programs with code to map yet"
        return fig
    vocabulary = defaultdict(int)
    for p in programs:
        for word in code[p["id"]]:
            vocabulary[word] += 1
    words = [w for w, df in sorted(vocabulary.items(), key=lambda kv: -kv[1]) if df < len(programs)][:400]
    if len(words) < 2:
        fig["caption"] = "programs share all their identifiers; nothing to separate"
        return fig
    index = {w: i for i, w in enumerate(words)}
    m = np.zeros((len(programs), len(words)))
    for row, p in enumerate(programs):
        for word, n in code[p["id"]].items():
            if word in index:
                m[row, index[word]] = 1 + math.log(n)
    m *= np.log(len(programs) / np.array([vocabulary[w] for w in words]))
    m /= np.linalg.norm(m, axis=1, keepdims=True) + 1e-12
    m -= m.mean(axis=0)
    u, s, _ = np.linalg.svd(m, full_matrices=False)
    explained = float((s[:2] ** 2).sum() / max((s ** 2).sum(), 1e-12))
    xy = u[:, :2] * s[:2]
    fx, rx = _scale(list(xy[:, 0]))
    fy, ry = _scale(list(xy[:, 1]) if xy.shape[1] > 1 else [0.0])
    coords = {p["id"]: [fx(xy[i, 0]), fy(xy[i, 1]) if xy.shape[1] > 1 else 0.5] for i, p in enumerate(programs)}
    fig["ranges"] = [rx, ry]
    fig["caption"] = f"same programs by code; two components hold {explained:.0%} of variance; shading is score"
    _blobs(fig, [(x, y, p["score"]) for p, (x, y) in zip(programs, coords.values())])
    _scatter(fig, programs, coords)
    return fig


def edits(snap, noise=None) -> dict:
    """Child score against parent score; points above the band are real steps up."""
    rows = snap.get("edits", [])
    fig = _figure(kind="edits", axes=["parent score", "child score"])
    if not rows:
        fig["caption"] = "no edits measured yet"
        return fig
    values = [v for r in rows for v in r[:2]]
    f, r = _scale(values)
    fig["ranges"] = [r, r]
    eps = noise or 0.0
    band = (f(r[0] + eps) - f(r[0])) if eps else 0.0
    if band:
        fig["items"].append({"t": "band", "w": band, "c": MUTED, "o": 0.18})
    fig["items"].append({"t": "line", "pts": [[0.04, 0.04], [0.96, 0.96]], "c": MUTED, "o": 0.9, "w": 0.8, "dash": True})
    above = 0
    for parent, child, island in rows:
        above += child > parent + eps
        fig["items"].append({"t": "dot", "x": f(parent), "y": f(child), "r": 0.009, "c": island_color(island),
                             "fill": True})
    share = above / len(rows)
    where = "above the noise band" if eps else "above the diagonal (noise band unknown)"
    fig["caption"] = f"{above} of {len(rows)} edits {where} → π ≈ {share:.2f}"
    fig["stats"] = {"above": above, "total": len(rows), "share": share}
    return fig


def all_views(snap, code, noise=None) -> dict:
    return {"culture": disk(snap), "landscape": landscape(snap), "code": code_map(snap, code),
            "edits": edits(snap, noise)}
