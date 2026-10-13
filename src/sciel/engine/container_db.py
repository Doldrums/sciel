"""Opt-in container experiments; the legacy Database contract stays unchanged.

All modes use the same active-population hard cap. Historical evaluated sources in
``_all`` are provenance, never extra candidates for parent selection or inspiration.
"""
from __future__ import annotations

import copy
import math

import numpy as np

from .db import Database
from .types import Program


class ContainerDatabase(Database):
    def __init__(self, *args, container, vector_fn=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.container = copy.deepcopy(container)
        self.kind = container["kind"]
        self.vector_fn = vector_fn
        self.geometries = {}
        self.vectors = {}
        self.progress = 0.0
        self.admissions = 0
        self.eviction_count = 0
        self.elite_eviction_count = 0
        if not container.get("strict_cap", True):
            raise ValueError("experimental containers require strict_cap=true")
        if self.pop_cap < 1:
            raise ValueError("container pop_cap must be positive")
        if self.kind in {"cvt", "vq", "annealed"} and vector_fn is None:
            raise ValueError("geometric containers require frozen vectors")

    def _geometry(self, island, dim):
        if island not in self.geometries:
            from ..components.containers import Geometry
            c = self.container
            self.geometries[island] = Geometry(
                self.kind, dim=dim, k=c.get("k", 4),
                max_centres=c.get("max_centres", 6), seed=c.get("seed", 0),
                alpha=c.get("temperature_alpha", .02),
                split_min_points=c.get("split_min_points", 4),
                novelty_threshold=c.get("vq_threshold", .25))
        return self.geometries[island]

    def _cells(self, island, *, update=True):
        isl = self.islands[island]
        if self.kind in {"pool", "pool_novelty"}:
            return {}
        if self.kind == "grid":
            return {p.id: super(ContainerDatabase, self)._cell(p)
                    for p in isl.programs.values()}
        ids = list(isl.programs)
        if not ids:
            return {}
        vectors = np.asarray([self.vectors[pid] for pid in ids], dtype=float)
        geo = self._geometry(island, vectors.shape[1])
        if update:
            labels = geo.update(ids, vectors, self.progress)
        else:
            probs = geo.probabilities(vectors)
            labels = {pid: geo.centre_ids[int(np.argmax(probs[i]))]
                      for i, pid in enumerate(ids)}
        # Geometry replaces the joint grid: one elite per centre. Quality still
        # decides the elite and retention order, not another score-bin axis.
        return {pid: (labels[pid], 0) for pid in ids}

    def _rebuild_grid(self, island, cells):
        isl = self.islands[island]
        isl.grid = {}
        for pid, p in isl.programs.items():
            if pid not in cells:
                continue
            cell = cells[pid]
            old = isl.grid.get(cell)
            if old is None or p.score > isl.programs[old].score:
                isl.grid[cell] = pid

    def add(self, code, score, cases, island, origin, parent_id=None, sig_hash="", card="", mechanism=""):
        if not math.isfinite(score):
            raise ValueError("container score must be finite")
        isl = self.islands[island]
        gen = self._all[parent_id].gen + 1 if parent_id in self._all else 0
        p = Program(self.new_id(), code, score, cases, island, origin,
                    parent_id=parent_id, gen=gen, sig_hash=sig_hash, card=card, mechanism=mechanism)
        if self.kind in {"cvt", "vq", "annealed"}:
            z = np.asarray(self.vector_fn(code, cases), dtype=float)
            if z.ndim != 1 or not len(z) or not np.isfinite(z).all():
                raise ValueError("invalid frozen container vector")
            self.vectors[p.id] = z.tolist()
        self._all[p.id] = p
        isl.programs[p.id] = p
        self.admissions += 1
        if isl.best_id is None or score > isl.best_score:
            isl.best_id = p.id
        if self.pop_best_id is None or score > self.pop_best:
            self.pop_best_id = p.id
        cells = self._cells(island)
        self._rebuild_grid(island, cells)
        elites = set(isl.grid.values())
        # Best is protected in every arm. Excess niche elites obey the SAME
        # score/id tie-break as other candidates, so capacity is a real bound.
        ordered = sorted(isl.programs, key=lambda pid: (
            pid == isl.best_id, pid in elites, isl.programs[pid].score,
            -int(pid[1:])), reverse=True)
        keep = set(ordered[:self.pop_cap])
        removed = False
        for pid in list(isl.programs):
            if pid in keep:
                continue
            victim = isl.programs.pop(pid)
            removed = True
            elite = pid in elites
            self.eviction_count += 1
            self.elite_eviction_count += int(elite)
            self.evictions.append({"island": island, "victim_id": pid,
                "victim_score": victim.score, "victim_gen": victim.gen,
                "victim_origin": victim.origin, "n_kept": len(keep),
                "container": self.kind, "was_elite": elite})
            self.vectors.pop(pid, None)
        if removed and island in self.geometries:
            ids = list(isl.programs)
            labels = self.geometries[island].refresh_active(
                ids, np.asarray([self.vectors[pid] for pid in ids]))
            cells = {pid: (label, 0) for pid, label in labels.items()}
        self._rebuild_grid(island, cells)
        # The archive is deliberately absent from every controlled mode.
        self.archive = []
        return p

    def reencode_niches(self):
        # Geometry updates exactly once per admission; SH may not update it again.
        return None

    def diagnostics(self):
        return {"kind": self.kind, "progress": self.progress,
                "admissions": self.admissions, "evictions": self.eviction_count,
                "elite_evictions": self.elite_eviction_count,
                "active_counts": [len(i.programs) for i in self.islands],
                "centres": {str(i): g.diagnostics() for i, g in self.geometries.items()}}

    def population_snapshot(self):
        out = super().population_snapshot()
        out["container"] = self.diagnostics()
        return out

    def select_parent(self, island, rng, floor=1e-3, tau_floor=1e-3, tilt=None):
        from ..components.policy.base import floor_clip
        from ..components.selection.novelty_chain import Selected
        isl = self.islands[island]
        ids = list(isl.programs)
        g = self.geometries[island]
        p = g.probabilities(np.asarray([self.vectors[i] for i in ids]))
        masses = p.sum(axis=0)
        cp = np.zeros_like(masses)
        np.divide(1.0, masses, out=cp, where=masses > 1e-12)
        cp /= cp.sum()
        scores = np.asarray([isl.programs[i].score for i in ids])
        quality = np.exp((scores - scores.max()) / max(float(scores.std()), tau_floor))
        quality /= quality.sum()
        if tilt is not None:
            from ..components.selection.novelty_chain import IslandView
            target = tilt(IslandView(ids, dict(zip(ids, scores)), {}))
            quality = (1 - tilt.beta) * quality + tilt.beta * np.asarray([target.get(i, 0) for i in ids])
        quality = np.maximum(quality, 1e-12)
        quality /= quality.sum()
        conditional = np.zeros_like(p)
        for j in range(p.shape[1]):
            if cp[j] == 0:
                continue
            weights = quality * p[:, j]
            weights /= weights.sum()
            dist = floor_clip(dict(zip(ids, weights)), floor)
            conditional[:, j] = [dist[i] for i in ids]
        j = rng.choices(range(len(cp)), weights=cp, k=1)[0]
        idx = rng.choices(range(len(ids)), weights=conditional[:, j], k=1)[0]
        marginal = conditional @ cp
        return Selected(ids[idx], [], float(marginal[idx]), {
            "container": self.kind, "center": g.centre_ids[j],
            "center_propensity": float(cp[j]),
            "conditional_parent_propensity": float(conditional[idx, j]),
            "joint_propensity": float(cp[j] * conditional[idx, j]),
            "rho_sum": float(marginal.sum()), "n": len(ids)})

    def state(self):
        state = super().state()
        state["container"] = {"schema": 1, "config": copy.deepcopy(self.container),
            "vectors": copy.deepcopy(self.vectors), "progress": self.progress,
            "admissions": self.admissions, "evictions": self.eviction_count,
            "elite_evictions": self.elite_eviction_count,
            "geometries": {str(i): g.state() for i, g in self.geometries.items()}}
        return state

    def load_state(self, state):
        data = state.get("container")
        if data is None or data.get("schema") != 1 or data["config"] != self.container:
            raise ValueError("checkpoint container contract differs from configured container")
        super().load_state(state)
        self.vectors = copy.deepcopy(data["vectors"])
        self.progress = data["progress"]
        self.admissions = data["admissions"]
        self.eviction_count = data["evictions"]
        self.elite_eviction_count = data["elite_evictions"]
        self.geometries = {}
        for i, gs in data["geometries"].items():
            dim = len(next(iter(self.vectors.values())))
            self._geometry(int(i), dim).load_state(gs)
        for isl in self.islands:
            if len(isl.programs) > self.pop_cap:
                raise ValueError("checkpoint exceeds container population cap")
