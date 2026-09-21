"""Population: islands, a MAP-Elites grid, global archive, origin, absorbing V_t."""

from __future__ import annotations

import hashlib

from .types import Program


def normalized_hash(code: str) -> str:
    lines = [ln.rstrip() for ln in code.strip().splitlines() if ln.strip()]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


class Island:
    def __init__(self, idx: int):
        self.idx = idx
        self.programs: dict[str, Program] = {}
        self.grid: dict[tuple, str] = {}
        self.best_id: str | None = None

    @property
    def best_score(self) -> float:
        if self.best_id is None:
            return float("-inf")
        return self.programs[self.best_id].score

    def top_k(self, k: int) -> list[Program]:
        return sorted(self.programs.values(), key=lambda p: p.score, reverse=True)[:k]


class Database:
    def __init__(self, num_islands: int, pop_cap: int = 50, archive_size: int = 20,
                 grid_bins: int = 10, descriptor_fn=None, niche_fn=None):
        self.islands = [Island(i) for i in range(num_islands)]
        self.pop_cap = pop_cap
        self.archive_size = archive_size
        self.grid_bins = grid_bins
        # arch-v4 #4a: learned behavioral descriptor for the MAP-Elites axis;
        # None -> legacy code-length axis
        self.descriptor_fn = descriptor_fn
        # arch-v4 #4b: VQ-Elites codebook -- when set, the map's first axis is a
        # learned niche id instead of a binned descriptor. Takes precedence over
        # descriptor_fn. Its codebook state is checkpointed with the population.
        self.niche_fn = niche_fn
        self.archive: list[str] = []
        self._all: dict[str, Program] = {}
        self._counter = 0
        self.pop_best_id: str | None = None
        # arch-v4 #4c: evicted programs buffer for the eviction head's label
        # collection (drained + logged by the engine when log_evictions is on)
        self.evictions: list = []

    def new_id(self) -> str:
        self._counter += 1
        return f"p{self._counter}"

    @property
    def pop_best(self) -> float:
        if self.pop_best_id is None:
            return float("-inf")
        return self._all[self.pop_best_id].score

    def island_best(self, island: int) -> float:
        return self.islands[island].best_score

    def pre_snapshot(self, island: int) -> dict:
        return {"island_best": self.islands[island].best_score,
                "pop_best": self.pop_best}

    def get(self, pid: str) -> Program:
        return self._all[pid]

    def code_hashes(self, island: int) -> set:
        return {normalized_hash(p.code) for p in self.islands[island].programs.values()}

    def _cell(self, program: Program) -> tuple:
        score_bin = max(0, min(int(program.score * self.grid_bins), self.grid_bins - 1))
        if self.niche_fn is not None:
            # VQ-Elites: first axis is the learned niche id (may grow the codebook)
            return (self.niche_fn.assign(program.code, program.cases), score_bin)
        if self.descriptor_fn is not None:
            d = self.descriptor_fn(program.code, program.cases)   # [0,1] behavior axis
            desc_bin = max(0, min(int(d * self.grid_bins), self.grid_bins - 1))
        else:
            desc_bin = min(len(program.code) // 200, self.grid_bins - 1)
        return (desc_bin, score_bin)

    def reencode_niches(self) -> None:
        """Re-encode every island's grid against the current codebook (AURORA
        re-encode): as the codebook grows, early cells drift, so on SH milestones
        we recompute each program's niche (without growing) and keep one elite per
        (niche, score_bin). No-op unless a VQ codebook is active."""
        if self.niche_fn is None:
            return
        for isl in self.islands:
            grid: dict = {}
            for prog in isl.programs.values():
                nid = self.niche_fn.assign_fixed(prog.code, prog.cases)
                sb = max(0, min(int(prog.score * self.grid_bins), self.grid_bins - 1))
                cell = (nid, sb)
                cur = grid.get(cell)
                if cur is None or prog.score > isl.programs[cur].score:
                    grid[cell] = prog.id
            isl.grid = grid

    def add(self, code: str, score: float, cases: dict, island: int, origin: str,
            parent_id: str | None = None, sig_hash: str = "", card: str = "") -> Program:
        isl = self.islands[island]
        gen = 0 if parent_id is None or parent_id not in self._all \
            else self._all[parent_id].gen + 1
        prog = Program(id=self.new_id(), code=code, score=score, cases=cases,
                       island=island, origin=origin, parent_id=parent_id, gen=gen,
                       sig_hash=sig_hash, card=card)
        self._all[prog.id] = prog
        isl.programs[prog.id] = prog
        cell = self._cell(prog)
        cur = isl.grid.get(cell)
        if cur is None or prog.score > isl.programs[cur].score:
            isl.grid[cell] = prog.id
        if isl.best_id is None or prog.score > isl.programs[isl.best_id].score:
            isl.best_id = prog.id
        if self.pop_best_id is None or prog.score > self._all[self.pop_best_id].score:
            self.pop_best_id = prog.id
        self._update_archive(prog)
        self._enforce_cap(isl)
        return prog

    def _update_archive(self, prog: Program) -> None:
        self.archive.append(prog.id)
        self.archive.sort(key=lambda pid: self._all[pid].score, reverse=True)
        self.archive = self.archive[:self.archive_size]

    def _enforce_cap(self, isl: Island) -> None:
        if len(isl.programs) <= self.pop_cap:
            return
        protected = set(isl.grid.values()) | {isl.best_id}
        removable = [p for p in isl.programs.values() if p.id not in protected]
        removable.sort(key=lambda p: p.score)
        while len(isl.programs) > self.pop_cap and removable:
            victim = removable.pop(0)
            # eviction label: the current score-min policy dropped this member while
            # a protected/higher set stayed -- the target signal for the 4c head
            self.evictions.append({"island": isl.idx, "victim_id": victim.id,
                                   "victim_score": round(victim.score, 6),
                                   "victim_gen": victim.gen, "victim_origin": victim.origin,
                                   "n_kept": len(isl.programs) - 1})
            del isl.programs[victim.id]

    def inspirations(self, island: int, parent_id: str, k: int = 3) -> list[Program]:
        pool = [p for p in self.islands[island].top_k(k + 1) if p.id != parent_id]
        return pool[:k]

    def migrate_unique_elites(self, src_island: int, survivors: list[int],
                              origin: str = "migrant") -> int:
        if not survivors:
            return 0
        src = self.islands[src_island]
        seen = set()
        moved = 0
        for prog in src.top_k(3):
            h = normalized_hash(prog.code)
            target = survivors[moved % len(survivors)]
            if h in self.code_hashes(target) or h in seen:
                continue
            seen.add(h)
            # Migration copies an observed program; it does not start a new
            # research branch. Keep the concrete source ID (including repeated
            # migrations) so ancestry and branch-scoped knowledge survive SH.
            self.add(prog.code, prog.score, prog.cases, target, origin=origin,
                     parent_id=prog.id, card=prog.card)
            moved += 1
        return moved

    def population_snapshot(self) -> dict:
        """Compact population state for meta-cognition training: scores, MAP-Elites
        coverage, and lineage depth per island + archive scores -- NO code strings,
        to keep the journal light. This is the population-level stream that the
        per-iteration scalar log discards."""
        islands = []
        for isl in self.islands:
            progs = list(isl.programs.values())
            scores = sorted((p.score for p in progs), reverse=True)
            islands.append({
                "idx": isl.idx,
                "best": isl.best_score if progs else None,
                "n": len(progs),
                "cells": len(isl.grid),                       # MAP-Elites coverage
                "top": [round(s, 4) for s in scores[:5]],
                "score_mean": round(sum(scores) / len(scores), 4) if scores else None,
                "max_gen": max((p.gen for p in progs), default=0),  # lineage depth
                # per-cell elite: (length_bin, score_bin, elite_score)
                "grid": [[c[0], c[1], round(isl.programs[pid].score, 4)]
                         for c, pid in isl.grid.items()],
                # per-member state (id, score, gen): the set-level training food
                # that island aggregates discard (trace audit 2026-08-14)
                "members": [[p.id, round(p.score, 6), p.gen] for p in progs],
            })
        return {"pop_best": self.pop_best,
                "archive": [round(self._all[pid].score, 4) for pid in self.archive],
                "islands": islands}

    def state(self) -> dict:
        return {
            "counter": self._counter,
            "pop_best_id": self.pop_best_id,
            "archive": list(self.archive),
            "programs": [self._prog_json(p) for p in self._all.values()],
            "islands": [{"idx": isl.idx, "best_id": isl.best_id,
                         "grid": {f"{c[0]},{c[1]}": pid for c, pid in isl.grid.items()},
                         "members": list(isl.programs.keys())}
                        for isl in self.islands],
            # arch-v4 #4b: the VQ codebook is population state -- resume must not
            # re-grow it from scratch or niche ids would renumber
            "niche": self.niche_fn.state() if self.niche_fn is not None else None,
        }

    def add_island(self) -> int:
        """Dynamic niche spawn: a fresh empty island; returns its index."""
        idx = len(self.islands)
        self.islands.append(Island(idx))
        return idx

    def load_state(self, d: dict) -> None:
        self._counter = d["counter"]
        # Rollback is an exact replacement, including islands spawned only in
        # an uncommitted wave and transient eviction notifications from it.
        self.islands = [Island(row['idx']) for row in d['islands']]
        self.evictions = []
        self.pop_best_id = d["pop_best_id"]
        self.archive = list(d["archive"])
        self._all = {}
        for pj in d["programs"]:
            p = Program(**pj)
            self._all[p.id] = p
        for isl, ij in zip(self.islands, d["islands"]):
            isl.best_id = ij["best_id"]
            isl.programs = {pid: self._all[pid] for pid in ij["members"]}
            isl.grid = {tuple(int(x) for x in c.split(",")): pid
                        for c, pid in ij["grid"].items()}
        if self.niche_fn is not None and d.get("niche") is not None:
            self.niche_fn.load_state(d["niche"])

    @staticmethod
    def _prog_json(p: Program) -> dict:
        return {"id": p.id, "code": p.code, "score": p.score, "cases": p.cases,
                "card": p.card,
                "island": p.island, "origin": p.origin, "parent_id": p.parent_id,
                "gen": p.gen, "sig_hash": p.sig_hash}
