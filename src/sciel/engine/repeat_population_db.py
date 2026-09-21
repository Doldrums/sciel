"""Immutable ordinary history with an explicit repeat-controlled active projection."""
from __future__ import annotations

import copy
import math

from .db import Database
from .types import Program


class RepeatPopulationDatabase(Database):
    """add records observations only; project is the sole membership writer.

    pop_best_id names the best ordinary *historical* observation, including
    rejected candidates. Neither it nor the legacy grid/archive selects parents.
    """
    def __init__(self, num_islands, pop_cap):
        super().__init__(num_islands, pop_cap, archive_size=0)

    def add(self, code, score, cases, island, origin, parent_id=None, sig_hash='', card=''):
        if (type(island) is not int or not 0 <= island < len(self.islands)
                or type(score) not in (int, float) or not math.isfinite(score)
                or type(code) is not str or type(cases) is not dict
                or (parent_id is not None and parent_id not in self._all)):
            raise ValueError('invalid ordinary history observation')
        gen = 0 if parent_id is None else self._all[parent_id].gen + 1
        p = Program(self.new_id(), code, score, copy.deepcopy(cases), island, origin,
                    parent_id, gen, sig_hash, card)
        self._all[p.id] = p
        if self.pop_best_id is None or score > self._all[self.pop_best_id].score:
            self.pop_best_id = p.id
        return p

    def project(self, population):
        plans = []
        for island in self.islands:
            ids = population.active_ids(island.idx)
            if (len(ids) > self.pop_cap or len(set(ids)) != len(ids)
                    or any(pid not in self._all or self._all[pid].island != island.idx for pid in ids)):
                raise ValueError('repeat membership differs from ordinary history')
            plans.append({pid: self._all[pid] for pid in ids})
        for island, programs in zip(self.islands, plans):
            island.programs = programs
            island.grid = {}
            # This field remains explicitly an ordinary-score display statistic.
            island.best_id = max(programs, key=lambda pid: programs[pid].score) if programs else None
        self.archive = []
        self.evictions.clear()

    def load_state(self, state):
        # Validate into a detached DB; do not partially mutate on malformed data.
        if len(state.get('islands', [])) != len(self.islands) or state.get('archive') != [] or state.get('niche') is not None:
            raise ValueError('repeat history topology/archive differs')
        other = RepeatPopulationDatabase(len(self.islands), self.pop_cap)
        Database.load_state(other, copy.deepcopy(state))
        if (len(other._all) != len(state['programs']) or type(other._counter) is not int
                or other._counter != len(other._all)
                or set(other._all) != {f'p{i}' for i in range(1, other._counter + 1)}):
            raise ValueError('repeat history program counter differs')
        for island in other.islands:
            if (island.grid or len(island.programs) > self.pop_cap
                    or any(p.island != island.idx for p in island.programs.values())):
                raise ValueError('invalid repeat active projection')
            expected = max(island.programs, key=lambda p: island.programs[p].score) if island.programs else None
            if island.best_id != expected:
                raise ValueError('ordinary island statistic differs')
        for p in other._all.values():
            if (type(p.score) not in (int, float) or not math.isfinite(p.score)
                    or type(p.code) is not str or type(p.cases) is not dict
                    or type(p.island) is not int or not 0 <= p.island < len(other.islands)
                    or (p.parent_id is not None and p.parent_id not in other._all)):
                raise ValueError('invalid stored ordinary observation')
        best = max(other._all, key=lambda pid: other._all[pid].score) if other._all else None
        if other.pop_best_id != best:
            raise ValueError('ordinary historical best differs')
        self.__dict__.update(other.__dict__)
