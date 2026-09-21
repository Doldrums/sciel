"""Deterministic island allocation with successive halving or round robin."""

from __future__ import annotations


class Saturated(Exception):
    pass


class Schedule:
    def __init__(self, num_islands: int, policy: str = "sh",
                 milestone_fracs: tuple = (0.5, 0.75), inflight_cap: int = 4,
                 min_fair_evals: int = 0, *, fair_dispatch: bool = False):
        if type(fair_dispatch) is not bool:
            raise ValueError('fair_dispatch must be a boolean')
        self.num_islands = num_islands
        self.policy = policy
        self.milestone_fracs = tuple(milestone_fracs)
        self.inflight_cap = inflight_cap
        self.min_fair_evals = min_fair_evals
        self.fair_dispatch = fair_dispatch
        self.survivors = list(range(num_islands))
        self.saturation_counter = {i: 0 for i in range(num_islands)}
        self.dispatched = {i: 0 for i in range(num_islands)}
        # island -> how many more dispatch decisions to skip it for. A duplicate storm
        # is a temporary property of one parent pool, not a death sentence: the old
        # code held a permanent `saturated` set that could only be cleared by a
        # successful child ON THAT ISLAND, which cannot happen while the island is
        # excluded from dispatch. One dup_cap iteration silently removed half the
        # search for the rest of the run, and when the last island followed it took
        # the whole run with it (outputs/heil_s56_pop ended at 92 of 100 evals).
        self.cooldown: dict = {}
        self.milestones_passed = 0
        self._rr_cursor = 0

    @property
    def phase(self) -> str:
        if self.policy == "round_robin":
            return "uniform"
        if self.milestones_passed == 0:
            return "uniform"
        if self.milestones_passed >= len(self.milestone_fracs):
            return "final"
        return f"sh_{self.milestones_passed}"

    def phase_code(self) -> int:
        p = self.phase
        if p == "uniform":
            return 0
        if p == "final":
            return -1
        return int(p.split("_")[1])

    SATURATION_COOLDOWN = 8          # dispatch decisions, not evaluations

    def pick_island(self, islands: list[int], in_flight: dict) -> int:
        pool = islands if self.phase == "uniform" else self.survivors
        if self.fair_dispatch:
            # A coordinator can omit islands whose parents are all reserved.
            # Retired islands must not re-enter just because a parent is free.
            allowed = set(islands)
            pool = [i for i in pool if i in allowed]
        hot = [i for i in pool if self.cooldown.get(i, 0) <= 0]
        if not hot and pool:
            # every island is cooling down: let them all back in rather than ending
            # the run. Runaway duplicates are already bounded by backstop_k.
            self.cooldown.clear()
            hot = list(pool)
        eligible = [i for i in hot if in_flight.get(i, 0) < self.inflight_cap]
        if not eligible:
            if not any(in_flight.values()):
                raise Saturated()
            return -1
        choice = (min(eligible, key=lambda i: (self.dispatched.get(i, 0), i))
                  if self.fair_dispatch else eligible[self._rr_cursor % len(eligible)])
        self._rr_cursor += 1
        self.dispatched[choice] += 1
        for i in list(self.cooldown):
            self.cooldown[i] -= 1
            if self.cooldown[i] <= 0:
                del self.cooldown[i]
        return choice

    def register_island(self, idx: int) -> None:
        """Dynamic spawn: a new island joins the live pool mid-run.

        Add it to survivors for future milestones and grow the island count for
        state restoration and round-robin scheduling."""
        if idx not in self.survivors:
            self.survivors.append(idx)
        self.saturation_counter.setdefault(idx, 0)
        self.dispatched.setdefault(idx, 0)
        self.num_islands = max(self.num_islands, idx + 1)

    def note_saturation(self, island: int) -> None:
        """This island just exhausted dup_cap. Rest it, do not retire it."""
        self.saturation_counter[island] = self.saturation_counter.get(island, 0) + 1
        self.cooldown[island] = self.SATURATION_COOLDOWN

    def note_progress(self, island: int) -> None:
        self.cooldown.pop(island, None)

    @property
    def saturated(self) -> set:
        """Islands currently resting. Read-only; kept for journals and tests."""
        return {i for i, left in self.cooldown.items() if left > 0}

    def milestones_for(self, budget: int) -> list[int]:
        return [max(1, int(round(f * budget))) for f in self.milestone_fracs]

    def on_milestone(self, milestone_value: int, db, log) -> None:
        self.cooldown.clear()
        if self.policy == "round_robin":
            log.sh_cut(milestone=milestone_value, ranking=list(range(self.num_islands)),
                       cut=[], protected=[], saturation=dict(self.saturation_counter),
                       migrations=0, policy="round_robin")
            return
        # Halve the live pool. register_island adds new islands to this pool;
        # previously cut islands must not be ranked or migrated a second time.
        alive = sorted(set(self.survivors))
        ranking = sorted(alive,
                         key=lambda i: (db.island_best(i),
                                        -self.saturation_counter.get(i, 0)),
                         reverse=True)
        keep = (len(ranking) + 1) // 2
        survivors = ranking[:keep]
        cut = ranking[keep:]
        # Fair-chance guard: don't cut an island that hasn't yet had min_fair_evals
        # dispatched to it -- a late bloomer (or a rr/inflight-starved island) may
        # still be pre-first-record. Defer its cut; it is re-ranked next milestone.
        protected = []
        if self.min_fair_evals > 0:
            protected = [i for i in cut if self.dispatched[i] < self.min_fair_evals]
            if protected:
                survivors = survivors + protected
                cut = [i for i in cut if i not in protected]
        migrations = 0
        for isl in cut:
            migrations += db.migrate_unique_elites(isl, survivors, origin="migrant")
        self.survivors = survivors
        self.milestones_passed += 1
        log.sh_cut(milestone=milestone_value, ranking=ranking, cut=cut,
                   protected=protected, saturation=dict(self.saturation_counter),
                   migrations=migrations, phase=self.phase)

    def state(self) -> dict:
        return {"policy": self.policy, "survivors": list(self.survivors),
                "saturation_counter": dict(self.saturation_counter),
                "dispatched": dict(self.dispatched),
                "min_fair_evals": self.min_fair_evals,
                "fair_dispatch": self.fair_dispatch,
                "cooldown": dict(self.cooldown),
                "num_islands": self.num_islands,
                "milestones_passed": self.milestones_passed,
                "rr_cursor": self._rr_cursor}

    def load_state(self, d: dict) -> None:
        fair_dispatch = d.get('fair_dispatch', False)
        if type(fair_dispatch) is not bool:
            raise ValueError('fair_dispatch must be a boolean')
        self.fair_dispatch = fair_dispatch
        self.policy = d["policy"]
        self.survivors = list(d["survivors"])
        self.saturation_counter = {int(k): v for k, v in d["saturation_counter"].items()}
        self.num_islands = max(self.num_islands,
                               max(self.saturation_counter, default=-1) + 1)
        self.dispatched = ({int(k): v for k, v in d["dispatched"].items()}
                           if d.get("dispatched") else {i: 0 for i in range(self.num_islands)})
        self.min_fair_evals = d.get("min_fair_evals", self.min_fair_evals)
        # pre-cooldown checkpoints stored a bare set under "saturated"
        self.cooldown = {int(k): int(v) for k, v in (d.get("cooldown") or {}).items()}
        for i in d.get("saturated", []):
            self.cooldown.setdefault(int(i), self.SATURATION_COOLDOWN)
        self.num_islands = int(d.get("num_islands", self.num_islands))
        self.milestones_passed = d["milestones_passed"]
        self._rr_cursor = d["rr_cursor"]
