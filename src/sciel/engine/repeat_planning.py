"""Repeat draw planning and bounded candidate scheduling."""
from __future__ import annotations

from copy import deepcopy
import hashlib
from .repeat_journal import encoded, immutable_file
from .repeat_policy import (RepeatPromotionError, SEED_CONTRACT, _finite, _source_sha)


class RepeatPlanning:

    def _seed(self, sequence, index):
        key = [SEED_CONTRACT, self.config["seed_master"], sequence, index]
        return int.from_bytes(hashlib.sha256(encoded(key)).digest()[:4], "big")

    def _planned_draws(self, sequence, kind, challenger, incumbent):
        batch_id = f"b{sequence:06d}"
        draws = []
        for index in range(3):
            roles = ["bootstrap"] if kind == "bootstrap" else ["incumbent", "challenger"]
            if kind != "bootstrap" and (sequence + index) % 2:
                roles.reverse()
            for role in roles:
                sha = incumbent if role == "incumbent" else challenger
                draws.append({"draw_id": f"{batch_id}-{role}-{index}", "role": role, "draw_index": index,
                              "source_sha256": sha, "seed": self._seed(sequence, index)})
        return draws

    def _can_plan(self):
        if self._data is None or self._lock is None:
            raise RepeatPromotionError("repeat coordinator not started")
        self._check_live()
        if self.pending:
            raise RepeatPromotionError("a repeat batch is already pending")
        return self.stopped_reason is None

    def plan_bootstrap(self, code):
        if not self._can_plan():
            return False
        if self._data["batches"]:
            return False
        if _source_sha(code) != self.identity["canonical_initial_source_sha256"]:
            raise RepeatPromotionError("bootstrap must use canonical initial source")
        return self._plan(code, "bootstrap", None, None)

    def plan_candidate(self, code, cases, program_id, *, research_context=None):
        if not self._can_plan():
            return False
        screening = self._screen_candidate(code, cases, program_id)
        if screening is None:
            return False
        return self._plan(code, "comparison", program_id, screening,
                          research_context=research_context)

    @staticmethod
    def _research_context(value, challenger):
        if value is None:
            return None
        if (not isinstance(value, dict) or set(value) !=
                {"branch", "origin_experiment_id", "source_sha256"}
                or not isinstance(value["branch"], str) or not value["branch"].strip()
                or len(value["branch"]) > 128 or value["source_sha256"] != challenger
                or (value["origin_experiment_id"] is not None and
                    (not isinstance(value["origin_experiment_id"], str)
                     or not value["origin_experiment_id"].strip()
                     or len(value["origin_experiment_id"]) > 128))):
            raise RepeatPromotionError("invalid source-bound repeat research context")
        return deepcopy(value)

    def _screen_candidate(self, code, cases, program_id):
        sha = _source_sha(code)
        if self._data["confirmed"] is None:
            raise RepeatPromotionError("candidate comparison requires confirmed bootstrap")
        if not isinstance(program_id, str) or not program_id:
            raise ValueError("candidate program ID required")
        if sha == self._data["confirmed"]["source_sha256"] or sha in self._data["claimed_sources"]:
            return None
        if (not isinstance(cases, dict) or not _finite(cases.get("validity"))
                or cases["validity"] != 1 or not _finite(cases.get("sum_radii"))):
            return None
        if cases["sum_radii"] <= self._data["confirmed"]["metrics"]["raw"]["minimum"]:
            return None
        return {"sum_radii": cases["sum_radii"], "validity": cases["validity"]}

    def note_unexamined_candidate(self, code, cases, program_id, *, reason):
        """Retain an eligible source when its search step cannot reach a safe point.

        This is not a draw reservation, measurement, or promotion. No further
        repeats are dispatched after the interrupted search step.
        """
        if not self._can_plan():
            return False
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 200:
            raise ValueError("unexamined candidate needs a bounded stop reason")
        screening = self._screen_candidate(code, cases, program_id)
        if screening is None:
            return False
        sha = _source_sha(code)
        immutable_file(self.directory / "sources" / (sha + ".py"), code.encode("utf-8"))
        self._append({"type": "unexamined_candidate", "source_sha256": sha,
                      "program_id": program_id, "screening": screening, "reason": reason})
        return True

    def _plan(self, code, kind, program_id, screening, *, research_context=None):
        sha = _source_sha(code)
        research_context = self._research_context(research_context, sha)
        count = 3 if kind == "bootstrap" else 6
        counters = self.summary()["counters"]
        allocation = count * (self.config["timeout_seconds"] + self.config["certification_seconds"])
        if (counters["reserved_draw_attempts"] + count > self.config["max_draw_attempts"]
                or counters["planned_wall_seconds"] + allocation > self.config["max_wall_seconds"]
                or (kind == "comparison" and counters["comparisons"] >= self.config["max_comparisons"])):
            self._append({"type": "budget_unavailable", "source_sha256": sha, "kind": kind,
                          "program_id": program_id, "screening": screening})
            return False
        immutable_file(self.directory / "sources" / (sha + ".py"), code.encode("utf-8"))
        sequence = len(self._data["batches"])
        incumbent = self._data["confirmed"]["source_sha256"] if kind == "comparison" else None
        plan = {"batch_id": f"b{sequence:06d}", "sequence": sequence, "kind": kind,
                "challenger": sha, "incumbent": incumbent, "program_id": program_id,
                "screening": screening,
                "draws": self._planned_draws(sequence, kind, sha, incumbent),
                "reserved_wall_seconds": allocation}
        if research_context is not None:
            plan["research_context"] = research_context
        self._append({"type": "plan", "plan": plan})
        self._ack = None
        return True
