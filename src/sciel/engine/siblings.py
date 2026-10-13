"""Sibling blocks: one parent, one operator, one evaluation seed, one child per knowledge item under test plus a
placebo control, so an item's effect is a paired score difference on real search instead of a rare island event.
"""

from __future__ import annotations

import difflib
import re
from copy import deepcopy
from statistics import median

BLOCK_SHARE = 0.5
ITEMS = 2
SCALE_WINDOW = 200
MIN_SCALE = 1e-6
PLACEBOS = (
    "Keep the program readable: descriptive names, short functions and no unused code.",
    "Prefer clear control flow over clever one-liners, and keep helper functions close to where they are used.",
    "Keep the public interface of the program exactly as it is, and make every change easy to read in a diff. "
    "Avoid introducing new global state or module-level side effects.",
    "Write the program so that a reader can follow it from top to bottom: put constants first, helpers next and "
    "the entry point last, and keep each function focused on one job. Remove code that no longer serves the "
    "current design.",
)
_WORD = re.compile(r"[a-z][a-z0-9_]{3,}")


def _placebo(length: int) -> str:
    return min(PLACEBOS, key=lambda text: abs(len(text) - length))


def _applied(text: str, parent: str, child: str) -> bool:
    words = set(_WORD.findall(text.lower()))
    added = " ".join(line[1:] for line in difflib.unified_diff(parent.splitlines(), child.splitlines(), lineterm="")
                     if line.startswith("+") and not line.startswith("+++")).lower()
    hits = {w for w in words if w in added}
    return bool(words) and (len(hits) >= 2 or len(hits) / len(words) >= 0.2)


class Siblings:
    def __init__(self):
        self.next_id = 0
        self.open: dict = {}
        self.launched: list[int] = []
        self.pairs: list[dict] = []
        self.counts: dict[str, int] = {}
        self.deltas: list[float] = []

    def candidates(self, engine, island: int) -> list[tuple]:
        from ..theory import schools
        strategies = [("strategy", s.id, "Strategy to try in this change: " + s.text)
                      for s in schools.trials(engine.strategies.pool, island)]
        findings = [("finding", f.id, "Knowledge from this run:\n" + f.card())
                    for f in schools.trials(engine.findings.pool, island)]
        return sorted(strategies + findings, key=lambda item: self.counts.get(item[1], 0))[:ITEMS]

    def scale(self) -> float:
        return max(median(self.deltas), MIN_SCALE) if self.deltas else 1.0

    def draw(self, engine, island: int, room: int):
        """Whether this step is a block, journaled like any decision; None when nothing is under test or no room."""
        items = self.candidates(engine, island)[:max(room - 1, 0)]
        if not items:
            return None
        choice = engine.decisions.sample("sibling", {"single": 1 - BLOCK_SHARE, "block": BLOCK_SHARE})
        if choice != "block":
            return None
        block = {"id": self.next_id, "items": [list(item) for item in items], "scale": self.scale(),
                 "seed": engine.rng.derive_seed("sibling") % 2 ** 31}
        self.next_id += 1
        return block

    def expand(self, plan: dict) -> list[dict]:
        """The control member keeps the step's decisions; item members share its parent, operator and seed."""
        block = plan.get("block")
        if block is None:
            return [plan]
        base = plan["snap"].directive
        texts = [item[2] for item in block["items"]]
        arms = [("control", "\n" + _placebo(max(len(t) for t in texts)) + "\n")] + [
            (item[1], "\n" + item[2] + "\n") for item in block["items"]]
        members = []
        for index, (arm, text) in enumerate(arms):
            member = plan if index == 0 else {**plan, "snap": deepcopy(plan["snap"]), "theorist": None,
                                               "assumptions": None,
                                               "decision_context": {**plan["decision_context"], "decisions": []}}
            if index:
                member["snap"].operator_decision = {**(member["snap"].operator_decision or {}), "on_policy": False}
                member.update(mode_on_policy=False, preset=None)
            member["snap"].directive = base + text
            member["arm"] = arm
            members.append(member)
        self.open[block["id"]] = {"items": block["items"], "scale": block["scale"],
                                  "parent": plan["snap"].parent_score, "parent_code": plan["snap"].parent_code,
                                  "scores": {}, "codes": {}, "arms": [arm for arm, _ in arms],
                                  "iteration": plan["snap"].iteration_no}
        self.launched.append(block["id"])
        return members

    def observe(self, parent_score: float, score: float) -> None:
        self.deltas = (self.deltas + [abs(score - parent_score)])[-SCALE_WINDOW:]

    def record(self, plan: dict, score: float | None, code: str | None) -> list[dict]:
        """Store a member's admitted score; return the pairs of blocks completed in launch order."""
        block = self.open.get(plan["block"]["id"])
        if block is None:
            return []
        block["scores"][plan["arm"]] = block["parent"] - block["scale"] if score is None else score
        block["codes"][plan["arm"]] = code or ""
        done = []
        while self.launched and len(self.open[self.launched[0]]["scores"]) == len(self.open[self.launched[0]]["arms"]):
            block_id = self.launched.pop(0)
            finished = self.open.pop(block_id)
            control = finished["scores"]["control"]
            for kind, item, text in finished["items"]:
                difference = (finished["scores"][item] - control) / finished["scale"]
                pair = {"block": block_id, "kind": kind, "item": item, "iteration": finished["iteration"],
                        "d": max(-1.0, min(1.0, difference)),
                        "applied": _applied(text, finished["parent_code"], finished["codes"][item])}
                self.counts[item] = self.counts.get(item, 0) + 1
                self.pairs.append(pair)
                done.append(pair)
        return done

    def state(self) -> dict:
        return {"next_id": self.next_id, "open": {str(k): v for k, v in self.open.items()},
                "launched": self.launched, "pairs": self.pairs, "counts": self.counts, "deltas": self.deltas}

    def load_state(self, data: dict) -> None:
        self.next_id = int(data["next_id"])
        self.open = {int(k): v for k, v in data["open"].items()}
        self.launched = [int(k) for k in data["launched"]]
        self.pairs = list(data["pairs"])
        self.counts = dict(data["counts"])
        self.deltas = list(data["deltas"])
