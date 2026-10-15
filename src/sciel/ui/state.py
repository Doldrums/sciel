"""A live, read-only view of one run directory: the journal tail, the runtime sidecar and the last checkpoint.

The viewer never writes to the run. It replays `sciel_events.jsonl` incrementally and folds it into a snapshot
that both the terminal and the web page draw from.
"""

from __future__ import annotations

import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

MAX_EVENTS = 60
MAX_IDEA = 160
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")
CELL_STATES = {"create": "writing a tool", "finish": "reached a conclusion", "cell_revision": "revising",
               "sharing_completed": "shared with islands", "abandon": "abandoned"}


def _tokens(code):
    return Counter(_WORD.findall(code)) if isinstance(code, str) else None


def _seconds(stamp):
    try:
        return datetime.fromisoformat(stamp).timestamp()
    except (TypeError, ValueError):
        return None


def _idea(row):
    bet = row.get("bet")
    text = bet.get("mechanism") if isinstance(bet, dict) else None
    text = " ".join((text or "").split())
    return text[:MAX_IDEA] if text else None


def _numbers(cases):
    if not isinstance(cases, dict):
        return {}
    return {k: float(v) for k, v in cases.items()
            if type(v) in (int, float) and v == v and k != "combined_score" and not k.startswith(("us_", "fit_"))}


class RunState:
    def __init__(self, run_dir):
        self.dir = Path(run_dir)
        self._offset = 0
        self._checkpoint_mtime = None
        self.run = {"dir": str(self.dir), "task": None, "run_id": None, "budget": None, "models": [],
                    "seed_score": None, "best": None, "B_evals": 0, "iteration": 0, "cost": 0.0, "started": None}
        self.history = []
        self.programs = {}
        self.alive = set()
        self.islands = {}
        self.transfers = []
        self.findings = {}
        self.claims = []
        self.cells = {}
        self.events = []
        self.edits = []
        self.code = {}
        self._finding_island = {}

    def refresh(self) -> dict:
        self._read_journal()
        self._read_checkpoint()
        return self.snapshot()

    def _read_journal(self):
        path = self.dir / "sciel_events.jsonl"
        if not path.exists():
            return
        with open(path, "rb") as stream:
            stream.seek(self._offset)
            data = stream.read()
        end = data.rfind(b"\n") + 1
        self._offset += end
        for line in data[:end].splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            self._fold(row)

    def _note(self, row, text, kind="info"):
        self.events.append({"t": row.get("timestamp"), "text": text, "kind": kind})
        del self.events[:-MAX_EVENTS]

    def _fold(self, row):
        event = row.get("event")
        if event == "run_start":
            self.run.update(task=row.get("task"), run_id=row.get("run_id"), budget=row.get("budget"),
                            models=row.get("models") or [], started=self.run["started"] or row.get("timestamp"))
            self._note(row, "run resumed" if row.get("resumed") else "run started")
        elif event == "iteration":
            self._iteration(row)
        elif event == "population":
            self._population(row)
        elif event == "research_tool":
            self._cell(row)
        elif event == "decision":
            self._decision(row)

    def _decision(self, row):
        policy, action = row.get("policy"), row.get("action")
        if policy == "initialization" and action == "seed_program":
            pid = row.get("program_id")
            if pid:
                self.programs[pid] = {"id": pid, "island": row.get("island"), "parent": None, "score": row.get("score"),
                                      "gen": 0, "op": "seed", "idea": None, "B_evals": row.get("B_evals"), "cases": {}}
                if _tokens(row.get("source")):
                    self.code[pid] = _tokens(row.get("source"))
            if row.get("seed_score") is not None:
                self.run["seed_score"] = row["seed_score"]
                self._best(row.get("B_evals") or 0, row["seed_score"])
        elif policy == "finding" and action == "add":
            for item in row.get("findings") or []:
                self._finding_island[item.get("id")] = item.get("island")
                self.findings.setdefault(item.get("id"), {"id": item.get("id"), "text": item.get("text"),
                                                          "island": item.get("island"), "status": "trial"})
                self._note(row, f"finding {item.get('id')} from I{item.get('island')}", "finding")
        elif policy == "cost_accounting":
            self._note(row, f"accounting: {action}", "warn")

    def _best(self, evals, score):
        if score is None:
            return
        if self.run["best"] is None or score > self.run["best"]:
            self.run["best"] = score
            self.history.append([evals, score])

    def _iteration(self, row):
        child, island = row.get("child_id"), row.get("island")
        parent = row.get("parent_id")
        score = row.get("child_score")
        self.run.update(B_evals=row.get("B_evals", self.run["B_evals"]),
                        iteration=row.get("iteration_no", self.run["iteration"]),
                        cost=self.run["cost"] + (row.get("cost") or 0.0))
        if child:
            gen = (self.programs.get(parent, {}).get("gen") or 0) + 1
            self.programs[child] = {"id": child, "island": island, "parent": parent, "score": score, "gen": gen,
                                    "op": row.get("op_used"), "idea": _idea(row), "B_evals": row.get("B_evals"),
                                    "cases": _numbers(row.get("cases"))}
            if _tokens(row.get("child_code")):
                self.code[child] = _tokens(row.get("child_code"))
            parent_score = self.programs.get(parent, {}).get("score")
            if parent_score is not None and score is not None:
                self.edits.append([parent_score, score, island])
            self._transfer(row, child, island, parent_score, score)
        self.islands.setdefault(island, {}).update(mode=row.get("effective_mode"), op=row.get("op_used"),
                                                   last=score, idea=_idea(row))
        best_before = self.run["best"]
        self._best(row.get("B_evals") or 0, score)
        if child and score is not None:
            mark = " ▲ best" if best_before is not None and score > best_before else ""
            kind = "best" if mark else ("drop" if score == 0 else "child")
            self._note(row, f"I{island} {child} {score:.3f}{mark}", kind)

    def _transfer(self, row, child, island, parent_score, score):
        context = row.get("decision_context")
        decisions = context.get("decisions", []) if isinstance(context, dict) else []
        for decision in decisions:
            if decision.get("decision") != "finding" or decision.get("choice") in (None, "none"):
                continue
            source = self._finding_island.get(decision["choice"])
            if source is None or source == island:
                continue
            rose = parent_score is not None and score is not None and score > parent_score
            self.transfers.append({"finding": decision["choice"], "from_island": source, "child": child,
                                   "island": island, "rose": rose})

    def _population(self, row):
        self.alive = set()
        for isl in row.get("islands") or []:
            idx = isl.get("idx")
            self.islands.setdefault(idx, {}).update(best=isl.get("best"), n=isl.get("n"))
            for member in isl.get("members") or []:
                pid, score, gen = member[0], member[1], member[2]
                self.alive.add(pid)
                known = self.programs.setdefault(pid, {"id": pid, "island": idx, "parent": None, "score": score,
                                                       "gen": gen, "op": None, "idea": None, "B_evals": None,
                                                       "cases": {}})
                known["gen"] = gen
        self.run["B_evals"] = row.get("B_evals", self.run["B_evals"])
        self._best(row.get("B_evals") or 0, row.get("pop_best"))

    def _cell(self, row):
        cell = row.get("cell_id")
        if not cell:
            return
        entry = self.cells.setdefault(cell, {"id": cell[:6], "cell": cell, "question": None,
                                             "island": row.get("island_id"), "program": None, "state": "open"})
        entry["program"] = row.get("parent_program_id") or entry["program"]
        if row.get("question"):
            entry["question"] = " ".join(str(row["question"]).split())[:200]
        action = row.get("action") or row.get("phase")
        if action:
            entry["state"] = CELL_STATES.get(action, str(action).replace("_", " "))
            if action == "create":
                self._note(row, f"cell {entry['id']} writes a tool", "cell")
            elif action == "finish":
                self._note(row, f"cell {entry['id']} reached a conclusion", "cell")

    def _read_checkpoint(self):
        path = self.dir / "checkpoint.json"
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return
        if mtime == self._checkpoint_mtime:
            return
        try:
            state = json.loads(path.read_text())
        except (OSError, ValueError):
            return
        self._checkpoint_mtime = mtime
        for item in (state.get("findings") or {}).get("pool", []):
            self.findings[item["id"]] = {"id": item["id"], "text": item.get("text"), "island": item.get("island"),
                                         "status": item.get("status"), "kind": item.get("kind"),
                                         "cell": item.get("cell_id"), "delivered": item.get("delivered", 0)}
            self._finding_island[item["id"]] = item.get("island")
        self.claims = [c for c in (state.get("theory") or {}).get("claims", []) if isinstance(c, dict)]
        if state.get("seed_score") is not None:
            self.run["seed_score"] = state["seed_score"]
        self.run["cost"] = max(self.run["cost"], state.get("cost_total") or 0.0)
        self.run["budget"] = state.get("budget", self.run["budget"])

    def _status(self):
        try:
            runtime = json.loads((self.dir / "runtime.json").read_text())
        except (OSError, ValueError):
            return {"status": "unknown", "stage": None}
        status = runtime.get("status")
        beat = _seconds(runtime.get("heartbeat_at"))
        stale = runtime.get("stale_after_seconds") or 10
        if status == "running" and (beat is None or time.time() - beat > stale):
            status = "lost"
        return {"status": status, "stage": runtime.get("stage"), "reason": runtime.get("reason"),
                "parallel": runtime.get("parallel")}

    def _eta(self):
        start, budget, done = _seconds(self.run["started"]), self.run["budget"], self.run["B_evals"]
        if start is None or not budget or not done or done >= budget:
            return None
        return (time.time() - start) / done * (budget - done)

    def snapshot(self) -> dict:
        best_id = max((p for p in self.programs.values() if p.get("score") is not None),
                      key=lambda p: p["score"], default=None)
        lineage = []
        node = best_id
        while node is not None and len(lineage) < 30:
            lineage.append(node)
            node = self.programs.get(node.get("parent")) if node.get("parent") else None
        return {
            "run": {**self.run, **self._status(), "eta_seconds": self._eta(),
                    "now": datetime.now(timezone.utc).isoformat()},
            "history": self.history,
            "islands": [{"id": k, **v} for k, v in sorted(self.islands.items(), key=lambda kv: kv[0] if kv[0] is not None else -1)],
            "programs": [{**p, "alive": p["id"] in self.alive} for p in self.programs.values()],
            "best": best_id,
            "lineage": lineage,
            "transfers": self.transfers,
            "findings": list(self.findings.values()),
            "claims": self.claims,
            "cells": list(self.cells.values()),
            "events": self.events,
            "edits": self.edits,
        }
