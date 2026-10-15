"""What the engine is doing this second: the jobs of the current parallel wave and their calls in flight.

A call is in flight from its durable dispatch record until its receipt appears. Finished calls never change,
so they are read once.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

OPS = {"diff": "edit", "full_rewrite": "rewrite", "cross": "crossover", "idea": "new idea",
       "error_repair": "repair", "portfolio": "portfolio seed"}
DOING = {"llm": "writing", "evaluation": "evaluating", "tool_execution": "running tool"}
JOBS = {"A": "assay", "R": "research cell", "portfolio": "first design"}


def _seconds(stamp):
    try:
        return datetime.fromisoformat(stamp).timestamp()
    except (TypeError, ValueError):
        return None


def _load(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


class Activity:
    def __init__(self, run_dir):
        self.dir = Path(run_dir) / ".parallel"
        self._calls = {}
        self._jobs = (None, None, {})

    def _wave(self):
        pointer = _load(self.dir / "current.json")
        return pointer.get("wave_id") if isinstance(pointer, dict) else None

    def _job_table(self, wave_dir: Path):
        path = wave_dir / "jobs.json"
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return {}
        if self._jobs[:2] == (str(wave_dir), mtime):
            return self._jobs[2]
        table = {}
        for job in (_load(path) or {}).get("jobs", []):
            snap = job.get("snapshot") or {}
            table[job.get("job_id")] = {"island": snap.get("island"), "op": snap.get("op"),
                                        "parent": snap.get("parent_id")}
        self._jobs = (str(wave_dir), mtime, table)
        return table

    def _call(self, call_dir: Path):
        known = self._calls.get(call_dir)
        if known is not None and known["done"]:
            return known
        if known is None:
            intent = _load(call_dir / "intent.json") or {}
            dispatched = _load(call_dir / "dispatched.json") or {}
            known = {"key": intent.get("key"), "kind": intent.get("kind"), "since": dispatched.get("timestamp"),
                     "done": False, "seconds": None, "status": None}
        receipt = _load(call_dir / "receipt.json")
        if receipt is not None:
            payload = receipt.get("payload", {})
            known.update(done=True, seconds=payload.get("elapsed_seconds"), status=payload.get("status"),
                         ended=payload.get("completed_at"))
        self._calls[call_dir] = known
        return known

    def read(self) -> dict:
        wave = self._wave()
        if wave is None:
            return {"wave": None, "jobs": [], "last": None}
        wave_dir = self.dir / f"wave-{wave}"
        table = self._job_table(wave_dir)
        calls = [self._call(d) for d in sorted((wave_dir / "calls").glob("*")) if d.is_dir()]
        by_job = {}
        for call in calls:
            if call["key"]:
                by_job.setdefault(call["key"].rsplit(":", 1)[0], []).append(call)
        now = time.time()
        jobs, last = [], None
        for job_id in sorted(set(table) | set(by_job)):
            own = sorted(by_job.get(job_id, []), key=lambda c: c.get("since") or "")
            info = table.get(job_id, {})
            current = next((c for c in reversed(own) if not c["done"]), None)
            for c in own:
                for stamp in (c.get("since"), c.get("ended")):
                    if stamp and (last is None or stamp > last):
                        last = stamp
            if current is not None:
                started = _seconds(current["since"])
                doing, seconds = DOING.get(current["kind"], current["kind"] or "working"), (
                    now - started if started else None)
            elif own:
                doing, seconds = "done", sum(c["seconds"] or 0 for c in own)
            else:
                doing, seconds = "queued", None
            prefix = job_id.split(":")[0]
            what = OPS.get(info.get("op"), info.get("op")) if prefix == "G" else JOBS.get(prefix, prefix)
            parent = info.get("parent") if prefix == "G" else None
            jobs.append({"job": job_id, "island": info.get("island"), "what": what, "parent": parent,
                         "doing": doing, "since": current["since"] if current else None, "seconds": seconds,
                         "calls": len(own)})
        return {"wave": wave, "jobs": jobs, "last": last}
