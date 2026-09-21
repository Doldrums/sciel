"""Re-evaluate stored programs to populate missing behavior descriptors.

Runs the evaluator's full stage without model calls and preserves the original
measurement alongside the recomputed result.

Usage: python scripts/backfill_behavior.py corpus.jsonl outputs --workers 8"""

from __future__ import annotations

import argparse
import concurrent.futures
import glob
import hashlib
import json
import os
import sys
import tempfile
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from sciel.eval.registry import load_task                     # noqa: E402
from sciel.eval.sandbox import run_stage                      # noqa: E402

TASKS_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tasks")


def collect(roots):
    """(task, code, stored_score) for every distinct program, deduped per task."""
    out, seen = [], set()
    for root in roots:
        for cp in sorted(glob.glob(os.path.join(root, "**", "checkpoint.json"),
                                   recursive=True)):
            try:
                state = json.load(open(cp))
            except (json.JSONDecodeError, OSError):
                continue
            task = ((state.get("config") or {}).get("task")) or ""
            if not task:
                continue
            for p in (state.get("db") or {}).get("programs") or []:
                code = p.get("code") or ""
                if not code:
                    continue
                key = (task, hashlib.blake2b(code.encode(), digest_size=16).digest())
                if key in seen:
                    continue
                seen.add(key)
                try:
                    stored = float(p.get("score") or 0.0)
                except (TypeError, ValueError):
                    stored = 0.0
                out.append({"task": task, "code": code, "stored_score": stored})
    return out


def stratify(items, per_task):
    """At most `per_task` per task, spread evenly over the stored score range.

    Taking the first N would take them in checkpoint order, which is run order, so the
    sample would be a handful of runs rather than the corpus."""
    by_task = {}
    for it in items:
        by_task.setdefault(it["task"], []).append(it)
    out = []
    for task, group in sorted(by_task.items()):
        if len(group) <= per_task:
            out += group
            continue
        group.sort(key=lambda x: x["stored_score"])
        step = len(group) / per_task
        out += [group[int(i * step)] for i in range(per_task)]
    return out


def evaluate_one(item, specs):
    spec = specs.get(item["task"])
    if spec is None:
        return None
    fd, path = tempfile.mkstemp(suffix=".py", prefix="backfill_")
    with os.fdopen(fd, "w") as f:
        f.write(item["code"])
    try:
        res = run_stage(spec.evaluator_path, "full", path,
                        timeout=spec.full_timeout, cpu_seconds=spec.cpu_seconds,
                        mem_bytes=spec.mem_bytes, python=sys.executable)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
    return {"task": item["task"], "code": item["code"], "ok": bool(res.ok),
            "score": float(res.score), "stored_score": item["stored_score"],
            "cases": {k: float(v) for k, v in (res.cases or {}).items()}}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0, help="stop after N programs")
    ap.add_argument("--per-task", type=int, default=0,
                    help="cap per task, sampled evenly across the score range. Re-running "
                         "every stored program is not affordable: evolved programs run "
                         "real optimisation loops, and a full heilbronn pass measured "
                         "~20s each, i.e. about nine hours on eight workers. A few "
                         "thousand spanning the range trains the embedder just as well.")
    ap.add_argument("--tasks-root", default=TASKS_ROOT)
    a = ap.parse_args(argv)

    items = collect(a.roots)
    if a.per_task:
        items = stratify(items, a.per_task)
    if a.limit:
        items = items[:a.limit]
    tasks = sorted({i["task"] for i in items})
    specs = {}
    for t in tasks:
        try:
            specs[t] = load_task(os.path.join(a.tasks_root, t))
        except Exception as e:                                  # noqa: BLE001
            print(f"  skip task {t}: {e}", file=sys.stderr)
    print(f"programs: {len(items)} over {len(specs)} tasks", flush=True)

    lock = threading.Lock()
    done = {"n": 0, "ok": 0, "moved": 0, "behaviour": 0}
    started = time.time()
    with open(a.out, "w") as fh, \
            concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        for row in pool.map(lambda it: evaluate_one(it, specs), items):
            if row is None:
                continue
            with lock:
                fh.write(json.dumps(row) + "\n")
                done["n"] += 1
                done["ok"] += row["ok"]
                if abs(row["score"] - row["stored_score"]) > 1e-9:
                    done["moved"] += 1
                if any(k.startswith("b_") for k in row["cases"]):
                    done["behaviour"] += 1
                if done["n"] % 500 == 0:
                    rate = done["n"] / max(time.time() - started, 1e-9)
                    left = (len(items) - done["n"]) / max(rate, 1e-9) / 60
                    print(f"  {done['n']}/{len(items)}  {rate:.1f}/s  "
                          f"~{left:.0f} min left", flush=True)

    print(f"\nwritten: {done['n']}  evaluated ok: {done['ok']}  "
          f"with behaviour: {done['behaviour']}")
    print(f"score moved vs the stored value: {done['moved']} "
          f"({done['moved'] / max(done['n'], 1):.1%})")
    if done["moved"]:
        print("  a moved score means an evaluator changed in a way that rescores old "
              "programs; expected here for matmul (finiteness) and for any task whose "
              "`full` path was touched.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
