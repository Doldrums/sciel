"""Freeze a bounded, disjoint corpus, then resume local sandbox evaluations.

Historical scores are used ONLY for deterministic sampling. Every selected source
is re-evaluated against exported, versioned local task files. Legacy backfills are
deliberately not accepted as caches because they lack evaluator provenance.

    python scripts/prepare_review_corpus.py prepare --revision HEAD --out /tmp/sciel-corpus /path/to/runs
    python scripts/prepare_review_corpus.py evaluate --out /tmp/sciel-corpus --pilot 12
    python scripts/prepare_review_corpus.py evaluate --out /tmp/sciel-corpus --workers 6
"""

from __future__ import annotations

import argparse
import collections
import concurrent.futures
import fcntl
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from journal_data import DirtyRun, completed_events, verify_checkpoint_bytes  # noqa: E402
from sciel.engine.db import normalized_hash  # noqa: E402
from sciel.eval.registry import load_task  # noqa: E402
from sciel.eval.sandbox import run_stage  # noqa: E402

VERSION = "review-corpus-v1"
TASKS = ("oe/heilbronn_triangle", "oe/circle_packing", "oe/function_minimization")
SPLITS = ("train", "validation", "test")
THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
              "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")
RUNTIME_FILES = ("src/sciel/eval/_runner.py", "src/sciel/eval/sandbox.py",
                 "src/sciel/eval/facade.py", "src/sciel/eval/registry.py")


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, separators=(",", ":"),
                           allow_nan=False).encode()
    return hashlib.sha256(value).hexdigest()


def file_hash(path):
    return digest(Path(path).read_bytes())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def write_jsonl(path, rows):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def finite(value):
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value))


def runtime_identity():
    dependencies = {}
    for name in ("numpy", "scipy", "PyYAML", "psutil"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    return {"files": {p: file_hash(ROOT / p) for p in RUNTIME_FILES},
            "python": sys.version, "python_executable": sys.executable,
            "platform": platform.platform(), "dependencies": dependencies,
            "thread_environment": {name: "1" for name in THREAD_ENV},
            "protocol": "smoke_then_full; no confirmation; failures retained",
            "randomness": "task-native stochastic sampling; labels are measurements, not records"}


def collect(roots, tasks=TASKS, exclude_root=None):
    """Retain exact evaluated sources, and membership in EVERY clean source run."""
    paths = sorted({p.resolve() for root in roots for p in Path(root).rglob("checkpoint.json")
                    if exclude_root is None or not p.resolve().is_relative_to(Path(exclude_root).resolve())})
    candidates, runs, excluded, trajectories = {}, [], [], []
    counters = collections.Counter()
    for cp in paths:
        run_path, journal = cp.parent, cp.parent / "sciel_events.jsonl"
        try:
            checkpoint_bytes = cp.read_bytes()
            state = json.loads(checkpoint_bytes)
            task = (state.get("config") or {}).get("task")
            if task not in tasks:
                counters["other_task_checkpoints"] += 1
                continue
            journal_hash = file_hash(journal)
            events, end = completed_events(journal)
            verify_checkpoint_bytes(checkpoint_bytes, end, cp)
            if file_hash(journal) != journal_hash or cp.read_bytes() != checkpoint_bytes:
                raise DirtyRun("source changed while taking the inventory snapshot")
            starts = [event for event in events if event.get("event") == "run_start"]
            if not starts or any(event.get("task") != task for event in starts):
                raise DirtyRun("missing or inconsistent run_start task")
            source_hashes = {event.get("task_hash") for event in starts}
            if len(source_hashes) != 1:
                raise DirtyRun("task hash changed within trajectory")
            programs = (state.get("db") or {}).get("programs") or []
            if not isinstance(programs, list):
                raise DirtyRun("checkpoint programs must be an array")
        except (OSError, ValueError, TypeError, AttributeError, DirtyRun) as exc:
            excluded.append({"run_path": str(run_path), "reason": str(exc)})
            continue
        checkpoint_hash = digest(checkpoint_bytes)
        # Content identity survives relocating the source directory. The path is
        # still preserved for inspection, but cannot alter an already frozen split.
        run_id = digest({"journal": journal_hash, "checkpoint": checkpoint_hash})
        if any(run["run_id"] == run_id for run in runs):
            excluded.append({"run_path": str(run_path), "reason": "duplicate run content"})
            continue
        source_task_hash = starts[-1].get("task_hash")
        run = {"run_id": run_id, "run_path": str(run_path), "task": task,
               "journal_sha256": journal_hash, "checkpoint_sha256": checkpoint_hash,
               "source_task_hash": source_task_hash,
               "budget": starts[-1].get("budget"), "B_evals": end.get("B_evals"),
               "end_reason": end.get("reason"), "program_count": 0}
        runs.append(run)
        iterations = [e for e in events if e.get("event") == "iteration"]
        by_child = {e["child_id"]: e for e in iterations if e.get("child_id")}
        mapped = {}
        for program in programs:
            if not isinstance(program, dict) or not isinstance(program.get("code"), str) \
                    or not program["code"].strip():
                counters["missing_source"] += 1
                continue
            event = by_child.get(program.get("id"), {})
            explicit_code = event.get("child_code")
            # A conflicting checkpoint is not a fallback for an explicit source.
            code = explicit_code if isinstance(explicit_code, str) and explicit_code.strip() else program["code"]
            source_kind = "journal.child_code" if code == explicit_code else "checkpoint.db.programs.code"
            key = (task, normalized_hash(code))
            score = event.get("child_score", program.get("score"))
            membership = {**run, "program_id": program.get("id"),
                          "child_id": event.get("child_id"), "parent_id": program.get("parent_id"),
                          "iteration_no": event.get("iteration_no"), "B_evals": event.get("B_evals"),
                          "source_kind": source_kind, "source_sha256": digest(code.encode()),
                          "historical_score": score if finite(score) else None,
                          "source_task_hash": source_task_hash}
            membership.pop("program_count", None)
            item = candidates.setdefault(key, {"task": task, "code_hash": key[1], "memberships": [],
                                                "sources": {}})
            item["memberships"].append(membership)
            item["sources"][membership["source_sha256"]] = code
            mapped[program.get("id")] = key[1]
            run["program_count"] += 1
        for event in iterations:
            code_hash = mapped.get(event.get("child_id"))
            trajectories.append({"run_id": run_id, "task": task, "iteration_no": event.get("iteration_no"),
                                 "B_evals": event.get("B_evals"), "child_id": event.get("child_id"),
                                 "parent_id": event.get("parent_id"), "code_hash": code_hash,
                                 "source_mapping": "exact_child_id" if code_hash else "unavailable",
                                 "historical_score": event.get("child_score") if finite(event.get("child_score")) else None,
                                 "historical_only": True, "reason": event.get("reason")})
        successful = [e for e in iterations if e.get("child_score") is not None]
        run["trajectory_source_complete"] = bool(successful) and all(mapped.get(e.get("child_id")) for e in successful)
    return list(candidates.values()), runs, excluded, trajectories, dict(counters)


def assign_splits(candidates, runs, seed=0):
    """Partition run groups before sampling; global source duplicates prefer test."""
    groups = collections.defaultdict(list)
    for run in runs:
        groups[run["task"]].append(run)
    for task, group in groups.items():
        # A random two-run holdout can contain only historical smoke runs. Balance
        # source counts without inspecting scores, while preserving run quotas.
        group.sort(key=lambda r: (-r["program_count"], digest([seed, task, r["run_id"]])))
        if len(group) < 3:
            raise ValueError(f"{task}: at least three clean runs required")
        n_test, n_validation = max(1, round(len(group) * .2)), max(1, round(len(group) * .2))
        capacities = {"test": n_test, "validation": n_validation,
                      "train": len(group) - n_test - n_validation}
        weights = {"train": .6, "validation": .2, "test": .2}
        loads = dict.fromkeys(SPLITS, 0)
        for run in group:
            split = min((s for s in SPLITS if capacities[s]),
                        key=lambda s: (loads[s] / weights[s], SPLITS.index(s)))
            run["split"] = split
            capacities[split] -= 1
            loads[split] += run["program_count"]
    run_splits = {run["run_id"]: run["split"] for run in runs}
    priorities = {split: i for i, split in enumerate(SPLITS)}
    owners = {}
    for item in candidates:
        for membership in item["memberships"]:
            membership["split"] = run_splits[membership["run_id"]]
            owners[item["code_hash"]] = max(owners.get(item["code_hash"], 0), priorities[membership["split"]])
    for item in candidates:
        item["split"] = SPLITS[owners[item["code_hash"]]]
        eligible = [m for m in item["memberships"] if m["split"] == item["split"]]
        # Cross-task duplicate may be held out due to membership in another task.
        eligible = eligible or item["memberships"]
        representative = min(eligible, key=lambda m: (m["run_id"], str(m["program_id"]), m["source_sha256"]))
        item["representative_run_id"] = representative["run_id"]
        item["source_sha256"] = representative["source_sha256"]
        item["code"] = item.pop("sources")[representative["source_sha256"]]
        scores = sorted(m["historical_score"] for m in item["memberships"] if finite(m["historical_score"]))
        item["historical_score"] = scores[len(scores) // 2] if scores else None
        item["row_id"] = digest([item["task"], item["code_hash"]])


def stratify(items, count):
    """Evenly spaced rank strata; all choices precede evaluation of new labels."""
    ordered = sorted(items, key=lambda r: (r["historical_score"] is None,
                                           r["historical_score"] or 0, r["code_hash"]))
    count = min(count, len(ordered))
    if not count:
        return []
    return [ordered[min(len(ordered) - 1, int((i + .5) * len(ordered) / count))] for i in range(count)]


def sample(candidates, runs, per_task=400, seed=0):
    quotas = {"train": round(per_task * .6), "validation": round(per_task * .2)}
    quotas["test"] = per_task - sum(quotas.values())
    selected, anchors, available = [], [], {}
    for task in sorted({item["task"] for item in candidates}):
        for split in SPLITS:
            pool = [item for item in candidates if item["task"] == task and item["split"] == split]
            available[f"{task}:{split}"] = len(pool)
            chosen = []
            if split == "test" and task != "oe/function_minimization":
                eligible_runs = sorted([r for r in runs if r["task"] == task and r["split"] == split
                                        and r["trajectory_source_complete"]],
                                       key=lambda r: digest([seed, "anchor", r["run_id"]]))
                for run in eligible_runs:
                    owned = [item for item in pool if any(m["run_id"] == run["run_id"] for m in item["memberships"])]
                    if 20 <= len(owned) <= quotas[split]:
                        chosen = owned
                        anchors.append({"task": task, "run_id": run["run_id"], "split": split,
                                        "run_path": run["run_path"], "program_count": len(owned)})
                        break
            used = {r["row_id"] for r in chosen}
            chosen.extend(stratify([item for item in pool if item["row_id"] not in used], quotas[split] - len(chosen)))
            selected.extend(chosen)
    return sorted(selected, key=lambda r: (r["task"], r["split"], r["code_hash"])), anchors, available, quotas


def export_tasks(out, revision, tasks):
    task_identities = {}
    commit = subprocess.check_output(["git", "rev-parse", f"{revision}^{{commit}}"], cwd=ROOT, text=True).strip()
    for task in tasks:
        prefix = f"tasks/{task}/"
        names = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", commit, "--", prefix],
                                        cwd=ROOT, text=True).splitlines()
        if not names:
            raise ValueError(f"task {task} absent from {commit}")
        hashes = {}
        for name in names:
            relative = name[len(prefix):]
            path = out / "task_sources" / task / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            data = subprocess.check_output(["git", "show", f"{commit}:{name}"], cwd=ROOT)
            path.write_bytes(data)
            hashes[relative] = digest(data)
        spec = load_task(str(out / "task_sources" / task))
        task_identities[task] = {"task_hash": spec.task_hash, "file_sha256": hashes,
                                 "source_commit": commit, "source_kind": "versioned_local",
                                 "stochastic": spec.stochastic, "full_timeout": spec.full_timeout,
                                 "smoke_timeout": spec.smoke_timeout}
    return commit, task_identities


def prepare(out, roots, revision, per_task=400, seed=0, tasks=TASKS):
    out = Path(out).resolve()
    if (out / "manifest.json").exists():
        raise ValueError("frozen manifest exists; evaluate to resume, or choose a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    items, runs, exclusions, trajectories, counters = collect(roots, tasks, out)
    assign_splits(items, runs, seed)
    selected, anchors, available, quotas = sample(items, runs, per_task, seed)
    commit, task_identities = export_tasks(out, revision, tasks)
    identity = runtime_identity()
    evaluation_id = digest({"tasks": task_identities, "runtime": identity, "version": VERSION})
    for item in selected:
        item.update({"task_hash": task_identities[item["task"]]["task_hash"],
                     "evaluation_id": evaluation_id, "schema_version": VERSION})
    split_by_run = {r["run_id"]: r["split"] for r in runs}
    for event in trajectories:
        event["split"] = split_by_run[event["run_id"]]
    index = [{k: v for k, v in item.items() if k != "code"} for item in items]
    write_jsonl(out / "candidates.jsonl", index)
    write_jsonl(out / "selected.jsonl", selected)
    write_jsonl(out / "trajectories.jsonl", trajectories)
    write_json(out / "excluded_runs.json", exclusions)
    manifest = {"schema_version": VERSION, "baseline_commit": commit, "seed": seed,
                "per_task_requested": per_task, "split_quotas": quotas,
                "source_roots": [str(Path(p).resolve()) for p in roots], "runs": runs,
                "tasks": task_identities, "runtime": identity, "evaluation_id": evaluation_id,
                "available": available, "inventory_counters": counters,
                "source_count": len(items), "selected_count": len(selected),
                "selected_counts": dict(collections.Counter(f'{r["task"]}:{r["split"]}' for r in selected)),
                "excluded_run_count": len(exclusions), "trajectory_anchors": anchors,
                "selection": "60/20/20 run groups balanced by source counts (no score labels); global normalized hash test>validation>train; historical-score rank strata; fixed full-trajectory test anchors",
                "normalization": "sciel.engine.db.normalized_hash: remove blank lines, strip trailing whitespace",
                "label_policy": "all scores and cases recomputed; historical/backfilled labels never accepted; failures retained without replacement",
                "stochastic_limit": "one smoke + one full measurement; no record certification or deterministic-score claim",
                "files": {name: file_hash(out / name) for name in
                          ("candidates.jsonl", "selected.jsonl", "trajectories.jsonl", "excluded_runs.json")}}
    write_json(out / "manifest.json", manifest)
    print(json.dumps({k: manifest[k] for k in ("selected_count", "selected_counts", "available", "excluded_run_count", "trajectory_anchors")}, indent=2))
    return manifest


def verify_manifest(out):
    manifest = json.loads((out / "manifest.json").read_text())
    if manifest["schema_version"] != VERSION or manifest["runtime"] != runtime_identity():
        raise ValueError("evaluation runtime changed; create a new corpus manifest")
    for name, expected in manifest["files"].items():
        if file_hash(out / name) != expected:
            raise ValueError(f"frozen source manifest changed: {name}")
    for task, identity in manifest["tasks"].items():
        root = out / "task_sources" / task
        if load_task(str(root)).task_hash != identity["task_hash"]:
            raise ValueError(f"task contract changed: {task}")
        for name, expected in identity["file_sha256"].items():
            if file_hash(root / name) != expected:
                raise ValueError(f"task file changed: {task}/{name}")
    return manifest


def evaluate_one(item, out):
    spec = load_task(str(out / "task_sources" / item["task"]))
    if spec.task_hash != item["task_hash"]:
        raise ValueError("task hash changed during evaluation")
    started, stages = time.monotonic(), []
    with tempfile.TemporaryDirectory(prefix="sciel_review_") as directory:
        source = Path(directory) / "program.py"
        source.write_text(item["code"])
        for stage in ("smoke", "full"):
            stage_started = time.monotonic()
            result = run_stage(spec.evaluator_path, stage, str(source), python=sys.executable,
                               timeout=getattr(spec, f"{stage}_timeout"),
                               cpu_seconds=spec.cpu_seconds, mem_bytes=spec.mem_bytes)
            stages.append({"stage": stage, "ok": result.ok, "score": result.score,
                           "wall_seconds": time.monotonic() - stage_started,
                           "artifacts": result.artifacts, "stderr": result.stderr})
            if not result.ok:
                break
    return {**item, "ok": bool(result.ok), "score": result.score, "cases": result.cases,
            "evaluation": {"recomputed": True, "evaluation_id": item["evaluation_id"],
                           "task_hash": spec.task_hash, "reached": stages[-1]["stage"],
                           "confirmed": False, "stages": stages,
                           "wall_seconds": time.monotonic() - started}}


def load_results(path, selected, evaluation_id):
    """Only own fully identified evaluations are reusable, never legacy backfills."""
    if not path.exists():
        return {}
    by_id = {r["row_id"]: r for r in selected}
    results, raw_lines = {}, path.read_bytes().splitlines(keepends=True)
    for i, line in enumerate(raw_lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if i == len(raw_lines) - 1 and not line.endswith(b"\n"):
                # An interrupted append can leave only its final record partial.
                path.with_suffix(".interrupted-tail").write_bytes(line)
                write_jsonl(path, results.values())
                break
            raise ValueError("corrupt evaluation cache")
        source = by_id.get(row.get("row_id"))
        if source is None or any(row.get(key) != value for key, value in source.items()) \
                or row.get("evaluation_id") != evaluation_id \
                or row.get("evaluation", {}).get("evaluation_id") != evaluation_id \
                or row.get("evaluation", {}).get("task_hash") != source["task_hash"] \
                or row.get("evaluation", {}).get("recomputed") is not True:
            raise ValueError("untrusted or incompatible evaluation cache row")
        if row["row_id"] in results:
            raise ValueError("duplicate evaluation cache row")
        if type(row.get("ok")) is not bool or not finite(row.get("score")) \
                or not isinstance(row.get("cases"), dict) or not all(finite(v) for v in row["cases"].values()):
            raise ValueError("invalid evaluation cache metrics")
        results[row["row_id"]] = row
    return results


def finalize(out, manifest, selected, results):
    rows = [results[item["row_id"]] for item in selected if item["row_id"] in results]
    for split in SPLITS:
        write_jsonl(out / f"{split}.jsonl", [row for row in rows if row["split"] == split])
    groups = {}
    for task in manifest["tasks"]:
        task_rows = [row for row in rows if row["task"] == task]
        groups[task] = {"evaluated": len(task_rows), "ok": sum(row["ok"] for row in task_rows),
                        "failed": sum(not row["ok"] for row in task_rows),
                        "wall_seconds_sum": sum(row["evaluation"]["wall_seconds"] for row in task_rows),
                        "valid_by_split": dict(collections.Counter(row["split"] for row in task_rows if row["ok"]))}
    status = {"complete": len(rows) == len(selected), "evaluated": len(rows),
              "pending": len(selected) - len(rows), "tasks": groups,
              "manifest_sha256": file_hash(out / "manifest.json"),
              "result_files": {name: file_hash(out / name) for name in
                               ("results.jsonl", "train.jsonl", "validation.jsonl", "test.jsonl") if (out / name).exists()}}
    write_json(out / "status.json", status)
    return status


def _evaluate_locked(out, workers=6, pilot=0):
    out = Path(out).resolve()
    if not 1 <= workers <= 8 or pilot < 0:
        raise ValueError("workers must be 1..8; pilot nonnegative")
    for name in THREAD_ENV:
        os.environ[name] = "1"
    manifest = verify_manifest(out)
    selected = [json.loads(line) for line in (out / "selected.jsonl").read_text().splitlines()]
    path = out / "results.jsonl"
    results = load_results(path, selected, manifest["evaluation_id"])
    desired = selected
    if pilot:
        desired = []
        for task in manifest["tasks"]:
            for index, split in enumerate(SPLITS):
                count = pilot // 3 + (index < pilot % 3)
                desired.extend(stratify([r for r in selected if r["task"] == task and r["split"] == split], count))
    pending = [row for row in desired if row["row_id"] not in results]
    started = time.monotonic()
    print(f"evaluating {len(pending)} pending rows, {workers} workers; {len(results)} cached", flush=True)
    with path.open("a") as fh, concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(evaluate_one, row, out): row for row in pending}
        for future in concurrent.futures.as_completed(futures):
            row = future.result()
            fh.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            results[row["row_id"]] = row
            print(f'{len(results)}/{len(selected)} {row["task"]} {row["split"]} ok={row["ok"]} '
                  f'{row["evaluation"]["wall_seconds"]:.1f}s elapsed={time.monotonic()-started:.1f}s', flush=True)
    # Refuse to publish labels if the runner changed during a long evaluation.
    verify_manifest(out)
    status = finalize(out, manifest, selected, results)
    print(json.dumps(status, indent=2), flush=True)
    return status


def evaluate(out, workers=6, pilot=0):
    out = Path(out).resolve()
    # Resuming a cache is safe only with one append owner. Crash/interrupt releases
    # the OS lock; its persistent empty file does not prevent a subsequent resume.
    with (out / ".evaluation.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("another evaluation owns this corpus") from exc
        return _evaluate_locked(out, workers, pilot)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("roots", nargs="+")
    prep.add_argument("--out", required=True)
    prep.add_argument("--revision", required=True,
        help="Git task-source revision, such as HEAD; its full commit is frozen in the manifest")
    prep.add_argument("--per-task", type=int, default=400)
    prep.add_argument("--seed", type=int, default=0)
    evaluation = sub.add_parser("evaluate")
    evaluation.add_argument("--out", required=True)
    evaluation.add_argument("--workers", type=int, default=6)
    evaluation.add_argument("--pilot", type=int, default=0, help="fixed samples per task across all three splits")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        if args.per_task < 3:
            parser.error("--per-task must be at least 3")
        prepare(args.out, args.roots, args.revision, args.per_task, args.seed)
    else:
        evaluate(args.out, args.workers, args.pilot)


if __name__ == "__main__":
    main()
