"""Offline record-statistics audit over the frozen corpus run inventory."""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shlex
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
from journal_data import DirtyRun, completed_events
from record_ratio import ratio_for
from sciel.eval.registry import load_task


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def summarize(rows):
    records = sum(r["records"] for r in rows)
    harmonic_sum = sum(r["expected"] for r in rows)
    return dict(runs=len(rows), generated_evaluations=sum(r["n"] for r in rows),
                records=records, harmonic_sum=harmonic_sum,
                pooled_ratio=records / harmonic_sum if harmonic_sum else None,
                median_ratio=statistics.median(r["ratio"] for r in rows) if rows else None,
                below_one=sum(r["ratio"] < 1 for r in rows))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path,
                        default=ROOT.parent / "sciel-data/review/corpus")
    parser.add_argument("--out", type=Path,
                        default=ROOT.parent / "sciel-data/review/record_metrics")
    args = parser.parse_args(argv)
    out, corpus = args.out.resolve(), args.corpus.resolve()
    manifest_path = corpus / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    for name, expected in manifest["files"].items():
        assert sha(corpus / name) == expected, f"corpus artifact changed: {name}"
    excluded = json.loads((corpus / "excluded_runs.json").read_text())
    assert len(excluded) == manifest["excluded_run_count"]
    excluded_paths = {r["run_path"] for r in excluded}
    current = {}
    for task, frozen in manifest["tasks"].items():
        task_root = ROOT / "tasks" / task
        for name, expected in frozen["file_sha256"].items():
            assert sha(task_root / name) == expected, f"current task changed: {task}/{name}"
        current[task] = load_task(str(task_root)).task_hash
        assert current[task] == frozen["task_hash"]

    rows, short = [], []
    task_groups = defaultdict(list)
    historical_groups = defaultdict(list)
    seen = set()
    for run in manifest["runs"]:
        run_dir = Path(run["run_path"])
        assert str(run_dir) not in seen and str(run_dir) not in excluded_paths
        seen.add(str(run_dir))
        journal = run_dir / "sciel_events.jsonl"
        assert sha(journal) == run["journal_sha256"], f"journal changed: {journal}"
        assert sha(run_dir / "checkpoint.json") == run["checkpoint_sha256"], str(run_dir)
        events, end = completed_events(journal)
        assert end["B_evals"] == run["B_evals"] and end.get("reason") == run["end_reason"]
        starts = [e for e in events if e.get("event") == "run_start"]
        assert starts and all(e.get("task_hash") == run["source_task_hash"] for e in starts)
        row = ratio_for(str(journal))
        if row is None:
            # All exclusions in this frozen inventory are the script's n>=20 rule.
            # A future failure due to absent/ambiguous events must stop the audit.
            assert end["B_evals"] < 20, f"unexpected missing metric: {run_dir}"
            short.append(dict(run_id=run["run_id"], run_path=str(run_dir),
                              task=run["task"], B_evals=end["B_evals"],
                              reason="fewer than 20 generated evaluations"))
            continue
        children = {}
        for event in events:
            if event.get("event") != "iteration" or event.get("child_score") is None:
                continue
            n = event["B_evals"]
            assert n not in children, f"ambiguous repeated generated evaluation: {run_dir}/{n}"
            record = event.get("is_pop_event")
            if type(record) is not bool:
                delta = float(event["d_pop"])
                assert math.isfinite(delta)
                record = delta > 0
            children[n] = int(record)
        assert set(children) == set(range(1, row["n"] + 1)), str(run_dir)
        assert sum(children.values()) == row["records"]
        row.update(run_id=run["run_id"], run_path=str(run_dir), split=run["split"],
                   journal_sha256=run["journal_sha256"],
                   source_task_hash=run["source_task_hash"], current_task_hash=current[run["task"]],
                   current_task_comparable=run["source_task_hash"] == current[run["task"]],
                   end_reason=run["end_reason"],
                   arm_source="logged config.arm or run_start.arm; no directory-name inference")
        rows.append(row)
        task_groups[row["task"]].append(row)
        historical_groups[(row["task"], row["source_task_hash"])].append(row)

    # Recheck that the frozen exclusions still fail trajectory hygiene. Their
    # journal bytes were not hashed by that historical exclusion manifest.
    for run in excluded:
        try:
            completed_events(Path(run["run_path"]) / "sciel_events.jsonl")
        except DirtyRun as exc:
            run["rechecked_reason"] = str(exc)
        else:
            raise AssertionError(f"previously excluded journal now passes: {run['run_path']}")

    audit = dict(schema="historical-record-audit/1", created_at=datetime.now(timezone.utc).isoformat(),
                 corpus_manifest=str(manifest_path), corpus_manifest_sha256=sha(manifest_path),
                 baseline_commit=manifest["baseline_commit"],
                 implementation={name:sha(ROOT / name) for name in
                     ("scripts/record_ratio.py", "scripts/journal_data.py")},
                 audit_script=str(Path(__file__).resolve()), audit_script_sha256=sha(__file__),
                 output_directory=str(out),
                 scope="descriptive historical generated-child records; no new evaluations or API calls",
                 metric="records / H_n, where H_n = sum(1/k, k=1..n)",
                 interpretation="H_n is a normalization, not an exchangeable null or significance test; ties, seeding, adaptation and mixed configurations preclude causal arm claims",
                 validation=dict(corpus_artifacts_verified=len(manifest["files"]),
                                 journals_verified=len(seen), checkpoints_verified=len(seen),
                                 completed_trajectories_verified=len(seen),
                                 current_tasks_match_frozen_manifest=True,
                                 exclusions_rechecked=len(excluded),
                                 excluded_journals_hash_frozen=False),
                 inventory_runs=len(seen), eligible_runs=len(rows), short_runs=len(short),
                 hygiene_excluded_runs=len(excluded),
                 current_task_comparable_runs=sum(r["current_task_comparable"] for r in rows),
                 summary=summarize(rows), by_task={k:summarize(v) for k,v in sorted(task_groups.items())},
                 by_task_identity=[dict(task=task, source_task_hash=digest,
                                       current_task_hash=current[task], **summarize(values))
                                   for (task,digest),values in sorted(historical_groups.items())],
                 inventory_task_hashes={task:dict(Counter(r["source_task_hash"] for r in manifest["runs"]
                                                      if r["task"] == task)) for task in current})
    out.mkdir(parents=True, exist_ok=True)
    write(out / "audit.json", audit)
    write(out / "excluded.json", {"journal_hygiene":excluded, "metric_minimum_length":short})
    (out / "rows.jsonl").write_text("".join(json.dumps(r, sort_keys=True, allow_nan=False) + "\n" for r in rows))
    table = ["| Task | Runs | Generated evals | Records | Sum H_n | Pooled ratio | Median ratio | Below 1 |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for task, values in audit["by_task"].items():
        table.append(f"| {task} | {values['runs']} | {values['generated_evaluations']} | {values['records']} | {values['harmonic_sum']:.6f} | {values['pooled_ratio']:.6f} | {values['median_ratio']:.6f} | {values['below_one']} |")
    exclusions = Counter(
        "ambiguous rollback/resume" if "rollback" in r["rechecked_reason"]
        else "missing run_end" if "no run_end" in r["rechecked_reason"]
        else "other trajectory hygiene failure" for r in excluded)
    exclusion_text = ", ".join(f"{count} {reason}" for reason, count in sorted(exclusions.items())) or "none"
    command = shlex.join(["venv/bin/python", "scripts/audit_review_record_metrics.py",
                          "--corpus", str(corpus), "--out", str(out)])
    (out / "report.md").write_text(
        "# Historical record-metrics audit\n\n"
        f"Corpus manifest SHA-256: `{audit['corpus_manifest_sha256']}`.\n\n"
        f"Audit script: `{Path(__file__).name}`; SHA-256: `{audit['audit_script_sha256']}`. "
        f"Artifacts written to `{out}`; other audit directories are unchanged.\n\n"
        f"Verified {len(seen)} journal hashes, {len(seen)} checkpoint hashes, "
        f"{len(manifest['files'])} corpus artifact hashes, "
        "and completed trajectory provenance with `completed_events`. Applied the existing "
        "`record_ratio.ratio_for` and independently checked unique generated-evaluation counters "
        f"and record totals. {len(rows)} runs qualify; {len(short)} have fewer than 20 generated evaluations.\n\n"
        f"The corpus's {len(excluded)} journal exclusions remain excluded: {exclusion_text}. "
        "Their current rejection reasons were "
        "rechecked; the exclusion inventory did not freeze their journal hashes.\n\n"
        + "\n".join(table) + "\n\n"
        f"{audit['current_task_comparable_runs']} of {len(rows)} eligible runs have the same "
        "task hash as the current task definitions. Hash equality alone does not establish "
        "equal runtime, engine or search settings. Current task "
        "files and hashes match the frozen corpus manifest. These numbers describe historical "
        "record events, not re-evaluated performance or certified records.\n\n"
        "H_n is only a normalization. Ties, prior seed bests, adaptive search, varying budgets "
        "and mixed configurations rule out treating it as an exact null expectation or a "
        "significance test. Task aggregates do not establish that one arm or embedder is better. "
        "No arm labels were guessed from directory names.\n\n"
        f"Reproduce with `{command}`. "
        "The audit performs no candidate evaluation and no API calls. Per-run results and all "
        "exclusions are retained in `rows.jsonl` and `excluded.json`.\n")
    print(json.dumps({"out":str(out), "eligible":len(rows), "short":len(short),
                      "hygiene_excluded":len(excluded), "current_comparable":audit["current_task_comparable_runs"],
                      "by_task":audit["by_task"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
