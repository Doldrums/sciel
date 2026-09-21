"""Compare an explicit available base revision against HEAD.

Both revisions must exist locally. --dry-run validates their isolated imports
without starting searches or making LLM calls.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sciel.config import load_config, validate
from sciel.engine.factory import load_env, resolve_tasks_root
from record_ratio import ratio_for
import yaml


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def arm_env(checkout: Path) -> dict:
    # The shared interpreter's editable .pth points at the original checkout.
    # Put this arm's src first, including for evaluator subprocesses it launches.
    return {**os.environ, "PYTHONPATH": str(checkout / "src")}


def source_sha(checkout: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted((checkout / "src").rglob("*.py")):
        digest.update(str(path.relative_to(checkout)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


_PREFLIGHT = '''
import json, pathlib, sys
import sciel
from sciel.config import load_config, validate
import sciel.engine.loop as loop
from sciel.engine.llm.mock import MockLLM
from sciel.eval.registry import load_task
checkout, config, tasks_root, out = map(pathlib.Path, sys.argv[1:])
module = pathlib.Path(sciel.__file__).resolve()
if not module.is_relative_to((checkout / "src").resolve()):
    raise RuntimeError(f"wrong engine imported: {module}")
cfg = validate(load_config(str(config)))
cfg["wandb"] = {"enabled": False}
task = load_task(str(tasks_root / cfg["task"]))
engine = loop.Engine(cfg, task, str(out), MockLLM(), python=sys.executable)
try:
    print(json.dumps({"module": str(module), "task_hash": task.task_hash,
                      "engine_contract": getattr(loop, "ENGINE_CONTRACT", "legacy_unversioned"),
                      "descriptor": getattr(engine, "descriptor_info", {})}))
finally:
    engine.log.close()
'''


def preflight(checkout: Path, config: Path, tasks: Path, output: Path) -> dict:
    proc = subprocess.run(
        [sys.executable, "-c", _PREFLIGHT, str(checkout), str(config), str(tasks), str(output)],
        cwd=checkout, env=arm_env(checkout), text=True, capture_output=True,
    )
    if proc.returncode:
        raise RuntimeError(f"preflight failed for {checkout}:\n{proc.stderr}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def run_arm(arm: str, seed: int, checkout: Path, config: Path, tasks: Path, out: Path) -> dict:
    run_id = f"{arm}_s{seed}"
    command = [sys.executable, "-m", "sciel.cli.main", "run", str(config),
               "--tasks-root", str(tasks), "--seed", str(seed),
               "--outputs", str(out), "--run-id", run_id]
    proc = subprocess.run(command, cwd=checkout, env=arm_env(checkout),
                          text=True, capture_output=True)
    (out / f"{run_id}.stdout.log").write_text(proc.stdout)
    (out / f"{run_id}.stderr.log").write_text(proc.stderr)
    if proc.returncode:
        raise RuntimeError(f"{run_id} exited {proc.returncode}; see {out / (run_id + '.stderr.log')}")
    try:
        summary = json.loads(proc.stdout.strip().splitlines()[-1])
    except (IndexError, ValueError) as exc:
        raise RuntimeError(f"{run_id} returned no JSON summary") from exc
    metric = ratio_for(str(out / run_id / "sciel_events.jsonl"))
    if metric is not None and metric["n"] != summary.get("B_evals"):
        raise RuntimeError(f"{run_id}: journal and summary disagree on B_evals")
    if metric is None and summary.get("B_evals", 0) >= 20:
        raise RuntimeError(f"{run_id}: no complete child-event metric")
    return {"arm": arm, "seed": seed, **summary,
            "records": metric["records"] if metric else None,
            "expected": metric["expected"] if metric else None,
            "ratio": metric["ratio"] if metric else None}


def paired_summary(rows: list[dict], seeds: list[int], budget: int) -> dict:
    """Compare only seeds with two complete, measurable runs at the same budget."""
    indexed = {(r["arm"], r["seed"]): r for r in rows}
    included, excluded = [], []
    for seed in seeds:
        reasons = {}
        for arm in ("base", "head"):
            row = indexed.get((arm, seed))
            if row is None:
                reasons[arm] = "missing run"
            elif row.get("B_evals") != budget:
                reasons[arm] = f"B_evals={row.get('B_evals')}, expected {budget} ({row.get('reason')})"
            elif row.get("expected") is None or row.get("records") is None:
                reasons[arm] = "no complete child-event metric (at least 20 evaluations required)"
        if reasons:
            excluded.append({"seed": seed, "reasons": reasons})
        else:
            included.append(seed)
    summary = {"included_seeds": included, "excluded_pairs": excluded, "arms": {}}
    for arm in ("base", "head"):
        valid = [indexed[arm, seed] for seed in included]
        expected = sum(r["expected"] for r in valid)
        records = sum(r["records"] for r in valid)
        summary["arms"][arm] = {"runs": len(valid), "records": records,
                                "expected": expected, "ratio": records / expected if expected else None}
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--outputs", type=Path, default=ROOT / "outputs" / "ab_revision",
                    help="parent directory; each comparison gets a fresh subdirectory")
    ap.add_argument("--budget", type=int)
    ap.add_argument("base_ref", help="explicit local Git revision for the base arm; candidate arm is HEAD")
    ap.add_argument("config", nargs="?", default="configs/heilbronn_default.yaml")
    ap.add_argument("seeds", nargs="*", type=int)
    a = ap.parse_args(argv)
    seeds = a.seeds or list(range(42, 50))
    if len(set(seeds)) != len(seeds):
        ap.error("seeds must be unique")
    config = Path(a.config)
    config = (config if config.is_absolute() else ROOT / config).resolve()
    # Materialize user settings, not HEAD defaults: each engine supplies its own
    # defaults. Resolve the objective ONCE and supply that same task to both arms.
    raw_cfg = load_config(str(config))
    if a.budget is not None:
        if a.budget < 1:
            ap.error("budget must be positive")
        raw_cfg["budget"] = a.budget
    cfg = validate(raw_cfg)
    tasks = Path(resolve_tasks_root(cfg, str(ROOT / "tasks"))).resolve()
    raw_cfg = {**raw_cfg, "sciel_tasks_repo": None, "sciel_tasks_revision": None}
    base_sha = git("rev-parse", "--verify", f"{a.base_ref}^{{commit}}")
    head_sha = git("rev-parse", "HEAD")
    a.outputs.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=f"{base_sha[:7]}_{head_sha[:7]}_", dir=a.outputs)).resolve()
    config_file = out / "config.yaml"
    config_file.write_text(yaml.safe_dump(raw_cfg, sort_keys=True))
    # Match the application's dotenv handling without executing or printing secrets.
    load_env()
    load_env(str(ROOT / ".env"))
    with tempfile.TemporaryDirectory(prefix="sciel_ab_") as temp:
        base = Path(temp) / "base"
        subprocess.run(["git", "worktree", "add", "--quiet", "--detach", str(base), base_sha],
                       cwd=ROOT, check=True)
        try:
            if not (base / "models").exists() and (ROOT / "models").exists():
                (base / "models").symlink_to(ROOT / "models", target_is_directory=True)
            arms = {"base": base, "head": ROOT}
            manifest = {"base_revision": base_sha, "head_revision": head_sha,
                        "task_source": str(tasks), "requested_task_source": {
                            "repo": cfg.get("sciel_tasks_repo"), "revision": cfg.get("sciel_tasks_revision")},
                        "seeds": seeds, "dry_run": a.dry_run, "arms": {}}
            for name, checkout in arms.items():
                info = preflight(checkout, config_file, tasks, Path(temp) / f"preflight_{name}")
                manifest["arms"][name] = {**info, "source_sha256": source_sha(checkout)}
            if manifest["arms"]["base"]["task_hash"] != manifest["arms"]["head"]["task_hash"]:
                raise RuntimeError("arms disagree on task identity; no runs started")
            (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"base: {base_sha}; head: {head_sha}; output: {out}", flush=True)
            if a.dry_run:
                print("preflight passed for both isolated imports; no searches started")
                return 0
            if not (cfg.get("llm") or {}).get("mock"):
                key = (cfg.get("llm") or {}).get("api_key_env", "OPENROUTER_API_KEY")
                if not os.environ.get(key):
                    raise RuntimeError(f"missing {key}; no searches started")
            rows = []
            with ThreadPoolExecutor(max_workers=2) as pool:
                for seed in seeds:
                    pending = [pool.submit(run_arm, name, seed, checkout, config_file, tasks, out)
                               for name, checkout in arms.items()]
                    failures = []
                    for future in pending:
                        try:
                            rows.append(future.result())
                        except Exception as exc:
                            failures.append(str(exc))
                    (out / "results.json").write_text(json.dumps(rows, indent=2) + "\n")
                    if failures:
                        raise RuntimeError("\n".join(failures))
                    print(f"seed {seed} done", flush=True)
            fields = ("arm", "seed", "pop_best", "records", "expected", "ratio", "B_evals", "cost_total", "reason")
            (out / "results.tsv").write_text("\t".join(fields) + "\n" + "".join(
                "\t".join(str(r.get(k)) for k in fields) + "\n" for r in rows))
            comparison = paired_summary(rows, seeds, cfg["budget"])
            (out / "comparison.json").write_text(json.dumps(comparison, indent=2) + "\n")
            for arm, stats in comparison["arms"].items():
                print(f"{arm}: {stats['runs']} paired measured runs, pooled harmonic record ratio {stats['ratio']}")
            for excluded in comparison["excluded_pairs"]:
                print(f"excluded seed {excluded['seed']}: {json.dumps(excluded['reasons'])}")
            if comparison["excluded_pairs"]:
                print("AB_INCOMPLETE: see comparison.json; raw results retained")
                return 2
            print("AB_DONE")
        finally:
            subprocess.run(["git", "worktree", "remove", "--force", str(base)],
                           cwd=ROOT, check=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (RuntimeError, OSError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)
