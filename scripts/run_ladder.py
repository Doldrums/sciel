"""Run-level randomized comparison of engine revisions or configurations.

Each arm is NAME=REVISION or NAME=REVISION:OVERLAY.yaml. Every revision runs from its own Git worktree, so local
uncommitted changes never enter a run. Seeds form blocks; the arm order inside each block is randomized from
--assignment-seed and recorded, together with the host load when each run starts and ends, because arms share the
LLM server. Every run gets a private host-capacity directory and single-threaded numerics.
Analyse the result with scripts/analyze_runs.py, one NAME=OUT/NAME/* argument per arm.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]


def git(*args) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def load(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    if "extends" in config:
        config["extends"] = str((path.parent / config["extends"]).resolve())
    return config


def merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for key, value in over.items():
        out[key] = merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def run(arm: dict, seed: int, out: Path, tasks: Path) -> dict:
    run_id = f"{arm['name']}_s{seed}"
    folder = out / arm["name"]
    folder.mkdir(parents=True, exist_ok=True)
    config = merge(arm["config"], {"parallel_search": {"capacity_dir": str(out / "capacity" / run_id)}})
    path = folder / f"{run_id}.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=True))
    env = {**os.environ, "PYTHONPATH": str(arm["checkout"] / "src"), "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    started, load_start = time.time(), os.getloadavg()[0]
    with open(folder / f"{run_id}.log", "w") as log:
        code = subprocess.run([sys.executable, "-m", "sciel.cli.main", "run", str(path), "--tasks-root", str(tasks),
                               "--seed", str(seed), "--outputs", str(folder), "--run-id", run_id],
                              cwd=arm["checkout"], env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    return {"arm": arm["name"], "seed": seed, "exit": code, "started": started, "seconds": time.time() - started,
            "load_start": load_start, "load_end": os.getloadavg()[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="base run config shared by every arm")
    parser.add_argument("arms", nargs="+", help="NAME=REVISION or NAME=REVISION:OVERLAY.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--outputs", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--assignment-seed", type=int, default=0)
    parser.add_argument("--tasks-root", type=Path, default=ROOT / "tasks")
    args = parser.parse_args()
    out = args.outputs.resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = load(args.config.resolve())
    with tempfile.TemporaryDirectory(prefix="sciel_ladder_") as temp:
        arms, trees = [], {}
        for spec in args.arms:
            name, _, target = spec.partition("=")
            revision, _, overlay = target.partition(":")
            sha = git("rev-parse", "--verify", f"{revision}^{{commit}}")
            if sha not in trees:
                trees[sha] = Path(temp) / sha[:12]
                subprocess.run(["git", "worktree", "add", "--quiet", "--detach", str(trees[sha]), sha],
                               cwd=ROOT, check=True)
            config = merge(base, load(Path(overlay).resolve())) if overlay else dict(base)
            arms.append({"name": name, "revision": sha, "overlay": overlay or None, "checkout": trees[sha],
                         "config": config})
        rng = random.Random(args.assignment_seed)
        schedule = []
        for seed in args.seeds:
            block = list(arms)
            rng.shuffle(block)
            schedule.extend((arm, seed) for arm in block)
        manifest = {"arms": [{k: str(v) if isinstance(v, Path) else v for k, v in a.items() if k != "config"}
                             for a in arms],
                    "seeds": args.seeds, "assignment_seed": args.assignment_seed,
                    "order": [[a["name"], s] for a, s in schedule]}
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
        try:
            with ThreadPoolExecutor(args.workers) as pool:
                results = list(pool.map(lambda job: run(job[0], job[1], out, args.tasks_root.resolve()), schedule))
        finally:
            for tree in trees.values():
                subprocess.run(["git", "worktree", "remove", "--force", str(tree)], cwd=ROOT, check=False)
    (out / "runs.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({"runs": len(results), "failed": [r for r in results if r["exit"]]}, indent=2))


if __name__ == "__main__":
    main()
