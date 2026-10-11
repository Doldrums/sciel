"""Ignition test: does an inherited repertoire make the next generation improve faster on a held-out task?

Generation 0 runs on the source config. Generation 1 runs on the target config with paired seeds,
once inheriting the generation-0 repertoire and once without it.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]


def run(config: dict, outputs: Path, run_id: str, seed: int) -> dict:
    path = outputs / f"{run_id}.yaml"
    path.write_text(yaml.safe_dump(config))
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    with open(outputs / f"{run_id}.log", "w") as log:
        subprocess.run([sys.executable, "-m", "sciel.cli.main", "run", str(path), "--seed", str(seed),
                        "--outputs", str(outputs), "--run-id", run_id], env=env, stdout=log,
                       stderr=subprocess.STDOUT, check=True)
    return curve(outputs / run_id / "sciel_events.jsonl")


def load(path: str) -> dict:
    config = yaml.safe_load(Path(path).read_text())
    if "extends" in config:
        config["extends"] = str((Path(path).parent / config["extends"]).resolve())
    return config


def curve(journal: Path) -> dict:
    best, points = None, []
    for line in journal.read_text().splitlines():
        event = json.loads(line)
        if event["event"] == "iteration" and event.get("pop_best") is not None:
            if best is None or event["pop_best"] > best:
                best = event["pop_best"]
                points.append((event["B_evals"], best))
    return {"final": best, "improvements": points}


def evals_to(points: list, threshold: float) -> int | None:
    return next((b for b, score in points if score >= threshold), None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="generation-0 config")
    parser.add_argument("target", help="held-out generation-1 config")
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    outputs = Path(args.outputs).resolve()
    outputs.mkdir(parents=True, exist_ok=True)
    source, target = load(args.source), load(args.target)
    run(source, outputs, "gen0", 0)
    repertoire = str(outputs / "gen0" / "repertoire.json")
    jobs = [(arm, seed) for seed in range(1, args.seeds + 1) for arm in ("inherit", "naive")]
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda job: run({**target, **({"inherit": repertoire} if job[0] == "inherit" else {})},
                                                outputs, f"gen1_{job[0]}_{job[1]}", job[1]), jobs))
    by_arm = {arm: {seed: r for (a, seed), r in zip(jobs, results) if a == arm} for arm in ("inherit", "naive")}
    threshold = statistics.median(r["final"] for r in results)
    summary = {"repertoire": repertoire, "threshold": threshold, "arms": {}}
    for arm, runs in by_arm.items():
        summary["arms"][arm] = {"final": [runs[s]["final"] for s in sorted(runs)],
                                "evals_to_threshold": [evals_to(runs[s]["improvements"], threshold) for s in sorted(runs)]}
    summary["paired_final_difference"] = [by_arm["inherit"][s]["final"] - by_arm["naive"][s]["final"]
                                          for s in sorted(by_arm["inherit"])]
    (outputs / "ignition.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
