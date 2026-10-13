"""Ignition test: does an inherited repertoire make the next generation improve faster on a held-out task?

Generation 0 runs on the source config. Generation 1 runs on the target config with paired seeds: without
inheritance, and once per transmission rule. The rules are transformations of the same repertoire file:
gate (ours: every item is re-tested by the recipient's ledger), greedy (AIDE2-style: everything is accepted),
certify (REUSE-style: only items proven at the donor are accepted, no learned models) and replay
(Dream-RSI-style: only the learned models are passed on).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_runs import curve


def run(config: dict, outputs: Path, run_id: str, seed: int) -> dict:
    path = outputs / f"{run_id}.yaml"
    capacity = {"capacity_dir": str(outputs / "capacity" / run_id)}
    config = {**config, "parallel_search": {**config.get("parallel_search", {}), **capacity}}
    path.write_text(yaml.safe_dump(config))
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "OMP_NUM_THREADS": "1",
           "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    with open(outputs / f"{run_id}.log", "w") as log:
        code = subprocess.run([sys.executable, "-m", "sciel.cli.main", "run", str(path), "--seed", str(seed),
                               "--outputs", str(outputs), "--run-id", run_id], env=env, stdout=log,
                              stderr=subprocess.STDOUT).returncode
    journal = outputs / run_id
    result = curve(journal) if (journal / "sciel_events.jsonl").exists() else {"points": [], "end": 0, "complete": False}
    return {**result, "exit": code, "final": result["points"][-1][1] if result["points"] else None}


def _accept(items: list, keep) -> list:
    return [{**item, "accept": "culture"} for item in items if keep(item)]


RULES = {
    "gate": lambda r: r,
    "greedy": lambda r: {**r, "strategies": _accept(r["strategies"], lambda i: True),
                         "knowledge": _accept(r["knowledge"], lambda i: True),
                         "procedures": _accept(r["procedures"], lambda i: True)},
    "certify": lambda r: {**r, "strategies": _accept(r["strategies"], lambda i: i["donor_status"] == "culture"),
                          "knowledge": _accept(r["knowledge"], lambda i: i["donor_status"] == "culture"),
                          "procedures": _accept(r["procedures"], lambda i: True),
                          "swm": {**r["swm"], "weights": None}, "policy": {**r["policy"], "weights": None}},
    "replay": lambda r: {**r, "claims": [], "strategies": [], "knowledge": [], "procedures": [], "methods": [],
                         "evidence": {"claims": {}, "items": {}, "programmes": []}},
}


def load(path: str) -> dict:
    config = yaml.safe_load(Path(path).read_text())
    if "extends" in config:
        config["extends"] = str((Path(path).parent / config["extends"]).resolve())
    return config


def evals_to(result: dict, threshold: float) -> dict:
    hit = next((b for b, score in result["points"] if score >= threshold), None)
    return {"evaluations": hit if hit is not None else result["end"], "reached": hit is not None}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="generation-0 config")
    parser.add_argument("target", help="held-out generation-1 config")
    parser.add_argument("plan", help="pre-registered configs/analysis/<task>.yaml of the target task")
    parser.add_argument("--outputs", required=True)
    parser.add_argument("--seeds", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--rules", nargs="+", default=list(RULES), choices=list(RULES))
    args = parser.parse_args()
    outputs = Path(args.outputs).resolve()
    outputs.mkdir(parents=True, exist_ok=True)
    source, target = load(args.source), load(args.target)
    threshold = yaml.safe_load(Path(args.plan).read_text())["threshold"]
    donor_run = run(source, outputs, "gen0", 0)
    if donor_run["exit"] != 0:
        raise SystemExit(f"generation 0 failed with exit code {donor_run['exit']}")
    donor = json.loads((outputs / "gen0" / "repertoire.json").read_text())
    files = {}
    for rule in args.rules:
        files[rule] = outputs / f"repertoire_{rule}.json"
        files[rule].write_text(json.dumps(RULES[rule](donor)))
    arms = list(args.rules) + ["naive"]
    jobs = [(arm, seed) for seed in range(1, args.seeds + 1) for arm in arms]
    with ThreadPoolExecutor(args.workers) as pool:
        results = list(pool.map(lambda job: run({**target, **({"inherit": str(files[job[0]])} if job[0] in files else {})},
                                                outputs, f"gen1_{job[0]}_{job[1]}", job[1]), jobs))
    by_arm = {arm: {seed: r for (a, seed), r in zip(jobs, results) if a == arm} for arm in arms}
    summary = {"repertoires": {k: str(v) for k, v in files.items()}, "threshold": threshold, "arms": {}}
    for arm, runs in by_arm.items():
        summary["arms"][arm] = {"final": [runs[s]["final"] for s in sorted(runs)],
                                "exit": [runs[s]["exit"] for s in sorted(runs)],
                                "complete": [runs[s]["complete"] for s in sorted(runs)],
                                "to_threshold": [evals_to(runs[s], threshold) for s in sorted(runs)]}
    summary["paired_final_difference_vs_naive"] = {
        arm: [by_arm[arm][s]["final"] - by_arm["naive"][s]["final"]
              if by_arm[arm][s]["final"] is not None and by_arm["naive"][s]["final"] is not None else None
              for s in sorted(by_arm[arm])] for arm in files}
    (outputs / "ignition.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
