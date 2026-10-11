"""Run-level analysis under a pre-registered plan: best-so-far curves, evaluations to threshold (Kaplan-Meier),
log gap to the reference, the seeds x budget frontier, and in-run arm values by DR with Hadad's adaptive weights.

Arms are given as NAME=GLOB over run directories (each with sciel_events.jsonl), or NAME=FILE.json holding
{"run": [[evaluations, best], ...]} curves exported from other systems. The plan is configs/analysis/<task>.yaml
with keys threshold, reference, budgets, seeds_k.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json
import math
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from sciel.causal.ledger import Ledger
from sciel.theory.claims import DECISIONS


def curve(run: Path) -> list:
    best, points = None, []
    for line in (run / "sciel_events.jsonl").read_text().splitlines():
        event = json.loads(line)
        if event.get("event") == "iteration" and event.get("pop_best") is not None:
            if best is None or event["pop_best"] > best:
                best = event["pop_best"]
                points.append((event["B_evals"], best))
    return points


def load_arm(spec: str) -> dict:
    name, source = spec.split("=", 1)
    if source.endswith(".json"):
        return {name: {run: [tuple(p) for p in points] for run, points in json.loads(Path(source).read_text()).items()}}
    return {name: {Path(path).name: curve(Path(path)) for path in sorted(glob.glob(source))
                   if (Path(path) / "sciel_events.jsonl").exists()}}


def best_at(points: list, budget: int) -> float | None:
    values = [score for evaluations, score in points if evaluations <= budget]
    return values[-1] if values else None


def kaplan_meier(times: list, events: list) -> dict:
    """Median and restricted mean of evaluations-to-threshold with right censoring at the run's last evaluation."""
    order = sorted(zip(times, events))
    at_risk, survival, median, area, last = len(order), 1.0, None, 0.0, 0
    for time, event in order:
        area += survival * (time - last)
        last = time
        if event:
            survival *= 1 - 1 / at_risk
            if median is None and survival <= 0.5:
                median = time
        at_risk -= 1
    return {"median": median, "restricted_mean": area, "reached": sum(events), "runs": len(order)}


def expected_max(values: list, k: int) -> float | None:
    """E[max of k runs drawn without replacement] from the order statistics."""
    values = sorted(v for v in values if v is not None)
    n = len(values)
    if n < k:
        return None
    return sum(v * math.comb(i, k - 1) for i, v in enumerate(values)) / math.comb(n, k)


def arm_values(run: Path) -> dict:
    """Value of always taking each action of each decision: AIPW scores averaged with weights sqrt(propensity)."""
    saved = json.loads((run / "checkpoint.json").read_text())
    if "ledger" not in saved:
        return {}
    ledger = Ledger()
    ledger.load_state(saved["ledger"])
    rows = defaultdict(list)
    for row in ledger.rows:
        if row.outcome is not None and len(row.distribution) > 1 and row.decision in DECISIONS:
            rows[row.decision].append(row)
    out = {}
    for decision, local in rows.items():
        cells = defaultdict(list)
        for row in local:
            cells[row.choice].append(row.outcome)
        mean = sum(r.outcome for r in local) / len(local)
        q = {a: (sum(v) + 2 * mean) / (len(v) + 2) for a, v in cells.items()}
        for action in sorted({a for r in local for a in r.distribution}):
            scores, weights = [], []
            for row in local:
                p = row.distribution.get(action, 0.0)
                if p <= 0:
                    continue
                base = q.get(action, mean)
                scores.append(base + (row.choice == action) * (row.outcome - base) / p)
                weights.append(math.sqrt(p))
            if not scores:
                continue
            total = sum(weights)
            value = sum(w * s for w, s in zip(weights, scores)) / total
            se = math.sqrt(sum(w * w * (s - value) ** 2 for w, s in zip(weights, scores))) / total
            out[f"{decision}:{action}"] = {"value": value, "low": value - 1.96 * se, "high": value + 1.96 * se,
                                           "rows": len(scores)}
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", help="configs/analysis/<task>.yaml")
    parser.add_argument("arms", nargs="+", help="NAME=GLOB of run directories or NAME=FILE.json of curves")
    parser.add_argument("--values", action="store_true", help="also report in-run arm values from checkpoints")
    args = parser.parse_args()
    plan = yaml.safe_load(Path(args.plan).read_text())
    arms = {}
    for spec in args.arms:
        arms.update(load_arm(spec))
    report = {"plan": plan, "arms": {}}
    for name, runs in arms.items():
        finals = {run: points[-1][1] for run, points in runs.items() if points}
        reached = [next(((e, True) for e, s in points if s >= plan["threshold"]), (points[-1][0] if points else 0, False))
                   for points in runs.values()]
        arm = {"runs": len(runs), "final": finals,
               "to_threshold": kaplan_meier([t for t, _ in reached], [r for _, r in reached]),
               "at_budget": {}, "frontier": {}}
        for budget in plan["budgets"]:
            values = [best_at(points, budget) for points in runs.values()]
            gaps = [math.log10(max(plan["reference"] - v, 1e-9)) for v in values if v is not None]
            present = [v for v in values if v is not None]
            arm["at_budget"][budget] = {"mean_best": sum(present) / len(present) if present else None,
                                        "mean_log_gap": sum(gaps) / len(gaps) if gaps else None}
            arm["frontier"][budget] = {k: expected_max(values, k) for k in plan["seeds_k"]}
        spec = next(s for s in args.arms if s.startswith(name + "="))
        if args.values and not spec.endswith(".json"):
            arm["in_run_values"] = {Path(p).name: arm_values(Path(p)) for p in sorted(glob.glob(spec.split("=", 1)[1]))
                                    if (Path(p) / "checkpoint.json").exists()}
        report["arms"][name] = arm
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
