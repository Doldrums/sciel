"""Run-level analysis under a pre-registered plan: best-so-far curves, evaluations to threshold (Kaplan-Meier),
log gap to the reference, the seeds x budget frontier, and in-run arm values by DR with Hadad's adaptive weights.

The evaluation axis counts every evaluator call of a run: seeds, spawns, failed candidates, research candidates and
record confirmations, the same unit as a competitor's evaluated programs. Arms are given as NAME=GLOB over run
directories (each with sciel_events.jsonl), or NAME=FILE.json holding {"run": {"points": [[evaluations, score], ...],
"end": evaluations, "complete": bool}} from other systems, with raw per-evaluation scores. The plan is
configs/analysis/<task>.yaml with keys threshold, reference, budgets, seeds_k and optionally gap_floor.
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

GAP_FLOOR = 1e-4
FINISHED = ("budget", "saturated")


def events(run: Path):
    for line in (run / "sciel_events.jsonl").read_text().splitlines():
        yield json.loads(line)


def curve(run: Path) -> dict:
    """Population best after each change, on the axis of all evaluator calls, and whether the run finished."""
    evaluations, best, points, reason = 0, None, [], None
    for event in events(run):
        if event.get("event") == "decision" and (event.get("policy"), event.get("action")) == ("evaluation", "result"):
            evaluations += 1
        elif event.get("event") == "decision" and event.get("policy") == "confirmation":
            evaluations += 1
        elif event.get("event") == "run_end":
            reason = event.get("reason")
        if event.get("event") == "iteration" and event.get("pop_best") is not None:
            if best is None or event["pop_best"] > best:
                best = event["pop_best"]
                points.append((evaluations, best))
    return {"points": points, "end": evaluations, "complete": reason in FINISHED}


def external(value) -> dict:
    raw = value["points"] if isinstance(value, dict) else value
    best, points = None, []
    for evaluations, score in raw:
        if score is not None and (best is None or score > best):
            best = score
            points.append((int(evaluations), best))
    end = value.get("end", raw[-1][0] if raw else 0) if isinstance(value, dict) else (raw[-1][0] if raw else 0)
    return {"points": points, "end": int(end), "complete": bool(value.get("complete", True)) if isinstance(value, dict) else True}


def load_arm(spec: str) -> dict:
    name, source = spec.split("=", 1)
    if source.endswith(".json"):
        return {name: {run: external(value) for run, value in json.loads(Path(source).read_text()).items()}}
    return {name: {Path(path).name: curve(Path(path)) for path in sorted(glob.glob(source))
                   if (Path(path) / "sciel_events.jsonl").exists()}}


ARMS = {"finding": lambda c: c != "none", "strategy": lambda c: c != "none", "method": lambda c: c != "none",
        "research_open": lambda c: c == "open", "origin": lambda c: c == "fresh", "planner": lambda c: c != "base",
        "inspiration_source": lambda c: c == "archive", "idea_target": lambda c: c != "none",
        "cell_scope": lambda c: c == "task"}


def record_attribution(run: Path) -> dict:
    """Which arms were active in the generation steps whose child set a population record, against all such steps."""
    steps, records = 0, 0
    active, at_record = defaultdict(int), defaultdict(int)
    for event in events(run):
        if event.get("event") != "iteration" or event.get("reason") == "remeasurement" or not event.get("decision_context"):
            continue
        record = bool(event.get("is_pop_event"))
        arms = {d["decision"] for d in event["decision_context"].get("decisions", [])
                if d["decision"] in ARMS and ARMS[d["decision"]](d["choice"])}
        steps += 1
        records += record
        for arm in arms:
            active[arm] += 1
            at_record[arm] += record
    return {"steps": steps, "records": records,
            "arms": {arm: {"share_all": active[arm] / max(steps, 1), "share_records": at_record[arm] / max(records, 1),
                           "records": at_record[arm]} for arm in sorted(active)}}


def delivery(run: Path) -> dict:
    """Generation steps that showed a finding arm, and the last counters of the knowledge pool by kind."""
    steps, shown, counters = 0, 0, []
    for event in events(run):
        if event.get("event") == "iteration" and event.get("decision_context"):
            steps += 1
            shown += any(d["decision"] == "finding" and d["choice"] != "none"
                         for d in event["decision_context"].get("decisions", []))
        elif event.get("event") == "decision" and (event.get("policy"), event.get("action")) == ("finding", "counters"):
            counters = event["counters"]
    kinds = defaultdict(lambda: {"items": 0, "created": 0, "delivered": 0, "culture": 0, "retired": 0})
    for item in counters:
        kind = kinds[item["kind"]]
        kind["items"] += 1
        kind["created"] += item["created"]
        kind["delivered"] += item["delivered"]
        kind[item["status"]] = kind.get(item["status"], 0) + 1
    return {"steps": steps, "share_with_finding_arm": shown / max(steps, 1), "pool": dict(kinds)}


def families(run: Path) -> list:
    """Occupied cells of the latent archive over time: distinct approach families in the population."""
    return [(event["iteration_no"], len(event.get("elites") or [])) for event in events(run)
            if event.get("event") == "decision" and (event.get("policy"), event.get("action")) == ("memory", "rebuild")]


def llm_usage(run: Path, evaluations: int) -> dict:
    """Calls, prompt size, tokens and cost per role, and per evaluation, from the run's LLM call log."""
    path = run / "artifacts" / "llm_calls.jsonl"
    if not path.exists():
        return {}
    roles = defaultdict(lambda: {"calls": 0, "prompt_chars": [], "prompt_tokens": 0, "completion_tokens": 0,
                                 "cost": 0.0, "unpriced": 0})
    for line in path.read_text().splitlines():
        call = json.loads(line)
        role = roles[call.get("role", "unknown")]
        if call.get("event") == "request":
            role["calls"] += 1
            role["prompt_chars"].append(call.get("prompt_chars", 0))
        elif call.get("event") == "response":
            role["prompt_tokens"] += call.get("prompt_tokens") or 0
            role["completion_tokens"] += call.get("completion_tokens") or 0
            role["cost"] += call.get("cost") or 0.0
            role["unpriced"] += call.get("cost") is None
    out = {}
    for name, role in roles.items():
        sizes = sorted(role.pop("prompt_chars"))
        out[name] = {**role, "mean_prompt_chars": sum(sizes) / max(len(sizes), 1),
                     "median_prompt_chars": sizes[len(sizes) // 2] if sizes else 0, "max_prompt_chars": max(sizes, default=0)}
    total = {key: sum(r[key] for r in out.values()) for key in ("prompt_tokens", "completion_tokens", "cost")}
    out["per_evaluation"] = {key: value / max(evaluations, 1) for key, value in total.items()}
    return out


def best_at(run: dict, budget: int) -> float | None:
    """Best within the budget; None when the run stopped before the budget without finishing its own budget."""
    if run["end"] < budget and not run["complete"]:
        return None
    values = [score for evaluations, score in run["points"] if evaluations <= budget]
    return values[-1] if values else None


def kaplan_meier(times: list, events: list, horizon: int) -> dict:
    """Median and mean restricted to a fixed horizon of evaluations-to-threshold; censored runs leave the risk set
    after events at the same time."""
    order = sorted(zip(times, events), key=lambda x: (x[0], not x[1]))
    at_risk, survival, median, area, last = len(order), 1.0, None, 0.0, 0
    for time, event in order:
        area += survival * (min(time, horizon) - min(last, horizon))
        last = time
        if event:
            survival *= 1 - 1 / at_risk
            if median is None and survival <= 0.5:
                median = time
        at_risk -= 1
    area += survival * max(horizon - min(last, horizon), 0)
    return {"median": median, "restricted_mean": area, "horizon": horizon, "reached": sum(events), "runs": len(order)}


def expected_max(values: list, k: int) -> float | None:
    """E[max of k runs drawn without replacement] from the order statistics."""
    values = sorted(v for v in values if v is not None)
    n = len(values)
    if n < k:
        return None
    return sum(v * math.comb(i, k - 1) for i, v in enumerate(values)) / math.comb(n, k)


def arm_values(run: Path) -> dict:
    """Value of always taking each action of each decision: AIPW scores with an outcome model fitted only on earlier
    rows, averaged with weights sqrt(propensity); a lineage row's outcome is its child's own event."""
    saved = json.loads((run / "checkpoint.json").read_text())
    if "ledger" not in saved:
        return {}
    ledger = Ledger()
    ledger.load_state(saved["ledger"])
    rows = defaultdict(list)
    for row in sorted(ledger.rows, key=lambda r: r.closed if r.closed is not None else r.iteration):
        if (row.scope == "lineage" or row.outcome is not None) and len(row.distribution) > 1 and row.decision in DECISIONS:
            rows[row.decision].append(row)
    out = {}
    for decision, local in rows.items():
        for action in sorted({a for r in local for a in r.distribution}):
            sums, counts, total, seen = defaultdict(float), defaultdict(int), 0.0, 0
            scores, weights = [], []
            for row in local:
                y = row.reward if row.scope == "lineage" else row.outcome
                p = row.distribution.get(action, 0.0)
                if p > 0:
                    mean = (total + 1.0) / (seen + 2)
                    base = (sums[action] + 2 * mean) / (counts[action] + 2)
                    scores.append(base + (row.choice == action) * (y - base) / p)
                    weights.append(math.sqrt(p))
                sums[row.choice] += y
                counts[row.choice] += 1
                total += y
                seen += 1
            if not scores:
                continue
            norm = sum(weights)
            value = sum(w * s for w, s in zip(weights, scores)) / norm
            se = math.sqrt(sum(w * w * (s - value) ** 2 for w, s in zip(weights, scores))) / norm
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
    floor = plan.get("gap_floor", GAP_FLOOR)
    arms = {}
    for spec in args.arms:
        arms.update(load_arm(spec))
    report = {"plan": plan, "arms": {}}
    for name, runs in arms.items():
        reached = [next(((e, True) for e, s in run["points"] if s >= plan["threshold"]), (run["end"], False))
                   for run in runs.values()]
        arm = {"runs": len(runs), "final": {k: r["points"][-1][1] if r["points"] else None for k, r in runs.items()},
               "evaluations": {k: r["end"] for k, r in runs.items()},
               "incomplete": sorted(k for k, r in runs.items() if not r["complete"]),
               "to_threshold": kaplan_meier([t for t, _ in reached], [r for _, r in reached], max(plan["budgets"])),
               "at_budget": {}, "frontier": {}}
        for budget in plan["budgets"]:
            values = [best_at(run, budget) for run in runs.values()]
            present = [v for v in values if v is not None]
            gaps = [math.log10(max(plan["reference"] - v, floor)) for v in present]
            arm["at_budget"][budget] = {"observed": len(present), "mean_best": sum(present) / len(present) if present else None,
                                        "mean_log_gap": sum(gaps) / len(gaps) if gaps else None,
                                        "at_or_above_reference": sum(v >= plan["reference"] for v in present)}
            arm["frontier"][budget] = {k: expected_max(values, k) if len(present) == len(values) else None
                                       for k in plan["seeds_k"]}
        spec = next(s for s in args.arms if s.startswith(name + "="))
        if not spec.endswith(".json"):
            dirs = [Path(p) for p in sorted(glob.glob(spec.split("=", 1)[1])) if (Path(p) / "sciel_events.jsonl").exists()]
            arm["llm_usage"] = {d.name: llm_usage(d, runs[d.name]["end"]) for d in dirs}
            arm["approach_families"] = {d.name: families(d) for d in dirs}
            arm["record_attribution"] = {d.name: record_attribution(d) for d in dirs}
            arm["finding_delivery"] = {d.name: delivery(d) for d in dirs}
            if args.values:
                arm["in_run_values"] = {d.name: arm_values(d) for d in dirs if (d / "checkpoint.json").exists()}
        report["arms"][name] = arm
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
