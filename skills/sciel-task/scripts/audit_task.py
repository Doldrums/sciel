"""Audit a task before a run: noise radius, leak probes, landscape from a trace, and admission.

Usage: audit_task.py TASK_DIR --programs DIR [--run RUN_DIR] [--budget B] [--upper U]
                     [--repeats K] [--delta D] [--json OUT]

SCiEL is imported from the environment, or from $SCIEL_REPO/src when that is set.

Programs in DIR are recognised by file name: seed.py (default: the task's initial
program), expert*.py (reference solutions), cheat*.py (leak probes that must not beat
the seed) and strawman*.py (strong lookups the expert must beat on the confirm slice).
The landscape and the feasible share come from RUN_DIR's checkpoint when given.
"""

import argparse
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
from scipy import stats

if os.environ.get("SCIEL_REPO"):
    sys.path.insert(0, str(Path(os.environ["SCIEL_REPO"]).expanduser() / "src"))
from sciel.eval.registry import load_task
from sciel.eval.sandbox import run_stage

K_ADMIT = 5


def measure(task, path, stage):
    timeout = task.full_timeout if stage == "full" else task.confirm_timeout
    n = task.confirm_n if stage == "confirm" else None
    r = run_stage(task.evaluator_path, stage, str(path), timeout=timeout, n=n,
                  cpu_seconds=task.cpu_seconds, mem_bytes=task.mem_bytes)
    return {"ok": r.ok, "score": float(r.score) if r.ok else None, "cases": dict(r.cases)}


def roles(directory, task):
    files = sorted(Path(directory).glob("*.py"))
    seed = next((f for f in files if f.stem == "seed"), Path(task.initial_program_path))
    pick = lambda prefix: [f for f in files if f.stem.startswith(prefix)]
    return {"seed": [seed], "expert": pick("expert"), "cheat": pick("cheat"), "strawman": pick("strawman")}


def noise(task, programs, measured, repeats):
    """Evaluator spread on one program, and the standard error of a paired difference."""
    paired = [f for role in ("expert", "strawman", "cheat") for f in programs[role]
              if measured[str(f)]["full"]["cases"].get("delta_wp")]
    probe = (paired or programs["expert"] or programs["seed"])[0]
    again = [measure(task, probe, "full")["score"] for _ in range(repeats)]
    again = [s for s in again if s is not None]
    sigma_e = float(np.std(again, ddof=1)) if len(again) > 1 else 0.0
    cases = measured[str(probe)]["full"]["cases"]
    seed = measured[str(programs["seed"][0])]
    if cases.get("delta_wp") and "delta_lo" in cases:
        scale = (measured[str(probe)]["full"]["score"] - seed["full"]["score"]) / cases["delta_wp"]
        return sigma_e, abs(scale) * (cases["delta_wp"] - cases["delta_lo"]) / 1.645, "evaluator cluster bootstrap"
    rows = [m for m in measured.values() if m["full"]["score"] and m.get("confirm", {}).get("score")]
    if len(rows) < 3 or not seed.get("confirm", {}).get("score"):
        return sigma_e, None, "not enough programs with two slices"
    gaps = [(m["full"]["score"] - seed["full"]["score"]) - (m["confirm"]["score"] - seed["confirm"]["score"])
            for m in rows]
    return sigma_e, float(np.std(gaps, ddof=1) / math.sqrt(2)), "select-vs-confirm disagreement (rough)"


def landscape(run_dir, eps):
    """Improvement probability and parent-child correlation along recorded edits."""
    ck = json.loads((Path(run_dir) / "checkpoint.json").read_text())
    progs = {p["id"]: p for p in ck["db"]["programs"]}
    pairs = [(float(progs[p["parent_id"]]["score"]), float(p["score"])) for p in progs.values()
             if p["parent_id"] in progs]
    ok = int(ck.get("n_evaluation_results", 0)) - int(ck.get("n_eval_fail", 0))
    feasible = ok / max(int(ck.get("n_evaluation_results", 0)), 1)
    if len(pairs) < 3:
        return {"pairs": len(pairs), "feasible_share": feasible}
    hp, hc = map(np.array, zip(*pairs))
    frontier = hp >= np.quantile(hp, 0.75)

    def rate(mask):
        n, x = int(mask.sum()), int(((hc > hp + eps) & mask).sum())
        upper = float(stats.beta.ppf(0.975, x + 1, n - x)) if x < n else 1.0
        return {"n": n, "up": x, "pi": x / n if n else None, "pi_upper": upper}

    return {"pairs": len(pairs), "feasible_share": feasible, "rho1": float(np.corrcoef(hp, hc)[0, 1]),
            "all": rate(np.ones_like(frontier)), "frontier": rate(frontier)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task")
    ap.add_argument("--programs", required=True)
    ap.add_argument("--run")
    ap.add_argument("--budget", type=int, default=200)
    ap.add_argument("--upper", type=float)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--delta", type=float, default=0.05)
    ap.add_argument("--json")
    args = ap.parse_args()
    task = load_task(args.task)
    programs = roles(args.programs, task)
    stages = ["full"] + (["confirm"] if "confirm" in task.tiers or task.confirm_timeout else [])
    measured = {}
    for role, files in programs.items():
        for f in files:
            measured[str(f)] = {"role": role, **{s: measure(task, f, s) for s in stages}}
            print(f"{role:9s} {f.name:28s} " + "  ".join(f"{s} {measured[str(f)][s]['score']}" for s in stages),
                  flush=True)
    sigma_e, sigma_d, how = noise(task, programs, measured, args.repeats)
    z = float(stats.norm.ppf(1 - args.delta / (2 * args.budget)))
    eps = (sigma_d or 0.0) * z + sigma_e
    full = lambda f: measured[str(f)]["full"]["score"] or 0.0
    conf = lambda f: (measured[str(f)].get("confirm") or {}).get("score") or 0.0
    seed = full(programs["seed"][0])
    lower = max([seed] + [full(f) for f in programs["expert"]])
    leaks = {f.name: full(f) for f in programs["cheat"]}
    leak_free = all(s <= seed + eps for s in leaks.values())
    best_expert = max((conf(f) for f in programs["expert"]), default=None)
    best_straw = max((conf(f) for f in programs["strawman"]), default=None)
    mechanism = None if best_expert is None or best_straw is None else best_expert > best_straw + eps
    land = landscape(args.run, eps) if args.run else None
    gap_ok = None if args.upper is None else (args.upper - lower) > 4 * eps
    pi_ok = None if not land or "frontier" not in land else land["frontier"]["pi_upper"] >= K_ADMIT / args.budget
    gates = [leak_free, mechanism, gap_ok, pi_ok]
    verdict = ("rejected" if any(g is False for g in gates)
               else "undetermined" if gap_ok is None or pi_ok is None else "admitted")
    report = {"task": task.qualified_name, "budget": args.budget, "sigma_e": sigma_e, "sigma_d": sigma_d,
              "sigma_d_method": how, "z": z, "eps": eps, "lower": lower, "upper": args.upper,
              "leaks": leaks, "leak_free": leak_free, "expert_vs_strawman_confirm": [best_expert, best_straw],
              "mechanism_distinguished": mechanism, "landscape": land, "gap_ok": gap_ok, "pi_ok": pi_ok,
              "min_detectable_gain": eps, "verdict": verdict, "programs": measured}
    print(json.dumps({k: v for k, v in report.items() if k != "programs"}, indent=1, default=str))
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1, default=str))


if __name__ == "__main__":
    main()
