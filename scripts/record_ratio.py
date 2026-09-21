"""Generated-child records, computed from the same events as Engine.record_stats.

Seeds, portfolio starts and dynamic island births set the current best, but do not
count as generated-child records or evaluations. H_n normalizes for run length;
with ties and a pre-existing best it is not an exact exchangeable null expectation.

Usage:
    python scripts/record_ratio.py outputs/ab_refine --by-arm
    python scripts/record_ratio.py outputs --glob
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from journal_data import DirtyRun, completed_events


def harmonic(n: int) -> float:
    return sum(1.0 / k for k in range(1, n + 1)) if n > 0 else 0.0


def ratio_for(journal_path: str) -> dict | None:
    """Count accepted child events, never differences between population snapshots.

    The shared reader selects checkpoint ancestry before counting. Evaluation
    indices must then be unique and contiguous; they cannot establish ancestry.
    Old journals need is_pop_event or d_pop: a population maximum alone cannot
    distinguish a generated child from a spawn.
    """
    children = []
    task = arm = identity = None
    try:
        events, end = completed_events(journal_path)
        for e in events:
            kind = e.get("event")
            if kind == "run_start":
                this_identity = (e.get("task"), e.get("task_hash"), e.get("seed"))
                if identity is not None and identity != this_identity:
                    raise DirtyRun("different run identities appended to one journal")
                identity = this_identity
                task = e.get("task")
                arm = (e.get("config") or {}).get("arm") or e.get("arm")
            elif kind == "iteration" and e.get("child_score") is not None:
                n = e.get("B_evals")
                if type(n) is not int or n != len(children) + 1:
                    raise DirtyRun("duplicate, missing or out-of-order generated-child B_evals")
                if isinstance(e.get("is_pop_event"), bool):
                    record = e["is_pop_event"]
                elif e.get("d_pop") is not None:
                    try:
                        gain = float(e["d_pop"])
                    except (TypeError, ValueError, OverflowError) as exc:
                        raise DirtyRun("invalid generated-child d_pop") from exc
                    if not math.isfinite(gain):
                        raise DirtyRun("non-finite generated-child d_pop")
                    record = gain > 0.0
                else:
                    raise DirtyRun("generated child lacks is_pop_event or d_pop")
                children.append(int(record))
        n = end.get("B_evals")
        if type(n) is not int or n != len(children):
            raise DirtyRun("run_end B_evals disagrees with generated-child events")
        if n < 20:
            raise DirtyRun(f"only {n} generated children; at least 20 required")
    except (OSError, DirtyRun) as exc:
        print(f"skip {journal_path}: {exc}", file=sys.stderr)
        return None
    records = sum(children)
    expected = harmonic(n)
    return {"run": journal_path, "task": task, "arm": arm, "n": n,
            "records": records, "expected": expected,
            "ratio": records / expected if expected else 0.0}


def _arm_of(run_path: str, row: dict) -> str:
    if row.get("arm"):
        return str(row["arm"])
    # fall back to the directory name with a trailing _sNN seed suffix removed
    name = os.path.basename(os.path.dirname(run_path))
    return name.rsplit("_s", 1)[0] if "_s" in name else name


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="+")
    ap.add_argument("--glob", action="store_true",
                    help="search the roots recursively for journals")
    ap.add_argument("--by-arm", action="store_true",
                    help="also aggregate by arm, pooling records and expectations")
    ap.add_argument("--json", action="store_true", help="emit rows as JSON")
    a = ap.parse_args(argv)

    paths = []
    for root in a.roots:
        if os.path.isfile(root):
            paths.append(root)
        elif a.glob:
            paths += glob.glob(os.path.join(root, "**", "sciel_events.jsonl"),
                               recursive=True)
        else:
            paths += glob.glob(os.path.join(root, "*", "sciel_events.jsonl"))
            paths += glob.glob(os.path.join(root, "sciel_events.jsonl"))
    rows = [r for r in (ratio_for(p) for p in sorted(set(paths))) if r]
    if not rows:
        print("no journals with enough observations", file=sys.stderr)
        return 2

    if a.json:
        print(json.dumps(rows, indent=1))
        return 0

    ratios = sorted(r["ratio"] for r in rows)
    below = sum(1 for r in ratios if r < 1.0)
    print(f"runs: {len(rows)}")
    print(f"  median ratio      {statistics.median(ratios):.3f}")
    print(f"  quartiles         {ratios[len(ratios)//4]:.3f} .. "
          f"{ratios[3*len(ratios)//4]:.3f}")
    print(f"  below 1.0         {below} of {len(rows)}  "
          f"(harmonic normalization; not a permutation test)")

    if a.by_arm:
        arms: dict = {}
        for r in rows:
            key = _arm_of(r["run"], r)
            slot = arms.setdefault(key, {"records": 0, "expected": 0.0, "runs": 0})
            slot["records"] += r["records"]
            slot["expected"] += r["expected"]
            slot["runs"] += 1
        print(f"\n{'arm':28s} {'runs':>5} {'records':>8} {'expected':>9} {'ratio':>7}")
        for key, v in sorted(arms.items(), key=lambda kv: -kv[1]["records"] / max(kv[1]["expected"], 1e-9)):
            # pooled, not a mean of ratios: short runs must not weigh as much as long
            print(f"{key[-28:]:28s} {v['runs']:>5} {v['records']:>8} "
                  f"{v['expected']:>9.1f} {v['records']/v['expected']:>7.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
