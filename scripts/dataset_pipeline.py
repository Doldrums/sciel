"""Phase-2 dataset smoke: extract features+propensity+outcome from run
journals, fit a logistic baseline, print a calibration report. Pipeline, not model.

Trains ONLY on new-engine (v1+) journals.
Usage: dataset_pipeline.py <run_dir> [<run_dir> ...]
"""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from journal_data import DirtyRun, completed_events


def extract(journal_path):
    events, _ = completed_events(journal_path)
    start = next((e for e in events if e.get("event") == "run_start"), None)
    if start is None or (start.get("non_comparable") is None
                         and start.get("schema_version") is None):
        raise DirtyRun(f"{journal_path}: not a v1 journal (refuse old-repo data)")
    rows = []
    for e in events:
        if e.get("event") != "iteration" or e.get("child_score") is None:
            continue
        rows.append({
            "parent_score": _f(e.get("island_best")) - _f(e.get("d_isl")),
            "island": _f(e.get("island")),
            "phase": _phase_code(e.get("op_stat")),
            "op_diff": 1.0 if e.get("op_used") == "diff" else 0.0,
            "op_full": 1.0 if e.get("op_used") == "full_rewrite" else 0.0,
            "parent_propensity": _f(e.get("parent_propensity"), 1.0),
            "op_propensity": _f(e.get("op_propensity"), 1.0),
            "y": 1.0 if e.get("is_island_event") else 0.0,
        })
    return rows


def _f(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _phase_code(op_stat):
    return 0.0


def _design(rows):
    feats = ["parent_score", "island", "op_diff", "op_full",
             "parent_propensity", "op_propensity"]
    x = [[r[f] for f in feats] for r in rows]
    y = [r["y"] for r in rows]
    return feats, x, y


def _standardize(x):
    n, d = len(x), len(x[0])
    means = [sum(row[j] for row in x) / n for j in range(d)]
    stds = []
    for j in range(d):
        var = sum((row[j] - means[j]) ** 2 for row in x) / n
        stds.append(math.sqrt(var) or 1.0)
    return [[(row[j] - means[j]) / stds[j] for j in range(d)] for row in x]


def fit_logistic(x, y, epochs=300, lr=0.1, l2=1e-3):
    n, d = len(x), len(x[0])
    w = [0.0] * d
    b = 0.0
    for _ in range(epochs):
        gw = [0.0] * d
        gb = 0.0
        for i in range(n):
            z = b + sum(w[j] * x[i][j] for j in range(d))
            p = 1.0 / (1.0 + math.exp(-z))
            err = p - y[i]
            for j in range(d):
                gw[j] += err * x[i][j]
            gb += err
        for j in range(d):
            w[j] -= lr * (gw[j] / n + l2 * w[j])
        b -= lr * gb / n
    return w, b


def predict(x, w, b):
    return [1.0 / (1.0 + math.exp(-(b + sum(w[j] * row[j] for j in range(len(w))))))
            for row in x]


def calibration_report(y, p, bins=5):
    base = sum(y) / len(y)
    brier = sum((p[i] - y[i]) ** 2 for i in range(len(y))) / len(y)
    brier_base = sum((base - y[i]) ** 2 for i in range(len(y))) / len(y)
    print(f"n={len(y)} base_rate={base:.3f} brier={brier:.4f} "
          f"brier_baseline(const)={brier_base:.4f} skill={1 - brier / brier_base:+.3f}")
    print("reliability (pred bin -> empirical):")
    for k in range(bins):
        lo, hi = k / bins, (k + 1) / bins
        idx = [i for i in range(len(p)) if lo <= p[i] < hi or (k == bins - 1 and p[i] == 1.0)]
        if not idx:
            continue
        emp = sum(y[i] for i in idx) / len(idx)
        avg = sum(p[i] for i in idx) / len(idx)
        print(f"  [{lo:.1f},{hi:.1f}) n={len(idx):3d} pred={avg:.3f} emp={emp:.3f}")


def main(argv):
    rows = []
    for run_dir in argv[1:]:
        try:
            rows += extract(f"{run_dir}/sciel_events.jsonl")
        except (OSError, DirtyRun) as exc:
            print(f"skip {run_dir}: {exc}", file=sys.stderr)
    if not rows:
        raise SystemExit("no gen rows extracted")
    feats, x, y = _design(rows)
    xs = _standardize(x)
    w, b = fit_logistic(xs, y)
    p = predict(xs, w, b)
    print("features:", feats)
    print("weights:", [round(v, 3) for v in w], "bias:", round(b, 3))
    calibration_report(y, p)


if __name__ == "__main__":
    main(sys.argv)
