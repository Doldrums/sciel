"""Fit and freeze a MAP-Elites behavior axis from pooled program observations.

Inputs pair saved code and evaluator cases with scores and task labels. Embed
programs with one fixed encoder, then select a principal component using explained
variance, per-task score correlation and bin occupancy. The artifact binds the
encoder hash and fitted normalization so archive coordinates remain stable.

Usage:
    python scripts/fit_behavior_axis.py OUT.npz EMBEDDER.pt RUN_DIR [RUN_DIR ...]
    python scripts/fit_behavior_axis.py OUT.npz EMBEDDER.pt /path/to/runs --glob
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from sciel.components.policy.code_embed import (            # noqa: E402
    BEHAVIOR_EXTRA_KEYS, BEHAVIOR_KEYS, DIM_HASH, BehaviorAxis, Embedder,
    embed_batch, file_sha)

MAX_COMPONENT = 6
CORR_LIMIT = 0.35          # a behavior axis this correlated with score is the score
MIN_EXPLAINED = 0.05       # below this a component is mostly noise


def collect_corpus(path, only_task=None):
    """Read successful JSONL rows with code, cases, score and task fields.

    Cases supply the encoder's behavior features. Missing feature keys become
    zero through the shared featurizer; observations are not recomputed here.
    """
    progs, scores, tasks, seen = [], [], [], set()
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if not r.get("ok") or (only_task and r.get("task") != only_task):
            continue
        code = r.get("code") or ""
        if not code or code in seen:
            continue
        seen.add(code)
        progs.append((code, r.get("cases") or {}))
        tasks.append(r.get("task") or "?")
        try:
            scores.append(float(r.get("score") or 0.0))
        except (TypeError, ValueError):
            scores.append(0.0)
    return progs, np.asarray(scores, dtype=np.float64), np.asarray(tasks)
UNIFORMITY_SLACK = 0.04    # fills 'essentially as well as' the best filler


def collect(run_dirs, only_task=None):
    """(programs, scores, tasks) for every distinct program in the pooled checkpoints.

    Checkpoints carry the code; journals do not. Programs are de-duplicated by code
    so that one popular ancestor copied across islands does not dominate the fit.
    The task label is kept because the selection rule below is per-task."""
    progs, scores, tasks, seen = [], [], [], set()
    for d in run_dirs:
        cp = os.path.join(d, "checkpoint.json")
        if not os.path.exists(cp):
            continue
        try:
            state = json.load(open(cp))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  skip {d}: {e}", file=sys.stderr)
            continue
        task = ((state.get("config") or {}).get("task")) or "?"
        if only_task and task != only_task:
            continue
        for p in (state.get("db") or {}).get("programs") or []:
            code = p.get("code") or ""
            if not code or code in seen:
                continue
            seen.add(code)
            cases = p.get("cases") or {}
            if not isinstance(cases, dict):
                cases = {}
            progs.append((code, cases))
            tasks.append(task)
            try:
                scores.append(float(p.get("score") or 0.0))
            except (TypeError, ValueError):
                scores.append(0.0)
    return progs, np.asarray(scores, dtype=np.float64), np.asarray(tasks)


MIN_TASK_N = 50          # below this a per-task correlation is noise, not evidence


def per_task_corr(u, scores, tasks):
    """Correlation of one axis with the score, computed SEPARATELY inside each task.

    This is the number that matters and the pooled one is not. A run's grid only ever
    contains one task, so an axis is a duplicate of the score axis whenever it
    correlates with the score inside SOME task -- and pooling tasks whose scores live
    on different scales can hide that relationship."""
    out = {}
    for t in sorted(set(tasks.tolist())):
        sel = tasks == t
        if sel.sum() < MIN_TASK_N or scores[sel].std() == 0 or u[sel].std() == 0:
            continue
        out[t] = float(np.corrcoef(u[sel], scores[sel])[0, 1])
    return out


def embed_all(model, progs, batch=512):
    return np.concatenate([embed_batch(model, progs[i:i + batch])
                           for i in range(0, len(progs), batch)])


def bin_profile(values, bins=10):
    idx = np.clip((np.asarray(values) * bins).astype(int), 0, bins - 1)
    counts = np.bincount(idx, minlength=bins) / max(len(idx), 1)
    nz = counts[counts > 0]
    uniformity = float(-(nz * np.log(nz)).sum() / np.log(bins)) if len(nz) > 1 else 0.0
    return counts, int((counts > 0).sum()), uniformity


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("embedder")
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--glob", action="store_true",
                    help="treat RUN_DIRs as roots and search them for checkpoints")
    ap.add_argument("--component", type=int, default=None,
                    help="force a 0-based component instead of the selection rule")
    ap.add_argument("--corr-limit", type=float, default=CORR_LIMIT)
    ap.add_argument("--min-explained", type=float, default=MIN_EXPLAINED,
                    help="ignore components below this share of the latent's variance; "
                         "a near-noise direction spreads beautifully and means nothing")
    ap.add_argument("--version", default="v1")
    ap.add_argument("--task", default=None,
                    help="fit on ONE task only; the safest option when no component "
                         "clears the limit on every task at once")
    ap.add_argument("--dry-run", action="store_true", help="report, write nothing")
    a = ap.parse_args(argv)

    dirs = list(a.run_dirs)
    if a.glob and not (len(dirs) == 1 and dirs[0].endswith(".jsonl")):
        dirs = sorted({os.path.dirname(p) for root in a.run_dirs
                       for p in glob.glob(os.path.join(root, "**", "checkpoint.json"),
                                          recursive=True)})
    print(f"run dirs: {len(dirs)}")

    if len(a.run_dirs) == 1 and a.run_dirs[0].endswith(".jsonl"):
        print(f"corpus: {a.run_dirs[0]}")
        progs, scores, tasks = collect_corpus(a.run_dirs[0], only_task=a.task)
    else:
        progs, scores, tasks = collect(dirs, only_task=a.task)
    if len(progs) < 200:
        print(f"only {len(progs)} distinct programs -- too few to fit 64 parameters",
              file=sys.stderr)
        return 2
    counts = {t: int((tasks == t).sum()) for t in sorted(set(tasks.tolist()))}
    print(f"distinct programs: {len(progs)}")
    print("by task: " + ", ".join(f"{t} {n}" for t, n in counts.items()))
    biggest = max(counts.values()) / len(progs) if counts else 0.0
    if biggest > 0.6 and not a.task:
        print(f"  note: one task is {biggest:.0%} of the pool, so the principal "
              f"directions are largely that task's. Consider --task.")

    model = Embedder.load(a.embedder)
    want = len(BEHAVIOR_KEYS) + DIM_HASH + 6 + len(BEHAVIOR_EXTRA_KEYS)
    if int(getattr(model, "in_dim", want)) < want:
        print(f"\n  WARNING: {a.embedder} has in_dim={model.in_dim} but the featurizer "
              f"produces {want}. embed_batch slices off the tail, which is exactly the "
              f"BEHAVIOR_EXTRA_KEYS block, so the evaluators' behaviour readout does NOT "
              f"reach this latent and cannot influence the axis fitted below. "
              f"Retrain the embedder (scripts/train_code_embedder.py) to close that gap.\n",
              file=sys.stderr)
    Z = embed_all(model, progs)
    mean = Z.mean(0)
    Zc = Z - mean
    # SVD of the centred latent: rows of Vt are the principal directions.
    _, S, Vt = np.linalg.svd(Zc, full_matrices=False)
    explained = (S ** 2) / float((S ** 2).sum())

    print(f"\n{'comp':>4} {'explained':>10} {'pooled':>8} {'worst task':>11} "
          f"{'which':>26} {'bins':>5} {'unif':>6}")
    stats = []
    for j in range(min(MAX_COMPONENT, Vt.shape[0])):
        u = Zc @ Vt[j]
        sd = float(u.std())
        if sd <= 0:
            continue
        corr = float(np.corrcoef(u, scores)[0, 1]) if scores.std() > 0 else 0.0
        by_task = per_task_corr(u, scores, tasks)
        worst_task, worst = max(by_task.items(), key=lambda kv: abs(kv[1])) \
            if by_task else ("", corr)
        vals = (np.tanh((u - u.mean()) / sd) + 1.0) / 2.0
        _, occupied, uniformity = bin_profile(vals)
        stats.append({"j": j, "corr": corr, "worst": worst, "worst_task": worst_task,
                      "by_task": by_task, "explained": float(explained[j]),
                      "center": float(u.mean()), "scale": sd,
                      "occupied": occupied, "uniformity": uniformity})
        print(f"{j:>4} {explained[j]:>10.4f} {corr:>8.3f} {worst:>11.3f} "
              f"{worst_task[-26:]:>26} {occupied:>5} {uniformity:>6.3f}")

    if a.component is not None:
        chosen = next((c for c in stats if c["j"] == a.component), None)
        if chosen is None:
            print(f"component {a.component} not available", file=sys.stderr)
            return 2
    else:
        # Two criteria, in this order.
        #
        # 1. The WORST per-task correlation, not the pooled one: choosing on the
        #    pooled number is how an axis that duplicates the score inside one task
        #    passes a gate meant to forbid exactly that.
        # 2. Among the survivors, the one that FILLS the axis best, not the one with
        #    the most variance. Taking the leading survivor picked a component sitting
        #    at 0.331 against a 0.35 limit that occupied 9 bins at uniformity 0.75,
        #    while a later component reached 10 bins at 0.97.
        #
        # The variance floor is what keeps criterion 2 honest: a near-noise direction
        # spreads perfectly and separates nothing.
        eligible = [c for c in stats if abs(c["worst"]) < a.corr_limit]
        signal = [c for c in eligible if c["explained"] >= a.min_explained]
        pool = signal or eligible
        if signal is not eligible and eligible and not signal:
            print(f"  note: no component clears both the correlation limit and "
                  f"{a.min_explained:.0%} of the variance; falling back to variance order.")
        # Among components that fill the axis essentially as well as the best filler,
        # take the one carrying the most signal. Maximising uniformity alone traded
        # 23% of the latent's variance for three hundredths of entropy.
        if pool:
            best_fill = max(c["uniformity"] for c in pool)
            well_filled = [c for c in pool if c["uniformity"] >= best_fill - UNIFORMITY_SLACK]
            chosen = max(well_filled, key=lambda c: c["explained"])
        else:
            chosen = None
        if chosen is None:
            print(f"\nno component keeps |corr with score| < {a.corr_limit} in EVERY "
                  f"task. Best worst-case was component "
                  f"{min(stats, key=lambda c: abs(c['worst']))['j']} at "
                  f"{min(abs(c['worst']) for c in stats):.3f}.\n"
                  "This embedder cannot supply one behavior axis that works for all of "
                  "them at once. Either fit per task (--task NAME) or refit the "
                  "embedder without the score head dominating the latent.",
                  file=sys.stderr)
            return 3
    print(f"\nchosen: component {chosen['j']}  pooled {chosen['corr']:+.3f}, "
          f"worst task {chosen['worst']:+.3f} ({chosen['worst_task']}) < {a.corr_limit}, "
          f"explains {chosen['explained']:.1%}, uniformity {chosen['uniformity']:.3f}")
    for t, c in sorted(chosen["by_task"].items()):
        print(f"    {t:34s} {c:+.3f}")

    axis = BehaviorAxis(Vt[chosen["j"]], mean, chosen["center"], chosen["scale"],
                        component=chosen["j"], version=a.version,
                        embed_sha=file_sha(a.embedder), corr_score=chosen["corr"],
                        explained=chosen["explained"], n_fit=len(progs),
                        corr_worst=chosen["worst"], corr_worst_task=chosen["worst_task"],
                        fit_task=a.task or "")
    vals = np.array([axis.value(z) for z in Z])
    counts, occupied, uniformity = bin_profile(vals)
    print(f"descriptor: std {vals.std():.3f}, bins occupied {occupied}/10, "
          f"uniformity {uniformity:.3f}")
    print("occupancy: " + " ".join(f"{c:.3f}" for c in counts))

    if a.dry_run:
        print("\ndry run: nothing written")
        return 0
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    axis.save(a.out)
    print(f"\nwrote {a.out} (embedder {axis.embed_sha}, {axis.n_fit} programs)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
