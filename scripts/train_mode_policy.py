"""Train the Stage-A return-conditioned mode policy on the journal corpus.

Usage: python scripts/train_mode_policy.py <out.npz> <run_dir> [<run_dir> ...]

Offline return-conditioned behaviour cloning. Reports a light OPE proxy (held-out
cross-entropy vs the marginal baseline + return-discrimination: does conditioning on a
high target-RTG shift the mode mix toward exploit relative to a low target-RTG?). Full
IPS/OPE (Open Bandit Pipeline) is Stage B; this is the honest data-feasible check.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))          # scripts/ for dt_dataloader
sys.path.insert(0, "src")

from dt_dataloader import held_out_split, load_corpus   # noqa: E402
from sciel.components.policy.mode_features import MODES  # noqa: E402
from sciel.components.policy.mode_policy_mlp import ReturnConditionedModeMLP  # noqa: E402


def _marginal_ce(train, holdout):
    counts = np.array([sum(s["mode"] == m for s in train) for m in MODES], dtype=float)
    p = counts / counts.sum()
    idx = [MODES.index(s["mode"]) for s in holdout]
    return float(-np.log(p[idx] + 1e-12).mean())


def main(out_path, run_dirs):
    all_samples, report = load_corpus(run_dirs)
    print(f"corpus: {len(all_samples)} samples from {len(report['loaded'])} runs "
          f"({len(report['skipped'])} skipped)")
    for run_dir, reason in report["skipped"]:
        print(f"skip {run_dir}: {reason}", file=sys.stderr)
    rtgs = np.array([s["rtg"] for s in all_samples])
    train, holdout = held_out_split(run_dirs, holdout_frac=0.2)
    print(f"split: train={len(train)} holdout={len(holdout)}")
    if len(train) < 20 or not holdout:
        print("too few samples to train/evaluate meaningfully; aborting."); return

    model = ReturnConditionedModeMLP(hidden=16, seed=0).fit(train, epochs=600)
    ce_tr, ce_ho = model.cross_entropy(train), model.cross_entropy(holdout)
    ce_base = _marginal_ce(train, holdout)
    print(f"cross-entropy: train={ce_tr:.3f} holdout={ce_ho:.3f} "
          f"marginal-baseline={ce_base:.3f}  (holdout < baseline => learned signal)")

    # Return-discrimination: at a mid-run context, low vs high target-RTG.
    lo, hi = float(np.percentile(rtgs, 10)), float(np.percentile(rtgs, 90))
    ctx = np.mean([s["features"] for s in train], axis=0).tolist()
    d_lo, d_hi = model.dist(ctx, lo), model.dist(ctx, hi)
    print(f"return-discrimination @mean-context: target_rtg={lo:.3f} -> "
          f"{{'exploit':{d_lo['exploit']:.2f},'explore':{d_lo['explore']:.2f},"
          f"'base':{d_lo['base']:.2f}}}")
    print(f"                                     target_rtg={hi:.3f} -> "
          f"{{'exploit':{d_hi['exploit']:.2f},'explore':{d_hi['explore']:.2f},"
          f"'base':{d_hi['base']:.2f}}}")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    model.save(out_path)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
