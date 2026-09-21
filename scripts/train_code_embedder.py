"""Train a code encoder with contrastive distance and score reconstruction heads.

Program inputs combine evaluator case metrics, token hashes and AST statistics;
the shared featurizer defines their layout. Near-score pairs within each task
are positive examples, and sufficiently separated scores provide negative pairs.
Run or program groups are split before fitting either objective. The distance
threshold is calibrated on training pairs and reported on held-out programs.
Loss = contrastive + LAMBDA * score-MSE.

Usage: python scripts/train_code_embedder.py <out.pt> <run_dir> [<run_dir> ...]
"""

import argparse
import json
import os
import sys

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:
    print("torch required"); sys.exit(1)

# The featurizer is IMPORTED, not copied. It used to be duplicated here, and a
# divergence between the training layout and the inference layout would be silent:
# the model would simply read the wrong slots at run time. Anything that changes the
# feature vector belongs in one place.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "..", "src"))
from sciel.components.policy.code_embed import (      # noqa: E402
    BEHAVIOR_EXTRA_KEYS, BEHAVIOR_KEYS, DIM_HASH, Embedder, featurize)
from sciel.engine.db import normalized_hash  # noqa: E402
from journal_data import (DirtyRun, checkpoint_programs, completed_events,
                          evaluated_code)  # noqa: E402

LAMBDA = 0.5
IN_DIM = len(BEHAVIOR_KEYS) + DIM_HASH + 6 + len(BEHAVIOR_EXTRA_KEYS)

# A retrained embedder invalidates every fitted behavior axis and every stored
# archive coordinate. Refit the axis (scripts/fit_behavior_axis.py) and start a
# new series; do not mix runs across embedder versions.


def load_corpus(path, with_metadata=False):
    """Read successful JSONL rows with code, task, score and evaluator cases.

    Supplied cases provide behavior features; missing feature keys become zero.
    Corpus rows are grouped by normalized code for the training/holdout split.
    This loader does not execute candidates or recompute their observations.
    """
    feats, scores, pos, neg = [], [], [], []
    groups, code_keys = [], []
    by_task = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if not r.get("ok"):
            continue
        idx = len(feats)
        feats.append(featurize(r["code"], r.get("cases") or {}))
        scores.append(float(r.get("score") or 0.0))
        key = normalized_hash(r["code"])
        groups.append(key)  # This corpus loader splits by normalized program identity.
        code_keys.append(key)
        # pairs within a task: near-equal scores are positives, far apart negatives
        prev = by_task.setdefault(r["task"], [])
        for j, sj in prev[-40:]:
            delta = abs(scores[idx] - sj)
            if delta < 0.005:
                pos.append((j, idx))
            elif delta > 0.1:
                neg.append((j, idx))
        prev.append((idx, scores[idx]))
    data = (np.array(feats, dtype=np.float32), np.array(scores, dtype=np.float32), pos, neg)
    return data + (groups, code_keys) if with_metadata else data


def load(run_dirs, with_metadata=False):
    feats, scores, pos, neg = [], [], [], []
    groups, code_keys = [], []
    for d in run_dirs:
        by_id, sc = {}, {}
        path = os.path.join(d, "sciel_events.jsonl")
        try:
            events, end = completed_events(path)
            programs = checkpoint_programs(path, end=end)
        except (OSError, DirtyRun) as exc:
            print(f"skip {d}: {exc}", file=sys.stderr)
            continue
        for e in events:
            if e.get("event") != "iteration" or e.get("child_score") is None:
                continue
            cid, pid = e.get("child_id"), e.get("parent_id")
            if not cid:
                print(f"skip {path} iteration {e.get('iteration_no')}: missing child_id",
                      file=sys.stderr)
                continue
            code = evaluated_code(e, programs)
            if code is None:
                print(f"skip {path} child {cid}: missing evaluated source", file=sys.stderr)
                continue
            idx = len(feats)
            feats.append(featurize(code, e.get("cases") or {}))
            scores.append(float(e["child_score"]))
            groups.append(os.path.abspath(d))
            code_keys.append(normalized_hash(code))
            by_id[cid] = idx
            sc[cid] = e["child_score"]
            if pid in by_id:
                delta = abs(e["child_score"] - sc.get(pid, 0.0))
                (pos if delta < 0.005 else neg if delta > 0.1 else []).append((by_id[pid], idx))
    data = (np.array(feats, dtype=np.float32), np.array(scores, dtype=np.float32), pos, neg)
    return data + (groups, code_keys) if with_metadata else data


def split_programs(groups, code_keys, holdout_frac=0.2, seed=0):
    """Split runs (or corpus program groups) before either objective sees data.

    Programs copied across runs cannot leak through the run boundary: remove
    overlapping code from training. A single run is split by distinct program.
    """
    if len(groups) != len(code_keys):
        raise ValueError("group and program identities must have the same length")
    groups = list(groups)
    if len(set(groups)) < 2:
        groups = list(code_keys)
    unique = sorted(set(groups))
    if len(unique) < 2:
        raise ValueError("at least two distinct programs/groups are needed for holdout")
    rng = np.random.default_rng(seed)
    n_hold = min(len(unique) - 1, max(1, round(holdout_frac * len(unique))))
    held_groups = {unique[i] for i in rng.permutation(len(unique))[:n_hold]}
    test = np.array([i for i, g in enumerate(groups) if g in held_groups], dtype=int)
    held_codes = {code_keys[i] for i in test}
    train = np.array([i for i, g in enumerate(groups)
                      if g not in held_groups and code_keys[i] not in held_codes], dtype=int)
    if not len(train):
        raise ValueError("holdout overlaps every training program; no independent split")
    return train, test


def split_pairs(pairs, train_indices, test_indices):
    train, test = set(train_indices), set(test_indices)
    return ([p for p in pairs if p[0] in train and p[1] in train],
            [p for p in pairs if p[0] in test and p[1] in test])


def fit_and_evaluate(X, Y, pos, neg, train_indices, test_indices, epochs=40):
    """Train on train programs only; calibrate the distance threshold there too."""
    train_indices = np.asarray(train_indices, dtype=int)
    test_indices = np.asarray(test_indices, dtype=int)
    if not len(train_indices) or not len(test_indices):
        raise ValueError("both program splits must be non-empty")
    if set(train_indices) & set(test_indices):
        raise ValueError("training and holdout programs overlap")
    pos_tr, pos_te = split_pairs(pos, train_indices, test_indices)
    neg_tr, neg_te = split_pairs(neg, train_indices, test_indices)
    if not all((pos_tr, neg_tr, pos_te, neg_te)):
        raise ValueError("each split needs positive and negative pairs")
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    model = Embedder(X.shape[1])
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    for _ in range(epochs):
        for pairs, is_pos in ((pos_tr, True), (neg_tr, False)):
            idx = rng.permutation(len(pairs))[:2000]
            a = Xt[[pairs[i][0] for i in idx]]
            b = Xt[[pairs[i][1] for i in idx]]
            distance = 1 - (model.embed(a) * model.embed(b)).sum(-1)
            loss_c = ((distance ** 2).mean() if is_pos else
                      (torch.clamp(0.4 - distance, 0) ** 2).mean())
            score_indices = rng.permutation(train_indices)[:2000]
            loss_s = ((model.predict_score(Xt[score_indices]) - Yt[score_indices]) ** 2).mean()
            opt.zero_grad()
            (loss_c + LAMBDA * loss_s).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        def distances(pairs):
            a = model.embed(Xt[[p[0] for p in pairs]])
            b = model.embed(Xt[[p[1] for p in pairs]])
            return (1 - (a * b).sum(-1)).numpy()
        threshold = float((distances(pos_tr).mean() + distances(neg_tr).mean()) / 2)
        dp, dn = distances(pos_te), distances(neg_te)
        accuracy = float(((dp < threshold).mean() + (dn >= threshold).mean()) / 2)
        prediction = model.predict_score(Xt[test_indices]).numpy()
        targets = Y[test_indices]
        correlation = (float(np.corrcoef(prediction, targets)[0, 1])
                       if prediction.std() > 0 and targets.std() > 0 else None)
    return model, {"separation_accuracy": accuracy, "score_correlation": correlation,
                   "distance_threshold": threshold,
                   "n_train": len(train_indices), "n_holdout": len(test_indices)}


# The model class is IMPORTED too, for the same reason as the featurizer: a second
# copy here would let the training architecture drift from the one that loads the
# checkpoint, and the mismatch would surface as a state-dict error at best and as
# silently different activations at worst.


def main(out, sources):
    if len(sources) == 1 and sources[0].endswith(".jsonl"):
        print(f"corpus: {sources[0]}  (using supplied evaluator cases)")
        X, Y, pos, neg, groups, code_keys = load_corpus(sources[0], with_metadata=True)
    else:
        print("source: run journals (missing behavior fields become zero inputs)")
        X, Y, pos, neg, groups, code_keys = load(sources, with_metadata=True)
    beh = 0
    if len(X):
        tail = X[:, -len(BEHAVIOR_EXTRA_KEYS):]
        beh = int((abs(tail).sum(axis=1) > 0).sum())
    print(f"programs with a behaviour readout: {beh} of {len(X)}; score head weight "
          f"{LAMBDA}")
    print(f"programs: {len(X)}, pos pairs: {len(pos)}, neg pairs: {len(neg)}, "
          f"feature dim: {X.shape[1] if len(X) else 0}")
    if len(pos) < 200 or len(neg) < 200:
        print("insufficient pairs"); return
    try:
        train_indices, test_indices = split_programs(groups, code_keys)
        m, metrics = fit_and_evaluate(X, Y, pos, neg, train_indices, test_indices)
    except ValueError as exc:
        print(f"insufficient independent training/holdout data: {exc}")
        return
    print(f"split: train={metrics['n_train']} holdout={metrics['n_holdout']}")
    print(f"held-out behavior separation acc: {metrics['separation_accuracy']:.3f}")
    print(f"held-out score-prediction corr:   {metrics['score_correlation']}")
    torch.save({"state": m.state_dict(), "in_dim": X.shape[1],
                "behavior_keys": list(BEHAVIOR_KEYS),
                "behavior_extra_keys": list(BEHAVIOR_EXTRA_KEYS),
                "dim_hash": DIM_HASH, "holdout_metrics": metrics,
                "holdout_protocol": "program_disjoint/1"}, out)
    print("saved ->", out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("sources", nargs="+",
                    help="run dirs, or a single .jsonl corpus with code, task, score and cases")
    ap.add_argument("--score-weight", type=float, default=LAMBDA,
                    help="weight of the score head in the loss. It is what organizes "
                         "the latent by quality: at 0.5 the first principal direction "
                         "correlated 0.85 with the score, which is the grid's OTHER "
                         "axis, so the behaviour axis had to be a later component")
    a = ap.parse_args()
    LAMBDA = a.score_weight
    main(a.out, a.sources)
