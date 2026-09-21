"""Train and evaluate a review candidate with immutable train/calibration/test roles.

The clean corpus owns run splits and evaluated program provenance. This pipeline
does not choose new splits or use test data for training, component selection,
distance thresholds, or the annealing temperature. Existing v3 has no exposure
manifest: its numbers are retrospective diagnostics, never a fresh holdout claim.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(HERE.parent / "src")]
from sciel.components.policy.code_embed import (
    BEHAVIOR_EXTRA_KEYS, BEHAVIOR_KEYS, DIM_HASH, BehaviorAxis, Embedder,
    embed_batch, featurize, file_sha)
from sciel.engine.db import normalized_hash
from fit_behavior_axis import bin_profile

PARTS = ("train", "validation", "test")
TASKS = ("oe/circle_packing", "oe/function_minimization", "oe/heilbronn_triangle")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def read_corpus(path):
    """Reject inconsistent splits; retain failures separately from trainable rows."""
    records = [json.loads(line) for line in open(path) if line.strip()]
    rows, rejected, seen, code_parts, run_parts = [], [], set(), {}, {}
    for index, original in enumerate(records):
        row = dict(original)
        task, part = row.get("task"), row.get("split")
        if part == "calibration":
            part = "validation"
        if task not in TASKS or part not in PARTS:
            raise ValueError(f"row {index}: unsupported task/split")
        code = row.get("code")
        if not isinstance(code, str) or not code.strip():
            raise ValueError(f"row {index}: evaluated source missing")
        key = normalized_hash(code)
        if row.get("code_hash") != key:
            raise ValueError(f"row {index}: normalized code hash disagrees")
        if key in code_parts and code_parts[key] != part:
            raise ValueError(f"program {key} leaks across splits")
        code_parts[key] = part
        run = row.get("representative_run_id")
        if not isinstance(run, str) or not run:
            raise ValueError(f"row {index}: representative_run_id missing")
        if run in run_parts and run_parts[run] != part:
            raise ValueError(f"run {run} leaks across splits")
        run_parts[run] = part
        identity = row.get("task_hash")
        evaluation = row.get("evaluation") or {}
        if not identity or not evaluation.get("recomputed"):
            raise ValueError(f"row {index}: fresh evaluation identity missing")
        sample_id = f"{task}:{key}"
        if sample_id in seen:
            raise ValueError(f"duplicate sample {sample_id}")
        seen.add(sample_id)
        row.update(split=part, sample_id=sample_id, representative_run_id=run)
        if not row.get("ok"):
            rejected.append({"sample_id": sample_id, "task": task, "split": part})
            continue
        score = row.get("score")
        cases = row.get("cases")
        if isinstance(score, bool) or not isinstance(score, (int, float)) \
                or not np.isfinite(score) or not isinstance(cases, dict):
            raise ValueError(f"row {index}: non-finite score or invalid cases")
        if not all(k in cases and np.isfinite(float(cases[k])) for k in BEHAVIOR_EXTRA_KEYS):
            raise ValueError(f"row {index}: six recomputed behavior descriptors required")
        rows.append(row)
    for task in TASKS:
        hashes = {r["task_hash"] for r in rows if r["task"] == task}
        if len(hashes) > 1:
            raise ValueError(f"{task}: mixed evaluator identities")
        if not all(any(r["task"] == task and r["split"] == p for r in rows) for p in PARTS):
            raise ValueError(f"{task}: an evaluation-success split is empty")
    return sorted(rows, key=lambda r: r["sample_id"]), rejected


def verify_frozen_corpus(path, manifest_path):
    """Require the entire frozen selection, including failed re-evaluations."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text())
    status = json.loads((manifest_path.parent / "status.json").read_text())
    if status.get("complete") is not True or status.get("manifest_sha256") != sha256(manifest_path) \
            or sha256(path) != status.get("result_files", {}).get("results.jsonl"):
        raise ValueError("corpus evaluation has not published a complete frozen result")
    selected_path = manifest_path.parent / "selected.jsonl"
    if sha256(selected_path) != manifest["files"]["selected.jsonl"]:
        raise ValueError("frozen corpus selection changed")
    selected = {r["row_id"]: r for r in map(json.loads, selected_path.read_text().splitlines())}
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len(records) != len(selected) or {r["row_id"] for r in records} != set(selected):
        raise ValueError("corpus evaluation is incomplete or contains duplicate/extra results")
    runs = {r["run_id"]: r for r in manifest["runs"]}
    for row in records:
        source = selected[row["row_id"]]
        for key in ("task", "task_hash", "code", "code_hash", "source_sha256", "split",
                    "memberships", "representative_run_id", "evaluation_id"):
            if row.get(key) != source.get(key):
                raise ValueError(f"evaluated row changed frozen source field {key}")
        evaluation = row.get("evaluation", {})
        if evaluation.get("evaluation_id") != manifest["evaluation_id"] \
                or evaluation.get("task_hash") != row["task_hash"]:
            raise ValueError("evaluation provenance disagrees with frozen corpus")
        if row.get("ok") and evaluation.get("reached") != "full":
            raise ValueError("successful corpus row did not reach full evaluation")
        if row["task_hash"] != manifest["tasks"][row["task"]]["task_hash"]:
            raise ValueError("task identity disagrees with frozen corpus")
        if hashlib.sha256(row["code"].encode()).hexdigest() != row["source_sha256"]:
            raise ValueError("evaluated source byte hash disagrees")
        for member in row["memberships"]:
            run = runs[member["run_id"]]
            if member["split"] != run["split"]:
                raise ValueError("source-run split changed")
        if runs[row["representative_run_id"]]["split"] != row["split"]:
            raise ValueError("representative run is outside the assigned split")
    return manifest


def indices(rows, part, task=None):
    return np.array([i for i, r in enumerate(rows)
                     if r["split"] == part and (task is None or r["task"] == task)], dtype=int)


def score_pairs(rows, subset, seed=0, cap=4000):
    """Legacy score-similarity objective, named as such rather than 'behavior'."""
    rng = np.random.default_rng(seed)
    positive, negative = [], []
    for task in TASKS:
        ids = np.array([i for i in subset if rows[i]["task"] == task], dtype=int)
        if len(ids) < 2:
            continue
        a, b = np.triu_indices(len(ids), 1)
        scores = np.array([rows[i]["score"] for i in ids])
        gaps = np.abs(scores[a] - scores[b])
        for condition, destination in ((gaps < .005, positive), (gaps > .1, negative)):
            choices = np.flatnonzero(condition)
            if len(choices) > cap:
                choices = rng.choice(choices, cap, replace=False)
            destination.extend((int(ids[a[j]]), int(ids[b[j]])) for j in choices)
    return positive, negative


def fit_candidate(rows, epochs=80, score_weight=.15, seed=0):
    train = indices(rows, "train")
    pos, neg = score_pairs(rows, train, seed)
    if len(train) < 32 or not pos or not neg:
        raise ValueError("training requires at least 32 programs and both pair classes")
    X = np.stack([featurize(r["code"], r["cases"]) for r in rows])
    Y = np.array([r["score"] for r in rows], dtype=np.float32)
    Xt, Yt = torch.tensor(X), torch.tensor(Y)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = Embedder(X.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    for _ in range(epochs):
        for pairs, is_positive in ((pos, True), (neg, False)):
            picked = rng.choice(len(pairs), min(2000, len(pairs)), replace=False)
            a = [pairs[i][0] for i in picked]
            b = [pairs[i][1] for i in picked]
            distance = 1 - (model.embed(Xt[a]) * model.embed(Xt[b])).sum(-1)
            contrast = ((distance ** 2).mean() if is_positive else
                        torch.clamp(.4 - distance, min=0).square().mean())
            score_ids = rng.choice(train, min(2000, len(train)), replace=False)
            loss = contrast + score_weight * (model.predict_score(Xt[score_ids]) - Yt[score_ids]).square().mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    model.eval()
    return model, {"n_train": len(train), "positive_pairs": len(pos), "negative_pairs": len(neg),
                   "epochs": epochs, "score_weight": score_weight, "seed": seed,
                   "optimizer": "Adam", "learning_rate": .001, "contrastive_margin": .4,
                   "positive_score_gap": .005, "negative_score_gap": .1,
                   "pair_cap_per_class_per_task": 4000, "batch_size_per_class": 2000,
                   "uses_calibration_in_loss": False, "uses_test_in_loss": False}


def corr(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if len(a) < 3 or a.std() < 1e-12 or b.std() < 1e-12:
        return None
    value = float(np.corrcoef(a, b)[0, 1])
    return value if np.isfinite(value) else None


def ranks(values):
    values = np.asarray(values)
    order = np.argsort(values, kind="stable")
    ranked = np.empty(len(values), dtype=float)
    cursor = 0
    while cursor < len(values):
        end = cursor + 1
        while end < len(values) and values[order[end]] == values[order[cursor]]:
            end += 1
        ranked[order[cursor:end]] = (cursor + end - 1) / 2
        cursor = end
    return ranked


def pair_distances(Z, pairs):
    return np.array([1 - Z[a] @ Z[b] for a, b in pairs], dtype=float)


def calibrated_threshold(Z, positive, negative):
    dp, dn = pair_distances(Z, positive), pair_distances(Z, negative)
    if not len(dp) or not len(dn):
        return None
    choices = np.unique(np.quantile(np.concatenate([dp, dn]), np.linspace(0, 1, 101)))
    return float(max(choices, key=lambda t: .5 * ((dp < t).mean() + (dn >= t).mean())))


def pair_accuracy(Z, positive, negative, threshold):
    if threshold is None or not positive or not negative:
        return None
    dp, dn = pair_distances(Z, positive), pair_distances(Z, negative)
    return float(.5 * ((dp < threshold).mean() + (dn >= threshold).mean()))


def fit_axis(Z, rows, task, embed_sha, version="review_v4"):
    """PCA/standardization on train; component selection on calibration only."""
    tr, ca = indices(rows, "train", task), indices(rows, "validation", task)
    mean = Z[tr].mean(0)
    centered = Z[tr] - mean
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    explained = singular ** 2
    explained = explained / max(float(explained.sum()), 1e-20)
    scores_cal = np.array([rows[i]["score"] for i in ca])
    candidates = []
    for j in range(min(6, len(vt))):
        projection = centered @ vt[j]
        scale = float(projection.std())
        if scale <= 1e-9:
            continue
        axis = BehaviorAxis(vt[j], mean, float(projection.mean()), scale,
                            component=j, version=version, embed_sha=embed_sha,
                            explained=float(explained[j]), n_fit=len(tr), fit_task=task)
        vals = np.array([axis.value(z) for z in Z[ca]])
        correlation = corr(vals, scores_cal)
        _, occupied, uniformity = bin_profile(vals)
        eligible = correlation is not None and abs(correlation) < .35 and explained[j] >= .05
        candidates.append((axis, {"component": j, "explained_train": float(explained[j]),
                                  "score_corr_calibration": correlation,
                                  "bins_calibration": occupied, "uniformity_calibration": uniformity,
                                  "eligible": bool(eligible)}))
    if not candidates:
        raise ValueError(f"{task}: degenerate latent cannot supply a behavior axis")
    eligible = [item for item in candidates if item[1]["eligible"]]
    if eligible:
        best_fill = max(item[1]["uniformity_calibration"] for item in eligible)
        selected = max((item for item in eligible if item[1]["uniformity_calibration"] >= best_fill - .04),
                       key=lambda item: item[1]["explained_train"])
    else:
        # Freeze a diagnostic axis even when unsuitable; never silently promote it.
        selected = min(candidates, key=lambda item: abs(item[1]["score_corr_calibration"])
                       if item[1]["score_corr_calibration"] is not None else float("inf"))
    axis, metadata = selected
    correlation = metadata["score_corr_calibration"]
    axis.corr_score = axis.corr_worst = correlation if correlation is not None else float("nan")
    axis.corr_worst_task = task
    covariance = np.cov(centered.T)
    t0 = max(4 * float(np.linalg.eigvalsh(covariance)[-1]), 1e-6)
    return axis, {**metadata, "n_train": len(tr), "n_calibration": len(ca),
                  "candidates": [item[1] for item in candidates],
                  "annealing_t0_train": t0, "train_mean": mean.tolist(),
                  "selection_used_test": False}


def cluster_ci(rows, ids, metric, repeats=300, seed=0):
    groups = sorted({rows[i]["representative_run_id"] for i in ids})
    if len(groups) < 3:
        return {"low": None, "high": None, "n_runs": len(groups), "reason": "fewer than 3 test runs"}
    by_run = {g: [i for i in ids if rows[i]["representative_run_id"] == g] for g in groups}
    rng, values = np.random.default_rng(seed), []
    for _ in range(repeats):
        selected = np.array([i for g in rng.choice(groups, len(groups), replace=True) for i in by_run[g]])
        value = metric(selected)
        if value is not None and np.isfinite(value):
            values.append(value)
    if not values:
        return {"low": None, "high": None, "n_runs": len(groups), "reason": "undefined statistic"}
    low, high = np.quantile(values, [.025, .975])
    return {"low": float(low), "high": float(high), "n_runs": len(groups),
            "method": "run-cluster percentile bootstrap", "repeats": repeats}


def model_metrics(model, rows, axes, seed=0):
    Z = embed_batch(model, [(r["code"], r["cases"]) for r in rows])
    X = np.stack([featurize(r["code"], r["cases"]) for r in rows])[:, :model.in_dim]
    with torch.no_grad():
        prediction = model.predict_score(torch.tensor(X)).numpy()
    scores = np.array([r["score"] for r in rows])
    result = {}
    for task in TASKS:
        ca, te = indices(rows, "validation", task), indices(rows, "test", task)
        pca, nca = score_pairs(rows, ca, seed)
        pte, nte = score_pairs(rows, te, seed)
        threshold = calibrated_threshold(Z, pca, nca)
        values = np.array([axes[task].value(z) for z in Z])
        _, bins, uniformity = bin_profile(values[te])
        rmse = lambda ix: float(np.sqrt(np.mean((prediction[ix] - scores[ix]) ** 2)))
        axis_corr = lambda ix: corr(values[ix], scores[ix])
        # Actual behavior-distance diagnostic; separate from score-pair labels.
        aa, bb = np.triu_indices(len(te), 1)
        rng = np.random.default_rng(seed)
        if len(aa) > 4000:
            choice = rng.choice(len(aa), 4000, replace=False)
            aa, bb = aa[choice], bb[choice]
        behavior = np.array([[rows[i]["cases"][k] for k in BEHAVIOR_EXTRA_KEYS] for i in te])
        bd = np.linalg.norm(behavior[aa] - behavior[bb], axis=1)
        ld = np.array([1 - Z[te[a]] @ Z[te[b]] for a, b in zip(aa, bb)])
        matched = np.abs(scores[te[aa]] - scores[te[bb]]) < .02
        result[task] = {
            "n_train": len(indices(rows, "train", task)), "n_calibration": len(ca), "n_test": len(te),
            "n_test_runs": len({rows[i]["representative_run_id"] for i in te}),
            "score_reconstruction_rmse": rmse(te), "score_reconstruction_rmse_ci95": cluster_ci(rows, te, rmse, seed=seed),
            "score_reconstruction_pearson": corr(prediction[te], scores[te]),
            "score_reconstruction_spearman": corr(ranks(prediction[te]), ranks(scores[te])),
            "score_pair_separation_accuracy": pair_accuracy(Z, pte, nte, threshold),
            "distance_threshold_calibration": threshold,
            "positive_test_pairs": len(pte), "negative_test_pairs": len(nte),
            "behavior_distance_spearman": corr(ranks(bd), ranks(ld)),
            "score_matched_behavior_distance_spearman": corr(ranks(bd[matched]), ranks(ld[matched])),
            "score_matched_behavior_pairs": int(matched.sum()),
            "axis_score_corr": axis_corr(te), "axis_score_corr_ci95": cluster_ci(rows, te, axis_corr, seed=seed),
            "axis_bins_occupied": bins, "axis_uniformity": uniformity,
        }
    return result


def simple_baselines(rows, seed=0, ridge_strength=1.0):
    """Fixed ridge reconstruction and untrained raw-feature distance baselines."""
    train = indices(rows, "train")
    X = np.stack([featurize(r["code"], r["cases"]) for r in rows]).astype(np.float64)
    Y = np.array([r["score"] for r in rows])
    mean, scale = X[train].mean(0), X[train].std(0)
    scale[scale < 1e-9] = 1.
    standardized = (X - mean) / scale
    intercept = float(Y[train].mean())
    A = standardized[train]
    coef = np.linalg.solve(A.T @ A + ridge_strength * np.eye(A.shape[1]), A.T @ (Y[train] - intercept))
    ridge_prediction = standardized @ coef + intercept
    raw_unit = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    means, ridge, raw_distance = {}, {}, {}
    for task in TASKS:
        tr, ca, te = (indices(rows, p, task) for p in PARTS)
        constant = float(Y[tr].mean())
        constant_rmse = lambda ix: float(np.sqrt(np.mean((constant - Y[ix]) ** 2)))
        ridge_rmse = lambda ix: float(np.sqrt(np.mean((ridge_prediction[ix] - Y[ix]) ** 2)))
        means[task] = {"train_mean": constant, "score_reconstruction_rmse": constant_rmse(te),
                       "score_reconstruction_rmse_ci95": cluster_ci(rows, te, constant_rmse, seed=seed)}
        ridge[task] = {"score_reconstruction_rmse": ridge_rmse(te),
                       "score_reconstruction_rmse_ci95": cluster_ci(rows, te, ridge_rmse, seed=seed),
                       "score_reconstruction_pearson": corr(ridge_prediction[te], Y[te])}
        pca, nca = score_pairs(rows, ca, seed)
        pte, nte = score_pairs(rows, te, seed)
        threshold = calibrated_threshold(raw_unit, pca, nca)
        raw_distance[task] = {"score_pair_separation_accuracy": pair_accuracy(raw_unit, pte, nte, threshold),
                              "distance_threshold_calibration": threshold,
                              "positive_test_pairs": len(pte), "negative_test_pairs": len(nte)}
    parameters = {"mean": mean, "scale": scale, "coef": coef,
                  "intercept": intercept, "ridge_strength": ridge_strength}
    return {"task_train_mean": means,
            "ridge_reconstruction": {"ridge_strength": ridge_strength, "fit": "pooled train only", "tasks": ridge},
            "raw_feature_cosine": raw_distance}, parameters


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--corpus-manifest", required=True)
    ap.add_argument("--out", type=Path, default=ROOT.parent / "sciel-data/review/models")
    ap.add_argument("--baseline", type=Path, default=ROOT.parent / "sciel-artifacts/code_embed_v3.pt")
    ap.add_argument("--axis-dir", type=Path,
        help="Baseline *_v3.npz directory; default: behavior_axis beside --baseline")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    out = Path(args.out)
    if out.exists() and any(out.iterdir()):
        raise ValueError("model output directory is not empty; choose a new --out to preserve frozen artifacts")
    corpus_manifest = verify_frozen_corpus(args.corpus, args.corpus_manifest)
    rows, rejected = read_corpus(args.corpus)
    out.mkdir(parents=True, exist_ok=True)
    provenance = {"corpus_sha256": sha256(args.corpus), "corpus_manifest_sha256": sha256(args.corpus_manifest),
                  "script_sha256": sha256(__file__), "baseline_sha256": sha256(args.baseline),
                  "feature_source_sha256": sha256(HERE.parent / "src/sciel/components/policy/code_embed.py"),
                  "axis_profile_source_sha256": sha256(HERE / "fit_behavior_axis.py"),
                  "runtime": {"python": sys.version, "numpy": np.__version__, "torch": torch.__version__},
                  "split_protocol": "source-run-grouped/program-disjoint/1", "seed": args.seed}
    split = [{k: r[k] for k in ("sample_id", "task", "task_hash", "code_hash", "split", "representative_run_id")}
             for r in rows]
    write_json(out / "splits.json", split)
    provenance["splits_sha256"] = sha256(out / "splits.json")
    (out / "samples.jsonl").write_text("".join(json.dumps(r, allow_nan=False) + "\n" for r in rows))
    provenance["samples_sha256"] = sha256(out / "samples.jsonl")
    model, training = fit_candidate(rows, epochs=args.epochs, seed=args.seed)
    model_path = out / "code_embed_candidate_v4.pt"
    torch.save({"state": model.state_dict(), "in_dim": model.in_dim,
                "behavior_keys": BEHAVIOR_KEYS, "behavior_extra_keys": BEHAVIOR_EXTRA_KEYS,
                "dim_hash": DIM_HASH, "provenance": provenance, "training": training}, model_path)
    Z = embed_batch(model, [(r["code"], r["cases"]) for r in rows])
    axes, axis_info = {}, {}
    (out / "axes").mkdir(exist_ok=True)
    for task in TASKS:
        axis, info = fit_axis(Z, rows, task, file_sha(model_path))
        path = out / "axes" / f"{task.replace('/', '_')}_v4.npz"
        axis.save(path)
        axes[task], axis_info[task] = axis, {**info, "path": str(path.resolve()), "sha256": sha256(path)}
    baseline = Embedder.load(args.baseline)
    axis_dir = args.axis_dir or Path(args.baseline).parent / "behavior_axis"
    baseline_paths = {task: axis_dir / f"{task.replace('/', '_')}_v3.npz" for task in TASKS}
    baseline_axes = {task: BehaviorAxis.load(path) for task, path in baseline_paths.items()}
    for task, axis in baseline_axes.items():
        if axis.embed_sha != file_sha(args.baseline):
            raise ValueError(f"{task}: baseline axis/encoder hash mismatch")
    metrics = {"candidate_v4": model_metrics(model, rows, axes, args.seed),
               "baseline_v3_retrospective": model_metrics(baseline, rows, baseline_axes, args.seed),
               "v3_exposure": {"status": "unknown_prior_exposure", "holdout": False,
                               "reason": "shipped v3 lacks source IDs or training split hashes; historical programs may have been exposed"}}
    metrics["simple_baselines"], ridge_parameters = simple_baselines(rows, args.seed)
    np.savez(out / "ridge_reconstruction.npz", **ridge_parameters)
    limitations = ["Offline representation evaluation does not establish a search-quality improvement.",
                   "The input contains evaluated score metrics: score-head evaluation is reconstruction, not prediction before evaluation.",
                   "Score-pair separation labels measure quality similarity, not independent behavioral classes.",
                   "Run-cluster confidence intervals are exploratory for few heldout runs.",
                   "Historical-score-stratified sampling defines a benchmark distribution; it is not a random sample of all generated programs.",
                   "Representation metrics condition on successful fresh evaluation; failed programs remain counted in the manifest.",
                   "v3 retrospective metrics cannot be interpreted as independent holdout generalization."]
    for task, info in axis_info.items():
        if info["n_train"] < 200 or info["n_calibration"] < 50 or len(indices(rows, "test", task)) < 50:
            limitations.append(f"{task}: below planned 200 train / 50 calibration / 50 test programs; evaluation is exploratory.")
    manifest = {"format": "review-model-pipeline/1", **provenance, "training": training,
                "corpus_manifest_path": str(Path(args.corpus_manifest).resolve()),
                "source_runs": corpus_manifest["runs"],
                "trajectory_anchors": corpus_manifest["trajectory_anchors"],
                "baseline_axes": {t: {"path": str(path.resolve()), "sha256": sha256(path)}
                                  for t, path in baseline_paths.items()},
                "model_path": str(model_path.resolve()), "model_sha256": sha256(model_path),
                "ridge_parameters_sha256": sha256(out / "ridge_reconstruction.npz"),
                "axes": axis_info, "evaluation_failures": rejected,
                "sample_counts": {task: {p: len(indices(rows, p, task)) for p in PARTS} for task in TASKS},
                "status": "offline_candidate_only", "default_promoted": False,
                "limitations": limitations}
    write_json(out / "metrics.json", metrics)
    manifest["metrics_sha256"] = sha256(out / "metrics.json")
    write_json(out / "candidate.json", {"embedder": str(model_path.resolve()),
                                        "axes": {task: info["path"] for task, info in axis_info.items()}})
    manifest["candidate_spec_sha256"] = sha256(out / "candidate.json")
    write_json(out / "manifest.json", manifest)
    print(json.dumps({"output": str(out), "counts": manifest["sample_counts"],
                      "calibration_eligible": {t: i["eligible"] for t, i in axis_info.items()},
                      "default_promoted": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
