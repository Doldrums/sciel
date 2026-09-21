"""Train a shared three-head trunk for promise, eviction and niche labels.

One Set-Transformer over frozen encoder embeddings, three heads trained jointly on
whatever labels each population set carries:

  - promise : per-set-normalized subtree_max derived from recorded lineage.
              Training this head does not enable a selection policy in the engine.
  - niche   : nearest centroid in one frozen VQ reference fitted to training
              embeddings. Class ids share a meaning across islands and runs;
              they are not the ids of each island's growing live VQ map.
  - eviction: 1 keep / 0 evict, read from `decision policy=eviction` events (only
              runs launched with log_evictions=True carry these; islands without
              an observed decision contribute no eviction loss). These are
              trajectory-level ever-evicted labels, not per-decision choice sets.

For held-out evaluation, reuse the training reference via build_sets(niche_book=...);
never fit a new reference on held-out runs. The CLI writes it to <out.pt>.niche.json.
Usage:
  python scripts/train_three_head.py <embedder.pt> <out.pt> <run_dir> [<run_dir> ...]
"""

import hashlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, "src")

from lineage_labels import labels_for                                          # noqa: E402
from journal_data import DirtyRun, completed_events                            # noqa: E402
from sciel.components.policy.code_embed import (                               # noqa: E402
    Embedder, VQBook, embed_batch, featurize)
from sciel.components.policy.three_head import ThreeHeadModel                  # noqa: E402

NICHE_TARGET_K, NICHE_KMAX = 6, 12


def _load_embedder(path):
    # Embedder.load records in_dim, which embed_batch needs in order to feed an
    # older model exactly the feature block it was trained on
    return Embedder.load(path)


def _embed(model, items):
    """Delegates to embed_batch. This used to stack features and call the model
    directly, which bypassed the feature-width rule and broke the moment the
    featurizer grew a tail block. One place owns that rule."""
    return embed_batch(model, [(it.get("code", ""), it.get("cases", {}))
                               for it in items])


def _evictions_by_island(journal_path, records, *, events=None):
    if events is None:
        events, _ = completed_events(journal_path)
    program_islands = {record['id']: record['island'] for record in records}
    evicted = {}
    for event in events:
        if event.get('event') != 'decision' or event.get('policy') != 'eviction':
            continue
        victim = event.get('victim_id')
        if victim is None:
            continue
        if type(victim) is not str or not victim:
            raise DirtyRun(f'{journal_path}: invalid eviction victim identity')
        known_island = program_islands.get(victim)
        # Old logs can omit island; only a selected victim record can supply it.
        island = event.get('island', known_island)
        if island is None:
            continue
        if (type(island) is not int or island < 0
                or known_island is not None and island != known_island):
            raise DirtyRun(f'{journal_path}: eviction island differs from selected victim')
        evicted.setdefault(island, set()).add(victim)
    return evicted


def fit_niche_book(sets):
    """Fit one corpus reference, invariant to run/island/member ordering.

    Only training embeddings belong here. Canonical unique rows make the online
    VQ growth deterministic; labels are assigned against the final frozen book,
    so subsequent populations cannot change the meaning of an earlier label.
    """
    vectors = np.concatenate(sets).astype(np.float32)
    if vectors.ndim != 2 or not vectors.shape[1] or not np.isfinite(vectors).all():
        raise ValueError("niche reference requires finite embedding vectors")
    book = VQBook(target_k=NICHE_TARGET_K, k_max=NICHE_KMAX)
    for vector in np.unique(vectors, axis=0):
        book.assign(vector)
    return book


def build_sets(embedder, run_dirs, min_set=3, *, niche_book=None, return_niche_book=False):
    """Build labels using one frozen reference; supplied books are never updated."""
    sets, labels = [], []
    for d in run_dirs:
        jp = f"{d}/sciel_events.jsonl"
        try:
            trajectory = completed_events(jp)
            recs = labels_for(jp, with_code=True, trajectory=trajectory)
            evicted = _evictions_by_island(jp, recs, events=trajectory[0])
        except (OSError, DirtyRun) as exc:
            print(f"skip {jp}: {exc}", file=sys.stderr)
            continue
        if not recs:
            continue
        by_isl = {}
        for r in recs:
            by_isl.setdefault(r["island"], []).append(r)
        for island, items in by_isl.items():
            if len(items) < min_set:
                continue
            Z = _embed(embedder, items)
            y = np.array([it["subtree_max"] for it in items], dtype=float)
            lo, hi = y.min(), y.max()
            promise = (y - lo) / (hi - lo) if hi > lo else np.zeros_like(y)
            lab = {"promise": promise}
            if island in evicted:
                lab["eviction"] = np.array(
                    [0.0 if it.get("id") in evicted[island] else 1.0 for it in items], float)
            sets.append(Z); labels.append(lab)
    if sets:
        if niche_book is None:
            niche_book = fit_niche_book(sets)
        if not niche_book.centroids:
            raise ValueError("niche reference must contain training centroids")
        for Z, lab in zip(sets, labels):
            lab["niche"] = np.array([niche_book.assign_fixed(z) for z in Z], dtype=np.int64)
    return (sets, labels, niche_book) if return_niche_book else (sets, labels)


def main(embedder_path, out, run_dirs):
    embedder = _load_embedder(embedder_path)
    sets, labels, niche_book = build_sets(embedder, run_dirs, return_niche_book=True)
    if not sets:
        raise SystemExit("no training sets found in the given run dirs")
    in_dim = sets[0].shape[1]
    n_evict = sum("eviction" in lab for lab in labels)
    print(f"sets={len(sets)} in_dim={in_dim} eviction-labeled sets={n_evict}")
    model = ThreeHeadModel(in_dim=in_dim, n_niche=NICHE_KMAX)
    model.fit(sets, labels, epochs=200)
    model.save(out)
    with open(embedder_path, "rb") as source:
        embedder_sha256 = hashlib.sha256(source.read()).hexdigest()
    with open(str(out) + ".niche.json", "w") as target:
        json.dump({"schema": "sciel-three-head-niche-reference/1",
                   "policy": "canonical_training_vq_frozen_nearest",
                   "embedder_sha256": embedder_sha256, "book": niche_book.state()},
                  target, sort_keys=True, allow_nan=False)
    print(f"saved -> {out}")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
