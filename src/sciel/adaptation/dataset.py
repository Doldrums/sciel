"""Content-bound supervised examples from committed, train-only evaluations.

Hashes establish artifact integrity, not truth of an arbitrary user-written
receipt. Callers must acquire episodes through the trusted native exporter or
host evaluator; model-authored success claims are never evaluator receipts.
"""
from __future__ import annotations

from collections import defaultdict
import json
import math
import os
from pathlib import Path
import random

from .storage import read_json, write_json
from .types import InferenceSpec, canonical_bytes, content_hash, require_hash, require_identifier, sha256_bytes
from .target_policy import DATASET_V2, EPISODE_V2, POLICY, manifest_policy, policy, verify_target

SCHEMA = "sciel-lora-dataset/1"
EPISODE_SCHEMA = "sciel-lora-episode/1"


def task_receipt(receipt: dict, *, evaluation_context=None) -> dict:
    """Native EvalResult serializes artifact values as JSON strings."""
    if not isinstance(receipt, dict):
        raise ValueError("evaluation receipt must be a mapping")
    if receipt.get("schema") == "sciel-saved-evaluation/1":
        if receipt.get("reached") not in ("full", "confirm"):
            raise ValueError("paired receipt requires full evaluation")
        nested = receipt.get("result", {}).get("artifacts", {}).get("evaluation_receipt", {})
    else:
        nested = receipt
    if isinstance(nested, str):
        if len(nested.encode()) > 1024 * 1024:
            raise ValueError("nested evaluation receipt is too large")
        def unique(items):
            result = {}
            for key, value in items:
                if key in result:
                    raise ValueError("duplicate key in evaluation receipt")
                result[key] = value
            return result
        try:
            nested = json.loads(nested, object_pairs_hook=unique)
        except (ValueError, RecursionError) as exc:
            raise ValueError("malformed nested evaluation receipt") from exc
    if (not isinstance(nested, dict) or nested.get("schema") != "sciel-lora-evaluation/1"
            or nested.get("split") != "train"
            or nested.get("source_sha256") != receipt.get("source_sha256")
            or nested.get("result", {}).get("score") != receipt.get("result", {}).get("score")
            or nested.get("result", {}).get("ok") is not True):
        raise ValueError("paired receipt is not a source-bound train evaluation")
    canonical_bytes(nested)
    require_hash(nested.get("instance_set_sha256"), "instance set hash")
    require_hash(nested.get("evaluator_version"), "evaluator version")
    if evaluation_context is not None:
        from .evaluation_evidence import RunContext
        if not isinstance(evaluation_context, RunContext):
            raise ValueError('task receipt requires a reconstructed native run context')
        expected = evaluation_context.resolve(receipt)['envelope']
        if canonical_bytes(nested) != canonical_bytes(expected):
            raise ValueError('oracle envelope differs from trusted retained evaluation')
    else:
        from ..eval.oracle_evidence import known_evaluator
        if (known_evaluator(receipt) or 'oracle_evaluation_schema' in nested or 'oracle_evaluation_sha256' in nested
                or 'oracle_evaluation' in receipt.get('result', {}).get('artifacts', {})):
            raise ValueError('oracle evaluation requires a trusted native evidence context')
    return nested


def verify_episode(row: dict, *, require_improvement: bool = True, ancestry_sources=None, evaluation_context=None) -> dict:
    if not isinstance(row, dict) or row.get("schema") not in (EPISODE_SCHEMA, EPISODE_V2):
        raise ValueError("unsupported episode schema")
    if row.get("committed") is not True or row.get("split") != "train":
        raise ValueError("only committed train episodes are eligible")
    require_identifier(row.get("owner"), "producer owner")
    require_identifier(row.get("episode_id"), "episode_id")
    if any(not isinstance(row.get(k), str) or not row[k].strip() for k in ("prompt", "target")):
        raise ValueError("empty prompt or target")
    source_hash = sha256_bytes(verify_target(row).encode("utf-8"))
    if row.get("source_sha256") != source_hash:
        raise ValueError("target does not match evaluated source")
    receipt = row.get("evaluation_receipt")
    if not isinstance(receipt, dict) or row.get("receipt_sha256") != content_hash(receipt):
        raise ValueError("evaluation receipt hash mismatch")
    if receipt.get("source_sha256") != source_hash:
        raise ValueError("receipt belongs to a different candidate source")
    if receipt.get("schema") == "sciel-saved-evaluation/1":
        result = receipt.get("result", {})
        if receipt.get("reached") not in ("full", "confirm"):
            raise ValueError("smoke-only result is not a supervised success")
        # Split is set by the trusted host exporter, never inferred from score.
        if row.get("provenance", {}).get("evaluation_split") != "train":
            raise ValueError("native evaluation split is not explicitly train")
    elif receipt.get("schema") == "sciel-lora-evaluation/1":
        if receipt.get("split") != "train":
            raise ValueError("evaluation is outside train split")
        result = receipt.get("result", {})
    else:
        raise ValueError("unrecognized host evaluation receipt")
    score, parent = result.get("score"), row.get("parent_score")
    if result.get("ok") is not True:
        raise ValueError("failed evaluation is not a supervised success")
    if any(type(v) not in (int, float) or not math.isfinite(v) for v in (score, parent)):
        raise ValueError("scores must be finite numbers")
    if require_improvement and score <= parent:
        raise ValueError("candidate did not improve the paired parent")
    provenance = row.get("provenance")
    if not isinstance(provenance, dict) or not provenance.get("run_id"):
        raise ValueError("missing producer provenance")
    spec = InferenceSpec.from_dict(provenance.get("inference_spec"))
    if spec.owner != row["owner"]:
        raise ValueError("inference owner differs from producer")
    parent_hash = require_hash(provenance.get("parent_source_sha256"), "parent source hash")
    parent_receipt = provenance.get("parent_evaluation_receipt")
    if (not isinstance(parent_receipt, dict) or parent_receipt.get("source_sha256") != parent_hash
            or parent_receipt.get("result", {}).get("ok") is not True
            or parent_receipt.get("result", {}).get("score") != parent):
        raise ValueError("parent score is not bound to a successful parent evaluation")
    from .evaluation_evidence import context as evaluation_context_type
    evaluation_context = evaluation_context_type(evaluation_context)
    evaluation_context = evaluation_context.for_row(row) if evaluation_context is not None else None
    if provenance.get('oracle_evaluation') is not None and evaluation_context is None:
        raise ValueError('oracle episode lost its native evaluation evidence')
    if task_receipt(parent_receipt, evaluation_context=evaluation_context)["instance_set_sha256"] != task_receipt(receipt, evaluation_context=evaluation_context)["instance_set_sha256"]:
        raise ValueError("parent and child were evaluated on different instance sets")
    if task_receipt(parent_receipt, evaluation_context=evaluation_context)["evaluator_version"] != task_receipt(receipt, evaluation_context=evaluation_context)["evaluator_version"]:
        raise ValueError("parent and child used different evaluator versions")
    if receipt.get("task_hash") != parent_receipt.get("task_hash"):
        raise ValueError("parent and child task identities differ")
    visible = row.get("visible_episode_ids")
    if not isinstance(visible, list) or any(not isinstance(v, str) for v in visible):
        raise ValueError("visible episode identities must be explicit")
    from .ancestry import validate_episode_resolution
    validate_episode_resolution(row, ancestry_sources=ancestry_sources)
    canonical_bytes(row)
    return row


def make_episode(*, episode_id: str, owner: str, prompt: str, target: str,
                 parent_score: float, evaluation_receipt: dict, provenance: dict,
                 visible_episode_ids: list[str], committed: bool = True,
                 split: str = "train", ancestry_sources=None, target_binding=None, target_evidence=None,
                 evaluation_context=None) -> dict:
    row = dict(schema=EPISODE_SCHEMA, episode_id=episode_id, owner=owner,
        prompt=prompt, target=target, source_sha256=sha256_bytes(target.encode("utf-8")),
        receipt_sha256=content_hash(evaluation_receipt), evaluation_receipt=evaluation_receipt,
        parent_score=parent_score, provenance=provenance,
        visible_episode_ids=visible_episode_ids, committed=committed, split=split)
    if target_binding is not None or target_evidence is not None:
        row.update(schema=EPISODE_V2, target_binding=target_binding, target_evidence=target_evidence,
            evaluated_source=target_binding["evaluated_source"], source_sha256=target_binding["source_sha256"],
            target_sha256=target_binding["target_sha256"])
    return verify_episode(row, ancestry_sources=ancestry_sources, evaluation_context=evaluation_context)


def _deduplicate(rows: list[dict]) -> list[dict]:
    seen, out = set(), []
    for row in rows:
        # Repeating an identical learned target must not manufacture diversity.
        key = row["source_sha256"]
        if key not in seen:
            seen.add(key)
            out.append(row)
    return out


def build_dataset(episodes: list[dict], output_dir: str | Path, *, owner: str,
                  mode: str = "local", max_examples: int = 128, seed: int = 0,
                  cutoff: str = "offline-frozen", minimum_examples: int = 1, ancestry_sources=None,
                  evaluation_context=None, corpus_policy=None) -> dict:
    from . import corpus_policy as C
    if corpus_policy is not None:
        C.validate(corpus_policy)
        if minimum_examples != 2 or max_examples != 2:
            raise ValueError("small-data selection requires exact two-example limits")
    require_identifier(owner, "recipient owner")
    if mode not in ("local", "mixed"):
        raise ValueError("mode must be local or mixed")
    if type(max_examples) is not int or max_examples < 1 or minimum_examples < 1:
        raise ValueError("example limits must be positive")
    valid, exclusions = [], []
    from .adapter_ancestry import context
    ancestry_sources = context(ancestry_sources)
    identities = set()
    for row in episodes:
        try:
            verify_episode(row, ancestry_sources=ancestry_sources, evaluation_context=evaluation_context)
            if row["episode_id"] in identities:
                raise ValueError("duplicate episode identity")
            identities.add(row["episode_id"])
            valid.append(row)
        except (ValueError, TypeError) as exc:
            exclusions.append({"episode_id": row.get("episode_id") if isinstance(row, dict) else None,
                               "reason": str(exc)})
    rng = random.Random(seed)
    policies = {policy(row) for row in valid}
    if len(policies) > 1:
        raise ValueError("a dataset cannot mix legacy and native target policies")
    rng.shuffle(valid)
    own = _deduplicate([r for r in valid if r["owner"] == owner])
    own_hashes = {r["source_sha256"] for r in own}
    donors = _deduplicate([r for r in valid if r["owner"] != owner and
                          r["source_sha256"] not in own_hashes])
    if not own:
        raise ValueError("no verified local examples; retain the previous adapter")
    if mode == "local":
        selected = own[:max_examples]
    else:
        groups = defaultdict(list)
        for row in donors:
            groups[row["owner"]].append(row)
        balanced = []
        while any(groups.values()):
            for donor in sorted(groups):
                if groups[donor]:
                    balanced.append(groups[donor].pop())
        count = min(len(own), len(balanced), max_examples // 2)
        if not count:
            raise ValueError("mixed training requires distinct local and donor examples")
        selected = own[:count] + balanced[:count]
    if len(selected) < minimum_examples:
        raise ValueError("insufficient distinct verified examples; retain the previous adapter")
    rng.shuffle(selected)
    rows = []
    for source in selected:
        C.check_ancestry(source, corpus_policy)
        row = {**source, "recipient_owner": owner,
               "origin": "local" if source["owner"] == owner else "donor"}
        row["record_sha256"] = content_hash(row)
        rows.append(row)
    if corpus_policy is not None:
        C.check_rows(rows, owner=owner, mode=mode)
    data = b"".join(canonical_bytes(r) + b"\n" for r in rows)
    manifest = dict(schema=SCHEMA, split="train", owner=owner, mode=mode,
        dataset_sha256=sha256_bytes(data), example_count=len(rows), cutoff=cutoff, seed=seed,
        candidates_seen=len(episodes), eligible_count=len(valid), exclusions=exclusions,
        minimum_examples=minimum_examples, selection="deduplicated; donor-balanced by producer",
        target_token_balance="not measured here; trainer measures and budgets loss tokens",
        require_improvement=True)
    if corpus_policy is not None:
        manifest["corpus_policy"] = corpus_policy
    if policies == {POLICY}:
        manifest.update(schema=DATASET_V2, target_policy=POLICY)
    out = Path(output_dir)
    # Corpus publication must never replace a previous dataset on rerun.
    out.mkdir(parents=True, exist_ok=True)
    from .evaluation_evidence import retain as retain_evaluations
    evaluated = retain_evaluations(rows, evaluation_context, out)
    if evaluated is not None:
        manifest['evaluation_evidence'] = evaluated
    if manifest["schema"] == DATASET_V2:
        from .retained_ancestry import retain
        evidence = retain(rows, ancestry_sources, out, evaluation_context=evaluation_context)
        if evidence is not None:
            manifest["ancestry_evidence"] = evidence
    dataset_path = out / "train.jsonl"
    if dataset_path.exists():
        if dataset_path.read_bytes() != data:
            raise ValueError("dataset output already contains different data")
    else:
        with dataset_path.open("xb") as stream:
            stream.write(data)
            if manifest["schema"] == DATASET_V2:
                stream.flush()
                os.fsync(stream.fileno())
    write_json(out / "manifest.json", manifest, immutable=True)
    verify_dataset(dataset_path, out / "manifest.json", ancestry_sources=ancestry_sources)
    return manifest


def verify_dataset(dataset_path: str | Path, manifest_path: str | Path, *, ancestry_sources=None) -> tuple[list[dict], dict]:
    from .adapter_ancestry import context
    manifest = read_json(manifest_path)
    target_policy = manifest_policy(manifest)
    if manifest.get("split") != "train":
        raise ValueError("training requires a train-only dataset manifest")
    from .retained_ancestry import retained_context
    retained = retained_context(manifest_path, manifest)
    ancestry_sources = retained if retained is not None else context(ancestry_sources)
    from .evaluation_evidence import retained_context as evaluation_context_loader
    evaluation_context = evaluation_context_loader(manifest_path, manifest)
    require_identifier(manifest.get("owner"), "recipient owner")
    if manifest.get("mode") not in ("local", "mixed") or manifest.get("require_improvement") is not True:
        raise ValueError("unsupported dataset eligibility policy")
    data = Path(dataset_path).read_bytes()
    if sha256_bytes(data) != manifest.get("dataset_sha256"):
        raise ValueError("dataset content hash mismatch")
    rows = [json.loads(line) for line in data.splitlines() if line.strip()]
    if (manifest["schema"] == DATASET_V2 and retained is None
            and any(node.get("kind") == "adapter_revision" for row in rows
                for node in row.get("provenance", {}).get("ancestry_resolution", {}).get("auxiliary_bindings", []))):
        raise ValueError("native dataset lost its retained adapter ancestry evidence")
    if (not rows or len(rows) != manifest.get("example_count")
            or len(rows) < manifest.get("minimum_examples", 1)):
        raise ValueError("dataset example count mismatch")
    from . import corpus_policy as C
    corpus_policy = C.selected(manifest)
    if corpus_policy is not None:
        if manifest.get("minimum_examples") != 2 or target_policy != POLICY:
            raise ValueError("small-data manifest lost its exact native policy")
        C.check_rows(rows, owner=manifest["owner"], mode=manifest["mode"])
    seen = set()
    origins = set()
    for row in rows:
        if policy(row) != target_policy:
            raise ValueError("episode target policy differs from dataset manifest")
        verify_episode(row, ancestry_sources=ancestry_sources, evaluation_context=evaluation_context)
        C.check_ancestry(row, corpus_policy)
        body = {k: v for k, v in row.items() if k != "record_sha256"}
        if row.get("record_sha256") != content_hash(body):
            raise ValueError("dataset record hash mismatch")
        expected = "local" if row["owner"] == manifest["owner"] else "donor"
        if row.get("origin") != expected or row.get("recipient_owner") != manifest["owner"]:
            raise ValueError("dataset owner/origin mismatch")
        if manifest["mode"] == "local" and expected != "local":
            raise ValueError("donor example in local dataset")
        if row["source_sha256"] in seen:
            raise ValueError("duplicate target in dataset")
        seen.add(row["source_sha256"])
        origins.add(expected)
    if manifest["mode"] == "mixed" and origins != {"local", "donor"}:
        raise ValueError("mixed dataset lacks an origin")
    return rows, manifest
