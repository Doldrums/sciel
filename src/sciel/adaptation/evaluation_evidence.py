"""Portable native oracle evidence. Locations never supply evaluator authority.

Checkpoint bytes and the exact committed journal prefix bind the public launch,
the unique evaluation decision and its saved-record/sidecar join. This remains
unsigned host evidence, not scientific or Docker qualification.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from ..eval import oracle_evidence as O
from ..methods.models import content_hash as native_hash
from .storage import read_json
from .types import sha256_bytes

from .evaluation_context import (
    DIRECTORY, SCHEMA, ROW, RunContext, Contexts, context, native_context)

def _files(root):
    return {k: v for k, v in O.files(root).items() if k != "manifest.json"}


def retained_context(manifest_path, manifest):
    desc = manifest.get("evaluation_evidence")
    if desc is None:
        return None
    O.need(
        type(desc) is dict
        and set(desc) == {"schema", "directory", "manifest_sha256"}
        and desc["schema"] == SCHEMA
        and desc["directory"] == DIRECTORY,
        "invalid retained oracle descriptor",
    )
    root = O.ordinary(Path(manifest_path).parent / DIRECTORY)
    raw = O.ordinary(root / "manifest.json").read_bytes()
    O.need(
        sha256_bytes(raw) == desc["manifest_sha256"],
        "retained oracle manifest hash differs",
    )
    value = json.loads(raw)
    O.need(
        type(value) is dict
        and set(value) == {"schema", "runs", "files"}
        and value["schema"] == SCHEMA
        and value["files"] == _files(root)
        and type(value["runs"]) is list
        and len(set(value["runs"])) == len(value["runs"]),
        "retained oracle inventory differs",
    )
    runs = []
    for ident in value["runs"]:
        O.task_module().panels.require_hash(ident)
        run = RunContext(root / "runs" / ident)
        O.need(run.checkpoint_sha256 == ident, "retained oracle checkpoint differs")
        runs.append(run)
    return Contexts(runs)


def _copy(source, destination):
    O.ordinary(source)
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with open(source, "rb") as reader, open(destination, "xb") as writer:
        shutil.copyfileobj(reader, writer)
        writer.flush()
        os.fchmod(writer.fileno(), 0o600)
        os.fsync(writer.fileno())


def retain(rows, sources, output):
    sources = context(sources)
    required = {}
    for row in rows:
        run = sources.for_row(row) if sources is not None else None
        O.need(
            run is not None or row["provenance"].get("oracle_evaluation") is None,
            "oracle evidence unavailable for retention",
        )
        if run is not None:
            required.setdefault(run.checkpoint_sha256, (run, set()))[1].update(
                (
                    row["evaluation_receipt"]["evaluation_id"],
                    row["provenance"]["parent_evaluation_receipt"]["evaluation_id"],
                )
            )
    if not required:
        return None
    output = Path(output)
    with tempfile.TemporaryDirectory(
        prefix=".oracle-evidence-", dir=output
    ) as temporary:
        root = Path(temporary) / DIRECTORY
        root.mkdir(mode=0o700)
        for ident, (run, evaluations) in required.items():
            dest = root / "runs" / ident
            dest.mkdir(parents=True, mode=0o700)
            _copy(run.checkpoint_path, dest / "checkpoint.json")
            with open(dest / "sciel_events.jsonl", "xb") as stream:
                stream.write(run.prefix)
                stream.flush()
                os.fchmod(stream.fileno(), 0o600)
                os.fsync(stream.fileno())
            for eid in sorted(evaluations):
                decision = run.decision(eid)
                original = run.root / O.relative_evaluation(eid)
                for path in sorted(original.rglob("*")):
                    O.ordinary(path)
                    if path.is_file():
                        _copy(
                            path,
                            dest
                            / O.relative_evaluation(eid)
                            / path.relative_to(original),
                        )
                index_name = (
                    "evaluation-" + hashlib.sha256(eid.encode()).hexdigest() + ".json"
                )
                index_root = run.root / "research_evidence/evaluations"
                index = read_json(index_root / index_name)
                for name in (index_name, index["sha256"] + ".json"):
                    target = dest / "research_evidence/evaluations" / name
                    if not target.exists():
                        _copy(index_root / name, target)
                O.need(
                    decision.get("oracle_evidence_sha256"),
                    "missing committed oracle sidecar",
                )
        for directory in (root, *[p for p in root.rglob("*") if p.is_dir()]):
            directory.chmod(0o700)
        value = dict(schema=SCHEMA, runs=sorted(required), files=_files(root))
        O.write(root / "manifest.json", value)
        descriptor = dict(
            schema=SCHEMA,
            directory=DIRECTORY,
            manifest_sha256=sha256_bytes((root / "manifest.json").read_bytes()),
        )
        copied = retained_context(
            root.parent / "manifest.json", dict(evaluation_evidence=descriptor)
        )
        for row in rows:
            copied.for_row(row)
        from .retained_ancestry import _sync_directories

        _sync_directories(root)
        destination = output / DIRECTORY
        if destination.exists():
            O.need(
                (destination / "manifest.json").read_bytes()
                == (root / "manifest.json").read_bytes()
                and _files(destination) == value["files"],
                "existing retained oracle evidence differs",
            )
        else:
            root.rename(destination)
        return descriptor


def copy_retained(manifest_path, manifest, output):
    if retained_context(manifest_path, manifest) is None:
        return
    source = Path(manifest_path).parent / DIRECTORY
    destination = Path(output) / DIRECTORY
    O.need(not destination.exists(), "retained oracle target already exists")
    destination.mkdir(mode=0o700)
    for path in sorted(source.rglob("*")):
        O.ordinary(path)
        if path.is_file():
            _copy(path, destination / path.relative_to(source))
    for directory in (destination, *[p for p in destination.rglob("*") if p.is_dir()]):
        directory.chmod(0o700)
    retained_context(Path(output) / "manifest.json", manifest)
    from .retained_ancestry import _sync_directories

    _sync_directories(destination)
