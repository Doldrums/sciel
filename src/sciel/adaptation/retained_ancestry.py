"""Portable, content-bound copies of the external evidence required by typed adapters.

These are locations for the existing recursive verifier, not a new trust source.
Only revisions/corpora named by verified auxiliary nodes are retained.
"""
from __future__ import annotations

from pathlib import Path
import os
import shutil
import tempfile

from .adapter_ancestry import context
from .storage import read_json, write_json
from .types import content_hash, require_hash, sha256_bytes

DIRECTORY = "ancestry_evidence"
SCHEMA = "sciel-retained-ancestry/1"


def _ordinary(path):
    path = Path(path).absolute()
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("retained ancestry may not traverse symlinks")
    return path


def _files(root):
    root = _ordinary(root)
    result = {}
    for path in sorted(root.rglob("*")):
        _ordinary(path)
        if path.is_file() and path != root / "manifest.json":
            result[path.relative_to(root).as_posix()] = sha256_bytes(path.read_bytes())
    return result


def _copy_tree(source, target):
    source = _ordinary(source)
    for path in sorted(source.rglob("*")):
        _ordinary(path)
        relative = path.relative_to(source)
        dest = target / relative
        oracle = any(name in relative.parts for name in ("evaluation_evidence", "native_qualification"))
        if path.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            _copy_file(path, dest)
            if oracle:
                dest.chmod(0o600)
        elif oracle and path.is_dir():
            dest.mkdir(parents=True, exist_ok=True, mode=0o700)
            dest.chmod(0o700)


def _copy_file(source, destination):
    with _ordinary(source).open("rb") as reader, destination.open("xb") as writer:
        shutil.copyfileobj(reader, writer)
        writer.flush()
        os.fsync(writer.fileno())


def _sync_directories(root):
    for path in sorted([root, *(p for p in root.rglob("*") if p.is_dir())],
                       key=lambda p: len(p.parts), reverse=True):
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def retained_context(manifest_path, manifest):
    descriptor = manifest.get("ancestry_evidence")
    if descriptor is None:
        return None
    if (manifest.get("schema") != "sciel-lora-dataset/2" or type(descriptor) is not dict
            or set(descriptor) != {"schema", "directory", "manifest_sha256"}
            or descriptor["schema"] != SCHEMA or descriptor["directory"] != DIRECTORY):
        raise ValueError("invalid retained ancestry descriptor")
    require_hash(descriptor["manifest_sha256"], "retained ancestry manifest")
    root = _ordinary(Path(manifest_path).parent / DIRECTORY)
    path = root / "manifest.json"
    if sha256_bytes(path.read_bytes()) != descriptor["manifest_sha256"]:
        raise ValueError("retained ancestry manifest hash differs")
    value = read_json(path)
    if (type(value) is not dict or set(value) != {"schema", "registry", "datasets", "files"}
            or value["schema"] != SCHEMA or value["registry"] != "registry"
            or value["files"] != _files(root) or type(value["datasets"]) is not list):
        raise ValueError("retained ancestry files differ")
    datasets = []
    if len(value["datasets"]) != len(set(value["datasets"])):
        raise ValueError("duplicate retained dataset identity")
    for item in value["datasets"]:
        if type(item) is not str:
            raise ValueError("invalid retained dataset identity")
        require_hash(item, "retained dataset manifest")
        ds = root / "datasets" / item
        if sha256_bytes((ds / "manifest.json").read_bytes()) != item:
            raise ValueError("retained dataset manifest identity differs")
        datasets.append(dict(dataset=str(ds / "train.jsonl"), manifest=str(ds / "manifest.json")))
    return context(dict(registry=str(root / "registry"), datasets=datasets))


def retain(rows, sources, output, *, evaluation_context=None):
    nodes = [node for row in rows for node in row["provenance"].get("ancestry_resolution", {})
             .get("auxiliary_bindings", []) if node["kind"] == "adapter_revision"]
    if not nodes:
        return None
    sources = context(sources)
    from .registry import AdapterRegistry
    registry = AdapterRegistry(sources.sources["registry"])
    destination = Path(output) / DIRECTORY
    with tempfile.TemporaryDirectory(prefix=".ancestry-", dir=output) as temporary:
        root = Path(temporary) / DIRECTORY
        root.mkdir()
        datasets, revisions = set(), set()
        for node in nodes:
            owner, ident = node["owner"], node["identity"]["adapter_id"]
            if (owner, ident) not in revisions:
                revision = registry.inspect(owner, ident)
                if content_hash(revision) != node["proof"]["registry_manifest_sha256"]:
                    raise ValueError("retained revision differs from verified node")
                _copy_tree(registry._path(owner, ident), root / "registry" / "revisions" / owner / ident)
                revisions.add((owner, ident))
            digest = node["proof"]["dataset_manifest_sha256"]
            if digest in datasets:
                continue
            matches = [p for p in sources.sources.get("datasets", [])
                       if sha256_bytes(_ordinary(p["manifest"]).read_bytes()) == digest]
            if not matches:
                raise ValueError("retained adapter corpus is missing")
            selected = matches[0]
            if any(_ordinary(p["dataset"]).read_bytes() != _ordinary(selected["dataset"]).read_bytes()
                   for p in matches[1:]):
                raise ValueError("retained adapter corpus is ambiguous")
            target = root / "datasets" / digest
            target.mkdir(parents=True)
            _copy_file(selected["dataset"], target / "train.jsonl")
            _copy_file(selected["manifest"], target / "manifest.json")
            nested = read_json(selected["manifest"])
            if nested.get("ancestry_evidence") is not None:
                copy_retained(selected["manifest"], nested, target)
            from .evaluation_evidence import copy_retained as copy_evaluations
            copy_evaluations(selected["manifest"], nested, target)
            datasets.add(digest)
        value = dict(schema=SCHEMA, registry="registry", datasets=sorted(datasets), files=_files(root))
        write_json(root / "manifest.json", value, immutable=True)
        descriptor = dict(schema=SCHEMA, directory=DIRECTORY,
                          manifest_sha256=sha256_bytes((root / "manifest.json").read_bytes()))
        # Verify copied revisions and recursively re-open every selected corpus.
        copied = retained_context(root.parent / "manifest.json",
            dict(schema="sciel-lora-dataset/2", ancestry_evidence=descriptor))
        from .dataset import verify_episode
        for row in rows:
            verify_episode(row, ancestry_sources=copied, evaluation_context=evaluation_context)
        _sync_directories(root)
        if destination.exists():
            if read_json(destination / "manifest.json") != value or _files(destination) != value["files"]:
                raise ValueError("existing retained ancestry differs")
        else:
            root.rename(destination)
        return descriptor


def copy_retained(manifest_path, manifest, output):
    """Copy and reverify a dataset's retained evidence beside its new manifest."""
    if retained_context(manifest_path, manifest) is None:
        return
    source = Path(manifest_path).parent / DIRECTORY
    destination = Path(output) / DIRECTORY
    if destination.exists():
        raise FileExistsError(destination)
    _copy_tree(source, destination)
    retained_context(Path(output) / "manifest.json", manifest)
    _sync_directories(destination)
