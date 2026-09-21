"""Content identity for experiments using versioned local tasks and model files."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def safe_config(value):
    if isinstance(value, dict):
        hidden = {"api_key", "token", "password", "secret", "authorization"}
        return {k: "<redacted>" if k.lower() in hidden else safe_config(v)
                for k, v in value.items()}
    if isinstance(value, list):
        return [safe_config(v) for v in value]
    return value


def runtime_identity():
    names = ("numpy", "scipy", "torch", "PyYAML", "openai", "tenacity",
             "pydantic", "huggingface-hub")
    deps = {}
    for name in names:
        try:
            deps[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            deps[name] = None
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "platform": platform.platform(), "machine": platform.machine(),
            "dependencies": deps}


def _build_metadata(path):
    return any(part == "__pycache__" or part.endswith((".egg-info", ".dist-info"))
               for part in Path(path).parts[:-1])


def _tree_files(root, subtree, *, suffixes=None, exclude_build_metadata=False):
    root = Path(root)
    return {str(p.relative_to(root)) for p in (root / subtree).rglob("*")
            if p.is_file() and "__pycache__" not in p.relative_to(root).parts
            and (suffixes is None or p.suffix in suffixes)
            and (not exclude_build_metadata or not _build_metadata(p.relative_to(root)))}


def source_identity(root):
    root = Path(root).resolve()
    files = {name: file_sha(root / name) for name in sorted(_tree_files(
        root, "src", suffixes=(".py", ".txt"), exclude_build_metadata=True))}
    project_files = {"pyproject.toml": file_sha(root / "pyproject.toml")}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    paths = ("src", "tasks", "configs", "models", "pyproject.toml")
    changed = subprocess.check_output(["git", "diff", "--name-only", "-z", "HEAD", "--", *paths],
                                      cwd=root, text=True).split("\0")
    untracked = subprocess.check_output(["git", "ls-files", "--others", "--exclude-standard", "-z", "--",
                                        *paths], cwd=root, text=True).split("\0")
    # Generated package metadata is neither source identity nor a reason to reject
    # an otherwise clean checkout. This also covers unignored dist-info directories.
    dirty = any(name and not (Path(name).parts[0] == "src" and _build_metadata(name))
                for name in changed + untracked)
    return {"revision": revision, "files": files, "source_sha256": object_sha(files),
            "project_files": project_files, "dirty": dirty}


def task_identity(task):
    root = Path(task.root).resolve()
    paths = [root / name for name in ("task.yaml", "evaluator.py", "initial_program.py")]
    if (root / "assets").exists():
        paths += [p for p in sorted((root / "assets").rglob("*")) if p.is_file()]
    return {"task": task.qualified_name, "task_hash": task.task_hash,
            "files": {str(p.relative_to(root)): file_sha(p) for p in paths}}


def verify_files(root, files, *, exact_root=None, suffixes=None, exclude_build_metadata=False):
    """Check content and, for frozen trees, also reject newly added files."""
    for name, expected in files.items():
        path = Path(root) / name
        if not path.is_file() or file_sha(path) != expected:
            raise ValueError(f"experimental identity drift: {name}")
    if exact_root is not None:
        actual = _tree_files(root, exact_root, suffixes=suffixes,
                             exclude_build_metadata=exclude_build_metadata)
        if actual != set(files):
            changed = sorted(actual.symmetric_difference(files))
            raise ValueError(f"experimental identity drift: file set changed: {changed}")
