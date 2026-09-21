"""Source closure for opt-in policies whose helpers span package modules."""
from __future__ import annotations

import hashlib
from pathlib import Path


def package_source_hashes(*, files=(), packages=('engine', 'hypothesis', 'knowledge_base')):
    """Bind every Python helper in the selected packages and explicit leaves."""
    root = Path(__file__).resolve().parents[1]
    names = set(files)
    for package in packages:
        names.update(path.relative_to(root).as_posix()
                     for path in (root / package).rglob('*.py'))
    return {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in sorted(names)}
