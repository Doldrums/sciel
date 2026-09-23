"""Small atomic stores; controller state is locked separately from artifacts."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import tempfile

from .types import canonical_bytes


def read_json(path: str | Path):
    def pairs(items):
        out = {}
        for key, value in items:
            if key in out:
                raise ValueError(f"duplicate JSON key: {key}")
            out[key] = value
        return out
    p = Path(path)
    if p.is_symlink():
        raise ValueError(f"symlink is not a trusted artifact: {p}")
    result = json.loads(p.read_bytes(), object_pairs_hook=pairs)
    canonical_bytes(result)
    return result


def write_json(path: str | Path, value, *, immutable: bool = False):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.is_symlink():
        raise ValueError("refusing symlink artifact")
    data = canonical_bytes(value) + b"\n"
    fd, temporary = tempfile.mkstemp(prefix=".publish-", dir=p.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            try:
                os.link(temporary, p)
            except FileExistsError:
                if p.read_bytes() != data:
                    raise ValueError(f"immutable artifact already exists: {p}")
        else:
            os.replace(temporary, p)
        directory = os.open(p.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def locked(root: str | Path):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("a") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)
