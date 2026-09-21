"""Canonical registry records, safe file access and atomic persistence."""
from __future__ import annotations

import json
import os
import re
import stat
import tempfile
from pathlib import Path

from .models import canonical_bytes


_ZERO = "0" * 64


_RESERVED = {"method_registered", "artifact_registered", "method_status", "lab_created", "lab_revised"}


_REVISION_FIELDS = {"active_methods", "active_adapters", "search_procedure", "generator",
                    "runtime_lock", "capabilities", "evidence_refs"}


_STATUSES = {"draft", "released", "quarantined", "superseded"}


class RegistryCorruption(ValueError):
    """A durable artifact, journal chain, or checkpoint failed validation."""


class RevisionConflict(ValueError):
    """A lab revision changed since the caller read it."""


def _name(value, label="name"):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise ValueError(f"invalid {label}")
    return value


def _text(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be nonempty text")
    return value


def _decode(data):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise RegistryCorruption("duplicate JSON key")
            result[key] = value
        return result
    try:
        value = json.loads(data, object_pairs_hook=pairs,
                           parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
        canonical_bytes(value)
        return value
    except (ValueError, UnicodeError, TypeError, RecursionError) as exc:
        raise RegistryCorruption("invalid registry JSON") from exc


def _read(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise RegistryCorruption("registry path is not a regular file")
        with os.fdopen(fd, "rb", closefd=False) as handle:
            return handle.read()
    finally:
        os.close(fd)


def _fsync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic(path, data, *, immutable=False):
    if path.is_symlink():
        raise RegistryCorruption("refusing symlink destination")
    fd, name = tempfile.mkstemp(prefix=".write-", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        if immutable:
            try:
                os.link(temporary, path)
            except FileExistsError:
                if _read(path) != data:
                    raise RegistryCorruption("immutable artifact already exists with different bytes")
        else:
            os.replace(temporary, path)
        _fsync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def file_stamp(path, *, missing=False):
    try:
        value = path.lstat()
    except FileNotFoundError:
        if missing:
            return None
        raise RegistryCorruption("referenced registry artifact is missing") from None
    if not stat.S_ISREG(value.st_mode):
        raise RegistryCorruption("registry path is not a regular file")
    # ctime catches in-place tampering even if size and mtime are restored.
    return (value.st_dev, value.st_ino, value.st_mode, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)
