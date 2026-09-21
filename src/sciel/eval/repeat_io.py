"""Bounded JSON transport for repeat-evaluation subprocesses."""

import json
import os
from pathlib import Path
import stat

_MAX_JSON_BYTES = 65536


class RepeatInfrastructureError(RuntimeError):
    """A preflight/helper failure; no numeric candidate quality was measured."""


def _write_json(path, value):
    with Path(path).open("xb") as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())
        stream.flush()
        os.fsync(stream.fileno())


def _read_json(path):
    """Bounded regular-file transport; output I/O failure is infrastructure."""
    path = Path(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > _MAX_JSON_BYTES:
        raise RepeatInfrastructureError("helper output is not a bounded regular file")
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) |
                         getattr(os, "O_NONBLOCK", 0))
    with os.fdopen(descriptor, "rb") as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode) or opened.st_size > _MAX_JSON_BYTES:
            raise RepeatInfrastructureError("helper output changed type or exceeds transport limit")
        raw = stream.read(_MAX_JSON_BYTES + 1)
    if len(raw) > _MAX_JSON_BYTES:
        raise RepeatInfrastructureError("helper output exceeds transport size limit")

    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("duplicate JSON key")
            value[key] = item
        return value

    def constant(value):
        raise ValueError("nonfinite JSON constant: " + value)
    result = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    if not isinstance(result, dict):
        raise RepeatInfrastructureError("helper output must be an object")
    return result
