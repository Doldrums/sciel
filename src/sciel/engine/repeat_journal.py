"""Durable, hash-chained storage for the opt-in repeat-promotion coordinator."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile


def encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def fsync_directory(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def immutable_file(path, data):
    """Publish complete bytes before any journal reference; never replace a value."""
    path = Path(path)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"repeat artifact differs: {path.name}")
        return
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        # link, unlike replace, cannot overwrite a receipt written by another actor.
        os.link(temporary, path)
        fsync_directory(path.parent)
    finally:
        os.unlink(temporary)


class Journal:
    def __init__(self, path):
        self.path = Path(path)
        self.rows = []

    @property
    def anchor(self):
        return {"entries": len(self.rows),
                "sha256": self.rows[-1]["sha256"] if self.rows else "0" * 64}

    def read(self):
        raw = self.path.read_bytes()
        if raw and not raw.endswith(b"\n"):
            raise ValueError("truncated repeat journal")
        rows, previous = [], "0" * 64
        for line in raw.splitlines():
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != {"sequence", "previous", "event", "sha256"}:
                raise ValueError("malformed repeat journal row")
            body = {key: row[key] for key in ("sequence", "previous", "event")}
            if (type(row["sequence"]) is not int or row["sequence"] != len(rows)
                    or row["previous"] != previous or row["sha256"] != digest(body)
                    or encoded(row) != line):
                raise ValueError("repeat journal identity/hash mismatch")
            rows.append(row)
            previous = row["sha256"]
        return rows

    def append(self, event):
        if self.path.exists():
            if self.read() != self.rows:
                raise ValueError("repeat journal changed outside this coordinator")
        elif self.rows:
            raise ValueError("repeat journal disappeared")
        body = {"sequence": len(self.rows), "previous": self.anchor["sha256"], "event": event}
        row = dict(body, sha256=digest(body))
        with self.path.open("ab") as stream:
            stream.write(encoded(row) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        fsync_directory(self.path.parent)
        self.rows.append(row)
        return row
