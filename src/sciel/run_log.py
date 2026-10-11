"""Append-only JSONL events with explicit checkpoint/segment provenance."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import hashlib
import logging
import os
import uuid
from time import monotonic

SCHEMA_VERSION = 2
EVENT_TYPES = {
    "run_start", "iteration", "sh_cut",
    "window_closed", "decision", "checkpoint", "checkpoint_restore", "run_end", "population",
    "research", "research_tool", "repeat_search",
}

_log = logging.getLogger("sciel.run_log")


def _timestamp():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class RunLog:
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._fh = open(path, "ab", buffering=0)
        self.run_id = self.segment_id = None
        self.event_seq = 0
        self._segment_started = None
        self._healthy = True
        self._llm_context = {}
        self._llm_attempt = {}
        # Hash a restored prefix once, in bounded chunks. Subsequent checkpoints
        # reuse the digest of our own append-only writes, under the run writer lock.
        self._prefix_hash = hashlib.sha256()
        self._offset = 0
        # Parallel staging can store only a new delta while inheriting the
        # authenticated digest/virtual offset of an existing journal prefix.
        # Ordinary logs keep zero: physical and checkpoint offsets coincide.
        self._prefix_bytes = 0
        self._last_record = None
        try:
            self._signature = self._file_signature()
            terminated = True
            with open(self.path, "rb") as source:
                while chunk := source.read(1024 * 1024):
                    self._prefix_hash.update(chunk)
                    self._offset += len(chunk)
                    terminated = chunk.endswith(b"\n")
            self._check_unchanged()
            if self._offset != self._signature[2]:
                raise OSError("journal was changed by another writer")
            # Construction also precedes checkpoint validation on resume. Let
            # that layer report its provenance error, but never append onto a
            # partial record even for a legacy checkpoint lacking an anchor.
            self._healthy = terminated
        except BaseException:
            self._fh.close()
            raise

    @staticmethod
    def _stat_signature(value):
        return (value.st_dev, value.st_ino, value.st_size,
                value.st_mtime_ns, value.st_ctime_ns)

    def _file_signature(self):
        opened = self._stat_signature(os.fstat(self._fh.fileno()))
        if self._stat_signature(os.stat(self.path)) != opened:
            raise OSError("journal was replaced by another writer")
        return opened

    def _check_unchanged(self):
        if self._file_signature() != self._signature:
            self._healthy = False
            raise OSError("journal was changed by another writer")

    def emit(self, event_type: str, **fields) -> bool:
        try:
            if not self._healthy:
                raise OSError("journal write previously failed")
            self._check_unchanged()
            if event_type not in EVENT_TYPES:
                raise ValueError(f"unknown event type {event_type}")
            # Logger-owned observation times; scientific payloads cannot supply
            # or replace them. Monotonic elapsed time is comparable only inside
            # this segment (a resume starts a new segment, excluding downtime).
            record = {"event": event_type, **fields, "timestamp": _timestamp()}
            record.pop("segment_elapsed_seconds", None)
            if self.segment_id is not None:
                record.update(run_id=self.run_id, segment_id=self.segment_id,
                              event_seq=self.event_seq + 1,
                              segment_elapsed_seconds=max(0.0, monotonic() - self._segment_started))
            data = (json.dumps(record, default=_fallback) + "\n").encode("utf-8")
            if self._fh.write(data) != len(data):
                raise OSError("short journal write")
            signature = self._file_signature()
            if signature[2] != self._offset - self._prefix_bytes + len(data):
                raise OSError("journal was changed by another writer")
            self._prefix_hash.update(data)
            self._offset += len(data)
            self._signature = signature
            self.event_seq += 1
            self._last_record = (record.get("segment_id"), record.get("event_seq"))
            return True
        except Exception as e:
            self._healthy = False
            _log.warning("run_log emit failed; checkpoint provenance unavailable: %s", e)
            return False

    def run_start(self, restored_checkpoint=None, **f):
        provenance = (restored_checkpoint or {}).get("provenance") or {}
        self.run_id = provenance.get("run_id") or str(uuid.uuid4())
        self.segment_id, self.event_seq = str(uuid.uuid4()), 0
        self._segment_started = monotonic()
        if not self.emit("run_start", schema_version=SCHEMA_VERSION,
                         resumed=restored_checkpoint is not None, **f):
            raise OSError("cannot start a journal segment")
        if restored_checkpoint is not None and not self.emit("checkpoint_restore", **restored_checkpoint):
            raise OSError("cannot record checkpoint restoration")

    def anchor(self):
        """Checkpoint state ends at this exact physical journal prefix."""
        if not self._healthy:
            raise OSError("journal write failed; refusing an unverifiable checkpoint")
        if self.segment_id is None:
            return None  # State saved before run_start has no provable ancestry.
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._check_unchanged()
        if self._last_record != (self.segment_id, self.event_seq):
            raise OSError("journal was changed by another writer")
        return {"segment_id": self.segment_id, "event_seq": self.event_seq,
                "byte_offset": self._offset, "prefix_sha256": self._prefix_hash.hexdigest()}

    def iteration(self, **f):
        self.emit("iteration", **f)

    def sh_cut(self, **f):
        self.emit("sh_cut", **f)

    def window_closed(self, **f):
        self.emit("window_closed", **f)

    def decision(self, **f):
        self.emit("decision", **f)

    def research_tool(self, **f):
        """Tool evidence is durable even when no candidate is produced."""
        if not self.emit('research_tool', **f):
            raise OSError('cannot persist research tool evidence')
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def research(self, **f):
        if not self.emit('research', **f):
            raise OSError('cannot persist research evidence')
        self._fh.flush()
        os.fsync(self._fh.fileno())
        if f.get('phase') == 'attempt':
            evidence = f.get('evidence') or {}
            self._llm_attempt = dict(experiment_id=f.get('experiment_id'),
                                    iteration=evidence.get('iteration'))
            self._llm_context = dict(self._llm_attempt, role='G',
                                    prompt_sha256=f.get('prompt_sha256'))
        elif f.get('phase') == 'aux_start':
            reference = f.get('reference')
            self._llm_context = dict(role={'conclusion': 'C', 'reflection': 'R',
                'idea': 'idea'}.get(f.get('kind'), 'idea' if 'idea' in str(f.get('kind')) else 'unknown'),
                reference=reference, call_index=f.get('call_index'), kind=f.get('kind'),
                prompt_sha256=f.get('prompt_sha256'))
            if reference == self._llm_attempt.get('experiment_id'):
                self._llm_context.update(self._llm_attempt)
        elif f.get('phase') == 'idea' and f.get('action') == 'request_start':
            self._llm_context = dict(role='G' if f.get('stage') == 'code' else 'idea',
                idea_id=f.get('idea_id'), stage=f.get('stage'), reference=f.get('idea_id'),
                experiment_id=f.get('experiment_id'), iteration=f.get('iteration'),
                prompt_sha256=f.get('prompt_sha256'))

    def repeat_search(self, **f):
        """H15 search intent: a crash must not silently replay an unknown call."""
        if not self.emit('repeat_search', **f):
            raise OSError('cannot persist repeat-promotion search intent')
        self._fh.flush()
        os.fsync(self._fh.fileno())

    def checkpoint(self, **f):
        if not self.emit("checkpoint", **f):
            raise OSError("cannot record checkpoint identity")

    def population(self, **f):
        self.emit("population", **f)

    def run_end(self, **f):
        if not self.emit("run_end", **f):
            raise OSError("cannot record completed journal segment")

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def _fallback(o):
    return str(o)
