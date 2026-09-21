"""Immutable methods and laboratory revisions backed by a verified journal."""
from __future__ import annotations

import fcntl
import os
import stat
import threading
from collections import OrderedDict
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from .models import (
    canonical_bytes,
    content_hash,
    detached,
    hash_list,
    require_hash,
    validate_manifest,
    validate_runtime_lock,
)
from .registry_records import (
    RegistryCorruption,
    RevisionConflict,
    _RESERVED,
    _REVISION_FIELDS,
    _STATUSES,
    _ZERO,
    _atomic,
    _decode,
    _fsync_dir,
    _name,
    _read,
    _text,
    file_stamp,
)


_OBJECT_CACHE_BYTES = 16 * 1024 * 1024

_JOURNAL_CACHE_BYTES = 16 * 1024 * 1024


class Registry:
    def __init__(self, root):
        self.root = Path(root).absolute()
        if self.root.is_symlink():
            raise RegistryCorruption("registry root cannot be a symlink")
        self.root.mkdir(parents=True, exist_ok=True)
        self.root = self.root.resolve()
        self._thread_lock = threading.RLock()
        # Cached bytes never confer trust after a path changes. Object bodies
        # have an LRU serialized-byte budget; compact verified envelope metadata
        # lets projection skip historical prompt/checkpoint bodies after eviction.
        self._object_cache = OrderedDict()
        self._object_cache_bytes = 0
        self._verified_objects = {}
        self._load_cache = None
        for name in ("methods", "artifacts", "revisions"):
            path = self.root / name
            if path.is_symlink():
                raise RegistryCorruption("registry directory cannot be a symlink")
            path.mkdir(exist_ok=True)
        # Validate existing storage before making it available to a resumed host.
        with self._locked():
            self._load()

    @contextmanager
    def _locked(self):
        with self._thread_lock:
            fd = os.open(self.root / ".registry.lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
            try:
                if not stat.S_ISREG(os.fstat(fd).st_mode):
                    raise RegistryCorruption("registry lock must be a regular file")
                fcntl.flock(fd, fcntl.LOCK_EX)
                # Do not follow a directory swapped after construction either.
                for name in ("methods", "artifacts", "revisions"):
                    if (self.root / name).is_symlink():
                        raise RegistryCorruption("registry directory cannot be a symlink")
                yield
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
                os.close(fd)

    _stamp = staticmethod(file_stamp)

    def _remember_object(self, category, object_id, stamp, value):
        key = category, object_id
        self._verified_objects[key] = dict(stamp=stamp, schema=value.get('schema'),
            kind=value.get('kind'), fields=frozenset(value))
        old = self._object_cache.pop(key, None)
        if old is not None:
            self._object_cache_bytes -= old[2]
        size = stamp[3]
        if size <= _OBJECT_CACHE_BYTES:
            self._object_cache[key] = (stamp, value, size)
            self._object_cache_bytes += size
        while self._object_cache_bytes > _OBJECT_CACHE_BYTES:
            _, (_, _, evicted) = self._object_cache.popitem(last=False)
            self._object_cache_bytes -= evicted

    def _object(self, category, object_id, *, copy=True):
        require_hash(object_id)
        path = self.root / category / f"{object_id}.json"
        before = self._stamp(path)
        key = category, object_id
        cached = self._object_cache.get(key)
        if cached is not None and cached[0] == before:
            self._object_cache.move_to_end(key)
            return deepcopy(cached[1]) if copy else cached[1]
        try:
            result = _decode(_read(path))
        except FileNotFoundError as exc:
            raise RegistryCorruption("referenced registry artifact is missing") from exc
        if not isinstance(result, dict) or content_hash(result) != object_id:
            raise RegistryCorruption("registry artifact content hash mismatch")
        if self._stamp(path) != before:
            raise RegistryCorruption("registry artifact changed while being read")
        self._remember_object(category, object_id, before, result)
        return deepcopy(result) if copy else result

    def _artifact_header(self, object_id):
        """A cached envelope is sufficient for projection, never for body reads."""
        require_hash(object_id)
        key = 'artifacts', object_id
        stamp = self._stamp(self.root / 'artifacts' / f'{object_id}.json')
        known = self._verified_objects.get(key)
        if known is None or known['stamp'] != stamp:
            self._object('artifacts', object_id, copy=False)
            known = self._verified_objects[key]
        return known

    def _store(self, category, body):
        object_id = content_hash(body)
        _atomic(self.root / category / f"{object_id}.json", canonical_bytes(body), immutable=True)
        return object_id

    def _load(self, *, detach=True):
        path = self.root / "journal.jsonl"
        stamp = self._stamp(path, missing=True)
        cached = self._load_cache
        if cached is not None and cached[0] == stamp:
            # Other cooperating processes and noncooperating file changes are
            # observed on every cache hit, not deferred until a later append.
            if all(self._stamp(self.root / category / f'{ident}.json') == old
                   for (category, ident), old in cached[3].items()):
                return (deepcopy(cached[1]), deepcopy(cached[2])) if detach else (cached[1], cached[2])
        try:
            raw = _read(path)
        except FileNotFoundError:
            raw = b""
        if raw and not raw.endswith(b"\n"):
            raise RegistryCorruption("incomplete journal append; manual audited repair required")
        events = []
        head = _ZERO
        for seq, line in enumerate(raw.splitlines(), 1):
            event = _decode(line)
            if not isinstance(event, dict) or set(event) != {"schema", "seq", "prev", "kind", "data", "at", "id"}:
                raise RegistryCorruption("invalid journal event shape")
            payload = {key: value for key, value in event.items() if key != "id"}
            if (event["schema"] != "sciel-method-event/1" or type(event["seq"]) is not int
                    or event["seq"] != seq or event["prev"] != head
                    or event["id"] != content_hash(payload)):
                raise RegistryCorruption("journal hash chain mismatch")
            _name(event["kind"], "event kind")
            if not isinstance(event["data"], dict) or not isinstance(event["at"], str):
                raise RegistryCorruption("invalid journal event payload")
            head = event["id"]
            events.append(event)
        state = self._project(events)
        if self._stamp(path, missing=True) != stamp:
            raise RegistryCorruption('registry journal changed while being read')
        if len(raw) <= _JOURNAL_CACHE_BYTES:
            references = {(category, ident): self._verified_objects[(category, ident)]['stamp']
                for category in ('methods', 'artifacts', 'revisions') for ident in state[category]}
            self._load_cache = stamp, events, state, references
        else:
            self._load_cache = None
        return (deepcopy(events), deepcopy(state)) if detach else (events, state)

    def _project(self, events):
        state = {"methods": {}, "artifacts": {}, "labs": {}, "revisions": {}}
        try:
            for event in events:
                kind, data = event["kind"], event["data"]
                if kind == "artifact_registered":
                    header = self._artifact_header(data["id"])
                    if header['schema'] != "sciel-method-artifact/1" or header['kind'] != data["kind"]:
                        raise RegistryCorruption("artifact kind mismatch")
                    if header['fields'] != {"schema", "kind", "body"}:
                        raise RegistryCorruption("invalid artifact envelope")
                    if data["id"] in state["artifacts"]:
                        raise RegistryCorruption("duplicate artifact registration")
                    state["artifacts"][data["id"]] = data["kind"]
                elif kind == "method_registered":
                    body = self._object("methods", data["id"], copy=False)
                    if set(body) != {"schema", "manifest", "files"} or body["schema"] != "sciel-method/1":
                        raise RegistryCorruption("invalid method envelope")
                    validate_manifest(body["manifest"], body["files"])
                    self._method_refs(body["manifest"], state)
                    if data["id"] in state["methods"]:
                        raise RegistryCorruption("duplicate method registration")
                    state["methods"][data["id"]] = {"status": "draft", "evidence_refs": []}
                elif kind == "method_status":
                    self._status(data["id"], data["status"], data["evidence_refs"], state)
                elif kind in ("lab_created", "lab_revised"):
                    revision = self._object("revisions", data["revision_id"], copy=False)
                    self._validate_revision(revision, state)
                    lab_id = revision["lab_id"]
                    if data.get("lab_id", lab_id) != lab_id:
                        raise RegistryCorruption("announced lab differs from revision")
                    if kind == "lab_created":
                        if lab_id in state["labs"] or revision["previous_revision"] is not None:
                            raise RegistryCorruption("duplicate lab or invalid first revision")
                        _text(data["agenda"], "agenda")
                        if data["mode"] not in ("A", "B", "C"):
                            raise ValueError("unsupported lab mode")
                        if any(lab["budget_account"] == revision["budget_account"] for lab in state["labs"].values()):
                            raise ValueError("a new lab requires its own budget account")
                        state["labs"][lab_id] = {"id": lab_id, "agenda": data["agenda"], "mode": data["mode"],
                                                "budget_account": revision["budget_account"], "status": "trial"}
                    else:
                        lab = state["labs"][lab_id]
                        if revision["previous_revision"] != lab["active_revision"]:
                            raise RegistryCorruption("revision activation is not a compare-and-swap")
                        if revision["budget_account"] != lab["budget_account"]:
                            raise RegistryCorruption("revision cannot change budget account")
                    state["revisions"][data["revision_id"]] = lab_id
                    state["labs"][lab_id]["active_revision"] = data["revision_id"]
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, RegistryCorruption):
                raise
            raise RegistryCorruption("invalid registry state transition") from exc
        return state

    @staticmethod
    def _method_refs(manifest, state):
        for key in ("dependencies", "parents"):
            for ref in manifest[key]:
                if ref not in state["methods"]:
                    raise ValueError(f"unknown method {key} reference")
        # References must precede the new immutable version in the journal. This
        # topological order excludes cycles, including self-referential packages.

    @staticmethod
    def _evidence(refs, state):
        hash_list(refs, "evidence_refs")
        if any(ref not in state["artifacts"] for ref in refs):
            raise ValueError("unknown evidence artifact")

    def _status(self, method_id, status, evidence_refs, state):
        if method_id not in state["methods"] or not isinstance(status, str) or status not in _STATUSES:
            raise ValueError("unknown method or status")
        self._evidence(evidence_refs, state)
        previous = state["methods"][method_id]
        if previous["status"] == "superseded" and status != "superseded":
            raise ValueError("superseded method is terminal; register a new version")
        if previous["status"] != "draft" and status == "draft":
            raise ValueError("a published status cannot become draft")
        refs = list(dict.fromkeys(previous["evidence_refs"] + evidence_refs))
        if status == "released" and not refs:
            raise ValueError("release requires evidence; coordinator assesses its sufficiency")
        state["methods"][method_id] = {"status": status, "evidence_refs": refs}

    def _validate_revision(self, revision, state):
        if set(revision) != _REVISION_FIELDS | {"schema", "lab_id", "previous_revision", "parent_revision", "budget_account", "reason"}:
            raise ValueError("unsupported lab revision fields")
        if revision["schema"] != "sciel-lab-revision/1":
            raise ValueError("unsupported lab revision schema")
        _name(revision["lab_id"], "lab ID")
        _name(revision["budget_account"], "budget account")
        _text(revision["reason"], "reason")
        previous, parent = revision["previous_revision"], revision["parent_revision"]
        if previous is not None:
            require_hash(previous)
            if state["revisions"].get(previous) != revision["lab_id"]:
                raise ValueError("previous revision must belong to this lab")
        if parent is not None:
            require_hash(parent)
            if parent not in state["revisions"]:
                raise ValueError("unknown fork parent revision")
            original = self._object("revisions", parent)
            if original["lab_id"] == revision["lab_id"] or original["budget_account"] == revision["budget_account"]:
                raise ValueError("fork needs its own lab and budget account")
        for key, expected in (("active_methods", None), ("active_adapters", "adapter")):
            for ref in hash_list(revision[key], key):
                if ref not in state["methods"]:
                    raise ValueError("unknown active method")
                if expected and self._object("methods", ref)["manifest"]["kind"] != expected:
                    raise ValueError("active adapter must reference adapter method")
        require_hash(revision["search_procedure"], "search procedure")
        for key in ("generator", "capabilities"):
            if not isinstance(revision[key], dict):
                raise ValueError(f"revision {key} must be an object")
        validate_runtime_lock(revision["runtime_lock"])
        self._evidence(revision["evidence_refs"], state)

    def _append(self, kind, data, events):
        payload = {"schema": "sciel-method-event/1", "seq": len(events) + 1,
                   "prev": events[-1]["id"] if events else _ZERO, "kind": kind,
                   "data": detached(data), "at": datetime.now(timezone.utc).isoformat()}
        event = {**payload, "id": content_hash(payload)}
        data_bytes = canonical_bytes(event) + b"\n"
        fd = os.open(self.root / "journal.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise RegistryCorruption("journal must be a regular file")
            view = memoryview(data_bytes)
            while view:
                count = os.write(fd, view)
                if count <= 0:
                    raise OSError("short journal write")
                view = view[count:]
            os.fsync(fd)
        finally:
            os.close(fd)
        _fsync_dir(self.root)
        return detached(event)

    def register(self, bundle, manifest):
        manifest = validate_manifest(manifest, bundle)
        body = {"schema": "sciel-method/1", "manifest": manifest, "files": detached(bundle)}
        method_id = content_hash(body)
        with self._locked():
            events, state = self._load()
            self._method_refs(manifest, state)
            if method_id not in state["methods"]:
                self._store("methods", body)
                self._append("method_registered", {"id": method_id}, events)
        return method_id

    def get_method(self, method_id):
        require_hash(method_id)
        with self._locked():
            _, state = self._load()
            if method_id not in state["methods"]:
                raise KeyError(method_id)
            value = self._object("methods", method_id)
        return {"id": method_id, "manifest": value["manifest"], "files": value["files"]}

    def put_artifact(self, kind, body):
        _name(kind, "artifact kind")
        value = {"schema": "sciel-method-artifact/1", "kind": kind, "body": detached(body)}
        artifact_id = content_hash(value)
        with self._locked():
            events, state = self._load()
            if artifact_id not in state["artifacts"]:
                self._store("artifacts", value)
                self._append("artifact_registered", {"id": artifact_id, "kind": kind}, events)
        return artifact_id

    def get_artifact(self, kind, artifact_id):
        _name(kind, "artifact kind")
        require_hash(artifact_id)
        with self._locked():
            _, state = self._load()
            if state["artifacts"].get(artifact_id) != kind:
                raise KeyError((kind, artifact_id))
            return self._object("artifacts", artifact_id)["body"]

    def append(self, kind, data):
        _name(kind, "event kind")
        if kind in _RESERVED:
            raise ValueError("use the typed registry transition API")
        if not isinstance(data, dict):
            raise ValueError("event data must be an object")
        with self._locked():
            events, _ = self._load()
            return self._append(kind, data, events)

    def events(self, kind=None):
        """Return detached events after verifying the entire journal and its artifacts."""
        if kind is not None:
            _name(kind, "event kind")
        with self._locked():
            events, _ = self._load(detach=False)
            selected = events if kind is None else [row for row in events if row["kind"] == kind]
            return deepcopy(selected)

    def create_lab(self, lab_id, agenda, budget_account, mode, initial_revision_fields):
        _name(lab_id, "lab ID")
        _name(budget_account, "budget account")
        _text(agenda, "agenda")
        if mode not in ("A", "B", "C") or not isinstance(initial_revision_fields, dict):
            raise ValueError("invalid lab mode or revision fields")
        supplied = detached(initial_revision_fields)
        if set(supplied) - _REVISION_FIELDS - {"parent_revision"}:
            raise ValueError("unsupported initial revision fields")
        with self._locked():
            events, state = self._load()
            if lab_id in state["labs"]:
                raise RevisionConflict("lab already exists")
            if any(lab["budget_account"] == budget_account for lab in state["labs"].values()):
                raise ValueError("a new lab requires its own budget account")
            parent = supplied.pop("parent_revision", None)
            fields = {"active_methods": [], "active_adapters": [], "generator": {},
                      "capabilities": {}, "evidence_refs": []}
            if parent is not None:
                require_hash(parent)
                if parent not in state["revisions"]:
                    raise ValueError("unknown fork parent revision")
                original = self._object("revisions", parent)
                fields.update({key: original[key] for key in _REVISION_FIELDS})
            fields.update(supplied)
            revision = {"schema": "sciel-lab-revision/1", "lab_id": lab_id, "previous_revision": None,
                        "parent_revision": parent, "budget_account": budget_account,
                        "reason": "fork" if parent else "initial", **fields}
            self._validate_revision(revision, state)
            revision_id = self._store("revisions", revision)
            self._append("lab_created", {"revision_id": revision_id, "lab_id": lab_id,
                                         "agenda": agenda, "mode": mode}, events)
            return {"id": lab_id, "agenda": agenda, "mode": mode, "budget_account": budget_account,
                    "status": "trial", "active_revision": revision_id}

    def lab(self, lab_id):
        _name(lab_id, "lab ID")
        with self._locked():
            _, state = self._load()
            return detached(state["labs"][lab_id])

    def revision(self, revision_id):
        require_hash(revision_id)
        with self._locked():
            _, state = self._load()
            if revision_id not in state["revisions"]:
                raise KeyError(revision_id)
            return self._object("revisions", revision_id)

    def revise_lab(self, lab_id, expected_revision, changes, reason):
        _name(lab_id, "lab ID")
        require_hash(expected_revision, "expected revision")
        _text(reason, "reason")
        if not isinstance(changes, dict) or set(changes) - _REVISION_FIELDS:
            raise ValueError("revision cannot change lab, lineage, or budget identity")
        changes = detached(changes)
        with self._locked():
            events, state = self._load()
            if state["labs"][lab_id]["active_revision"] != expected_revision:
                raise RevisionConflict("lab revision changed; reload before applying changes")
            revision = {**self._object("revisions", expected_revision), **changes,
                        "previous_revision": expected_revision, "reason": reason}
            self._validate_revision(revision, state)
            revision_id = self._store("revisions", revision)
            self._append("lab_revised", {"revision_id": revision_id, "lab_id": lab_id}, events)
            return revision_id

    def status_method(self, method_id, status, evidence_refs=None):
        require_hash(method_id)
        refs = [] if evidence_refs is None else detached(evidence_refs)
        with self._locked():
            events, state = self._load()
            self._status(method_id, status, refs, state)
            self._append("method_status", {"id": method_id, "status": status, "evidence_refs": refs}, events)
            return detached(state["methods"][method_id])

    def method_status(self, method_id):
        require_hash(method_id)
        with self._locked():
            _, state = self._load()
            return detached(state["methods"][method_id])

    def checkpoint(self, path):
        path = Path(path).absolute()
        if path.is_symlink():
            raise RegistryCorruption("checkpoint destination cannot be a symlink")
        destination = path.resolve()
        # A checkpoint must not overwrite any authoritative registry object.
        if destination == self.root / "journal.jsonl" or destination == self.root / ".registry.lock" or any(
                folder in destination.parents for folder in (self.root / "methods", self.root / "artifacts", self.root / "revisions")):
            raise ValueError("checkpoint path conflicts with authoritative registry storage")
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked():
            events, state = self._load()
            body = {"schema": "sciel-method-checkpoint/1", "journal": {"count": len(events),
                    "head": events[-1]["id"] if events else _ZERO}, "state": state}
            value = {**body, "id": content_hash(body)}
            _atomic(path, canonical_bytes(value))
            return detached(value)

    def verify_checkpoint(self, path):
        with self._locked():
            events, _ = self._load()
            value = _decode(_read(Path(path)))
            if not isinstance(value, dict) or set(value) != {"schema", "journal", "state", "id"}:
                raise RegistryCorruption("invalid checkpoint shape")
            body = {key: item for key, item in value.items() if key != "id"}
            if value["schema"] != "sciel-method-checkpoint/1" or content_hash(body) != value["id"]:
                raise RegistryCorruption("checkpoint hash mismatch")
            anchor = value["journal"]
            if not isinstance(anchor, dict) or set(anchor) != {"count", "head"}:
                raise RegistryCorruption("invalid checkpoint journal anchor")
            count = anchor["count"]
            if type(count) is not int or not 0 <= count <= len(events):
                raise RegistryCorruption("checkpoint journal prefix is missing")
            if anchor["head"] != (events[count - 1]["id"] if count else _ZERO):
                raise RegistryCorruption("checkpoint journal anchor mismatch")
            if value["state"] != self._project(events[:count]):
                raise RegistryCorruption("checkpoint state does not match anchored journal")
            return detached(value)
