"""Durable shared resource ledger; reservations precede dispatch and never replay it."""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal, localcontext
from pathlib import Path

from .funding import FundingPolicy
from .resource_values import (
    BudgetExceeded,
    BudgetUnknown,
    RESOURCES,
    _SCHEMA,
    _ZERO_HASH,
    _balance,
    _canonical,
    _cap_exceeds,
    _limit_vector,
    _text,
    _vector,
    _violations,
    _wire,
    resolution,
)


class ResourceLedger:
    """Immutable ceilings, append-only receipts and campaign-wide reservations."""

    def __init__(self, path, limits: dict):
        self.path = Path(path)
        self.limits = _limit_vector(limits)
        # One current verified state, never a cache of historical snapshots or
        # journal bodies. Its memory is linear in the durable reservation index.
        self._read_cache = None
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._locked() as (fd, state):
            if not state["_seq"]:
                self._append(fd, state, "init", {"limits": self.limits})
                directory = os.open(self.path.parent, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            elif state["limits"] != self.limits:
                raise ValueError("campaign limits cannot change on resume")

    @staticmethod
    def _open(path):
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0), 0o600)
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            os.close(fd)
            raise ValueError("ledger and lock must be regular files")
        return fd

    @contextmanager
    def _locked(self):
        lock = self._open(str(self.path) + ".lock")
        fd = None
        try:
            fcntl.flock(lock, fcntl.LOCK_EX)
            fd = self._open(self.path)
            digest, size = self._fingerprint(fd)
            cached = self._read_cache
            if (cached is not None and cached['_source_size'] == size
                    and cached['_source_hash'].digest() == digest):
                state = cached
            else:
                state = self._read(fd)
            if state["_seq"] and state["limits"] != self.limits:
                raise ValueError("campaign limits cannot change on resume")
            yield fd, state
            # Own writes update the expected raw-byte hash only after fsync.
            # A peer append or even same-size/restored-mtime tamper therefore
            # cannot be mistaken for our known state on the next operation.
            digest, size = self._fingerprint(fd)
            if state['_source_size'] != size or state['_source_hash'].digest() != digest:
                raise ValueError('resource ledger changed outside the locked transition')
            self._read_cache = state
        except BaseException:
            # _apply precedes persistence. Never retain a partly applied state
            # after validation, short-write, fsync, or user operation failure.
            self._read_cache = None
            raise
        finally:
            if fd is not None:
                os.close(fd)
            fcntl.flock(lock, fcntl.LOCK_UN)
            os.close(lock)

    @staticmethod
    def _fingerprint(fd):
        """Stream a full byte hash without JSON replay or trusting timestamps."""
        before = os.fstat(fd)
        digest, offset = hashlib.sha256(), 0
        while chunk := os.pread(fd, 1024 * 1024, offset):
            digest.update(chunk)
            offset += len(chunk)
        after = os.fstat(fd)
        if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
                != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
                or offset != after.st_size):
            raise ValueError('resource ledger changed while being fingerprinted')
        return digest.digest(), offset

    @staticmethod
    def _read(fd):
        state = {"limits": {}, "accounts": {}, "reservations": {}, "requests": {},
                 "halted": [], "_seq": 0, "_head": _ZERO_HASH, "_genesis": None,
                 "_source_size": 0, "_source_hash": hashlib.sha256()}
        os.lseek(fd, 0, os.SEEK_SET)
        with os.fdopen(os.dup(fd), "rb") as stream:
            while line := stream.readline(1024 * 1024):
                if not line.endswith(b"\n"):
                    raise ValueError("incomplete or oversized resource ledger event; manual recovery required")
                try:
                    event = json.loads(line)
                    claimed = event.pop("sha256")
                    if (set(event) != {"schema", "seq", "prev_sha256", "op", "data", "at"}
                            or event["schema"] != _SCHEMA or type(event["seq"]) is not int
                            or event["seq"] != state["_seq"] + 1 or event["prev_sha256"] != state["_head"]
                            or claimed != hashlib.sha256(_canonical(event)).hexdigest()):
                        raise ValueError("resource ledger hash chain mismatch")
                    ResourceLedger._apply(state, event["op"], event["data"])
                    state["_seq"], state["_head"] = event["seq"], claimed
                    if event["seq"] == 1:
                        state["_genesis"] = claimed
                    state['_source_hash'].update(line)
                    state['_source_size'] += len(line)
                except (KeyError, TypeError, json.JSONDecodeError) as exc:
                    raise ValueError("malformed resource ledger event") from exc
        return state

    @staticmethod
    def _apply(state, op, data):
        if op == "init":
            if state["_seq"] or set(data) != {"limits"}:
                raise ValueError("invalid ledger initialization")
            state["limits"] = _limit_vector(data["limits"])
        elif not state["_seq"]:
            raise ValueError("ledger must start with limits")
        elif op == "account":
            ident = _text(data["account_id"], "account ID")
            if set(data) != {"account_id", "limits"} or ident in state["accounts"]:
                raise ValueError("invalid account event")
            limits = _limit_vector(data["limits"])
            if any(_cap_exceeds(limits[k], state["limits"][k]) for k in RESOURCES):
                raise ValueError("account cap exceeds campaign ceiling")
            state["accounts"][ident] = limits
        elif op == "reserve":
            if set(data) != {"id", "account_id", "request_id", "bounds", "category"}:
                raise ValueError("invalid reservation event")
            ident, account = data["id"], data["account_id"]
            request = _text(data["request_id"], "request ID")
            if account not in state["accounts"] or request in state["requests"] or ident in state["reservations"]:
                raise ValueError("duplicate reservation or missing account")
            if ident != "r-" + hashlib.sha256(request.encode()).hexdigest():
                raise ValueError("reservation identity mismatch")
            _text(data["category"], "category", 128)
            ResourceLedger._admit(state, account, _vector(data["bounds"]))
            state["requests"][request] = ident
            state["reservations"][ident] = dict(deepcopy(data), status="reserved", actual={},
                actual_is_lower_bound=True, resolution=None, violations=[])
        elif op in ("settle", "resolve"):
            if set(data) != {"id", "actual", "status", "resolution"}:
                raise ValueError("invalid settlement event")
            row = state["reservations"][data["id"]]
            required = "unknown" if op == "resolve" else "reserved"
            if row["status"] != required:
                raise ValueError("invalid settlement transition")
            ResourceLedger._actual(row, data["actual"], data["status"], resolve=op == "resolve")
            if (op == "resolve") != (data["resolution"] is not None):
                raise ValueError("resolution audit missing or unexpected")
            if op == "resolve":
                ResourceLedger._resolution(**data["resolution"])
            row.update(actual=deepcopy(data["actual"]), status=data["status"],
                       actual_is_lower_bound=data["status"] == "unknown", resolution=deepcopy(data["resolution"]))
            row["violations"] = _violations(state, row)
            if row["violations"] and row["id"] not in state["halted"]:
                state["halted"].append(row["id"])
        else:
            raise ValueError("unknown resource ledger operation")

    @staticmethod
    def _append(fd, state, op, data):
        event = dict(schema=_SCHEMA, seq=state["_seq"] + 1, prev_sha256=state["_head"],
                     op=op, data=data, at=datetime.now(timezone.utc).isoformat())
        digest = hashlib.sha256(_canonical(event)).hexdigest()
        ResourceLedger._apply(state, op, data)
        encoded = _canonical(dict(event, sha256=digest)) + b"\n"
        content = memoryview(encoded)
        while content:
            n = os.write(fd, content)
            if n <= 0:
                raise OSError("resource ledger short write")
            content = content[n:]
        os.fsync(fd)
        state['_source_hash'].update(encoded)
        state['_source_size'] += len(encoded)
        state["_seq"], state["_head"] = event["seq"], digest
        if event["seq"] == 1:
            state["_genesis"] = digest

    @staticmethod
    def _admit(state, account, bounds):
        if state["halted"]:
            raise BudgetExceeded("resource overrun halted campaign dispatch")
        if any(r["account_id"] == account and r["status"] == "unknown" for r in state["reservations"].values()):
            raise BudgetUnknown("account has unresolved resource usage")
        with localcontext() as ctx:
            ctx.prec = 160
            for scope, limits in ((None, state["limits"]), (account, state["accounts"][account])):
                spent, held = _balance(state, scope)
                for key in RESOURCES:
                    if (limits[key] is not None
                            and spent[key] + held[key] + Decimal(str(bounds[key])) > Decimal(str(limits[key]))):
                        raise BudgetExceeded(f"{'campaign' if scope is None else 'account'} {key} ceiling leaves no room")

    def create_account(self, account_id, limits):
        account_id = _text(account_id, "account ID")
        limits = _limit_vector(limits, defaults=self.limits)
        with self._locked() as (fd, state):
            existing = state["accounts"].get(account_id)
            if existing is not None:
                if existing != limits:
                    raise ValueError("account limits cannot change with a revision or on resume")
            else:
                self._append(fd, state, "account", {"account_id": account_id, "limits": limits})
        return {"account_id": account_id, "limits": deepcopy(limits)}

    def reserve(self, account_id, request_id, bounds, category):
        account_id, request_id = _text(account_id, "account ID"), _text(request_id, "request ID")
        data = dict(id="r-" + hashlib.sha256(request_id.encode()).hexdigest(), account_id=account_id,
                    request_id=request_id, bounds=_vector(bounds), category=_text(category, "category", 128))
        with self._locked() as (fd, state):
            if request_id in state["requests"]:
                row = state["reservations"][state["requests"][request_id]]
                if any(row[key] != value for key, value in data.items()):
                    raise ValueError("request ID already binds a different reservation")
                if row["status"] in ("reserved", "unknown"):
                    raise BudgetUnknown("existing request cannot be dispatched again", record=row)
                return dict(deepcopy(row), reused=True, dispatch_allowed=False)
            if account_id not in state["accounts"]:
                raise ValueError("account must be explicitly created before dispatch")
            self._append(fd, state, "reserve", data)
            return dict(deepcopy(state["reservations"][data["id"]]), reused=False, dispatch_allowed=True)

    @staticmethod
    def _actual(row, actual, status, *, resolve=False):
        allowed = ("completed", "failed") if resolve else ("completed", "failed", "unknown")
        if status not in allowed:
            raise ValueError("invalid settlement status")
        actual = _vector(actual, partial=True)
        if status != "unknown":
            missing = [k for k in RESOURCES if Decimal(str(row["bounds"][k])) > 0 and k not in actual]
            if missing:
                raise ValueError("actual usage missing reserved resources: " + ", ".join(missing))
            actual = _vector(actual)
        if any(Decimal(str(actual.get(k, 0))) < Decimal(str(v)) for k, v in row["actual"].items()):
            raise ValueError("known usage cannot be reduced by settlement")
        return actual

    _resolution = staticmethod(resolution)

    def _settle(self, reservation_id, actual, status, resolution):
        with self._locked() as (fd, state):
            if reservation_id not in state["reservations"]:
                raise ValueError("unknown reservation ID")
            row = state["reservations"][reservation_id]
            actual = self._actual(row, actual, status, resolve=resolution is not None)
            if row["status"] == status and row["actual"] == actual and row["resolution"] == resolution:
                result = deepcopy(row)
            else:
                if row["status"] == "unknown" and resolution is None:
                    raise BudgetUnknown("unknown usage requires explicit audited resolution", record=row)
                if row["status"] != ("unknown" if resolution is not None else "reserved"):
                    raise ValueError("conflicting settlement of an existing receipt")
                self._append(fd, state, "resolve" if resolution is not None else "settle",
                             dict(id=reservation_id, actual=actual, status=status, resolution=resolution))
                result = deepcopy(state["reservations"][reservation_id])
        if result["violations"]:
            raise BudgetExceeded("observed resource usage exceeded its bound; charge recorded", record=result)
        return result

    def settle(self, reservation_id, actual, status="completed"):
        return self._settle(reservation_id, actual, status, None)

    def resolve_unknown(self, reservation_id, actual, *, reason, evidence_refs, status="completed"):
        """Resolve once from external evidence; no timeout-based/free-call guesses."""
        return self._settle(reservation_id, actual, status, self._resolution(reason, evidence_refs))

    def snapshot(self, account_id=None):
        with self._locked() as (_, state):
            if account_id is not None and account_id not in state["accounts"]:
                raise ValueError("unknown account")
            rows = [r for r in state["reservations"].values() if account_id is None or r["account_id"] == account_id]
            spent, held = _balance(state, account_id)
            limits = state["limits"] if account_id is None else state["accounts"][account_id]
            with localcontext() as ctx:
                ctx.prec = 160
                remaining = {k: None if limits[k] is None else
                             max(Decimal(0), Decimal(str(limits[k])) - spent[k] - held[k]) for k in RESOURCES}
                overrun = {k: Decimal(0) if limits[k] is None else
                           max(Decimal(0), spent[k] + held[k] - Decimal(str(limits[k]))) for k in RESOURCES}
                global_spent, global_held = _balance(state)
                shared = {k: None if state["limits"][k] is None else _wire(
                    max(Decimal(0), Decimal(str(state["limits"][k])) - global_spent[k] - global_held[k]), k) for k in RESOURCES}
            return deepcopy(dict(schema=_SCHEMA, account_id=account_id, limits=limits,
                spent={k: _wire(v, k) for k, v in spent.items()}, reserved={k: _wire(v, k) for k, v in held.items()},
                remaining={k: None if v is None else _wire(v, k) for k, v in remaining.items()}, overrun={k: _wire(v, k) for k, v in overrun.items()},
                shared_remaining=shared, halted=state["halted"],
                unknown_reservations=[r["id"] for r in rows if r["status"] == "unknown"],
                pending_reservations=[r["id"] for r in rows if r["status"] == "reserved"],
                reservations=rows, accounts=state["accounts"],
                journal={"events": state["_seq"], "head_sha256": state["_head"], "genesis_sha256": state["_genesis"]}))
