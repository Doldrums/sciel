"""Durable calls for research episodes; cached receipts are not new trials."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import re
import stat
import time

from .budget import BudgetUnknown, ResourceLedger, _vector
from .models import canonical_bytes, content_hash, detached, hash_list
from .registry import _atomic, _decode, _read
from ..campaign.budget import RequestNotDispatched


class DispatchStore:
    def __init__(self, root, registry, ledger):
        self.root = Path(root)
        if self.root.is_symlink():
            raise ValueError('dispatch root cannot be a symlink')
        self.root.mkdir(parents=True, exist_ok=True)
        self.root = self.root.resolve()
        self.registry, self.ledger = registry, ledger

    @staticmethod
    def _read(path):
        value = _decode(_read(path))
        if (not isinstance(value, dict) or value.get('sha256') !=
                content_hash({k: v for k, v in value.items() if k != 'sha256'})):
            raise ValueError('dispatch receipt integrity mismatch')
        return value

    @staticmethod
    def _write(path, body):
        _atomic(path, canonical_bytes(dict(body, sha256=content_hash(body))), immutable=True)

    @contextmanager
    def _locked(self, key, account):
        for value in (key, account):
            if not isinstance(value, str) or not value.strip() or len(value) > 256:
                raise ValueError('dispatch key/account must be bounded nonempty strings')
        directory = self.root / content_hash({'key': key, 'account': account})
        if directory.is_symlink():
            raise ValueError('dispatch directory cannot be a symlink')
        directory.mkdir(exist_ok=True)
        fd = os.open(directory / '.lock', os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError('dispatch lock must be a regular file')
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield directory
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    def _artifact(self, receipt):
        return self.registry.put_artifact('call_receipt',
            {k: v for k, v in receipt.items() if k != 'sha256'})

    @staticmethod
    def _bound(receipt, intent):
        if (receipt.get('request_sha256') != intent['request_sha256']
                or receipt.get('reservation_id') != intent['reservation_id']
                or any(receipt.get(k) != intent[k] for k in ('key', 'account', 'category'))):
            raise ValueError('receipt belongs to another dispatch')

    def _finish(self, receipt, *, replayed, resolution=None):
        if resolution is None:
            self.ledger.settle(receipt['reservation_id'], receipt['actual'], status=receipt['status'])
        else:
            self.ledger.resolve_unknown(receipt['reservation_id'], receipt['actual'],
                status=receipt['status'], reason=resolution['reason'], evidence_refs=resolution['evidence_refs'])
        artifact_id = self._artifact(receipt)
        if receipt['status'] == 'unknown':
            raise BudgetUnknown('saved unknown outcome needs an audited resolution', record={'call_receipt_id': artifact_id})
        return dict(receipt['result'], call_receipt_id=artifact_id,
                    reservation_id=receipt['reservation_id'], replayed=replayed)

    def run(self, key, account, category, request, bounds, function):
        """``function(reservation_id) -> (JSON result object, resource actuals)``.

        Call-specific OS locks serialize concurrent hosts. Intent and dispatch
        markers precede the function. Missing terminal receipts fail closed;
        known terminal receipts finish settlement without calling it again.
        """
        request = detached(request)
        bounds = _vector(bounds)
        identity = dict(key=key, account=account, category=category, request=request, bounds=bounds)
        digest = content_hash(identity)
        with self._locked(key, account) as directory:
            intent_path, result_path = directory / 'intent.json', directory / 'receipt.json'
            if intent_path.exists():
                intent = self._read(intent_path)
                if (intent.get('request_sha256') != digest or
                        any(intent.get(k) != value for k, value in identity.items())):
                    raise ValueError('logical call reused with another request or budget')
                resolution_path = directory / 'resolution.json'
                if resolution_path.exists():
                    resolution = self._read(resolution_path)
                    self._bound(resolution, intent)
                    original = self._read(result_path) if result_path.exists() else None
                    if resolution['original_receipt_sha256'] != (original['sha256'] if original else None):
                        raise ValueError('audited resolution no longer binds the original receipt')
                    return self._finish(resolution, replayed=True, resolution=resolution['audit'])
                if not result_path.exists():
                    rows = {row['id']: row for row in self.ledger.snapshot(account)['reservations']}
                    row = rows[intent['reservation_id']]
                    if row['status'] in ('reserved', 'unknown'):
                        self.ledger.settle(intent['reservation_id'], row['actual'], status='unknown')
                    raise BudgetUnknown('dispatch has no durable terminal receipt')
                receipt = self._read(result_path)
                self._bound(receipt, intent)
                return self._finish(receipt, replayed=True)

            reservation_key = 'methods-' + content_hash({'key': key, 'account': account})
            reservation = self.ledger.reserve(account, reservation_key, bounds, category)
            if not reservation['dispatch_allowed']:
                raise BudgetUnknown('settled request has no recoverable dispatch receipt')
            reservation_id = reservation['id']
            intent = dict(schema='sciel-method-dispatch/1', request_sha256=digest,
                          reservation_id=reservation_id, **identity)
            self._write(intent_path, intent)
            self._write(directory / 'dispatched.json', dict(schema='sciel-method-dispatched/1',
                request_sha256=digest, reservation_id=reservation_id,
                at=datetime.now(timezone.utc).isoformat()))
            started = time.monotonic()
            cpu_started = time.thread_time()
            returned = False
            value, actual = None, None
            try:
                value, actual = function(reservation_id)
                returned = True
                if not isinstance(value, dict):
                    raise ValueError('call result must be a JSON object')
                canonical_bytes(value)
                status = value.get('status', 'completed')
                status = 'completed' if status == 'ok' else status
                if status not in ('completed', 'failed', 'unknown'):
                    raise ValueError('call result has an unsupported status')
                actual = ResourceLedger._actual({'bounds': bounds, 'actual': {}}, actual, status)
            except BaseException as exc:
                name = type(exc).__name__
                if not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,79}', name):
                    name = 'Exception'
                not_dispatched = isinstance(exc, RequestNotDispatched)
                status = 'failed' if not_dispatched else 'unknown'
                known = {}
                if returned and isinstance(actual, dict):
                    for key, amount in actual.items():
                        try:
                            known.update(_vector({key: amount}, partial=True))
                        except ValueError:
                            pass
                if returned and isinstance(value, dict):
                    try:
                        value = detached(value)
                    except (ValueError, TypeError, RecursionError):
                        value = {}
                else:
                    value = {}
                value.update(status=status, reason=name, dispatched=not not_dispatched)
                actual = {'wall_seconds': round(time.monotonic() - started, 9),
                          'cpu_seconds': round(time.thread_time() - cpu_started, 9),
                          'steps': bounds['steps'], **known}
                if not_dispatched:
                    actual.update(dollars=0, gpu_seconds=0, storage_bytes=0)
            receipt = dict(schema='sciel-method-call/1', request_sha256=digest,
                reservation_id=reservation_id, key=key, account=account,
                category=category, result=value, actual=actual, status=status)
            try:
                self._write(result_path, receipt)
            except BaseException:
                self.ledger.settle(reservation_id, actual, status='unknown')
                raise BudgetUnknown('terminal receipt could not be persisted; reservation retained') from None
            return self._finish(receipt, replayed=False)

    def resolve_unknown(self, key, account, *, result, actual, reason, evidence_refs, status='failed'):
        """Explicit host audit; never retry or overwrite the original call.

        ``completed`` requires an externally recovered actual result. Abandoning
        a call uses ``failed``; resolving charges is not evidence of success.
        """
        if status not in ('completed', 'failed') or not isinstance(result, dict):
            raise ValueError('resolution requires a terminal result object')
        result = detached(result)
        terminal_status = 'completed' if result.get('status') == 'ok' else result.get('status')
        if terminal_status != status:
            raise ValueError('resolution result status must match its terminal status')
        audit = ResourceLedger._resolution(reason, evidence_refs)
        hash_list(evidence_refs, 'resolution evidence')
        recorded = {event['data']['id'] for event in self.registry.events()
                    if event['kind'] == 'artifact_registered'}
        if any(ref not in recorded for ref in evidence_refs):
            raise ValueError('resolution evidence must be registered artifacts')
        with self._locked(key, account) as directory:
            intent = self._read(directory / 'intent.json')
            original_path = directory / 'receipt.json'
            original = self._read(original_path) if original_path.exists() else None
            if original is not None:
                self._bound(original, intent)
                if original['status'] != 'unknown':
                    raise ValueError('only unknown calls can be resolved')
            rows = {row['id']: row for row in self.ledger.snapshot(account)['reservations']}
            row = rows[intent['reservation_id']]
            if row['status'] == 'reserved':
                self.ledger.settle(row['id'], original['actual'] if original else {}, status='unknown')
                row = {**row, 'status': 'unknown', 'actual': original['actual'] if original else {}}
            actual = ResourceLedger._actual(row, actual, status, resolve=True)
            receipt = dict(schema='sciel-method-call-resolution/1',
                **{key: intent[key] for key in ('request_sha256', 'reservation_id', 'key', 'account', 'category')},
                result=result, actual=actual, status=status, audit=audit,
                original_receipt_sha256=original['sha256'] if original else None)
            self._write(directory / 'resolution.json', receipt)
            return self._finish(receipt, replayed=True, resolution=audit)
