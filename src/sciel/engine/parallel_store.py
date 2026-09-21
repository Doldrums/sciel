"""Durable, single-writer scientific commits for cooperative parallel waves.

External-call receipts live in the wave directory and are never deleted here.
An active wave replays from its published baseline; a prepared wave completes
its exact journal append/checkpoint replacement before Engine construction.
Callers must own RunDirectoryLock during begin/commit and must not write the
root checkpoint while a transaction is active. ``recover`` acquires that lock.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import uuid

from sciel.run_log import RunLog
from . import checkpoint
from .run_lock import RunDirectoryLock

SCHEMA = 'sciel-wave-transaction/2'
_METADATA_EVENTS = {'run_start', 'checkpoint_restore', 'checkpoint', 'run_end'}
_LOG_METADATA = ('run_id', 'segment_id', 'event_seq', '_segment_started',
                 '_last_record', '_llm_context', '_llm_attempt')


class WaveTransactionError(RuntimeError):
    """An unpublished wave or its committed prefix cannot be authenticated."""


def _sync_dir(path):
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _directory(path):
    path = Path(path)
    if path.is_symlink():
        raise WaveTransactionError('transaction directory is a symlink')
    path.mkdir(parents=True, exist_ok=True)
    if not path.is_dir():
        raise WaveTransactionError('transaction path is not a directory')
    return path


def _regular(path, *, missing=False):
    path = Path(path)
    if path.is_symlink() or (path.exists() and not path.is_file()):
        raise WaveTransactionError('transaction input is not a regular file')
    if not missing and not path.exists():
        raise WaveTransactionError('transaction input is missing: ' + path.name)
    return path


def _atomic_bytes(path, data):
    _regular(path, missing=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with open(temporary, 'xb') as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _sync_dir(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path, value):
    _atomic_bytes(path, json.dumps(value, sort_keys=True, allow_nan=False).encode())


def _read_json(path):
    try:
        return json.loads(_regular(path).read_bytes())
    except (ValueError, UnicodeError) as exc:
        raise WaveTransactionError('invalid transaction JSON: ' + path.name) from exc


def _copy(source, target):
    _regular(source)
    _regular(target, missing=True)
    temporary = target.with_name(target.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with open(source, 'rb') as reader, open(temporary, 'xb') as writer:
            shutil.copyfileobj(reader, writer, 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        os.replace(temporary, target)
        _sync_dir(target.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _digest(path, *, missing=False):
    if missing and not path.exists():
        return None
    _regular(path)
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _wave_id(value):
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        value = str(value)
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,95}', value):
        raise ValueError('invalid parallel wave ID')
    return value


def _file(directory, name):
    if not isinstance(name, str) or not re.fullmatch(r'[a-f0-9]{32}\.(baseline|journal|candidate)\.json[l]?', name):
        raise WaveTransactionError('unsafe transaction filename')
    return _regular(directory / name)


def _manifest(directory):
    value = _read_json(directory / 'transaction.json')
    if (not isinstance(value, dict) or value.get('schema') != SCHEMA
            or value.get('wave_id') != directory.name.removeprefix('wave-')
            or value.get('phase') not in {'initializing', 'active', 'prepared', 'committed'}):
        raise WaveTransactionError('invalid wave transaction manifest')
    if value['phase'] == 'initializing':
        return value
    for key in ('baseline_sha256', 'base_journal_sha256'):
        if not isinstance(value.get(key), str) or not re.fullmatch('[a-f0-9]{64}', value[key]):
            raise WaveTransactionError('invalid transaction digest')
    if type(value.get('base_journal_bytes')) is not int or value['base_journal_bytes'] < 0:
        raise WaveTransactionError('invalid transaction journal boundary')
    return value


def _authenticate(directory, manifest, field, digest):
    path = _file(directory, manifest[field])
    if _digest(path) != manifest[digest]:
        raise WaveTransactionError('transaction ' + field + ' hash mismatch')
    return path


def _base_digest(main, boundary):
    """Authenticate history once at recovery, never on normal wave commits."""
    digest = hashlib.sha256()
    with open(_regular(main), 'rb') as stream:
        remaining = boundary
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                raise WaveTransactionError('journal baseline prefix is incomplete')
            digest.update(chunk)
            remaining -= len(chunk)
    return digest


def _tail_matches(main, delta, boundary, length):
    if not length:
        return True
    with open(main, 'rb') as actual, open(delta, 'rb') as expected:
        actual.seek(boundary)
        while length:
            size = min(length, 1024 * 1024)
            if actual.read(size) != expected.read(size):
                return False
            length -= size
    return True


def _baseline(directory, manifest, run_dir):
    baseline = _authenticate(directory, manifest, 'baseline', 'baseline_sha256')
    authoritative = run_dir / 'checkpoint.json'
    current = _digest(authoritative, missing=True)
    if current == manifest['baseline_sha256']:
        return
    if current != manifest.get('previous_checkpoint_sha256'):
        raise WaveTransactionError('authoritative checkpoint changed during active wave')
    _copy(baseline, authoritative)


def _active_prefix(directory, manifest, run_dir):
    _file(directory, manifest['journal'])
    main = _regular(run_dir / 'sciel_events.jsonl')
    boundary = manifest['base_journal_bytes']
    if _base_digest(main, boundary).hexdigest() != manifest['base_journal_sha256']:
        raise WaveTransactionError('active wave baseline journal hash mismatch')
    with open(main, 'rb') as stream:
        stream.seek(boundary)
        for line in stream:
            try:
                record = json.loads(line)
            except (ValueError, UnicodeError) as exc:
                raise WaveTransactionError('unpublished wave has malformed journal suffix') from exc
            if not line.endswith(b'\n') or not isinstance(record, dict) or record.get('event') not in _METADATA_EVENTS:
                raise WaveTransactionError('unpublished wave has scientific journal suffix')


def _finish_prepared(directory, manifest, run_dir, *, base_hash=None):
    delta = _authenticate(directory, manifest, 'journal', 'delta_sha256')
    candidate = _authenticate(directory, manifest, 'candidate', 'candidate_sha256')
    _authenticate(directory, manifest, 'baseline', 'baseline_sha256')
    boundary, total = manifest['base_journal_bytes'], manifest['journal_bytes']
    delta_bytes = delta.stat().st_size
    if delta_bytes != manifest['delta_bytes'] or total != boundary + delta_bytes:
        raise WaveTransactionError('prepared journal length mismatch')
    main = _regular(run_dir / 'sciel_events.jsonl')
    size = main.stat().st_size
    if not boundary <= size <= total or not _tail_matches(main, delta, boundary, size - boundary):
        raise WaveTransactionError('prepared wave conflicts with authoritative journal')
    # The live writer checked its unchanged inode/stat and cached prefix hash.
    # Recovery has no such process-local proof and streams the prefix once.
    digest = base_hash.copy() if base_hash is not None else _base_digest(main, boundary)
    if digest.hexdigest() != manifest['base_journal_sha256']:
        raise WaveTransactionError('prepared wave baseline journal hash mismatch')
    with open(delta, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != manifest['journal_sha256']:
        raise WaveTransactionError('prepared full journal hash mismatch')
    authoritative = run_dir / 'checkpoint.json'
    current = _digest(authoritative, missing=True)
    if current not in {manifest['baseline_sha256'], manifest['candidate_sha256']}:
        raise WaveTransactionError('prepared wave conflicts with authoritative checkpoint')
    candidate_state = _read_json(candidate)
    anchor = candidate_state.get('provenance', {}).get('journal_anchor')
    if not isinstance(anchor, dict) or anchor.get('byte_offset') != total or anchor.get('prefix_sha256') != manifest['journal_sha256']:
        raise WaveTransactionError('candidate checkpoint does not bind prepared journal')
    with open(delta, 'rb') as reader, open(main, 'ab', buffering=0) as writer:
        reader.seek(size - boundary)
        for chunk in iter(lambda: reader.read(1024 * 1024), b''):
            remaining = memoryview(chunk)
            while remaining:
                written = writer.write(remaining)
                if written is None or written <= 0:
                    raise OSError('short parallel journal append')
                remaining = remaining[written:]
        os.fsync(writer.fileno())
        if os.fstat(writer.fileno()).st_size != total:
            raise WaveTransactionError('journal changed during prepared append')
    if current != manifest['candidate_sha256']:
        _copy(candidate, authoritative)
    committed = dict(manifest, phase='committed')
    _write_json(directory / 'transaction.json', committed)
    return committed


def _cleanup_committed(directory):
    """Remove disposable checkpoints/delta only after the commit marker is durable.

    This reserved filename namespace includes copies from interrupted rebases;
    immutable call receipts/jobs/results and manifest hashes remain untouched.
    A crash during cleanup is harmless and the next recovery finishes it.
    """
    owned = re.compile(r'[a-f0-9]{32}\.(baseline|journal|candidate)\.json[l]?'
                       r'(?:\.(?:[a-f0-9]{32}\.)?tmp)?')
    removed = False
    for path in directory.iterdir():
        if owned.fullmatch(path.name):
            _regular(path)
            path.unlink()
            removed = True
    if removed:
        _sync_dir(directory)


def recover(run_dir):
    """Finish a prepared commit, or authenticate an active replay baseline.

    Call before constructing an Engine. This never dispatches/retries external
    work and never discards call receipts. Returns None if no wave exists.
    """
    run_dir = Path(run_dir)
    pointer = run_dir / '.parallel' / 'current.json'
    if not pointer.exists():
        return None
    lock = RunDirectoryLock(run_dir)
    try:
        _directory(run_dir / '.parallel')
        current = _read_json(pointer)
        if not isinstance(current, dict) or current.get('schema') != SCHEMA:
            raise WaveTransactionError('invalid current wave pointer')
        directory = _directory(pointer.parent / ('wave-' + _wave_id(current.get('wave_id'))))
        manifest = _manifest(directory)
        if manifest['phase'] == 'prepared':
            manifest = _finish_prepared(directory, manifest, run_dir)
        elif manifest['phase'] == 'active':
            _active_prefix(directory, manifest, run_dir)
            _baseline(directory, manifest, run_dir)
        if manifest['phase'] == 'committed':
            _cleanup_committed(directory)
        return {'phase': manifest['phase'], 'wave_id': manifest['wave_id'], 'directory': str(directory)}
    finally:
        lock.close()


def _metadata(log):
    return {name: deepcopy(getattr(log, name)) for name in _LOG_METADATA}


def begin(engine, wave_id):
    return WaveTransaction(engine, wave_id)


class WaveTransaction:
    def __init__(self, engine, wave_id):
        if getattr(engine, '_parallel_transaction', None) is not None:
            raise WaveTransactionError('nested parallel wave transaction')
        self.engine, self.wave_id = engine, _wave_id(wave_id)
        self.run_dir = Path(engine.run_dir)
        root = _directory(self.run_dir / '.parallel')
        self.directory = _directory(root / ('wave-' + self.wave_id))
        self._restored = False
        self._committed = False
        self._original = None
        manifest_path = self.directory / 'transaction.json'
        if (root / 'current.json').exists():
            pointer = _read_json(root / 'current.json')
            previous = _directory(root / ('wave-' + _wave_id(pointer.get('wave_id'))))
            previous_state = _manifest(previous)
            if previous != self.directory and previous_state['phase'] != 'committed':
                raise WaveTransactionError('another wave remains unpublished')
        if manifest_path.exists():
            previous = _manifest(self.directory)
            if previous['phase'] == 'prepared':
                raise WaveTransactionError('prepared wave must recover before Engine construction')
            if previous['phase'] == 'committed':
                raise WaveTransactionError('wave ID already committed')
            if previous['phase'] == 'active':
                _active_prefix(self.directory, previous, self.run_dir)
                _baseline(self.directory, previous, self.run_dir)
        else:
            _write_json(manifest_path, {'schema': SCHEMA, 'wave_id': self.wave_id, 'phase': 'initializing'})
        _write_json(root / 'current.json', {'schema': SCHEMA, 'wave_id': self.wave_id})
        main = self.run_dir / 'sciel_events.jsonl'
        if Path(engine.log.path).resolve() != main.resolve():
            raise WaveTransactionError('engine journal is not the authoritative journal')
        anchor = engine.log.anchor()
        if not isinstance(anchor, dict):
            raise WaveTransactionError('parallel wave requires an initialized journal segment')
        generation = uuid.uuid4().hex
        baseline = self.directory / (generation + '.baseline.json')
        staged_path = self.directory / (generation + '.journal.jsonl')
        previous_sha = _digest(self.run_dir / 'checkpoint.json', missing=True)
        checkpoint.save(engine, 'parallel_baseline', path=baseline)
        with open(baseline, 'rb') as stream:
            os.fsync(stream.fileno())
        _atomic_bytes(staged_path, b'')
        manifest = {'schema': SCHEMA, 'wave_id': self.wave_id, 'phase': 'active',
            'baseline': baseline.name, 'baseline_sha256': _digest(baseline),
            'previous_checkpoint_sha256': previous_sha,
            'journal': staged_path.name, 'base_journal_bytes': anchor['byte_offset'],
            'base_journal_sha256': anchor['prefix_sha256']}
        engine.log._check_unchanged()
        _write_json(manifest_path, manifest)
        _baseline(self.directory, manifest, self.run_dir)
        self.manifest = manifest
        self._saved_checkpoint = deepcopy(engine._saved_checkpoint)
        staged = RunLog(str(staged_path))
        staged.__dict__.update(_metadata(engine.log))
        staged._prefix_hash = engine.log._prefix_hash.copy()
        staged._offset = staged._prefix_bytes = anchor['byte_offset']
        self._original = engine.log.__dict__.copy()
        engine.log.__dict__.clear()
        engine.log.__dict__.update(staged.__dict__)
        engine._parallel_transaction = self

    def _check_original(self):
        original = self._original
        opened = RunLog._stat_signature(os.fstat(original['_fh'].fileno()))
        if (opened != original['_signature']
                or opened != RunLog._stat_signature(os.stat(original['path']))
                or opened[2] != self.manifest['base_journal_bytes']
                or original['_prefix_hash'].hexdigest() != self.manifest['base_journal_sha256']):
            raise WaveTransactionError('authoritative journal changed while wave was staged')

    def commit(self):
        if self._committed:
            return str(self.run_dir / 'checkpoint.json')
        if self._restored:
            raise WaveTransactionError('cannot commit restored wave')
        engine = self.engine
        self._check_original()
        if _digest(self.run_dir / 'checkpoint.json') != self.manifest['baseline_sha256']:
            raise WaveTransactionError('authoritative checkpoint changed during wave')
        anchor = engine.log.anchor()
        candidate = self.directory / (uuid.uuid4().hex + '.candidate.json')
        checkpoint.save(engine, 'parallel_wave', path=candidate)
        with open(candidate, 'rb') as stream:
            os.fsync(stream.fileno())
        prepared = dict(self.manifest, phase='prepared', candidate=candidate.name,
            candidate_sha256=_digest(candidate), journal_bytes=anchor['byte_offset'],
            journal_sha256=anchor['prefix_sha256'],
            delta_bytes=engine.log._offset - engine.log._prefix_bytes,
            delta_sha256=_digest(Path(engine.log.path)))
        _write_json(self.directory / 'transaction.json', prepared)
        self.manifest = prepared
        self._check_original()
        self.manifest = _finish_prepared(self.directory, prepared, self.run_dir,
                                        base_hash=self._original['_prefix_hash'])
        updated = dict(self._original, **_metadata(engine.log))
        updated.update(_prefix_hash=engine.log._prefix_hash.copy(),
                       _offset=engine.log._offset, _prefix_bytes=0)
        engine.log.close()
        engine.log.__dict__.clear()
        engine.log.__dict__.update(updated)
        engine.log._signature = engine.log._file_signature()
        if engine.log._signature[2] != engine.log._offset:
            raise WaveTransactionError('committed journal length changed')
        engine._parallel_transaction = None
        self._committed = True
        _cleanup_committed(self.directory)
        return str(self.run_dir / 'checkpoint.json')

    def restore_log(self):
        """Unbind unpublished staged events; keep every WAL/stage artifact.

        In-memory scientific state is not rolled back: the caller must exit and
        load the authenticated baseline before replaying the wave.
        """
        if self._restored or self._committed or self._original is None:
            return
        self.engine.log.close()
        self.engine.log.__dict__.clear()
        self.engine.log.__dict__.update(self._original)
        self.engine._saved_checkpoint = deepcopy(self._saved_checkpoint)
        self.engine._parallel_transaction = None
        self._restored = True
