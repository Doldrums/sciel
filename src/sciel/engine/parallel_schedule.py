"""Immutable coordinator resume order, separate from disposable wave journals."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import uuid

SCHEMA = 'sciel-parallel-resume-order/1'


def _bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
                      allow_nan=False).encode('utf-8')


def _hash(value):
    return hashlib.sha256(_bytes(value)).hexdigest()


def _read(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Resume-order input is not a regular file')
    value = json.loads(path.read_bytes())
    if (type(value) is not dict or set(value) != {'payload', 'sha256'}
            or _hash(value['payload']) != value['sha256']):
        raise ValueError('Resume-order checksum mismatch')
    return value


def _exclusive(path, value):
    """Install a complete fsynced record without replacing existing evidence."""
    if path.exists() or path.is_symlink():
        raise ValueError('Resume-order record already exists')
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
    try:
        with temporary.open('xb') as stream:
            stream.write(_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)  # Exclusive, atomic installation on the same filesystem.
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


class ResumeOrder:
    def __init__(self, wave_directory, identity, run_index, job_ids):
        if (type(identity) is not dict or set(identity) != {'run_id', 'wave', 'engine_contract'}
                or not all(type(identity[key]) is str and identity[key]
                           for key in ('run_id', 'engine_contract'))
                or type(identity['wave']) is not int or identity['wave'] < 0):
            raise ValueError('Invalid resume-order run identity')
        self._run_index = run_index
        if type(run_index) is not int or run_index < 0:
            raise ValueError('Invalid resume-order phase index')
        if len(set(job_ids)) != len(job_ids) or any(type(key) is not str or not key for key in job_ids):
            raise ValueError('Invalid resume-order job identities')
        parent = Path(wave_directory) / 'resume-order'
        self.directory = parent / f'run-{run_index:06d}'
        if parent.is_symlink() or self.directory.is_symlink():
            raise ValueError('Resume-order directory is a symlink')
        declarations = Path(wave_directory) / 'resume-order-phases'
        if declarations.is_symlink():
            raise ValueError('Resume-order phase index is a symlink')
        declarations.mkdir(parents=True, exist_ok=True)
        self.declaration_path = declarations / f'phase-{run_index:06d}.json'
        declared = self.declaration_path.exists()
        if declared and (not (self.directory / 'header.json').is_file()
                         or not (self.directory / 'head.json').is_file()):
            raise ValueError('Resume-order declared phase/header/head is missing')
        if not declared and list(self.directory.glob('turn-*.json')):
            raise ValueError('Resume-order phase declaration is missing')
        self.directory.mkdir(parents=True, exist_ok=True)
        header = {'schema': SCHEMA, 'identity': deepcopy(identity),
                  'run_index': run_index, 'job_ids': list(job_ids)}
        path = self.directory / 'header.json'
        existing = list(self.directory.glob('turn-*.json'))
        if not path.exists():
            if existing:
                raise ValueError('Resume-order header is missing')
            _exclusive(path, {'payload': header, 'sha256': _hash(header)})
        saved = _read(path)
        if saved['payload'] != header:
            raise ValueError('Resume-order identity/jobs/phase differ on replay')
        self.header_sha256 = saved['sha256']
        predecessor = None
        if run_index:
            previous = declarations / f'phase-{run_index - 1:06d}.json'
            if not previous.is_file():
                raise ValueError('Resume-order previous phase declaration is missing')
            predecessor = _read(previous)['sha256']
        declaration = {'schema': SCHEMA, 'run_index': run_index,
                       'header_sha256': self.header_sha256, 'previous_sha256': predecessor}
        if not declared:
            if list(declarations.glob('phase-*.json')) and any(
                    p.name > self.declaration_path.name for p in declarations.glob('phase-*.json')):
                raise ValueError('Resume-order phase index has a gap')
        elif _read(self.declaration_path)['payload'] != declaration:
            raise ValueError('Resume-order phase declaration differs')
        self.declaration_sha256 = _hash(declaration)
        self.head = saved['sha256']
        self.turns = []
        for index, path in enumerate(sorted(existing)):
            if path.name != f'turn-{index:08d}.json':
                raise ValueError('Resume-order sequence has a gap or invalid filename')
            saved = _read(path)
            turn = saved['payload']
            if (type(turn) is not dict or set(turn) != {'schema', 'sequence', 'previous_sha256',
                    'job_id', 'call_key', 'kind', 'request_sha256', 'receipt_sha256'}
                    or turn['schema'] != SCHEMA or turn['sequence'] != index
                    or turn['previous_sha256'] != self.head or turn['job_id'] not in job_ids
                    or type(turn['call_key']) is not str or not turn['call_key'].startswith(turn['job_id'] + ':')
                    or type(turn['kind']) is not str
                    or any(type(turn[k]) is not str or not re.fullmatch('[a-f0-9]{64}', turn[k])
                           for k in ('request_sha256', 'receipt_sha256'))):
                raise ValueError('Resume-order turn binding differs')
            self.turns.append(turn)
            self.head = saved['sha256']
        self.cursor = 0
        self.tip_path = self.directory / 'head.json'
        if self.tip_path.is_symlink():
            raise ValueError('Resume-order head is a symlink')
        if self.tip_path.exists():
            tip = _read(self.tip_path)['payload']
            if (type(tip) is not dict or set(tip) != {'header_sha256', 'count', 'head_sha256'}
                    or tip['header_sha256'] != _hash(header) or type(tip['count']) is not int
                    or tip['count'] < 0 or tip['count'] > len(self.turns)):
                raise ValueError('Resume-order head is invalid or its tail was truncated')
            known_head = (_hash(header) if tip['count'] == 0 else
                          _hash(self.turns[tip['count'] - 1]))
            if tip['head_sha256'] != known_head or len(self.turns) > tip['count'] + 1:
                raise ValueError('Resume-order head differs from its turn chain')
        elif self.turns:
            raise ValueError('Resume-order head is missing')
        self.header_sha256 = _hash(header)
        # At most one complete turn may precede its head update after a crash;
        # no scientific continuation has consumed it until both writes finish.
        self._save_head()
        if not declared:
            _exclusive(self.declaration_path, {'payload': declaration, 'sha256': _hash(declaration)})

    def _save_head(self):
        from .parallel_calls import durable_json
        payload = {'header_sha256': self.header_sha256, 'count': len(self.turns),
                   'head_sha256': self.head}
        durable_json(self.tip_path, {'payload': payload, 'sha256': _hash(payload)})

    def bind_call(self, directory, key, request_hash):
        path = Path(directory) / 'resume-order.json'
        payload = {'schema': SCHEMA, 'run_index': self._run_index,
                   'header_sha256': self.header_sha256,
                   'phase_sha256': self.declaration_sha256,
                   'call_key': key, 'request_sha256': request_hash}
        if path.exists():
            if _read(path)['payload'] != payload:
                raise ValueError('Resume-order call belongs to a different phase')
        else:
            _exclusive(path, {'payload': payload, 'sha256': _hash(payload)})

    @staticmethod
    def verify_call(wave_directory, directory, intent):
        path = Path(directory) / 'resume-order.json'
        if not path.is_file():
            raise ValueError('Resume-order physical call binding is missing')
        binding = _read(path)['payload']
        if (type(binding) is not dict or set(binding) != {'schema', 'run_index', 'header_sha256',
                'phase_sha256', 'call_key', 'request_sha256'} or binding['schema'] != SCHEMA
                or type(binding['run_index']) is not int or binding['run_index'] < 0
                or binding['call_key'] != intent['key']
                or binding['request_sha256'] != intent['request_sha256']):
            raise ValueError('Resume-order physical call binding differs')
        index = binding['run_index']
        phase = Path(wave_directory) / 'resume-order-phases' / f'phase-{index:06d}.json'
        header = Path(wave_directory) / 'resume-order' / f'run-{index:06d}' / 'header.json'
        if not phase.is_file() or not header.is_file():
            raise ValueError('Resume-order physical call phase is missing')
        declared, saved = _read(phase), _read(header)
        if (declared['sha256'] != binding['phase_sha256']
                or saved['sha256'] != binding['header_sha256']
                or declared['payload']['header_sha256'] != saved['sha256']):
            raise ValueError('Resume-order physical call phase differs')
        return binding

    @staticmethod
    def binding(job_id, operation, receipt):
        # Abandonment is an independently authenticated recovery authorization;
        # it never changes the immutable physical receipt identity.
        physical = {key: value for key, value in receipt.items() if key != 'abandonment'}
        return {'job_id': job_id, 'call_key': operation['key'], 'kind': operation['kind'],
                'request_sha256': _hash({'kind': operation['kind'], 'request': operation['request']}),
                'receipt_sha256': _hash(physical)}

    def consume(self, job_id, operation, receipt):
        binding = self.binding(job_id, operation, receipt)
        if self.cursor < len(self.turns):
            turn = self.turns[self.cursor]
            if any(turn[key] != value for key, value in binding.items()):
                raise ValueError('Resume-order differs from pending call/physical receipt')
        else:
            turn = {'schema': SCHEMA, 'sequence': self.cursor,
                    'previous_sha256': self.head, **binding}
            envelope = {'payload': turn, 'sha256': _hash(turn)}
            _exclusive(self.directory / f'turn-{self.cursor:08d}.json', envelope)
            self.turns.append(turn)
            self.head = envelope['sha256']
            self._save_head()
        self.cursor += 1
