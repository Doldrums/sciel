"""Checkpoint-owned research cells, independent of candidate/island populations.

Artifacts are immutable provenance, not a second coordinator or a publication
index. Only checkpoint heads grant visibility. The caller owns wave transactions,
scientific transition checks, execution, accounting and proof of recorded claims.
There is no lifetime cell/tool limit; only active work and individual records are
bounded. No method is imported or executed by this module.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path
import stat
import tempfile

from ..methods.models import canonical_bytes, content_hash, hash_list, require_hash

SCHEMA = 'sciel-research-cells/1'
GENESIS_SCHEMA = 'sciel-research-cell/1'
REVISION_SCHEMA = 'sciel-research-cell-revision/1'
MAX_ARTIFACT_BYTES = 256 * 1024
MAX_QUESTION_BYTES = 16384
MAX_EVENT_REFS = 256  # per revision, never a lifetime tool/evidence limit
_ROLES = {'creator', 'replayer', 'recipient'}


def _text(value, label, limit):
    if type(value) is not str or not value.strip() or len(value.encode('utf-8')) > limit:
        raise ValueError(f'{label} must be nonempty text within {limit} UTF-8 bytes')
    return value


def _island(value):
    if type(value) is int and 0 <= value <= 2**63 - 1:
        return value
    if (type(value) is str and value.strip() and len(value.encode('utf-8')) <= 128
            and all(ord(char) >= 32 for char in value)):
        return value
    raise ValueError('island must be a nonnegative integer or bounded identifier')


def _bounded_json(value, label, limit):
    if len(canonical_bytes(value)) > limit:
        raise ValueError(label + ' exceeds its per-record byte limit')
    return deepcopy(value)


def _roles(value, origin):
    if type(value) is not dict or set(value) - _ROLES or value.get('creator') != origin:
        raise ValueError('cell roles must retain the immutable creator')
    for role, island in value.items():
        _island(island)
        if role != 'creator' and str(island) == str(origin):
            raise ValueError(role + ' must be independent of the creator island')
    return deepcopy(value)


def _evidence(value):
    if type(value) is not list or len(value) > MAX_EVENT_REFS:
        raise ValueError('evidence must be a bounded list of typed references')
    result = []
    for row in value:
        if type(row) is not dict or set(row) - {'kind', 'id', 'tool_id'} or not {'kind', 'id'} <= set(row):
            raise ValueError('evidence reference requires kind and id, with optional tool_id')
        _text(row['kind'], 'evidence kind', 80)
        require_hash(row['id'], 'evidence id')
        if 'tool_id' in row:
            require_hash(row['tool_id'], 'evidence tool_id')
        if row in result:
            raise ValueError('duplicate evidence reference')
        result.append(deepcopy(row))
    return result


def _decode(raw):
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError('duplicate cell artifact JSON key')
            value[key] = item
        return value
    def constant(_):
        raise ValueError('nonfinite cell artifact JSON')
    value = json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)
    canonical_bytes(value)
    return value


class CellRegistry:
    """Pure scheduling state plus immutable research lineage artifacts.

    ``ensure_cell`` reuses a committed origin/question pair. Explicit ``cell_id``
    reuses a cell for another island, retaining its original question and origin.
    A full active set leaves a newly recorded cell dormant, not rejected. Reload
    a prior ``state()`` to roll back a wave; abandoned artifacts grant no rights.

    Revision ``tool_ids`` are exact existing source versions, not newly copied
    packages. Evidence references and phase/decision labels are recorded claims;
    the controller must authenticate receipts and enforce its lifecycle policy.
    """
    def __init__(self, run_dir, max_active=4):
        if type(max_active) is not int or max_active < 1:
            raise ValueError('max_active must be a positive integer')
        supplied = Path(run_dir).expanduser().absolute()
        if supplied.is_symlink():
            raise ValueError('cell run directory cannot be a symbolic link')
        supplied = supplied.parent.resolve() / supplied.name
        self.run_dir = supplied
        self.path = supplied / 'research_tools' / 'cells'
        self.genesis = self.path / 'genesis'
        self.revisions = self.path / 'revisions'
        for directory in (supplied, supplied / 'research_tools', self.path, self.genesis, self.revisions):
            self._refuse_links(directory)
            directory.mkdir(parents=True, exist_ok=True)
            if not directory.is_dir():
                raise ValueError('cell storage must be a directory')
        self.max_active = max_active
        self._state = {'schema': SCHEMA, 'max_active': max_active, 'cells': {}}
        self._questions = {}

    @staticmethod
    def _refuse_links(path):
        if any(item.is_symlink() for item in (path, *path.parents)):
            raise ValueError('cell storage refuses symbolic links')

    def _read(self, directory, ident):
        require_hash(ident, 'cell artifact id')
        target = directory / (ident + '.json')
        self._refuse_links(target)
        fd = os.open(target, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_ARTIFACT_BYTES:
                raise ValueError('cell artifact is oversized or not a regular file')
            with os.fdopen(fd, 'rb', closefd=False) as stream:
                raw = stream.read(MAX_ARTIFACT_BYTES + 1)
            after = os.fstat(fd)
            signature = lambda s: (s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
            if len(raw) != before.st_size or signature(before) != signature(after):
                raise ValueError('cell artifact changed during read')
        finally:
            os.close(fd)
        body = _decode(raw)
        if content_hash(body) != ident:
            raise ValueError('cell artifact hash mismatch')
        return body

    def _write(self, directory, body):
        raw = canonical_bytes(body)
        if len(raw) > MAX_ARTIFACT_BYTES:
            raise ValueError('cell artifact exceeds its per-record byte limit')
        ident = content_hash(body)
        target = directory / (ident + '.json')
        self._refuse_links(target)
        fd, temporary = tempfile.mkstemp(prefix='.cell-write-', dir=directory)
        try:
            with os.fdopen(fd, 'wb') as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, target)
            except FileExistsError:
                if canonical_bytes(self._read(directory, ident)) != raw:
                    raise ValueError('immutable cell artifact differs')
            directory_fd = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            Path(temporary).unlink(missing_ok=True)
        return ident

    @staticmethod
    def _question_key(origin, question):
        return content_hash({'origin_island': str(origin), 'question': question})

    @staticmethod
    def _validate_genesis(body):
        if (type(body) is not dict or set(body) != {'schema', 'origin_island', 'question', 'trigger'}
                or body.get('schema') != GENESIS_SCHEMA):
            raise ValueError('invalid cell genesis')
        _island(body['origin_island'])
        _text(body['question'], 'cell question', MAX_QUESTION_BYTES)
        trigger = body['trigger']
        if type(trigger) is not dict or len(trigger) < 2:
            raise ValueError('cell trigger requires kind and recorded evidence fields')
        _text(trigger.get('kind'), 'trigger kind', 80)
        _bounded_json(trigger, 'cell trigger', 32768)

    @staticmethod
    def _validate_revision(body, cell_id, origin):
        keys = {'schema', 'cell_id', 'previous_revision', 'ordinal', 'phase', 'tool_ids',
                'evidence', 'roles', 'summary', 'decision'}
        if (type(body) is not dict or set(body) != keys or body.get('schema') != REVISION_SCHEMA
                or body.get('cell_id') != cell_id):
            raise ValueError('invalid cell revision binding')
        if body['previous_revision'] is not None:
            require_hash(body['previous_revision'], 'previous revision')
        if type(body['ordinal']) is not int or body['ordinal'] < 1:
            raise ValueError('invalid cell revision ordinal')
        _text(body['phase'], 'cell phase', 80)
        tools = hash_list(body['tool_ids'], 'cell tool_ids')
        if len(tools) > MAX_EVENT_REFS:
            raise ValueError('too many tools in a single cell revision')
        _evidence(body['evidence'])
        _roles(body['roles'], origin)
        if type(body['summary']) is not str or len(body['summary'].encode('utf-8')) > 8192:
            raise ValueError('cell summary must be text within 8192 UTF-8 bytes')
        _bounded_json(body['decision'], 'cell decision', 8192)

    def _visible(self, cell_id):
        require_hash(cell_id, 'cell id')
        if cell_id not in self._state['cells']:
            raise KeyError('cell is not visible in committed checkpoint state: ' + cell_id)
        return self._state['cells'][cell_id]

    def get(self, cell_id):
        entry = self._visible(cell_id)
        genesis = self._read(self.genesis, cell_id)
        self._validate_genesis(genesis)
        revision = None
        if entry['head_revision'] is not None:
            revision = self._read(self.revisions, entry['head_revision'])
            self._validate_revision(revision, cell_id, genesis['origin_island'])
        return {'id': cell_id, **deepcopy(genesis), **deepcopy(entry),
                'revision': revision, 'roles': deepcopy(revision['roles']) if revision else {'creator': genesis['origin_island']},
                'tool_ids': list(revision['tool_ids']) if revision else [],
                'phase': revision['phase'] if revision else 'proposed'}

    def all_cells(self):
        """Detached visible records; this does not scan disk or include orphans."""
        return [self.get(ident) for ident in self._state['cells']]

    @property
    def cell_count(self):
        return len(self._state['cells'])

    @property
    def active_count(self):
        """Count visible leases without copying lineage or evidence records."""
        return sum(entry['active'] for entry in self._state['cells'].values())

    def state(self):
        return deepcopy(self._state)

    def load_state(self, state):
        """Authenticate all reachable history before atomically replacing state."""
        if (type(state) is not dict or set(state) != {'schema', 'max_active', 'cells'}
                or state.get('schema') != SCHEMA or type(state.get('max_active')) is not int
                or state['max_active'] != self.max_active or type(state.get('cells')) is not dict):
            raise ValueError('cell checkpoint shape or active-work bound differs')
        canonical_bytes(state)
        questions, active = {}, 0
        for cell_id, entry in state['cells'].items():
            require_hash(cell_id, 'cell id')
            if (type(entry) is not dict or set(entry) != {'head_revision', 'active', 'affinities'}
                    or type(entry['active']) is not bool or type(entry['affinities']) is not list
                    or not entry['affinities']):
                raise ValueError('invalid cell checkpoint entry')
            affinities = [str(_island(item)) for item in entry['affinities']]
            if len(set(affinities)) != len(affinities):
                raise ValueError('duplicate cell affinity')
            genesis = self._read(self.genesis, cell_id)
            self._validate_genesis(genesis)
            if str(genesis['origin_island']) not in affinities:
                raise ValueError('cell origin is absent from affinities')
            key = self._question_key(genesis['origin_island'], genesis['question'])
            if key in questions:
                raise ValueError('duplicate origin/question cell in checkpoint')
            questions[key] = cell_id
            head, expected, seen = entry['head_revision'], None, set()
            if head is not None:
                require_hash(head, 'head revision')
            while head is not None:
                if head in seen:
                    raise ValueError('cyclic cell history')
                seen.add(head)
                revision = self._read(self.revisions, head)
                self._validate_revision(revision, cell_id, genesis['origin_island'])
                if expected is not None and revision['ordinal'] != expected:
                    raise ValueError('cell revision history is not contiguous')
                expected = revision['ordinal'] - 1
                head = revision['previous_revision']
            if expected not in (None, 0):
                raise ValueError('cell revision history is incomplete')
            active += entry['active']
        if active > self.max_active:
            raise ValueError('cell checkpoint exceeds active-work bound')
        self._state = deepcopy(state)
        self._questions = questions

    def ensure_cell(self, island, question, trigger, *, cell_id=None):
        island = _island(island)
        _text(question, 'cell question', MAX_QUESTION_BYTES)
        if cell_id is not None:
            existing = self.get(cell_id)
            if question != existing['question']:
                raise ValueError('explicit cell reuse must retain its original question')
        else:
            key = self._question_key(island, question)
            cell_id = self._questions.get(key)
            if cell_id is None:
                body = {'schema': GENESIS_SCHEMA, 'origin_island': island,
                        'question': question, 'trigger': deepcopy(trigger)}
                self._validate_genesis(body)
                cell_id = self._write(self.genesis, body)
                self._state['cells'][cell_id] = {'head_revision': None, 'active': False, 'affinities': [island]}
                self._questions[key] = cell_id
        entry = self._visible(cell_id)
        if str(island) not in {str(item) for item in entry['affinities']}:
            entry['affinities'].append(island)
        self.activate(cell_id)
        return self.get(cell_id)

    def activate(self, cell_id):
        entry = self._visible(cell_id)
        if entry['active']:
            return True
        if self.active_count >= self.max_active:
            return False
        entry['active'] = True
        return True

    def deactivate(self, cell_id):
        self._visible(cell_id)['active'] = False
        return self.get(cell_id)

    def revise(self, cell_id, *, phase, tool_ids=None, evidence=(), summary='', roles=None, decision=None):
        cell = self.get(cell_id)
        previous = cell['revision']
        body = {'schema': REVISION_SCHEMA, 'cell_id': cell_id,
                'previous_revision': cell['head_revision'], 'ordinal': previous['ordinal'] + 1 if previous else 1,
                'phase': phase, 'tool_ids': cell['tool_ids'] if tool_ids is None else list(tool_ids),
                'evidence': list(evidence), 'roles': cell['roles'] if roles is None else deepcopy(roles),
                'summary': summary, 'decision': deepcopy(decision)}
        self._validate_revision(body, cell_id, cell['origin_island'])
        revision_id = self._write(self.revisions, body)
        self._state['cells'][cell_id]['head_revision'] = revision_id
        return self.get(cell_id)

    def assign_role(self, cell_id, role, island):
        if role not in _ROLES:
            raise ValueError('unsupported cell role')
        island = _island(island)
        cell = self.get(cell_id)
        roles = {**cell['roles'], role: island}
        _roles(roles, cell['origin_island'])
        if roles == cell['roles']:
            return cell
        return self.revise(cell_id, phase='role_assignment', roles=roles,
                           decision={'role': role, 'island': island})

    def compact_context_cards(self, cell_ids=None, *, max_chars=4096):
        """Bounded presentation only; questions/evidence never grow into a card.

        Default selection is active checkpoint cells. Exact full questions and
        event evidence remain in immutable artifacts even when excerpts are tiny.
        """
        if type(max_chars) is not int or max_chars < 0:
            raise ValueError('max_chars must be a nonnegative integer')
        identifiers = cell_ids if cell_ids is not None else (
            ident for ident, entry in self._state['cells'].items() if entry['active'])
        pieces, remaining = [], max_chars
        for ident in identifiers:
            if remaining < 1:
                break
            cell = self.get(ident)
            revision = cell['revision'] or {}
            def excerpt(text, limit):
                return text if len(text) <= limit else text[:limit - 1] + '…'
            tools = cell['tool_ids'][:4]
            tool_text = ', '.join(tools) or 'none'
            if len(tools) < len(cell['tool_ids']):
                tool_text += f" (+{len(cell['tool_ids']) - len(tools)} pinned in revision)"
            # A long original question cannot evict the latest observation or
            # make this presentation grow with cumulative research history.
            card = (f"Cell {cell['id']} | {cell['phase']}\n"
                    f"Latest observation: {excerpt(revision.get('summary', ''), 512)}\n"
                    f"Question: {excerpt(cell['question'], 384)}\n"
                    f"Exact tools: {tool_text}\n"
                    f"Revision: {cell['head_revision'] or 'none'}")
            allowance = remaining - (2 if pieces else 0)
            if allowance <= 0:
                break
            if len(card) > allowance:
                card = card[:max(0, allowance - 1)] + '…'
            pieces.append(card)
            remaining -= len(card) + (2 if len(pieces) > 1 else 0)
        return '\n\n'.join(pieces)
