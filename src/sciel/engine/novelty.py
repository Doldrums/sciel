"""Whole-run source novelty, independent of evaluator scores and behavior features.

SHA-256 and structural AST equality are duplicate gates. Token-embedding cosine
neighbors and AST diffs are explanations only, never grounds for rejection. AST
equality is syntactic equality, not a proof of equivalent runtime behavior.
"""
from __future__ import annotations

import ast
from array import array
from copy import deepcopy
import difflib
import hashlib
import heapq
import json
import sys
from threading import RLock

from ..components.code_novelty import DESCRIPTOR_VERSION, _descriptor

SCHEMA = 'sciel-candidate-novelty/1'
REPRESENTATION = DESCRIPTOR_VERSION
AST_VERSION = 'python-ast-fields-constants/1'
_LOCATIONS = {'lineno', 'col_offset', 'end_lineno', 'end_col_offset'}
_DIFF_CHARS = 12000
_DIFF_LINES = 160


def _sha(text):
    return hashlib.sha256(text.encode('utf-8', errors='surrogatepass')).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def _structural_ast(code):
    """An iterative AST dump retaining every non-location field and constant.

    Explicit closing markers preserve structure even beyond the indentation cap.
    TypeIgnore.lineno is an AST *field*, so include_attributes=False alone would
    not make type-ignore comments independent of their source line numbers.
    """
    try:
        root = ast.parse(code, type_comments=True)
    except (SyntaxError, ValueError, RecursionError, UnicodeError) as exc:
        return None, type(exc).__name__
    lines, stack = [], [(root, '', 0)]
    while stack:
        value, label, depth = stack.pop()
        indent = '  ' * min(depth, 12)
        if isinstance(value, ast.AST):
            lines.append(indent + label + type(value).__name__ + '(')
            stack.append((')', '', depth))
            fields = [(name, child) for name, child in ast.iter_fields(value) if name not in _LOCATIONS]
            stack.extend((child, name + '=', depth + 1) for name, child in reversed(fields))
        elif isinstance(value, list):
            lines.append(indent + label + '[')
            stack.append((']', '', depth))
            stack.extend((child, '', depth + 1) for child in reversed(value))
        elif label == '' and value in (')', ']'):
            lines.append(indent + value)
        else:
            # Type tags distinguish bool/int/float; repr keeps strings, bytes,
            # docstrings and all constant values without multiline ambiguity.
            lines.append(indent + label + type(value).__name__ + ':' + repr(value))
    return '\n'.join(lines), None


def _identifier(value):
    if type(value) is not str or not value or len(value) > 512:
        raise ValueError('program ID must be a nonempty string of at most 512 characters')
    return value


def _island(value):
    if value is None:
        return None
    if not ((type(value) is int and value >= 0)
            or (type(value) is str and value and len(value) <= 256)):
        raise ValueError('island must be a nonnegative integer, nonempty string or None')
    return value


def _evaluation(value):
    if value is not None and type(value) is not bool:
        raise ValueError('evaluation_ok must be a boolean or None')
    return value


def _ast_diff(before, after, program_id):
    if before is None or after is None:
        return None, False
    pieces, size = [], 0
    for index, line in enumerate(difflib.unified_diff(
            before.splitlines(), after.splitlines(), fromfile=program_id,
            tofile='candidate', n=2, lineterm='')):
        if index >= _DIFF_LINES:
            return ''.join(pieces), True
        if size + len(line) + 1 > _DIFF_CHARS:
            available = _DIFF_CHARS - size
            if available > 0:
                pieces.append((line + '\n')[:available])
            return ''.join(pieces), True
        pieces.append(line + '\n')
        size += len(line) + 1
    return ''.join(pieces), False


class CandidateNoveltyRegistry:
    """All registered evaluation attempts; population eviction never removes any.

    inspect is pure. Register a unique evaluation ID before dispatch, finalize
    that same ID with evaluation_ok, then bind admitted program IDs as aliases.
    A failed attempt may be retried under a NEW evaluation ID. Unknown/pending
    attempts block automatic and explicit remeasurement until reconciled.
    """

    def __init__(self, neighbors=3):
        if type(neighbors) is not int or not 1 <= neighbors <= 10:
            raise ValueError('neighbors must be an integer in [1, 10]')
        self.neighbors = neighbors
        self._sources, self._vectors, self._records, self._ids = {}, {}, {}, {}
        self._by_source, self._by_ast = {}, {}
        self._reservation_lock = RLock()

    def __deepcopy__(self, memo):
        # Checkpoint validation works on a detached registry. Locks are process
        # resources, never checkpoint data or shared between those copies.
        with self._reservation_lock:
            restored = type(self)(self.neighbors)
            memo[id(self)] = restored
            restored.__dict__.update({key: deepcopy(value, memo)
                for key, value in self.__dict__.items() if key != '_reservation_lock'})
            return restored

    @property
    def identity(self):
        return dict(schema=SCHEMA, representation=REPRESENTATION, ast_version=AST_VERSION,
                    python_ast_version=f'{sys.version_info.major}.{sys.version_info.minor}',
                    neighbors=self.neighbors)

    @staticmethod
    def _program_id(record):
        return record['aliases'][-1] if record['aliases'] else record['id']

    def reserve(self, code, evaluation_id, island=None, *, remeasurement=False):
        """Atomically inspect and claim a final source before evaluator dispatch.

        Concurrent calls to this method cannot both claim the same SHA or AST.
        Other mutations remain coordinator-owned. A pending reservation is kept
        across checkpoints: unknown outcomes must be reconciled, never released
        as a failed evaluation just to permit another call.
        """
        evaluation_id, island = _identifier(evaluation_id), _island(island)
        with self._reservation_lock:
            if evaluation_id in self._ids:
                raise ValueError('evaluation ID already registered; use a new evaluation ID')
            inspection = self.inspect(code, remeasurement=remeasurement)
            if inspection['accept']:
                self.register(code, evaluation_id, island, evaluation_ok=None)
                inspection['reservation_id'] = evaluation_id
            return inspection

    def register(self, code, program_id, island=None, evaluation_ok=None):
        if type(code) is not str:
            raise TypeError('code must be a string')
        program_id, island = _identifier(program_id), _island(island)
        evaluation_ok = _evaluation(evaluation_ok)
        source_sha = _sha(code)
        existing = self._records.get(self._ids.get(program_id))
        if existing is not None:
            if existing['source_sha256'] != source_sha or self._sources[source_sha]['code'] != code:
                raise ValueError('program ID is already bound to different source')
            previous = existing['evaluation_ok']
            if previous is not None and evaluation_ok is not None and previous != evaluation_ok:
                raise ValueError('a completed evaluation cannot be rewritten; use a new evaluation ID')
            if evaluation_ok is not None:
                existing['evaluation_ok'] = evaluation_ok
            if island is not None and island not in existing['islands']:
                existing['islands'].append(island)
            return deepcopy(existing)
        if source_sha not in self._sources:
            dump, _ = _structural_ast(code)
            source = dict(source_sha256=source_sha, ast_sha256=_sha(dump) if dump is not None else None, code=code)
            try:
                vector = array('d', _descriptor(code))
            except UnicodeError:
                vector = array('d', [0.0] * 512)
            self._sources[source_sha] = source
            self._vectors[source_sha] = vector
            self._by_source[source_sha] = []
            if source['ast_sha256'] is not None:
                self._by_ast.setdefault(source['ast_sha256'], []).append(source_sha)
        elif self._sources[source_sha]['code'] != code:
            raise ValueError('source SHA-256 collision')
        record = dict(id=program_id, source_sha256=source_sha, aliases=[],
                      islands=[] if island is None else [island], evaluation_ok=evaluation_ok)
        self._records[program_id] = record
        self._ids[program_id] = program_id
        self._by_source[source_sha].append(program_id)
        return deepcopy(record)

    def bind_program(self, code, program_id, island=None, *, evaluation_id=None):
        """Bind an admitted pID to an exact-source evaluation.

        Parallel jobs name their immutable evaluation ID. Serial callers retain
        the historical default of binding the most recent exact-source attempt.
        """
        if type(code) is not str:
            raise TypeError('code must be a string')
        program_id, island = _identifier(program_id), _island(island)
        source_sha = _sha(code)
        if source_sha not in self._by_source or self._sources[source_sha]['code'] != code:
            raise ValueError('cannot bind a program before its source is registered')
        if evaluation_id is None:
            record = self._records[self._by_source[source_sha][-1]]
        else:
            evaluation_id = _identifier(evaluation_id)
            record = self._records.get(evaluation_id)
            if record is None:
                raise ValueError('cannot bind an unknown evaluation ID')
            if record['source_sha256'] != source_sha:
                raise ValueError('evaluation ID is bound to different source')
        owner = self._ids.get(program_id)
        if owner is not None and owner != record['id']:
            raise ValueError('program ID is already bound to another evaluation')
        if program_id != record['id'] and program_id not in record['aliases']:
            record['aliases'].append(program_id)
        self._ids[program_id] = record['id']
        if island is not None and island not in record['islands']:
            record['islands'].append(island)
        return deepcopy(record)

    def inspect(self, code, remeasurement=False):
        if type(code) is not str:
            raise TypeError('code must be a string')
        if type(remeasurement) is not bool:
            raise TypeError('remeasurement must be a boolean')
        source_sha = _sha(code)
        known = self._sources.get(source_sha)
        if known is not None and known['code'] != code:
            raise ValueError('source SHA-256 collision')
        if known is not None and known['ast_sha256'] is not None:
            # SHA is the first gate; reuse its verified structural identity.
            dump, ast_error, ast_sha = None, None, known['ast_sha256']
        else:
            dump, ast_error = _structural_ast(code)
            ast_sha = _sha(dump) if dump is not None else None
        matching_sources = list(self._by_ast.get(ast_sha, ())) if ast_sha else []
        if source_sha in self._by_source and source_sha not in matching_sources:
            matching_sources.append(source_sha)
        matches = [self._records[ident] for sha in matching_sources for ident in self._by_source[sha]]
        pending = [r for r in matches if r['evaluation_ok'] is None]
        successful = [r for r in matches if r['evaluation_ok'] is True]
        blockers = pending or (successful if not remeasurement else [])
        choices = blockers or matches
        exact = [r for r in choices if r['source_sha256'] == source_sha]
        matched = (exact or choices)[-1] if choices else None
        result = dict(schema=SCHEMA, accept=not blockers, match_kind='new',
            matched_program_id=self._program_id(matched) if matched else None,
            matched_evaluation_id=matched['id'] if matched else None,
            reason='no_previous_candidate', source_sha256=source_sha, ast_sha256=ast_sha,
            ast_error=ast_error, representation=REPRESENTATION, neighbors=[], neighbors_scope='all_history',
            remeasurement=bool(matches and not blockers),
            remeasurement_of=list(dict.fromkeys(self._program_id(r) for r in matches)) if matches and not blockers else [])
        if matched:
            result['match_kind'] = 'sha256' if matched['source_sha256'] == source_sha else 'ast'
            result['reason'] = ('previous_evaluation_pending_or_unknown' if pending else
                'exact_source_duplicate' if blockers and result['match_kind'] == 'sha256' else
                'structural_ast_duplicate' if blockers else
                'explicit_remeasurement' if successful else 'previous_evaluation_failed')
            # Exact/structural equality already identifies the relevant source.
            # Do not scan every historical embedding to explain a known match.
            sha = matched['source_sha256']
            reference = self._vectors[sha]
            if sha == source_sha:
                similarity = 1.0 if any(reference) else 0.0
            else:
                try:
                    vector = _descriptor(code)
                except UnicodeError:
                    vector = [0.0] * 512
                similarity = max(-1.0, min(1.0, sum(a * b for a, b in zip(vector, reference))))
            result['neighbors_scope'] = 'identity_match'
            result['neighbors'].append(dict(program_id=self._program_id(matched),
                evaluation_id=matched['id'], aliases=list(matched['aliases']),
                islands=list(matched['islands']), evaluation_ok=matched['evaluation_ok'],
                similarity=similarity, source_sha256=sha, ast_sha256=self._sources[sha]['ast_sha256'],
                ast_diff='' if ast_sha is not None else None, ast_diff_truncated=False))
            return result
        try:
            candidate = _descriptor(code)
        except UnicodeError:
            candidate = [0.0] * 512
        if any(candidate):
            def scored():
                for index, (sha, vector) in enumerate(self._vectors.items()):
                    if any(vector):
                        similarity = max(-1.0, min(1.0, sum(a * b for a, b in zip(candidate, vector))))
                        yield similarity, -index, sha
            for similarity, _, sha in heapq.nlargest(self.neighbors, scored()):
                record = self._records[self._by_source[sha][-1]]
                program_id = self._program_id(record)
                before, _ = _structural_ast(self._sources[sha]['code'])
                difference, truncated = _ast_diff(before, dump, program_id)
                result['neighbors'].append(dict(program_id=program_id, evaluation_id=record['id'],
                    aliases=list(record['aliases']), islands=list(record['islands']),
                    evaluation_ok=record['evaluation_ok'], similarity=similarity,
                    source_sha256=sha, ast_sha256=self._sources[sha]['ast_sha256'],
                    ast_diff=difference, ast_diff_truncated=truncated))
            if matched is None and result['neighbors']:
                result.update(match_kind='embedding', reason='code_similarity_only',
                    matched_program_id=result['neighbors'][0]['program_id'],
                    matched_evaluation_id=result['neighbors'][0]['evaluation_id'])
        return result

    def state(self):
        state = dict(schema=SCHEMA, identity=self.identity,
                     sources=deepcopy(list(self._sources.values())), records=deepcopy(list(self._records.values())))
        return {**state, 'sha256': _sha(_canonical(state))}

    def load_state(self, state):
        """Validate fully into a detached registry before replacing live state."""
        if type(state) is not dict or set(state) != {'schema', 'identity', 'sources', 'records', 'sha256'}:
            raise ValueError('invalid novelty registry state fields')
        if state['schema'] != SCHEMA or _canonical(state['identity']) != _canonical(self.identity):
            raise ValueError('novelty registry identity mismatch')
        body = {key: value for key, value in state.items() if key != 'sha256'}
        try:
            digest = _sha(_canonical(body))
        except (TypeError, ValueError, RecursionError) as exc:
            raise ValueError('invalid novelty registry state encoding') from exc
        if state['sha256'] != digest:
            raise ValueError('novelty registry state hash mismatch')
        if type(state['sources']) is not list or type(state['records']) is not list:
            raise ValueError('novelty registry history must be lists')
        sources = {}
        for source in state['sources']:
            if type(source) is not dict or set(source) != {'source_sha256', 'ast_sha256', 'code'} or type(source['code']) is not str:
                raise ValueError('invalid novelty registry source')
            sha = _sha(source['code'])
            dump, _ = _structural_ast(source['code'])
            if source['source_sha256'] != sha or source['ast_sha256'] != (_sha(dump) if dump is not None else None):
                raise ValueError('novelty registry source or AST hash mismatch')
            if sha in sources:
                raise ValueError('duplicate novelty registry source')
            sources[sha] = source['code']
        restored = type(self)(self.neighbors)
        all_ids = set()
        for record in state['records']:
            if type(record) is not dict or set(record) != {'id', 'source_sha256', 'aliases', 'islands', 'evaluation_ok'}:
                raise ValueError('invalid novelty registry record')
            ident = _identifier(record['id'])
            if type(record['aliases']) is not list or type(record['islands']) is not list:
                raise ValueError('invalid novelty registry aliases or islands')
            ids = [ident] + [_identifier(alias) for alias in record['aliases']]
            if len(set(ids)) != len(ids) or any(alias in all_ids for alias in ids):
                raise ValueError('duplicate novelty registry program ID')
            all_ids.update(ids)
            islands = [_island(island) for island in record['islands']]
            if None in islands or len(set(islands)) != len(islands):
                raise ValueError('invalid novelty registry islands')
            if type(record['source_sha256']) is not str or record['source_sha256'] not in sources:
                raise ValueError('novelty registry record references missing source')
            restored.register(sources[record['source_sha256']], ident, evaluation_ok=_evaluation(record['evaluation_ok']))
            restored._records[ident]['islands'] = list(islands)
            restored._records[ident]['aliases'] = list(record['aliases'])
            for alias in record['aliases']:
                restored._ids[alias] = ident
        if set(restored._sources) != set(sources):
            raise ValueError('novelty registry contains unreferenced source')
        # The first occurrence of a source defines deterministic neighbor ties.
        if list(restored._sources) != list(sources):
            raise ValueError('novelty registry source order differs from evaluation history')
        self.__dict__.update(restored.__dict__)
