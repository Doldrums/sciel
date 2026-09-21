"""Immutable, source-bound evaluator data made available to research tools.

This is a copy of the returned evaluator record, not another evaluation. Only
content-addressed records belonging to this run can be read; tools never receive
a host path. An admitted program resolves through its exact novelty-registry
evaluation identity, including migrant aliases, never through a latest-source
heuristic. The existing registry checkpoint therefore preserves the binding.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile

from ..methods.models import canonical_bytes, content_hash, require_hash

SCHEMA = 'sciel-saved-evaluation/1'
INDEX_SCHEMA = 'sciel-saved-evaluation-index/1'
MAX_RECORD_BYTES = 32 * 1024 * 1024
MAX_INLINE_BYTES = 512 * 1024


def _sha(text):
    return hashlib.sha256(text.encode('utf-8', errors='surrogatepass')).hexdigest()


def _validate_record(record):
    fields = {'schema', 'run_id', 'task_hash', 'evaluation_id', 'source_sha256',
              'result', 'reached', 'confirmed', 'raw_first_score', 'conditions'}
    if type(record) is dict and record.get('schema') == 'sciel-saved-evaluation/2':
        fields.add('execution')
        from ..eval.candidate_backend import validate_execution
        validate_execution(record.get('execution'))
    if (type(record) is not dict or set(record) != fields or record['schema'] not in (SCHEMA, 'sciel-saved-evaluation/2')
            or any(type(record[key]) is not str or not record[key]
                   for key in ('run_id', 'task_hash', 'evaluation_id'))
            or record['reached'] not in ('smoke', 'full', 'confirm')
            or type(record['confirmed']) is not bool
            or type(record['raw_first_score']) not in (int, float)):
        raise ValueError('invalid saved evaluation record schema')
    require_hash(record['source_sha256'], 'saved evaluation source')
    result = record['result']
    if (type(result) is not dict
            or set(result) != {'ok', 'score', 'cases', 'stderr', 'timing', 'cost', 'artifacts', 'confirmed_sampled'}
            or any(type(result[key]) is not bool for key in ('ok', 'confirmed_sampled'))
            or any(type(result[key]) not in (int, float) for key in ('score', 'timing', 'cost'))
            or result['timing'] < 0 or result['cost'] < 0
            or type(result['cases']) is not dict or type(result['artifacts']) is not dict
            or type(result['stderr']) is not str):
        raise ValueError('invalid saved returned evaluator fields')
    conditions = record['conditions']
    if (type(conditions) is not dict
            or set(conditions) != {'evaluator_sha256', 'stage_timeouts_seconds', 'per_run_timeout_seconds',
                'cpu_limit_seconds', 'memory_limit_bytes', 'confirm_n', 'search_seed_config',
                'evaluator_seed_status', 'timing_scope', 'result_scope'}
            or type(conditions['stage_timeouts_seconds']) is not dict
            or set(conditions['stage_timeouts_seconds']) != {'smoke', 'full', 'confirm'}):
        raise ValueError('invalid saved evaluation conditions')
    if conditions['evaluator_sha256'] is not None:
        require_hash(conditions['evaluator_sha256'], 'saved evaluator source')
    canonical_bytes(record)
    return record


def _directory(engine, *, create=False):
    root = getattr(engine, 'run_dir', None)
    if root is None:
        return None
    root = Path(root)
    for part in ('research_evidence', 'evaluations'):
        root = root / part
        if root.is_symlink():
            raise ValueError('saved evaluation directory must not be a symlink')
        if create:
            root.mkdir(exist_ok=True)
        elif not root.exists():
            return None
    return root


def _read(path, limit):
    if path.is_symlink():
        raise ValueError('saved evaluation artifact must not be a symlink')
    with path.open('rb') as stream:
        data = stream.read(limit + 1)
    if len(data) > limit:
        raise ValueError('saved evaluation artifact exceeds its read bound')
    value = json.loads(data)
    canonical_bytes(value)  # Reject non-finite JSON without normalizing numbers.
    return value


def _write(path, value):
    """Publish once atomically; a replay must reproduce the exact saved value."""
    data = canonical_bytes(value)
    if path.is_symlink():
        raise ValueError('saved evaluation artifact must not be a symlink')
    if path.exists():
        if canonical_bytes(_read(path, MAX_RECORD_BYTES)) != data:
            raise ValueError('saved evaluation artifact changed on replay')
        return
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix='.evaluation-', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if canonical_bytes(_read(path, MAX_RECORD_BYTES)) != data:
                raise ValueError('saved evaluation artifact changed during publication')
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def capture_evaluation(engine, source, outcome, evaluation_id):
    """Save the exact returned cascade result before admission; never run code."""
    if type(getattr(engine.log, 'run_id', None)) is not str or not engine.log.run_id:
        # Direct engine helpers can be used before run_start. They are not an
        # identified experiment and cannot fabricate run-bound research data.
        return None
    if type(evaluation_id) is not str or not evaluation_id:
        raise ValueError('saved evaluation requires an explicit evaluation identity')
    task = engine.task
    evaluator = getattr(task, 'evaluator_path', None)
    evaluator_sha256 = hashlib.sha256(Path(evaluator).read_bytes()).hexdigest() if evaluator else None
    body = dict(schema=SCHEMA, run_id=engine.log.run_id, task_hash=task.task_hash,
        evaluation_id=evaluation_id, source_sha256=_sha(source),
        result=deepcopy(outcome.result.to_json()), reached=outcome.reached,
        confirmed=outcome.confirmed, raw_first_score=outcome.raw_first_score,
        conditions=dict(evaluator_sha256=evaluator_sha256,
            stage_timeouts_seconds={name: getattr(task, name + '_timeout', None)
                                    for name in ('smoke', 'full', 'confirm')},
            per_run_timeout_seconds=getattr(task, 'per_run_timeout', None),
            cpu_limit_seconds=getattr(task, 'cpu_seconds', None),
            memory_limit_bytes=getattr(task, 'mem_bytes', None),
            confirm_n=getattr(task, 'confirm_n', None),
            search_seed_config=getattr(engine, 'cfg', {}).get('random_seed'),
            evaluator_seed_status='Only evaluator-returned artifacts can attest its actual RNG state or seed.',
            timing_scope='result.timing is the evaluator-reported duration, not measured discovery CPU.',
            result_scope='Returned cascade stage only; earlier stages are not reproduced in this record.'))
    if getattr(outcome, 'execution', None) is not None:
        body['schema']='sciel-saved-evaluation/2'
        body['execution']=deepcopy(outcome.execution)
        body['conditions']['timing_scope']='Returned stage host wall time including candidate transport and host scoring; full cascade resources are separately bound in execution.'
        body['conditions']['result_scope']='Returned host-scored stage; execution binds all physical stage/scorer receipts and measured CPU.'
    data = canonical_bytes(_validate_record(body))
    if len(data) > MAX_RECORD_BYTES:
        raise ValueError('returned evaluation exceeds saved research artifact bound')
    reference = dict(sha256=content_hash(body), bytes=len(data), evaluation_id=evaluation_id,
                     source_sha256=body['source_sha256'], task_hash=body['task_hash'])
    directory = _directory(engine, create=True)
    if directory is None:
        return reference
    _write(directory / (reference['sha256'] + '.json'), body)
    index = dict(schema=INDEX_SCHEMA, run_id=engine.log.run_id, **reference)
    _write(directory / ('evaluation-' + _sha(evaluation_id) + '.json'), index)
    return deepcopy(reference)


def unavailable(reason, reference=None):
    return dict(status='unavailable', reason=reason, reference=deepcopy(reference), record=None)


def saved_evaluation(engine, source, evaluation_id, *, max_bytes=MAX_INLINE_BYTES):
    """Retrieve one exact identity; oversized values are whole omissions."""
    if type(max_bytes) is not int or not 0 < max_bytes <= MAX_INLINE_BYTES:
        raise ValueError('invalid saved evaluation materialization bound')
    return saved_evaluation_by_hash(engine, _sha(source), evaluation_id, max_bytes=max_bytes)


def saved_evaluation_by_hash(engine, source_sha256, evaluation_id, *, max_bytes=MAX_RECORD_BYTES,
                             expected_run_id=None):
    """Authenticate complete host evidence, including evicted/rejected sources.

    Internal admission recovery may need the whole returned record; tool-facing
    retrieval above keeps its smaller materialization bound.
    """
    require_hash(source_sha256, 'saved evaluation source')
    if type(max_bytes) is not int or not 0 < max_bytes <= MAX_RECORD_BYTES:
        raise ValueError('invalid saved evaluation materialization bound')
    if type(evaluation_id) is not str or not evaluation_id:
        return unavailable('No exact evaluation identity is bound to this target.')
    directory = _directory(engine)
    if directory is None:
        return unavailable('No saved evaluator data exists in this run.')
    path = directory / ('evaluation-' + _sha(evaluation_id) + '.json')
    if not path.exists() and not path.is_symlink():
        return unavailable('The exact evaluation has no saved research artifact.')
    index = _read(path, 4096)
    run_id = expected_run_id or getattr(engine.log, 'run_id', None)
    if run_id is None:
        # checkpoint.load authenticates provenance before making restored state
        # available; read-only projections can precede the next run_start.
        restored = getattr(engine, '_restored_checkpoint', None) or {}
        run_id = (restored.get('provenance') or {}).get('run_id')
    fields = {'schema', 'run_id', 'sha256', 'bytes', 'evaluation_id', 'source_sha256', 'task_hash'}
    if (type(index) is not dict or set(index) != fields or index['schema'] != INDEX_SCHEMA
            or index['evaluation_id'] != evaluation_id or index['source_sha256'] != source_sha256
            or index['run_id'] != run_id or index['task_hash'] != engine.task.task_hash
            or type(index['bytes']) is not int or not 0 < index['bytes'] <= MAX_RECORD_BYTES):
        raise ValueError('saved evaluator index differs from the requested run/source/evaluation')
    require_hash(index['sha256'], 'saved evaluator artifact')
    reference = {key: index[key] for key in fields - {'schema', 'run_id'}}
    record = _validate_record(_read(directory / (index['sha256'] + '.json'), MAX_RECORD_BYTES))
    if (content_hash(record) != index['sha256'] or len(canonical_bytes(record)) != index['bytes']
            or record.get('schema') not in (SCHEMA, 'sciel-saved-evaluation/2') or record.get('run_id') != run_id
            or any(record.get(key) != index[key] for key in ('task_hash', 'evaluation_id', 'source_sha256'))):
        raise ValueError('saved evaluator artifact identity differs')
    if index['bytes'] > max_bytes:
        return unavailable('Exact saved evaluation exceeds the bounded tool input; no values were truncated.', reference)
    return dict(status='available', reason=None, reference=reference, record=record)


def evaluation_id_for_program(engine, program):
    """Resolve explicit checkpoint bindings, never source or score similarity.

    This only resolves a declared identity. ``saved_evaluation`` authenticates
    its immutable run/task/source-bound record before it is called measured
    evidence. Missing legacy bindings remain unknown.
    """
    registry = getattr(engine, 'candidate_novelty', None)
    observed = getattr(engine, 'observed_changes', None)
    aliases = getattr(observed, '_aliases', {})
    bindings = getattr(engine, 'research_evaluation_bindings', {})
    ident, seen, evaluation_id = program.id, set(), None
    while True:
        if ident in seen:
            raise ValueError('cyclic saved evaluation program alias')
        seen.add(ident)
        registered = getattr(registry, '_ids', {}).get(ident)
        binding = bindings.get(ident)
        if binding is not None:
            validate_evaluation_bindings({ident: binding}, program_sources={ident: program.code})
        for candidate in (registered, binding['evaluation_id'] if binding else None):
            if candidate is not None:
                if type(candidate) is not str or not candidate:
                    raise ValueError('invalid saved evaluation program identity')
                if evaluation_id is not None and candidate != evaluation_id:
                    raise ValueError('saved evaluation program bindings disagree')
                evaluation_id = candidate
        alias = aliases.get(ident)
        if alias is None:
            return evaluation_id
        if alias.get('source_sha256') != _sha(program.code):
            raise ValueError('saved evaluation alias source differs')
        ident = alias['reference_program_id']


def saved_evaluation_for_program(engine, program):
    return saved_evaluation(engine, program.code, evaluation_id_for_program(engine, program))


def bind_evaluation_program(engine, program, evaluation_id):
    """Bind one admission in checkpoint state, including when novelty is off.

    Immutable evaluator files may outlive an uncommitted wave. Program IDs may
    be reused after rollback, so their authority must live in checkpoint state,
    never an external singleton file which survives that rollback.
    """
    saved = saved_evaluation(engine, program.code, evaluation_id)
    if saved['reference'] is None:
        # Mocked/legacy measurements without a returned record remain unavailable.
        return None
    binding = dict(source_sha256=_sha(program.code), evaluation_id=evaluation_id)
    bindings = getattr(engine, 'research_evaluation_bindings', {})
    if program.id in bindings and bindings[program.id] != binding:
        raise ValueError('admitted program is already bound to another saved evaluation')
    bindings[program.id] = binding
    engine.research_evaluation_bindings = bindings
    return deepcopy(binding)


def validate_evaluation_bindings(value, *, program_sources=None):
    if type(value) is not dict:
        raise ValueError('saved evaluator bindings must be a dictionary')
    for ident, binding in value.items():
        if (type(ident) is not str or not ident or type(binding) is not dict
                or set(binding) != {'source_sha256', 'evaluation_id'}
                or type(binding['evaluation_id']) is not str or not binding['evaluation_id']):
            raise ValueError('invalid saved evaluator program binding')
        require_hash(binding['source_sha256'], 'saved evaluator program source')
        if program_sources is not None and (ident not in program_sources
                or _sha(program_sources[ident]) != binding['source_sha256']):
            raise ValueError('saved evaluator binding differs from the checkpoint program source')
    return deepcopy(value)


def validate_saved_evaluation(value, *, source_sha256, task_hash=None, run_id=None):
    fields = {'status', 'reason', 'reference', 'record'}
    if type(value) is not dict or set(value) != fields:
        raise ValueError('invalid saved evaluation access contract')
    reference, record = value['reference'], value['record']
    if reference is not None:
        if (type(reference) is not dict
                or set(reference) != {'sha256', 'bytes', 'evaluation_id', 'source_sha256', 'task_hash'}
                or reference['source_sha256'] != source_sha256
                or (task_hash is not None and reference['task_hash'] != task_hash)
                or type(reference['bytes']) is not int or not 0 < reference['bytes'] <= MAX_RECORD_BYTES
                or type(reference['evaluation_id']) is not str or not reference['evaluation_id']):
            raise ValueError('saved evaluation reference differs from source/task')
        require_hash(reference['sha256'], 'saved evaluation reference')
    if value['status'] == 'unavailable':
        if record is not None or type(value['reason']) is not str or not value['reason']:
            raise ValueError('unavailable saved evaluation contains claimed values')
        return
    if (value['status'] != 'available' or value['reason'] is not None or reference is None
            or type(record) is not dict or record.get('schema') not in (SCHEMA, 'sciel-saved-evaluation/2')
            or (run_id is not None and record.get('run_id') != run_id)
            or content_hash(record) != reference['sha256']
            or len(canonical_bytes(record)) != reference['bytes'] or reference['bytes'] > MAX_INLINE_BYTES
            or any(record.get(key) != reference[key] for key in ('source_sha256', 'task_hash', 'evaluation_id'))):
        raise ValueError('saved evaluation value differs from its immutable reference')
    _validate_record(record)
