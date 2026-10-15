"""Explicit abandonment of eligible provider errors in an unpublished wave.

Successful leaf receipts replay from the authenticated baseline. Failed requests
keep their original receipts and unknown charges; typed adapters skip only their
missing output. No physical request is retried by this protocol.
"""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path

from ..campaign.budget import CampaignStopped
from .llm.client import LLMError, accounted_cost, llm_error_metadata
from .parallel_calls import digest, durable_json, dispatch_evidence, _decode
from .recovery import ProviderRecoveryError, _provider_failure

SCHEMA = 'sciel-parallel-provider-abandonment/1'
STOP_SCHEMA = 'sciel-parallel-provider-stop/1'


def _require(value, message):
    if not value:
        raise ProviderRecoveryError(message)


def _read(path):
    _require(not path.is_symlink() and path.is_file(), 'recovery evidence must be a regular file')
    return json.loads(path.read_bytes())


def _sha(path):
    _require(not path.is_symlink() and path.is_file(), 'recovery evidence is missing or not regular')
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _hex(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _transient(metadata):
    try:
        _, automatic = _provider_failure(metadata, legacy=False, allow_legacy=False)
        return automatic
    except ProviderRecoveryError:
        return False


def _protocol_failure(metadata):
    """Recognize only the new bounded parser diagnostic, never a ValueError."""
    from .llm.codex_protocol import RECOVERABLE_CODES, validate_codex_protocol
    if (type(metadata) is not dict or metadata.get('stage') != 'response'
            or metadata.get('reason') != 'malformed_response'
            or metadata.get('cause_type') != '_CodexProtocolError'
            or 'http_status' in metadata):
        return None
    diagnostic = validate_codex_protocol(metadata.get('codex_protocol'))
    if (diagnostic and diagnostic['code'] in (
            'transport_incomplete_read', 'transport_tls_eof')
            and diagnostic['terminal_status'] != 'none'):
        return None  # A read interruption cannot hide an observed terminal event.
    return diagnostic if diagnostic and diagnostic['code'] in RECOVERABLE_CODES else None


def _recoverable_call(directory, row, config=None):
    """Bind an unusable response to its exact saved Codex request.

    The initial authorization also reconstructs metadata from the frozen run
    configuration, without reading credentials or opening a connection. Later
    marker replay relies on the already authorized, unchanged intent hashes.
    This does not broaden the separate serial recovery policy.
    """
    metadata = row.get('error')
    if _transient(metadata):
        return True
    diagnostic = _protocol_failure(metadata)
    if diagnostic is None:
        return False
    from .llm.codex import CONTRACT, ENDPOINT, CodexResponsesLLM
    try:
        intent = _read(directory / 'calls' / row['directory'] / 'intent.json')
        request = intent['request']
        provider = request['provider_request']
        if (type(provider) is not dict or provider.get('schema') != CONTRACT
                or provider.get('provider') != 'codex_responses'
                or provider.get('endpoint') != ENDPOINT
                or provider.get('body_sha256') != diagnostic['body_sha256']
                or provider.get('model') != diagnostic['model']
                or request.get('model') != diagnostic['model']):
            return False
        if config is not None:
            llm = config['llm']
            if llm.get('provider') != 'codex_responses' or llm.get('mock'):
                return False
            expected = CodexResponsesLLM(llm).request_metadata(
                request['prompt'], request['model'], request.get('temperature'),
                **request['options'])
            if provider != expected:
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def _classify(error):
    from .parallel_calls import ParallelAborted
    from .provider_abandonment import AbandonedProviderRequest
    seen = set()
    while error is not None and id(error) not in seen:
        seen.add(id(error))
        if isinstance(error, ParallelAborted):
            return {'kind': 'undispatched_sibling'}
        if isinstance(error, AbandonedProviderRequest):
            return {'kind': 'fatal'}  # Missing adapter; never loop on an already abandoned call.
        if isinstance(error, LLMError):
            metadata = llm_error_metadata(error)
            eligible = _transient(metadata) or _protocol_failure(metadata) is not None
            return {'kind': 'provider' if eligible else 'fatal', 'metadata': metadata}
        if not isinstance(error, CampaignStopped):
            break
        error = error.__cause__ or error.__context__
    return {'kind': 'fatal'}


def _calls(directory):
    """Authenticate every observed leaf, retaining dispatch uncertainty."""
    rows = []
    for call in sorted((directory / 'calls').iterdir()) if (directory / 'calls').exists() else []:
        _require(call.is_dir() and not call.is_symlink(), 'invalid call evidence directory')
        intent = _read(call / 'intent.json')
        key, kind = intent.get('key'), intent.get('kind')
        _require(type(key) is str and hashlib.sha256(key.encode()).hexdigest() == call.name
                 and intent.get('schema') == 'sciel-parallel-call/1'
                 and intent.get('request_sha256') == digest({'kind': kind, 'request': intent.get('request')}),
                 'parallel recovery intent differs')
        row = {'directory': call.name, 'key': key, 'kind': kind,
               'intent_sha256': _sha(call / 'intent.json'), 'request_sha256': intent['request_sha256']}
        receipt_path = call / 'receipt.json'
        dispatch_path = call / 'dispatched.json'
        if dispatch_path.exists():
            try:
                row['dispatch_sha256'] = dispatch_evidence(call, intent['request_sha256'])
            except ValueError as exc:
                raise ProviderRecoveryError('parallel recovery dispatch differs') from exc
        if receipt_path.exists():
            _require('dispatch_sha256' in row, 'terminal receipt lacks physical dispatch evidence')
            receipt = _read(receipt_path)
            payload = receipt.get('payload')
            _require(type(payload) is dict and receipt.get('sha256') == digest(payload)
                     and payload.get('request_sha256') == intent['request_sha256'],
                     'parallel recovery receipt differs')
            row.update(receipt_sha256=receipt['sha256'], receipt_file_sha256=_sha(receipt_path),
                       status=payload.get('status'))
            if payload.get('status') == 'returned':
                # A returned response with unknown/invalid accounting cannot be
                # hidden by an unrelated transient sibling failure.
                try:
                    value = _decode(kind, payload['value'])
                    if kind == 'llm':
                        accounted_cost(value)
                except Exception as exc:
                    raise ProviderRecoveryError('returned receipt is unusable: ' + type(exc).__name__) from exc
            elif payload.get('status') == 'error':
                row.update(error=payload.get('error'), not_dispatched=payload.get('not_dispatched'))
            else:
                raise ProviderRecoveryError('unknown parallel receipt status')
        else:
            _require(not (call / 'dispatched.json').exists(),
                     'dispatched operation has no terminal receipt; reconciliation required')
            row['status'] = 'not_dispatched'
        rows.append(row)
    return rows


def seal_stop(transaction, errors):
    """Record the full coordinator failure set before partial state is discarded."""
    directory = transaction.directory
    manifest = transaction.manifest
    body = {'schema': STOP_SCHEMA, 'wave_id': transaction.wave_id,
            'manifest': deepcopy(manifest), 'jobs_sha256': _sha(directory / 'jobs.json'),
            'errors': [_classify(error) for error in errors], 'calls': _calls(directory)}
    ident = digest(body)
    target = directory / 'provider_stops' / (ident + '.json')
    if target.exists():
        _require(_read(target) == body, 'provider stop evidence changed')
    else:
        durable_json(target, body)
    durable_json(directory / 'provider-stop.json', {'schema': STOP_SCHEMA, 'id': ident})


def _active(run_dir):
    from .parallel_store import _manifest, _authenticate, _active_prefix
    pointer = _read(run_dir / '.parallel/current.json')
    wave = pointer.get('wave_id')
    _require(pointer.get('schema') == 'sciel-wave-transaction/2'
             and type(wave) is str and wave.isdigit(), 'invalid recovery wave identity')
    directory = run_dir / '.parallel' / ('wave-' + wave)
    manifest = _manifest(directory)
    _require(manifest['phase'] == 'active', 'provider recovery requires an unpublished active wave')
    baseline = _authenticate(directory, manifest, 'baseline', 'baseline_sha256')
    _require(_sha(run_dir / 'checkpoint.json') == manifest['baseline_sha256'],
             'provider recovery checkpoint differs from wave baseline')
    _active_prefix(directory, manifest, run_dir)
    return directory, manifest, _read(baseline)


def inspect_parallel_stop(checkpoint_path):
    """Read-only, fail-closed eligibility for ordinary parallel search."""
    run_dir = Path(checkpoint_path).resolve().parent
    try:
        _require(not (run_dir / 'STOP').exists(), 'explicit STOP forbids provider recovery')
        directory, manifest, state = _active(run_dir)
        cfg = state['config']
        _require(cfg.get('parallel_search', {}).get('enabled') is True
                 and cfg.get('research', {}).get('enabled') is True
                 and cfg['research'].get('max_aux_calls') is None,
                 'parallel recovery requires ordinary research without an auxiliary call cap')
        _require(not any(cfg.get(name, {}).get('enabled') or name in state for name in
                         ('repeat_population', 'repeat_promotion', 'circle_generation', 'circle_intervention')),
                 'specialized study protocols cannot abandon provider requests')
        pointer = _read(directory / 'provider-stop.json')
        stop_id = pointer.get('id')
        _require(pointer.get('schema') == STOP_SCHEMA and _hex(stop_id), 'invalid provider stop pointer')
        stop = _read(directory / 'provider_stops' / (stop_id + '.json'))
        _require(digest(stop) == stop_id and stop.get('schema') == STOP_SCHEMA
                 and stop.get('manifest') == manifest and stop.get('wave_id') == manifest['wave_id']
                 and stop.get('jobs_sha256') == _sha(directory / 'jobs.json'),
                 'provider stop does not bind the current frozen wave')
        errors = stop.get('errors')
        _require(type(errors) is list and errors
                 and all(row.get('kind') in ('provider', 'undispatched_sibling') for row in errors)
                 and any(row.get('kind') == 'provider' for row in errors),
                 'wave contains an unclassified or non-provider failure')
        calls = _calls(directory)
        _require(calls == stop.get('calls'), 'call evidence changed after provider stop')
        failed = [row for row in calls if row['status'] == 'error']
        _require(failed and all(row['kind'] == 'llm' and row.get('not_dispatched') is False
                               and _recoverable_call(directory, row, cfg) for row in failed),
                 'wave contains a nonrecoverable physical operation')
        _require(all(any(row['error'] == error['metadata'] for row in failed)
                     for error in errors if error['kind'] == 'provider'),
                 'coordinator provider failure lacks a matching physical receipt')
        body = {'schema': SCHEMA, 'wave_id': manifest['wave_id'],
                'stop_id': stop_id, 'source_checkpoint_sha256': manifest['baseline_sha256'],
                'jobs_sha256': stop['jobs_sha256'], 'unknown_cost': True, 'retry': False,
                'auto_recoverable': True, 'failed_calls': failed}
        return {**body, 'recovery_id': digest(body)}
    except ProviderRecoveryError:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        raise ProviderRecoveryError('cannot verify parallel provider stop: ' + type(exc).__name__) from exc


def authorize_parallel_recovery(engine, checkpoint_path, receipt):
    """Own the run while authenticating and persisting explicit authorization.

    Engine.run closes its writer before the supervisor catches its exception.
    Reacquire ownership, then recheck every byte; never assume that gap was idle.
    """
    from contextlib import ExitStack
    from .run_lock import RunDirectoryLock
    lock = getattr(engine, '_writer_lock', None)
    _require(Path(engine.run_dir).resolve() == Path(checkpoint_path).resolve().parent,
             'parallel provider authorization belongs to another run')
    _require(not hasattr(engine.llm, 'ledger'), 'budget-ledger requests cannot be abandoned')
    with ExitStack() as cleanup:
        if lock is None or lock._file.closed:
            acquired = RunDirectoryLock(engine.run_dir)
            cleanup.callback(acquired.close)
        else:
            _require(lock._pid == os.getpid(), 'parallel recovery writer belongs to another process')
        _authorize_locked(checkpoint_path, receipt)


def _authorize_locked(checkpoint_path, receipt):
    current = inspect_parallel_stop(checkpoint_path)
    _require(current == receipt, 'provider recovery evidence changed before authorization')
    directory = Path(checkpoint_path).resolve().parent / '.parallel' / ('wave-' + receipt['wave_id'])
    target = directory / 'provider_recoveries' / (receipt['recovery_id'] + '.json')
    if target.exists():
        _require(_read(target) == receipt, 'recovery authorization changed')
    else:
        durable_json(target, receipt)
    for row in receipt['failed_calls']:
        call = directory / 'calls' / row['directory']
        marker = call / 'abandonment.json'
        if marker.exists():
            # An earlier recovery of this same wave may already own this error.
            load_abandonment(call, row['request_sha256'], row['receipt_sha256'])
        else:
            durable_json(marker, {'schema': SCHEMA, 'recovery_id': receipt['recovery_id']})


def has_unabandoned_provider_failure(checkpoint_path):
    """Do not turn an old stopped invocation into a newly authorized failure."""
    root = Path(checkpoint_path).resolve().parent
    pointer = root / '.parallel/current.json'
    if not pointer.exists():
        return False
    from .parallel_store import _manifest, _wave_id
    wave = _wave_id(_read(pointer).get('wave_id'))
    directory = root / '.parallel' / ('wave-' + wave)
    if _manifest(directory)['phase'] != 'active':
        return False
    for row in _calls(directory):
        if row['kind'] == 'llm' and row['status'] == 'error':
            if load_abandonment(directory / 'calls' / row['directory'],
                                row['request_sha256'], row['receipt_sha256']) is None:
                return True
    return False


def load_abandonment(call_directory, request_hash, receipt_hash):
    """Validate a saved authorization, never infer it from an HTTP status alone."""
    call = Path(call_directory)
    marker = call / 'abandonment.json'
    if not marker.exists():
        return None
    value = _read(marker)
    ident = value.get('recovery_id')
    _require(value.get('schema') == SCHEMA and _hex(ident), 'invalid abandonment identity')
    wave = call.parent.parent
    receipt = _read(wave / 'provider_recoveries' / (ident + '.json'))
    body = {k: v for k, v in receipt.items() if k != 'recovery_id'}
    _require(receipt.get('recovery_id') == ident and digest(body) == ident
             and receipt.get('schema') == SCHEMA and receipt.get('unknown_cost') is True
             and receipt.get('retry') is False and receipt.get('auto_recoverable') is True
             and receipt.get('wave_id') == wave.name.removeprefix('wave-')
             and receipt.get('jobs_sha256') == _sha(wave / 'jobs.json'),
             'abandonment authorization differs')
    stop_id = receipt.get('stop_id')
    _require(_hex(stop_id), 'invalid abandonment stop reference')
    stop = _read(wave / 'provider_stops' / (stop_id + '.json'))
    _require(digest(stop) == stop_id and stop.get('schema') == STOP_SCHEMA
             and stop.get('jobs_sha256') == receipt['jobs_sha256']
             and stop['manifest']['baseline_sha256'] == receipt['source_checkpoint_sha256'],
             'abandonment stop evidence differs')
    rows = [row for row in receipt['failed_calls'] if row['directory'] == call.name]
    _require(len(rows) == 1, 'abandonment does not uniquely identify a call')
    row = rows[0]
    _require(row in stop['calls'] and row['kind'] == 'llm' and row['status'] == 'error'
             and row['not_dispatched'] is False and _recoverable_call(wave, row)
             and row['request_sha256'] == request_hash and row['receipt_sha256'] == receipt_hash
             and row['intent_sha256'] == _sha(call / 'intent.json')
             and row['dispatch_sha256'] == _sha(call / 'dispatched.json')
             and row['receipt_file_sha256'] == _sha(call / 'receipt.json'),
             'abandonment call evidence differs')
    return {'recovery_id': ident, 'job_call': row['key'], 'request_sha256': request_hash,
            'receipt_sha256': receipt_hash, 'unknown_cost': True, 'retry': False}
