"""Explicit abandonment of a finalized, unknown-charge provider request.

This is continuation, never request replay or cost reconciliation. The original
checkpoint and journal remain evidence; only a matched research stop is cleared.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path

SCHEMA = 'sciel-provider-recovery/1'
_AUX_STOP = 'research auxiliary request stopped: LLMError'
_GEN_STOP = 'generation outcome unknown'
_IDEA_STOP = 'two-stage idea stopped: LLMError'
_SPECIAL = ('repeat_population', 'repeat_promotion', 'circle_generation', 'circle_intervention')


class ProviderRecoveryError(ValueError):
    """A stop cannot be abandoned under this deliberately narrow protocol."""


def _require(condition, message):
    if not condition:
        raise ProviderRecoveryError(message)


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _digest(value):
    return _sha(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode())


def _hex(value):
    return type(value) is str and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def validate_recoveries(value):
    """Validate optional checkpoint metadata without loading execution modules."""
    _require(type(value) is list, 'invalid provider recovery history')
    seen = set()
    for row in value:
        _require(type(row) is dict and row.get('schema') == SCHEMA,
                 'invalid provider recovery receipt schema')
        ident = row.get('recovery_id')
        _require(_hex(ident) and ident not in seen, 'invalid or duplicate provider recovery id')
        seen.add(ident)
        _require(_hex(row.get('source_checkpoint_sha256'))
                 and _hex(row.get('terminal_event_sha256')),
                 'invalid provider recovery evidence hashes')
        _require(row.get('unknown_cost') is True and row.get('retry') is False
                 and row.get('stop_reason') in (_AUX_STOP, _GEN_STOP, _IDEA_STOP)
                 and type(row.get('auto_recoverable')) is bool,
                 'invalid provider recovery accounting/stop metadata')
        cost = row.get('known_cost_total')
        _require(type(cost) in (int, float) and math.isfinite(cost) and cost >= 0,
                 'invalid provider recovery known cost')
        _require(type(row.get('request')) is dict and type(row.get('source_anchor')) is dict
                 and type(row.get('provider_failure')) is dict,
                 'invalid provider recovery provenance')
        if row['stop_reason'] == _IDEA_STOP:
            request = row['request']
            _require(request.get('kind') == 'idea' and request.get('stage') in ('card', 'code')
                     and type(request.get('reference')) is str and request['reference'].startswith('idea')
                     and type(request.get('call_index')) is int and request['call_index'] > 0
                     and _hex(row.get('idea_stop_event_sha256'))
                     and type(row.get('known_attempt_cost')) in (int, float)
                     and math.isfinite(row['known_attempt_cost']) and row['known_attempt_cost'] >= 0
                     and row.get('auto_recoverable') is True,
                     'invalid stage-bound idea recovery receipt')
    return deepcopy(value)


def _provider_failure(metadata, *, legacy, allow_legacy):
    if metadata is None:
        _require(legacy and allow_legacy, 'provider diagnostics are absent; automatic recovery is forbidden')
        return {'category': 'legacy_generation_unknown'}, False
    _require(type(metadata) is dict and metadata.get('stage') == 'request',
             'only request-stage provider failures can be abandoned')
    reason, status = metadata.get('reason'), metadata.get('http_status')
    transient = (reason in ('timeout', 'connection_error') and status is None) or (
        reason == 'http_error' and type(status) is int and (status == 429 or 500 <= status <= 599))
    _require(transient, 'provider failure is not a classified transient transport failure')
    info = {'category': reason}
    if status is not None:
        info['http_status'] = status
    return info, True


def _idea_request(state, terminal, prior):
    """Prove a sealed card/code request; no response or evaluation is abandoned."""
    idea = state['idea_two_stage']
    attempts, started, returned = (idea.get(k) for k in ('attempts', 'calls_started', 'calls_returned'))
    retries = idea.get('code_retries', 0)
    card_retries = idea.get('card_retries', 0)
    card_limit = state['config'].get('idea_two_stage', {}).get('card_repair_attempts', 0)
    _require(idea.get('schema') == 'sciel-two-stage-idea/1'
             and state['config'].get('idea_two_stage', {}).get('enabled') is True
             and all(type(n) is int and n >= 0 for n in (attempts, started, returned))
             and type(retries) is int and 0 <= retries <= 2 * attempts
             and type(card_limit) is int and 0 <= card_limit <= 1
             and type(idea.get('card_repair_attempts', 0)) is int
             and idea.get('card_repair_attempts', 0) == card_limit
             and type(card_retries) is int and 0 <= card_retries <= attempts * card_limit
             and attempts > 0 and attempts + retries + card_retries <= started <= 2 * attempts + retries + card_retries
             and returned < started, 'inconsistent two-stage idea counters')
    past_unknown = sum(r.get('stop_reason') == _IDEA_STOP for r in state.get('provider_recoveries', []))
    _require(started - returned == past_unknown + 1, 'idea history contains unexplained missing responses')
    ident = f'idea{attempts:06d}'
    rows = prior + [terminal]
    idea_rows = [r for r in rows if r.get('phase') == 'idea' and r.get('idea_id') == ident]
    _require(idea_rows and all(r.get('idea_contract') == 'sciel-two-stage-idea/1' for r in idea_rows),
             'idea stop lacks its versioned attempt history')
    stop = idea_rows[-1]
    stage = stop.get('stage')
    _require(stage in ('card', 'code') and stop.get('action') == 'stopped'
             and stop.get('error_type') == 'LLMError'
             and stop.get('generation_returned') is False
             and stop.get('generation_dispatch') == 'unknown'
             and stop.get('current_call_cost_unknown') is True,
             'idea stop is not an unresolved provider request')
    # A format repair may follow one explicitly rejected, fully returned card.
    # The unresolved call is abandoned; preceding paid responses are retained.
    prefix, spent = 0, 0.0
    if len(idea_rows) >= 3 and idea_rows[2].get('action') == 'card_rejected':
        first, reply, rejected = idea_rows[:3]
        paid = reply.get('cost')
        _require(card_limit == 1 and card_retries > 0
                 and first.get('action') == 'request_start' and reply.get('action') == 'request_result'
                 and first.get('stage') == reply.get('stage') == 'card'
                 and first.get('experiment_id') is reply.get('experiment_id') is None
                 and _hex(first.get('prompt_sha256'))
                 and reply.get('generation_returned') is True
                 and type(paid) in (int, float) and math.isfinite(paid) and paid >= 0
                 and rejected.get('reason') == 'invalid_card_schema'
                 and rejected.get('card_attempt') == 1 and rejected.get('retry_scheduled') is True
                 and type(rejected.get('diagnostics')) is list and rejected['diagnostics']
                 and rejected.get('response_sha256') == reply.get('response_sha256'),
                 'idea card repair lacks a bound, rejected, known-cost response')
        from .idea import validate_card, digest as card_digest
        raw = reply.get('response')
        if reply.get('response_is_text') is True:
            _require(type(raw) is str and _hex(reply.get('response_sha256')),
                     'idea card repair lacks its returned text')
            if reply.get('response_truncated') is False:
                parsed, errors = validate_card(raw)
                _require(card_digest(raw) == reply['response_sha256'] and parsed is None
                         and errors == rejected['diagnostics'], 'idea card repair rejection cannot be verified')
            else:
                errors = rejected['diagnostics']
                _require(reply.get('response_truncated') is True and len(raw) == 8192
                         and len(errors) == 1 and errors[0].get('kind') == 'response_too_long'
                         and type(errors[0].get('length')) is int and errors[0]['length'] > 8192
                         and errors[0].get('maximum') == 8192,
                         'idea card repair truncation differs from the recorded format limit')
        else:
            _require(reply.get('response_is_text') is False and raw is None
                     and reply.get('response_sha256') is None
                     and rejected['diagnostics'] == validate_card(None)[1],
                     'idea non-text card rejection cannot be verified')
        prefix, spent = 3, paid
    remaining = idea_rows[prefix:]
    completed = remaining[3:-2] if stage == 'code' else []
    _require(len(completed) % 3 == 0 and len(completed) <= 6
             and len(completed) // 3 <= retries, 'idea retry history exceeds recorded bounds')
    if completed:
        novelty = state['config'].get('candidate_novelty', {})
        _require(novelty.get('enabled') is True
                 and type(novelty.get('max_resamples')) is int
                 and len(completed) // 3 <= novelty['max_resamples'] <= 2,
                 'idea retries are not enabled by the recorded novelty policy')
    expected = ['request_start', 'stopped'] if stage == 'card' else [
        'request_start', 'request_result', 'card_decision',
        *(['request_start', 'request_result', 'source_binding'] * (len(completed) // 3)),
        'request_start', 'stopped']
    _require([r.get('action') for r in remaining] == expected,
             'idea attempt has a response, replay, or unexpected transition')
    request = idea_rows[-2]
    _require(request.get('stage') == stage and _hex(request.get('prompt_sha256'))
             and type(request.get('iteration')) is int
             and request['iteration'] == state['iteration_no'] + 1,
             'idea request does not match the stopped iteration/stage')
    # After dispatch there must be only the stop and (for code) its matching
    # generation_stopped outcome. Nothing measured or partly returned is skipped.
    _require(rows[-(2 if stage == 'card' else 3):] == (
        [request, stop] if stage == 'card' else [request, stop, terminal]),
        'work intervenes between idea request and terminal stop')
    if stage == 'card':
        _require(terminal == stop and request.get('experiment_id') is None
                 and stop.get('experiment_id') is None and stop.get('known_cost') == spent,
                 'card stop contains a research outcome or an unexplained charge')
        experiment = None
    else:
        start_card, card, decision = remaining[:3]
        card_cost = card.get('cost')
        _require(start_card.get('stage') == card.get('stage') == 'card'
                 and card.get('generation_returned') is True
                 and type(card_cost) in (int, float) and math.isfinite(card_cost) and card_cost >= 0
                 and card.get('response_is_text') is True and _hex(card.get('response_sha256'))
                 and decision.get('accepted') is True and decision.get('matches') == []
                 and _hex(decision.get('card_sha256'))
                 and decision['card_sha256'] == request.get('card_sha256'),
                 'code request lacks the prior accepted, paid card')
        spent += card_cost
        seen = set()
        for offset in range(0, len(completed), 3):
            prior_start, prior_reply, binding = completed[offset:offset + 3]
            previous_id = prior_start.get('experiment_id')
            paid = prior_reply.get('cost')
            _require(type(previous_id) is str and previous_id and previous_id not in seen
                     and previous_id != request.get('experiment_id')
                     and prior_start.get('stage') == prior_reply.get('stage') == 'code'
                     and prior_reply.get('experiment_id') == binding.get('experiment_id') == previous_id
                     and prior_start.get('card_sha256') == prior_reply.get('card_sha256')
                         == binding.get('card_sha256') == decision['card_sha256']
                     and prior_reply.get('generation_returned') is True
                     and prior_reply.get('response_is_text') is True
                     and type(paid) in (int, float) and math.isfinite(paid) and paid >= 0
                     and _hex(prior_reply.get('response_sha256'))
                     and prior_reply['response_sha256'] == binding.get('response_sha256')
                     and _hex(prior_start.get('prompt_sha256'))
                     and prior_start['prompt_sha256'] == binding.get('prompt_sha256'),
                     'idea retry lacks a bound, known-cost returned code request')
            intents = [r for r in prior if r.get('phase') == 'attempt'
                       and r.get('experiment_id') == previous_id]
            outcomes = [r for r in prior if r.get('phase') == 'outcome'
                        and r.get('experiment_id') == previous_id]
            hypotheses = [r for r in prior if r.get('phase') == 'hypothesis'
                          and r.get('experiment_id') == previous_id]
            _require(len(intents) == len(outcomes) == len(hypotheses) == 1
                     and intents[0].get('prompt_sha256') == prior_start['prompt_sha256']
                     and hypotheses[0].get('response_sha256') == prior_reply['response_sha256'],
                     'idea retry lacks one completed research attempt')
            prior_evidence = outcomes[0].get('evidence') or {}
            _require(prior_evidence.get('status') in ('no_child', 'hypothesis_invalid')
                     and all(prior_evidence.get(k) is None for k in ('score', 'actual_delta', 'evaluation'))
                     and prior_evidence.get('generation_dispatch') == 'returned'
                     and prior_evidence.get('generation_cost') == paid
                     and prior_evidence.get('source_sha256') == binding.get('candidate_source_sha256'),
                     'idea retry would abandon a measured or unresolved proposal')
            seen.add(previous_id)
            spent += paid
        _require(stop.get('known_cost') == spent, 'idea stopped request lost prior known retry charges')
        experiment = request.get('experiment_id')
        _require(type(experiment) is str and experiment
                 and stop.get('experiment_id') == terminal.get('experiment_id') == experiment,
                 'idea code request differs from its research outcome')
        evidence = terminal.get('evidence') or {}
        _require(terminal.get('phase') == 'outcome'
                 and evidence.get('status') == 'generation_stopped'
                 and evidence.get('failure') == _IDEA_STOP
                 and evidence.get('generation_dispatch') == 'unknown'
                 and evidence.get('generation_cost') is None and evidence.get('score') is None
                 and evidence.get('hypothesis') is None,
                 'idea code stop has a returned, proposed, or measured outcome')
        attempts_rows = [r for r in prior if r.get('phase') == 'attempt'
                         and r.get('experiment_id') == experiment]
        _require(len(attempts_rows) == 1
                 and attempts_rows[0].get('prompt_sha256') == request['prompt_sha256'],
                 'idea code request does not match a single research intent')
    cost = idea.get('known_cost')
    _require(type(cost) in (int, float) and math.isfinite(cost) and cost >= spent
             and state['cost_total'] >= spent, 'known idea costs were not retained')
    return ({'kind': 'idea', 'stage': stage, 'reference': ident,
             'experiment_id': experiment, 'call_index': started}, stop,
            {'idea_stop_event_sha256': _digest(stop), 'known_attempt_cost': spent})


def inspect_provider_stop(checkpoint_path, journal_path=None, *, allow_legacy=True):
    """Read-only eligibility check. ``auto_recoverable`` excludes legacy failures.

    The caller must still load the checkpoint normally (identity, state and
    replay validation) and hold the run directory writer lock before applying.
    """
    path = Path(checkpoint_path)
    try:
        raw = path.read_bytes()
        state = json.loads(raw)
        cfg = state['config']
        research = state['research']
        provenance = state['provenance']
        anchor = provenance['journal_anchor']
        digest = _sha(raw)
        _require(state.get('milestone') == 'final', 'provider recovery requires a finalized checkpoint')
        _require('budget' in cfg and cfg['budget'] is None and 'max_iterations' in cfg
                 and cfg['max_iterations'] is None and cfg.get('research', {}).get('enabled') is True
                 and 'max_aux_calls' in cfg['research'] and cfg['research']['max_aux_calls'] is None,
                 'provider recovery is restricted to unlimited ordinary research runs')
        _require(not any(cfg.get(name, {}).get('enabled') or name in state for name in _SPECIAL),
                 'specialized study protocols cannot abandon provider requests')
        _require(not state.get('_accounting_stopped_reason'), 'accounting response stops cannot be abandoned')
        reason = research.get('stopped_reason')
        _require(reason in (_AUX_STOP, _GEN_STOP, _IDEA_STOP), 'checkpoint does not contain an eligible provider stop')
        _require(research.get('pending') == {} and research.get('queue') == []
                 and type(research.get('attempts')) is int
                 and research.get('attempts') == research.get('outcomes'),
                 'checkpoint has unresolved research work')
        idea = state.get('idea_two_stage')
        _require((idea is not None or reason != _IDEA_STOP)
                 and (idea is None or (idea.get('pending') is None
                     and idea.get('stopped_reason') == (_IDEA_STOP if reason == _IDEA_STOP else None)
                     and idea.get('attempts') == idea.get('finished'))),
                 'checkpoint has unresolved two-stage idea work')
        validate_recoveries(state.get('provider_recoveries', []))
        data = Path(journal_path or path.with_name('sciel_events.jsonl')).read_bytes()
        offset = anchor['byte_offset']
        _require(type(offset) is int and 0 < offset <= len(data)
                 and data.endswith(b'\n') and data[:offset].endswith(b'\n')
                 and _sha(data[:offset]) == anchor['prefix_sha256'], 'checkpoint journal prefix is not intact')
        prefix = [json.loads(line) for line in data[:offset].splitlines()]
        tail = [json.loads(line) for line in data[offset:].splitlines()]
        terminal = prefix[-1]
        _require(terminal.get('segment_id') == anchor['segment_id']
                 and terminal.get('event_seq') == anchor['event_seq']
                 and terminal.get('run_id') == provenance['run_id']
                 and terminal.get('event') == 'research', 'provider stop is not the checkpoint terminal event')
        # Match only the same segment: an old request with the same reference is
        # not evidence for the request being abandoned now.
        prior = [r for r in prefix[:-1] if r.get('segment_id') == anchor['segment_id']
                 and r.get('event') == 'research']
        idea_fields = {}
        if reason == _IDEA_STOP:
            request, diagnostic, idea_fields = _idea_request(state, terminal, prior)
        elif reason == _AUX_STOP:
            _require(terminal.get('phase') == 'aux_stopped' and terminal.get('stage') == 'request'
                     and terminal.get('error_type') == 'LLMError', 'auxiliary stop does not match its journal evidence')
            keys = ('kind', 'reference', 'call_index')
            requests = [r for r in prior if all(r.get(k) == terminal.get(k) for k in keys)]
            _require(requests and requests[-1].get('phase') == 'aux_start'
                     and terminal.get('kind') in ('conclusion', 'reflection')
                     and terminal.get('call_index') == research.get('aux_calls'), 'auxiliary request start is missing or resolved')
            request = {k: terminal[k] for k in keys}
            diagnostic = terminal
        else:
            evidence = terminal.get('evidence') or {}
            _require(terminal.get('phase') == 'outcome' and evidence.get('status') == 'generation_stopped'
                     and evidence.get('failure') == _GEN_STOP
                     and evidence.get('generation_dispatch') == 'unknown'
                     and evidence.get('generation_cost') is None and evidence.get('score') is None,
                     'generation stop does not match its journal evidence')
            ident = terminal['experiment_id']
            requests = [r for r in prior if r.get('experiment_id') == ident]
            _require(requests and requests[0].get('phase') == 'attempt'
                     and all(r.get('phase') in ('attempt', 'generation_error') for r in requests)
                     and sum(r.get('phase') == 'attempt' for r in requests) == 1,
                     'generation request start is missing or already resolved')
            diagnostic = next((r for r in reversed(requests) if r.get('phase') == 'generation_error'), None)
            request = {'kind': 'generation', 'reference': ident, 'call_index': None}
        if diagnostic is not None:
            _require(diagnostic.get('generation_returned') is False
                     and diagnostic.get('generation_dispatch') == 'unknown'
                     and diagnostic.get('current_call_cost_unknown') is True
                     and diagnostic.get('cost') is None, 'request does not have an unresolved charge')
        failure, automatic = _provider_failure((diagnostic or {}).get('llm_error'),
                                               legacy=reason == _GEN_STOP and diagnostic is None,
                                               allow_legacy=allow_legacy)
        receipt = {'schema': SCHEMA, 'source_checkpoint_sha256': digest,
                   'source_checkpoint_id': provenance['checkpoint_id'],
                   'source_anchor': deepcopy(anchor), 'run_id': provenance['run_id'],
                   'terminal_event_sha256': _digest(terminal), 'request': request,
                   'stop_reason': reason, 'provider_failure': failure,
                   'auto_recoverable': automatic, 'unknown_cost': True, 'retry': False,
                   'known_cost_total': state['cost_total'],
                   'authorization': 'explicit_abandon_provider_failure', **idea_fields}
        receipt['recovery_id'] = _digest(receipt)
        validate_recoveries([receipt])
        checkpoint_seen = ended = False
        for row in tail:
            event = row.get('event')
            _require(row.get('run_id') == provenance['run_id'], 'journal tail belongs to another run')
            if event in ('checkpoint', 'run_end'):
                _require(row.get('checkpoint_id') == provenance['checkpoint_id']
                         and row.get('checkpoint_sha256') == digest and row.get('reason') == 'research_stopped'
                         and row.get('iteration_no') == state['iteration_no']
                         and row.get('B_evals') == state['B_evals'], 'terminal checkpoint declaration differs')
                if event == 'checkpoint':
                    _require(row.get('provenance') == provenance, 'checkpoint declaration provenance differs')
                    checkpoint_seen = True
                else:
                    _require(checkpoint_seen, 'run ended without checkpoint declaration')
                    ended = True
            elif event == 'checkpoint_restore':
                _require(ended and row.get('checkpoint_sha256') == digest
                         and row.get('provenance') == provenance, 'later restore uses another checkpoint')
            elif event == 'run_start':
                _require(ended and row.get('resumed') is True, 'unexpected run start after provider stop')
            elif event == 'decision':
                _require(ended and row.get('policy') == 'provider_recovery'
                         and row.get('action') == 'abandon_unknown_request'
                         and row.get('receipt') == receipt, 'unexpected decision after provider stop')
            else:
                raise ProviderRecoveryError('journal contains work after the stopped checkpoint')
        _require(ended, 'provider stop has no finalized run_end')
        return receipt
    except ProviderRecoveryError:
        raise
    except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ProviderRecoveryError('cannot verify provider stop: ' + type(exc).__name__) from exc


def prepare_provider_recovery(engine, checkpoint_path, *, allow_legacy=True):
    receipt = inspect_provider_stop(checkpoint_path, engine.log.path, allow_legacy=allow_legacy)
    restored = getattr(engine, '_restored_checkpoint', None) or {}
    _require(restored.get('checkpoint_sha256') == receipt['source_checkpoint_sha256']
             and not restored.get('identity_mismatches') and restored.get('ancestry') == 'bound',
             'provider recovery requires a normally restored, identity-matching checkpoint')
    _require(not hasattr(engine.llm, 'ledger'), 'budget-ledger requests cannot be abandoned')
    _require(engine.research.stopped_reason == receipt['stop_reason'], 'restored research stop differs')
    if receipt['stop_reason'] == _IDEA_STOP:
        _require(getattr(engine, 'idea', None) is not None and engine.idea.stopped_reason == _IDEA_STOP,
                 'restored idea stop differs')
    engine._pending_provider_recovery = {'path': str(Path(checkpoint_path).resolve()), 'receipt': receipt}
    return deepcopy(receipt)


def _sync(path):
    with open(path, 'rb') as stream:
        os.fsync(stream.fileno())
    fd = os.open(str(Path(path).parent), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def apply_provider_recovery(engine):
    """Called after run_start, before the stopped-research gate or any request."""
    intent = getattr(engine, '_pending_provider_recovery', None)
    if intent is None:
        return False
    from . import checkpoint
    receipt = inspect_provider_stop(intent['path'], engine.log.path)
    _require(receipt == intent['receipt'], 'provider recovery evidence changed after preparation')
    restored = engine._restored_checkpoint
    _require(engine.log.run_id == receipt['run_id'] and engine.log.segment_id is not None
             and checkpoint.counters(engine) == restored['restored_counters']
             and engine.research.stopped_reason == receipt['stop_reason'], 'restored state changed before recovery')
    idea_recovery = receipt['stop_reason'] == _IDEA_STOP
    if idea_recovery:
        _require(engine.idea.stopped_reason == _IDEA_STOP, 'restored idea stop changed before recovery')
    source = Path(intent['path'])
    backup = source.parent / 'recovery' / ('checkpoint-' + receipt['source_checkpoint_sha256'] + '.json')
    backup.parent.mkdir(exist_ok=True)
    try:
        with backup.open('xb') as stream:
            stream.write(source.read_bytes())
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        pass
    _require(_sha(backup.read_bytes()) == receipt['source_checkpoint_sha256'], 'original checkpoint backup differs')
    _sync(backup)
    # An earlier attempt may have durably written the receipt and failed before
    # saving the new state. That receipt is reusable; the request is never replayed.
    tail = Path(engine.log.path).read_bytes()[receipt['source_anchor']['byte_offset']:]
    already = any(json.loads(line).get('receipt') == receipt for line in tail.splitlines())
    if not already:
        _require(engine.log.emit('decision', policy='provider_recovery', action='abandon_unknown_request',
                                 receipt=receipt), 'cannot persist provider recovery receipt')
    engine.log.anchor()
    previous = deepcopy(getattr(engine, 'provider_recoveries', []))
    _require(not any(r['recovery_id'] == receipt['recovery_id'] for r in previous), 'request was already abandoned')
    engine.provider_recoveries = validate_recoveries(previous + [receipt])
    engine.research.stopped_reason = None
    if idea_recovery:
        engine.idea.stopped_reason = None
    try:
        path = checkpoint.save(engine, 'provider_recovery')
        _sync(path)
        engine.log.checkpoint(milestone='provider_recovery', reason='provider_recovery', path=path,
                              iteration_no=engine.iteration_no, B_evals=engine.B_evals,
                              **engine._saved_checkpoint)
        engine.log.anchor()
    except BaseException:
        engine.research.stopped_reason = receipt['stop_reason']
        if idea_recovery:
            engine.idea.stopped_reason = _IDEA_STOP
        engine.provider_recoveries = previous
        raise
    engine._pending_provider_recovery = None
    return True
