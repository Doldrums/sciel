"""Cheap guard tests; no model loading, candidate execution, or real run writes."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location('recover_circle_tail',
    Path(__file__).resolve().parents[2] / 'scripts' / 'recover_circle_tail.py')
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def fixture():
    first = dict(event='run_start', run_id='run', segment_id='segment', event_seq=1)
    prefix = (json.dumps(first) + '\n').encode()
    state = dict(iteration_no=190, B_evals=169,
        config=dict(task='oe/circle_packing', research={'enabled': True}),
        research=dict(pending={}, queue=[], stopped_reason=None),
        provenance=dict(run_id='run', journal_anchor=dict(byte_offset=len(prefix),
            prefix_sha256=recovery.digest(prefix), segment_id='segment', event_seq=1)))
    checkpoint = json.dumps(state).encode()
    response = 'saved response'
    source = 'def run_packing():\n    pass\n'
    rows = [dict(event='checkpoint', checkpoint_sha256=recovery.digest(checkpoint),
                 provenance=state['provenance'])]
    for iteration, ident in [(191, 'r211'), (192, 'r212')]:
        evidence = dict(iteration=iteration, attempt=1, operator='diff', generation_cost=.01,
                        generation_dispatch='returned', source_sha256=recovery.digest(source),
                        status='evaluated' if iteration == 191 else 'no_child')
        rows.append(dict(event='research', phase='attempt', experiment_id=ident,
                         evidence=copy.deepcopy(evidence), prompt_sha256=recovery.digest('prompt')))
        rows.append(dict(event='research', phase='hypothesis', experiment_id=ident,
                         evidence=copy.deepcopy(evidence), response=response,
                         response_sha256=recovery.digest(response), response_truncated=False))
        if iteration == 191:
            rows.append(dict(event='decision', policy='evaluation', action='result',
                             schema='sciel-evaluation-result/1', source_sha256=recovery.digest(source)))
        rows.append(dict(event='research', phase='outcome', experiment_id=ident, evidence=evidence))
        if iteration == 191:
            rows.extend([
                dict(event='iteration', iteration_no=191, B_evals=170, child_id='p181', child_code=source),
                dict(event='research', phase='aux_start', kind='conclusion', reference=ident,
                     prompt_sha256=recovery.digest('aux prompt')),
                dict(event='research', phase='aux_result', kind='conclusion', reference=ident,
                     cost=.002, response=response, response_sha256=recovery.digest(response),
                     response_truncated=False),
                dict(event='research', phase='conclusion', experiment_id=ident,
                     status='parsed', lesson_error=None, accepted_lesson_ids=['l46'])])
    for sequence, row in enumerate(rows, 2):
        row.update(event_seq=sequence, run_id='run', segment_id='segment')
    journal = prefix + b''.join((json.dumps(row) + '\n').encode() for row in rows)
    return state, checkpoint, journal, rows


def test_tail_requires_complete_verified_response_and_bound_evaluation():
    state, checkpoint, journal, rows = fixture()
    plan = recovery.validate_tail(state, checkpoint, journal)
    assert plan['iteration']['child_id'] == 'p181'
    assert plan['conclusion']['accepted_lesson_ids'] == ['l46']
    for mutation in ('truncated', 'response', 'evaluation', 'unknown', 'sequence'):
        changed = copy.deepcopy(rows)
        hypothesis = next(row for row in changed if row.get('phase') == 'hypothesis')
        if mutation == 'truncated':
            hypothesis['response_truncated'] = True
        elif mutation == 'response':
            hypothesis['response'] += ' changed'
        elif mutation == 'evaluation':
            next(row for row in changed if row.get('policy') == 'evaluation')['source_sha256'] = '0' * 64
        elif mutation == 'unknown':
            changed[-1]['evidence']['generation_dispatch'] = 'unknown'
        else:
            changed[-1]['event_seq'] += 1
        bad = journal[:plan['offset']] + b''.join((json.dumps(row) + '\n').encode() for row in changed)
        with pytest.raises(recovery.RecoveryError):
            recovery.validate_tail(state, checkpoint, bad)


def test_replay_compares_nested_parent_operator_preset_and_full_payload():
    expected = dict(event='research', phase='attempt', event_seq=20,
                    evidence=dict(parent_id='p59', operator='diff', injected_ids=['l46']))
    for altered in ({'parent_id': 'p48'}, {'operator': 'full_rewrite'}, {'injected_ids': []}):
        actual = recovery.payload(expected)
        actual['evidence'] = {**actual['evidence'], **altered}
        event = actual.pop('event')
        with pytest.raises(recovery.RecoveryError, match='replay mismatch'):
            recovery.ReplayLog([expected]).emit(event, **actual)
    log = recovery.ReplayLog([expected])
    actual = recovery.payload(expected)
    log.emit(actual.pop('event'), **actual)
    log.emit('iteration', child=None, reason='gate_rejected', iteration_no=192)
    assert log.index == 1 and len(log.extension) == 1
    with pytest.raises(recovery.RecoveryError, match='unexpected replay extension'):
        log.emit('iteration', child=None, reason='gate_rejected', iteration_no=193)


def test_saved_reply_requires_exact_prompt_and_no_extra_dispatch():
    state, checkpoint, journal, _ = fixture()
    plan = recovery.validate_tail(state, checkpoint, journal)
    cfg = dict(llm=dict(models=['model'], weights=[1]))
    log = recovery.ReplayLog([])
    llm = recovery.ReplayLLM(cfg, plan, log)
    log.last = recovery.payload(plan['attempts'][0])
    with pytest.raises(recovery.RecoveryError, match='prompt/model'):
        llm.generate('changed prompt', 'model')
    assert llm.used == 0
    response = llm.generate('prompt', 'model')
    assert response.cost == .01 and response.text == 'saved response'
    llm.used = 3
    with pytest.raises(recovery.RecoveryError, match='unexpected LLM dispatch'):
        llm.generate('anything', 'model')


def test_append_preserves_original_prefix_and_does_not_invent_monotonic_time(tmp_path):
    journal = tmp_path / 'journal.jsonl'
    original = b'{"event":"research","run_id":"r","segment_id":"s","event_seq":10}\n'
    journal.write_bytes(original)
    log = recovery.RecoveryAppendLog(journal, json.loads(original))
    log.emit('decision', policy='offline_tail_recovery', action='close_rejected_dispatch')
    assert journal.read_bytes().startswith(original)
    row = json.loads(journal.read_bytes().splitlines()[-1])
    assert row['event_seq'] == 11 and row['recovery_schema'] == recovery.SCHEMA
    assert row['timestamp'] and 'segment_elapsed_seconds' not in row
    assert log.anchor()['prefix_sha256'] == recovery.digest(journal.read_bytes())


def test_network_and_process_guard_is_not_swallowed_by_exception_handlers():
    with pytest.raises(recovery.RecoveryError, match='forbidden'):
        try:
            recovery.forbidden()
        except Exception:
            pytest.fail('ordinary worker handling must not swallow replay guard')
