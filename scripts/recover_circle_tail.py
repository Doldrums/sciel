#!/usr/bin/env python3
"""Offline, fail-closed recovery of one completed step and one rejected proposal.

This deliberately narrow tool never edits the source run. It replays saved LLM
replies and a saved evaluator receipt on a detached checkpoint, comparing every
original journal payload. The interrupted final dispatch is explicitly closed
after its one recorded rejection, without issuing its remaining retries.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
from unittest.mock import patch


SCHEMA = 'sciel-offline-tail-recovery/1'
TRANSPORT_FIELDS = {'timestamp', 'segment_elapsed_seconds', 'run_id', 'segment_id', 'event_seq'}


class RecoveryError(BaseException):
    """Cannot be swallowed by the worker's ordinary request-error handling."""


def require(condition, message):
    if not condition:
        raise RecoveryError(message)


def digest(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def payload(row):
    return {key: value for key, value in row.items() if key not in TRANSPORT_FIELDS}


def difference(a, b, path='$'):
    if type(a) is not type(b) and not (type(a) in (float, int) and type(b) in (float, int)):
        return path + ': type differs'
    if isinstance(a, dict):
        if a.keys() != b.keys():
            return path + ': keys differ ' + str(sorted(a.keys() ^ b.keys()))
        return next((item for key in a if (item := difference(a[key], b[key], path + '.' + key))), None)
    if isinstance(a, list):
        if len(a) != len(b):
            return path + ': length differs'
        return next((item for i, (left, right) in enumerate(zip(a, b))
                     if (item := difference(left, right, f'{path}[{i}]'))), None)
    return None if a == b else path + ': value differs'


def validate_tail(state, checkpoint_bytes, journal_bytes):
    """Reject other crash shapes instead of guessing their missing transitions."""
    anchor = state.get('provenance', {}).get('journal_anchor', {})
    offset = anchor.get('byte_offset')
    require(type(offset) is int and 0 < offset < len(journal_bytes), 'missing post-checkpoint tail')
    prefix = journal_bytes[:offset]
    require(prefix.endswith(b'\n') and digest(prefix) == anchor.get('prefix_sha256'), 'checkpoint prefix hash differs')
    require(journal_bytes.endswith(b'\n'), 'incomplete journal tail')
    head = json.loads(prefix.rstrip(b'\n').rsplit(b'\n', 1)[-1])
    require((head.get('event_seq'), head.get('segment_id')) ==
            (anchor.get('event_seq'), anchor.get('segment_id')), 'anchor event differs')
    rows = [json.loads(line) for line in journal_bytes[offset:].splitlines()]
    require(len(rows) <= 40, 'tail exceeds this bounded recovery protocol')
    for index, row in enumerate(rows, anchor['event_seq'] + 1):
        require(row.get('event_seq') == index and row.get('segment_id') == anchor['segment_id']
                and row.get('run_id') == state['provenance']['run_id'], 'tail sequence/identity differs')
    declaration = rows[0]
    require(declaration.get('event') == 'checkpoint'
            and declaration.get('checkpoint_sha256') == digest(checkpoint_bytes)
            and declaration.get('provenance') == state['provenance'], 'checkpoint declaration differs')
    attempts = [row for row in rows if row.get('event') == 'research' and row.get('phase') == 'attempt']
    hypotheses = [row for row in rows if row.get('event') == 'research' and row.get('phase') == 'hypothesis']
    outcomes = [row for row in rows if row.get('event') == 'research' and row.get('phase') == 'outcome']
    iterations = [row for row in rows if row.get('event') == 'iteration']
    receipts = [row for row in rows if row.get('policy') == 'evaluation' and row.get('action') == 'result']
    auxiliaries = [row for row in rows if row.get('phase') == 'aux_result']
    conclusions = [row for row in rows if row.get('phase') == 'conclusion']
    require(len(attempts) == len(hypotheses) == len(outcomes) == 2 and len(iterations) == 1
            and len(receipts) == len(auxiliaries) == len(conclusions) == 1, 'unsupported tail shape')
    require(rows[-1] is outcomes[-1] and outcomes[-1]['evidence']['status'] == 'no_child'
            and outcomes[0]['evidence']['status'] == 'evaluated', 'tail must end in one resolved rejection')
    for index, (attempt, hypothesis, outcome) in enumerate(zip(attempts, hypotheses, outcomes), 1):
        require(attempt['experiment_id'] == hypothesis['experiment_id'] == outcome['experiment_id'], 'attempt binding differs')
        evidence = attempt['evidence']
        require(evidence['iteration'] == state['iteration_no'] + index and evidence['attempt'] == 1
                and evidence['operator'] in {'diff', 'full_rewrite', 'error_repair'}, 'unsupported dispatch')
        require(outcome['evidence']['generation_dispatch'] == 'returned', 'generation outcome unknown')
    require(iterations[0]['iteration_no'] == state['iteration_no'] + 1
            and iterations[0]['B_evals'] == state['B_evals'] + 1
            and iterations[0].get('child_id') and receipts[0].get('schema') == 'sciel-evaluation-result/1',
            'completed iteration/evaluation binding differs')
    require(conclusions[0].get('status') == 'parsed' and not conclusions[0].get('lesson_error')
            and conclusions[0]['experiment_id'] == attempts[0]['experiment_id'], 'conclusion not safely admitted')
    require(auxiliaries[0].get('kind') == 'conclusion'
            and auxiliaries[0]['reference'] == attempts[0]['experiment_id'], 'auxiliary binding differs')
    for row in [*hypotheses, *auxiliaries]:
        require(row.get('response_truncated') is False and isinstance(row.get('response'), str)
                and digest(row['response']) == row.get('response_sha256'), 'response is truncated or hash differs')
        cost = row.get('cost') if row.get('phase') == 'aux_result' else row['evidence'].get('generation_cost')
        require(type(cost) in (int, float) and math.isfinite(cost) and cost >= 0, 'response cost unknown')
    require(receipts[0]['source_sha256'] == digest(iterations[0]['child_code'])
            == outcomes[0]['evidence']['source_sha256'], 'evaluated source differs')
    cfg = state['config']
    require(cfg.get('task') == 'oe/circle_packing' and cfg.get('research', {}).get('enabled'), 'ordinary Circle research only')
    require(not any(cfg.get(key, {}).get('enabled') for key in
                    ('repeat_promotion', 'repeat_population', 'circle_generation', 'circle_intervention')), 'unsupported coordinator')
    require(not cfg.get('sciel_tasks_repo') and not cfg.get('wandb', {}).get('enabled'), 'remote assets/sinks forbidden')
    require(not state.get('_accounting_stopped_reason') and not state['research'].get('stopped_reason')
            and not state['research'].get('pending') and not state['research'].get('queue'), 'checkpoint already stopped or pending')
    return dict(rows=rows, attempts=attempts, hypotheses=hypotheses, receipt=receipts[0],
                auxiliary=auxiliaries[0], iteration=iterations[0], conclusion=conclusions[0], offset=offset)


class ReplayLog:
    def __init__(self, rows):
        self.rows, self.index, self.extension = rows, 0, []
        self.last = None

    def emit(self, event, **fields):
        actual = json.loads(canonical(dict(event=event, **fields)))
        if self.index < len(self.rows):
            expected = self.rows[self.index]
            mismatch = difference(payload(expected), actual)
            require(not mismatch, f'event {expected["event_seq"]} replay mismatch: {mismatch}; '
                    f'actual={event}/{fields.get("phase")}/{(fields.get("evidence") or {}).get("failure")}')
            self.index += 1
        else:
            require(not self.extension and event == 'iteration' and fields.get('child') is None
                    and fields.get('reason') == 'gate_rejected', 'unexpected replay extension')
            self.extension.append(actual)
        self.last = actual
        return True

    def __getattr__(self, name):
        if name in ('research', 'decision', 'iteration', 'population', 'postmortem', 'sh_cut', 'window_closed'):
            return lambda **fields: self.emit(name, **fields)
        raise AttributeError(name)

    def close(self):
        pass


class ReplayLLM:
    def __init__(self, cfg, plan, log):
        self.models = list(cfg['llm']['models'])
        self.weights = list(cfg['llm']['weights'])
        require(len(self.models) == 1, 'this recovery requires a single recorded model')
        aux_start = next(row for row in plan['rows'] if row.get('phase') == 'aux_start')
        self.calls = [(plan['attempts'][0], plan['hypotheses'][0]),
                      (aux_start, plan['auxiliary']), (plan['attempts'][1], plan['hypotheses'][1])]
        self.log, self.used, self.verified = log, 0, []

    def choose_model(self, rng):
        return rng.choices(self.models, weights=self.weights, k=1)[0]

    def generate(self, prompt, model, temperature=None, *, response_format=None):
        from sciel.engine.llm.client import LLMResponse
        require(self.used < len(self.calls), 'unexpected LLM dispatch; network is forbidden')
        start, reply = self.calls[self.used]
        require(self.log.last == payload(start), 'request is not bound to its recorded start')
        require(digest(prompt) == start['prompt_sha256'] and model == start.get('model', self.models[0]),
                'request prompt/model differs')
        require(response_format == start.get('response_format'), 'request response schema differs')
        auxiliary = reply['phase'] == 'aux_result'
        cost = reply['cost'] if auxiliary else reply['evidence']['generation_cost']
        self.verified.append(dict(event_seq=reply['event_seq'], prompt_sha256=digest(prompt),
            response_sha256=reply['response_sha256'], model=model, cost=cost))
        self.used += 1
        return LLMResponse(reply['response'], model, cost, reply.get('prompt_tokens'),
                           reply.get('completion_tokens'), reply.get('finish_reason'), reply.get('refusal'))


class RecoveryAppendLog:
    """Append explicit recovery events; never fabricate old monotonic timing."""
    def __init__(self, path, last):
        self.path = str(path)
        self.run_id, self.segment_id, self.event_seq = last['run_id'], last['segment_id'], last['event_seq']

    def emit(self, event, **fields):
        self.event_seq += 1
        row = dict(event=event, **fields, run_id=self.run_id, segment_id=self.segment_id,
                   event_seq=self.event_seq, timestamp=datetime.now(timezone.utc).isoformat(),
                   recovery_schema=SCHEMA)
        with open(self.path, 'ab') as stream:
            stream.write((json.dumps(row, allow_nan=False) + '\n').encode())
            stream.flush()
            os.fsync(stream.fileno())
        return True

    def checkpoint(self, **fields):
        return self.emit('checkpoint', **fields)

    def anchor(self):
        data = Path(self.path).read_bytes()
        last = json.loads(data.splitlines()[-1])
        require((last['segment_id'], last['event_seq']) == (self.segment_id, self.event_seq), 'recovery journal changed')
        return dict(segment_id=self.segment_id, event_seq=self.event_seq,
                    byte_offset=len(data), prefix_sha256=digest(data))

    def close(self):
        pass


def forbidden(*args, **kwargs):
    raise RecoveryError('network, process launch, or live evaluation forbidden during recovery')


def recover(source, destination, tasks_root):
    source, destination, tasks_root = Path(source).resolve(), Path(destination).resolve(), Path(tasks_root).resolve()
    require(not destination.exists() and source not in destination.parents and destination not in source.parents,
            'destination must be a new detached directory')
    checkpoint_bytes = (source / 'checkpoint.json').read_bytes()
    journal_bytes = (source / 'sciel_events.jsonl').read_bytes()
    state = json.loads(checkpoint_bytes)
    plan = validate_tail(state, checkpoint_bytes, journal_bytes)
    before = dict(checkpoint_sha256=digest(checkpoint_bytes), journal_sha256=digest(journal_bytes))
    destination.mkdir(parents=True)
    (destination / 'checkpoint.original.json').write_bytes(checkpoint_bytes)
    (destination / 'sciel_events.jsonl').write_bytes(journal_bytes[:plan['offset']])
    if (source / 'config.yaml').exists():
        shutil.copyfile(source / 'config.yaml', destination / 'config.yaml')
    os.environ['OMP_NUM_THREADS'] = os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    engine = None
    verified = False
    with ExitStack() as guard:
        guard.enter_context(patch('socket.socket.connect', forbidden))
        guard.enter_context(patch('socket.socket.connect_ex', forbidden))
        guard.enter_context(patch('subprocess.Popen', forbidden))
        guard.enter_context(patch('os.system', forbidden))
        from sciel.engine import checkpoint
        from sciel.engine.loop import Engine
        from sciel.eval.registry import load_task
        from sciel.eval.cascade import CascadeOutcome
        from sciel.eval.facade import EvalResult
        guard.enter_context(patch.object(Engine, '_evaluate', forbidden))
        import torch
        torch.set_num_threads(1)
        replay_log = ReplayLog(plan['rows'][1:])
        llm = ReplayLLM(state['config'], plan, replay_log)
        task = load_task(str(tasks_root / state['config']['task']))
        try:
            engine = Engine(state['config'], task, str(destination), llm, python=sys.executable)
            checkpoint.load(engine, str(destination / 'checkpoint.original.json'))
            engine.log.close()
            engine.log = engine.research.log = replay_log
            # Offline replay is not a fresh provider dispatch. Do not create
            # ordinary live-call transcripts for these historical responses.
            engine.llm = engine.research.llm = llm
            evaluation_calls = []

            def evaluate(code):
                receipt = plan['receipt']
                require(not evaluation_calls and digest(code) == receipt['source_sha256'], 'unexpected evaluator call/source')
                require(engine.iteration_no == receipt['iteration_no_before']
                        and engine.B_evals == receipt['B_evals_before'], 'evaluator counters differ')
                fields = payload(receipt)
                event = fields.pop('event')
                replay_log.emit(event, **fields)
                evaluation_calls.append(receipt['event_seq'])
                return CascadeOutcome(EvalResult.from_json(receipt), receipt['raw_first_score'],
                                      receipt['reached'], receipt['confirmed'])

            engine._evaluate = evaluate
            for index in range(2):
                islands = list(range(len(engine.db.islands)))
                island = engine.schedule.pick_island(islands, {item: 0 for item in islands})
                require(island == plan['attempts'][index]['evidence']['island'], 'scheduled island differs')
                original_cap = engine.cfg['dup_cap']
                if index == 1:
                    engine.cfg['dup_cap'] = 1
                try:
                    engine._step(island)
                finally:
                    engine.cfg['dup_cap'] = original_cap
                engine._maybe_milestone()
            require(replay_log.index == len(replay_log.rows) and len(replay_log.extension) == 1
                    and llm.used == 3 and len(evaluation_calls) == 1, 'replay did not consume exactly the evidence')
            require(not engine.research.pending and not engine.research.queue
                    and not engine.research.stopped_reason, 'recovery left unfinished research')
            expected_cost = state['cost_total'] + sum(call['cost'] for call in llm.verified)
            require(math.isclose(engine.cost_total, expected_cost, rel_tol=0, abs_tol=1e-12), 'recovered cost differs')
            admitted = plan['conclusion']['accepted_lesson_ids']
            require(set(admitted) <= {lesson['id'] for lesson in engine.research.knowledge.lessons}, 'admitted lesson missing')
            child_id = plan['iteration']['child_id']
            require(digest(engine.db.get(child_id).code) == plan['receipt']['source_sha256'], 'admitted child missing')
            closure = dict(policy='offline_tail_recovery', action='close_rejected_dispatch', schema=SCHEMA,
                source_checkpoint_sha256=before['checkpoint_sha256'], source_journal_sha256=before['journal_sha256'],
                source_last_event_seq=plan['rows'][-1]['event_seq'],
                iteration_no=engine.iteration_no, experiment_id=plan['attempts'][1]['experiment_id'],
                closure='One recorded rejected proposal closes this interrupted dispatch; remaining retries were not issued.',
                recorded_dup_cap=engine.cfg['dup_cap'], recovery_dispatch_attempts=1,
                replayed_calls=llm.verified, new_provider_calls=0, new_candidate_evaluations=0)
            (destination / 'sciel_events.jsonl').write_bytes(journal_bytes)
            append = RecoveryAppendLog(destination / 'sciel_events.jsonl', plan['rows'][-1])
            engine.log = engine.research.log = append
            append.emit('decision', **closure)
            addition = dict(replay_log.extension[0])
            event = addition.pop('event')
            addition['meta']['offline_recovery'] = dict(schema=SCHEMA, closure='remaining_retries_cancelled',
                                                       original_dup_cap=engine.cfg['dup_cap'])
            append.emit(event, **addition)
            candidate = checkpoint.save(engine, 'offline_tail_recovery')
            append.checkpoint(milestone='offline_tail_recovery', iteration_no=engine.iteration_no,
                B_evals=engine.B_evals, path=candidate, **engine._saved_checkpoint)
            recovered = json.loads(Path(candidate).read_bytes())
            manifest = dict(schema=SCHEMA, source=str(source), destination=str(destination), original=before,
                source_checkpoint_id=state['provenance']['checkpoint_id'],
                checked_event_seq=[row['event_seq'] for row in plan['rows'][1:]],
                calls=llm.verified, evaluator_receipt_event_seq=evaluation_calls[0],
                preserved_child=child_id, preserved_lessons=admitted, closure=closure,
                counters=checkpoint.counters(engine), known_cost_added=sum(call['cost'] for call in llm.verified),
                checkpoint_sha256=digest(Path(candidate).read_bytes()),
                component_sha256={key: digest(canonical(recovered[key])) for key in
                    (*checkpoint._COMPONENTS, 'research', 'preset_weights') if key in recovered},
                original_bytes_unchanged=False, ordinary_resume_load_verified=False)
            engine.close()
            engine = None
            # Verify the candidate through the ordinary guarded loader; no run.
            engine = Engine(state['config'], task, str(destination), llm, python=sys.executable)
            checkpoint.load(engine, candidate)
            require(checkpoint.counters(engine) == manifest['counters'], 'restored counters differ')
            for key in checkpoint._COMPONENTS:
                require(digest(canonical(getattr(engine, key).state())) == manifest['component_sha256'][key],
                        'restored component differs: ' + key)
            require(digest(canonical(engine.research.state())) == manifest['component_sha256']['research'],
                    'restored research differs')
            require(digest(canonical(engine.preset_weights.state())) == manifest['component_sha256']['preset_weights'],
                    'restored presets differ')
            manifest['ordinary_resume_load_verified'] = True
            require(digest((source / 'checkpoint.json').read_bytes()) == before['checkpoint_sha256']
                    and digest((source / 'sciel_events.jsonl').read_bytes()) == before['journal_sha256'], 'source changed')
            manifest['original_bytes_unchanged'] = True
            (destination / 'recovery_manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False) + '\n')
            verified = True
            return manifest
        finally:
            if engine is not None:
                engine.close()
            if not verified and (destination / 'checkpoint.json').exists():
                # An unsuccessful verification must not leave an ordinary
                # resume target that looks like an approved recovery.
                (destination / 'checkpoint.json').replace(destination / 'checkpoint.unverified.json')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source')
    parser.add_argument('destination')
    parser.add_argument('--tasks-root', default='tasks')
    args = parser.parse_args()
    try:
        manifest = recover(args.source, args.destination, args.tasks_root)
    except RecoveryError as exc:
        raise SystemExit('Recovery refused: ' + str(exc)) from None
    print(json.dumps({key: manifest[key] for key in ('destination', 'counters', 'known_cost_added',
                     'checkpoint_sha256', 'ordinary_resume_load_verified', 'original_bytes_unchanged')}, indent=2))


if __name__ == '__main__':
    main()
