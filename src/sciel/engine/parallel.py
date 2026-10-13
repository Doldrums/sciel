"""Bounded parallel waves with one scientific-state writer per run."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from time import monotonic
from urllib.parse import urlsplit

from ..campaign.budget import CampaignStopped
from . import checkpoint, portfolio_allocation, research_queue
from .capacity import HostCapacity
from .llm.trace import TracedLLM
from .parallel_calls import CallLedger, CooperativeIO, ParallelPause, ParallelAborted, durable_json, digest
from .prompt.build import compose_directive
from .schedule import Saturated
from .types import NoChild, Snapshot
from .worker import run_worker

CONTRACT = 'sciel-parallel-search/1'


class ParallelWavePaused(CampaignStopped):
    """Unpublished receipts are durable; continuation starts at the baseline."""


def _caused_by(error, kinds):
    """Recognize cancellation through worker/research exception wrappers."""
    seen = set()
    while error is not None and id(error) not in seen:
        if isinstance(error, kinds):
            return True
        seen.add(id(error))
        error = error.__cause__ or error.__context__
    return False


def _pause_error(error):
    return _caused_by(error, (ParallelPause, ParallelWavePaused))


def _raise_errors(errors):
    # A STOP may coincide with a dispatched provider failure. That failure must
    # remain visible and require reconciliation, even if siblings were cancelled.
    if errors:
        raise next((error for error in errors if not _caused_by(error,
            (ParallelPause, ParallelWavePaused, ParallelAborted))), errors[0])


def _status(engine, **fields):
    engine._parallel_status = {**(engine._parallel_status or {}),
        'schema': 'sciel-parallel-runtime/1', **fields}


class WaveLLM:
    def __init__(self, engine, original, io):
        self.engine, self.original, self.io = engine, original, io

    def __getattr__(self, name):
        return getattr(self.original, name)

    def generate(self, prompt, model, temperature=None, **options):
        # Capture the journal context before suspension. A sibling may begin its
        # request before this transport starts; no shared context slot is read
        # from a worker thread.
        context = self.engine._llm_trace_context(prompt)
        context.update(parallel_job=self.io.current,
                       parallel_wave=getattr(self.engine, '_parallel_wave', 0))
        provider = self.original._provider if isinstance(self.original, TracedLLM) else self.original
        trace = TracedLLM(provider, self.engine.run_dir, lambda _: dict(context))
        request = {'prompt': prompt, 'model': model, 'temperature': temperature, 'options': options,
                   'trace_context': {key: context.get(key) for key in
                                     ('role', 'reference', 'experiment_id')}}
        from .llm.client import transport_parameters
        parameters = transport_parameters(provider, model, temperature)
        if parameters is not None:
            request['transport_parameters'] = parameters
        prepare_metadata = getattr(provider, 'request_metadata', None)
        if callable(prepare_metadata):
            request['provider_request'] = prepare_metadata(prompt, model, temperature, **options)
        response = self.io.call('llm', request, lambda: trace.generate(prompt, model, temperature, **options))
        account = getattr(self.engine, '_record_llm_accounting', None)
        if account is not None:
            account(response)
        return response


def _snapshot(snapshot):
    value = asdict(snapshot)
    if value.get('tool_episode_plan') is None:
        value.pop('tool_episode_plan', None)
    if value.get('research_cells_context') is None:
        value.pop('research_cells_context', None)
    if value.get('research_targets') is None:
        value.pop('research_targets', None)
    value['code_hashes'] = sorted(value['code_hashes'])
    return value


def _intent(engine, key, plan):
    snap = plan['snap']
    return {'schema': CONTRACT, 'job_id': key, 'snapshot': _snapshot(snap),
        'parent_source_sha256': hashlib.sha256(snap.parent_code.encode()).hexdigest(),
        'selection': asdict(plan['selected']) if plan.get('selected') is not None else None,
        'attribution': {name: plan.get(name) for name in ('mode', 'mode_prop', 'eff_mode',
            'phase', 'mode_on_policy', 'op_prop', 'op_stat', 'model', 'model_prop',
            'preset', 'preset_prop', 'decision_context', 'theorist', 'assumptions', 'block', 'arm')},
        'novelty_max_resamples': engine.cfg['candidate_novelty']['max_resamples'],
        'output_repeats': engine.cfg['candidate_novelty']['output_repeats'],
        'output_resolution': engine.cfg['candidate_novelty'].get('output_resolution')}


def _save_intents(directory, intents):
    path = directory / 'jobs.json'
    value = {'schema': CONTRACT, 'jobs': intents}
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Parallel wave job intents differ on replay')
    else:
        durable_json(path, value)


def _prepare_seed(engine, island):
    from .loop_contracts import _read, paradigms
    initial = engine.cfg.get('warm_start')
    code = initial['source'] if initial else _read(engine.task.initial_program_path)
    parent = next(program for program in engine.db.islands[island].programs.values()
                  if program.code == code)
    model = engine.llm.choose_model(engine.rng.stream('prompt'))
    snap = Snapshot(island=island, op='full_rewrite', model=model,
        iteration_no=engine.iteration_no, parent_id=parent.id,
        parent_code=code, parent_score=parent.score,
        directive=('Bootstrap portfolio: write a new program using '
                   + paradigms(engine.task)[island % len(paradigms(engine.task))] + '.\n'
                   + compose_directive(engine._per_run_budget(), None, plateau=False)
                   + engine.observed_changes.block(code)),
        task_prompt=engine.task_prompt, required_entrypoints=engine.required_entrypoints)
    return {'snap': snap, 'op': 'full_rewrite', 'parent': parent, 'model': model}


def _commit_seed(engine, plan, result):
    engine.cost_total += result.cost
    ok = not isinstance(result, NoChild) and result.ok
    if ok:
        repeated = result.meta.get('remeasurement')
        program = engine.db.add(result.child_code, result.score, result.cases,
            plan['snap'].island, origin='remeasurement' if repeated else 'portfolio',
            parent_id=None if repeated else plan['parent'].id)
        engine._bind_novelty_program(program, evaluation_id=result.meta.get('evaluation_id'))
        engine._observe_program(program, result, parent=None if repeated else plan['parent'])
    engine.log.decision(policy='portfolio_seed', action='portfolio', island=plan['snap'].island,
        ok=ok, cost=result.cost,
        meta=result.meta, model_used=plan['model'])


def _wave(engine, capacity, provider, *, bootstrap=False, seed_only=False, limit=None):
    from .loop import ENGINE_CONTRACT
    from .parallel_store import begin
    wave = getattr(engine, '_parallel_wave', 0)
    transaction = begin(engine, wave)
    config = engine.cfg['parallel_search']
    kind = 'canonical_seed' if seed_only else 'portfolio' if bootstrap else 'evolution'
    allocation_before = portfolio_allocation.metadata(engine, kind)
    started = monotonic()
    records = []
    _status(engine, wave=wave, phase='bootstrap' if seed_only or bootstrap else 'generation',
        workers=config['workers'], jobs=1 if seed_only else 0,
        pending_calls=0, pending_llm_calls=0, pending_evaluations=0,
        **allocation_before,
        **({'pending_tool_calls': 0} if getattr(engine, 'research_tools', None) else {}))

    def receipt(operation, payload):
        record = {'job_call': operation['key'], 'kind': operation['kind'],
            'request_sha256': payload['request_sha256'],
            'receipt_sha256': digest({key: value for key, value in payload.items() if key != 'abandonment'}),
            'status': payload['status'], 'queue_seconds': payload['queue_seconds'],
            'elapsed_seconds': payload['elapsed_seconds'], 'started_at': payload['started_at'],
            'completed_at': payload['completed_at']}
        records.append(record)
        engine.log.decision(policy='parallel_search', action='io_receipt', wave=wave, **record)
        if payload.get('abandonment') is not None:
            engine.log.decision(policy='parallel_provider_recovery', action='abandon_unknown_request',
                wave=wave, receipt=payload['abandonment'], current_call_cost_unknown=True)

    ledger = CallLedger(transaction.directory, capacity, provider)
    research_slots = (engine.research_tools.config['research_jobs_per_wave']
                      if research_queue.enabled(engine) else 0)
    io = CooperativeIO(ledger, config['workers'] + research_slots, receipt,
                       on_state=lambda fields: _status(engine, **fields),
                       stop_requested=engine._pause_requested,
                       schedule_identity={'run_id': engine.log.run_id, 'wave': wave,
                                          'engine_contract': ENGINE_CONTRACT})
    original_llm = engine.llm
    engine.llm = WaveLLM(engine, original_llm, io)
    engine._parallel_io = io
    errors = []
    try:
        if portfolio_allocation.enabled(engine.cfg):
            # Bind the allocation choice before any background poll or paid call.
            # Replaying an active transaction must use exactly this baseline/mode.
            allocation_path = transaction.directory / 'allocation.json'
            if allocation_path.exists():
                if json.loads(allocation_path.read_text()) != allocation_before:
                    raise ValueError('Parallel portfolio allocation differs on replay')
            else:
                durable_json(allocation_path, allocation_before)
        tools = getattr(engine, 'research_tools', None)
        if not seed_only and getattr(tools, 'background_enabled', False) is True:
            # Collect against the wave baseline before freezing any generation
            # plans. Cached poll receipts keep replay independent of jobs that
            # finish later, and do not depend on island selection or stagnation.
            _status(engine, phase='research_jobs')
            _, failed = io.run([('research_jobs', lambda: tools.collect_background(io))])
            _raise_errors(list(failed.values()))
            if io.aborted:
                raise CampaignStopped('Background tool collection aborted')
            if io.paused or engine._pause_requested():
                raise ParallelWavePaused('Background tool collection paused with durable receipts')
            _status(engine, phase='bootstrap' if bootstrap else 'generation')
        if seed_only:
            _save_intents(transaction.directory, [{'job_id': 'canonical_seed', 'schema': CONTRACT,
                'task_hash': engine.task.task_hash}])
            _, failed = io.run([('canonical_seed', lambda: engine.seed(portfolio=False))])
            errors.extend(failed.values())
        else:
            plans, intents = [], []
            pending_initial = portfolio_allocation.enabled(engine.cfg) and portfolio_allocation.pending(engine)
            if not bootstrap and not pending_initial and engine.cfg.get('dynamic_islands'):
                _, failed = io.run([('spawn', engine._maybe_spawn_island)])
                if failed:
                    raise next(iter(failed.values()))
            if bootstrap:
                cursor = engine._portfolio_bootstrap
                for island in range(cursor['next_island'], min(cursor['islands'], cursor['next_island'] + config['workers'])):
                    key = f'portfolio:{island}'
                    plan = _prepare_seed(engine, island)
                    plans.append((key, plan)); intents.append(_intent(engine, key, plan))
            else:
                in_flight = {island: 0 for island in portfolio_allocation.eligible_islands(engine)
                             if not engine.incubator.holds(island)}
                programmes = set()
                parents = {}
                base_iteration = engine.iteration_no
                slots, slot = min(config['workers'], limit or config['workers']), 0
                while slot < slots:
                    programme = engine.incubator.next_island(engine, busy=programmes)
                    if programme is not None:
                        plan = engine.incubator.prepare(engine, programme, base_iteration + slot)
                        key = f'G:{base_iteration + slot + 1}'
                        plans.append((key, plan)); intents.append(_intent(engine, key, plan))
                        programmes.add(programme)
                        slot += 1
                        continue
                    eligible = [island for island in in_flight
                        if in_flight[island] < config['max_inflight_per_island']
                        and any(parents.get(pid, 0) < config['max_inflight_per_parent']
                                for pid in engine.db.islands[island].programs)]
                    if not eligible:
                        break
                    island = engine.schedule.pick_island(eligible, in_flight)
                    if island < 0:
                        break
                    excluded = {pid for pid, count in parents.items()
                                if count >= config['max_inflight_per_parent']}
                    plan = engine._prepare_step(island, excluded_parents=excluded, spawn=False,
                        iteration_no=base_iteration + slot, block_room=slots - slot)
                    plan['snap'] = deepcopy(plan['snap'])
                    for member in engine.siblings.expand(plan):
                        key = f'G:{base_iteration + slot + 1}'
                        plans.append((key, member)); intents.append(_intent(engine, key, member))
                        slot += 1
                    in_flight[island] += 1
                    parent = plan['snap'].parent_id
                    parents[parent] = parents.get(parent, 0) + 1
            research_jobs = research_queue.prepare(engine, ordinary_jobs=len(plans))
            intents.extend(_intent(engine, key, {'snap': snapshot}) for key, snapshot in research_jobs)
            _save_intents(transaction.directory, intents)
            _status(engine, jobs=len(plans) + len(research_jobs))
            engine.log.decision(policy='parallel_search', action='wave_dispatch', schema=CONTRACT,
                wave=wave, jobs=len(plans) + len(research_jobs), bootstrap=bootstrap,
                **allocation_before,
                **({'ordinary_jobs': len(plans), 'research_jobs': len(research_jobs)} if research_slots else {}),
                workers=config['workers'], commit_order=[key for key, _ in plans + research_jobs])
            functions = []
            for key, plan in plans:
                if bootstrap:
                    def execute(plan=plan):
                        return run_worker('full_rewrite', plan['snap'], engine.llm,
                            engine._evaluation_callback(plan['snap'].island),
                            dup_cap=1, base_temp=.9,
                            novelty_check=engine._candidate_novelty_callback(plan['snap'].island))
                else:
                    def execute(plan=plan):
                        return engine._execute_step(plan)
                functions.append((key, execute))
            functions.extend((key, lambda snapshot=snapshot: research_queue.execute(engine, snapshot))
                             for key, snapshot in research_jobs)
            results, failed = io.run(functions)
            errors.extend(failed.values())
            for key, plan in plans:
                if key not in results:
                    continue
                result = results[key]
                durable_json(transaction.directory / ('result-' + key.replace(':', '-') + '.json'),
                    {'type': type(result).__name__, 'result': asdict(result)})
                if bootstrap:
                    _commit_seed(engine, plan, result)
                else:
                    engine._commit_step(plan, result)
                engine.log.decision(policy='parallel_search', action='job_committed', wave=wave, job_id=key)
            for key, snapshot in research_jobs:
                if key not in results:
                    continue
                result = results[key]
                durable_json(transaction.directory / ('result-' + key.replace(':', '-') + '.json'),
                    {'type': type(result).__name__, 'result': asdict(result)})
                research_queue.commit(engine, snapshot, result)
                engine.log.decision(policy='parallel_search', action='job_committed', wave=wave, job_id=key)
            if bootstrap:
                engine._portfolio_bootstrap['next_island'] += len(plans)
            _raise_errors(errors)
            if io.aborted:
                raise CampaignStopped('Parallel I/O aborted after a terminal failure')
            if io.paused or engine._pause_requested():
                raise ParallelWavePaused('Parallel work paused with durable receipts')
            if not bootstrap and not errors and not pending_initial:
                engine._maybe_milestone()
        # All submitted work has drained. Returned receipts stay in the WAL;
        # the baseline remains authoritative until a complete transaction.
        _raise_errors(errors)
        if io.aborted:
            raise CampaignStopped('Parallel I/O aborted after a terminal failure')
        if io.paused:
            raise ParallelWavePaused('Parallel work paused with durable receipts')
        engine._parallel_wave = wave + 1
        portfolio_allocation.advance(engine, kind)
        allocation_after = portfolio_allocation.metadata(engine, kind)
        _status(engine, phase='commit')
        engine.log.decision(policy='parallel_search', action='wave_complete', wave=wave,
            elapsed_seconds=monotonic() - started, calls=len(records),
            **allocation_after,
            evaluations=sum(row['kind'] == 'evaluation' for row in records),
            **({'tool_executions': sum(row['kind'] == 'tool_execution' for row in records)}
               if getattr(engine, 'research_tools', None) else {}),
            cost_total=engine.cost_total, llm_unpriced_calls=engine.llm_unpriced_calls, B_evals=engine.B_evals)
        io.validate_resume_complete()
        transaction.commit()
        _status(engine, **allocation_after)
    except BaseException as error:
        if not _pause_error(error) and not isinstance(error, (KeyboardInterrupt, SystemExit)):
            # Eligibility must bind the entire failure set, not merely find one
            # transient receipt among unrelated coordinator or accounting errors.
            try:
                from .parallel_recovery import seal_stop
                stop_errors = list(errors)
                if not any(item is error for item in stop_errors):
                    stop_errors.append(error)
                seal_stop(transaction, stop_errors)
            except Exception:
                # Missing/unverifiable certification forbids automatic recovery;
                # preserve the original failure and every physical WAL receipt.
                pass
        transaction.restore_log()
        if _pause_error(error):
            raise ParallelWavePaused('Parallel work paused with durable receipts') from error
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
        # Never let the serial exception handler publish the partially replayed
        # in-memory state. Its authoritative baseline and paid receipts survive.
        raise CampaignStopped('Parallel wave is durably stopped: ' + type(error).__name__
                              + ': ' + str(error)[:512]) from error
    finally:
        engine.llm = original_llm
        engine._parallel_io = None
        io.close()


def run_parallel(engine):
    engine._parallel_wave = getattr(engine, '_parallel_wave', 0)
    if portfolio_allocation.enabled(engine.cfg) and getattr(engine, '_portfolio_allocation', None) is None:
        engine._portfolio_allocation = portfolio_allocation.initial()
    cfg = engine.cfg['parallel_search']
    provider = urlsplit(engine.cfg.get('llm', {}).get('api_base', 'https://local')).hostname or 'local'
    capacity = HostCapacity(cfg['capacity_dir'], llm_slots=cfg['host_llm_slots'],
        evaluator_slots=cfg['host_evaluator_slots'], research_slots=cfg.get('host_research_slots'),
        provider_limits={provider: cfg['provider_slots']})
    try:
        if not Path(engine.run_dir, 'checkpoint.json').exists():
            checkpoint.save(engine, 'parallel_initial')
        if engine.db.pop_best_id is None:
            _wave(engine, capacity, provider, seed_only=True)
        while (engine._portfolio_bootstrap is not None
               and not portfolio_allocation.enabled(engine.cfg)
               and engine._portfolio_bootstrap['next_island'] < engine._portfolio_bootstrap['islands']
               and not engine._pause_requested()):
            _wave(engine, capacity, provider, bootstrap=True)
        reason = 'budget'
        backstop = engine.cfg.get('max_iterations')
        if engine.budget is not None:
            failure_limit = engine.cfg['backstop_k'] * engine.budget
            backstop = min(backstop, failure_limit) if backstop is not None else failure_limit
        while engine.budget is None or engine.B_evals < engine.budget:
            if engine._pause_requested() or (engine._stop_at is not None and engine.B_evals >= engine._stop_at):
                reason = 'paused'; break
            if backstop is not None and engine.iteration_no >= backstop:
                reason = 'backstop'; break
            if (portfolio_allocation.enabled(engine.cfg) and portfolio_allocation.pending(engine)
                    and engine._portfolio_allocation['next_kind'] == 'portfolio'):
                _wave(engine, capacity, provider, bootstrap=True)
                continue
            limit = cfg['workers']
            if engine.budget is not None:
                limit = min(limit, engine.budget - engine.B_evals)
            if backstop is not None:
                limit = min(limit, backstop - engine.iteration_no)
            if engine._stop_at is not None:
                limit = min(limit, engine._stop_at - engine.B_evals)
            _wave(engine, capacity, provider, limit=limit)
    except ParallelWavePaused:
        # The active manifest owns this unchanged baseline. Publishing a new
        # checkpoint here would make its cached paid receipts unrecoverable.
        checkpoint.load(engine, str(Path(engine.run_dir) / 'checkpoint.json'))
        reason = 'paused'
        _status(engine, phase='paused', pending_calls=0, pending_llm_calls=0,
                pending_evaluations=0,
                **({'pending_tool_calls': 0} if getattr(engine, 'research_tools', None) else {}))
        engine.log.run_end(reason=reason, B_evals=engine.B_evals,
            iteration_no=engine.iteration_no, cost_total=engine.cost_total,
            llm_unpriced_calls=engine.llm_unpriced_calls,
            parallel_wave_pending=engine._parallel_wave)
    else:
        engine._finish(reason)
    result = {'reason': reason, 'B_evals': engine.B_evals, 'pop_best': engine.db.pop_best,
        'iteration_no': engine.iteration_no, 'cost_total': engine.cost_total,
        'llm_unpriced_calls': engine.llm_unpriced_calls,
        'n_events': engine.n_events, 'n_remeasurements': engine.n_remeasurements,
        'n_evaluation_results': engine.n_evaluation_results,
        'n_output_repeats': engine.n_output_repeats,
        'parallel_search': {'contract': CONTRACT, 'workers': cfg['workers'],
                            'waves': engine._parallel_wave}, **engine.record_stats()}
    return result
