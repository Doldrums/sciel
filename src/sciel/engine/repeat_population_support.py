"""Opt-in Engine trajectory controlled by fresh paired population decisions.

The same worker/evaluator/research machinery observes ordinary outcomes. This
route deliberately has no legacy single-score admission, reward or parent tilt.
"""
from __future__ import annotations

from copy import deepcopy
from functools import partial
import hashlib
import json
from pathlib import Path

from ..campaign.budget import CampaignStopped
from ..evidence.measurement import metric_context
from . import checkpoint
from .db import normalized_hash
from .paired_batches import CONTRACT as BATCH_CONTRACT, PairedBatches
from .repeat_population import RepeatPopulation
from .repeat_support import (validate_circle_task, durable_write,
                             circle_initial_source_policy, traced_execution_policy)
from .prompt.build import compose_directive
from .types import NoChild, Snapshot
from .worker import run_worker
from .runtime import observed_stage


def create_batches(cfg, task, directory, python):
    from ..eval.repeat_draw import adapter_identity, draw
    policy = circle_initial_source_policy(cfg)
    initial, evaluator = validate_circle_task(task, initial_source_policy=policy)
    trace_policy = traced_execution_policy(policy)
    repeat = cfg['repeat_population']
    runtime = repeat.get('runtime', {'backend': 'native'})
    root = Path(__file__).resolve().parents[1]
    identity = {'contract': BATCH_CONTRACT, 'task': task.qualified_name, 'task_hash': task.task_hash,
        'canonical_initial_source_sha256': hashlib.sha256(initial).hexdigest(),
        'task_evaluator_sha256': hashlib.sha256(evaluator).hexdigest(),
        'task_metric': metric_context(task), 'adapter': adapter_identity(python=python, runtime=runtime),
        'integration_sha256': {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(root.rglob('*.py'))}, **trace_policy}
    return PairedBatches(directory, repeat, identity,
        partial(draw, python=python, runtime=runtime, expected_identity=identity['adapter'],
                **({'restart_trace': True} if trace_policy else {})))


def initialize(engine):
    batches = engine.paired_batches
    engine.repeat_population = RepeatPopulation(
        {'num_islands': engine.cfg['num_islands'], 'pop_cap': engine.cfg.get('pop_cap', 50)},
        {'task_metric': engine.research.metric, 'adapter_identity': batches.identity['adapter'],
         'limits': batches._receipt_limits(), 'task_identity': {key: batches.identity[key] for key in
            ('task', 'task_hash', 'canonical_initial_source_sha256', 'task_evaluator_sha256')}})


def save(engine, phase):
    path = checkpoint.save(engine, phase)
    engine.log.checkpoint(milestone=phase, iteration_no=engine.iteration_no,
                         B_evals=engine.B_evals, path=path, **engine._saved_checkpoint)


def validate_history(population, db, seed_score, log_path, anchor=None):
    """Bind ordinary seed observations even before the first terminal batch."""
    state = population.state()['projection']
    if population.initialized:
        if set(state['history']) != set(db._all):
            raise ValueError('ordinary history differs from repeat component history')
    elif db._all and (len(db._all) != len(db.islands)
                     or any(p.origin != 'seed' for p in db._all.values())):
        raise ValueError('pre-bootstrap history must contain only one seed per island')
    data = Path(log_path).read_bytes()
    if anchor is not None:
        data = data[:anchor['byte_offset']]
    rows = [json.loads(line) for line in data.splitlines()]
    first_generation = next((i for i, row in enumerate(rows)
        if row.get('event') == 'repeat_search'
        and row.get('action') == 'population_generation_intent'), len(rows))
    source_sha = population.identity['bindings']['task_identity']['canonical_initial_source_sha256']
    results = [row for row in rows[:first_generation]
        if row.get('event') == 'decision' and row.get('policy') == 'evaluation'
        and row.get('action') == 'result' and row.get('schema') == 'sciel-evaluation-result/1'
        and row.get('source_sha256') == source_sha
        and row.get('iteration_no_before') == 0 and row.get('B_evals_before') == 0]
    seeds = sorted((p for p in db._all.values() if p.origin == 'seed'), key=lambda p: p.island)
    if not seeds:
        if db._all or any(row.get('ok') for row in results):
            raise ValueError('ordinary seed history missing its measured baseline')
        return
    if len(seeds) != len(db.islands) or len(results) != 1 or results[0].get('ok') is not True:
        raise ValueError('ordinary seed history lacks one measured baseline')
    result = results[0]
    def encoded(value):
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    if encoded(seed_score) != encoded(result['score']):
        raise ValueError('ordinary seed score differs from the journal')
    for island, program in enumerate(seeds):
        if (program.id != f'p{island + 1}' or program.island != island
                or program.parent_id is not None or type(program.gen) is not int or program.gen != 0
                or hashlib.sha256(program.code.encode()).hexdigest() != source_sha
                or encoded(program.score) != encoded(result['score'])
                or encoded(program.cases) != encoded(result['cases'])):
            raise ValueError('ordinary seed observation differs from the journal')


def validate_projection(engine):
    population = engine.repeat_population
    state = population.state()['projection']
    batches = engine.paired_batches.state()['data']['batches']
    plan = None
    if batches:
        last = batches[-1]
        if last['decision'] is None or last['decision']['decision_id'] not in state['applied']:
            plan = last['plan']
    population.assert_pending(plan)
    population.assert_batches(engine.paired_batches.terminal_batches())
    validate_history(population, engine.db, engine.seed_score, engine.log.path)
    for island in engine.db.islands:
        if list(island.programs) != population.active_ids(island.idx):
            raise ValueError('DB projection differs from repeat membership')


def sync(engine):
    """Pure terminal projection; no journal tail between decision and checkpoint."""
    population, batches = engine.repeat_population, engine.paired_batches
    packets = batches.terminal_batches()
    for packet in packets:
        if packet['kind'] == 'bootstrap' and not population.initialized:
            seeds = [p for p in engine.db._all.values() if p.origin == 'seed']
            population.bootstrap(seeds, packet)
        else:
            population.apply(packet)
    population.assert_batches(packets)
    engine.db.project(population)
    engine._sync_repeat_comparisons(batches)


@observed_stage("evaluating")
def run_pending(engine):
    sync(engine)
    if engine.paired_batches.pending and not engine.paired_batches.stopped_reason:
        save(engine, 'population_before_draws')
        engine.paired_batches.run_pending()
        sync(engine)
    save(engine, 'population_after_draws')
    complete_research(engine)


@observed_stage("research")
def complete_research(engine):
    """Flush only an owed ordinary reflection after a known repeat boundary.

    The preceding checkpoint owns the imported terminal facts. A crash after
    auxiliary dispatch but before the following checkpoint leaves a research
    journal tail, which checkpoint loading refuses to replay.
    """
    research, batches = engine.research, engine.paired_batches
    if (batches.pending or batches.stopped_reason
            or engine.repeat_population.stopped_reason):
        return
    interval = research.config['reflection_interval']
    if interval and research.outcomes - research.reflected_through >= interval:
        research.complete_step()
        save(engine, 'population_after_reflection')


@observed_stage("generating")
def step(engine, island):
    population = engine.repeat_population
    selected = population.select(island, engine.rng.stream('parent'))
    parent = engine.db.get(selected['parent_id'])
    op = engine.cfg['op_fixed']
    model = engine.llm.choose_model(engine.rng.stream('prompt'))
    context = engine.research.context(engine.db, parent, island, engine.iteration_no + 1)
    heads = population.state()['projection']['evidence_heads']
    repeat_evidence = {role: deepcopy(heads[selected[key]]) for role, key in
        (('parent', 'parent_source_sha256'), ('comparator', 'comparator_source_sha256'))}
    directive = compose_directive(engine._per_run_budget(), engine.eval_times.get(parent.id),
                                  plateau=False, parent_timing_scope='evaluation_stage_mean')
    directive += engine.observed_changes.block(parent.code)
    directive += ('\nREPEAT_SELECTION_EVIDENCE_JSON:\n' + json.dumps(repeat_evidence, sort_keys=True,
        separators=(',', ':'), allow_nan=False) + '\nThese are separate repeated raw/exact-certified observations. '
        'Parent selection ranks are a heuristic. Forecast the ordinary child score minus the measured '
        'ordinary parent_score in RESEARCH_CONTEXT_JSON, not these repeat metrics or comparator score.')
    snap = Snapshot(island=island, op=op, model=model, iteration_no=engine.iteration_no,
        parent_id=parent.id, parent_code=parent.code, parent_score=parent.score,
        inspirations=[{'code': p.code, 'score': p.score, 'card': p.card} for p in
            (engine.db.get(pid) for pid in population.inspirations(island, parent.id))],
        directive=directive, kb_lessons=context['lesson_texts'],
        code_hashes={normalized_hash(p.code) for p in engine.db._all.values()},
        escalation_seed=engine.rng.derive_seed('escalation'), task_prompt=engine.task_prompt,
        bet_enabled=False, required_entrypoints=engine.required_entrypoints, research_context=context)
    # A crash after this fsynced intent cannot replay generation from an older
    # checkpoint. The exact prospective selection precedes the provider request.
    engine.log.repeat_search(action='population_generation_intent', iteration_no=engine.iteration_no,
        B_evals=engine.B_evals, island=island, selection=selected, operator=op, model=model,
        island_assignment='deterministic_round_robin', parent_propensity=selected['parent_propensity'])
    options = {'research': engine.research}
    if engine.circle_generation is not None:
        options['circle_generation'] = engine.circle_generation
    result = run_worker(op, snap, engine.llm, engine._evaluate, dup_cap=1,
                        base_temp=engine.cfg.get('base_temp', .7), **options)
    engine.iteration_no += 1
    engine.cost_total += result.cost
    reason, child, staged = None, None, None
    if isinstance(result, NoChild):
        engine.n_nochild += 1
        engine.health = 1
        reason = result.reason
        population.finish_attempt(reason)
    elif not result.ok:
        engine.n_eval_fail += 1
        engine.health = 1
        reason = 'eval_failed'
        population.finish_attempt(reason)
    else:
        # Record the ordinary observation once. This cannot activate or evict it.
        pre_best = engine.db.pop_best
        child = engine.db.add(result.child_code, result.score, result.cases, island,
            'gen', parent_id=parent.id, card=(result.meta.get('hypothesis') or {}).get('text', '')[:200])
        engine._observe_program(child, result, parent=parent)
        engine.B_evals += 1
        engine.n_records += int(result.score > pre_best)
        engine.health = 0
        from .evaluation_timing import trial_count
        engine._observed_trials = trial_count(result.cases.get('n_trials')) or 1
        binding = engine._repeat_research_binding(child.id, result.meta.get('research_id'))
        staged = population.stage(child, selected, attribution={**binding, 'operator': op,
            'model': model, 'iteration': engine.iteration_no, 'operator_propensity': 1.,
            'parent_propensity': selected['parent_propensity']})
        reason = 'pending_comparison' if staged['eligible'] else staged['reason']
    engine.log.iteration(iteration_no=engine.iteration_no, B_evals=engine.B_evals,
        island=island, child=child.id if child else None, reason=reason, parent_id=parent.id,
        model_used=model, model_propensity=engine._model_propensity(model),
        mode='repeat_population', parent_propensity=selected['parent_propensity'], op_propensity=1.,
        cost=result.cost, meta=result.meta,
        **({'score': result.score, 'cases': result.cases} if child else {}))
    # The ordinary conclusion keeps its original forecast/measurement binding.
    # Defer the due reflection until the terminal comparison is imported. If
    # this conclusion stops, no draw or population fallback occurs.
    engine.research.complete_step(reflect=False)
    if staged and staged['eligible']:
        comparator = engine.db.get(selected['comparator_program_id'])
        planned = engine.paired_batches.plan_comparison(child.code, comparator.code, child.id,
                                                        context=staged['context'])
        if not planned:
            population.stop('comparison_not_reserved')
        else:
            run_pending(engine)
    else:
        save(engine, 'population_engineering_outcome')
        complete_research(engine)


def run(engine):
    from .loop import ENGINE_CONTRACT, _model_names
    population, batches = engine.repeat_population, engine.paired_batches
    engine.log.run_start(seed=engine.cfg['random_seed'], budget=engine.budget,
        counters=checkpoint.counters(engine), restored_checkpoint=getattr(engine, '_restored_checkpoint', None),
        task=engine.task.qualified_name, task_hash=engine.task.task_hash, engine_contract=ENGINE_CONTRACT,
        num_islands=engine.cfg['num_islands'], schedule_policy='round_robin',
        population_policy=population.identity, bet_enabled=False, models=_model_names(engine.llm),
        descriptor=engine.descriptor_info, known_best=engine.task.known_best, tiers=list(engine.task.tiers),
        config=engine.cfg)
    if engine.research.stopped_reason:
        raise CampaignStopped(engine.research.stopped_reason)
    batches.start(engine.log.run_id)
    try:
        # A saved pending plan may already have a durable terminal journal tail.
        run_pending(engine)
        if engine.db.pop_best_id is None:
            engine.log.repeat_search(action='population_seed_intent',
                source_sha256=batches.identity['canonical_initial_source_sha256'])
            engine.seed()  # One ordinary evaluation, shared across static seed IDs.
        if not population.initialized and batches.plan_bootstrap(Path(engine.task.initial_program_path).read_text()):
            run_pending(engine)
        reason = 'budget'
        failure_limit = engine.cfg['backstop_k'] * engine.budget
        explicit_limit = engine.cfg.get('max_iterations')
        backstop = min(failure_limit, explicit_limit) if explicit_limit is not None else failure_limit
        while engine.B_evals < engine.budget:
            if population.stopped_reason or batches.stopped_reason or not population.ready:
                reason = 'repeat_stopped'; break
            if engine._pause_requested() or (engine._stop_at is not None and engine.B_evals >= engine._stop_at):
                reason = 'paused'; break
            if engine.iteration_no >= backstop:
                reason = 'backstop'; break
            if not batches.can_compare():
                population.stop('full_comparison_budget_unavailable')
                reason = 'repeat_budget'; break
            island = engine.iteration_no % engine.cfg['num_islands']
            step(engine, island)
        if population.stopped_reason or batches.stopped_reason:
            reason = 'repeat_budget' if population.stopped_reason == 'full_comparison_budget_unavailable' else 'repeat_stopped'
        engine._finish(reason)
    except CampaignStopped:
        if not population.stopped_reason:
            population.stop('research_stopped')
        engine._finish('research_stopped')
        raise
    return {'reason': reason, 'B_evals': engine.B_evals, 'iteration_no': engine.iteration_no,
        'pop_best': engine.db.pop_best, 'cost_total': engine.cost_total, 'n_events': engine.n_events,
        **engine.record_stats(), 'research': engine.research.summary(),
        'repeat_population': engine._repeat_population_export,
        **({'circle_generation': engine.circle_generation.summary()} if engine.circle_generation else {})}


def export(engine):
    observed = engine.db.get(engine.db.pop_best_id) if engine.db.pop_best_id else None
    result = engine.paired_batches.export(observed)
    result['population'] = engine.repeat_population.summary()
    result['population']['ranking'] = {str(i): engine.repeat_population.ranked(i)
                                     for i in range(engine.cfg['num_islands'])}
    # The service knows only reserved batches. A research failure can leave a
    # staged candidate with no batch, so its coverage cannot describe the search.
    result['repeat_summary']['coverage_scope'] = 'reserved_batches_only'
    result['coverage_complete'] = bool(result['repeat_summary']['coverage_complete']
        and engine.repeat_population.ready and not engine.repeat_population.pending)
    result['status'] = (engine.repeat_population.stopped_reason
        or ('population_pending' if engine.repeat_population.pending else
            'population_ready' if engine.repeat_population.ready else result['status']))
    result['scope'] = ('observed_search_best is ordinary history, including rejected candidates; '
        'active membership uses repeats; confirmed output remains bootstrap until a separate publication comparison')
    confirmed = result['confirmed_output_incumbent']
    if confirmed is not None:
        target = Path(engine.run_dir) / 'confirmed_program.py'
        durable_write(target, confirmed['code'].encode())
        result['confirmed_output_incumbent'] = {k: v for k, v in confirmed.items() if k != 'code'}
        result['confirmed_output_incumbent']['path'] = str(target.resolve())
    else:
        target = Path(engine.run_dir) / 'confirmed_program.py'
        if target.exists():
            raise ValueError('unbound confirmed program exists without a publication source')
    durable_write(Path(engine.run_dir) / 'repeat_population.json',
                  (json.dumps(result, indent=2, allow_nan=False) + '\n').encode())
    return result
