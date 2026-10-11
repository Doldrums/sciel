"""Validated restoration of complete Engine state."""
from __future__ import annotations

import json
import hashlib
import os
from contextlib import ExitStack
from copy import deepcopy
from .checkpoint_identity import (
    IdentityMismatch,
    PROVENANCE_SCHEMA,
    _COMPONENTS,
    _SCALARS,
    _assert_novelty_anchor,
    _bootstrap_state,
    _validate_attempt_anchor,
    _validate_research_failures,
    counters,
)


def _load(engine, path: str, allow_identity_mismatch: bool, rollback: ExitStack) -> None:
    from . import checkpoint as _public
    with open(path, "rb") as f:
        data = f.read()
    state = json.loads(data)
    unpriced = state.get('llm_unpriced_calls', 0)
    if type(unpriced) is not int or unpriced < 0:
        raise IdentityMismatch('invalid completed unpriced LLM count')
    _validate_attempt_anchor(state, getattr(engine, 'cfg', {}))
    _validate_research_failures(state, getattr(engine, 'cfg', {}))
    parallel = state.get('parallel_search')
    if getattr(engine, 'cfg', {}).get('parallel_search', {}).get('enabled'):
        if (type(parallel) is not dict or parallel.get('schema') != 'sciel-parallel-search/1'
                or type(parallel.get('wave')) is not int or parallel['wave'] < 0):
            raise IdentityMismatch('parallel search requires a valid wave checkpoint')
    elif parallel is not None:
        raise IdentityMismatch('cannot resume a parallel experiment in serial mode')
    remeasurements = state.get('n_remeasurements', 0)
    if type(remeasurements) is not int or remeasurements < 0:
        raise IdentityMismatch('checkpoint n_remeasurements must be a nonnegative integer')
    remeasurement_successes = state.get('n_remeasurement_successes', 0)
    if (type(remeasurement_successes) is not int
            or not 0 <= remeasurement_successes <= remeasurements):
        raise IdentityMismatch('checkpoint n_remeasurement_successes must be a nonnegative integer '
                               'no greater than n_remeasurements')
    bootstrap = _bootstrap_state(state.get('portfolio_bootstrap'), getattr(engine, 'cfg', {}))
    from . import portfolio_allocation
    try:
        allocation = portfolio_allocation.validate(
            parallel.get('allocation') if parallel else None, getattr(engine, 'cfg', {}), bootstrap)
        if allocation is not None and bootstrap is None and state.get('db', {}).get('programs'):
            raise ValueError('interleaved portfolio is missing its committed attempt cursor')
    except ValueError as exc:
        raise IdentityMismatch(str(exc)) from exc
    from .research_artifacts import validate_evaluation_bindings
    from .evaluation_timing import validate as validate_evaluation_timings
    try:
        research_evaluation_bindings = validate_evaluation_bindings(
            state.get('research_evaluation_bindings', {}),
            program_sources={row['id']: row['code'] for row in state.get('db', {}).get('programs', [])})
        evaluation_timings = validate_evaluation_timings(
            state.get('evaluation_timings', {}),
            program_sources={row['id']: row['code'] for row in state.get('db', {}).get('programs', [])},
            program_cases={row['id']: row['cases'] for row in state.get('db', {}).get('programs', [])},
            evaluation_bindings=research_evaluation_bindings)
        if any(state.get('eval_times', {}).get(ident) != row[
                'mean_evaluation_stage_wall_seconds_per_reported_trial']
               for ident, row in evaluation_timings.items()):
            raise ValueError('evaluation timing cache differs from source-bound timing records')
    except (ValueError, TypeError, KeyError) as exc:
        raise IdentityMismatch('invalid saved evaluator program bindings: ' + str(exc)) from exc
    from .recovery import ProviderRecoveryError, validate_recoveries
    try:
        provider_recoveries = validate_recoveries(state.get('provider_recoveries', []))
    except ProviderRecoveryError as exc:
        raise IdentityMismatch(str(exc)) from exc
    accounting_stop = state.get("_accounting_stopped_reason")
    if accounting_stop is not None and (type(accounting_stop) is not str or not accounting_stop.strip()):
        raise IdentityMismatch("checkpoint accounting stop reason is invalid")
    provenance = state.get("provenance")
    if provenance is not None and (not isinstance(provenance, dict)
            or provenance.get("schema") != PROVENANCE_SCHEMA):
        raise IdentityMismatch("unsupported checkpoint provenance schema")
    if provenance is not None:
        expected = {**{name: state[name] for name in _SCALARS if name in state},
                    "db_counter": state.get("db", {}).get("counter")}
        if provenance.get("counters") != expected:
            raise IdentityMismatch("checkpoint provenance counters disagree with state")
    problems = _public.check_identity(engine, state)
    saved_backend=(state.get('identity') or {}).get('evaluation_backend')
    current_backend=getattr(engine, 'evaluation_backend', None)
    if saved_backend is not None or current_backend is not None:
        if current_backend is None or saved_backend != current_backend.identity or problems:
            raise IdentityMismatch('cannot force or change candidate-only evaluation identity; start a fresh run')
    tools = getattr(engine, 'research_tools', None)
    restored_tools = None
    tools_bound = (tools is not None or 'research_tools' in state
                   or bool((state.get('identity') or {}).get('research_tools'))
                   or bool(((state.get('config') or {}).get('research_tools') or {}).get('enabled')))
    if tools_bound:
        if (tools is None or 'research_tools' not in state or problems
                or not (state.get('identity') or {}).get('research_tools') or not provenance):
            raise IdentityMismatch('cannot force or change research tools checkpoint identity/state')
        try:
            from copy import copy
            restored_tools = copy(tools)
            restored_tools.load_state(state['research_tools'], expected_run_id=provenance['run_id'])
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise IdentityMismatch(f'cannot resume research tools: {exc}') from exc
    observed = getattr(engine, 'observed_changes', None)
    restored_observed = None
    if observed is not None or 'observed_changes' in state or (state.get('identity') or {}).get('observed_changes'):
        if (observed is None or 'observed_changes' not in state
                or (state.get('identity') or {}).get('observed_changes') != observed.identity):
            raise IdentityMismatch('cannot drop or change observed change feedback history; start a new run')
        try:
            restored_observed = deepcopy(observed)
            restored_observed.load_state(state['observed_changes'])
        except (ValueError, KeyError, TypeError) as exc:
            raise IdentityMismatch(f'cannot resume observed change feedback: {exc}') from exc
    novelty = getattr(engine, 'candidate_novelty', None)
    novelty_enabled = bool(getattr(engine, 'cfg', {}).get('candidate_novelty', {}).get('enabled'))
    novelty_bound = (novelty_enabled or 'candidate_novelty' in state
        or bool((state.get('identity') or {}).get('candidate_novelty'))
        or bool(((state.get('config') or {}).get('candidate_novelty') or {}).get('enabled')))
    restored_novelty = None
    if novelty_bound:
        # Population/archive snapshots cannot reconstruct evicted source history.
        # Never silently begin an incomplete registry or force it across identity.
        if (not novelty_enabled or novelty is None or 'candidate_novelty' not in state
                or not (state.get('identity') or {}).get('candidate_novelty') or problems):
            raise IdentityMismatch('cannot force or change candidate novelty checkpoint identity/state; start a new run')
        try:
            _assert_novelty_anchor(engine.log.path, (provenance or {}).get('journal_anchor'))
            restored_novelty = deepcopy(novelty)
            restored_novelty.load_state(state['candidate_novelty'])
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise IdentityMismatch(f'cannot resume candidate novelty: {exc}') from exc
    outputs = getattr(engine, 'output_novelty', None)
    restored_outputs = None
    if outputs is not None or 'output_novelty' in state or (state.get('identity') or {}).get('output_novelty'):
        if (outputs is None or 'output_novelty' not in state or problems
                or (state.get('identity') or {}).get('output_novelty') != outputs.identity):
            raise IdentityMismatch('cannot drop or change output novelty history; start a new run')
        try:
            restored_outputs = deepcopy(outputs)
            from .output_measurements import restore_measurement
            restored_outputs.load_state_dict(state['output_novelty'], measurement_validator=
                lambda evaluation_id, identity, binding: restore_measurement(
                    engine, evaluation_id, identity, binding, run_id=provenance['run_id']))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise IdentityMismatch(f'cannot resume output novelty: {exc}') from exc
        for name in ('n_evaluation_results', 'n_output_repeats'):
            if type(state.get(name)) is not int or state[name] < 0:
                raise IdentityMismatch('output novelty requires measured evaluation counters')
        if state['n_output_repeats'] > state['n_evaluation_results']:
            raise IdentityMismatch('output repeat count exceeds evaluation results')
        measured = restored_outputs.summary()
        if (measured['evaluations'] != state['n_evaluation_results']
                or measured['exact_recorded_repeat'] + measured.get('non_improving_near_repeat', 0)
                    != state['n_output_repeats']):
            raise IdentityMismatch('output novelty history differs from measured evaluation counters')
    generation = getattr(engine, 'circle_generation', None)
    generation_bound = (generation is not None or 'circle_generation' in state
        or bool((state.get('identity') or {}).get('circle_generation'))
        or bool(((state.get('config') or {}).get('circle_generation') or {}).get('enabled')))
    restored_generation, restored_research = None, None
    if generation_bound:
        if (generation is None or 'circle_generation' not in state or problems
                or not (state.get('identity') or {}).get('circle_generation') or not provenance):
            raise IdentityMismatch('cannot force or change circle generation checkpoint identity/state')
        try:
            from copy import copy
            from .journal_anchor import assert_no_uncheckpointed_research
            if 'research' not in state:
                raise ValueError('checkpoint has no full research state')
            assert_no_uncheckpointed_research(engine.log.path, provenance.get('journal_anchor'))
            restored_research = copy(engine.research)
            restored_research.load_state(state['research'])
            # Generation restore only replaces its own counters/lists after
            # validation. Check source proofs and the anchored journal on a
            # detached component before loading Research or taking repeat locks.
            restored_generation = copy(generation)
            restored_generation.research = restored_research
            restored_generation.load_state(state['circle_generation'])
            restored_generation.assert_journal(engine.log.path, provenance.get('journal_anchor'))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise IdentityMismatch(f'cannot resume circle generation: {exc}') from exc
    intervention = getattr(engine, 'intervention', None)
    intervention_bound = (intervention is not None or 'circle_intervention' in state
        or bool((state.get('identity') or {}).get('circle_intervention'))
        or bool(((state.get('config') or {}).get('circle_intervention') or {}).get('enabled')))
    if intervention_bound:
        if (intervention is None or 'circle_intervention' not in state or problems
                or not (state.get('identity') or {}).get('circle_intervention') or not provenance):
            raise IdentityMismatch('cannot force or change circle intervention checkpoint identity/state')
    repeat = getattr(engine, 'repeat_promotion', None)
    repeat_bound = (repeat is not None or 'repeat_promotion' in state
                    or bool((state.get('identity') or {}).get('repeat_promotion'))
                    or bool(((state.get('config') or {}).get('repeat_promotion') or {}).get('enabled')))
    if repeat_bound:
        if repeat is None or 'repeat_promotion' not in state or problems:
            raise IdentityMismatch('cannot force or change repeat promotion checkpoint identity/state')
        if not (state.get('identity') or {}).get('repeat_promotion') or not provenance:
            raise IdentityMismatch('repeat promotion requires bound checkpoint identity and provenance')
        try:
            from .repeat_support import assert_search_anchor
            assert_search_anchor(engine.log.path, provenance.get('journal_anchor'))
            repeat.load_state(state['repeat_promotion'])
            rollback.callback(repeat.close)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise IdentityMismatch(f'cannot resume repeat promotion: {exc}') from exc
    population = getattr(engine, 'repeat_population', None)
    batches = getattr(engine, 'paired_batches', None)
    population_bound = (population is not None or 'repeat_population' in state or 'paired_batches' in state
        or bool((state.get('identity') or {}).get('repeat_population'))
        or bool(((state.get('config') or {}).get('repeat_population') or {}).get('enabled')))
    restored_population = None
    if population_bound:
        if (population is None or batches is None or 'repeat_population' not in state
                or 'paired_batches' not in state or problems or not provenance
                or not (state.get('identity') or {}).get('repeat_population')):
            raise IdentityMismatch('cannot force or change repeat population checkpoint identity/state')
        try:
            from .repeat_support import assert_search_anchor
            from .repeat_population_db import RepeatPopulationDatabase
            assert_search_anchor(engine.log.path, provenance.get('journal_anchor'))
            batches.load_state(state['paired_batches'])
            rollback.callback(batches.close)
            restored_db = RepeatPopulationDatabase(len(engine.db.islands), engine.db.pop_cap)
            restored_db.load_state(state['db'])
            restored_population = deepcopy(population)
            restored_population.load_state(state['repeat_population'], restored_db._all)
            restored_population.assert_batches(batches.terminal_batches())
            saved_batches = state['paired_batches']['data']['batches']
            pending_plan = (saved_batches[-1]['plan'] if saved_batches
                            and saved_batches[-1]['decision'] is None else None)
            restored_population.assert_pending(pending_plan)
            from .repeat_population_support import validate_history
            validate_history(restored_population, restored_db, state['seed_score'],
                             engine.log.path, provenance['journal_anchor'])
            for island in restored_db.islands:
                if list(island.programs) != restored_population.active_ids(island.idx):
                    raise ValueError('stored DB projection differs from repeat membership')
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise IdentityMismatch(f'cannot resume repeat population: {exc}') from exc
    if problems and not allow_identity_mismatch:
        raise IdentityMismatch(
            "refusing to resume: this run was produced under a different identity.\n  "
            + "\n  ".join(problems)
            + "\nThe archived programs and the new children would not be comparable. "
              "Start a new run, or pass --force-identity if you know why the change is safe.")
    research = getattr(engine, 'research', None)
    if research is not None:
        if 'research' not in state:
            raise IdentityMismatch('checkpoint has no full research state; start a new run')
        try:
            from .journal_anchor import assert_no_uncheckpointed_research
            assert_no_uncheckpointed_research(engine.log.path, (provenance or {}).get('journal_anchor'))
            if restored_research is None:
                research.load_state(state['research'])
            else:
                research.__dict__.update(restored_research.__dict__)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise IdentityMismatch(f'cannot resume research: {exc}') from exc
    if getattr(engine, 'idea', None) is not None:
        try:
            engine.idea.load_state(state['idea_two_stage'])
        except (ValueError, KeyError, TypeError) as exc:
            raise IdentityMismatch(f'cannot resume two-stage idea: {exc}') from exc
    if intervention is not None:
        try:
            intervention.load_state(state['circle_intervention'])
            intervention.assert_journal(engine.log.path, provenance.get('journal_anchor'))
        except (ValueError, KeyError, TypeError, OSError) as exc:
            raise IdentityMismatch(f'cannot resume circle intervention: {exc}') from exc
    if restored_generation is not None:
        # Retain the Engine's component object and shared Research reference;
        # install only the already validated detached generation state.
        restored_generation.research = research
        generation.__dict__.update(restored_generation.__dict__)
    if restored_tools is not None:
        tools.__dict__.update(restored_tools.__dict__)
    engine.llm_unpriced_calls = unpriced
    for name in _SCALARS:
        if name in state:
            setattr(engine, name, state[name])
    engine.n_remeasurements = remeasurements
    engine.n_remeasurement_successes = remeasurement_successes
    engine._accounting_stopped_reason = accounting_stop
    engine.provider_recoveries = provider_recoveries
    engine._portfolio_bootstrap = bootstrap
    engine._portfolio_allocation = allocation
    engine.research_evaluation_bindings = research_evaluation_bindings
    if parallel is not None:
        engine._parallel_wave = parallel['wave']
    engine.milestones_fired = set(state["milestones_fired"])
    engine.lineage_fail = {k: v for k, v in state["lineage_fail"].items()}
    # Legacy unscoped values cannot be relabelled as per-trial measurements.
    engine.evaluation_timings = evaluation_timings
    engine.eval_times = {ident: row['mean_evaluation_stage_wall_seconds_per_reported_trial']
                         for ident, row in evaluation_timings.items()}
    engine.noop_parents = set(state.get("noop_parents", []))
    if restored_observed is not None:
        observed.__dict__.update(restored_observed.__dict__)
    for name in _COMPONENTS:
        if name in state:
            getattr(engine, name).load_state(state[name])
    if restored_novelty is not None:
        # Keep references held by the worker's novelty callback valid.
        novelty.__dict__.update(restored_novelty.__dict__)
    if restored_outputs is not None:
        outputs.__dict__.update(restored_outputs.__dict__)
    if restored_population is not None:
        restored_population.load_state(state['repeat_population'], engine.db._all)
        engine.repeat_population = restored_population
    if getattr(engine, "preset_weights", None) is not None:
        if "preset_weights" not in state:
            raise IdentityMismatch("checkpoint is missing enabled preset selection state")
        engine.preset_weights.load_state(state["preset_weights"])
    if engine.kb_on and "kb" in state:
        for name in ("kb", "base_rate"):
            getattr(engine, name).load_state(state[name])
    if engine.c2_on and "branch_cal" in state:
        engine.branch_cal.load_state(state["branch_cal"])
    controller = getattr(engine, "controller", None)
    if controller is not None and getattr(controller, "enabled", False) and "controller" in state:
        controller.load_state(state["controller"])
    if engine.plateau_shadow and engine.plateau_monitors is not None and "plateau" in state:
        from ..stats import PlateauMonitor
        for i, st in state["plateau"].items():
            idx = int(i)
            if idx not in engine.plateau_monitors:
                engine.plateau_monitors[idx] = PlateauMonitor()
            engine.plateau_monitors[idx].load_state(st)
    if engine.oc_monitors is not None and "oc" in state:
        pol = engine.cfg["policy"]
        for i, st in state["oc"].items():
            idx = int(i)
            if idx not in engine.oc_monitors:      # island spawned mid-run (dynamic_islands)
                from ..stats import OpportunityCostMonitor
                engine.oc_monitors[idx] = OpportunityCostMonitor(
                    threshold=pol["plateau_threshold"], horizon=pol["plateau_horizon"],
                    min_n=pol["plateau_min_n"])
            engine.oc_monitors[idx].load_state(st)
    engine._restored_checkpoint = {
        "checkpoint_id": (provenance or {}).get("checkpoint_id"),
        "checkpoint_sha256": hashlib.sha256(data).hexdigest(),
        "path": os.path.abspath(path), "provenance": provenance,
        "restored_counters": counters(engine), "identity_mismatches": problems,
        "ancestry": "bound" if (provenance or {}).get("journal_anchor") else "legacy_unbound"}
