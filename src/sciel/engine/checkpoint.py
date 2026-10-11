"""Full-state checkpoints for search resume."""

from __future__ import annotations
import json
import hashlib
import os
import uuid
from contextlib import ExitStack
from copy import deepcopy
from .checkpoint_identity import (
    IdentityMismatch,
    PROVENANCE_SCHEMA,
    _COMPONENTS,
    _SCALARS,
    _assert_novelty_anchor,
    _bootstrap_state,
    _identity,
    _validate_attempt_anchor,
    _validate_research_failures,
    check_identity,
    counters,
)
from .checkpoint_restore import _load


def save(engine, milestone, *, path=None) -> str:
    state = {
        "milestone": milestone,
        "milestones_fired": list(engine.milestones_fired),
        "lineage_fail": engine.lineage_fail,
        "eval_times": engine.eval_times,
        # A set, so it cannot ride in _SCALARS. Leaving it out made a resumed run
        # build a different prompt than the continuous one: the no-op warning is
        # keyed on this and silently vanished at every resume.
        "noop_parents": sorted(engine.noop_parents),
        "identity": _identity(engine),
        "config": engine.cfg,
    }
    if engine.cfg.get('parallel_search', {}).get('enabled'):
        state['parallel_search'] = {'schema': 'sciel-parallel-search/1',
            'wave': getattr(engine, '_parallel_wave', 0)}
        from . import portfolio_allocation
        if portfolio_allocation.enabled(engine.cfg):
            state['parallel_search']['allocation'] = portfolio_allocation.validate(
                getattr(engine, '_portfolio_allocation', None), engine.cfg,
                getattr(engine, '_portfolio_bootstrap', None))
    for name in _SCALARS:
        if name in ('n_remeasurements', 'n_remeasurement_successes', '_last_pop_improve_iteration',
                    'n_evaluation_results', 'n_output_repeats', 'llm_unpriced_calls') and not hasattr(engine, name):
            continue
        state[name] = getattr(engine, name)
    _validate_attempt_anchor(state, engine.cfg)
    _validate_research_failures(state, engine.cfg)
    if getattr(engine, "_accounting_stopped_reason", None) is not None:
        state["_accounting_stopped_reason"] = engine._accounting_stopped_reason
    if getattr(engine, 'provider_recoveries', None):
        from .recovery import validate_recoveries
        state['provider_recoveries'] = validate_recoveries(engine.provider_recoveries)
    if getattr(engine, '_portfolio_bootstrap', None) is not None:
        state['portfolio_bootstrap'] = _bootstrap_state(engine._portfolio_bootstrap, engine.cfg)
    if engine.cfg.get('candidate_novelty', {}).get('enabled'):
        state['candidate_novelty'] = engine.candidate_novelty.state()
    if getattr(engine, 'observed_changes', None) is not None:
        state['observed_changes'] = engine.observed_changes.state()
    from .research_artifacts import validate_evaluation_bindings
    from .evaluation_timing import validate as validate_evaluation_timings
    state['evaluation_timings'] = validate_evaluation_timings(
        getattr(engine, 'evaluation_timings', {}),
        program_sources={program.id: program.code for program in getattr(engine.db, '_all', {}).values()},
        program_cases={program.id: program.cases for program in getattr(engine.db, '_all', {}).values()},
        evaluation_bindings=getattr(engine, 'research_evaluation_bindings', {}))
    state['research_evaluation_bindings'] = validate_evaluation_bindings(
        getattr(engine, 'research_evaluation_bindings', {}),
        program_sources={program.id: program.code for program in getattr(engine.db, '_all', {}).values()})
    if getattr(engine, 'output_novelty', None) is not None:
        state['output_novelty'] = engine.output_novelty.state_dict()
    for name in _COMPONENTS:
        state[name] = getattr(engine, name).state()
    if getattr(engine, "preset_weights", None) is not None:
        state["preset_weights"] = engine.preset_weights.state()
    if getattr(engine, 'research', None) is not None:
        state['research'] = engine.research.state()
    if getattr(engine, 'research_tools', None) is not None:
        state['research_tools'] = engine.research_tools.state()
    if getattr(engine, 'idea', None) is not None:
        state['idea_two_stage'] = engine.idea.state()
    if getattr(engine, 'intervention', None) is not None:
        state['circle_intervention'] = engine.intervention.state()
    if getattr(engine, 'circle_generation', None) is not None:
        state['circle_generation'] = engine.circle_generation.state()
    repeat = getattr(engine, 'repeat_promotion', None)
    if repeat is not None:
        state['repeat_promotion'] = repeat.state()
    population = getattr(engine, 'repeat_population', None)
    batches = getattr(engine, 'paired_batches', None)
    if population is not None:
        from .repeat_population_support import validate_projection
        validate_projection(engine)
        state['repeat_population'] = population.state()
        state['paired_batches'] = batches.state()
    controller = getattr(engine, "controller", None)
    if controller is not None and getattr(controller, "enabled", False):
        state["controller"] = controller.state()
    if engine.plateau_shadow and engine.plateau_monitors is not None:
        state["plateau"] = {str(i): m.state() for i, m in engine.plateau_monitors.items()}
    if engine.oc_monitors is not None:
        state["oc"] = {str(i): m.state() for i, m in engine.oc_monitors.items()}
    provenance = {"schema": PROVENANCE_SCHEMA, "checkpoint_id": str(uuid.uuid4()),
                  "run_id": engine.log.run_id, "journal_anchor": engine.log.anchor(),
                  "counters": counters(engine)}
    state["provenance"] = provenance
    path = os.fspath(path) if path is not None else os.path.join(engine.run_dir, "checkpoint.json")
    tmp = path + ".tmp"
    data = json.dumps(state).encode("utf-8")
    with open(tmp, "wb") as f:
        f.write(data)
        if repeat is not None or batches is not None or engine.cfg.get('parallel_search', {}).get('enabled'):
            f.flush()
            os.fsync(f.fileno())
    os.replace(tmp, path)
    if repeat is not None or batches is not None or engine.cfg.get('parallel_search', {}).get('enabled'):
        directory = os.open(os.path.dirname(path), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        if repeat is not None:
            repeat.checkpoint_ack(state['repeat_promotion']['ledger_anchor'])
        if batches is not None:
            batches.checkpoint_ack(state['paired_batches']['ledger_anchor'])
    engine._saved_checkpoint = {"checkpoint_id": provenance["checkpoint_id"],
        "checkpoint_sha256": hashlib.sha256(data).hexdigest(), "provenance": provenance}
    return path

def load(engine, path: str, allow_identity_mismatch: bool = False) -> None:
    # A restored repeat coordinator owns a process lock before the remaining
    # components are validated. Release that newly acquired resource on any
    # later failure, even when Engine.run (and its cleanup) is never entered.
    try:
        with ExitStack() as rollback:
            _load(engine, path, allow_identity_mismatch, rollback)
            rollback.pop_all()
    except BaseException:
        close = getattr(engine, "close", None)
        if callable(close):
            close()
        raise
