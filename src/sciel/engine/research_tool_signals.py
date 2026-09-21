"""Read-only, phase-aware snapshots of existing search stagnation signals."""
from math import isfinite


def _finite(value):
    return value if type(value) in (int, float) and isfinite(value) else None


def snapshot_signals(engine, *, island, iteration_no, phase, mode, effective_mode,
                     refine, forced_operator, stall, stalled, controller_plateau):
    signals = {
        'schema': 'sciel-stagnation-signals/1', 'island_id': island,
        'iteration': iteration_no + 1, 'committed_iteration': engine.iteration_no,
        'B_evals': engine.B_evals, 'phase': phase,
        'population_stall': {
            'algorithm': 'accepted_evaluations_since_generated_population_record',
            'counter': 'B_evals', 'anchor': '_last_pop_improve',
            'last_record_B_evals': engine._last_pop_improve,
            'elapsed_evaluations': stall, 'threshold': int(engine.cfg.get('refine_stall', 8)),
            'threshold_reached': stalled,
            'scope': 'B_evals includes successful remeasurements; only ordinary generated population records reset the anchor.'},
        'search_attempt_stall': {
            'algorithm': 'committed_attempts_since_generated_population_record',
            'counter': 'iteration_no', 'anchor': '_last_pop_improve_iteration',
            'last_record_iteration': engine._last_pop_improve_iteration,
            # The planned iteration_no can include uncommitted parallel slots.
            'elapsed_attempts': engine.iteration_no - engine._last_pop_improve_iteration,
            'threshold': int(engine.cfg.get('research_tools', {}).get('attempt_stall_threshold', 8)),
            'threshold_reached': (engine.iteration_no - engine._last_pop_improve_iteration
                >= int(engine.cfg.get('research_tools', {}).get('attempt_stall_threshold', 8))),
            'scope': 'Committed ordinary search attempts, including failed evaluations, no-child outcomes and mutation-side remeasurements; bootstrap and in-flight work are excluded. Only ordinary generated population records reset the anchor.'},
        'controller_plateau': {
            'enabled': bool(engine.controller.enabled), 'active': bool(controller_plateau),
            'until_iteration': engine.controller.plateau_until.get(island, -1),
            'selected_mode': mode, 'effective_mode': effective_mode},
        'refinement': {'applied': bool(refine), 'forced_operator': bool(forced_operator)},
    }
    for name, monitors, shadow in (
            ('opportunity_cost', engine.oc_monitors, not engine.oc_shift),
            ('betting_shadow', engine.plateau_monitors, True)):
        monitor = (monitors or {}).get(island)
        value = {'enabled': monitor is not None,
                 'mode': 'shadow' if shadow else 'intervention',
                 'observation_unit': 'successful ordinary candidate admissions',
                 'phase': None, 'current_phase': False, 'detected': None}
        if monitor is not None:
            state = monitor.state()
            same_phase = state['phase'] == phase
            value.update(phase=state['phase'], current_phase=same_phase,
                         detected=bool(monitor.is_plateau()) if same_phase else None)
            keys = (('n', 'k', 'horizon', 'min_n', 'threshold') if name == 'opportunity_cost'
                    else ('t', 't_seg', 's', 'wealth', 'alpha', 'threshold', 'min_t'))
            value.update({key: _finite(state[key]) for key in keys})
            if name == 'opportunity_cost':
                value['forecast'] = _finite(monitor.forecast())
        signals[name] = value
    return signals
