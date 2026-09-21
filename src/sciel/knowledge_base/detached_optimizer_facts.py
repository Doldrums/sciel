"""Retained optimizer and matched-input observations from authenticated receipts.

This reducer runs no candidate, helper or solver. It declares no forecast and
does not credit a local interval gain to a particular intervention. Its center
Hausdorff distance is distinct from the existing restart result_changed rule.
"""
from __future__ import annotations

import copy
import math

from .catalogs import canonical_sha
from .paired_comparisons import validate_comparison
from ..hypothesis import circle_optimizer_traced_body as builder
from ..hypothesis import restart_motion as motion
from ..hypothesis import restart_motion_observations as observations
from ..eval.packing_geometry import validate_packing

SCHEMA = 'sciel-detached-optimizer-facts/1'
INPUTS = ('best', 'initial', 'noise', 'ordinary')
SCOPE = ('Retained scheduled-call prefix only. Center Hausdorff compares the optimizer '
    'result with incoming best; it is not result_changed, an initializer measurement, '
    'a forecast verdict or basin proof. Local gain is terminal incumbent minus '
    'optimizer-entered incumbent, not causal credit or final paired improvement. '
    'Matched inputs require all four captured input vectors to be byte-identical '
    'at the same seed, ordinal and restart index. Later trajectories and elapsed '
    'budgets can differ. Success, tolerance validity and strict validity differ. '
    'Omitted or unavailable snapshots remain unknown; full-run counts do not reveal '
    'unrecorded geometry. No native outcome, lesson, promotion or knowledge benefit.')


def _hausdorff(left, right):
    a, b = [list(zip(v[:52:2], v[1:52:2])) for v in (left, right)]
    value = max(max(min(math.dist(p, q) for q in b) for p in a),
                max(min(math.dist(q, p) for p in a) for q in b))
    return value if math.isfinite(value) else None


def _draw(receipt, *, role, seed, group, scaffold):
    source = group[role + '_source_sha256']
    value = dict(role=role, seed=seed, source_sha256=source,
        receipt_sha256=None if receipt is None else canonical_sha(receipt),
        receipt_status='not_dispatched' if receipt is None else receipt['status'],
        trace_status='unavailable', reason='no_trusted_terminal_receipt',
        trace_sha256=None, observation_sha256=None, prefix=None, run=None, calls=[])
    if receipt is None or receipt['status'] == 'unknown':
        return value
    try:
        observed = observations.build_observation(receipt, source_sha256=source, seed=seed,
            adapter_identity=group['adapter_identity'], scaffold_sha256=scaffold)
    except (ValueError, KeyError, TypeError, OverflowError):
        # A diagnostic cannot invalidate otherwise authenticated final geometry.
        # Preserve the unavailable diagnostic, without deriving partial numbers.
        value['reason'] = 'diagnostic_invalid_or_unsupported'
        return value
    value.update(trace_status=observed['status'], reason=observed['reason'],
                 observation_sha256=observed['observation_sha256'])
    if observed['status'] != 'available':
        return value
    # The observation builder has authenticated this trace and its quality join.
    trace = receipt['restart_trace']['trace']
    value.update(trace_sha256=trace['trace_sha256'], prefix=copy.deepcopy(trace['prefix']),
                 run=copy.deepcopy(observed['run']))
    cache = {}
    for call in trace['calls']:
        best, best_error = motion._decode(trace, call['inputs']['best'], 'best', cache)
        result, result_error = motion._decode(trace, call.get('optimizer_result'), 'result', cache)
        distance = None if best is None or result is None else _hausdorff(result, best)
        distance_reason = best_error or result_error or ('nonfinite_distance' if distance is None else None)
        geometry = (None if result is None else validate_packing(dict(
            centers=[list(result[2*i:2*i+2]) for i in range(26)], radii=list(result[52:]))))
        start, terminal = [call['phases'][key] for key in ('optimizer_entered', 'terminal')]
        gain = None if start is None or terminal is None else terminal['sum'] - start['sum']
        if gain is not None and not math.isfinite(gain):
            gain = None
        input_refs = {key: call['inputs'][key]['ref'] for key in INPUTS}
        unchanged = {key: None if input_refs[key] is None or call['post_inputs'][key]['ref'] is None
                     else input_refs[key] == call['post_inputs'][key]['ref'] for key in INPUTS}
        outcome = call.get('optimizer_outcome')
        value['calls'].append(dict(ordinal=call['ordinal'], restart_index=call['restart_index'],
            helper_finished=call['helper_finished'], effective_origin=call['effective']['origin'],
            terminal=call['terminal'], input_refs=input_refs, input_unchanged=unchanged,
            optimizer_result_center_hausdorff_from_best=distance, distance_reason=distance_reason,
            optimizer_success=None if outcome is None else outcome['success']['value'],
            optimizer_status=None if outcome is None else outcome['status']['value'],
            result_tolerance_valid=None if geometry is None else geometry['valid'],
            result_strict_valid=None if geometry is None else geometry['strict_valid'],
            incumbent_gain=gain,
            complete_accepted_return=bool(call['helper_finished']
                and call['effective']['origin'] == 'accepted_return'
                and call['terminal'] == 'completed' and start is not None and terminal is not None)))
    return value


def build_facts(group):
    """Recompute source/seed-bound diagnostics, with no author or execution permit."""
    group = validate_comparison(group)
    scaffold = builder.identity()['common_scaffold_sha256']
    draws, matched = [], []
    for pair in group['pairs']:
        current = {role: _draw(pair[role], role=role, seed=pair['seed'], group=group, scaffold=scaffold)
                   for role in ('comparator', 'challenger')}
        draws.extend(current.values())
        left = {(row['ordinal'], row['restart_index']): row for row in current['comparator']['calls']}
        for candidate in current['challenger']['calls']:
            if candidate['effective_origin'] != 'accepted_return':
                continue
            control = left.get((candidate['ordinal'], candidate['restart_index']))
            refs = candidate['input_refs']
            available = (control is not None and all(refs[k] is not None
                         and control['input_refs'][k] is not None for k in INPUTS))
            equal = (all(refs[k] == control['input_refs'][k] for k in INPUTS) if available else None)
            # Only true matches expose a local contrast; unmatched rows retain no
            # numerical contrast which could be mistaken for matched evidence.
            a = control['incumbent_gain'] if equal is True else None
            b = candidate['incumbent_gain'] if equal is True else None
            matched.append(dict(seed=pair['seed'], ordinal=candidate['ordinal'],
                restart_index=candidate['restart_index'], inputs_byte_identical=equal,
                reason=None if available else 'missing_control_call_or_input_snapshot',
                comparator_gain=a, challenger_gain=b,
                local_gain_delta=None if a is None or b is None else b-a,
                comparator_result_distance=(control['optimizer_result_center_hausdorff_from_best']
                                            if equal is True else None),
                challenger_result_distance=(candidate['optimizer_result_center_hausdorff_from_best']
                                            if equal is True else None),
                inputs_sha256=canonical_sha(refs) if equal is True else None))
    value = dict(schema=SCHEMA, comparison_id=group['id'], comparison_sha256=canonical_sha(group),
        scope=SCOPE, forecasts_declared=False, draws=draws, matched_inputs=matched)
    value['facts_sha256'] = canonical_sha(value)
    return value


def compact_facts(facts):
    """All accepted-return rows and matched contrasts; inactive rows are counted.

    Input refs and per-call proofs stay in the full receipt-bound reducer, whose
    digest accompanies this compact view. No rows are selected for their gain.
    """
    columns = ['ordinal', 'restart_index', 'terminal', 'complete_accepted_return',
        'optimizer_success', 'optimizer_status', 'result_tolerance_valid',
        'result_strict_valid', 'optimizer_result_center_hausdorff_from_best',
        'distance_reason', 'incumbent_gain']
    matched_columns = ['seed', 'ordinal', 'restart_index', 'inputs_byte_identical',
        'reason', 'comparator_gain', 'challenger_gain', 'local_gain_delta',
        'comparator_result_distance', 'challenger_result_distance']
    draws = []
    for draw in facts['draws']:
        selected = [c for c in draw['calls'] if c['effective_origin'] == 'accepted_return']
        run = draw['run']
        run_view = None if run is None else dict(status=run['status'],
            total_starts=run['total_starts'], total_scheduled_calls=run['total_scheduled_calls'],
            accepted_return=copy.deepcopy(run['origins']['accepted_return']))
        draws.append(dict(role=draw['role'], seed=draw['seed'], receipt_status=draw['receipt_status'],
            trace_status=draw['trace_status'], reason=draw['reason'],
            trace_sha256=draw['trace_sha256'], prefix=copy.deepcopy(draw['prefix']),
            run=run_view,
            retained_nonaccepted_calls_omitted=len(draw['calls'])-len(selected),
            accepted_return_calls=[[row[k] for k in columns] for row in selected]))
    return dict(schema=SCHEMA, facts_sha256=facts['facts_sha256'], scope=SCOPE,
        forecasts_declared=False, call_columns=columns, matched_columns=matched_columns,
        draws=draws, matched_inputs=[[row[k] for k in matched_columns] for row in facts['matched_inputs']])
