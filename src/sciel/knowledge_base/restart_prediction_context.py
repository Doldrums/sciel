"""Deterministic, source-bound forecast feedback; never independent evidence.

Results are derived from immutable comparisons and their original ordinary
records. Nothing is added to the KB, scored as a lesson, or executed here.
"""
from __future__ import annotations

import copy

from .catalogs import canonical_json, canonical_sha
from .paired_comparisons import UNKNOWN_RECEIPT, validate_comparison
from ..hypothesis import restart_motion_observations as observations
from ..hypothesis import restart_predictions as predictions

SCHEMA = 'sciel-restart-prediction-context/1'
PREFIX = 'RESTART_PREDICTIONS_JSON:\n'
MAX_CHARS = 6000
RULES = (
    'RESTART FORECAST FEEDBACK: columns label rows; expected/observed/verdicts use '
    'the fixed observable order. Each row concerns ONLY the challenger draw and '
    'accepted cases within its retained scheduled-call prefix, not omitted calls or the comparator. These '
    'predicates are existential over at most the first 16 scheduled calls: '
    'effective_locations_changed means the exact numeric center-location multiset differs best to effective; '
    'result_changed requires tolerance-valid result geometry, a unique nearest-center bijection to best, '
    'and matched center OR radius residual >1e-6; accepted_interval_gain means positive '
    'terminal-minus-optimizer-entered incumbent sum using exact represented binary64 values. '
    'A true witness wins other unknowns; otherwise missing/ambiguous relevant facts or no accepted '
    'cases mean unknown, and all known false means false. These '
    'pre-evaluation model predictions are separate from the ordinary score forecast '
    'and paired quality. INCONCLUSIVE/null is unknown. CONFIRMED does not establish '
    'improvement, a causal mechanism, replication, or lesson support. Source/spec/'
    'receipt hashes bind the original claim and captured facts; never infer '
    'missing forecasts. result_sha256 binds the full per-case calculation.\n'
)
NAMES = ('effective_locations_changed', 'result_changed', 'accepted_interval_gain')
COLUMNS = ['seed', 'receipt_sha256', 'result_sha256', 'observed', 'verdicts',
           'coverage', 'unavailable_reason']


def _seal(value):
    value = copy.deepcopy(value)
    value['context_sha256'] = canonical_sha(value)
    return value


def _unavailable(group, reason):
    return dict(group_id=group['id'], group_sha256=canonical_sha(group),
                origin_id=group['origin_experiment_id'], unavailable_reason=reason)


def _card(group, records):
    record = records.get(group['origin_experiment_id'])
    if record is None:
        return _unavailable(group, 'original_hypothesis_unavailable')
    spec = (record.get('hypothesis') or {}).get('restart_predictions')
    if spec is None:
        return _unavailable(group, 'predictions_not_recorded')
    spec = predictions.validate_spec(spec)
    binding = record.get('restart_prediction_binding')
    if binding is None:
        return _unavailable(group, 'pre_evaluation_binding_unavailable')
    source = group['challenger_source_sha256']
    if (record.get('status') != 'evaluated' or record.get('source_repaired') is not False
            or record.get('branch') != group['branch']
            or record.get('source_sha256') != source or record.get('proposed_sha256') != source
            or binding.get('spec_sha256') != canonical_sha(spec)
            or binding.get('proposed_sha256') != source or binding.get('source_sha256') != source):
        return _unavailable(group, 'original_source_binding_mismatch')
    card = dict(group_id=group['id'], group_sha256=canonical_sha(group),
        origin_id=record['id'], origin_sha256=canonical_sha(record),
        binding=copy.deepcopy(binding), expected=[spec['expected'][name] for name in NAMES], rows=[])
    for pair in group['pairs']:
        receipt = pair['challenger']
        reason = ('receipt_not_dispatched' if receipt is None else
                  'dispatch_unknown' if receipt == UNKNOWN_RECEIPT else None)
        result = None
        if reason is None:
            try:
                observation = observations.build_observation(receipt, source_sha256=source,
                    seed=pair['seed'], adapter_identity=group['adapter_identity'])
            except ValueError:
                reason = 'diagnostic_unavailable'
            else:
                result = predictions.evaluate(spec, observation)
                if result['status'] != 'available':
                    reason = result['reason'] or 'diagnostic_unavailable'
        card['rows'].append([pair['seed'], canonical_sha(receipt) if type(receipt) is dict else None,
            result['result_sha256'] if result else None,
            [result['observed'][name] for name in NAMES] if result else [None] * 3,
            [result['verdicts'][name] for name in NAMES] if result else ['INCONCLUSIVE'] * 3,
            result['coverage'] if result else None, reason])
    return card


def build_context(groups, records, *, max_chars=MAX_CHARS):
    """Preserve whole groups in input order, with explicit budget omissions."""
    if type(max_chars) is not int or max_chars < 0:
        raise ValueError('invalid restart prediction context limit')
    if not groups:
        return None
    value = dict(schema=SCHEMA, policy_sha256=canonical_sha(predictions.POLICY),
        observables=list(NAMES), columns=list(COLUMNS), max_chars=max_chars,
        cards=[], omitted=[g['id'] for g in groups])
    if len(canonical_json(_seal(value))) > max_chars:
        return None
    for group in groups:
        group = validate_comparison(group)
        candidate = copy.deepcopy(value)
        candidate['cards'].append(_card(group, records))
        candidate['omitted'].remove(group['id'])
        if len(canonical_json(_seal(candidate))) <= max_chars:
            value = candidate
    return _seal(value)


def validate_context(value, *, groups, records):
    """Re-derive all claims from the caller's authoritative KB, not wire hashes."""
    if type(value) is not dict or type(value.get('max_chars')) is not int:
        raise ValueError('invalid restart prediction context')
    expected = build_context(groups, records, max_chars=value['max_chars'])
    if canonical_json(value) != canonical_json(expected):
        raise ValueError('restart prediction context differs from bound evidence')
    return copy.deepcopy(value)


def block(context):
    # Public callers that receive a wire object must validate_context against
    # actual records first. Research builds or rederives this projection.
    if context is None:
        return ''
    if (type(context) is not dict or context.get('schema') != SCHEMA
            or context.get('context_sha256') != canonical_sha(
                {k: v for k, v in context.items() if k != 'context_sha256'})):
        raise ValueError('invalid restart prediction projection digest')
    return RULES + PREFIX + canonical_json(context) + '\n'
