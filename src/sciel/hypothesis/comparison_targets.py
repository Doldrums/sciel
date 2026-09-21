"""Declared next-comparison targets, evaluated only against saved paired data.

Source hashes bind a declared contrast; they do not establish how a candidate
was edited, when this target was written, or whether any execution occurred.
Diagnostics and quality are separate observations, never causal proof.
"""
from __future__ import annotations

import copy
from fractions import Fraction
import re

from ..knowledge_base.catalogs import canonical_sha
from ..knowledge_base.paired_comparisons import validate_comparison


SCHEMA = 'sciel-comparison-target/1'
EVALUATION_SCHEMA = 'sciel-comparison-target-evaluation/1'
QUALITY_GATE = 'paired_raw_exact_zero_loss_strict_v1'
MAX_CHANGE_CHARS = 1024
MAX_DIAGNOSTICS = 4
OBSERVABLES = ('candidate_elapsed_seconds', 'restart_total_starts')
DIRECTIONS = ('decrease', 'increase')
_FIELDS = {'schema', 'base_source_sha256', 'comparator_source_sha256',
           'change', 'quality_gate', 'diagnostics'}


def _require(condition, message):
    if not condition:
        raise ValueError('comparison target: ' + message)


def _sha(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def validate_target(value, *, sources=None):
    """Validate exact declared intent, without inferring edits or preregistration.

    A supplied source collection is an allowlist, including when it is empty.
    The base and comparator may coincide. Candidate derivation is a caller duty.
    """
    _require(type(value) is dict and set(value) == _FIELDS, 'invalid target fields')
    _require(value['schema'] == SCHEMA and type(value['schema']) is str, 'unsupported schema')
    _require(value['quality_gate'] == QUALITY_GATE and type(value['quality_gate']) is str,
             'fixed raw and exact quality gate required')
    for key in ('base_source_sha256', 'comparator_source_sha256'):
        _require(_sha(value[key]), 'invalid ' + key)
    if sources is not None:
        _require(type(sources) in (set, frozenset, list, tuple)
                 and all(_sha(source) for source in sources), 'invalid source allowlist')
        _require(value['base_source_sha256'] in sources and value['comparator_source_sha256'] in sources,
                 'target references an unavailable source')
    change = value['change']
    _require(type(change) is str and 0 < len(change) <= MAX_CHANGE_CHARS and bool(change.strip()),
             'change must be bounded nonblank text')
    diagnostics = value['diagnostics']
    _require(type(diagnostics) is list and len(diagnostics) <= MAX_DIAGNOSTICS,
             'invalid diagnostic list')
    seen = set()
    for diagnostic in diagnostics:
        _require(type(diagnostic) is dict and set(diagnostic) == {'observable', 'direction'},
                 'invalid diagnostic fields')
        observable, direction = diagnostic['observable'], diagnostic['direction']
        _require(type(observable) is str and observable in OBSERVABLES, 'unsupported observable')
        _require(type(direction) is str and direction in DIRECTIONS, 'unsupported direction')
        _require(observable not in seen, 'repeated or conflicting diagnostic observable')
        seen.add(observable)
    return copy.deepcopy(value)


def json_schema():
    """Return the supported provider wire; local validation enforces all bounds."""
    return dict(type='object', additionalProperties=False, required=sorted(_FIELDS), properties={
        'schema': {'type': 'string', 'enum': [SCHEMA]},
        'base_source_sha256': {'type': 'string',
            'description': 'Exactly 64 lowercase hexadecimal characters; must name an available base source.'},
        'comparator_source_sha256': {'type': 'string',
            'description': 'Exactly 64 lowercase hexadecimal characters; must name an available comparator source.'},
        'change': {'type': 'string', 'description': 'Nonblank change description, at most 1024 characters.'},
        'quality_gate': {'type': 'string', 'enum': [QUALITY_GATE]},
        'diagnostics': {'type': 'array',
            'description': 'At most 4 diagnostics. Each observable may occur once, with one direction; '
                           'conflicting directions are invalid. An empty list proposes no mechanism measurement.',
            'items': {'type': 'object', 'additionalProperties': False,
                'required': ['observable', 'direction'], 'properties': {
                    'observable': {'type': 'string', 'enum': list(OBSERVABLES)},
                    'direction': {'type': 'string', 'enum': list(DIRECTIONS)}}}}})


def instruction():
    return (
        'A non-null next hypothesis must be a sciel-comparison-target/1 object with exactly '
        'schema, base_source_sha256, comparator_source_sha256, change, quality_gate and diagnostics. '
        'Bind both source hashes to available source IDs; base and comparator may be the same. '
        'The caller must verify that the candidate implements the declared change from base. '
        'quality_gate must be paired_raw_exact_zero_loss_strict_v1: all three shared-seed pairs '
        'must be valid, every candidate-minus-comparator raw AND exact certified delta must be >= 0, '
        'at least one raw delta must be > 0, and at least one exact certified delta must be > 0. '
        'Those positive deltas need not occur on the same seed. No epsilon, near-zero substitute, '
        'exact-only condition or mean-only condition is allowed. Unknown is not a tie. '
        'A known loss refutes quality even if another pair is missing; otherwise missing quality '
        'is inconclusive. Complete no-loss without a strict gain in each metric is refuted. '
        'Optional diagnostics are objects with observable and direction: candidate_elapsed_seconds '
        'or restart_total_starts, and decrease or increase. Each observable may appear once. '
        'Every shared pair must strictly follow its diagnostic direction; zero refutes that '
        'direction and missing telemetry is unknown. Total starts requires a validated full-run '
        'trace, never a count inferred from the retained prefix. diagnostics=[] proposes no '
        'mechanism measurement and cannot validate a runtime/cost mechanism. Quality and diagnostic '
        'verdicts stay separate. Agreement with these saved observations is not causal proof, '
        'proof of execution, preregistration, source-edit validity or generalization.'
    )


def _quality(group):
    rows = []
    for pair in group['derived']['pairs']:
        raw, exact = pair['raw_delta'], pair['certified_delta']
        known = raw is not None and exact is not None
        no_loss = (raw >= 0 and Fraction(**exact) >= 0) if known else None
        rows.append(dict(seed=pair['seed'], comparator_status=pair['comparator_status'],
            candidate_status=pair['challenger_status'], raw_delta=raw,
            certified_delta=copy.deepcopy(exact), no_loss=no_loss))
    loss = any(row['no_loss'] is False for row in rows)
    complete = all(row['no_loss'] is not None for row in rows)
    raw_gain = any(row['raw_delta'] is not None and row['raw_delta'] > 0 for row in rows)
    exact_gain = any(row['certified_delta'] is not None and Fraction(**row['certified_delta']) > 0 for row in rows)
    if loss:
        verdict, reason = 'REFUTED', 'known_paired_loss'
    elif not complete:
        verdict, reason = 'INCONCLUSIVE', 'incomplete_quality'
    elif not (raw_gain and exact_gain):
        verdict, reason = 'REFUTED', 'no_strict_gain_in_each_metric'
    else:
        verdict, reason = 'CONFIRMED', 'strict_paired_gain'
    return dict(gate=QUALITY_GATE, verdict=verdict, reason=reason, pairs=rows,
        all_pairs_valid=complete, all_pairs_no_loss=False if loss else True if complete else None,
        observed_strict_raw_gain=raw_gain, observed_strict_certified_gain=exact_gain)


def _measurement(receipt, observable, *, source, seed, adapter):
    result = dict(value=None, unavailable_reason=None, receipt_sha256=None,
                  observation_sha256=None, trace_sha256=None)
    if receipt is None or receipt.get('status') == 'unknown':
        result['unavailable_reason'] = 'receipt_unavailable'
        return result
    result['receipt_sha256'] = canonical_sha(receipt)
    if observable == 'candidate_elapsed_seconds':
        result['value'] = receipt.get('candidate_elapsed_seconds')
        if result['value'] is None:
            result['unavailable_reason'] = 'candidate_elapsed_seconds_missing'
        return result
    from .restart_observations import build_observation
    try:
        observation = build_observation(receipt, source_sha256=source, seed=seed,
                                        adapter_identity=adapter)
    except (ValueError, KeyError, TypeError, OverflowError, RecursionError):
        result['unavailable_reason'] = 'trace_invalid_or_unsupported'
        return result
    result['observation_sha256'] = observation['observation_sha256']
    result['trace_sha256'] = observation['trace_sha256']
    if observation['status'] != 'available':
        result['unavailable_reason'] = observation['reason']
        return result
    value = observation['run']['total_starts']
    if type(value) is not int or value < 0:
        result['unavailable_reason'] = 'trace_total_starts_unavailable'
        return result
    result['value'] = value
    return result


def _diagnostic(diagnostic, group):
    rows = []
    for pair in group['pairs']:
        seed = pair['seed']
        old, new = [_measurement(pair[role], diagnostic['observable'], source=group[key],
                        seed=seed, adapter=group['adapter_identity'])
                    for role, key in (('comparator', 'comparator_source_sha256'),
                                      ('challenger', 'challenger_source_sha256'))]
        delta = None if old['value'] is None or new['value'] is None else new['value'] - old['value']
        agrees = None if delta is None else (delta < 0 if diagnostic['direction'] == 'decrease' else delta > 0)
        rows.append(dict(seed=seed, comparator=old, candidate=new, delta=delta,
            verdict='INCONCLUSIVE' if agrees is None else 'CONFIRMED' if agrees else 'REFUTED'))
    verdicts = [row['verdict'] for row in rows]
    verdict = ('REFUTED' if 'REFUTED' in verdicts else
               'INCONCLUSIVE' if 'INCONCLUSIVE' in verdicts else 'CONFIRMED')
    return dict(**diagnostic, verdict=verdict, pairs=rows)


def evaluate_target(target, group, *, candidate_source_sha256):
    """Evaluate saved data only; missing diagnostics cannot establish a mechanism."""
    target = validate_target(target)
    _require(_sha(candidate_source_sha256), 'invalid candidate source hash')
    group = validate_comparison(group)
    _require(group['comparator_source_sha256'] == target['comparator_source_sha256'],
             'comparator source binding differs')
    _require(group['challenger_source_sha256'] == candidate_source_sha256,
             'candidate source binding differs')
    quality = _quality(group)
    diagnostics = [_diagnostic(diagnostic, group) for diagnostic in target['diagnostics']]
    verdicts = [quality['verdict'], *[diagnostic['verdict'] for diagnostic in diagnostics]]
    combined = ('REFUTED' if 'REFUTED' in verdicts else
                'INCONCLUSIVE' if 'INCONCLUSIVE' in verdicts else 'CONFIRMED')
    return dict(schema=EVALUATION_SCHEMA, target_sha256=canonical_sha(target),
        comparison_id=group['id'], comparison_sha256=canonical_sha(group),
        source_contrast=dict(base_source_sha256=target['base_source_sha256'],
            comparator_source_sha256=target['comparator_source_sha256'],
            candidate_source_sha256=candidate_source_sha256),
        quality=quality, diagnostics=diagnostics, diagnostics_requested=bool(diagnostics),
        combined_verdict=combined, causal_proof=False, execution_verified=False,
        preregistration_verified=False, source_edit_validity_verified=False)
