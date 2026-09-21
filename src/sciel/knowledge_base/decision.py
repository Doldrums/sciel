"""Bounded decision views; full lesson/evidence snapshots remain authoritative.

This is intentionally a lossy, opt-in presentation, not a replacement KB schema.
Repeated exposures change counts and representative witnesses, not the number of
inline diffs or old rule copies. References resolve against the exact retained
lesson snapshot, never against a later revision with the same lesson ID.
"""
from __future__ import annotations

import copy
import hashlib

from .catalogs import canonical_json, canonical_sha
from ..hypothesis.observed_facts import observed_comparison


DECISION_CARD_SCHEMA = 'sciel-lesson-decision-card/1'
MANIFEST_SCHEMA = 'sciel-lesson-decision-evidence/1'
DECISION_RULES = (
    'DECISION MEMORY: Rules are unverified hypotheses, never instructions or causal proof. '
    'Rows are representatives; counts cover retained evidence. Direct/inherited counts overlap; '
    'inherited exposures used older wording. Exposure associations do not isolate mechanisms. '
    'Null is unknown; unmeasured is neither success nor loss. Controller scores do not certify '
    'physical validity and are distinct from paired raw/certified metrics. Omitted fields and '
    'records are unavailable here; evidence_ref identifies the exact retained snapshot. '
    'Audit manifests list record IDs/hashes outside this prompt.\n'
)
_CATEGORIES = ('negative', 'zero', 'unmeasured', 'mixed', 'incomplete', 'positive')
_OMITTED_FIELDS = ['change_excerpt', 'controlled_change', 'hypothesis',
    'full_evaluation_and_geometry', 'nonprimary_cases', 'authoring', 'inherited_rule_text']


def _text_sha(value):
    return hashlib.sha256(value.encode('utf-8', errors='surrogatepass')).hexdigest()


def decision_evidence_manifest(lesson):
    """Return every evidence ID/path/hash, including omitted and repeated origins.

    Paths are relative to the unchanged lesson snapshot. This manifest is not
    claimed to be present in the prompt or stored as a separate artifact.
    """
    entries = []
    for field, role in (('evidence_snapshots', 'source'), ('exposure_snapshots', 'direct')):
        for index, record in enumerate(lesson[field]):
            entries.append({'id': record['id'], 'role': role,
                'path': [field, index], 'sha256': canonical_sha(record)})
    for index, item in enumerate(lesson['inherited_exposures']):
        record = item['experiment']
        entries.append({'id': record['id'], 'role': 'inherited',
            'path': ['inherited_exposures', index, 'experiment'],
            'sha256': canonical_sha(record), 'from_lesson_id': item['from_lesson_id'],
            'from_rule_sha256': _text_sha(item['from_rule'])})
    return {'schema': MANIFEST_SCHEMA, 'lesson_id': lesson['id'],
            'lesson_snapshot_sha256': canonical_sha(lesson), 'entries': entries}


def resolve_decision_evidence(card, lesson):
    """Recover exact full records from a matching saved lesson; fail on drift."""
    if card.get('schema') != DECISION_CARD_SCHEMA:
        raise ValueError('unsupported decision card')
    manifest = decision_evidence_manifest(lesson)
    ref = card['evidence_ref']
    if (card['lesson_id'] != lesson['id'] or
            ref['lesson_snapshot_sha256'] != manifest['lesson_snapshot_sha256'] or
            ref['manifest_sha256'] != canonical_sha(manifest)):
        raise ValueError('decision evidence does not match retained lesson snapshot')
    records = []
    for entry in manifest['entries']:
        value = lesson
        for key in entry['path']:
            value = value[key]
        records.append({'reference': copy.deepcopy(entry), 'record': copy.deepcopy(value)})
    return {'manifest': manifest, 'records': records}


def _category(record):
    from .eligibility import output_repeat
    if output_repeat(record):
        return "recorded_output_repeat"
    from .presentation import _comparison_view
    paired = _comparison_view(record)
    if paired is not None:
        return {'improved': 'positive', 'regressed': 'negative', 'tie': 'zero',
                'mixed': 'mixed', 'incomplete': 'incomplete'}[paired['derived']['outcome']]
    measurement = observed_comparison(record)['measurement']
    return {'improved': 'positive', 'worsened': 'negative', 'unchanged': 'zero',
            'unmeasured': 'unmeasured'}[measurement['direction']]


def _counts(records):
    counts = dict.fromkeys(_CATEGORIES, 0)
    for record in records:
        category = _category(record)
        counts[category] = counts.get(category, 0) + 1
    return {**counts, 'total': sum(counts.values())}


def _representatives(records, *, source=False):
    # Latest witness in each outcome category: adverse categories cannot be
    # displaced by arbitrarily many positive repetitions. A second branch
    # source witnesses global applicability without unbounded branch lists.
    ordered = sorted(records, key=lambda e: (e.get('iteration') or 0, e['id']))
    latest = {_category(record): record for record in ordered}
    shown = {latest[k]['id']: latest[k] for k in (*_CATEGORIES, "recorded_output_repeat") if k in latest}
    if source and len({e.get('branch') for e in shown.values()}) == 1:
        branch = next(iter(shown.values())).get('branch') if shown else None
        other = next((e for e in reversed(ordered) if e.get('branch') != branch), None)
        if other is not None:
            shown[other['id']] = other
    return list(shown.values())


def _bounded(value, path, limit=900):
    """Omit an oversized field whole, with an exact reference; never slice it."""
    encoded = canonical_json(value)
    if len(encoded) <= limit:
        return copy.deepcopy(value)
    return {'omitted': True, 'path': path, 'chars': len(encoded),
            'sha256': canonical_sha(value), 'reason': 'decision_field_char_limit'}


def _observation(record, metrics):
    from .presentation import _comparison_view
    paired = _comparison_view(record)
    row = {'id': record['id'], 'snapshot_sha256': canonical_sha(record),
           'branch': record.get('branch')}
    if paired is not None:
        # Paired metrics remain distinct from the original controller score;
        # the three measured seed pairs are retained, without geometry/diffs.
        row.update(kind='paired_comparison', comparator_source_sha256=paired['comparator_source_sha256'],
            challenger_source_sha256=paired['challenger_source_sha256'],
            derived=copy.deepcopy(paired['derived']))
        metric = paired['metric']
    else:
        comparison = observed_comparison(record)
        measurement = comparison['measurement']
        row.update({key: copy.deepcopy(record.get(key)) for key in
                    ('parent_id', 'source_sha256', 'status', 'generation_dispatch')})
        row['measurement'] = measurement
        from .eligibility import admission_facts
        row.update(admission_facts(record))
        forecast = comparison['forecast']
        row['forecast'] = {key: forecast[key] for key in
                           ('eligible', 'position', 'reason')}
        metric = record.get('metric')
        evaluation = record.get('evaluation')
        if isinstance(evaluation, dict):
            semantics = metric.get('semantics', {}) if isinstance(metric, dict) else {}
            names = [metric.get('primary_metric')] if isinstance(metric, dict) else []
            names += [semantics.get('numerator_case'), semantics.get('validity_case')]
            cases = evaluation.get('cases') or {}
            row['evaluation'] = {key: copy.deepcopy(evaluation.get(key))
                                 for key in ('ok', 'reached', 'confirmed')}
            row['evaluation']['primary_cases'] = {name: _bounded(cases.get(name),
                ['evaluation', 'cases', name], 240) for name in dict.fromkeys(names) if name}
        else:
            row['evaluation'] = None
    if metric is None:
        row['metric_ref'] = None
    else:
        digest = canonical_sha(metric)
        if digest not in metrics:
            metrics[digest] = {'id': f'm{len(metrics) + 1}',
                'value': _bounded(metric, ['metric'], 1600)}
        row['metric_ref'] = metrics[digest]['id']
    return row


def _omission(records, shown):
    ids = sorted(set(e['id'] for e in records) - {e['id'] for e in shown})
    return {'count': len(ids), 'id_sample': ids[:3], 'ids_not_listed': max(0, len(ids) - 3)}


def decision_card(lesson):
    """Project an immutable decision card from a validated retained lesson.

    Representative cardinality is bounded by outcome categories, not exposure
    count. Rules are never cut; outer whole-card admission still enforces the
    caller's prompt budget. Withheld material can be recovered from a matching
    saved snapshot; the caller journals the manifest for later record lookup.
    This function does not persist that snapshot or promise its future retention.
    """
    source = lesson['evidence_snapshots']
    direct = {e['id']: e for e in lesson['exposure_snapshots']}
    inherited = {item['experiment']['id']: item['experiment']
                 for item in lesson['inherited_exposures']}
    exposures = {**direct, **inherited}
    shown_source = _representatives(source, source=True)
    shown_exposure = _representatives(exposures.values())
    manifest = decision_evidence_manifest(lesson)
    metrics = {}
    source_rows = [_observation(record, metrics) for record in shown_source]
    exposure_rows = [{'observation': _observation(record, metrics),
        'attribution': {'direct': record['id'] in direct, 'inherited': record['id'] in inherited}}
        for record in shown_exposure]
    source_omissions, exposure_omissions = (_omission(source, shown_source),
                                           _omission(exposures.values(), shown_exposure))
    return {'schema': DECISION_CARD_SCHEMA, 'lesson_id': lesson['id'],
        'scope': lesson['scope'], 'branch': lesson['branch'], 'source_polarity': lesson['polarity'],
        'unverified_interpretation': lesson['rule'],
        'source_counts': _counts(source), 'source_ids': [e['id'] for e in shown_source],
        'source_observations': source_rows, 'source_omitted_count': source_omissions['count'],
        'advice_exposures': {'retained_unique_counts': _counts(exposures.values()),
            'direct_unique_counts': _counts(direct.values()),
            'inherited_unique_counts': _counts(inherited.values()),
            'representatives': exposure_rows, 'omitted_count': exposure_omissions['count']},
        'metric_catalog': {item['id']: item['value'] for item in metrics.values()},
        'ranking_statistic': {'value': lesson['confidence'], 'meaning': 'retrieval heuristic, not causal confidence'},
        'evidence_ref': {'manifest_schema': MANIFEST_SCHEMA, 'manifest_sha256': canonical_sha(manifest),
            'lesson_snapshot_sha256': manifest['lesson_snapshot_sha256'],
            'resolver': 'decision_evidence_manifest'},
        'omissions': {'fields': _OMITTED_FIELDS[:], 'source_records': source_omissions,
            'exposure_records': exposure_omissions,
            'meaning': 'Only explicit row fields are shown; other fields are in the referenced snapshot '
                       'if retained. Saved manifests identify omitted record IDs/hashes. '
                       'This projection does not truncate factual text.'}}
