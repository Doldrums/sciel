"""Explicit prompt projections followed by lossless measurement transport.

Full native geometry is deliberately projected to its metadata-only view;
arrays remain in durable evidence. Only schema-owned measurement slots are
then factored losslessly. Opaque task data is never walked for lookalike fields.
A smaller wire does not imply equivalent model behavior; checkpoint identity
therefore versions this presentation change.
"""
from __future__ import annotations

import json
import copy
import hashlib
from pathlib import Path

from ..knowledge_base.authoring import SCHEMA as AUTHORING_SCHEMA
from ..knowledge_base.catalogs import (SCHEMA as MEASUREMENT_SCHEMA,
    canonical_json, canonical_sha, expand_measurements, pack_measurements)
from ..knowledge_base.presentation import (CARD_SCHEMA, REFLECTION_SCHEMA, COMPARISON_CARD_SCHEMA,
    INITIALIZER_CONTEXT_SCHEMA, PAIRED_LESSON_SCHEMA,
    reflection_payload, compact_paired_lesson_card, expand_paired_lesson_card,
    DECISION_CARD_SCHEMA, decision_card)
from .observed_facts import SCHEMA as COMPARISON_SCHEMA
from .parsing import AUX_TRANSPORT
from .circle_proposal_group import CARD_SCHEMA as PROPOSAL_CARD_SCHEMA
from .measurement import evaluation_view, NATIVE_GEOMETRY_VIEW_SCHEMA
from .conclusion_feedback import (SCHEMA as CONCLUSION_FEEDBACK_SCHEMA,
    REFLECTION_REVIEW_SCHEMA, LESSON_ADMISSION_SCHEMA, BATCH_FEEDBACK_SCHEMA,
    BATCH_CARD_SCHEMA, BATCH_REFLECTION_REVIEW_SCHEMA, BATCH_REFLECTION_REVIEW_CARD_SCHEMA,
    TYPED_FEEDBACK_SCHEMA, TYPED_CARD_SCHEMA,
    TYPED_REFLECTION_REVIEW_SCHEMA, TYPED_REFLECTION_REVIEW_CARD_SCHEMA)
from .mechanism_review import (SCHEMA as MECHANISM_REVIEW_SCHEMA,
                               source_quote_identity, identity as mechanism_identity)
from ..knowledge_base.restart_context import SCHEMA as RESTART_CONTEXT_SCHEMA
from ..knowledge_base.restart_context import admission_identity as restart_admission_identity
from .restart_observations import SCHEMA as RESTART_OBSERVATION_SCHEMA
from .restart_optimizer_observations import SCHEMA as OPTIMIZER_OBSERVATION_SCHEMA, geometry_policy
from .restart_motion_observations import SCHEMA as MOTION_OBSERVATION_SCHEMA
from .restart_motion import POLICY as MOTION_POLICY
from . import restart_motion, restart_motion_observations
from ..knowledge_base.memory_references import SCHEMA as SHOWN_COMPARISON_SCHEMA
from ..knowledge_base.comparison_facts import identity as comparison_facts_identity

MAX_CONTEXT_CHARS = 131072
REFERENCE_RULES = (
    "MEASUREMENT REFERENCES: If the JSON uses sciel-measurement-catalog/1, read its "
    "payload after resolving ONLY paths listed in bindings: each catalog_ref selects "
    "the exact dictionary in catalog with that SHA256. Other lookalike fields are "
    "literal task data. These references preserve measurements, not new observations; "
    "missing/null remains unknown.\n"
)
PAIRED_REFERENCE_RULES = (
    "PAIRED REFERENCES: Only comparison_catalog paired_ref=true copies "
    "schema,id,evidence_sha256,derived,metric,forecast from its same-ID "
    "source_observations card.\n"
)


def presentation_identity(*, batch_only_lessons=False, restart_predictions=False, comparison_reviews=False,
                          compact_lesson_memory=False):
    # A new copy: callers cannot mutate checkpoint identity through this value.
    if type(batch_only_lessons) is not bool:
        raise ValueError('batch_only_lessons must be boolean')
    if type(restart_predictions) is not bool:
        raise ValueError('restart_predictions must be boolean')
    if type(comparison_reviews) is not bool:
        raise ValueError('comparison_reviews must be boolean')
    if type(compact_lesson_memory) is not bool:
        raise ValueError('compact_lesson_memory must be boolean')
    from .conclusion_source import identity as conclusion_source_identity
    identity = {'conclusion_source': conclusion_source_identity(), 'measurement': MEASUREMENT_SCHEMA, 'reflection': REFLECTION_SCHEMA,
            'observed_comparison': COMPARISON_SCHEMA,
            'comparison_facts': comparison_facts_identity(),
            'auxiliary_transport': AUX_TRANSPORT,
            'failure_context': 'sciel-failure-context/2',
            'native_geometry': NATIVE_GEOMETRY_VIEW_SCHEMA,
            'proposal_group_card': PROPOSAL_CARD_SCHEMA,
            'conclusion_feedback': CONCLUSION_FEEDBACK_SCHEMA,
            'reflection_review': REFLECTION_REVIEW_SCHEMA,
            'mechanism_review': MECHANISM_REVIEW_SCHEMA,
            'typed_mechanism': {**mechanism_identity(),
                'conclusion_feedback': TYPED_FEEDBACK_SCHEMA, 'conclusion_feedback_card': TYPED_CARD_SCHEMA,
                'reflection_review': TYPED_REFLECTION_REVIEW_SCHEMA,
                'reflection_review_card': TYPED_REFLECTION_REVIEW_CARD_SCHEMA},
            'mechanism_source_quotation': source_quote_identity(),
            'lesson_authoring': AUTHORING_SCHEMA,
            'initializer_context': INITIALIZER_CONTEXT_SCHEMA,
            'restart_context': RESTART_CONTEXT_SCHEMA,
            'restart_observation_admission': restart_admission_identity(),
            'restart_observation': RESTART_OBSERVATION_SCHEMA,
            'restart_optimizer_observation': OPTIMIZER_OBSERVATION_SCHEMA,
            'restart_motion_observation': MOTION_OBSERVATION_SCHEMA,
            'restart_motion_policy': canonical_sha(MOTION_POLICY),
            'restart_motion_sources': {
                'hypothesis/restart_motion.py': hashlib.sha256(Path(restart_motion.__file__).read_bytes()).hexdigest(),
                'hypothesis/restart_motion_observations.py': hashlib.sha256(
                    Path(restart_motion_observations.__file__).read_bytes()).hexdigest()},
            'restart_result_geometry': geometry_policy(),
            'shown_comparison_references': SHOWN_COMPARISON_SCHEMA,
            'paired_lesson_references': PAIRED_LESSON_SCHEMA,
            'observation_card': CARD_SCHEMA, 'paired_comparison': COMPARISON_CARD_SCHEMA, 'sources': 'sciel-prompt-source-catalog/1',
            'memory': 'sciel-memory-presentation/5', 'json': 'strict-canonical/1'}
    if batch_only_lessons:
        identity['conclusion_lesson_admission'] = {
            'schema': LESSON_ADMISSION_SCHEMA,
            'conclusion_feedback': BATCH_FEEDBACK_SCHEMA, 'conclusion_feedback_card': BATCH_CARD_SCHEMA,
            'reflection_review': BATCH_REFLECTION_REVIEW_SCHEMA,
            'reflection_review_card': BATCH_REFLECTION_REVIEW_CARD_SCHEMA}
    if restart_predictions:
        from . import restart_predictions as forecasts
        from ..knowledge_base import restart_prediction_context as projection
        identity['restart_predictions'] = {
            'schema': forecasts.SCHEMA, 'result_schema': forecasts.RESULT_SCHEMA,
            'context_schema': projection.SCHEMA, 'policy_sha256': canonical_sha(forecasts.POLICY),
            'source_sha256': {name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                for name, module in (('hypothesis/restart_predictions.py', forecasts),
                                     ('knowledge_base/restart_prediction_context.py', projection))}}
        identity['restart_predictions']['source_sha256']['engine/research.py'] = hashlib.sha256(
            (Path(__file__).resolve().parents[1] / 'engine/research.py').read_bytes()).hexdigest()
        identity['restart_predictions']['memory_priority'] = (
            'comparisons_lessons_motion_forecasts_proposal_history_reviews/1')
    if comparison_reviews:
        from . import comparison_review
        identity['comparison_reviews'] = comparison_review.identity()
        identity['comparison_reviews']['integration_sha256'] = {
            name: hashlib.sha256((Path(__file__).resolve().parents[1] / name).read_bytes()).hexdigest()
            for name in ('engine/research.py', 'hypothesis/prompts.py', 'hypothesis/parsing.py')}
        identity['comparison_reviews']['memory_priority'] = 'comparisons_lessons_comparison_review_imported_conclusion_motion/1'
    if compact_lesson_memory:
        from ..knowledge_base import decision
        identity['compact_lesson_memory'] = {
            'schema': DECISION_CARD_SCHEMA, 'manifest': decision.MANIFEST_SCHEMA,
            'source_sha256': hashlib.sha256(Path(decision.__file__).read_bytes()).hexdigest(),
            'guard_fact_scope': 'shown_decision_source_rows_only/1'}
    return identity


def serialize_decision_card(lesson_or_card):
    """Serialize the explicit lossy decision view, without changing legacy wire."""
    card = (lesson_or_card if lesson_or_card.get('schema') == DECISION_CARD_SCHEMA
            else decision_card(lesson_or_card))
    return canonical_json(card)


def _measurement_paths(prefix=()):
    # Controlled source-edit proofs, operators and exact diff strings stay
    # literal. Only these declared measurement dictionaries may be factored.
    return [prefix + ('metric',), prefix + ('parent_cases',),
            prefix + ('evaluation', 'cases')]


def _project_evaluations(value, paths):
    value = copy.deepcopy(value)
    # The existing schema-owned case paths also identify the only evaluation
    # objects eligible for projection. Never walk arbitrary nested task data.
    for path in paths:
        if path[-2:] != ('evaluation', 'cases'):
            continue
        node = value
        for key in path[:-2]:
            if isinstance(node, dict) and key in node:
                node = node[key]
            elif isinstance(node, list) and type(key) is int and 0 <= key < len(node):
                node = node[key]
            else:
                node = None
                break
        if isinstance(node, dict) and 'evaluation' in node:
            node['evaluation'] = evaluation_view(node['evaluation'])
    return value


def _serialize_projected(value, paths):
    inline = canonical_json(value)
    envelope = pack_measurements(value, paths)
    compact = canonical_json(envelope)
    # Account for the extra explanation as well as ALL envelope overhead. The
    # caller emits these exact rules only when this transport was selected.
    data = (compact if envelope['bindings'] and
            len(compact) + len(REFERENCE_RULES) < len(inline) else inline)
    return data


def _serialize(value, paths):
    return _serialize_projected(_project_evaluations(value, paths), paths)


def decode_evidence(value):
    """Expand the declared outer layers only, never inferred nested values."""
    if type(value) is dict and value.get('schema') == MEASUREMENT_SCHEMA:
        value = expand_measurements(value)
    if type(value) is dict and value.get('schema') == PAIRED_LESSON_SCHEMA:
        return expand_paired_lesson_card(value)
    return json.loads(canonical_json(value))


def reference_rules(serialized):
    """Instructions selected from a serializer result, not from opaque text."""
    value = json.loads(serialized)
    rules = ''
    if type(value) is dict and value.get('schema') == MEASUREMENT_SCHEMA:
        rules += REFERENCE_RULES
        value = value['payload']
    if type(value) is dict and value.get('schema') == PAIRED_LESSON_SCHEMA:
        rules += PAIRED_REFERENCE_RULES
    return rules


def serialize_record(record):
    return _serialize(record, _measurement_paths())


def serialize_context(context):
    paths = _measurement_paths()
    for index, _ in enumerate(context.get('fragments', [])):
        paths.extend(_measurement_paths(('fragments', index, 'evidence')))
    return _serialize(context, paths)


def serialize_card(card):
    paths = []
    for index, _ in enumerate(card['source_observations']):
        paths.extend(_measurement_paths(('source_observations', index)))
    for index, _ in enumerate(card['advice_exposures']['representatives']):
        paths.extend(_measurement_paths(('advice_exposures', 'representatives', index, 'observation')))
    # Project native geometry before hashing the expanded card. The hash must
    # bind the exact prompt view, not the larger durable evidence behind it.
    projected = _project_evaluations(card, paths)
    legacy = _serialize_projected(projected, paths)
    if projected.get('schema') != CARD_SCHEMA:
        return legacy
    wire = compact_paired_lesson_card(projected)
    if wire.get('schema') != PAIRED_LESSON_SCHEMA:
        return legacy
    compact = _serialize_projected(wire, paths)
    # Framing is identical for both choices. Charge every conditional rule and
    # retain the established representation on ties or a net size increase.
    return (compact if len(compact) + len(reference_rules(compact)) <
            len(legacy) + len(reference_rules(legacy)) else legacy)


def serialize_reflection(records, lessons):
    payload = reflection_payload(records, lessons)
    paths = []
    for ident in payload['observation_catalog']:
        paths.extend(_measurement_paths(('observation_catalog', ident)))
    return _serialize(payload, paths)
