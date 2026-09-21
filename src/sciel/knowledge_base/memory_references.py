"""Exact references to comparison cards in already selected, shown lessons.

This codec neither selects lessons nor changes evidence. Callers keep selection
and exposure accounting unchanged, charge RULES once, and must ensure referenced
whole lesson texts really occur in the delivered prompt. Freed space must not
silently trigger a second lesson retrieval/admission pass.
"""
from __future__ import annotations

import hashlib
import json
import re

from .catalogs import canonical_json, canonical_sha
from .presentation import (CARD_SCHEMA, COMPARISON_CARD_SCHEMA,
                           LEGACY_COMPARISON_CARD_SCHEMA, _comparison_view)

SCHEMA = 'sciel-paired-comparison-context/2'
LEGACY_SCHEMA = 'sciel-paired-comparison-context/1'
REFERENCE_SCHEMA = 'sciel-shown-comparison-ref/1'
MARKER = 'RESEARCH_LESSON_JSON:\n'
RULES = ('SHOWN COMPARISON REFERENCES: Resolve sciel-shown-comparison-ref/1 only to the '
         'same-ID full source_observations card in the already shown lesson_id. Verify '
         'the whole lesson-text and card hashes. No new evidence or support; missing '
         'targets are errors, never inferred facts.\n')
_FIELDS = {'schema', 'cards', 'presented_ids', 'omitted_ids', 'omitted_count',
           'omitted_ids_truncated', 'retained_count', 'rendered_chars'}
_REF_FIELDS = {'schema', 'id', 'lesson_id', 'lesson_text_sha256', 'view_sha256',
               'evidence_sha256', 'card_sha256'}
_CARD_SCHEMAS = (COMPARISON_CARD_SCHEMA, LEGACY_COMPARISON_CARD_SCHEMA)


def _require(ok, message):
    if not ok:
        raise ValueError('shown comparison references: ' + message)


def _sha(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def _text_sha(text):
    try:
        return hashlib.sha256(text.encode('utf-8')).hexdigest()
    except UnicodeError as exc:
        raise ValueError('shown comparison references: invalid lesson UTF8') from exc


def _clone(value):
    data = canonical_json(value)
    _require(len(data) <= 262144, 'context byte limit')
    return json.loads(data)


def _resize(value):
    value['rendered_chars'] = 0
    while value['rendered_chars'] != len(canonical_json(value)):
        value['rendered_chars'] = len(canonical_json(value))
    return value


def _context(value, schema):
    value = _clone(value)
    _require(type(value) is dict and set(value) == _FIELDS and value['schema'] == schema, 'context schema')
    _require(type(value['cards']) is list and len(value['cards']) <= 200 and
             all(type(card) is dict and type(card.get('id')) is str and card['id'] for card in value['cards']), 'context cards')
    ids = [card['id'] for card in value['cards']]
    _require(type(value['presented_ids']) is list and value['presented_ids'] == ids and
             len(set(ids)) == len(ids), 'context presented IDs')
    _require(type(value['rendered_chars']) is int and value['rendered_chars'] == len(canonical_json(value)), 'context rendered length')
    return value


def _shown(lesson_texts_by_id):
    """Read the single owned suffix and declared sources, never nested data."""
    _require(type(lesson_texts_by_id) is dict and len(lesson_texts_by_id) <= 200, 'shown lessons mapping')
    _require(all(type(k) is str and k and type(v) is str for k, v in lesson_texts_by_id.items()), 'shown lesson types')
    _require(sum(len(text) for text in lesson_texts_by_id.values()) <= 262144, 'shown lesson text limit')
    from ..hypothesis.rendering import decode_evidence  # avoid rendering import cycle
    result = {}
    for ident, text in lesson_texts_by_id.items():
        if MARKER not in text:
            continue
        prefix, _, payload = text.partition(MARKER)
        _require(text.count(MARKER) == 1 and (not prefix or prefix.endswith('\n')), 'lesson marker framing')
        try:
            wire = json.loads(payload)
        except (ValueError, RecursionError) as exc:
            raise ValueError('shown comparison references: lesson JSON') from exc
        _require(canonical_json(wire) == payload, 'lesson suffix is not canonical JSON')
        card = decode_evidence(wire)
        _require(type(card) is dict and card.get('schema') == CARD_SCHEMA and card.get('lesson_id') == ident, 'shown lesson identity')
        sources = card.get('source_observations')
        _require(type(sources) is list and all(type(row) is dict and type(row.get('id')) is str for row in sources), 'lesson sources')
        ids = [row['id'] for row in sources]
        _require(card.get('source_ids') == ids and len(set(ids)) == len(ids), 'lesson source membership')
        result[ident] = (_text_sha(text), {row['id']: _comparison_view(row)
            for row in sources if row.get('schema') in _CARD_SCHEMAS})
    return result


def factor_comparisons(context, lesson_texts_by_id):
    """Return detached context and net chars saved AFTER the one RULES charge.

    Mapping order is the caller's already shown lesson order; the first exact
    target wins. A source-context omission or any other dictionary difference
    prevents factoring, even if IDs, measurements or evidence hashes agree.
    """
    if context is None:
        return None, 0
    original = _context(context, LEGACY_SCHEMA)
    shown = _shown(lesson_texts_by_id)
    result = _clone(original)
    replaced = False
    for index, card in enumerate(original['cards']):
        _require(card.get('schema') in _CARD_SCHEMAS, 'original comparison schema')
        _comparison_view(card)
        for lesson_id, (text_sha, sources) in shown.items():
            target = sources.get(card['id'])
            if target is None or canonical_json(target) != canonical_json(card):
                continue
            reference = dict(schema=REFERENCE_SCHEMA, id=card['id'], lesson_id=lesson_id,
                lesson_text_sha256=text_sha, view_sha256=card['view_sha256'],
                evidence_sha256=card['evidence_sha256'], card_sha256=canonical_sha(card))
            if len(canonical_json(reference)) < len(canonical_json(card)):
                result['cards'][index] = reference
                replaced = True
            break
    if not replaced:
        return original, 0
    result['schema'] = SCHEMA
    _resize(result)
    saved = original['rendered_chars'] - result['rendered_chars'] - len(RULES)
    return (result, saved) if saved > 0 else (original, 0)


def expand_comparisons(context, lesson_texts_by_id):
    """Verify exact shown targets and restore /1; never guess missing cards."""
    if context is None:
        return None
    if type(context) is dict and context.get('schema') == LEGACY_SCHEMA:
        result = _context(context, LEGACY_SCHEMA)
        for card in result['cards']:
            _require(card.get('schema') in _CARD_SCHEMAS, 'original comparison schema')
            _comparison_view(card)
        return result
    result = _context(context, SCHEMA)
    shown = _shown(lesson_texts_by_id)
    found = False
    for index, reference in enumerate(result['cards']):
        if reference.get('schema') != REFERENCE_SCHEMA:
            _require(reference.get('schema') in _CARD_SCHEMAS, 'inline comparison schema')
            _comparison_view(reference)
            continue
        found = True
        _require(set(reference) == _REF_FIELDS and type(reference['lesson_id']) is str and
            all(_sha(reference[k]) for k in ('lesson_text_sha256','view_sha256','evidence_sha256','card_sha256')), 'reference fields')
        _require(reference['lesson_id'] in shown, 'referenced lesson is not shown')
        text_sha, sources = shown[reference['lesson_id']]
        _require(text_sha == reference['lesson_text_sha256'], 'shown lesson text digest')
        card = sources.get(reference['id'])
        _require(card is not None and card['view_sha256'] == reference['view_sha256'] and
            card['evidence_sha256'] == reference['evidence_sha256'] and
            canonical_sha(card) == reference['card_sha256'], 'shown comparison target digest')
        result['cards'][index] = card
    _require(found, 'factored context has no references')
    result['schema'] = LEGACY_SCHEMA
    return _resize(result)
