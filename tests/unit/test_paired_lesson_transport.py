"""Pure lossless card codec checks; no model, helper, evaluator or solver runs."""
import copy
import json
import math

import pytest

from sciel.knowledge_base import KnowledgeBase
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
from sciel.knowledge_base.presentation import (
    CARD_SCHEMA, COMPARISON_CARD_SCHEMA, LEGACY_COMPARISON_CARD_SCHEMA,
    PAIRED_LESSON_SCHEMA, compact_paired_lesson_card, expand_paired_lesson_card,
    observation_card,
)

from test_knowledge_paired_comparisons import group
from test_paired_source_context import build as source_group
from test_research_knowledge import experiment, proposal


@pytest.fixture(scope='module')
def cards():
    """Real KB views from fixed saved-number receipts and inert source text."""
    result = {}
    for kind in ('ordinary', 'legacy', 'source', 'mixed'):
        kb = KnowledgeBase()
        kb.record(experiment(1, branch='p1'))
        ids = ['r000001a1'] if kind in ('ordinary', 'mixed') else []
        if kind != 'ordinary':
            paired = group() if kind == 'legacy' else source_group()
            kb.record_comparison(paired)
            ids.append(paired['id'])
        kb.add_lessons([proposal(*ids)], allowed_evidence_ids=ids,
            writer_context={'phase': 'reflection', 'reference': 1, 'raw_response_sha256': 'a'*64})
        result[kind] = observation_card(kb.lessons[0])
    return result


def paired_id(card):
    return next(row['id'] for row in card['source_observations']
                if row.get('schema') in (COMPARISON_CARD_SCHEMA, LEGACY_COMPARISON_CARD_SCHEMA))


@pytest.mark.parametrize('kind', ['legacy', 'source', 'mixed'])
def test_exact_roundtrip_preserves_full_sources_authoring_and_facts(cards, kind):
    card = copy.deepcopy(cards[kind]); before = canonical_json(card)
    wire = compact_paired_lesson_card(card)
    ident = paired_id(card)
    assert wire['schema'] == PAIRED_LESSON_SCHEMA
    assert wire['comparison_catalog'][ident] == {'paired_ref': True}
    assert wire['expanded_sha256'] == canonical_sha(card)
    assert wire['source_observations'] == card['source_observations']
    assert wire['authoring_facts'] == card['authoring_facts']
    assert len(canonical_json(wire)) < len(before)
    expanded = expand_paired_lesson_card(json.loads(canonical_json(wire)))
    assert expanded['schema'] == CARD_SCHEMA and canonical_json(expanded) == before
    expanded['source_observations'][0]['id'] = 'changed'
    wire['source_observations'][0]['id'] = 'also_changed'
    assert canonical_json(card) == before


def test_no_eligible_source_and_plain_legacy_decoder_are_detached_passthrough(cards):
    for card in (cards['ordinary'], {'schema': CARD_SCHEMA, 'opaque': {'paired_ref': True}},
                 {'schema': CARD_SCHEMA, 'source_observations': None}):
        before = canonical_json(card)
        compact = compact_paired_lesson_card(card)
        expanded = expand_paired_lesson_card(card)
        assert canonical_json(compact) == canonical_json(expanded) == before
        assert compact is not card and expanded is not card


def test_only_same_id_source_slot_is_referenced_and_exposure_copy_stays_inline(cards):
    source = cards['source']['source_observations'][0]
    ident = source['id']
    projection = cards['source']['comparison_catalog'][ident]
    for kind in ('ordinary', 'source'):
        card = copy.deepcopy(cards[kind])
        card['advice_exposures']['representatives'].append({
            'observation': copy.deepcopy(source), 'attribution': []})
        card['comparison_catalog'][ident] = copy.deepcopy(projection)
        wire = compact_paired_lesson_card(card)
        assert wire['advice_exposures'] == card['advice_exposures']
        if kind == 'ordinary':
            assert canonical_json(wire) == canonical_json(card)
        else:
            assert wire['comparison_catalog'][ident] == {'paired_ref': True}
        assert canonical_json(expand_paired_lesson_card(wire)) == canonical_json(card)


def test_opaque_reference_lookalikes_and_json_distinctions_remain_literal(cards):
    card = copy.deepcopy(cards['mixed'])
    opaque = {'paired_ref': True, 'nested': {'schema': PAIRED_LESSON_SCHEMA,
        'comparison_catalog': {'x': {'paired_ref': True}}},
        'null': None, 'integer': 0, 'float': 0.0, 'negative_zero': -0.0, 'bool': False}
    ordinary = next(row for row in card['source_observations'] if 'schema' not in row)
    ordinary['parent_cases'] = opaque
    card['unverified_interpretation'] = '{"paired_ref":true}; literal text only'
    expanded = expand_paired_lesson_card(compact_paired_lesson_card(card))
    assert canonical_json(expanded) == canonical_json(card)
    value = next(row for row in expanded['source_observations'] if 'schema' not in row)['parent_cases']
    assert 'absent' not in value and value['null'] is None
    assert type(value['integer']) is int and type(value['float']) is float
    assert math.copysign(1, value['negative_zero']) == -1
    assert type(value['bool']) is bool and value['paired_ref'] is True


@pytest.mark.parametrize('damage', ['absent', 'null', 'extra', 'bool_alias', 'int_float', 'signed_zero'])
def test_eligible_projection_mismatch_rejects_without_silent_repair(cards, damage):
    card = copy.deepcopy(cards['source']); ident = paired_id(card)
    target = card['comparison_catalog'][ident]
    if damage == 'absent':
        del card['comparison_catalog'][ident]
    elif damage == 'null':
        card['comparison_catalog'][ident] = None
    elif damage == 'extra':
        target['extra'] = 'not part of the six-field projection'
    elif damage == 'bool_alias':
        target['derived']['all_valid'] = 1
    elif damage == 'int_float':
        target['derived']['pairs'][0]['seed'] = float(target['derived']['pairs'][0]['seed'])
    else:
        source = card['source_observations'][0]
        source['derived']['pairs'][0]['raw_delta'] = 0.0
        source['view_sha256'] = canonical_sha({k:v for k,v in source.items() if k!='view_sha256'})
        target['derived']['pairs'][0]['raw_delta'] = -0.0
    before = canonical_json(card)
    with pytest.raises(ValueError, match='projection mismatch'):
        compact_paired_lesson_card(card)
    assert canonical_json(card) == before


@pytest.mark.parametrize('damage', ['source_ids_missing', 'source_ids_duplicate', 'sources_duplicate',
    'source_id_changed', 'view_hash', 'expanded_hash', 'false_ref', 'integer_ref', 'extra_ref',
    'missing_target', 'unsupported_target', 'exposure_target', 'no_references'])
def test_malformed_transport_or_source_bindings_reject(cards, damage):
    wire = compact_paired_lesson_card(cards['source']); ident = paired_id(cards['source'])
    if damage == 'source_ids_missing': wire.pop('source_ids')
    elif damage == 'source_ids_duplicate': wire['source_ids'].append(ident)
    elif damage == 'sources_duplicate': wire['source_observations'].append(copy.deepcopy(wire['source_observations'][0]))
    elif damage == 'source_id_changed': wire['source_observations'][0]['id'] = 'different'
    elif damage == 'view_hash': wire['source_observations'][0]['view_sha256'] = '0'*64
    elif damage == 'expanded_hash': wire['expanded_sha256'] = '0'*64
    elif damage == 'false_ref': wire['comparison_catalog'][ident]['paired_ref'] = False
    elif damage == 'integer_ref': wire['comparison_catalog'][ident]['paired_ref'] = 1
    elif damage == 'extra_ref': wire['comparison_catalog'][ident]['extra'] = None
    elif damage == 'missing_target':
        wire['source_ids'] = []; wire['source_observations'] = []
    elif damage == 'unsupported_target': wire['source_observations'][0]['schema'] = 'unsupported-paired-view/99'
    elif damage == 'exposure_target':
        wire['advice_exposures']['representatives'] = [{'observation': wire['source_observations'].pop()}]
        wire['source_ids'] = []
    else: wire['comparison_catalog'][ident] = copy.deepcopy(cards['source']['comparison_catalog'][ident])
    before = canonical_json(wire)
    with pytest.raises(ValueError): expand_paired_lesson_card(wire)
    assert canonical_json(wire) == before


def test_invalid_eligible_view_is_rejected_before_compaction(cards):
    card = copy.deepcopy(cards['legacy'])
    card['source_observations'][0]['derived']['all_valid'] = False
    with pytest.raises(ValueError, match='integrity'):
        compact_paired_lesson_card(card)


def test_rehashed_transport_cannot_repair_inconsistent_inline_projection(cards):
    wire = compact_paired_lesson_card(cards['source']); ident = paired_id(cards['source'])
    wire['comparison_catalog'][ident] = copy.deepcopy(cards['source']['comparison_catalog'][ident])
    wire['comparison_catalog'][ident]['derived']['all_valid'] = 1
    forged = {k: copy.deepcopy(v) for k, v in wire.items() if k != 'expanded_sha256'}
    forged['schema'] = CARD_SCHEMA
    wire['expanded_sha256'] = canonical_sha(forged)
    with pytest.raises(ValueError, match='projection mismatch'):
        expand_paired_lesson_card(wire)


def test_unknown_plain_source_schema_is_not_silently_upgraded(cards):
    card = copy.deepcopy(cards['legacy'])
    card['source_observations'][0]['schema'] = 'future-paired-card/99'
    assert canonical_json(compact_paired_lesson_card(card)) == canonical_json(card)
