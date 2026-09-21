"""Exact shown-memory transport only: no calls, candidates or numerical runs."""
from copy import deepcopy
import hashlib
import json

import pytest

from sciel.engine.research import ResearchController
from sciel.hypothesis.rendering import decode_evidence, reference_rules
from sciel.knowledge_base import KnowledgeBase
from sciel.knowledge_base import memory_references as refs
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha, pack_measurements
from sciel.knowledge_base.presentation import compact_paired_lesson_card, omit_source_context
from test_paired_source_context import build as source_group
from test_research_knowledge import proposal


def resize(value):
    value['rendered_chars'] = 0
    while value['rendered_chars'] != len(canonical_json(value)):
        value['rendered_chars'] = len(canonical_json(value))
    return value


@pytest.fixture
def shown():
    kb = KnowledgeBase()
    group = source_group()
    kb.record_comparison(group)
    kb.add_lessons([proposal(group['id'])], allowed_evidence_ids=[group['id']])
    lesson = kb.lessons[0]
    return kb, kb.comparison_context('p1'), {lesson['id']: ResearchController.lesson_text(lesson)}


def test_actual_framed_lesson_roundtrip_preserves_whole_context_and_state(shown):
    kb, context, texts = shown
    original, state, old_texts = canonical_json(context), canonical_json(kb.state()), deepcopy(texts)
    factored, saved = refs.factor_comparisons(context, texts)
    assert factored['schema'] == refs.SCHEMA and saved > 0
    card = factored['cards'][0]
    assert card['schema'] == refs.REFERENCE_SCHEMA
    assert card['card_sha256'] == canonical_sha(context['cards'][0])
    assert card['lesson_text_sha256'] == hashlib.sha256(next(iter(texts.values())).encode()).hexdigest()
    assert saved == len(original) - len(canonical_json(factored)) - len(refs.RULES)
    expanded = refs.expand_comparisons(factored, texts)
    assert canonical_json(expanded) == original
    assert {k:v for k,v in expanded.items() if k != 'cards'} == {k:v for k,v in context.items() if k != 'cards'}
    expanded['cards'][0]['branch'] = 'detached'
    assert canonical_json(kb.state()) == state and canonical_json(context) == original and texts == old_texts


def test_source_context_omission_or_any_full_card_difference_prevents_factoring(shown):
    _, context, texts = shown
    context['cards'][0] = omit_source_context(context['cards'][0])
    resize(context)
    assert refs.factor_comparisons(context, texts) == (context, 0)
    # Same ID and evidence digest are intentionally insufficient.
    assert context['cards'][0]['source_context']['omission_reason'] == 'presentation_budget'


def test_first_actual_shown_lesson_order_wins_without_reranking(shown):
    _, context, texts = shown
    original_id, text = next(iter(texts.items()))
    prefix, payload = text.split(refs.MARKER)
    decoded = decode_evidence(json.loads(payload))
    decoded['lesson_id'] = 'shown-first'
    first_text = prefix + refs.MARKER + canonical_json(decoded)
    mapping = {'shown-first': first_text, original_id: text}
    factored, _ = refs.factor_comparisons(context, mapping)
    assert factored['cards'][0]['lesson_id'] == 'shown-first'
    assert refs.expand_comparisons(factored, mapping) == context


@pytest.mark.parametrize('damage', ['missing_text', 'changed_text', 'lesson_id', 'text_sha', 'card_hashes'])
def test_reference_tampering_never_resolves_from_another_lesson_or_id(shown, damage):
    _, context, texts = shown
    factored, _ = refs.factor_comparisons(context, texts)
    target = factored['cards'][0]
    if damage == 'missing_text': texts = {}
    elif damage == 'changed_text': texts[target['lesson_id']] = 'Changed framing.\n' + texts[target['lesson_id']]
    elif damage == 'lesson_id': target['lesson_id'] = 'not-shown'
    elif damage == 'text_sha': target['lesson_text_sha256'] = '0'*64
    else:
        for key in ('view_sha256', 'evidence_sha256', 'card_sha256'):
            damaged = deepcopy(factored)
            damaged['cards'][0][key] = '0'*64
            resize(damaged)
            with pytest.raises(ValueError): refs.expand_comparisons(damaged, texts)
        return
    resize(factored)
    with pytest.raises(ValueError): refs.expand_comparisons(factored, texts)


def test_measurement_outer_and_paired_lesson_references_expand_before_matching(shown):
    _, context, texts = shown
    ident, text = next(iter(texts.items()))
    prefix, payload = text.split(refs.MARKER)
    card = decode_evidence(json.loads(payload))
    metric = {'literal': 'large repeated exact measurement; ' * 60, 'float': -0., 'nullable': None}
    card['source_observations'].append({'id':'ordinary', 'metric':deepcopy(metric), 'parent_cases':deepcopy(metric)})
    card['source_ids'].append('ordinary')
    compact = compact_paired_lesson_card(card)
    assert compact['schema'] == 'sciel-paired-lesson-references/1'
    outer = pack_measurements(compact, [('source_observations',1,'metric'), ('source_observations',1,'parent_cases')])
    assert outer['bindings']
    wire = canonical_json(outer)
    texts = {ident: prefix + reference_rules(wire) + refs.MARKER + wire}
    assert canonical_json(decode_evidence(json.loads(wire))) == canonical_json(card)
    factored, saved = refs.factor_comparisons(context, texts)
    assert saved > 0 and refs.expand_comparisons(factored, texts) == context


def test_no_recursive_search_of_opaque_lookalikes_or_exposure_only_cards(shown):
    _, context, texts = shown
    ident, text = next(iter(texts.items()))
    prefix, payload = text.split(refs.MARKER)
    card = decode_evidence(json.loads(payload))
    card['opaque'] = {'source_observations': card['source_observations']}
    card['advice_exposures']['representatives'] = [{'observation':card['source_observations'][0]}]
    card['source_ids'] = []; card['source_observations'] = []
    texts = {ident:prefix + refs.MARKER + canonical_json(card)}
    assert refs.factor_comparisons(context, texts) == (context, 0)


def test_rules_cost_and_missing_marker_keep_original_detached(shown, monkeypatch):
    _, context, texts = shown
    monkeypatch.setattr(refs, 'RULES', 'long rules ' * len(canonical_json(context)))
    factored, saved = refs.factor_comparisons(context, texts)
    assert factored == context and saved == 0 and factored is not context
    monkeypatch.undo()
    assert refs.factor_comparisons(context, {'unknown':'{"source_observations":[]}'} ) == (context, 0)
    assert refs.expand_comparisons(context, {}) == context
    assert refs.factor_comparisons(None, {}) == (None, 0)


def test_changed_shown_identity_and_noncanonical_suffix_reject(shown):
    _, context, texts = shown
    ident, text = next(iter(texts.items()))
    with pytest.raises(ValueError, match='shown lesson identity'):
        refs.factor_comparisons(context, {'another-id':text})
    with pytest.raises(ValueError, match='canonical JSON'):
        refs.factor_comparisons(context, {ident:text+'\n'})
