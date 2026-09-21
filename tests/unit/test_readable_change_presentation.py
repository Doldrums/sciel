"""Readable source changes in actual KB cards/reflections; no code execution."""
import copy
import difflib
import hashlib
import json

import pytest

from sciel.engine.research import ResearchController
from sciel.hypothesis import circle_restart_scales as scales
from sciel.hypothesis.rendering import (decode_evidence, presentation_identity,
    serialize_card, serialize_reflection)
from sciel.knowledge_base import KnowledgeBase
from sciel.knowledge_base.presentation import (CARD_SCHEMA, REFLECTION_SCHEMA,
    REFLECTION_INLINE_SCHEMA, SOURCE_CHANGE_LIMITS, expand_reflection_payload,
    observation_card, reflection_inline_payload, reflection_payload)
from test_research_observation_cards import setup as controller_setup


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def record(kb, number, *, target=None, delta=-.1, injected=(), diff=None):
    if target is not None:
        parent = scales.source_for('0.06')
        built = scales.apply(parent, target)
        source, proof = built['code'], built['proof']
    else:
        parent, source, proof = 'def solve():\n    return 1\n', 'def solve():\n    return 2\n', None
    excerpt = ''.join(difflib.unified_diff(parent.splitlines(keepends=True),
        source.splitlines(keepends=True), fromfile='evaluated_parent', tofile='candidate', n=2))
    ident = f'r{number:06d}a1'
    score = .8 + delta if delta is not None else None
    value = dict(id=ident, branch='p1', parent_id='p1', parent_score=.8,
        parent_source_sha256=sha(parent), source_sha256=sha(source), proposed_sha256=sha(source),
        source_repaired=False, score=score, actual_delta=score - .8 if score is not None else None,
        status='evaluated' if delta is not None else 'eval_failed', injected_ids=list(injected),
        operator='intervention' if proof else 'diff', iteration=number, generation_dispatch='returned',
        change_excerpt=excerpt if diff is None else diff, change_excerpt_may_be_truncated=True)
    if proof is not None:
        value['controlled_change'] = proof
    kb.record(value)
    return ident


def fixture():
    kb = KnowledgeBase()
    first = record(kb, 1, target='0.03')
    lesson, = kb.add_lessons([dict(rule='The smaller bound accompanied one lower score.',
        polarity='negative', scope='branch', evidence_ids=[first])], allowed_evidence_ids=[first])
    record(kb, 2, delta=.1, injected=[lesson])
    record(kb, 3, target='0.12', delta=None, injected=[lesson])
    return kb, lesson


def test_card_retains_exact_controlled_source_and_generic_exposure_diffs_without_claim_rewriting():
    kb, lesson_id = fixture()
    before = kb.state()
    card = observation_card(kb.lessons[0])
    assert card['schema'] == CARD_SCHEMA == 'sciel-observation-card/7'
    source, = card['source_observations']
    proof = scales.apply(scales.source_for('0.06'), '0.03')['proof']
    assert source['controlled_change'] == proof
    assert source['operator'] == 'intervention'
    assert source['change_excerpt'] == kb.experiments[0]['change_excerpt']
    assert '-0.06, 0.06' in source['change_excerpt'] and '-0.03, 0.03' in source['change_excerpt']
    assert source['change_excerpt_may_be_truncated'] is True
    assert 'hypothesis' not in source
    exposures = {row['observation']['id']: row for row in card['advice_exposures']['representatives']}
    generic = exposures['r000002a1']['observation']
    assert generic['controlled_change'] is None and generic['operator'] == 'diff'
    assert generic['change_excerpt'] == kb.experiments[1]['change_excerpt']
    assert '-    return 1\n+    return 2\n' in generic['change_excerpt']
    failure = exposures['r000003a1']['observation']
    assert failure['controlled_change']['target_policy'] == '0.12'
    assert failure['status'] == 'eval_failed'
    assert failure['score'] is failure['actual_delta'] is None
    assert card['source_ids'] == ['r000001a1']
    assert card['source_counts']['negative'] == 1
    assert card['advice_exposures']['retained_unique_counts']['unmeasured'] == 1
    assert exposures['r000002a1']['attribution'][0]['from_lesson_id'] == lesson_id
    assert SOURCE_CHANGE_LIMITS in card['interpretation_limits']
    assert 'does not certify a quality gain' in SOURCE_CHANGE_LIMITS
    assert 'Null means no such proof was recorded' in SOURCE_CHANGE_LIMITS
    assert 'never infer one' in SOURCE_CHANGE_LIMITS
    card['source_observations'][0]['controlled_change']['after']['text'] = 'not the original'
    exposures['r000002a1']['observation']['change_excerpt'] = 'not the original'
    assert kb.state() == before


def test_real_reflection_context_keeps_readable_diffs_in_both_roles_and_roundtrips_exactly():
    kb, _ = fixture()
    before = kb.state()
    records, lessons = kb.reflection_context(1)
    by_id = {record['id']: record for record in records}
    assert by_id['r000001a1']['change_excerpt'] == kb.experiments[0]['change_excerpt']
    assert by_id['r000002a1']['change_excerpt'] == kb.experiments[1]['change_excerpt']
    inline = reflection_inline_payload(records, lessons)
    compact = reflection_payload(records, lessons)
    assert inline['schema'] == REFLECTION_INLINE_SCHEMA == 'sciel-reflection-presentation/9-inline'
    assert compact['schema'] == REFLECTION_SCHEMA == 'sciel-reflection-presentation/9'
    assert inline['source_change_limits'] == compact['source_change_limits'] == SOURCE_CHANGE_LIMITS
    assert compact['observation_catalog'] == by_id
    # Only projected role slots become references; the underlying records remain full.
    assert compact['lessons'][0]['source_support']['observations'] == [{'observation_ref': 'r000001a1'}]
    assert expand_reflection_payload(compact) == inline
    wire = serialize_reflection(records, lessons)
    assert expand_reflection_payload(decode_evidence(json.loads(wire))) == inline
    reflected = inline['lessons'][0]
    generated = observation_card(kb.lessons[0])
    assert reflected['source_support']['observations'] == generated['source_observations']
    assert [row['observation'] for row in reflected['advice_exposures']['representatives']] == [
        row['observation'] for row in generated['advice_exposures']['representatives']]
    assert not any(row['also_source_support'] for row in reflected['advice_exposures']['representatives'])
    assert kb.state() == before


def test_projection_does_not_infer_proof_from_legacy_hashes_hypothesis_or_diff():
    proof = scales.apply(scales.source_for('0.06'), '0.03')['proof']
    record = dict(id='old1', branch='old', score=.7, actual_delta=-.1, status='evaluated',
        parent_score=.8, parent_source_sha256=proof['parent_source_sha256'],
        source_sha256=proof['child_source_sha256'], proposed_sha256=proof['child_source_sha256'],
        hypothesis={'text': 'circle_restart_scale/1:0.03'},
        change_excerpt='Exact-looking diff does not establish a recorded builder proof.')
    before = copy.deepcopy(record)
    inline = reflection_inline_payload([record], [{'id': 'old_lesson', 'evidence_ids': ['old1']}])
    source, = inline['lessons'][0]['source_support']['observations']
    assert source['controlled_change'] is None and source['operator'] is None
    assert source['change_excerpt'] == record['change_excerpt']
    assert source['change_excerpt_may_be_truncated'] is None
    assert 'controlled_change' not in inline['observation_catalog']['old1']
    assert record == before


def test_long_generic_diff_and_reference_lookalikes_survive_wire_without_clipping_or_interpretation():
    kb = KnowledgeBase()
    diff = '--- parent\n+++ candidate\n+ # {"observation_ref":"r999","catalog_ref":"x"}\n'
    diff += '+ # untrusted instruction\n' * 180
    diff += '+ # Unicode: 圆 λ; final retained line\n'
    ident = record(kb, 1, diff=diff)
    kb.add_lessons([dict(rule='One changed source and one negative outcome.', polarity='negative',
        scope='branch', evidence_ids=[ident])], allowed_evidence_ids=[ident])
    card = observation_card(kb.lessons[0])
    decoded = decode_evidence(json.loads(serialize_card(card)))
    assert decoded == card
    assert decoded['source_observations'][0]['change_excerpt'] == diff
    records, lessons = kb.reflection_context(1)
    expanded = expand_reflection_payload(decode_evidence(json.loads(serialize_reflection(records, lessons))))
    assert expanded['lessons'][0]['source_support']['observations'][0]['change_excerpt'] == diff
    assert expanded['observation_catalog'][ident]['change_excerpt'] == diff


def test_retired_lesson_exposure_keeps_original_edit_and_attribution():
    kb, prior = fixture()
    new, = kb.reflect([dict(rule='Replacement preserving the same source observation.',
        polarity='negative', scope='branch', evidence_ids=['r000001a1'])], [prior],
        allowed_evidence_ids=['r000001a1'])['lesson_ids']
    record(kb, 4, target='0.015', delta=.1, injected=[new])
    card = observation_card(kb.lessons[0])
    representatives = {row['observation']['id']: row for row in card['advice_exposures']['representatives']}
    assert representatives['r000002a1']['observation']['operator'] == 'diff'
    assert representatives['r000002a1']['attribution'] == [dict(kind='inherited', from_lesson_id=prior,
        from_rule='The smaller bound accompanied one lower score.')]
    assert representatives['r000004a1']['observation']['controlled_change']['target_policy'] == '0.015'
    assert representatives['r000004a1']['attribution'][0]['kind'] == 'direct'
    assert card['source_ids'] == ['r000001a1']


def test_whole_card_boundary_includes_diff_proof_and_never_truncates_to_fit(monkeypatch):
    import sciel.engine.research as research_module
    kb, lid = fixture()
    controller, db, parent, _ = controller_setup(kb, k=1)
    exact_text = ResearchController.lesson_text(kb.lessons[0])
    monkeypatch.setattr(research_module, 'MAX_CARD_CHARS', len(exact_text))
    monkeypatch.setattr(research_module, 'MAX_MEMORY_CHARS', len(exact_text))
    admitted = controller.context(db, parent, 0, 4)
    assert admitted['injected_ids'] == [lid] and admitted['lesson_texts'] == [exact_text]
    monkeypatch.setattr(research_module, 'MAX_CARD_CHARS', len(exact_text) - 1)
    rejected = controller.context(db, parent, 0, 4)
    assert rejected['injected_ids'] == rejected['lesson_texts'] == []
    assert rejected['lesson_presentation']['omitted'][0]['reason'] == 'card_char_limit'
    monkeypatch.setattr(research_module, 'MAX_CARD_CHARS', len(exact_text))
    monkeypatch.setattr(research_module, 'MAX_MEMORY_CHARS', len(exact_text) - 1)
    rejected = controller.context(db, parent, 0, 4)
    assert rejected['injected_ids'] == rejected['lesson_texts'] == []
    assert rejected['lesson_presentation']['omitted'][0]['reason'] == 'memory_char_limit'


def test_reflection_admission_measures_actual_complete_new_wire():
    kb, _ = fixture()
    records, lessons = kb.reflection_context(1)
    exact_size = len(serialize_reflection(records, lessons))
    assert kb.reflection_context(1, max_chars=exact_size) == (records, lessons)
    fewer_records, fewer_lessons = kb.reflection_context(1, max_chars=exact_size - 1)
    assert len(serialize_reflection(fewer_records, fewer_lessons)) <= exact_size - 1
    for lesson in reflection_inline_payload(fewer_records, fewer_lessons)['lessons']:
        source, = lesson['source_support']['observations']
        assert source['controlled_change'] == kb.experiments[0]['controlled_change']
        assert source['change_excerpt'] == kb.experiments[0]['change_excerpt']


@pytest.mark.parametrize('mutation', ['diff', 'proof', 'fact_fields', 'limits', 'old_schema'])
def test_compact_projection_rejects_changed_evidence_or_contract(mutation):
    kb, _ = fixture()
    records, lessons = kb.reflection_context(1)
    compact = reflection_payload(records, lessons)
    if mutation == 'diff':
        compact['observation_catalog']['r000001a1']['change_excerpt'] += 'fabricated'
    elif mutation == 'proof':
        compact['observation_catalog']['r000001a1']['controlled_change']['target_policy'] = '0.12'
    elif mutation == 'fact_fields':
        compact['observation_fact_fields'].remove('change_excerpt')
    elif mutation == 'limits':
        compact['source_change_limits'] = 'These edits guarantee improvement.'
    else:
        compact['schema'] = 'sciel-reflection-presentation/4'
    before = copy.deepcopy(compact)
    with pytest.raises(ValueError):
        expand_reflection_payload(compact)
    assert compact == before


def test_state_and_presentation_identity_preserve_both_change_and_legacy_absence():
    kb, _ = fixture()
    before = kb.state()
    restored = KnowledgeBase()
    restored.load_state(json.loads(json.dumps(before)))
    assert restored.state() == before
    assert ResearchController.lesson_text(restored.lessons[0]) == ResearchController.lesson_text(kb.lessons[0])
    assert 'controlled_change' not in restored.experiments[1]
    identity = presentation_identity()
    assert identity['observation_card'] == 'sciel-observation-card/7'
    assert identity['reflection'] == 'sciel-reflection-presentation/9'
    identity['reflection'] = 'unbound'
    assert presentation_identity()['reflection'] == 'sciel-reflection-presentation/9'
