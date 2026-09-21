"""Lossless prompt delivery: pure serialization and inert actual Engine routes."""
import copy
import json
import sys

import pytest

from sciel.engine import checkpoint
from sciel.engine.loop import Engine
from sciel.engine.research import ResearchController, digest
from sciel.hypothesis import rendering
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha, pack_measurements
from sciel.knowledge_base.presentation import (CARD_SCHEMA, PAIRED_LESSON_SCHEMA,
    compact_paired_lesson_card, observation_card)
from test_paired_lesson_transport import cards
from test_repeat_engine_integration import config, FakeDraws, task, search_observations, journal
from test_repeat_reflection_lifecycle import ReflectingModel


def parse(text, marker):
    return json.JSONDecoder().raw_decode(text.split(marker, 1)[1].lstrip())[0]


def build(task, folder, measured, *, start=0):
    cfg = config(budget=2, max_iterations=2)
    cfg['research'].update(batch_only_lessons=True, reflection_interval=1, max_aux_calls=4)
    model = ReflectingModel(measured, start_step=start)
    return Engine(cfg, task, str(folder), model, python=sys.executable), model


def test_reflected_paired_lesson_reaches_next_generation_with_exact_facts(
        task, tmp_path, monkeypatch, search_observations):
    measured = FakeDraws(monkeypatch)
    folder = tmp_path/'delivery'
    first, first_model = build(task, folder, measured)
    first._stop_at = 1
    first.run()
    lesson, = first.research.knowledge.lessons
    card = observation_card(lesson)
    framed = ResearchController.lesson_text(lesson)
    wire = parse(framed, 'RESEARCH_LESSON_JSON:')
    assert wire['schema'] == PAIRED_LESSON_SCHEMA
    assert canonical_json(rendering.decode_evidence(wire)) == canonical_json(card)
    assert len(framed) <= 16000
    second, second_model = build(task, folder, measured, start=1)
    checkpoint.load(second, str(folder/'checkpoint.json'))
    second.run()
    generation, = [c for c in second_model.calls if c['kind'] == 'mutation']
    assert framed in generation['prompt']
    assert rendering.PAIRED_REFERENCE_RULES in generation['prompt']
    restored = rendering.decode_evidence(parse(generation['prompt'], 'RESEARCH_LESSON_JSON:'))
    assert canonical_json(restored) == canonical_json(card)
    attempt = next(e for e in journal(folder)
                   if e.get('phase') == 'attempt' and e['experiment_id'] == 'r000002a1')
    presentation = attempt['lesson_presentation']
    assert presentation['presented'] == [dict(id=lesson['id'], chars=len(framed),
        sha256=digest(framed))]
    assert presentation['omitted'] == []
    assert presentation['rendered_chars'] == len(framed)
    assert presentation['max_card_chars'] == 16000 and presentation['max_memory_chars'] == 32000
    direct = attempt['paired_comparisons']
    assert card['source_ids'][0] in [c['id'] for c in direct['cards']]
    assert len(framed) + len(canonical_json(direct)) <= 32000
    exposed = next(l for l in second.research.knowledge.lessons if l['id'] == lesson['id'])
    assert exposed['exposure_ids'] == ['r000002a1']
    assert exposed['source_support_ids'] == lesson['source_support_ids']
    assert len(measured.calls) == 15 and len(search_observations) == 3
    calls = first_model.calls + second_model.calls
    assert [c['kind'] for c in calls] == ['mutation', 'conclusion', 'reflection'] * 2
    assert second.research.aux_calls == 4 and second.cost_total == pytest.approx(.006)


def test_exact_card_cap_admits_and_one_character_less_omits_without_exposure(
        task, tmp_path, monkeypatch, search_observations):
    measured = FakeDraws(monkeypatch)
    engine, _ = build(task, tmp_path/'boundary', measured)
    engine._stop_at = 1
    engine.run()
    lesson, = engine.research.knowledge.lessons
    before = canonical_json(engine.research.knowledge.state())
    size = len(engine.research.lesson_text(lesson))
    group, = engine.research.knowledge.comparisons
    parent = engine.db._all[group['branch']]
    monkeypatch.setattr('sciel.engine.research.MAX_CARD_CHARS', size)
    visible = engine.research.context(engine.db, parent, 0, 2)
    assert visible['injected_ids'] == [lesson['id']]
    assert visible['lesson_presentation']['rendered_chars'] == size
    monkeypatch.setattr('sciel.engine.research.MAX_CARD_CHARS', size - 1)
    hidden = engine.research.context(engine.db, parent, 0, 2)
    assert hidden['injected_ids'] == [] and hidden['lesson_texts'] == []
    assert hidden['lesson_presentation']['omitted'] == [
        dict(id=lesson['id'], reason='card_char_limit', chars=size)]
    assert hidden['paired_comparisons'] == visible['paired_comparisons']
    assert canonical_json(engine.research.knowledge.state()) == before
    assert lesson['exposure_ids'] == [] and len(measured.calls) == 9


def test_old_presentation_identity_cannot_resume_and_does_not_mutate_state(
        task, tmp_path, monkeypatch, search_observations):
    measured = FakeDraws(monkeypatch)
    folder = tmp_path/'identity'
    engine, _ = build(task, folder, measured)
    engine._stop_at = 1
    engine.run()
    state = json.loads((folder/'checkpoint.json').read_text())
    del state['identity']['research']['presentation']['paired_lesson_references']
    old = folder/'old-presentation.json'
    old.write_text(canonical_json(state))
    fresh, model = build(task, folder, measured, start=1)
    before = fresh.research.state()
    try:
        with pytest.raises(checkpoint.IdentityMismatch):
            checkpoint.load(fresh, str(old))
        assert fresh.research.state() == before and model.calls == []
        assert len(measured.calls) == 9
        bad_research = copy.deepcopy(engine.research.state())
        del bad_research['presentation']['paired_lesson_references']
        with pytest.raises(ValueError):
            fresh.research.load_state(bad_research)
        assert fresh.research.state() == before
    finally:
        fresh.repeat_promotion.close()
        fresh.log.close()


def test_paired_wire_and_measurement_outer_both_roundtrip_and_charge_rules(cards):
    card = copy.deepcopy(cards['mixed'])
    ordinary = card['source_observations'][0]
    # Opaque equal measurements can factor without interpreting their content.
    measurement = {'literal': 'Ж' * 2000, 'paired_ref': True, 'null': None, 'zero': -0.0}
    ordinary['parent_cases'] = copy.deepcopy(measurement)
    ordinary['metric'] = copy.deepcopy(measurement)
    data = rendering.serialize_card(card)
    value = json.loads(data)
    assert value['schema'] == rendering.MEASUREMENT_SCHEMA
    assert value['payload']['schema'] == PAIRED_LESSON_SCHEMA
    assert rendering.reference_rules(data) == rendering.REFERENCE_RULES + rendering.PAIRED_REFERENCE_RULES
    assert canonical_json(rendering.decode_evidence(value)) == canonical_json(card)
    assert len(rendering.PAIRED_REFERENCE_RULES) == 159


def test_native_evaluation_projection_precedes_paired_expanded_hash(cards):
    from test_native_geometry_knowledge import outcome
    from sciel.hypothesis.measurement import evaluation_observation, evaluation_view
    card = copy.deepcopy(cards['mixed'])
    full = evaluation_observation(outcome())  # Pure fixed arrays; no run_stage.
    card['source_observations'][0]['evaluation'] = full
    expected = copy.deepcopy(card)
    expected['source_observations'][0]['evaluation'] = evaluation_view(full)
    before = canonical_json(card)
    raw = json.loads(rendering.serialize_card(card))
    wire = (raw['payload'] if raw['schema'] == rendering.MEASUREMENT_SCHEMA else raw)
    assert wire['schema'] == PAIRED_LESSON_SCHEMA
    assert wire['expanded_sha256'] == canonical_sha(expected)
    assert wire['expanded_sha256'] != canonical_sha(card)
    assert canonical_json(rendering.decode_evidence(raw)) == canonical_json(expected)
    assert canonical_json(card) == before


def test_plain_ordinary_bytes_legacy_decode_and_nested_lookalikes_are_unchanged(cards):
    card = copy.deepcopy(cards['ordinary'])
    expected = canonical_json(card)
    assert rendering.serialize_card(card) == expected
    assert rendering.reference_rules(expected) == ''
    assert canonical_json(rendering.decode_evidence(json.loads(expected))) == expected
    opaque = {'schema': 'opaque-task', 'nested': compact_paired_lesson_card(cards['source'])}
    outer = pack_measurements(opaque, [])
    assert canonical_json(rendering.decode_evidence(outer)) == canonical_json(opaque)
    assert rendering.reference_rules(canonical_json(outer)) == rendering.REFERENCE_RULES


def test_net_wire_cost_uses_legacy_when_rules_erase_saving(cards, monkeypatch):
    card = cards['source']
    # A deliberately expensive explanation tests net accounting, not payload only.
    monkeypatch.setattr(rendering, 'PAIRED_REFERENCE_RULES', 'x' * 100000)
    data = rendering.serialize_card(card)
    assert json.loads(data)['schema'] == CARD_SCHEMA
    assert data == canonical_json(card) and rendering.reference_rules(data) == ''


@pytest.mark.parametrize('batch', [False, True])
def test_identity_distinguishes_new_transport_in_both_modes(batch):
    identity = rendering.presentation_identity(batch_only_lessons=batch)
    assert identity['paired_lesson_references'] == PAIRED_LESSON_SCHEMA
    assert ('conclusion_lesson_admission' in identity) is batch
