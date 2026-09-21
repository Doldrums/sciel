"""Whole-history feedback uses saved outputs, never candidate execution."""
from copy import deepcopy
import hashlib
import json

import pytest

from sciel.engine.observed_changes import ObservedChanges, _validate_record
from sciel.engine.types import Program
from sciel.eval.registry import load_task
from sciel.hypothesis.measurement import metric_context
from test_observed_changes import (ROOT, canonical, geometry_observation, observation,
                                   program, prompt_records, reseal, sha)


def history(task='tasks/test/toy_quadratic'):
    return ObservedChanges(metric_context(load_task(str(ROOT / task))))


def cases_observation(score=0., **changes):
    result = observation(score, validity=None, reached='full')
    result['cases'] = {'loss': 0., 'accuracy': .25}
    result.update(changes)
    return result


def match_card(h, code, **kwargs):
    block = h.block(code, **kwargs)
    return next((r['history_matches'] for r in prompt_records(block)
                 if 'history_matches' in r), None) if block else None


def test_cross_island_nonancestor_cases_are_delivered_after_population_eviction_and_resume():
    h = history()
    a, b = program('p29'), program('p33')
    a.island, b.island = 2, 8
    h.observe(a, cases_observation())
    h.observe(b, cases_observation())
    # No shared parent and no active-population object is required by history.
    active_population = {b.id: b}
    assert a.id not in active_population
    saved = h.state()
    card = match_card(h, b.code)
    assert card['kind'] == 'exact_case_vector_and_score'
    assert card['query']['program_id'] == b.id
    assert [r['program_id'] for r in card['matches']] == [a.id]
    assert card['matching_record_count'] == card['matching_distinct_sources'] == 1
    assert card['validity'] is None  # Generic success is not invented physical validity.
    assert card['query']['source_sha256'] == sha(b.code)
    assert card['matches'][0]['record_sha256'] == saved['records'][0]['record_sha256']
    restored = history(); restored.load_state(json.loads(json.dumps(saved)))
    assert restored.block(b.code) == h.block(b.code)
    assert restored.state() == saved
    assert 'post-evaluation' in restored.block(b.code)
    assert 'not matching behavior' in restored.block(b.code)


def test_exact_zero_geometry_is_not_confused_with_invalid_zero_sentinel():
    h = history('tasks/oe/circle_packing')
    a, b, invalid = program('p29'), program('p33'), program('p34')
    for p in (a, b):
        measured = geometry_observation(p)
        geo = json.loads(measured['artifacts']['native_geometry'])
        geo['numeric']['radii'] = [0.] * 26
        geo['numeric_sha256'] = hashlib.sha256(canonical(geo['numeric']).encode()).hexdigest()
        geo.update(native_sum_radii=0., native_score=0.)
        measured['artifacts']['native_geometry'] = canonical(geo)
        measured['score'] = 0.
        measured['cases'].update(sum_radii=0., target_ratio=0., combined_score=0.)
        h.observe(p, measured)
    h.observe(invalid, geometry_observation(invalid, invalid=True))
    card = match_card(h, b.code)
    assert card['kind'] == 'exact_validated_geometry' and card['validity'] is True
    assert card['score'] == 0. and card['matching_record_count'] == 1
    assert match_card(h, invalid.code) is None
    assert 'numeric' not in card['geometry'] and 'centers' not in card['geometry']


@pytest.mark.parametrize('change', ['cases', 'score', 'empty', 'nonfinite', 'bool', 'failed',
                                   'missing_flags', 'stage', 'confirmation'])
def test_generic_matches_require_whole_exact_successful_case_vector(change):
    h = history(); a, b = program('p1'), program('p2')
    h.observe(a, cases_observation())
    measured = cases_observation()
    if change == 'cases': measured['cases']['loss'] = 1e-12
    elif change == 'score': measured['score'] = 5e-10  # Within old score tolerance, not exact.
    elif change == 'empty': measured['cases'] = {}
    elif change == 'nonfinite': measured['cases']['loss'] = float('nan')
    elif change == 'bool': measured['cases']['loss'] = False
    elif change == 'failed': measured['ok'] = False
    elif change == 'missing_flags': measured['confirmed'] = None
    elif change == 'stage': measured['reached'] = 'smoke'
    else: measured.update(confirmed=True, reached='confirm')
    h.observe(b, measured)
    assert match_card(h, b.code) is None


@pytest.mark.parametrize('change', ['signed_zero', 'int_float'])
def test_numeric_case_representation_is_not_rounded_or_coerced(change):
    h = history(); a, b = program('p1'), program('p2')
    h.observe(a, cases_observation())
    measured = cases_observation(); measured['cases']['loss'] = -0. if change == 'signed_zero' else 0
    h.observe(b, measured)
    assert match_card(h, b.code) is None


def test_declared_invalid_or_missing_validity_cannot_create_generic_ties():
    h = history('tasks/oe/circle_packing')
    for i, validity in enumerate((0., 0., None, None), 1):
        measured = cases_observation(); measured['cases']['validity'] = validity
        h.observe(program(f'p{i}'), measured)
    for i in range(1, 5):
        assert match_card(h, program(f'p{i}').code) is None


def test_alias_chains_are_named_as_copies_and_do_not_inflate_measurement_count():
    h = history(); a, b = program('p1'), program('p2')
    measured = cases_observation(); a.score = measured['score']; a.cases = measured['cases']
    h.observe(a, measured)
    alias = Program('p3', a.code, a.score, deepcopy(a.cases), 4, 'migrant', a.id)
    copied = Program('p4', a.code, a.score, deepcopy(a.cases), 7, 'migrant', alias.id)
    h.bind_alias(alias, a); h.bind_alias(copied, alias)
    h.observe(b, measured)
    card = match_card(h, b.code)
    assert card['matching_record_count'] == 1
    assert card['matches'][0]['alias_program_ids'] == ['p3', 'p4']
    assert match_card(h, copied.code)['query']['alias_program_ids'] == ['p3', 'p4']
    restored = history(); restored.load_state(h.state())
    assert restored.block(b.code) == h.block(b.code)


def test_alias_cases_must_match_the_saved_observation_not_just_each_other():
    h = history(); a = program('p1'); measured = cases_observation()
    a.score, a.cases = measured['score'], {'other': 123.}
    h.observe(a, measured)
    alias = Program('p2', a.code, a.score, deepcopy(a.cases), 3, 'migrant', a.id)
    saved = h.state()
    with pytest.raises(ValueError, match='alias cases'):
        h.bind_alias(alias, a)
    assert h.state() == saved


def test_same_source_seed_registrations_do_not_pose_as_different_methods():
    h = history(); a = program('p1')
    for i in range(1, 25):
        h.observe(program(f'p{i}', a.code), cases_observation())
    assert h.block(a.code) == ''
    h.observe(program('p25'), cases_observation())
    card = match_card(h, program('p25').code)
    assert card['matching_distinct_sources'] == 1
    assert card['matches'][0]['same_source_record_count'] == 24
    assert len(card['matches']) == 1


@pytest.mark.parametrize('mismatch', ['evaluator', 'dtype', 'geometry'])
def test_geometry_mismatch_never_falls_back_to_a_misleading_case_tie(mismatch):
    h = history('tasks/oe/circle_packing'); a, b = program('p1'), program('p2')
    h.observe(a, geometry_observation(a))
    kwargs = {'evaluator': 'e' * 64} if mismatch == 'evaluator' else (
        {'dtype': 'float32'} if mismatch == 'dtype' else {'x_shift': .001})
    h.observe(b, geometry_observation(b, **kwargs))
    assert match_card(h, b.code) is None


def test_case_fingerprint_requires_a_bound_metric_and_cannot_change_on_restore():
    a, b = program('p1'), program('p2'); unbound = ObservedChanges()
    unbound.observe(a, cases_observation()); unbound.observe(b, cases_observation())
    assert match_card(unbound, b.code) is None
    h = history(); h.observe(a, cases_observation()); saved = h.state()
    damaged = deepcopy(saved)
    damaged['records'][0]['child_observation']['case_vector']['metric_sha256'] = 'f' * 64
    damaged['records'][0] = reseal(damaged['records'][0], 'record_sha256')
    damaged = reseal(damaged, 'state_sha256')
    with pytest.raises(ValueError, match='metric binding'):
        h.load_state(damaged)
    assert h.state() == saved


def test_legacy_record_without_case_fingerprint_remains_readable_without_fabrication():
    h = history(); a, b = program('p1'), program('p2')
    for p in (a, b):
        record = h.observe(p, cases_observation())
        record['child_observation'].pop('case_vector')
        record = reseal(record, 'record_sha256')
        assert 'case_vector' not in _validate_record(record)['child_observation']
    legacy = history()
    for record in h.state()['records']:
        record['child_observation'].pop('case_vector')
        legacy._insert(_validate_record(reseal(record, 'record_sha256')))
    assert legacy.block(b.code) == ''


def test_global_cards_remain_whole_json_under_budget_with_explicit_omissions():
    h = history()
    for i in range(1, 51):
        h.observe(program(f'p{i}'), cases_observation())
    block = h.block(program('p50').code)
    assert 0 < len(block) <= 4000
    card = match_card(h, program('p50').code)
    assert card['matching_distinct_sources'] == card['matching_record_count'] == 49
    assert 1 <= len(card['matches']) <= 2
    assert card['match_sources_omitted'] == 49 - len(card['matches'])
    assert card['query']['program_id'] == 'p50'
    for budget in (0, 10, 1000, 2000, 3000, 4000, 9000):
        rendered = h.block(program('p50').code, max_chars=budget)
        assert len(rendered) <= budget
        if rendered:
            rows = prompt_records(rendered)
            assert isinstance(rows, list)
            assert any('history_matches' in row or row.get('history_matches_omitted') ==
                       {'reason': 'char_budget'} for row in rows)
    assert h.block('never observed source') == ''


def test_small_budget_preserves_lineage_with_explicit_global_witness_omission():
    h = history(); parent, child = program('p1'), program('p2', parent_id='p1')
    h.observe(parent, cases_observation())
    h.observe(child, cases_observation(), parent=parent)
    assert match_card(h, child.code)['kind'] == 'exact_case_vector_and_score'
    rendered = h.block(child.code, max_chars=2000)
    assert 0 < len(rendered) <= 2000
    rows = prompt_records(rendered)
    assert rows[0]['id'] == 'program:p2'
    assert rows[0]['status'] == 'score_tie_unpaired'
    assert rows[0]['score_comparison'] == 'tie_within_abs_tolerance'
    assert rows[0]['history_matches_omitted'] == {'reason': 'char_budget'}
    assert 'history_matches' not in rows[0]


def test_actual_v3_p29_p33_cross_island_zero_geometry_reaches_generator_block():
    fixture = json.loads((ROOT / 'tests/fixtures/observed_global_matches_rl_full_v3.json').read_text())
    programs = {p['id']: p for p in fixture['programs']}
    assert programs['p29']['island'] == 8 and programs['p33']['island'] == 14
    assert programs['p29']['parent_id'] != programs['p33']['parent_id']
    h = ObservedChanges(fixture['metric'])
    for record in fixture['records']:
        # Exact old records and hashes, without invented generic case fields.
        assert 'case_vector' not in record['child_observation']
        h._insert(_validate_record(record))
    for ident, other in (('p29', 'p33'), ('p33', 'p29')):
        rendered = h.block(programs[ident]['code'])
        assert len(rendered) <= 4000
        card = match_card(h, programs[ident]['code'])
        assert card['kind'] == 'exact_validated_geometry'
        assert card['query']['program_id'] == ident
        assert [m['program_id'] for m in card['matches']] == [other]
        assert card['score'] == 0. and card['validity'] is True
        assert card['case_vector'] is None
    restored = ObservedChanges(fixture['metric']); restored.load_state(h.state())
    assert restored.block(programs['p33']['code']) == h.block(programs['p33']['code'])
