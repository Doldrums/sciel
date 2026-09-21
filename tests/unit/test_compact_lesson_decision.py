"""Pure prompt projections from retained evidence; no model/evaluator execution."""
from copy import deepcopy
import hashlib
import json
import random

import pytest

from sciel.hypothesis.rendering import (presentation_identity, serialize_card,
                                       serialize_decision_card)
from sciel.knowledge_base import KnowledgeBase
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
from sciel.knowledge_base.decision import (DECISION_CARD_SCHEMA, DECISION_RULES,
    decision_card, decision_evidence_manifest, resolve_decision_evidence)
from sciel.knowledge_base.guarded import guarded_admission
from sciel.knowledge_base.presentation import observation_card
from sciel.knowledge_base.retrieval import selection_identity
from test_research_knowledge import experiment, lesson


def render(value):
    return DECISION_RULES + 'RESEARCH_LESSON_JSON:\n' + serialize_decision_card(value)


def fixture():
    kb = KnowledgeBase(max_experiments=500)
    kb.record(experiment(1, branch='p1', change_excerpt='FULL_DIFF_MARKER\n' * 350))
    ident = lesson(kb, 'r000001a1', rule='Use smaller steps only on this branch; preserve the boundary constraint.')
    for number, delta, status in [(2, -.1, 'evaluated'), (3, 0., 'evaluated'),
                                   (4, None, 'eval_failed'), (5, .125, 'evaluated')]:
        kb.record(experiment(number, branch='p1', delta=delta, status=status, injected=[ident],
            change_excerpt='FULL_DIFF_MARKER\n' * 350, iteration=number))
    return kb


def test_full_rule_exact_measurements_and_counterevidence_survive_without_recursive_diffs():
    kb = fixture()
    original, rng = kb.state(), random.getstate()
    retained = kb.lessons[0]
    card = decision_card(retained)
    text = render(retained)
    assert card['schema'] == DECISION_CARD_SCHEMA
    assert card['unverified_interpretation'] == retained['rule']
    assert card['scope'] == 'branch' and card['branch'] == 'p1'
    assert 'FULL_DIFF_MARKER' not in text
    assert 'change_excerpt' in card['omissions']['fields']
    counts = card['advice_exposures']['retained_unique_counts']
    assert counts == dict(positive=1, negative=1, zero=1, unmeasured=1, mixed=0, incomplete=0, total=4)
    shown = {row['observation']['id']: row['observation']
             for row in card['advice_exposures']['representatives']}
    assert shown['r000002a1']['measurement']['actual_delta'] == (.5 - .1) - .5
    assert shown['r000003a1']['measurement']['actual_delta'] == 0.
    assert shown['r000004a1']['measurement']['actual_delta'] is None
    assert shown['r000004a1']['measurement']['direction'] == 'unmeasured'
    assert len(text) < 8000 < len(serialize_card(observation_card(retained)))
    assert kb.state() == original and random.getstate() == rng


def test_repeated_exposures_have_bounded_rows_and_recoverable_omissions():
    kb = fixture()
    before = kb.lessons[0]
    size = len(render(before))
    for number in range(6, 206):
        kb.record(experiment(number, branch='p1', delta=.125, injected=[before['id']],
            change_excerpt='FULL_DIFF_MARKER\n' * 350, iteration=number))
    retained = kb.lessons[0]
    state_before = canonical_json(kb.state())
    card = decision_card(retained)
    assert len(card['advice_exposures']['representatives']) == 4
    assert len(render(retained)) < size + 300
    assert card['advice_exposures']['retained_unique_counts']['positive'] == 201
    assert card['advice_exposures']['omitted_count'] == 200
    omissions = card['omissions']['exposure_records']
    assert omissions['count'] == 200 and len(omissions['id_sample']) == 3
    assert omissions['ids_not_listed'] == 197
    recovered = resolve_decision_evidence(card, retained)
    assert len(recovered['records']) == 205
    assert {r['record']['id'] for r in recovered['records']} == {
        r['id'] for r in [*retained['evidence_snapshots'], *retained['exposure_snapshots']]}
    for item in recovered['records']:
        assert canonical_sha(item['record']) == item['reference']['sha256']
    assert canonical_json(kb.state()) == state_before
    recovered['records'][0]['record']['change_excerpt'] = 'mutation of returned copy'
    assert canonical_json(kb.state()) == state_before


def test_manifest_requires_exact_snapshot_not_latest_lesson_with_same_id():
    retained = fixture().lessons[0]
    card = decision_card(retained)
    changed = deepcopy(retained)
    changed['exposure_snapshots'][0]['change_excerpt'] += 'new unseen evidence'
    assert len(render(changed)) == len(render(retained))
    with pytest.raises(ValueError, match='does not match'):
        resolve_decision_evidence(card, changed)
    assert canonical_sha(decision_evidence_manifest(retained)) == card['evidence_ref']['manifest_sha256']


def test_inherited_wording_is_referenced_not_recursively_quoted_or_credited_to_new_rule():
    retained = fixture().lessons[0]
    exposure = retained['exposure_snapshots'][0]
    retained['inherited_exposures'] = [{'experiment': deepcopy(exposure),
        'from_lesson_id': 'l000000', 'from_rule': 'OLD_WORDING_MARKER ' * 100}]
    card = decision_card(retained)
    rows = card['advice_exposures']['representatives']
    inherited = next(r for r in rows if r['observation']['id'] == exposure['id'])
    assert inherited['attribution'] == {'direct': True, 'inherited': True}
    assert card['advice_exposures']['retained_unique_counts']['total'] == 4
    assert card['advice_exposures']['direct_unique_counts']['total'] == 4
    assert card['advice_exposures']['inherited_unique_counts']['total'] == 1
    assert 'OLD_WORDING_MARKER' not in render(retained)
    manifest = decision_evidence_manifest(retained)
    origin = next(e for e in manifest['entries'] if e['role'] == 'inherited')
    assert origin['from_lesson_id'] == 'l000000'
    assert origin['from_rule_sha256'] == hashlib.sha256(
        retained['inherited_exposures'][0]['from_rule'].encode()).hexdigest()


def test_global_applicability_preserves_two_source_branches_and_exact_omission_counts():
    kb = KnowledgeBase()
    for number in range(1, 21):
        kb.record(experiment(number, branch='p1' if number == 1 else 'p2', iteration=number))
    sources = [r['id'] for r in kb.experiments]
    lesson(kb, *sources, scope='global')
    retained = kb.lessons[0]
    card = decision_card(retained)
    assert {r['branch'] for r in card['source_observations']} == {'p1', 'p2'}
    assert card['source_counts']['positive'] == 20
    assert card['source_omitted_count'] == 18
    assert set(card['source_ids']) == {'r000001a1', 'r000020a1'}
    assert len(resolve_decision_evidence(card, retained)['records']) == 20


def test_primary_validity_measurements_not_a_physical_improvement_inference():
    retained = fixture().lessons[0]
    record = retained['evidence_snapshots'][0]
    record['metric'] = dict(primary_metric='combined_score', semantics=dict(
        kind='validity_gated_ratio', numerator_case='sum_radii', denominator=2.635,
        validity_case='validity', invalid_numerator_semantics='penalty_sentinel_not_physical_measurement'))
    record.update(parent_score=.5, score=0., actual_delta=-.5)
    record['evaluation'] = dict(ok=True, score=0., reached='full', confirmed=False,
        cases=dict(combined_score=0., sum_radii=0., validity=0., arbitrary_large_payload='OPAQUE_MARKER'*2000))
    row = decision_card(retained)['source_observations'][0]
    assert row['evaluation']['primary_cases'] == {'combined_score': 0., 'sum_radii': 0., 'validity': 0.}
    assert row['evaluation']['confirmed'] is False
    assert 'OPAQUE_MARKER' not in render(retained)
    metric = decision_card(retained)['metric_catalog'][row['metric_ref']]
    assert metric['semantics']['invalid_numerator_semantics'] == 'penalty_sentinel_not_physical_measurement'


def test_large_opaque_measurement_field_is_omitted_whole_with_hash_not_sliced():
    retained = fixture().lessons[0]
    record = retained['evidence_snapshots'][0]
    record['metric'] = {'primary_metric': 'score', 'opaque': 'UNKNOWN_TEXT_MARKER' * 300}
    card = decision_card(retained)
    omitted = card['metric_catalog']['m1']
    assert omitted['omitted'] is True and omitted['reason'] == 'decision_field_char_limit'
    assert omitted['sha256'] == canonical_sha(record['metric'])
    assert omitted['chars'] == len(canonical_json(record['metric']))
    assert 'UNKNOWN_TEXT_MARKER' not in serialize_decision_card(card)
    assert resolve_decision_evidence(card, retained)['records'][0]['record']['metric'] == record['metric']


def test_paired_rows_keep_exact_three_pair_measurements_separate_from_controller_score():
    from test_knowledge_paired_comparisons import group
    kb = KnowledgeBase()
    comparison = group()
    kb.record_comparison(comparison)
    lesson(kb, comparison['id'])
    card = decision_card(kb.lessons[0])
    row, = card['source_observations']
    assert row['kind'] == 'paired_comparison'
    assert row['derived'] == comparison['derived']
    assert len(row['derived']['pairs']) == 3
    assert row['comparator_source_sha256'] == comparison['comparator_source_sha256']
    assert 'measurement' not in row and 'score' not in row
    assert card['metric_catalog'][row['metric_ref']] == comparison['metric']


def test_guard_uses_shown_compact_source_facts_and_retains_adverse_card_anchor():
    kb = fixture()
    old = kb.lessons[0]
    for number in range(6, 9):
        kb.record(experiment(number, branch='p1'))
    lesson(kb, 'r000006a1', 'r000007a1', 'r000008a1', rule='Use smaller steps alternate proposal')
    candidates = kb.lessons
    result = guarded_admission(candidates, 'smaller steps', 2, render, compact_decision=True,
                               max_memory_chars=16000)
    assert old['id'] in [l['id'] for l in result['lessons']]
    assert len(result['lessons']) == 2
    metadata = result['selection']
    assert metadata['decision_presentation'] == DECISION_CARD_SCHEMA
    assert metadata['source_fact_scope'] == 'shown_decision_source_rows_only'
    # Three retained positive source records in second card are one shown row.
    assert metadata['final']['source_fact_count'] == 2
    assert metadata['protected_cards'][0]['id'] == old['id']
    assert metadata['protected_cards'][0]['any_retained_advice_exposure'] is True
    blocked = guarded_admission(candidates, 'smaller steps', 2, render,
                                compact_decision=True, max_card_chars=1)
    assert blocked['lessons'] == []
    assert {r['reason'] for r in blocked['omitted']} == {'card_char_limit'}


def test_opt_in_identity_and_disabled_legacy_bytes_are_unchanged():
    retained = fixture().lessons[0]
    before = serialize_card(observation_card(retained))
    baseline = presentation_identity()
    assert baseline == presentation_identity(compact_lesson_memory=False)
    enabled = presentation_identity(compact_lesson_memory=True)
    assert set(enabled) - set(baseline) == {'compact_lesson_memory'}
    assert enabled['compact_lesson_memory']['schema'] == DECISION_CARD_SCHEMA
    assert len(enabled['compact_lesson_memory']['source_sha256']) == 64
    config = {'lesson_selection': {'mode': 'source_diverse_guarded'}, 'compact_lesson_memory': True}
    assert selection_identity(config)['rendering']['compact_lesson_memory'] == enabled['compact_lesson_memory']
    assert serialize_decision_card(retained) == serialize_decision_card(decision_card(retained))
    assert serialize_card(observation_card(retained)) == before
    with pytest.raises(ValueError, match='must be boolean'):
        presentation_identity(compact_lesson_memory=1)
    with pytest.raises(ValueError, match='must be boolean'):
        guarded_admission([], '', 1, render, compact_decision=1)


def test_full_rule_is_not_truncated_to_force_admission_under_tiny_budget():
    retained = fixture().lessons[0]
    retained['rule'] = 'Full applicability constraint. ' * 125
    assert decision_card(retained)['unverified_interpretation'] == retained['rule']
    result = guarded_admission([retained], 'constraint', 1, render,
        max_memory_chars=100, compact_decision=True)
    assert result['lessons'] == [] and result['omitted'][0]['reason'] == 'memory_char_limit'
