"""Pure evidence joins and projections; no model, tool or evaluator is called."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace
import pytest
from sciel.engine.observed_changes import ObservedChanges
from sciel.engine.research_cells_context import compact_context
from sciel.hypothesis.measurement import metric_context
from sciel.methods.models import canonical_bytes, content_hash
from sciel.research_tools.followups import (
    FollowupLedger, build_followup, capture_exposures, compact_view,
    validate_exposures, validate_followup,
)
from test_research_cell_context import card
from method_followup_fixtures import (
    sha, seal, record, ledger,
    observed, output_record,
)


def test_exact_postbudget_order_receipt_identity_and_detached_hashes():
    a, b = card('a'), card('d')
    b['source']['receipt_id'] = '9' * 64
    context = compact_context([b, a])
    context['omitted_count'] = 2
    context['omission_reasons'] = {'character_limit': 2}
    original = deepcopy(context)
    result = capture_exposures(context, prompt_sha256=sha('ACTUAL final prompt'))
    assert result['prompt_sha256'] == sha('ACTUAL final prompt')
    assert [r['card_sha256'] for r in result['cards']] == [content_hash(b), content_hash(a)]
    assert [r['source']['receipt_id'] for r in result['cards']] == ['9' * 64, 'e' * 64]
    assert result['cards'][0]['observations_sha256'] == content_hash(b['observations'])
    assert result['omitted_count'] == 2 and context == original
    context['cards'][0]['source']['receipt_id'] = '1' * 64
    assert result['cards'][0]['source']['receipt_id'] == '9' * 64
    assert 'output_excerpt' not in json.dumps(result)


def test_no_context_empty_and_omitted_only_context_do_not_invent_exposure():
    assert capture_exposures(None, prompt_sha256=sha('prompt')) is None
    item = record()
    del item['method_exposures']
    assert build_followup(item) is None
    item['method_exposures'] = capture_exposures({'schema': 'sciel-research-cells-context/1',
        'cards': [], 'omitted_count': 1, 'omission_reasons': {'research_context_limit': 1}},
        prompt_sha256=sha('prompt'))
    assert build_followup(item) is None
    assert ledger().add(None) is False


def test_unlinked_error_card_is_preserved_but_not_indexed_as_a_method():
    unlinked = card()
    for name in ('tool_id', 'invocation_id', 'receipt_id'):
        unlinked['source'][name] = None
    item = record()
    item['method_exposures'] = capture_exposures(compact_context([unlinked]), prompt_sha256=sha('p'))
    assert len(item['method_exposures']['cards']) == 1
    assert build_followup(item) is None


@pytest.mark.parametrize('mutation', [
    lambda x: x['cards'][0]['source'].update(receipt_id='x' * 64),
    lambda x: x['cards'][0]['source'].update(candidate_id=None),
    lambda x: x['cards'][0]['source'].update(tool_id=None),
    lambda x: x.update(omitted_count=True),
    lambda x: x.update(omission_reasons={'omitted': 1}),
])
def test_resealed_invalid_exposure_shapes_fail(mutation):
    value = record()['method_exposures']
    mutation(value)
    with pytest.raises(ValueError):
        validate_exposures(seal(value))


def test_capture_rejects_invalid_final_card_instead_of_describing_different_prompt():
    context = compact_context([card()])
    context['cards'][0]['source']['target_kind'] = 'fake'
    with pytest.raises(ValueError):
        capture_exposures(context, prompt_sha256=sha('p'))


def test_valid_zero_is_retained_without_tool_benefit_or_physical_reconstruction():
    evidence = record(score=0.0)
    row = build_followup(evidence, child_program_id='p2', observed_record=observed(),
                         evidence_artifact_id='8' * 64)
    assert row['outcome']['score'] == 0.0
    assert row['outcome']['evaluation'] == evidence['evaluation']
    assert row['outcome']['metric'] == evidence['metric']
    assert row['evidence_sha256'] == content_hash(evidence)
    assert row['evidence_artifact_id'] == '8' * 64
    assert row['proposal']['claim_status'] == 'unverified_model_statement'
    assert row['model_use'] == row['causal_benefit'] == 'unknown'
    assert row['outcome']['actual_delta'] == -1.0  # unpaired recorded fact, not method reward
    current = ledger()
    current.add(row)
    view = current.planner_view(['c' * 64], max_chars=8000)
    assert view['records'][0]['canonical_evaluation']['score'] == 0.0
    assert 'native_score' not in json.dumps(view)


@pytest.mark.parametrize('status', ['eval_failed', 'no_child', 'hypothesis_invalid'])
def test_failed_or_rejected_generation_has_no_admitted_child_and_no_quality_score(status):
    evidence = record(status=status, score=-99.0)
    row = build_followup(evidence)
    assert row['outcome']['child_program_id'] is None
    assert row['outcome']['score'] is None and row['outcome']['actual_delta'] is None
    if status == 'eval_failed':
        assert row['outcome']['evaluation']['score'] == -99.0
        assert row['outcome']['evaluation']['ok'] is False
    else:
        assert row['outcome']['evaluation'] is None


@pytest.mark.parametrize('dispatch', ['returned', 'unknown', 'not_dispatched'])
def test_generation_dispatch_states_remain_distinct(dispatch):
    row = build_followup(record(status='generation_stopped', dispatch=dispatch))
    assert row['generation']['dispatch'] == dispatch
    assert (row['generation']['response_sha256'] is not None) == (dispatch == 'returned')
    assert row['outcome']['child_program_id'] is None


def test_legacy_returned_response_without_saved_hash_stays_unavailable():
    evidence = record()
    del evidence['generation_response_sha256']
    assert build_followup(evidence)['generation']['response_sha256'] is None


def test_unknown_dispatch_cannot_be_relabelled_as_returned_child():
    evidence = record(dispatch='unknown')
    with pytest.raises(ValueError, match='canonical evaluation'):
        build_followup(evidence)


@pytest.mark.parametrize('kwargs', [
    {'child_program_id': 'p2'},
    {'child_program_id': 'p3', 'observed_record': observed()},
    {'child_program_id': 'p2', 'observed_record': observed(source='another source')},
    {'child_program_id': 'p2', 'observed_record': observed(score=9)},
    {'observed_record': observed()},
])
def test_admission_requires_matching_saved_program_source_and_measurement(kwargs):
    with pytest.raises(ValueError):
        build_followup(record(), **kwargs)


def test_remeasurement_is_a_separate_unmeasured_generation_link():
    row = build_followup(record(status='no_child'), link_kind='remeasurement')
    assert row['outcome']['link_kind'] == 'remeasurement'
    assert row['outcome']['score'] is None and row['outcome']['child_program_id'] is None


@pytest.mark.parametrize('delta', [999, None, True])
def test_measured_delta_must_match_existing_binary_subtraction_convention(delta):
    evidence = record(score=0.0)
    evidence['actual_delta'] = delta
    with pytest.raises(ValueError, match='delta'):
        build_followup(evidence)


def test_tiny_and_extreme_finite_differences_keep_raw_numeric_values():
    for parent, score in ((0.1, 0.3), (1.0, 1.0000000000000002), (-1e308, -1e307)):
        evidence = record(score=score)
        evidence['parent_score'] = parent
        evidence['actual_delta'] = score - parent
        assert build_followup(evidence)['outcome']['actual_delta'] == score - parent


def test_no_child_cannot_gain_a_mutation_program_even_with_matching_source():
    with pytest.raises(ValueError, match='mutation child'):
        build_followup(record(status='no_child'), child_program_id='p2', observed_record=observed())
    row = build_followup(record(status='no_child'), child_program_id='p2',
                         observed_record=observed(), link_kind='remeasurement')
    assert row['outcome']['link_kind'] == 'remeasurement' and row['outcome']['score'] is None


@pytest.mark.parametrize('changed', ['cases', 'metric'])
def test_observed_case_vector_must_bind_same_cases_and_metric(changed):
    metric = metric_context(SimpleNamespace(qualified_name='synthetic/ratio-free',
        task_hash='a' * 64, primary_metric='quality'))
    evidence = record(score=0.0)
    evidence['metric'] = metric
    other = metric_context(SimpleNamespace(qualified_name='synthetic/ratio-free',
        task_hash='b' * 64, primary_metric='quality')) if changed == 'metric' else metric
    actual = ObservedChanges(other).observe(SimpleNamespace(id='p2', code='child', parent_id=None),
        dict(ok=True, score=0.0, cases={'score': 99.0 if changed == 'cases' else 0.0},
             reached='full', confirmed=False))
    with pytest.raises(ValueError, match='cases/metric'):
        build_followup(evidence, child_program_id='p2', observed_record=actual)


def test_observed_geometry_cannot_appear_without_canonical_geometry_evidence():
    actual = observed()
    actual['child_observation'].update(geometry_status='validated', validity=True,
        status='valid', reason='observed', geometry={'numeric_sha256': 'a' * 64,
        'evaluator_sha256': 'b' * 64, 'centers_dtype': 'float64', 'radii_dtype': 'float64'})
    actual['record_sha256'] = content_hash({k: v for k, v in actual.items() if k != 'record_sha256'})
    with pytest.raises(ValueError, match='canonical geometry'):
        build_followup(record(), child_program_id='p2', observed_record=actual)


def test_repaired_proposal_keeps_both_source_identities():
    evidence = record()
    evidence.update(proposed_sha256=sha('before syntax repair'), source_repaired=True)
    row = build_followup(evidence)
    assert row['proposal']['source_sha256'] == sha('child')
    assert row['proposal']['proposed_sha256'] == sha('before syntax repair')


def test_only_final_delivered_card_not_every_episode_receipt_is_a_followup():
    evidence = record()
    unrelated_first_attempt = '9' * 64
    row = build_followup(evidence)
    seen = []
    current = ledger(exposure=lambda item: seen.append(item['source']['receipt_id']))
    current.add(row)
    assert seen == ['e' * 64] and unrelated_first_attempt not in seen


def test_idempotency_conflict_and_caller_detachment():
    evidence = record()
    row = build_followup(evidence)
    current = ledger()
    assert current.add(row) is True
    assert current.add(deepcopy(row)) is False
    saved = current.state()
    evidence['failure'] = 'Changed history'
    with pytest.raises(ValueError, match='conflicting'):
        current.add(build_followup(evidence))
    row['outcome']['failure'] = 'Caller mutation'
    assert current.state() == saved
    saved['records'].clear()
    assert len(current.state()['records']) == 1


def test_callbacks_authenticate_resealed_artifact_and_outcome_tampering():
    expected = record()
    def authenticate_exposure(item):
        assert item['source']['receipt_id'] == 'e' * 64
    def authenticate_outcome(item):
        if item['evidence_sha256'] != content_hash(expected):
            raise ValueError('canonical evidence differs')
    current = ledger(exposure=authenticate_exposure, outcome=authenticate_outcome)
    current.add(build_followup(expected))
    bad = record()
    bad['method_exposures']['cards'][0]['source']['receipt_id'] = '9' * 64
    bad['method_exposures'] = seal(bad['method_exposures'])
    with pytest.raises(AssertionError):
        current.add(build_followup(bad))
    bad = record(score=1000)
    with pytest.raises(ValueError, match='canonical evidence'):
        current.add(build_followup(bad))


def test_restore_reauthenticates_atomically_and_rejects_duplicate_experiments():
    current = ledger()
    current.add(build_followup(record()))
    saved = current.state()
    checked = []
    restored = ledger(exposure=lambda item: checked.append(item['card_sha256']))
    restored.load_state(saved)
    assert restored.state() == saved and len(checked) == 1
    broken = deepcopy(saved)
    broken['records'].append(build_followup(record(2)))
    broken['records'][-1]['outcome']['score'] = 3
    broken['records'][-1] = seal(broken['records'][-1])
    with pytest.raises(ValueError):
        restored.load_state(seal(broken))
    assert restored.state() == saved
    broken = deepcopy(saved)
    broken['records'] *= 2
    with pytest.raises(ValueError, match='duplicate experiment'):
        restored.load_state(seal(broken))
    broken = deepcopy(saved)
    broken['identity_sha256'] = '9' * 64
    with pytest.raises(ValueError, match='identity'):
        restored.load_state(seal(broken))
    assert restored.state() == saved


def test_visible_methods_latest_bounded_whole_rows_and_frozen_views():
    current = ledger()
    for number, tool in ((1, 'c'), (3, 'c'), (2, 'c'), (4, 'd'), (5, 'e')):
        current.add(build_followup(record(number, tool=tool)))
    view = current.planner_view(['c' * 64, 'd' * 64], max_chars=16000, per_method=2)
    assert [r['experiment_id'] for r in view['records']] == ['r000003a1', 'r000004a1', 'r000002a1']
    assert view['eligible_count'] == 4 and view['omitted_count'] == 1
    assert not any(r['tool_id'] == 'e' * 64 for r in view['records'])
    before = deepcopy(view)
    current.add(build_followup(record(6, tool='c')))
    assert view == before
    for budget in (256, 800, 2000, 4096, 8000):
        small = compact_view(view, max_chars=budget)
        assert len(canonical_bytes(small)) <= budget
        assert json.loads(canonical_bytes(small)) == small
        assert len(small['records']) + small['omitted_count'] == 4
        assert small['records'] == view['records'][:len(small['records'])]
    assert compact_view(view, max_chars=256)['records'] == []
    with pytest.raises(ValueError, match='omission metadata'):
        compact_view(view, max_chars=10)


def test_large_unicode_claim_cases_failure_remain_bounded_projection_with_full_evidence_retained():
    evidence = record(status='eval_failed')
    evidence['hypothesis'] = {'statement': '测量' * 5000}
    evidence['failure'] = 'Failure context ' * 5000
    evidence['evaluation']['cases'] = {f'case{i}': i for i in range(2000)}
    row = build_followup(evidence)
    current = ledger()
    current.add(row)
    view = current.planner_view(['c' * 64], max_chars=4096)
    assert len(canonical_bytes(view)) <= 4096
    projected, = view['records']
    assert projected['canonical_evaluation']['case_count'] == 2000
    assert projected['proposed_claim']['omitted_chars'] > 0
    assert projected['failure']['omitted_chars'] > 0
    assert current.state()['records'][0]['outcome']['evaluation'] == evidence['evaluation']
    assert current.state()['records'][0]['proposal']['claim'] == evidence['hypothesis']


@pytest.mark.parametrize('field,value', [('model_use', 'yes'), ('causal_benefit', 'positive')])
def test_resealed_usage_or_benefit_claim_is_forbidden(field, value):
    row = build_followup(record())
    row[field] = value
    with pytest.raises(ValueError, match='cannot assert'):
        validate_followup(seal(row))


def test_output_repeat_preserves_positive_measurement_without_efficacy_or_admission(tmp_path):
    evidence = output_record(tmp_path)
    before = deepcopy(evidence)
    row = build_followup(evidence)
    assert evidence == before
    assert row['outcome']['actual_delta'] == 1.0
    assert row['outcome']['evaluation'] == evidence['evaluation']
    assert row['outcome']['output_novelty'] == evidence['output_novelty']
    assert row['outcome']['child_program_id'] is None
    current = ledger()
    current.add(row)
    projected, = current.planner_view(['c' * 64], max_chars=4096)['records']
    assert projected['canonical_evaluation']['score'] == 2.0
    assert projected['efficacy_eligible'] is False
    view = projected['output_admission']
    assert view['status'] == 'exact_recorded_repeat' and view['accept'] is False
    assert view['evaluation_id'] == 'child-evaluation'
    assert view['verdict_sha256'] == evidence['output_novelty']['sha256']
    assert view['match'] == {'evaluation_id': 'historical-evaluation',
        'source_sha256': sha('historical'), 'island_id': 0, 'program_id': 'historical-program'}
    assert 'not efficacy evidence or program equivalence' in projected['output_admission_scope']
    assert projected['model_use'] == projected['causal_benefit'] == 'unknown'
    assert 'output_novelty' not in projected and 'identity' not in view
    evidence['output_novelty']['match']['program_id'] = 'caller-mutation'
    assert current.state()['records'][0] == row


def test_exact_repeat_cannot_gain_mutation_child_but_explicit_remeasurement_stays_separate(tmp_path):
    evidence = output_record(tmp_path)
    with pytest.raises(ValueError, match='repeat cannot acquire a mutation child'):
        build_followup(evidence, child_program_id='p2', observed_record=observed(score=2.0))
    row = build_followup(evidence, child_program_id='p2', link_kind='remeasurement',
                         observed_record=observed(score=2.0))
    assert row['outcome']['link_kind'] == 'remeasurement'
    current = ledger()
    current.add(row)
    assert current.planner_view(['c' * 64], max_chars=8000)['records'][0]['efficacy_eligible'] is False


@pytest.mark.parametrize('disposition,status,eligible', [
    ('new_output', 'evaluated', True), ('unsupported', 'evaluated', True),
    ('unsupported', 'eval_failed', False),
])
def test_nonrepeat_dispositions_do_not_invent_repeat_or_method_benefit(tmp_path, disposition, status, eligible):
    evidence = output_record(tmp_path, disposition=disposition, status=status)
    row = build_followup(evidence, **({'child_program_id': 'p2', 'observed_record': observed(score=2.0)}
                                     if status == 'evaluated' else {}))
    current = ledger()
    current.add(row)
    projected, = current.planner_view(['c' * 64], max_chars=8000)['records']
    assert projected['output_admission']['status'] == disposition
    assert projected['output_admission']['match'] is None
    assert projected['efficacy_eligible'] is eligible
    assert 'output_admission_scope' not in projected
    assert projected['causal_benefit'] == 'unknown'


@pytest.mark.parametrize('damage', ['seal', 'source', 'island', 'stage', 'task', 'missing_evaluation',
                                   'compact_substitute', 'mutation_child'])
def test_output_verdict_restore_is_bound_and_atomic(tmp_path, damage):
    row = build_followup(output_record(tmp_path))
    current = ledger()
    current.add(row)
    saved = current.state()
    restored = ledger()
    restored.load_state(saved)
    assert restored.state() == saved
    broken = deepcopy(saved)
    target = broken['records'][0]
    if damage == 'seal': target['outcome']['output_novelty']['sha256'] = '9' * 64
    elif damage == 'source': target['proposal']['source_sha256'] = '9' * 64
    elif damage == 'island': target['island_id'] += 1
    elif damage == 'stage': target['outcome']['evaluation']['reached'] = 'smoke'
    elif damage == 'task':
        target['outcome']['metric'] = metric_context(SimpleNamespace(task_hash='9' * 64))
    elif damage == 'missing_evaluation':
        target['outcome'].update(status='no_child', score=None, actual_delta=None, evaluation=None)
    elif damage == 'compact_substitute':
        projected, = current.planner_view(['c' * 64], max_chars=8000)['records']
        target['outcome']['output_novelty'] = projected['output_admission']
    elif damage == 'mutation_child':
        target['outcome'].update(child_program_id='p2', observed_record=observed(score=2.0))
    broken['records'][0] = seal(target)
    with pytest.raises(ValueError):
        restored.load_state(seal(broken))
    assert restored.state() == saved


def test_legacy_followup_shape_and_compact_repeat_omissions_are_preserved(tmp_path):
    legacy = build_followup(record())
    assert 'output_novelty' not in legacy['outcome']
    current = ledger()
    current.add(legacy)
    projected, = current.planner_view(['c' * 64], max_chars=8000)['records']
    assert not {'output_admission', 'efficacy_eligible', 'output_admission_scope'} & projected.keys()
    restored = ledger()
    restored.load_state(current.state())
    assert canonical_bytes(restored.state()) == canonical_bytes(current.state())
    evidence = output_record(tmp_path)
    evidence['id'], evidence['iteration'] = 'r000002a1', 2
    current.add(build_followup(evidence))
    view = current.planner_view(['c' * 64], max_chars=16000)
    before = deepcopy(view)
    for budget in (256, 2048, 4096, 8000):
        compact = compact_view(view, max_chars=budget)
        assert len(canonical_bytes(compact)) <= budget
        assert len(compact['records']) + compact['omitted_count'] == 2
        for entry in compact['records']:
            if entry['experiment_id'] == 'r000002a1':
                assert entry['efficacy_eligible'] is False
                assert entry['output_admission']['verdict_sha256'] == evidence['output_novelty']['sha256']
                assert 'not efficacy evidence' in entry['output_admission_scope']
    assert view == before
