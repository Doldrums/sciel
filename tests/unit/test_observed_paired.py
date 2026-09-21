"""Saved synthetic numbers only; no candidate, provider or evaluator calls."""
import copy
import hashlib
import json
from types import SimpleNamespace as NS

import pytest

from sciel.engine.observed_paired import paired_record, validate_record
from sciel.knowledge_base.paired_comparisons import build_comparison, UNKNOWN_RECEIPT
from test_paired_comparisons import inputs, receipt, SOURCE_A, SOURCE_B


def same_inputs():
    return inputs(old=(.1, .1, .1), new=(.1, .1, .1))


def sealed(value):
    value = copy.deepcopy(value)
    value.pop('record_sha256', None)
    raw = json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)
    value['record_sha256'] = hashlib.sha256(raw.encode('ascii')).hexdigest()
    return value


def test_same_geometry_is_saved_array_fact_with_explicit_comparator_and_dtype_scope():
    group = build_comparison(**same_inputs())
    before = copy.deepcopy(group)
    value = paired_record(group)
    assert group == before
    assert value['status'] == 'same_geometry_on_paired_seeds'
    assert value['parent_role'] == 'comparator' and value['scope'] == 'paired'
    assert value['original_array_dtypes'] == 'not_recorded'
    assert value['geometry_representation'] == 'validated_capture_binary64_json'
    assert all(row['scores_equal'] is True and row['geometry_equal'] is True for row in value['pairs'])
    assert all(row['parent_raw_sum'] == .1 for row in value['pairs'])
    assert not value['edited_units_available'] and value['edited_units'] == []
    assert validate_record(json.loads(json.dumps(value))) == value
    assert len(json.dumps(value)) < 5000


def test_equal_scores_do_not_imply_equal_geometry():
    data = same_inputs()
    for pair in data['pairs']:
        pair['challenger'] = receipt(SOURCE_B, pair['seed'], .1, x=.55)
    value = paired_record(build_comparison(**data))
    assert value['status'] == 'same_scores_on_paired_seeds'
    assert all(row['scores_equal'] is True and row['geometry_equal'] is False for row in value['pairs'])


def test_different_valid_results_keep_both_metrics_without_quality_or_speed_label():
    value = paired_record(build_comparison(**inputs(new=(.11, .09, .1))))
    assert value['status'] == 'different_geometry_on_paired_seeds'
    assert [row['geometry_equal'] for row in value['pairs']] == [False, False, True]
    assert [row['child_raw_sum'] for row in value['pairs']] == [.11, .09, .1]
    assert 'reward' not in value and 'admission' not in value and 'speedup' not in value


@pytest.mark.parametrize('missing', [None, UNKNOWN_RECEIPT, 'candidate_failure', 'infrastructure'])
def test_partial_and_failed_draws_are_incomplete_not_zero_or_tie(missing):
    data = same_inputs()
    data['pairs'][1]['challenger'] = (receipt(SOURCE_B, 12, status=missing)
        if isinstance(missing, str) else copy.deepcopy(missing))
    value = paired_record(build_comparison(**data))
    assert value['status'] == 'incomplete'
    row = value['pairs'][1]
    assert row['parent_raw_sum'] == .1
    assert row['child_raw_sum'] is row['child_certified_sum'] is row['child_geometry_sha256'] is None
    assert row['scores_equal'] is row['geometry_equal'] is None
    assert value['pairs'][0]['geometry_equal'] is True


@pytest.mark.parametrize('field,value', [('source_sha256', 'c'*64), ('seed', 19),
    ('adapter_identity', {'fixture': 'other runtime'}), ('limits', {})])
def test_unmatched_source_seed_runtime_or_limits_rejected(field, value):
    group = build_comparison(**same_inputs())
    group['pairs'][1]['challenger'][field] = value
    with pytest.raises(ValueError):
        paired_record(group)


def test_capture_and_derived_claims_are_recomputed_not_trusted():
    group = build_comparison(**same_inputs())
    group['pairs'][0]['challenger']['raw_sum'] = .2
    with pytest.raises(ValueError, match='metrics differ'):
        paired_record(group)
    group = build_comparison(**same_inputs())
    group['derived']['outcome'] = 'improved'
    with pytest.raises(ValueError, match='claims differ'):
        paired_record(group)


def test_record_hash_and_shape_reject_tampering_and_unsafe_prose():
    value = paired_record(build_comparison(**same_inputs()))
    changed = copy.deepcopy(value)
    changed['parent_source_sha256'] = 'c'*64
    with pytest.raises(ValueError, match='digest differs'):
        validate_record(changed)
    changed = sealed({**value, 'status': 'different_geometry_on_paired_seeds'})
    with pytest.raises(ValueError, match='derived status'):
        validate_record(changed)
    with pytest.raises(ValueError, match='record fields'):
        validate_record(sealed({**value, 'instruction': 'Untrusted prose must not become prompt rules.'}))
    changed = copy.deepcopy(value)
    changed['pairs'][0]['scores_equal'] = 1
    with pytest.raises(ValueError, match='pair comparisons'):
        validate_record(sealed(changed))


def test_identity_binds_limits_and_group_but_not_invented_generative_parent():
    first = build_comparison(**same_inputs())
    data = same_inputs()
    data['limits']['candidate_wall_seconds'] = 40.
    for pair in data['pairs']:
        for role in ('comparator', 'challenger'):
            pair[role]['limits'] = copy.deepcopy(data['limits'])
    second = build_comparison(**data)
    a, b = paired_record(first), paired_record(second)
    assert a['id'] == b['id']
    assert a['group_sha256'] != b['group_sha256']
    assert a['comparison_identity_sha256'] != b['comparison_identity_sha256']
    data = same_inputs()
    data['provenance']['original_parent_source_sha256'] = 'f'*64
    value = paired_record(build_comparison(**data))
    assert value['parent_source_sha256'] == SOURCE_A


def test_optional_source_pair_supplies_bounded_direct_unit_names():
    sources = {'comparator': 'def search():\n    return 1\n',
               'challenger': 'def search():\n    return 2\n'}
    data = same_inputs()
    data['source_pair'] = sources
    for role, field in [('comparator', 'comparator_source_sha256'), ('challenger', 'challenger_source_sha256')]:
        source_hash = hashlib.sha256(sources[role].encode()).hexdigest()
        data[field] = source_hash
        for pair in data['pairs']:
            pair[role]['source_sha256'] = source_hash
    value = paired_record(build_comparison(**data))
    assert value['edited_units_available'] is True
    assert value['edited_units'] == ['search'] and value['edited_units_omitted'] == 0
    # Distinct executable AST alone says nothing about measured geometry.
    assert value['status'] == 'same_geometry_on_paired_seeds'


def test_specialized_step_uses_feedback_and_observes_after_add_without_admission_change(monkeypatch):
    from sciel.engine import repeat_population_support as support
    from sciel.engine.types import Program

    parent = Program('p1', 'parent source', .1, {}, 0, 'seed')
    child = Program('p2', 'child source', .1, {}, 0, 'gen', 'p1')
    selected = dict(parent_id='p1', parent_source_sha256=SOURCE_A,
                    comparator_source_sha256=SOURCE_A, comparator_program_id='p1', parent_propensity=1.)
    events, sent = [], []
    programs = {'p1': parent}

    def add(*args, **kwargs):
        programs[child.id] = child
        events.append('add')
        return child

    def observe(program, outcome, *, parent):
        assert programs[program.id] is child and parent.id == 'p1'
        assert events == ['intent', 'worker', 'add']
        events.append('observe')

    def stage(*args, **kwargs):
        events.append('stage')
        return {'eligible': False, 'reason': 'fixture_original_decision'}

    def worker(op, snapshot, *args, **kwargs):
        sent.append(snapshot)
        events.append('worker')
        return NS(ok=True, child_code=child.code, score=.1, cases={}, meta={}, cost=.01, timing=1.)

    engine = NS(repeat_population=NS(select=lambda *a: selected,
            state=lambda: {'projection': {'evidence_heads': {SOURCE_A: {}}}}, inspirations=lambda *a: [], stage=stage),
        db=NS(get=lambda ident: programs[ident], _all=programs, add=add, pop_best=.1),
        rng=NS(stream=lambda name: None, derive_seed=lambda name: 1), cfg={'op_fixed': 'full_rewrite'},
        llm=NS(choose_model=lambda rng: 'fixture'), research=NS(context=lambda *a: {'lesson_texts': []},
            complete_step=lambda **k: events.append('conclusion')),
        observed_changes=NS(block=lambda code: '\nOBSERVED_FIXTURE: current parent only\n'),
        _observe_program=observe, _evaluate=lambda *a: pytest.fail('unexpected evaluation'),
        _per_run_budget=lambda: 30., eval_times={}, iteration_no=0, cost_total=0.,
        task_prompt='fixture', required_entrypoints=[], circle_generation=None,
        log=NS(repeat_search=lambda **kw: events.append('intent'), iteration=lambda **kw: events.append('iteration')),
        B_evals=0, n_records=0, _repeat_research_binding=lambda *a: {}, _model_propensity=lambda model: 1.)
    monkeypatch.setattr(support, 'run_worker', worker)
    monkeypatch.setattr(support, 'save', lambda *a: events.append('checkpoint'))
    monkeypatch.setattr(support, 'complete_research', lambda *a: events.append('reflection'))
    support.step(engine, 0)
    assert 'OBSERVED_FIXTURE: current parent only' in sent[0].directive
    assert events == ['intent', 'worker', 'add', 'observe', 'stage', 'iteration', 'conclusion', 'checkpoint', 'reflection']
    assert engine.B_evals == 1 and engine.n_records == 0 and engine.cost_total == .01
