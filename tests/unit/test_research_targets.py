"""Research inputs preserve measured source identity across repair and replay.

All providers/evaluators are local fixtures; generated programs are never executed.
"""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from sciel.engine.observed_changes import ObservedChanges
from sciel.engine.parallel import _snapshot
from sciel.engine.research_targets import (
    failed_provenance, freeze_targets, input_binding, source_sha256,
    tool_inputs, validate_targets,
)
from sciel.engine.types import Program, Snapshot
from sciel.methods.models import content_hash
from test_parallel_engine import evaluator
from test_research_tool_engine import setup_tools


def measured(score):
    return dict(ok=True, score=score, cases={'quality': score}, artifacts={},
                reached='full', confirmed=False)


@pytest.fixture
def fixture():
    parent = Program('p1', 'def solve(): return 1\n', .7, {'quality': .2}, 0, 'gen')
    best = Program('p2', 'def solve(): return 2\n', 999., {'quality': .8}, 1, 'gen')
    history = ObservedChanges()
    history.observe(parent, measured(.2))
    history.observe(best, measured(.8))
    programs = {p.id: p for p in (parent, best)}
    engine = SimpleNamespace(log=SimpleNamespace(run_id='run-fixture'),
        task=SimpleNamespace(task_hash='task-fixture'), iteration_no=12, B_evals=9,
        db=SimpleNamespace(pop_best_id=best.id, get=programs.__getitem__),
        observed_changes=history)
    snapshot = Snapshot(island=0, op='diff', model='fixture', iteration_no=12,
        parent_id=parent.id, parent_code=parent.code, parent_score=parent.score,
        task_prompt='A generic fixture task.')
    snapshot.research_targets = freeze_targets(engine, parent, parent.code)
    return SimpleNamespace(engine=engine, parent=parent, best=best,
                           history=history, snapshot=snapshot)


def reseal(value):
    value['sha256'] = content_hash({k: v for k, v in value.items() if k != 'sha256'})
    return value


def test_real_engine_error_repair_names_failed_source_without_borrowing_parent_id(tmp_path, monkeypatch):
    build, _, _, _ = setup_tools(tmp_path, monkeypatch, workers=1,
                                 max_iterations=1, num_islands=1)
    evaluator(monkeypatch, fail_first_candidate=True)
    engine = build()
    engine.run()
    assert len(engine.lineage_fail) == 1
    parent_id, failure = next(iter(engine.lineage_fail.items()))
    parent = engine.db.get(parent_id)
    assert failure['_failed_code'] != parent.code
    engine.cfg['op_fixed'] = 'error_repair'
    snapshot = engine._prepare_step(0, spawn=False)['snap']
    inputs = tool_inputs(snapshot, {'task_hash': engine.task.task_hash}, {})
    selected = inputs['research_targets']['selected']
    assert snapshot.parent_id == inputs['ancestry_parent_id'] == parent_id
    assert inputs['parent_source'] == selected['source'] == failure['_failed_code']
    assert inputs['parent_program_id'] is inputs['parent_score'] is None
    assert selected['kind'] == 'failed_candidate' and selected['selection_score'] is None
    assert selected['measurement'] == dict(status='unavailable', record=None, aliases=[])
    assert selected['failure'] == failure['_failure_provenance']
    assert selected['failure']['source_sha256'] == source_sha256(failure['_failed_code'])
    assert selected['failure']['artifacts'] == snapshot.artifacts
    assert inputs['research_targets']['population_best']['program_id'] == parent_id
    assert inputs['research_targets']['population_best']['source'] == parent.code
    binding = input_binding(inputs)
    assert binding['candidate_id'] is None and binding['ancestry_parent_id'] == parent_id
    assert binding['target_source_sha256'] != binding['population_best_source_sha256']


def test_champion_is_full_frozen_source_with_measurement_separate_from_selection_score(fixture):
    inputs = tool_inputs(fixture.snapshot, {'task_hash': 'task-fixture'}, {'probe': 1})
    target, best = (inputs['research_targets'][key] for key in ('selected', 'population_best'))
    assert target['source'] == fixture.parent.code and target['selection_score'] == .7
    assert inputs['parent_score'] == target['measurement']['record']['child_observation']['score'] == .2
    assert best['source'] == fixture.best.code and best['program_id'] == fixture.best.id
    assert best['selection_score'] == 999.
    assert best['measurement']['record']['child_observation']['score'] == .8
    assert best['measurement']['record']['child_source_sha256'] == source_sha256(best['source'])
    assert input_binding(inputs)['population_best_source_sha256'] == source_sha256(fixture.best.code)


def test_unmeasured_source_never_uses_selection_score_as_tool_measurement(fixture):
    fixture.engine.observed_changes = None
    fixture.snapshot.research_targets = freeze_targets(fixture.engine, fixture.parent, fixture.parent.code)
    inputs = tool_inputs(fixture.snapshot, {'task_hash': 'task-fixture'}, {})
    assert inputs['parent_score'] is None
    for key in ('selected', 'population_best'):
        assert inputs['research_targets'][key]['measurement'] == dict(status='unavailable', record=None, aliases=[])


def test_frozen_parallel_snapshot_does_not_follow_population_or_caller_mutations(fixture):
    before = deepcopy(fixture.snapshot.research_targets)
    serialized = json.loads(json.dumps(_snapshot(fixture.snapshot)))
    fixture.best.code = 'def solve(): return 500\n'
    fixture.best.score = 500.
    fixture.engine.db.pop_best_id = fixture.parent.id
    fixture.engine.iteration_no += 1
    assert fixture.snapshot.research_targets == before
    assert serialized['research_targets'] == before
    restored = Snapshot(**serialized)
    arguments = {'nested': {'value': 1}}
    inputs = tool_inputs(restored, {'task_hash': 'task-fixture'}, arguments)
    arguments['nested']['value'] = 2
    inputs['research_targets']['population_best']['source'] = 'mutated caller copy'
    assert inputs['arguments']['nested']['value'] == 1
    assert restored.research_targets == before
    assert input_binding(tool_inputs(restored, {'task_hash': 'task-fixture'}, {}))[
        'population_best_source_sha256'] == before['population_best']['source_sha256']


@pytest.mark.parametrize('damage', ['source', 'measurement', 'ancestry', 'task', 'snapshot'])
def test_resealed_tampering_cannot_rebind_saved_source_or_measurement(fixture, damage):
    value = deepcopy(fixture.snapshot.research_targets)
    snapshot = deepcopy(fixture.snapshot)
    if damage == 'source':
        value['selected']['source'] += '# changed bytes\n'
    elif damage == 'measurement':
        value['population_best']['measurement'] = deepcopy(value['selected']['measurement'])
    elif damage == 'ancestry':
        value['ancestry_parent']['program_id'] = 'p_wrong'
    elif damage == 'task':
        value['task_hash'] = 'another-task'
    else:
        snapshot.parent_code += '# different generation input\n'
    reseal(value)
    with pytest.raises(ValueError, match='research targets'):
        validate_targets(value, snapshot=snapshot, task_hash='task-fixture')


def test_failed_target_rejects_admitted_identity_and_cross_source_provenance(fixture):
    source = 'def solve(): raise ValueError("fixture")\n'
    result = SimpleNamespace(child_code=source, meta={'evaluation_id': 'eval-failed'},
        reached='full', confirmed=False, artifacts={'error': 'fixture failure'})
    failure = failed_provenance(result, iteration=12, B_evals=9)
    frozen = freeze_targets(fixture.engine, fixture.parent, source, failed=True, failure=failure)
    assert frozen['selected']['failure']['artifacts'] == result.artifacts
    for key, value in [('program_id', fixture.parent.id), ('selection_score', fixture.parent.score)]:
        forged = deepcopy(frozen)
        forged['selected'][key] = value
        with pytest.raises(ValueError, match='failed source cannot claim'):
            validate_targets(reseal(forged))
    with pytest.raises(ValueError, match='failure source/digest differs'):
        freeze_targets(fixture.engine, fixture.parent, source+'# different\n',
                       failed=True, failure=failure)


def test_tool_compatibility_fields_cannot_borrow_another_program_identity(fixture):
    inputs = tool_inputs(fixture.snapshot, {'task_hash': 'task-fixture'}, {})
    for field, value in [('parent_program_id', fixture.best.id),
                         ('parent_source', fixture.best.code),
                         ('parent_score', fixture.parent.score),
                         ('ancestry_parent_id', fixture.best.id)]:
        forged = deepcopy(inputs)
        forged[field] = value
        with pytest.raises(ValueError, match='compatibility input aliases differ'):
            input_binding(forged)
