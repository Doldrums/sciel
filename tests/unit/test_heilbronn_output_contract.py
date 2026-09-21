"""Trusted evaluator arithmetic on fixed geometry; mocked candidate loading only."""
from copy import deepcopy
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sciel.engine.novelty import CandidateNoveltyRegistry
from sciel.engine.observed_changes import ObservedChanges
from sciel.engine.output_novelty import OutputNoveltyRegistry
from sciel.engine.research_artifacts import capture_evaluation, bind_evaluation_program
from sciel.engine.research_targets import freeze_targets, tool_inputs
from sciel.engine.types import Program, Snapshot
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.eval.output_identity import adapt_output, canonical, digest
from sciel.eval.registry import load_task
from sciel.hypothesis.measurement import metric_context

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / 'tasks/oe/heilbronn_triangle'


@pytest.fixture
def scene(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('heilbronn_output_fixture', TASK/'evaluator.py')
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    x = np.linspace(.1, .9, 11)
    points = np.column_stack((x, .03 + .15*x*x))
    source = tmp_path/'inert_source.py'
    source.write_text('# Fixed geometry fixture, never imported or executed.\n')
    monkeypatch.setattr(evaluator, '_load', lambda *_: SimpleNamespace(heilbronn_triangle11=lambda: points))
    return evaluator, source, points, load_task(str(TASK))


def outcome(raw, reached='full'):
    confirmed = reached == 'confirm'
    result = EvalResult(**deepcopy(raw), confirmed_sampled=confirmed)
    return CascadeOutcome(result, result.score, reached, confirmed)


def identity(task, source, result, island=0):
    binding = dict(task_hash=task.task_hash,
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(), island_id=island)
    return adapt_output(task, binding['source_sha256'], result, binding), binding


def test_full_points_exactly_match_scored_float64_geometry_and_native_metric(scene):
    ev, source, points, task = scene
    raw = ev.full(str(source))
    assert raw['ok'] is True and raw['cases']['validity'] == 1.
    # Independent parabola identity, not the evaluator's shoelace helper.
    expected = min(.15/2*(b-a)*(c-a)*(c-b)
                   for a,b,c in itertools.combinations(points[:, 0], 3))
    assert raw['cases']['min_area'] == pytest.approx(expected, abs=1e-16)
    assert raw['cases']['min_area_normalized'] == pytest.approx(expected/(np.sqrt(3)/4))
    assert raw['score'] == raw['cases']['min_area_normalized']/ev.BENCHMARK
    text = raw['artifacts']['full_output']
    saved = json.loads(text)
    assert text == canonical(saved)
    assert np.asarray(saved['output']['points'], dtype=np.float64).tobytes() == points.tobytes()
    assert saved['source_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert saved['evaluator_sha256'] == hashlib.sha256((TASK/'evaluator.py').read_bytes()).hexdigest()
    assert saved['stage'] == 'full' and saved['confirmation'] is None
    assert saved['input_sha256'] is saved['seed_sha256'] is None
    measured, _ = identity(task, source, outcome(raw))
    assert measured['output_sha256'] == digest({'points': points.tolist()})
    assert measured['scope']['adapter'] == 'canonical-json/1'
    assert 'native_geometry' not in raw['artifacts']
    metric = metric_context(task)['semantics']
    assert metric['numerator_case'] == 'min_area_normalized'
    assert metric['denominator'] == ev.BENCHMARK == 0.036529889880030156


def test_valid_degenerate_zero_seed_and_signed_zero_are_not_failed_measurements(scene):
    ev, source, points, task = scene
    points[:] = 0.
    points[0, 0] = -0.
    raw = ev.smoke(str(source))
    assert raw['ok'] and raw['score'] == 0 and raw['cases']['validity'] == 1
    saved = json.loads(raw['artifacts']['full_output'])
    assert np.asarray(saved['output']['points']).tobytes() == points.tobytes()
    assert identity(task, source, outcome(raw, 'smoke'))[0] is not None


@pytest.mark.parametrize('invalid', ['shape', 'nan', 'inf', 'outside'])
def test_invalid_geometry_has_no_full_output_or_false_identity(scene, monkeypatch, invalid):
    ev, source, points, task = scene
    if invalid == 'shape': points = points[:10]
    elif invalid == 'nan': points[0, 0] = np.nan
    elif invalid == 'inf': points[0, 0] = np.inf
    else: points[0] = [-.1, .5]
    monkeypatch.setattr(ev, '_load', lambda *_: SimpleNamespace(heilbronn_triangle11=lambda: points))
    raw = ev.full(str(source))
    assert raw['ok'] is False and raw['cases']['validity'] == 0.
    assert raw['cases']['min_area'] == raw['cases']['min_area_normalized'] == 0.
    assert 'full_output' not in raw['artifacts']
    assert identity(task, source, outcome(raw))[0] is None


def test_changed_source_during_invocation_fails_without_witness(scene, monkeypatch):
    ev, source, points, task = scene
    def fixed_result():
        source.write_text('# changed after invocation began\n')
        return points
    monkeypatch.setattr(ev, '_load', lambda *_: SimpleNamespace(heilbronn_triangle11=fixed_result))
    raw = ev.full(str(source))
    assert raw['ok'] is False and 'changed during invocation' in raw['artifacts']['error']
    assert 'full_output' not in raw['artifacts']


@pytest.mark.parametrize('damage', ['source', 'evaluator', 'stage', 'partial', 'noncanonical'])
def test_malformed_or_misbound_saved_witness_cannot_create_identity(scene, damage):
    ev, source, _, task = scene
    result = outcome(ev.full(str(source)))
    witness = json.loads(result.result.artifacts['full_output'])
    if damage == 'source': witness['source_sha256'] = '0' * 64
    elif damage == 'evaluator': witness['evaluator_sha256'] = '0' * 64
    elif damage == 'stage': witness['stage'] = 'smoke'
    elif damage == 'partial': del witness['output']
    result.result.artifacts['full_output'] = (json.dumps(witness, indent=2)
        if damage == 'noncanonical' else canonical(witness))
    assert identity(task, source, result)[0] is None


def test_same_output_different_source_is_gated_but_actual_coordinate_change_is_allowed(scene):
    ev, source, points, task = scene
    registry = OutputNoveltyRegistry({'task_hash': task.task_hash})
    first, bind = identity(task, source, outcome(ev.full(str(source))))
    assert registry.observe('e1', first, bind)['status'] == 'new_output'
    registry.bind_program('e1', 'p1')
    source.write_text('# A different source producing the exact same measured geometry.\n')
    second, bind = identity(task, source, outcome(ev.full(str(source))), island=7)
    repeated = registry.observe('e2', second, bind)
    assert repeated['status'] == 'exact_recorded_repeat' and repeated['accept'] is False
    assert repeated['match']['program_id'] == 'p1'
    # Alter one actual output, without relying on score or case-vector equality.
    points[5, 1] += .001
    third_raw = ev.full(str(source))
    third, bind = identity(task, source, outcome(third_raw))
    assert third['output_sha256'] != first['output_sha256']
    assert registry.observe('e3', third, bind)['status'] == 'new_output'


def test_worst_confirmation_keeps_its_exact_points_and_selected_trial(scene, monkeypatch):
    ev, source, points, task = scene
    raw = []
    for factor in (1., .7, .9):
        original = points.copy()
        points[:] *= factor
        raw.append(ev.full(str(source)))
        points[:] = original
    before = deepcopy(raw)
    values = iter(raw)
    monkeypatch.setattr(ev, 'full', lambda _: next(values))
    confirmed = ev.confirm(str(source), 3)
    assert confirmed['score'] == raw[1]['score'] and confirmed['cases']['n_trials'] == 3.
    witness = json.loads(confirmed['artifacts']['full_output'])
    assert witness['output'] == json.loads(raw[1]['artifacts']['full_output'])['output']
    assert witness['confirmation'] == {'trials':3, 'selected_trial':1,
        'selection':'first_failure_else_lowest_score'}
    assert witness['stage'] == 'full'
    assert raw == before  # selecting a trial must not mutate saved trial records
    adapted, _ = identity(task, source, outcome(confirmed, 'confirm'))
    assert adapted['evidence']['selected_trial'] == 1
    assert adapted['scope']['stage']['selected_output_only'] is True
    wrong = outcome(confirmed, 'confirm')
    wrong.result.cases['n_trials'] = 2.
    assert identity(task, source, wrong)[0] is None


def test_confirmation_failure_never_certifies_surviving_geometry(scene, monkeypatch):
    ev, source, points, task = scene
    good = ev.full(str(source))
    failed = ev._fail('known failed fixture')
    results = iter([good, failed, good])
    monkeypatch.setattr(ev, 'full', lambda _: next(results))
    result = ev.confirm(str(source), 3)
    assert result['ok'] is False and 'full_output' not in result['artifacts']
    assert json.loads(result['artifacts']['confirmation_selection'])['selected_trial'] == 1
    assert identity(task, source, outcome(result, 'confirm'))[0] is None


def test_saved_geometry_reaches_frozen_research_tool_input_exactly(scene, tmp_path):
    ev, source_path, points, task = scene
    raw = ev.full(str(source_path))
    source = source_path.read_text()
    program = Program('p1', source, raw['score'], raw['cases'], 0, 'gen')
    engine = SimpleNamespace(run_dir=tmp_path, task=task, cfg={'random_seed':0},
        log=SimpleNamespace(run_id='heilbronn-fixture'), candidate_novelty=CandidateNoveltyRegistry(),
        observed_changes=ObservedChanges(), iteration_no=1, B_evals=1,
        db=SimpleNamespace(pop_best_id='p1', get=lambda _: program))
    reference = capture_evaluation(engine, source, outcome(raw), 'e1')
    bind_evaluation_program(engine, program, 'e1')
    engine.candidate_novelty.register(source, 'e1', evaluation_ok=True)
    engine.candidate_novelty.bind_program(source, 'p1', evaluation_id='e1')
    engine.observed_changes.observe(program, dict(ok=True, score=raw['score'], cases=raw['cases'],
        artifacts=raw['artifacts'], reached='full', confirmed=False))
    snapshot = Snapshot(0, 'diff', 'fake', 1, 'p1', source, program.score)
    snapshot.research_targets = freeze_targets(engine, program, source)
    inputs = tool_inputs(snapshot, {'task_hash':task.task_hash}, {})
    saved = inputs['research_targets']['selected']['saved_evaluation']
    assert saved['status'] == 'available' and saved['reference'] == reference
    delivered = saved['record']['result']['artifacts']['full_output']
    assert delivered == raw['artifacts']['full_output']
    assert np.asarray(json.loads(delivered)['output']['points']).tobytes() == points.tobytes()
