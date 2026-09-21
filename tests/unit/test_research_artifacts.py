"""Research reads immutable returned measurements; no candidate code is executed."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sciel.engine.novelty import CandidateNoveltyRegistry
from sciel.engine.observed_changes import ObservedChanges
from sciel.engine.research_artifacts import (
    MAX_INLINE_BYTES, bind_evaluation_program, capture_evaluation, saved_evaluation,
    saved_evaluation_for_program, validate_evaluation_bindings, validate_saved_evaluation,
)
from sciel.engine.research_targets import freeze_targets, tool_inputs, validate_targets
from sciel.engine.types import Program, Snapshot
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.methods.models import canonical_bytes, content_hash


@pytest.fixture
def scene(tmp_path):
    source = 'def solve(): return "fixture source, never executed"\n'
    program = Program('p1', source, .5, {'quality': .5}, 0, 'gen')
    task = SimpleNamespace(task_hash='fixture-task', full_timeout=28., per_run_timeout=25.,
                           cpu_seconds=30, confirm_n=3)
    engine = SimpleNamespace(run_dir=tmp_path, task=task, cfg={'random_seed': 42},
        log=SimpleNamespace(run_id='fixture-run'), candidate_novelty=CandidateNoveltyRegistry(),
        observed_changes=ObservedChanges(), iteration_no=4, B_evals=3,
        db=SimpleNamespace(pop_best_id=program.id, get=lambda _: program))
    outputs = {'saved_output': {'array': [1., -0., .12345678901234567], 'seed': 17,
                              'state': {'counter': 30, 'note': '雪🙂'}}}
    outcome = CascadeOutcome(EvalResult(True, .5, {'quality': .5}, timing=1.625,
                                       artifacts=outputs), raw_first_score=.5, reached='full')
    return engine, program, outcome


def capture(scene, ident='evaluation-1', *, bind=True):
    engine, program, outcome = scene
    reference = capture_evaluation(engine, program.code, outcome, ident)
    if bind:
        bind_evaluation_program(engine, program, ident)
        if engine.candidate_novelty is not None:
            engine.candidate_novelty.register(program.code, ident, evaluation_ok=True)
            engine.candidate_novelty.bind_program(program.code, program.id, evaluation_id=ident)
    return reference


def test_exact_saved_output_and_conditions_reach_frozen_tool_input(scene):
    engine, program, outcome = scene
    original = deepcopy(outcome.result.artifacts)
    reference = capture(scene)
    engine.observed_changes.observe(program, dict(ok=True, score=.5, cases={'quality': .5},
        artifacts=original, reached='full', confirmed=False))
    snapshot = Snapshot(0, 'diff', 'fixture', 4, program.id, program.code, program.score)
    snapshot.research_search_evidence = {'constraint': 'measured, not inferred'}
    snapshot.research_targets = freeze_targets(engine, program, program.code)
    inputs = tool_inputs(snapshot, {'task_hash': engine.task.task_hash}, {'seed': 9})
    saved = inputs['research_targets']['selected']['saved_evaluation']
    assert saved['status'] == 'available' and saved['reference'] == reference
    assert canonical_bytes(saved['record']['result']['artifacts']) == canonical_bytes(original)
    assert saved['record']['result']['timing'] == 1.625
    assert saved['record']['conditions']['stage_timeouts_seconds']['full'] == 28.
    assert saved['record']['conditions']['search_seed_config'] == 42
    assert 'not measured discovery CPU' in saved['record']['conditions']['timing_scope']
    assert inputs['search_evidence'] == snapshot.research_search_evidence
    inputs['research_targets']['selected']['saved_evaluation']['record']['result']['artifacts'].clear()
    assert saved_evaluation_for_program(engine, program)['record']['result']['artifacts'] == original
    assert outcome.result.artifacts == original


def test_planner_saved_evaluation_paths_resolve_to_exact_available_record(scene):
    from sciel.research_tools.controller import ResearchTools
    from test_tool_input_projection import pointer_value, rendered_view
    engine, program, outcome = scene
    reference = capture(scene)
    current = Snapshot(0, 'diff', 'fixture', 4, program.id, program.code, program.score)
    current.research_targets = freeze_targets(engine, program, program.code)
    inputs = tool_inputs(current, {'task_hash': engine.task.task_hash}, {})
    view = rendered_view(ResearchTools._target_prompt(current.research_targets, 0))
    paths = view['access_paths']
    for row in view['references']:
        base = paths[row['role']]
        assert row['saved_evaluation_status'] == 'available'
        assert row['saved_evaluation_sha256'] == reference['sha256']
        assert pointer_value(inputs, base + paths['fields']['saved_artifacts']) == outcome.result.artifacts
        assert pointer_value(inputs, base + paths['fields']['saved_timing']) == outcome.result.timing
        assert pointer_value(inputs, base + paths['fields']['saved_conditions'])['search_seed_config'] == 42


def test_resume_restores_exact_program_measurement_not_last_same_source(scene):
    engine, program, outcome = scene
    first = capture(scene)
    outcome.result.artifacts = {'saved_output': {'value': 'a later stochastic measurement'}}
    second = capture(scene, 'evaluation-2', bind=False)
    assert first != second
    restored = CandidateNoveltyRegistry()
    restored.load_state(json.loads(json.dumps(engine.candidate_novelty.state())))
    engine.candidate_novelty = restored
    saved = saved_evaluation_for_program(engine, program)
    assert saved['reference'] == first
    assert saved['record']['evaluation_id'] == 'evaluation-1'


def test_disabled_novelty_and_observation_migrant_alias_keep_saved_measurement(scene):
    engine, program, _ = scene
    engine.candidate_novelty = None
    reference = capture(scene)
    engine.observed_changes.observe(program, dict(ok=True, score=.5, cases={'quality': .5},
                                                reached='full', confirmed=False))
    migrant = Program('p_migrant', program.code, program.score, program.cases,
                      4, 'migrant', parent_id=program.id)
    engine.observed_changes.bind_alias(migrant, program)
    assert saved_evaluation_for_program(engine, migrant)['reference'] == reference


def test_capture_is_idempotent_but_identity_cannot_be_overwritten(scene):
    engine, program, outcome = scene
    reference = capture(scene)
    assert capture_evaluation(engine, program.code, outcome, 'evaluation-1') == reference
    outcome.result.artifacts['new'] = 'a different output'
    with pytest.raises(ValueError, match='changed on replay'):
        capture_evaluation(engine, program.code, outcome, 'evaluation-1')
    assert saved_evaluation_for_program(engine, program)['reference'] == reference


@pytest.mark.parametrize('location', ['directory', 'index', 'artifact'])
def test_saved_access_rejects_symlinks(scene, tmp_path, location):
    engine, program, _ = scene
    reference = capture(scene)
    root = tmp_path / 'research_evidence' / 'evaluations'
    if location == 'directory':
        root.rename(tmp_path / 'redirected')
        root.symlink_to(tmp_path / 'redirected', target_is_directory=True)
    else:
        path = root / ((reference['sha256'] if location == 'artifact' else
            'evaluation-' + hashlib.sha256(b'evaluation-1').hexdigest()) + '.json')
        moved = tmp_path / 'redirected.json'
        path.rename(moved)
        path.symlink_to(moved)
    with pytest.raises(ValueError, match='symlink'):
        saved_evaluation_for_program(engine, program)


def test_bad_identity_or_modified_content_is_rejected(scene):
    engine, program, _ = scene
    reference = capture(scene)
    with pytest.raises(ValueError, match='source/evaluation'):
        saved_evaluation(engine, program.code + '# different', 'evaluation-1')
    path = Path(engine.run_dir) / 'research_evidence' / 'evaluations' / (reference['sha256'] + '.json')
    body = json.loads(path.read_text())
    body['result']['artifacts'] = {}
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match='identity differs'):
        saved_evaluation_for_program(engine, program)


def test_oversize_is_whole_omission_with_saved_reference(scene):
    engine, program, outcome = scene
    outcome.result.artifacts = {'large': 'x' * MAX_INLINE_BYTES}
    reference = capture(scene)
    saved = saved_evaluation_for_program(engine, program)
    assert saved['status'] == 'unavailable' and saved['record'] is None
    assert saved['reference'] == reference
    assert 'no values were truncated' in saved['reason']
    validate_saved_evaluation(saved, source_sha256=reference['source_sha256'], task_hash=engine.task.task_hash)


def test_missing_identity_is_unavailable_and_never_latest_source(scene):
    engine, program, _ = scene
    capture(scene, bind=False)
    assert saved_evaluation_for_program(engine, program)['status'] == 'unavailable'
    assert saved_evaluation(engine, program.code, '../../not-a-path')['status'] == 'unavailable'


def test_unstarted_engine_does_not_fabricate_a_run_bound_measurement(scene):
    engine, program, outcome = scene
    engine.log.run_id = None
    assert capture_evaluation(engine, program.code, outcome, 'not-started') is None
    assert saved_evaluation_for_program(engine, program)['status'] == 'unavailable'


def test_resealed_frozen_target_cannot_claim_other_task_artifact(scene):
    engine, program, _ = scene
    capture(scene)
    targets = freeze_targets(engine, program, program.code)
    saved = targets['selected']['saved_evaluation']
    saved['record']['task_hash'] = saved['reference']['task_hash'] = 'another-task'
    saved['reference']['sha256'] = content_hash(saved['record'])
    saved['reference']['bytes'] = len(canonical_bytes(saved['record']))
    targets['sha256'] = content_hash({k: v for k, v in targets.items() if k != 'sha256'})
    with pytest.raises(ValueError, match='source/task'):
        validate_targets(targets)


def test_binding_rollback_can_reuse_discarded_program_id_without_old_measurement(scene):
    engine, program, outcome = scene
    engine.candidate_novelty = None
    capture(scene)
    checkpoint_bindings = json.loads(json.dumps(engine.research_evaluation_bindings))
    later = Program('p2', 'def solve(): return 2\n', .6, {}, 0, 'gen')
    capture_evaluation(engine, later.code, outcome, 'evaluation-discarded')
    bind_evaluation_program(engine, later, 'evaluation-discarded')
    engine.research_evaluation_bindings = checkpoint_bindings
    assert saved_evaluation_for_program(engine, later)['status'] == 'unavailable'
    later.code = 'def solve(): return 3\n'
    outcome.result.artifacts = {'state': 'new committed measurement'}
    capture_evaluation(engine, later.code, outcome, 'evaluation-after-rollback')
    bind_evaluation_program(engine, later, 'evaluation-after-rollback')
    saved = saved_evaluation_for_program(engine, later)
    assert saved['record']['evaluation_id'] == 'evaluation-after-rollback'
    assert saved['record']['result']['artifacts'] == {'state': 'new committed measurement'}


def test_checkpoint_binding_rejects_program_source_substitution(scene):
    engine, program, _ = scene
    capture(scene)
    with pytest.raises(ValueError, match='checkpoint program source'):
        validate_evaluation_bindings(engine.research_evaluation_bindings,
                                    program_sources={program.id: program.code + '# edited'})


def test_real_checkpoint_roundtrip_restores_binding_authority(tmp_path, monkeypatch):
    from sciel.engine import checkpoint
    from test_evaluation_artifacts import make_engine
    engine = make_engine(tmp_path)
    engine.log.run_start(seed=1)
    measured = CascadeOutcome(EvalResult(True, .5, {'quality': .5},
                                        artifacts={'data': '[1,2,3]'}), .5, 'full')
    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate', lambda *a, **k: measured)
    try:
        engine.seed()
        program = engine.db.get(engine.db.pop_best_id)
        expected = deepcopy(engine.research_evaluation_bindings)
        path = checkpoint.save(engine, 0)
        saved = json.loads(Path(path).read_text())
        assert saved['research_evaluation_bindings'] == expected
        engine.research_evaluation_bindings = {'discarded': {'source_sha256': 'f' * 64,
                                                           'evaluation_id': 'uncommitted'}}
        checkpoint.load(engine, path)
        assert engine.research_evaluation_bindings == expected
        assert saved_evaluation_for_program(engine, program)['record']['result']['artifacts'] == {'data': '[1,2,3]'}
    finally:
        engine.close()
