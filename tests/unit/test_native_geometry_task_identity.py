"""Captured geometry changes task identity; old evidence is never relabelled."""
import hashlib
import json
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import shutil

import pytest

from sciel.engine.repeat_support import (
    BODY_TASK_HASH, CANONICAL_TASK_HASH, EVALUATOR_SHA,
    TRACED_BODY_TASK_HASH, OPTIMIZER_TRACED_TASK_HASH,
    HISTORICAL_NATIVE_GEOMETRY_TASK_EVALUATORS, LEGACY_GEOMETRY_EVALUATOR_SHA,
    HISTORICAL_CIRCLE_TASK_HASHES, KNOWN_CIRCLE_TASK_HASHES,
    NATIVE_GEOMETRY_TASK_EVALUATORS, validate_circle_task,
)
from sciel.eval.registry import load_task
from sciel.eval.native_geometry import SCHEMA, validate_binding
from sciel.hypothesis import circle_restart_body
from sciel.hypothesis.circle_source_family import builder_for_operation
from sciel.hypothesis.measurement import metric_context

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "tasks/oe/circle_packing"


def copied_task(tmp_path, *, body=False):
    for name in ("task.yaml", "evaluator.py", "initial_program.py"):
        shutil.copyfile(TASK / name, tmp_path / name)
    if body:
        (tmp_path / "initial_program.py").write_text(circle_restart_body.source_for())
    return load_task(str(tmp_path))


@pytest.mark.parametrize("body", [False, True])
def test_new_evaluator_gets_new_exact_registered_task_identity(tmp_path, body):
    task = copied_task(tmp_path, body=body)
    assert task.task_hash == (BODY_TASK_HASH if body else CANONICAL_TASK_HASH)
    assert task.task_hash not in HISTORICAL_CIRCLE_TASK_HASHES
    initial, evaluator = validate_circle_task(task,
        initial_source_policy="circle_restart_body/1" if body else None)
    assert initial == Path(task.initial_program_path).read_bytes()
    assert hashlib.sha256(evaluator).hexdigest() == EVALUATOR_SHA
    assert NATIVE_GEOMETRY_TASK_EVALUATORS[task.task_hash] == EVALUATOR_SHA


def test_historical_native_without_geometry_retains_its_original_metric(tmp_path):
    # Old descriptor identity remains old evidence; current bytes do not prove it.
    task = replace(copied_task(tmp_path), task_hash=HISTORICAL_CIRCLE_TASK_HASHES[1])
    metric = metric_context(task)
    assert metric["task_hash"] == HISTORICAL_CIRCLE_TASK_HASHES[1]
    assert metric["task_hash"] in KNOWN_CIRCLE_TASK_HASHES
    assert metric["task_hash"] not in NATIVE_GEOMETRY_TASK_EVALUATORS
    with pytest.raises(ValueError, match="pinned"):
        validate_circle_task(task, initial_source_policy="circle_restart_body/1")
    assert metric_context(task) == metric  # Rejection never rewrites old facts.


def test_even_nonsemantic_task_edit_is_not_silently_admitted(tmp_path):
    task = copied_task(tmp_path, body=True)
    with Path(task.evaluator_path).open("a") as stream:
        stream.write("\n# Different evaluator bytes.\n")
    with pytest.raises(ValueError, match="pinned"):
        validate_circle_task(task, initial_source_policy="circle_restart_body/1")
    changed = load_task(str(tmp_path))
    assert changed.task_hash not in KNOWN_CIRCLE_TASK_HASHES
    assert changed.task_hash not in NATIVE_GEOMETRY_TASK_EVALUATORS


@pytest.mark.parametrize('operation,current_hash,previous_hash', [
    (None, CANONICAL_TASK_HASH,
     'a597f167ece11256458997773abbf009ce569b06518f88b17325eb585289d614'),
    ('circle_restart_body/1', BODY_TASK_HASH,
     'e3549f4cb473e8ae8bc319680f9f12a9a761f6520ee31c182682bb17b7acbc0e'),
    ('circle_restart_body_traced/1', TRACED_BODY_TASK_HASH,
     '976d9ae145dd3cd7c897bfde304c913e6c16b4e4fc0391a4137032191f823926'),
    ('circle_restart_body_traced/2', OPTIMIZER_TRACED_TASK_HASH,
     '3fc57bfd26a6164671df0bf4fe3013387a1f9dda5b603b806f82c1b598fbf944'),
])
def test_candidate_loader_identity_preserves_old_observations_but_refuses_old_execution(
        tmp_path, operation, current_hash, previous_hash):
    """Re-hash actual task bytes; validate saved synthetic arrays without dispatch."""
    copied_task(tmp_path)
    if operation is not None:
        (tmp_path / 'initial_program.py').write_text(builder_for_operation(operation).source_for())
    task = load_task(str(tmp_path))
    assert task.task_hash == current_hash != previous_hash
    source, evaluator = validate_circle_task(task, initial_source_policy=operation)
    assert hashlib.sha256(evaluator).hexdigest() == EVALUATOR_SHA
    old_sha = '44366fd28850b02da1303a7f1616b0e724f28b837d6ea506b26d4a91096523f4'
    assert HISTORICAL_NATIVE_GEOMETRY_TASK_EVALUATORS[previous_hash] == old_sha
    assert NATIVE_GEOMETRY_TASK_EVALUATORS[current_hash] == EVALUATOR_SHA != old_sha

    historical = replace(task, task_hash=previous_hash)
    with pytest.raises(ValueError, match='pinned'):
        validate_circle_task(historical, initial_source_policy=operation)

    # A legitimate zero-radius observation is still authenticated under its old
    # task/evaluator identity, never relabelled with the current evaluator SHA.
    numeric = {'centers': [[.5, .5] for _ in range(26)], 'radii': [0.] * 26}
    numeric_sha = hashlib.sha256(json.dumps(numeric, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode()).hexdigest()
    geometry = {'schema': SCHEMA, 'source_sha256': hashlib.sha256(source).hexdigest(),
        'evaluator_sha256': old_sha, 'invocation_id': 'a'*32, 'stage': 'full',
        'confirmation': None, 'numeric': numeric, 'numeric_sha256': numeric_sha,
        'centers_dtype': 'float64', 'radii_dtype': 'float64',
        'native_validity': True, 'native_sum_radii': 0., 'native_score': 0.}
    observation = {'ok': True, 'score': 0., 'confirmed': False, 'reached': 'full',
        'cases': {'sum_radii': 0., 'validity': 1., 'target_ratio': 0., 'combined_score': 0.}}
    before = deepcopy(geometry)
    assert validate_binding(geometry, observation, metric=metric_context(historical)) == before
    with pytest.raises(ValueError, match='unregistered native evaluator/task tuple'):
        validate_binding(geometry, observation, metric=metric_context(task))
    assert geometry == before


def test_pre_binary64_native_history_keeps_its_own_evaluator_mapping():
    for task_hash in (
        '34f919d0c1937a79ce039bcdf75b863443f1f63edb77734bc7b506ae17f91aeb',
        '149a3887108c5ff96ac783036ea62b318433ec8a0104f8bcf6db9e31253297ef',
        '5f65a12e159ced391a1635833cb2d24f60813763e5a6e237de228eea7d68987e',
        'a936b3365f80fcff7da567c5985e5d327295c66b75ce3144253405f1ebde6481',
    ):
        assert HISTORICAL_NATIVE_GEOMETRY_TASK_EVALUATORS[task_hash] == LEGACY_GEOMETRY_EVALUATOR_SHA
        assert NATIVE_GEOMETRY_TASK_EVALUATORS[task_hash] == LEGACY_GEOMETRY_EVALUATOR_SHA


@pytest.mark.parametrize('operation,previous_hash', [
    (None, '4a2bfdd6e0eb925e88560bbde9e3f6fab9b63f7e9058c9966393476ee7adff60'),
    ('circle_restart_body/1', 'fe5a424943169db9b29798713ae0465c4ed02c4e7bc215eff2e3edf0457a17c8'),
    ('circle_restart_body_traced/1', '8dc9f2f9e1fa842e22fc68c5642691aba41c8fadec7128c4430b63cf32ba3f67'),
    ('circle_restart_body_traced/2', '85b24214a299c3ef6e01c127a4e7875774d2902741d167be839ff445f462856d'),
])
def test_public_prompt_identity_preserves_pre_prompt_native_observations(tmp_path, operation, previous_hash):
    from sciel.engine.repeat_support import validate_registered_native_task
    from test_native_geometry_knowledge import geometry, outcome
    from sciel.hypothesis.measurement import evaluation_observation
    copied_task(tmp_path)
    if operation is not None:
        (tmp_path / 'initial_program.py').write_text(builder_for_operation(operation).source_for())
    task = load_task(str(tmp_path))
    assert validate_registered_native_task(task) == {
        'task_hash': task.task_hash, 'evaluator_sha256': EVALUATOR_SHA}
    assert HISTORICAL_NATIVE_GEOMETRY_TASK_EVALUATORS[previous_hash] == EVALUATOR_SHA
    historical = replace(task, task_hash=previous_hash)
    for descriptor in (task, historical):
        assert validate_binding(geometry(), evaluation_observation(outcome()),
                                metric=metric_context(descriptor)) == geometry()
    with pytest.raises(ValueError, match='pinned'):
        validate_circle_task(historical, initial_source_policy=operation)
    # Current native evaluator plus unreviewed task metadata must fail before
    # provider/evaluator dispatch, while historical evidence remains readable.
    with (tmp_path / 'task.yaml').open('a') as stream:
        stream.write('\npreflight_fixture: changed\n')
    with pytest.raises(ValueError, match='unregistered native evaluator/task tuple'):
        validate_registered_native_task(load_task(str(tmp_path)))


def test_native_preflight_leaves_unregistered_third_party_evaluators_alone(tmp_path):
    from sciel.engine.repeat_support import validate_registered_native_task
    copied_task(tmp_path)
    (tmp_path / 'evaluator.py').write_text('# unrelated custom evaluator; never executed\n')
    assert validate_registered_native_task(load_task(str(tmp_path))) is None
