"""Pure task preparation and inert adapter routing; never execute task source."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest

from sciel.engine import repeat_support as support
from sciel.engine.repeat_population_support import create_batches
from sciel.eval.registry import load_task
from sciel.hypothesis import circle_restart_body as legacy
from sciel.hypothesis import circle_traced_body as traced

ROOT = Path(__file__).resolve().parents[2]
CANONICAL = ROOT / 'tasks/oe/circle_packing'
spec = importlib.util.spec_from_file_location('prepare_traced_circle_task',
    ROOT / 'scripts/prepare_traced_circle_task.py')
preparation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preparation)


def hashes(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts}


@pytest.fixture
def traced_task(tmp_path):
    target = tmp_path / 'traced-task'
    preparation.prepare(CANONICAL, target)
    return load_task(str(target))


def test_preparation_preserves_canonical_files_and_binds_new_native_task(tmp_path, capsys):
    before = hashes(CANONICAL)
    target = tmp_path / 'empty'
    target.mkdir()
    preparation.main(['--canonical-task', str(CANONICAL), '--output', str(target)])
    receipt = json.loads(capsys.readouterr().out)
    task = load_task(str(target))
    source, evaluator = support.validate_circle_task(task, initial_source_policy=traced.OPERATION)
    assert source == traced.source_for().encode()
    assert hashlib.sha256(source).hexdigest() == support.TRACED_BODY_INITIAL_SHA
    assert task.task_hash == receipt['task_hash'] == support.TRACED_BODY_TASK_HASH
    assert task.task_hash not in (support.CANONICAL_TASK_HASH, support.BODY_TASK_HASH)
    assert task.task_hash in support.KNOWN_CIRCLE_TASK_HASHES
    assert support.NATIVE_GEOMETRY_TASK_EVALUATORS[task.task_hash] == support.EVALUATOR_SHA
    assert evaluator == (CANONICAL / 'evaluator.py').read_bytes()
    assert (target / 'task.yaml').read_bytes() == (CANONICAL / 'task.yaml').read_bytes()
    assert receipt['files'] == hashes(target) and receipt['executed'] is False
    assert receipt['initial_source_policy'] == traced.OPERATION
    assert receipt['execution_profile'] == support.TRACED_EXECUTION_PROFILE
    assert hashes(CANONICAL) == before
    assert not list(target.rglob('__pycache__'))


@pytest.mark.parametrize('damage', ['yaml', 'evaluator', 'source', 'asset', 'metadata'])
def test_traced_task_rejects_changed_task_inputs(traced_task, damage):
    path = Path(traced_task.root)
    if damage == 'yaml':
        with (path / 'task.yaml').open('a') as stream:
            stream.write('\nfull_timeout: 121.0\n')
    elif damage == 'evaluator':
        with (path / 'evaluator.py').open('a') as stream:
            stream.write('\n# changed bytes\n')
    elif damage == 'source':
        (path / 'initial_program.py').write_text(legacy.source_for())
    elif damage == 'asset':
        (path / 'assets').mkdir()
        (path / 'assets/new.txt').write_text('changes task identity')
    else:
        traced_task.full_timeout += 1
    with pytest.raises(ValueError, match='pinned|metadata'):
        support.validate_circle_task(traced_task, initial_source_policy=traced.OPERATION)


@pytest.mark.parametrize('profile', [None, 'unknown', False])
def test_explicit_invalid_profile_is_not_legacy(profile):
    with pytest.raises(ValueError, match='profile'):
        support.circle_initial_source_policy({'circle_generation':
            {'enabled': True, 'execution_profile': profile}})


def test_disabled_profile_and_cross_family_tasks_reject(traced_task):
    with pytest.raises(ValueError, match='disabled'):
        support.circle_initial_source_policy({'circle_generation':
            {'enabled': False, 'execution_profile': support.TRACED_EXECUTION_PROFILE}})
    with pytest.raises(ValueError, match='pinned'):
        support.validate_circle_task(traced_task, initial_source_policy=legacy.OPERATION)
    with pytest.raises(ValueError, match='pinned|metadata'):
        support.validate_circle_task(load_task(str(CANONICAL)), initial_source_policy=traced.OPERATION)


def settings():
    return dict(enabled=True, seed_master=31, max_comparisons=2, max_draw_attempts=15,
        max_wall_seconds=600, timeout_seconds=30, certification_seconds=10,
        cpu_seconds=35, mem_bytes=2**31)


@pytest.mark.parametrize('route', ['promotion', 'population'])
@pytest.mark.parametrize('family', ['canonical', 'legacy', 'traced'])
def test_actual_repeat_factories_bind_both_roles_and_keep_default_shape(
        tmp_path, monkeypatch, route, family):
    if family == 'canonical':
        task = load_task(str(CANONICAL)); circle = {'enabled': False}
    else:
        target = tmp_path / 'task'
        if family == 'traced':
            preparation.prepare(CANONICAL, target)
            circle = {'enabled': True, 'execution_profile': support.TRACED_EXECUTION_PROFILE}
        else:
            shutil.copytree(CANONICAL, target, ignore=shutil.ignore_patterns('__pycache__'))
            (target / 'initial_program.py').write_text(legacy.source_for())
            circle = {'enabled': True}
        task = load_task(str(target))
    identity = {'schema': 'fixture-adapter/1', 'runtime': {'kind': 'inert'}}
    probes, calls = [], []
    def adapter(**kwargs):
        probes.append(kwargs)
        return deepcopy(identity)
    def draw(source, **kwargs):
        calls.append((source, kwargs))
        return {'inert': True}
    monkeypatch.setattr('sciel.eval.repeat_draw.adapter_identity', adapter)
    monkeypatch.setattr('sciel.eval.repeat_draw.draw', draw)
    policy = support.circle_initial_source_policy({'circle_generation': circle})
    if route == 'promotion':
        instance = support.create(settings(), task, tmp_path / 'run', 'fixture-python',
                                  initial_source_policy=policy)
    else:
        instance = create_batches({'circle_generation': circle, 'repeat_population': settings()},
                                  task, tmp_path / 'run', 'fixture-python')
    try:
        assert len(probes) == 1 and not calls
        assert instance.identity['canonical_initial_source_sha256'] == hashlib.sha256(
            Path(task.initial_program_path).read_bytes()).hexdigest()
        assert instance.identity['task_evaluator_sha256'] == support.EVALUATOR_SHA
        assert instance.identity['task_hash'] == task.task_hash
        if family == 'traced':
            expected = support.traced_execution_policy(traced.OPERATION)
            assert {key: instance.identity[key] for key in expected} == expected
            assert expected['restart_trace']['scaffold_sha256'] == traced.identity()['common_scaffold_sha256']
        else:
            assert 'execution_profile' not in instance.identity and 'restart_trace' not in instance.identity
        # The single adapter closure serves both sources; no candidate is run.
        for source in ('inert comparator', 'inert challenger'):
            assert instance.draw_fn(source, seed=17)['inert'] is True
        assert len(calls) == 2
        for source, kwargs in calls:
            assert kwargs['expected_identity'] == identity and kwargs['seed'] == 17
            assert kwargs['python'] == 'fixture-python' and kwargs['runtime'] == {'backend': 'native'}
            if family == 'traced':
                assert kwargs['restart_trace'] is True
            else:
                assert 'restart_trace' not in kwargs
        assert not (tmp_path / 'run').exists()
    finally:
        instance.close()


def test_rejected_task_never_reaches_adapter(traced_task, tmp_path, monkeypatch):
    def forbidden(**kwargs):
        raise AssertionError('invalid task reached runtime identity')
    monkeypatch.setattr('sciel.eval.repeat_draw.adapter_identity', forbidden)
    traced_task.mem_bytes = 1
    with pytest.raises(ValueError, match='metadata'):
        support.create(settings(), traced_task, tmp_path / 'run', None,
                       initial_source_policy=traced.OPERATION)


def test_prepare_refuses_existing_data_and_unpinned_canonical_source(tmp_path):
    target = tmp_path / 'kept'; target.mkdir()
    (target / 'keep.txt').write_text('unchanged')
    before = hashes(target)
    with pytest.raises(ValueError, match='empty'):
        preparation.prepare(CANONICAL, target)
    assert hashes(target) == before
    altered = tmp_path / 'canonical'
    shutil.copytree(CANONICAL, altered, ignore=shutil.ignore_patterns('__pycache__'))
    with (altered / 'evaluator.py').open('a') as stream:
        stream.write('\n# unpinned\n')
    with pytest.raises(ValueError, match='pinned'):
        preparation.prepare(altered, tmp_path / 'never-published')
    assert not (tmp_path / 'never-published').exists()


def test_prepare_validates_copied_snapshot_before_publication(tmp_path, monkeypatch):
    original = Path.read_bytes
    def altered_copy(path):
        raw = original(path)
        if path == CANONICAL / 'task.yaml':
            return raw + b'\nfull_timeout: 121.0\n'
        return raw
    monkeypatch.setattr(Path, 'read_bytes', altered_copy)
    target = tmp_path / 'not-published'
    with pytest.raises(ValueError, match='pinned'):
        preparation.prepare(CANONICAL, target)
    assert not target.exists()
