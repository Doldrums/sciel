"""No candidate/evaluator subprocesses: mocked dispatch and one inert seed shim."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest

from sciel.diagnostics import paired_eval as paired
from sciel.eval.facade import EvalResult
from sciel.eval.registry import load_task


ROOT = Path(__file__).resolve().parents[2]
SOURCE = '# inert diagnostic candidate; never executed by these tests\n'


@pytest.fixture
def task():
    return load_task(str(ROOT / 'tasks/oe/circle_packing'))


def setup_receipt(folder, seed=11):
    setup = {'schema': paired.SEED_SCHEMA,
        'source_sha256': hashlib.sha256((folder / 'candidate.py').read_bytes()).hexdigest(),
        'evaluator_sha256': hashlib.sha256((folder / 'original_evaluator.py').read_bytes()).hexdigest(),
        'seed_applied': {'python_random': seed, 'numpy_legacy': seed,
            'before_candidate_import': True, 'determinism_proven': False,
            'candidate_reseeding_intercepted': False, 'python_hash_seed': None,
            'python_hash_seed_policy': 'inherited_uncontrolled'},
        'runtime': {'scope': 'mocked dispatcher; no subprocess'}}
    (folder / 'seed_receipt.json').write_text(json.dumps(setup))
    return setup


def native_result(folder, *, valid=True, wrong_source=False, stage='smoke'):
    centers = [[.1 + .15 * (i % 6), .1 + .15 * (i // 6)] for i in range(26)]
    if not valid:
        centers = [[.5, .5] for _ in range(26)]
    numeric = {'centers': centers, 'radii': [.03125] * 26}
    total = .8125 if valid else 0.
    score = total / 2.635
    geometry = {'schema': 'sciel-native-circle-geometry/1',
        'source_sha256': ('f' * 64 if wrong_source else
                         hashlib.sha256((folder / 'candidate.py').read_bytes()).hexdigest()),
        'evaluator_sha256': hashlib.sha256((folder / 'original_evaluator.py').read_bytes()).hexdigest(),
        'invocation_id': '1' * 32, 'stage': stage, 'confirmation': None,
        'numeric': numeric, 'numeric_sha256': hashlib.sha256(paired._canonical(numeric).encode()).hexdigest(),
        'centers_dtype': 'float64', 'radii_dtype': 'float64', 'native_validity': valid,
        'native_sum_radii': total, 'native_score': score}
    return EvalResult(ok=True, score=score, timing=.01,
        cases={'sum_radii': total, 'validity': float(valid), 'target_ratio': score, 'combined_score': score},
        artifacts={'native_geometry': paired._canonical(geometry)})


def test_exactly_one_original_smoke_request_and_persistent_separate_receipts(task, tmp_path, monkeypatch):
    calls = []
    original_bytes = Path(task.evaluator_path).read_bytes()
    def dispatch(evaluator, stage, program, **kwargs):
        folder = Path(evaluator).parent
        calls.append((evaluator, stage, program, kwargs))
        assert stage == 'smoke' and Path(program).read_bytes() == SOURCE.encode()
        assert (folder / 'original_evaluator.py').read_bytes() == original_bytes
        assert kwargs == dict(timeout=30., cpu_seconds=None, mem_bytes=None, python=None)
        setup_receipt(folder)
        return native_result(folder)
    monkeypatch.setattr(paired.sandbox, 'run_stage', dispatch)
    first = paired.paired_draw(SOURCE, task, 11, work_dir=tmp_path)
    second = paired.paired_draw(SOURCE, task, 11, work_dir=tmp_path)
    assert len(calls) == 2  # exactly one dispatch for each explicit draw
    assert first['stage_invocations'] == first['native_entrypoint_calls'] == 1
    assert first['status'] == 'evaluated' and first['ok'] is True and first['valid'] is True
    assert first['score'] == .8125 / 2.635 and first['cases']['sum_radii'] == .8125
    assert first['native_geometry']['evaluator_sha256'] == hashlib.sha256(original_bytes).hexdigest()
    assert first['artifact_dir'] != second['artifact_dir']
    assert first['shim_sha256'] == second['shim_sha256']
    assert first['task'] == second['task']
    assert first['timing'] >= 0
    assert json.loads(Path(first['receipt_path']).read_text()) == first
    assert Path(task.evaluator_path).read_bytes() == original_bytes
    assert first['seed_applied']['determinism_proven'] is False
    assert first['seed_applied']['candidate_reseeding_intercepted'] is False
    for name, path in paired._LOADER_FILES.items():
        assert first[name] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_changed_loader_cannot_authenticate_a_standalone_draw(task, tmp_path, monkeypatch):
    helper = tmp_path / 'loader.py'
    helper.write_text('original helper fixture')
    monkeypatch.setattr(paired, '_LOADER_FILES', {'candidate_loader_sha256': helper})
    def dispatch(evaluator, *args, **kwargs):
        folder = Path(evaluator).parent
        setup_receipt(folder)
        result = native_result(folder)
        helper.write_text('changed helper fixture')
        return result
    monkeypatch.setattr(paired.sandbox, 'run_stage', dispatch)
    receipt = paired.paired_draw(SOURCE, task, 11, work_dir=tmp_path)
    assert receipt['status'] == 'infrastructure' and receipt['score'] is None
    assert 'diagnostic loader bytes changed' in receipt['reason']
    assert receipt['stage_invocations'] == 1


def test_native_invalid_geometry_is_measured_penalty_not_execution_failure(task, tmp_path, monkeypatch):
    def dispatch(evaluator, *args, **kwargs):
        folder = Path(evaluator).parent
        setup_receipt(folder)
        return native_result(folder, valid=False)
    monkeypatch.setattr(paired.sandbox, 'run_stage', dispatch)
    receipt = paired.paired_draw(SOURCE, task, 11, work_dir=tmp_path)
    assert receipt['ok'] is True and receipt['valid'] is False
    assert receipt['score'] == 0. and receipt['status'] == 'evaluated'
    assert receipt['native_geometry']['numeric']['radii'][0] > 0
    assert receipt['cases']['sum_radii'] == 0.  # evaluator penalty, not physical sum


@pytest.mark.parametrize('damage', ['missing_seed', 'wrong_seed', 'bool_seed', 'wrong_source', 'full_stage', 'changed_bytes'])
def test_uncorroborated_seed_source_or_stage_never_becomes_matched_quality(task, tmp_path, monkeypatch, damage):
    calls = []
    def dispatch(evaluator, *args, **kwargs):
        folder = Path(evaluator).parent
        calls.append(evaluator)
        if damage != 'missing_seed':
            setup = setup_receipt(folder, seed=12 if damage == 'wrong_seed' else 11)
            if damage == 'bool_seed':
                setup['seed_applied']['python_random'] = True
                (folder / 'seed_receipt.json').write_text(json.dumps(setup))
        result = native_result(folder, wrong_source=damage == 'wrong_source',
                               stage='full' if damage == 'full_stage' else 'smoke')
        if damage == 'changed_bytes':
            (folder / 'candidate.py').write_text('# changed after return\n')
        return result
    monkeypatch.setattr(paired.sandbox, 'run_stage', dispatch)
    receipt = paired.paired_draw(SOURCE, task, 11, work_dir=tmp_path)
    assert len(calls) == 1
    assert receipt['status'] == 'infrastructure'
    assert receipt['ok'] is False and receipt['score'] is None and receipt['valid'] is None
    assert receipt['native_entrypoint_calls'] is None
    assert receipt['native_result']['ok'] is True  # retained, never used as corroborated quality
    assert json.loads(Path(receipt['receipt_path']).read_text()) == receipt


def test_timeout_after_seed_setup_is_failure_without_zero_quality_or_retry(task, tmp_path, monkeypatch):
    calls = []
    def dispatch(evaluator, *args, **kwargs):
        calls.append(evaluator)
        setup_receipt(Path(evaluator).parent)
        return EvalResult(ok=False, score=0., artifacts={'error_type': 'TimeoutError', 'error': 'timeout30s'})
    monkeypatch.setattr(paired.sandbox, 'run_stage', dispatch)
    receipt = paired.paired_draw(SOURCE, task, 11, work_dir=tmp_path)
    assert len(calls) == 1 and receipt['status'] == 'evaluation_failed'
    assert receipt['score'] is None and receipt['valid'] is None and receipt['ok'] is False
    assert receipt['native_result']['score'] == 0. and receipt['reason'] == 'timeout30s'
    assert receipt['seed_applied']['numpy_legacy'] == 11


@pytest.mark.parametrize('seed', [True, -1, 2**32, 1.5, '11'])
def test_invalid_seed_rejected_before_any_dispatch_or_directory(task, tmp_path, monkeypatch, seed):
    monkeypatch.setattr(paired.sandbox, 'run_stage', lambda *a, **kw: pytest.fail('must not dispatch'))
    work = tmp_path / 'not-created'
    with pytest.raises(ValueError, match='ambient seed'):
        paired.paired_draw(SOURCE, task, seed, work_dir=work)
    assert not work.exists()


@pytest.mark.parametrize('field,value', [('task_hash', 'f'*64), ('smoke_timeout', 120.),
                                        ('primary_metric', 'different_score'), ('cpu_seconds', True)])
def test_task_or_budget_changes_fail_before_dispatch(task, tmp_path, monkeypatch, field, value):
    setattr(task, field, value)
    monkeypatch.setattr(paired.sandbox, 'run_stage', lambda *a, **kw: pytest.fail('must not dispatch'))
    with pytest.raises(ValueError):
        paired.paired_draw(SOURCE, task, 11, work_dir=tmp_path / 'not-created')
    assert not (tmp_path / 'not-created').exists()


def test_nonregular_and_oversized_setup_files_are_not_read(tmp_path):
    large = tmp_path / 'large.json'
    large.write_bytes(b'x' * 16385)
    with pytest.raises(ValueError, match='receipt file'):
        paired._read_setup(large)
    link = tmp_path / 'link.json'
    link.symlink_to(large)
    with pytest.raises((ValueError, OSError)):
        paired._read_setup(link)


def test_shim_seeds_before_one_inert_original_entrypoint_without_loading_candidate(tmp_path):
    # Trusted fixture evaluator only observes RNG state; deliberately never loads
    # candidate.py. No archived/generated candidate or actual evaluator executes.
    original = b'''import random\nimport numpy as np\ndef smoke(path):\n    return {"python": random.random(), "numpy": float(np.random.random()), "path": path}\n'''
    source = b'raise AssertionError("candidate must never execute in this test")\n'
    (tmp_path / 'original_evaluator.py').write_bytes(original)
    (tmp_path / 'candidate.py').write_bytes(source)
    cfg = dict(seed=19, source_sha256=hashlib.sha256(source).hexdigest(),
        evaluator_sha256=hashlib.sha256(original).hexdigest(), seed_schema=paired.SEED_SCHEMA,
        thread_env=list(paired._THREAD_ENV))
    shim = tmp_path / 'shim.py'
    shim.write_text(paired._SHIM.replace('__CONFIG__', repr(cfg)))
    spec = importlib.util.spec_from_file_location('inert_seed_fixture', shim)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        observed = module.smoke(str(tmp_path / 'candidate.py'))
        again = module.smoke(str(tmp_path / 'candidate.py'))
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
    assert observed == again
    assert observed['python'] == random.Random(19).random()
    assert observed['numpy'] == float(np.random.RandomState(19).random_sample())
    assert (tmp_path / 'candidate.py').read_bytes() == source
    receipt = paired._read_setup(tmp_path / 'seed_receipt.json')
    assert receipt['seed_applied']['before_candidate_import'] is True
    assert receipt['seed_applied']['candidate_reseeding_intercepted'] is False
    assert receipt['seed_applied']['python_hash_seed_policy'] == 'inherited_uncontrolled'
