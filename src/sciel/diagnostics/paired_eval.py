"""One native Circle smoke draw with explicitly limited ambient RNG control.

The candidate and evaluator are copied byte-for-byte. A separate shim seeds
Python's global random generator and NumPy's legacy generator before the
original smoke entrypoint imports the candidate. It does not intercept reseeding,
default_rng(), wall time, OS entropy, or library nondeterminism. Matched ambient
seeds are therefore neither proof of deterministic output nor independent trials.

Existing run_stage owns timeout/process-group cleanup and credential filtering.
Its native resource limits are best effort, not a hostile-code security sandbox.
No cascade, confirmation, retry, certification, or historical score update occurs.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import sys
import tempfile
import time

from ..eval import sandbox


SCHEMA = 'sciel-diagnostic-native-circle-draw/1'
SEED_SCHEMA = 'sciel-diagnostic-ambient-seed/1'
_THREAD_ENV = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
               'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS')
_LOADER_FILES = {
    'candidate_loader_sha256': Path(__file__).resolve().parents[1] / 'candidate_modules/__init__.py',
    'package_init_sha256': Path(__file__).resolve().parents[1] / '__init__.py',
}
_SHIM = '''"""Diagnostic seed boundary; original evaluator/candidate bytes unchanged."""
import hashlib
from sciel.candidate_modules import load_candidate
import json
import os
from pathlib import Path
import platform
import random
import sys

CONFIG = __CONFIG__

def _check(path, expected):
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError("diagnostic input bytes changed")

def smoke(program_path):
    directory = Path(__file__).resolve().parent
    evaluator = directory / "original_evaluator.py"
    source = Path(program_path).resolve()
    if source != directory / "candidate.py":
        raise ValueError("diagnostic candidate path differs")
    _check(source, CONFIG["source_sha256"])
    _check(evaluator, CONFIG["evaluator_sha256"])
    original = load_candidate(evaluator, "diagnostic_original_evaluator")
    import numpy as np
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    setup = {"schema": CONFIG["seed_schema"], "source_sha256": CONFIG["source_sha256"],
        "evaluator_sha256": CONFIG["evaluator_sha256"],
        "seed_applied": {"python_random": CONFIG["seed"], "numpy_legacy": CONFIG["seed"],
            "before_candidate_import": True, "determinism_proven": False,
            "candidate_reseeding_intercepted": False,
            "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
            "python_hash_seed_policy": "inherited_uncontrolled"},
        "runtime": {"python_executable": sys.executable, "python_version": platform.python_version(),
            "prefix": sys.prefix, "platform": platform.platform(), "numpy_version": np.__version__,
            "thread_env": {k: os.environ.get(k) for k in CONFIG["thread_env"]}}}
    (directory / "seed_receipt.json").write_text(
        json.dumps(setup, sort_keys=True, separators=(",", ":"), allow_nan=False), encoding="utf-8")
    result = original.smoke(str(source))
    _check(source, CONFIG["source_sha256"])
    _check(evaluator, CONFIG["evaluator_sha256"])
    return result
'''


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write(path, value):
    """Atomic replacement only inside this draw's newly allocated directory."""
    temporary = path.with_suffix('.pending.json')
    with temporary.open('x', encoding='utf-8') as stream:
        stream.write(_canonical(value))
    temporary.replace(path)


def _read_setup(path):
    # Do not block on candidate-created FIFO/symlinks or read unbounded text.
    flags = os.O_RDONLY | os.O_NONBLOCK | getattr(os, 'O_NOFOLLOW', 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, 'rb') as stream:
        metadata = os.fstat(stream.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 16384:
            raise ValueError('invalid ambient seed receipt file')
        raw = stream.read(16385)
    if len(raw) > 16384:
        raise ValueError('ambient seed receipt exceeds byte limit')
    return json.loads(raw)


def _task_identity(task):
    from ..engine.repeat_support import NATIVE_GEOMETRY_TASK_EVALUATORS
    from ..eval.registry import load_task
    from ..hypothesis.measurement import metric_context

    if task.qualified_name != 'oe/circle_packing' or task.primary_metric != 'combined_score':
        raise ValueError('diagnostic requires the native Circle task')
    loaded = load_task(task.root)
    evaluator = Path(task.evaluator_path).read_bytes()
    evaluator_sha = _sha(evaluator)
    if (loaded.task_hash != task.task_hash or
            NATIVE_GEOMETRY_TASK_EVALUATORS.get(task.task_hash) != evaluator_sha):
        raise ValueError('native Circle task/evaluator identity differs')
    if metric_context(task) != metric_context(loaded):
        raise ValueError('native Circle measurement declaration differs')
    if task.smoke_timeout != loaded.smoke_timeout:
        raise ValueError('diagnostic must retain the original smoke wall limit')
    timeout = task.smoke_timeout
    if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError('invalid smoke wall limit')
    for name in ('cpu_seconds', 'mem_bytes'):
        value = getattr(task, name)
        if value is not None and (type(value) is not int or value <= 0):
            raise ValueError('invalid diagnostic resource limit: ' + name)
    return {'task_name': task.qualified_name, 'task_hash': task.task_hash,
            'evaluator_sha256': evaluator_sha, 'metric': metric_context(task)}, evaluator


def paired_draw(source, task, seed, *, work_dir, python=None):
    """Persist and return one original smoke draw; never execute full/confirm.

    A unique directory is created below work_dir, so repeated calls never replace
    earlier receipts. stage_invocations counts calls to sandbox.run_stage, including
    unsuccessful dispatch attempts. native_entrypoint_calls is 1 only when the
    original evaluator's returned geometry proves a completed native invocation;
    otherwise unknown. Failure score is None; native_result retains the evaluator's
    explicit zero penalty separately. Returned paths identify durable artifacts.
    """
    if type(source) is not str or not source.strip():
        raise ValueError('candidate source must be nonempty text')
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError('ambient seed must be an integer in [0, 2**32)')
    encoded = source.encode('utf-8')
    identity, evaluator = _task_identity(task)
    source_sha = _sha(encoded)
    base = Path(work_dir)
    base.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f'draw_{seed}_', dir=base)).resolve()
    config = {'seed': seed, 'source_sha256': source_sha,
              'evaluator_sha256': identity['evaluator_sha256'],
              'seed_schema': SEED_SCHEMA, 'thread_env': list(_THREAD_ENV)}
    shim = _SHIM.replace('__CONFIG__', repr(config)).encode('utf-8')
    inputs = {'candidate.py': encoded, 'original_evaluator.py': evaluator, 'seed_evaluator.py': shim}
    for name, data in inputs.items():
        with (directory / name).open('xb') as stream:
            stream.write(data)
    receipt = {'schema': SCHEMA, 'status': 'prepared', 'source_sha256': source_sha,
        'seed': seed, 'task': identity, 'shim_sha256': _sha(shim),
        'adapter_sha256': _sha(Path(__file__).read_bytes()),
        'runner_sha256': _sha(Path(sandbox._RUNNER).read_bytes()),
        'sandbox_sha256': _sha(Path(sandbox.__file__).read_bytes()),
        **{name: _sha(path.read_bytes()) for name, path in _LOADER_FILES.items()},
        'requested_python': python or sys.executable,
        'protocol': 'one_original_native_smoke_no_retry_no_selection',
        'limits': {'wall_seconds': task.smoke_timeout, 'cpu_seconds': task.cpu_seconds,
                   'mem_bytes': task.mem_bytes, 'resource_limits_verified': False},
        'stage_invocations': 0, 'native_entrypoint_calls': None,
        'ok': False, 'valid': None, 'score': None, 'cases': {}, 'native_geometry': None,
        'native_result': None, 'seed_applied': None, 'runtime': None, 'reason': None,
        'started_at': _now(), 'finished_at': None, 'timing': None,
        'artifact_dir': str(directory), 'receipt_path': str(directory / 'receipt.json'),
        'limitations': ['Matched ambient RNG seeds are not independent algorithm seeds.',
            'Candidate code may reseed or use default_rng, time, OS entropy, or nondeterministic libraries.',
            'Python hash seeding is inherited, not controlled by this adapter.',
            'Native smoke uses the task tolerance/score; no exact certificate is performed.',
            'Native subprocess limits do not isolate hostile code from host files/network.']}
    _write(directory / 'receipt.json', receipt)
    started = time.monotonic()
    try:
        receipt.update(status='running', stage_invocations=1)
        _write(directory / 'receipt.json', receipt)
        result = sandbox.run_stage(str(directory / 'seed_evaluator.py'), 'smoke',
            str(directory / 'candidate.py'), timeout=task.smoke_timeout,
            cpu_seconds=task.cpu_seconds, mem_bytes=task.mem_bytes, python=python)
        receipt['native_result'] = result.to_json()
        for name, path in _LOADER_FILES.items():
            if _sha(path.read_bytes()) != receipt[name]:
                raise ValueError('diagnostic loader bytes changed: ' + name)
        for name, data in inputs.items():
            if _sha((directory / name).read_bytes()) != _sha(data):
                raise ValueError('diagnostic input bytes changed: ' + name)
        setup = _read_setup(directory / 'seed_receipt.json')
        applied = setup.get('seed_applied', {})
        if (set(setup) != {'schema', 'source_sha256', 'evaluator_sha256', 'seed_applied', 'runtime'} or
                setup.get('schema') != SEED_SCHEMA or setup.get('source_sha256') != source_sha or
                setup.get('evaluator_sha256') != identity['evaluator_sha256'] or
                type(applied.get('python_random')) is not int or type(applied.get('numpy_legacy')) is not int or
                applied.get('python_random') != seed or applied.get('numpy_legacy') != seed or
                applied.get('before_candidate_import') is not True or
                applied.get('determinism_proven') is not False or
                applied.get('candidate_reseeding_intercepted') is not False or
                applied.get('python_hash_seed_policy') != 'inherited_uncontrolled' or
                type(setup.get('runtime')) is not dict):
            raise ValueError('ambient seed receipt differs from requested draw')
        receipt.update(seed_applied=applied, runtime=setup.get('runtime'))
        if result.ok:
            from ..eval.native_geometry import parse_artifact, validate_binding
            geometry = parse_artifact(result.artifacts.get('native_geometry'))
            validate_binding(geometry, {'ok': result.ok, 'score': result.score,
                'cases': result.cases, 'confirmed': False, 'reached': 'smoke'},
                source_sha256=source_sha, metric=identity['metric'])
            if geometry['evaluator_sha256'] != identity['evaluator_sha256']:
                raise ValueError('native geometry evaluator differs from original task')
            receipt.update(status='evaluated', ok=True, valid=geometry['native_validity'], score=result.score,
                cases=result.cases, native_geometry=geometry, native_entrypoint_calls=1)
        else:
            receipt.update(status='evaluation_failed', reason=result.artifacts.get('error') or result.stderr)
    except Exception as exc:
        receipt.update(status='infrastructure', ok=False, score=None,
                       reason=f'{type(exc).__name__}: {exc}'[:2000])
    except BaseException:
        receipt.update(status='interrupted', reason='draw interrupted; no retry performed')
        raise
    finally:
        receipt.update(finished_at=_now(), timing=time.monotonic() - started)
        _write(directory / 'receipt.json', receipt)
    return receipt
