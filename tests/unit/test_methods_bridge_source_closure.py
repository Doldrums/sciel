"""Execute a bounded authored toy from a standalone packaged cascade.

No Docker, providers or scientific candidates. The outer interpreter and each
real evaluator stage must import SCiEL exclusively from the materialized bundle.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from sciel.eval.registry import load_task
from sciel.methods.engine_bridge import EngineCascadeBridge
from sciel.methods.models import content_hash


CANDIDATE = '''from dataclasses import dataclass
import multiprocessing
from pathlib import Path
import sciel.candidate_modules

@dataclass
class Value:
    amount: int

def saved_value():
    return Value(3).amount, str(Path(sciel.candidate_modules.__file__).resolve())

def solve():
    # A new interpreter must import the exact dynamically named candidate too.
    with multiprocessing.get_context("spawn").Pool(1) as pool:
        return pool.apply(saved_value)
'''

EVALUATOR = '''import json
from pathlib import Path
import sys
import sciel
import sciel.candidate_modules
from sciel.candidate_modules import load_candidate

def trial(path, stage):
    bundle = Path(__file__).resolve().parents[1]
    assert list(sciel.__path__) == [str(bundle / "sciel")]
    candidate = load_candidate(path, "toy_" + stage)
    value, worker_loader = candidate.solve()
    assert value == 3
    assert worker_loader == str(bundle / "sciel/candidate_modules/__init__.py")
    imported = {name: str(Path(module.__file__).resolve())
                for name, module in tuple(sys.modules.items())
                if (name == "sciel" or name.startswith("sciel."))
                and getattr(module, "__file__", None)}
    assert imported
    assert all(Path(value).is_relative_to(bundle) for value in imported.values())
    with (Path.cwd() / "stage-events.jsonl").open("a") as stream:
        stream.write(json.dumps({"stage": stage, "imports": imported}) + "\\n")
    return {"ok": True, "score": float(value), "cases": {"toy_value": float(value)},
            "artifacts": {"spawn_loader": worker_loader}}

def smoke(path):
    return trial(path, "smoke")

def full(path):
    return trial(path, "full")

def confirm(path, n):
    results = [trial(path, "confirm") for _ in range(n)]
    result = min(results, key=lambda row: row["score"])
    result["cases"]["n_trials"] = float(n)
    return result
'''


DRIVER = '''import importlib, importlib.util, json, sys
from pathlib import Path
bundle = Path(sys.argv[1]).resolve()
sys.path.insert(0, str(bundle))
import sciel
assert list(sciel.__path__) == [str(bundle / "sciel")]
# Import every packaged runtime module, including geometry's NumPy dependency.
for name in ("cascade", "facade", "sandbox", "_runner", "native_geometry", "packing_geometry"):
    importlib.import_module("sciel.eval." + name)
loader_path = bundle.parent / "runtime/_loader.py"
spec = importlib.util.spec_from_file_location("_fixture_method_loader", loader_path)
loader = importlib.util.module_from_spec(spec)
spec.loader.exec_module(loader)
invoke = loader.make_method_loader({"fixture": {"root": str(bundle), "entrypoint": "adapter.py:run"}})
output = invoke("fixture", {"record_candidate": sys.argv[2] == "true"})
imports = {name: str(Path(module.__file__).resolve())
           for name, module in tuple(sys.modules.items())
           if (name == "sciel" or name.startswith(("sciel.", "_sciel_bundle_")))
           and getattr(module, "__file__", None)}
assert all(Path(value).is_relative_to(bundle) for value in imports.values())
print(json.dumps({"output": output, "imports": imports}))
'''


def assembled(tmp_path):
    task = tmp_path / 'authored-task'
    task.mkdir()
    (task / 'task.yaml').write_text(textwrap.dedent('''\
        suite: fixture
        name: source_closure
        origin: math
        primary_metric: score
        stochastic: true
        confirm_n: 3
        smoke_timeout: 8
        full_timeout: 8
        confirm_timeout: 12
        '''))
    (task / 'evaluator.py').write_text(EVALUATOR)
    (task / 'initial_program.py').write_text(CANDIDATE)
    bridge = EngineCascadeBridge(load_task(str(task)))
    bundle = bridge.solution_bundle(CANDIDATE, 'fixture',
        {'image_id': 'sha256:' + 'a' * 64},
        dict(wall_seconds=30, cpu_seconds=10, memory_bytes=268435456,
             pids=16, max_output_bytes=131072, max_stderr_bytes=4096))
    packaged = tmp_path / 'standalone-bundle'
    for name, text in bundle['files'].items():
        path = packaged / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    runtime = tmp_path / 'runtime'
    runtime.mkdir()
    loader = Path(__file__).resolve().parents[2] / 'src/sciel/methods/_loader.py'
    (runtime / '_loader.py').write_bytes(loader.read_bytes())
    work = tmp_path / 'outside-repository'
    work.mkdir()
    # An intentionally unusable ambient source must not supply missing modules.
    ambient = tmp_path / 'ambient/sciel/candidate_modules'
    ambient.mkdir(parents=True)
    (ambient.parent / '__init__.py').write_text('raise RuntimeError("ambient sciel imported")\n')
    (ambient / '__init__.py').write_text('raise RuntimeError("ambient loader imported")\n')
    env = {'PATH': os.defpath, 'PYTHONPATH': str(ambient.parents[1]),
           'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONNOUSERSITE': '1',
           'OPENBLAS_NUM_THREADS': '1', 'OMP_NUM_THREADS': '1'}
    return bridge, bundle, packaged, work, env


def invoke(packaged, work, env, record):
    result = subprocess.run([sys.executable, '-I', '-B', '-c', DRIVER,
                             str(packaged), str(record).lower()],
        cwd=work, env=env, stdin=subprocess.DEVNULL, close_fds=True,
        capture_output=True, text=True, timeout=35)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


@pytest.mark.parametrize('record', [False, True])
def test_standalone_bundle_executes_complete_runtime_closure_and_spawn(tmp_path, record):
    bridge, bundle, packaged, work, env = assembled(tmp_path)
    returned = invoke(packaged, work, env, record)
    outcome = bridge.extract(returned['output'],
        expected_source_sha256=hashlib.sha256(CANDIDATE.encode()).hexdigest())
    assert outcome.result.ok, outcome.result.to_json()
    assert outcome.result.score == outcome.raw_first_score == 3.0
    assert outcome.reached == ('confirm' if record else 'full')
    assert outcome.confirmed is record
    if record:
        assert outcome.result.cases['n_trials'] == 3.0
        assert outcome.result.confirmed_sampled
    events = [json.loads(line) for line in (work / 'stage-events.jsonl').read_text().splitlines()]
    assert [row['stage'] for row in events] == ['smoke', 'full'] + (['confirm'] * 3 if record else [])
    assert all('sciel.candidate_modules' in row['imports'] for row in events)
    assert all(any(name.startswith('sciel.candidate_modules._p') for name in row['imports'])
               for row in events)
    expected_loader = Path(__file__).resolve().parents[2] / 'src/sciel/candidate_modules/__init__.py'
    assert bundle['files']['sciel/candidate_modules/__init__.py'].encode() == expected_loader.read_bytes()
    assert content_hash(bridge._files) == bridge.identity['runtime_sources_sha256']
    assert returned['output']['task_identity_sha256'] == content_hash(bridge.identity)


def test_missing_packaged_loader_fails_closed_without_ambient_fallback(tmp_path):
    bridge, _bundle, packaged, work, env = assembled(tmp_path)
    (packaged / 'sciel/candidate_modules/__init__.py').unlink(missing_ok=True)
    # Removing the directory prevents creation of a namespace-only package.
    directory = packaged / 'sciel/candidate_modules'
    if directory.exists():
        directory.rmdir()
    returned = invoke(packaged, work, env, False)
    outcome = bridge.extract(returned['output'])
    assert outcome.reached == 'smoke' and not outcome.result.ok
    assert outcome.result.score == 0.0 and not outcome.confirmed
    assert outcome.result.artifacts['error_type'] == 'ModuleNotFoundError'
    assert 'sciel.candidate_modules' in outcome.result.stderr
    assert 'ambient' not in outcome.result.stderr
    assert not (work / 'stage-events.jsonl').exists()
