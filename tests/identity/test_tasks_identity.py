"""Task copies preserve scores and content identity across independent roots.

The reference values are authored scalar expectations, not a retired checkout.
Only constant fixture programs execute; there is no model or numerical search.
"""
import importlib.util
from pathlib import Path
import shutil

import pytest

from sciel.eval.registry import load_task

ROOT = Path(__file__).resolve().parents[2]
TASK = ROOT / "tasks/oe/function_minimization"

# Fixed values derived independently with scalar math.sin/cos/hypot and the
# declared stage weights. Neither copied evaluator supplies these expectations.
FIXTURES = {
    "perfect": ((-1.704, .678, -1.5186854322511936), 1.3997358461852736, 1.4997641483797084),
    "origin_2tuple": ((0., 0.), .3793364943308036, .5043515172348038),
    "far": ((5., 5., 2.095637694457542), .13966578319158848, .23924023098403077),
    "region": ((-1., .5, -1.1553858012083318), .7724965754155833, .8885632524790412),
}


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def evaluators(tmp_path_factory):
    root = tmp_path_factory.mktemp('portable-task-roots')
    paths = [root / 'checkout/tasks/oe/function_minimization',
             root / 'external-bundle/oe/function_minimization']
    for path in paths:
        shutil.copytree(TASK, path, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    specs = [load_task(str(path)) for path in paths]
    assert specs[0].root != specs[1].root
    assert specs[0].task_hash == specs[1].task_hash == load_task(str(TASK)).task_hash
    assert specs[0].qualified_name == specs[1].qualified_name == 'oe/function_minimization'
    modules = [_load(path / 'evaluator.py', f'portable_function_evaluator_{i}')
               for i, path in enumerate(paths)]
    assert [Path(module.__file__).parent for module in modules] == paths
    return modules


def _source(returned):
    return f'def run_search():\n    return {returned!r}\n'


def _compare(key, stage, evaluators, tmp_path):
    returned, smoke_score, full_score = FIXTURES[key]
    expected = smoke_score if stage == 'smoke' else full_score
    program = tmp_path / 'constant_fixture.py'
    program.write_text(_source(returned))
    results = [getattr(evaluator, stage)(str(program)) for evaluator in evaluators]
    for result in results:
        assert result['ok'], result.get('artifacts')
        assert result['score'] == pytest.approx(expected, rel=0, abs=1e-9)
        assert result['cases']['n_trials'] == (1 if stage == 'smoke' else 10)
    # Equal hashes alone do not establish that both imported roots behave alike.
    assert results[0] == results[1]


@pytest.mark.parametrize("key", list(FIXTURES))
def test_full_matches_reference(key, evaluators, tmp_path):
    _compare(key, 'full', evaluators, tmp_path)


@pytest.mark.parametrize("key", list(FIXTURES))
def test_smoke_matches_reference(key, evaluators, tmp_path):
    _compare(key, 'smoke', evaluators, tmp_path)


def test_reported_value_is_not_trusted(evaluators, tmp_path):
    """Both copies score coordinates, ignoring a false third return value."""
    returned, _, expected = FIXTURES['region']
    honest, liar = tmp_path / 'honest.py', tmp_path / 'liar.py'
    honest.write_text(_source(returned))
    liar.write_text(_source((*returned[:2], -99999.)))
    for evaluator in evaluators:
        a, b = evaluator.full(str(honest)), evaluator.full(str(liar))
        assert a['ok'] and b['ok']
        assert a['score'] == pytest.approx(expected, rel=0, abs=1e-9)
        assert b['score'] == pytest.approx(a['score'], rel=0, abs=1e-9), 'the lie changed the score'
