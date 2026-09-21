"""Small deterministic fixtures; no candidate imports or subprocess execution."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def evaluator(task):
    path = Path(__file__).resolve().parents[2] / 'tasks' / task / 'evaluator.py'
    spec = importlib.util.spec_from_file_location('numerical_evaluator', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measure_result(monkeypatch, weights):
    ev = evaluator('math/beat_the_average')
    monkeypatch.setattr(ev, '_load', lambda *_: SimpleNamespace(
        beat_the_average_measure=lambda: (np.arange(len(weights)), weights)))
    return ev.full('not loaded')


@pytest.mark.parametrize('weights', [
    [-1e-9, -1e-9, 0., 3.1e-9], [-1e-300, 1.], [0., 0.], [np.inf, 1.], [np.nan, 1.],
])
def test_measure_rejects_negative_and_invalid_mass(monkeypatch, weights):
    result = measure_result(monkeypatch, weights)
    assert result['ok'] is False
    assert result['score'] == 0


@pytest.mark.parametrize('scale', [1e-300, 1., 1e308])
def test_valid_measure_is_scale_invariant_and_probability_bounded(monkeypatch, scale):
    weights = np.array([.5, 1., .25, .75])
    reference = measure_result(monkeypatch, weights)
    result = measure_result(monkeypatch, weights * scale)
    assert result['ok'] and reference['ok']
    assert result['score'] == pytest.approx(reference['score'])
    assert 0 <= result['cases']['success_prob'] <= 1
    assert all(np.isfinite(v) for v in result['cases'].values())


@pytest.mark.parametrize('probability', [float('nan'), float('inf'), -.01, 1.01])
def test_out_of_range_probability_is_a_failed_evaluation(monkeypatch, probability):
    ev = evaluator('math/beat_the_average')
    monkeypatch.setattr(ev, '_load', lambda *_: SimpleNamespace(
        beat_the_average_measure=lambda: ([0., 1.], [1., 1.])))
    monkeypatch.setattr(ev, '_success_probability', lambda *_: probability)
    assert ev.full('not loaded')['ok'] is False


def geometry_result(monkeypatch, points, tmp_path):
    ev = evaluator('oe/heilbronn_triangle')
    monkeypatch.setattr(ev, '_load', lambda *_: SimpleNamespace(heilbronn_triangle11=lambda: points))
    source = tmp_path / 'inert_geometry_source.py'
    source.write_text('# Known geometry is supplied by the mocked loader; never executed.\n')
    return ev.full(str(source))


def test_float16_collinear_triple_cannot_earn_a_positive_area(monkeypatch, tmp_path):
    collinear = [[.544921875, .3974609375], [.673828125, .4619140625], [.330078125, .2900390625]]
    extra = np.random.default_rng(9).uniform([.35, .1], [.65, .4], size=(8, 2))
    points = np.array([*collinear, *extra], dtype=np.float16)
    for candidate in (points, points[::-1], points.astype(np.float64)):
        result = geometry_result(monkeypatch, candidate, tmp_path)
        assert result['ok']
        assert result['score'] == 0
        assert result['cases']['min_area_normalized'] == 0


def test_float16_geometry_matches_float64_and_permutations(monkeypatch, tmp_path):
    points = np.random.default_rng(21).uniform([.35, .1], [.65, .4], size=(11, 2)).astype(np.float16)
    results = [geometry_result(monkeypatch, candidate, tmp_path) for candidate in
               (points, points[::-1], points[[3, 0, 9, 5, 8, 1, 7, 10, 2, 6, 4]], points.astype(np.float64))]
    assert all(result['ok'] for result in results)
    assert results[0]['score'] > 0
    assert all(result['score'] == results[0]['score'] for result in results)
