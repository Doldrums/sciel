"""Exact small-measure oracle; the candidate loader is always inert."""
from fractions import Fraction
from itertools import product
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tracemalloc

import numpy as np
import pytest


@pytest.fixture
def evaluator():
    path = Path(__file__).resolve().parents[2] / 'tasks/math/beat_the_average/evaluator.py'
    spec = importlib.util.spec_from_file_location('measure_support_fixture', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def measure(evaluator, monkeypatch, support, weights):
    monkeypatch.setattr(evaluator, '_load', lambda *_: SimpleNamespace(
        beat_the_average_measure=lambda: (support, weights)))
    return evaluator.full('never-loaded.py')


def exact_probability(support, weights):
    points = [Fraction(float(x)) for x in support]
    mass = [Fraction(float(x)) for x in weights]
    total = sum(mass)
    mass = [w / total for w in mass]
    return sum((mass[i] * mass[j] * mass[k] * mass[m]
                for i, j, k, m in product(range(len(points)), repeat=4)
                if points[i] + points[j] + points[k] < 2 * points[m]), Fraction())


@pytest.mark.parametrize('negative', [-5e-10, -np.nextafter(0., 1.), -1.])
def test_every_negative_support_is_rejected(evaluator, monkeypatch, negative):
    result = measure(evaluator, monkeypatch, [negative], [1.])
    assert result['ok'] is False
    assert result['score'] == 0
    assert '[0, inf)' in result['artifacts']['error']


@pytest.mark.parametrize('zero', [0., -0.])
def test_zero_atom_is_valid_but_never_wins(evaluator, monkeypatch, zero):
    result = measure(evaluator, monkeypatch, [zero], [1.])
    assert result['ok'] is True
    assert result['cases']['success_prob'] == 0


@pytest.mark.parametrize('support,weights', [
    ([6e307, 1e308], [1., 1.]),
    ([0., np.finfo(float).max], [1., 3.]),
    ([0., np.nextafter(0., 1.), np.finfo(float).max], [1., 2., 3.]),
    ([np.nextafter(0., 1.), 2 * np.nextafter(0., 1.)], [1., 1.]),
    ([0., 1., 2.], [1., 2., 4.]),
    ([0., np.nextafter(1., 0.), 1., np.nextafter(1., 2.)], [1., 2., 3., 4.]),
    ([.1, .2, .3], [2., 3., 5.]),
    ([0., 0., 1., 1.], [1., 2., 3., 4.]),
    ([0., np.nextafter(0., 1.), 1e308], [0., 1., 0.]),
])
def test_finite_measure_probability_matches_exact_binary64_event(
        evaluator, monkeypatch, support, weights):
    expected = float(exact_probability(support, weights))
    for points, mass in ((support, weights), (support[::-1], weights[::-1])):
        result = measure(evaluator, monkeypatch, points, mass)
        assert result['ok'] is True, result['artifacts']
        assert result['cases']['success_prob'] == pytest.approx(expected, abs=2e-15)


@pytest.mark.parametrize('exponent', [-1074, -900, 0, 900, 1022])
def test_exact_power_of_two_rescaling_keeps_strict_event(evaluator, exponent):
    support = np.ldexp(np.array([0., 1., 2., 3.]), exponent)
    weights = [1., 2., 3., 4.]
    assert evaluator._success_probability(support, weights) == pytest.approx(
        float(exact_probability([0., 1., 2., 3.], weights)), abs=2e-15)


def test_fixed_tiny_random_measures_match_independent_exact_enumeration(evaluator):
    rng = np.random.default_rng(738)
    for _ in range(20):
        support = np.ldexp(rng.random(4), rng.integers(-1070, 1024, size=4))
        weights = rng.integers(1, 9, size=4)
        assert evaluator._success_probability(support, weights) == pytest.approx(
            float(exact_probability(support, weights)), abs=2e-15)


def test_wide_exponent_path_does_not_allocate_quadratic_big_integer_storage(evaluator):
    # Repeat a known three-atom measure to increase n without changing its law.
    # Its full exponent range would make an n^2 big-integer array very large.
    atoms = [0., np.nextafter(0., 1.), np.finfo(float).max]
    support = atoms * 64
    expected = float(exact_probability(atoms, [1., 1., 1.]))
    tracemalloc.start()
    try:
        result = evaluator._success_probability(support, np.ones(len(support)))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert result == pytest.approx(expected, abs=2e-15)
    assert peak < 2 * 1024 * 1024
