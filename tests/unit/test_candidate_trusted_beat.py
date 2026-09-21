"""Inert arrays only: exact-event oracle and pinned trusted evaluator parity."""
import ast
from fractions import Fraction
import importlib.util
from itertools import product
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from sciel.eval.trusted_scoring import beat


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def legacy():
    spec = importlib.util.spec_from_file_location("inert_legacy_beat", ROOT / "tasks/math/beat_the_average/evaluator.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def legacy_score(legacy, monkeypatch, support, weights):
    # The evaluator module is trusted source. Its candidate import is replaced
    # before invocation; neither this label nor any candidate file is opened.
    monkeypatch.setattr(legacy, "_load", lambda *_: SimpleNamespace(
        beat_the_average_measure=lambda: (support, weights)))
    result = legacy._evaluate("never-opened-inert-label", "inert-module")
    return {key: result[key] for key in ("ok", "score", "cases")}


def exact_event_probability(support, weights):
    points = [Fraction(float(x)) for x in support]
    mass = [Fraction(float(w)) for w in weights]
    total = sum(mass)
    mass = [w / total for w in mass]
    probability = Fraction(0)
    for a, b, c, d in product(range(len(points)), repeat=4):
        if points[a] + points[b] + points[c] < 2 * points[d]:
            probability += mass[a] * mass[b] * mass[c] * mass[d]
    return probability


SMALL_MEASURES = [
    ([0.0], [1.0]),
    ([1.0], [1.0]),
    ([0.0, 1.0], [1.0, 1.0]),
    ([0.0, 1.0, 2.0], [1.0, 3.0, 7.0]),
    ([2.0, 0.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]),
    ([-0.0, 0.0, 1.0], [0.0, 1.0, 1.0]),
    ([0.0, np.nextafter(0.0, 1.0), 1.0], [1.0, 1.0, 1.0]),
    ([0.0, np.nextafter(1.0, 2.0), 2.0], [1.0, 2.0, 1.0]),
    ([0.0, np.nextafter(1.0, 0.0), 2.0], [1.0, 2.0, 1.0]),
    ([0.0, 1e-300, 1e300], [1.0, 2.0, 3.0]),
    ([0.0, np.finfo(float).max / 2, np.finfo(float).max], [1.0, 1.0, 1.0]),
    ([0.0, 1.0, 2.0], [np.finfo(float).max] * 3),
    ([0.0, 1.0, 2.0], [np.nextafter(0.0, 1.0)] * 3),
    ([0.0, 1.0, 2.0], [np.nextafter(0.0, 1.0), 1.0, np.finfo(float).max]),
]


@pytest.mark.parametrize("support,weights", SMALL_MEASURES)
def test_small_measures_match_fraction_oracle_and_exact_legacy(
        legacy, monkeypatch, support, weights):
    support = np.array(support, dtype=np.float64)
    weights = np.array(weights, dtype=np.float64)
    old_support, old_weights = support.copy(), weights.copy()
    result = beat.score(support, weights)
    assert result == legacy_score(legacy, monkeypatch, support, weights)
    assert result["cases"]["success_prob"] == pytest.approx(
        float(exact_event_probability(support, weights)), rel=3e-14, abs=3e-15)
    assert result["score"] == result["cases"]["success_prob"] / beat.BENCHMARK
    assert set(result) == {"ok", "score", "cases"}
    assert set(result["cases"]) == {"success_prob", "combined_score", *beat._BEHAVIOR_ZERO}
    assert result["ok"] is True
    assert all(np.isfinite(value) for value in result["cases"].values())
    assert np.array_equal(support, old_support) and np.array_equal(weights, old_weights)


def test_strict_ties_exclude_equality_mass():
    # For uniform {0,1}, only X4=1 and at most one of the first three=1 wins:
    # (1+3)/16. Replacing < with <= would give (1+7)/16.
    assert exact_event_probability([0.0, 1.0], [1.0, 1.0]) == Fraction(1, 4)
    assert beat.score([0.0, 1.0], [1.0, 1.0])["cases"]["success_prob"] == 0.25


@pytest.mark.parametrize("scale", [1e-300, 1.0, 1e308])
def test_nonnegative_weight_scale_does_not_overflow(scale):
    support = np.array([0.0, 1.0, 2.0, 3.0])
    weights = np.array([0.5, 1.0, 0.25, 0.75])
    expected = beat.score(support, weights)
    result = beat.score(support, weights * scale)
    assert result["score"] == pytest.approx(expected["score"], rel=1e-14)
    assert 0 <= result["cases"]["success_prob"] <= 1


@pytest.mark.parametrize("support,weights,message", [
    ([], [], "at least one"),
    ([0.0], [1.0, 2.0], "same shape"),
    ([-np.nextafter(0.0, 1.0), 1.0], [1.0, 1.0], "Support points"),
    ([0.0, 1.0], [-np.nextafter(0.0, 1.0), 1.0], "non-negative"),
    ([0.0, 1.0], [0.0, 0.0], "positive"),
    ([float("inf")], [1.0], "NaN or infinite"),
    ([float("nan")], [1.0], "NaN or infinite"),
    ([0.0], [float("inf")], "NaN or infinite"),
    ([0.0], [float("nan")], "NaN or infinite"),
    ([0.0] * 2001, [1.0] * 2001, "limit is 2000"),
])
def test_domain_rejections_match_legacy(legacy, monkeypatch, support, weights, message):
    with pytest.raises(ValueError, match=message) as actual:
        beat.score(support, weights)
    with pytest.raises(ValueError) as expected:
        legacy_score(legacy, monkeypatch, support, weights)
    assert str(actual.value) == str(expected.value)


def test_support_cap_precedes_quadratic_scorer(monkeypatch):
    def forbidden(*_):
        pytest.fail("Oversized support reached quadratic scorer")
    monkeypatch.setattr(beat, "_success_probability", forbidden)
    with pytest.raises(ValueError, match="limit is 2000"):
        beat.score(np.zeros(2001), np.ones(2001))


def test_double_normalization_and_first_normalized_behavior_are_preserved(monkeypatch):
    original_normalize, original_behavior = beat._normalized_weights, beat._behavior
    normalized, behavior_inputs = [], []
    def normalize(weights):
        normalized.append(np.array(weights, copy=True))
        return original_normalize(weights)
    def behavior(support, weights):
        behavior_inputs.append(np.array(weights, copy=True))
        return original_behavior(support, weights)
    monkeypatch.setattr(beat, "_normalized_weights", normalize)
    monkeypatch.setattr(beat, "_behavior", behavior)
    weights = np.array([1.0, 3.0, 7.0])
    beat.score(np.array([0.0, 1.0, 2.0]), weights)
    first = original_normalize(weights)
    assert len(normalized) == 2 and len(behavior_inputs) == 1
    assert np.array_equal(normalized[0], weights)
    assert np.array_equal(normalized[1], first)
    assert np.array_equal(behavior_inputs[0], first)


@pytest.mark.parametrize("probability", [float("nan"), float("inf"), -0.01, 1.01])
def test_nonphysical_probability_is_rejected(monkeypatch, probability):
    monkeypatch.setattr(beat, "_success_probability", lambda *_: probability)
    with pytest.raises(ValueError, match="Success probability"):
        beat.score([0.0, 1.0], [1.0, 1.0])


@pytest.mark.parametrize("probability,clipped", [(-5e-13, 0.0), (1.0 + 5e-13, 1.0)])
def test_legacy_probability_roundoff_clip_preserved(monkeypatch, probability, clipped):
    monkeypatch.setattr(beat, "_success_probability", lambda *_: probability)
    assert beat.score([0.0, 1.0], [1.0, 1.0])["cases"]["success_prob"] == clipped


def test_pure_helpers_are_verbatim_ast_and_no_candidate_loader_exists():
    old = ast.parse((ROOT / "tasks/math/beat_the_average/evaluator.py").read_text())
    new = ast.parse((ROOT / "src/sciel/eval/trusted_scoring/beat.py").read_text())
    old_functions = {node.name: node for node in old.body if isinstance(node, ast.FunctionDef)}
    new_functions = {node.name: node for node in new.body if isinstance(node, ast.FunctionDef)}
    helpers = {"_success_probability", "_wide_success_probability", "_normalized_weights",
               "_squash", "_finite", "_behavior"}
    assert set(new_functions) == helpers | {"score"}
    for name in helpers:
        assert ast.dump(new_functions[name]) == ast.dump(old_functions[name])
    assert {name.name for node in new.body if isinstance(node, ast.Import)
            for name in node.names} == {"heapq", "math", "numpy"}
    assert not any(isinstance(node, ast.ImportFrom) for node in new.body)


@pytest.fixture(autouse=True)
def forbid_candidate_import(monkeypatch):
    from sciel import candidate_modules
    def forbidden(*args, **kwargs):
        raise AssertionError("Only inert evaluator parity fixtures may run")
    monkeypatch.setattr(candidate_modules, "load_candidate", forbidden)
