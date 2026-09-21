"""Strict adapter status validation without importing an evaluator or candidate."""
from types import SimpleNamespace
import json

import numpy as np
import pytest

from sciel.eval import _runner
from sciel.eval.facade import EvalResult


@pytest.mark.parametrize('value', [None, 'false', '0', 0, 1, [], [False], {}, {'failed': True}, np.bool_(True)])
@pytest.mark.parametrize('shape', ['score', 'metrics_mapping', 'metrics_object', 'typed'])
def test_explicit_status_requires_bool_in_every_adapter_shape(value, shape):
    if shape == 'score':
        raw = {'ok': value, 'score': .9}
    elif shape == 'metrics_mapping':
        raw = {'ok': value, 'metrics': {'combined_score': .9}}
    elif shape == 'metrics_object':
        raw = SimpleNamespace(ok=value, metrics={'combined_score': .9})
    else:
        raw = EvalResult(value, .9)
    with pytest.raises(ValueError, match='ok.*bool'):
        _runner._normalize(raw)


@pytest.mark.parametrize('value', [True, False])
@pytest.mark.parametrize('raw', ['score', 'metrics_mapping', 'metrics_object'])
def test_boolean_status_is_preserved(value, raw):
    result = ({'ok': value, 'score': .9} if raw == 'score' else
              {'ok': value, 'metrics': {'combined_score': .9}} if raw == 'metrics_mapping' else
              SimpleNamespace(ok=value, metrics={'combined_score': .9}))
    assert EvalResult.from_json(_runner._normalize(result)).ok is value


@pytest.mark.parametrize('raw', [
    {'score': .9}, {'combined_score': .9}, {'metrics': {'combined_score': .9}},
    SimpleNamespace(metrics={'combined_score': .9}), EvalResult(True, .9),
])
def test_absent_legacy_status_still_works(raw):
    assert EvalResult.from_json(_runner._normalize(raw)).ok is True


def test_explicit_true_does_not_override_reported_error():
    result = _runner._normalize({'ok': True, 'score': .9, 'artifacts': {'error': 'failed'}})
    assert result['ok'] is False


def test_malformed_status_becomes_a_failed_runner_envelope(tmp_path, monkeypatch):
    monkeypatch.setattr(_runner, '_apply_limits', lambda *_: None)
    monkeypatch.setattr(_runner, '_load', lambda _: SimpleNamespace(
        full=lambda _: {'ok': 'false', 'score': .9}))
    output = tmp_path / 'result.json'
    _runner.main(['runner', 'never-loaded', 'full', 'never-loaded', str(output)])
    result = EvalResult.from_json(json.loads(output.read_text()))
    assert result.ok is False and result.score == 0
    assert result.artifacts['error_type'] == 'ValueError'
    assert 'ok must be boolean' in result.stderr
