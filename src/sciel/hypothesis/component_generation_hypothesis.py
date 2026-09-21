"""A prospective component contrast, without an invented native score interval.

This wire is deliberately separate from the ordinary ``Hypothesis`` codec.
Decoding never repairs prose, strips fences, executes a body or measures a gate.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re

from . import restart_predictions
from .parsing import _unique_object, _reject_constant

SCHEMA = 'sciel-component-generation-hypothesis/1'
FORECAST_SCHEMA = 'sciel-component-quality-forecast/1'
MAX_RESPONSE_BYTES = 32768
MAX_TEXT_CHARS = 2048
MAX_REASONING_CHARS = 4096
MAX_ABSTENTION_CHARS = 1024
MAX_BODY_BYTES = 8192


def _sha(value):
    return type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None


def _text(value, limit):
    return type(value) is str and 0 < len(value) <= limit and bool(value.strip())


@dataclass(frozen=True)
class ComponentGenerationHypothesis:
    text: str
    reasoning: str
    confidence: float
    target_sha256: str
    expected_gate_pass: bool
    restart_expected: tuple[bool, bool, bool]

    def to_dict(self):
        return dict(text=self.text, reasoning=self.reasoning, confidence=self.confidence,
            paired_forecast=dict(schema=FORECAST_SCHEMA, target_sha256=self.target_sha256,
                                 expected_gate_pass=self.expected_gate_pass),
            restart_predictions=dict(schema=restart_predictions.SCHEMA,
                expected=dict(zip(restart_predictions.NAMES, self.restart_expected))))


def validate_hypothesis(value, *, target_sha256):
    """Validate an authored forecast; preserve original prose and numeric value."""
    if not _sha(target_sha256):
        raise ValueError('component hypothesis: invalid target binding')
    if type(value) is not dict or set(value) != {
            'text', 'reasoning', 'confidence', 'paired_forecast', 'restart_predictions'}:
        raise ValueError('component hypothesis: invalid fields')
    if not _text(value['text'], MAX_TEXT_CHARS) or not _text(value['reasoning'], MAX_REASONING_CHARS):
        raise ValueError('component hypothesis: invalid bounded prose')
    confidence = value['confidence']
    if type(confidence) not in (int, float) or not math.isfinite(confidence) or not 0 <= confidence <= 1:
        raise ValueError('component hypothesis: invalid confidence')
    forecast = value['paired_forecast']
    if (type(forecast) is not dict or set(forecast) != {'schema', 'target_sha256', 'expected_gate_pass'}
            or forecast['schema'] != FORECAST_SCHEMA or forecast['target_sha256'] != target_sha256
            or type(forecast['expected_gate_pass']) is not bool):
        raise ValueError('component hypothesis: invalid paired forecast')
    spec = restart_predictions.validate_spec(value['restart_predictions'])
    return ComponentGenerationHypothesis(value['text'], value['reasoning'], confidence,
        target_sha256, forecast['expected_gate_pass'], tuple(spec['expected'][key] for key in restart_predictions.NAMES))


def decode_response(raw, *, target_sha256):
    """Return a detached proposal or safe diagnostic, never rejected model prose."""
    result = dict(status='invalid', body=None, hypothesis=None, abstention_reason=None,
                  error=dict(code='invalid_response', path='response'))
    try:
        if not _sha(target_sha256):
            raise ValueError('invalid target')
        if type(raw) is not str or len(raw.encode('utf-8')) > MAX_RESPONSE_BYTES:
            result['error'] = dict(code='response_bound', path='response')
            return result
        value = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
        if type(value) is not dict or set(value) != {'target_sha256', 'body', 'hypothesis', 'abstention_reason'}:
            return result
        if value['target_sha256'] != target_sha256:
            result['error'] = dict(code='target_binding', path='target_sha256')
            return result
        if value['body'] is None and value['hypothesis'] is None:
            if not _text(value['abstention_reason'], MAX_ABSTENTION_CHARS):
                result['error'] = dict(code='invalid_abstention', path='abstention_reason')
                return result
            return dict(status='abstained', body=None, hypothesis=None,
                        abstention_reason=value['abstention_reason'], error=None)
        if (value['abstention_reason'] is not None or type(value['body']) is not str
                or not value['body'].strip() or len(value['body'].encode('utf-8')) > MAX_BODY_BYTES):
            result['error'] = dict(code='invalid_proposal', path='body')
            return result
        try:
            hypothesis = validate_hypothesis(value['hypothesis'], target_sha256=target_sha256)
        except (ValueError, TypeError, OverflowError):
            result['error'] = dict(code='invalid_hypothesis', path='hypothesis')
            return result
        return dict(status='proposed', body=value['body'], hypothesis=hypothesis,
                    abstention_reason=None, error=None)
    except (ValueError, TypeError, UnicodeError, RecursionError, OverflowError):
        return result


def response_format(target_sha256):
    if not _sha(target_sha256):
        raise ValueError('component hypothesis: invalid target binding')
    forecast = dict(type='object', additionalProperties=False,
        required=['schema', 'target_sha256', 'expected_gate_pass'], properties={
            'schema':dict(type='string', enum=[FORECAST_SCHEMA]),
            'target_sha256':dict(type='string', enum=[target_sha256]),
            'expected_gate_pass':dict(type='boolean')})
    hypothesis = dict(type=['object', 'null'], additionalProperties=False,
        required=['text', 'reasoning', 'confidence', 'paired_forecast', 'restart_predictions'],
        properties=dict(text=dict(type='string', description='Nonblank, at most 2048 characters.'),
            reasoning=dict(type='string', description='Nonblank, at most 4096 characters; unverified explanation.'),
            confidence=dict(type='number', description='Finite number from 0 to 1 inclusive.'),
            paired_forecast=forecast, restart_predictions=restart_predictions.json_schema()))
    return dict(type='json_schema', json_schema=dict(name='sciel_component_target_generation_v1', strict=True,
        schema=dict(type='object', additionalProperties=False,
            required=['target_sha256', 'body', 'hypothesis', 'abstention_reason'], properties={
                'target_sha256':dict(type='string', enum=[target_sha256]),
                'body':dict(type=['string', 'null'], description='Complete editable body, at most 8192 UTF-8 bytes; no fences.'),
                'hypothesis':hypothesis,
                'abstention_reason':dict(type=['string', 'null'], description=
                    'When abstaining, nonblank at most 1024 characters and body/hypothesis both null. '
                    'When proposing, null and body/hypothesis both present.')})))


INSTRUCTION = (
    'Return one bare JSON object matching the supplied schema. Edit the declared base body '
    'to implement the selected target; the comparator is a future paired evaluation control, '
    'not the edit parent. Return the complete editable body, never scaffold code. '
    'If no justified implementation is possible, return body=null and hypothesis=null with '
    'a concrete abstention_reason (at most 1024 characters). Otherwise abstention_reason=null. '
    'Preserve target_sha256 in the envelope and paired_forecast. The paired forecast is a '
    'boolean prediction of the fixed three-pair raw/exact quality gate against the declared '
    'comparator, not a native score or a measured result. There is no measured native parent '
    'score and no numeric native forecast interval. Prose, confidence and target fidelity '
    'remain unverified. Do not assert a gain or causal mechanism from source binding alone. '
    'This request authorizes no solver, evaluation, promotion, lesson or research outcome. '
)
