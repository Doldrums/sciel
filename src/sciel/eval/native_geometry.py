"""Validate saved native Circle observations without executing candidate code.

The original NumPy dtype arithmetic is authoritative for the native score.
Exact radius-decreased certification is an explicit, separate operation. Source
hashes bind recorded bytes; they do not authenticate execution or the producer.
"""
from copy import deepcopy
import hashlib
import json
import math
import re
import warnings

import numpy as np

from .packing_geometry import certify_packing

SCHEMA = 'sciel-native-circle-geometry/1'
VIEW_SCHEMA = 'sciel-native-circle-geometry-view/1'
CERTIFICATE_SCHEMA = 'sciel-native-circle-geometry-certificate/1'
MAX_ARTIFACT_BYTES = 16384
_FIELDS = {'schema', 'source_sha256', 'evaluator_sha256', 'invocation_id', 'stage',
           'confirmation', 'numeric', 'numeric_sha256', 'centers_dtype', 'radii_dtype',
           'native_validity', 'native_sum_radii', 'native_score'}
_VIEW_FIELDS = (_FIELDS - {'numeric'}) | {'artifact_sha256', 'arrays_omitted', 'certification'}
_DTYPES = {'float16', 'float32', 'float64'}


def _require(ok, message):
    if not ok:
        raise ValueError(message)


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True, allow_nan=False)


def _sha(value):
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def _hash(value, name, length=64):
    _require(type(value) is str and re.fullmatch('[0-9a-f]{' + str(length) + '}', value) is not None,
             'invalid ' + name)


def _float(value, name):
    _require(type(value) is float and math.isfinite(value), name + ' must be a finite binary64 JSON float')


def _equal_float(a, b):
    return type(a) is float and type(b) is float and a.hex() == b.hex()


def _finite_number(value):
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _metadata(value):
    for key in ('source_sha256', 'evaluator_sha256', 'numeric_sha256'):
        _hash(value[key], key)
    _hash(value['invocation_id'], 'invocation_id', 32)
    _require(type(value['stage']) is str and value['stage'] in ('smoke', 'full'), 'invalid native stage')
    confirmation = value['confirmation']
    if confirmation is not None:
        _require(type(confirmation) is dict and set(confirmation) == {'trials', 'selected_trial', 'selection'},
                 'invalid confirmation fields')
        _require(type(confirmation['trials']) is int and confirmation['trials'] >= 1
                 and type(confirmation['selected_trial']) is int
                 and 0 <= confirmation['selected_trial'] < confirmation['trials']
                 and confirmation['selection'] == 'first_failure_else_lowest_score'
                 and value['stage'] == 'full', 'invalid selected confirmation trial')
    for key in ('centers_dtype', 'radii_dtype'):
        _require(type(value[key]) is str and value[key] in _DTYPES, 'unsupported original numeric dtype')
    _require(type(value['native_validity']) is bool, 'native_validity must be boolean')
    for key in ('native_sum_radii', 'native_score'):
        _float(value[key], key)
    _require(value['native_sum_radii'] >= 0 and value['native_score'] >= 0, 'negative native metric')
    expected_sum = value['native_sum_radii'] if value['native_validity'] else 0.0
    _require(_equal_float(value['native_sum_radii'], expected_sum)
             and _equal_float(value['native_score'], expected_sum / 2.635), 'native metric relationship differs')


def _array(values, dtype):
    # JSON values must describe exactly the original dtype values, not values
    # that merely round/truncate to them. Preserve float signed zero as well.
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            array = np.array(values, dtype=dtype, copy=True)
    except (ValueError, TypeError, OverflowError, Warning) as exc:
        raise ValueError('numeric values cannot reconstruct original dtype') from exc
    originals = [x for row in values for x in row] if type(values[0]) is list else values
    for original, restored in zip(originals, array.flat):
        _require(_equal_float(original, float(restored)), 'inexact floating dtype representation')
    return array


def _native_valid(centers, radii):
    """The evaluator's exact operation order, including dtype promotion rules."""
    # Overflow warnings are diagnostics in the original evaluator, not a
    # changed arithmetic rule; all input elements have already been finite.
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        for i in range(26):
            if radii[i] < 0:
                return False
        for i in range(26):
            x, y = centers[i]
            r = radii[i]
            if x-r < -1e-6 or x+r > 1+1e-6 or y-r < -1e-6 or y+r > 1+1e-6:
                return False
        for i in range(26):
            for j in range(i+1, 26):
                dist = np.sqrt(np.sum((centers[i]-centers[j]) ** 2))
                if dist < radii[i]+radii[j]-1e-6:
                    return False
    return True


def validate_geometry(value):
    """Return detached full evidence, or ValueError; a view is not evidence."""
    _require(type(value) is dict and set(value) == _FIELDS and value['schema'] == SCHEMA,
             'invalid native geometry fields/schema')
    _metadata(value)
    numeric = value['numeric']
    _require(type(numeric) is dict and set(numeric) == {'centers', 'radii'}, 'invalid numeric fields')
    centers, radii = numeric['centers'], numeric['radii']
    _require(type(centers) is list and len(centers) == 26
             and all(type(row) is list and len(row) == 2 for row in centers)
             and type(radii) is list and len(radii) == 26, 'expected centers26x2 and radii26')
    for x in [v for row in centers for v in row] + radii:
        _float(x, 'coordinate/radius')
    _require(_sha(numeric) == value['numeric_sha256'], 'numeric digest differs')
    _require(len(_canonical(value).encode()) <= MAX_ARTIFACT_BYTES, 'native artifact exceeds byte cap')
    c, r = _array(centers, value['centers_dtype']), _array(radii, value['radii_dtype'])
    # Reproduce the evaluator bound to this evidence. Old records used candidate
    # dtype arithmetic; the corrected evaluator checks constraints in binary64.
    # Scoring still uses the recorded reduction dtype in both versions.
    from ..engine.repeat_support import LEGACY_GEOMETRY_EVALUATOR_SHA
    legacy = value['evaluator_sha256'] == LEGACY_GEOMETRY_EVALUATOR_SHA
    valid = _native_valid(c, r) if legacy else _native_valid(c.astype(np.float64), r.astype(np.float64))
    total = float(np.sum(r)) if valid else 0.0
    _require(math.isfinite(total) and value['native_validity'] is valid
             and _equal_float(value['native_sum_radii'], total)
             and _equal_float(value['native_score'], float(total / 2.635)), 'native dtype score/validity differs')
    return deepcopy(value)


def parse_artifact(text):
    """Parse only the canonical bounded JSON string emitted by the evaluator."""
    _require(type(text) is str, 'native artifact must be a JSON string')
    try:
        _require(len(text.encode('utf-8')) <= MAX_ARTIFACT_BYTES, 'native artifact exceeds byte cap')
        def unique(pairs):
            result = {}
            for key, item in pairs:
                _require(key not in result, 'duplicate artifact JSON key')
                result[key] = item
            return result
        def constant(_):
            raise ValueError('nonfinite artifact JSON number')
        value = json.loads(text, object_pairs_hook=unique, parse_constant=constant)
        validated = validate_geometry(value)
        _require(text == _canonical(validated), 'native artifact JSON must be canonical')
        return validated
    except (UnicodeError, RecursionError, OverflowError) as exc:
        raise ValueError('invalid native artifact JSON') from exc


def validate_view(value):
    """Check a presentation-only view; never establish missing numeric facts."""
    _require(type(value) is dict and set(value) == _VIEW_FIELDS and value['schema'] == VIEW_SCHEMA,
             'invalid native geometry view fields/schema')
    _metadata(value)
    _hash(value['artifact_sha256'], 'artifact_sha256')
    _require(value['arrays_omitted'] is True and value['certification'] == 'not_performed', 'invalid view scope')
    _require(len(_canonical(value).encode()) <= MAX_ARTIFACT_BYTES, 'native view exceeds byte cap')
    return deepcopy(value)


def view_geometry(geometry):
    value = validate_geometry(geometry)
    view = {k:v for k,v in value.items() if k != 'numeric'}
    view.update(schema=VIEW_SCHEMA, artifact_sha256=_sha(value), arrays_omitted=True, certification='not_performed')
    return validate_view(view)


def validate_binding(geometry, observation, source_sha256=None, metric=None):
    """Bind geometry to its selected native result, never to a repeat result.

    When supplied, the metric must match a registered task/evaluator tuple.
    A descriptor or source hash cannot authenticate an actual execution.
    Failed outer cascades may override the score while retaining the selected
    trial's cases (e.g. insufficient confirmation); this is not native success.
    """
    value = validate_geometry(geometry)
    _require(type(observation) is dict and type(observation.get('ok')) is bool
             and type(observation.get('confirmed')) is bool, 'invalid observation flags')
    reached = observation.get('reached')
    _require(type(reached) is str and reached in ('smoke', 'full', 'confirm'), 'invalid observation stage')
    confirmation = value['confirmation']
    if reached == 'confirm':
        _require(value['stage'] == 'full' and confirmation is not None, 'missing selected confirmation trial')
    else:
        _require(value['stage'] == reached and confirmation is None and observation['confirmed'] is False,
                 'native stage/confirmation differs')
    cases = observation.get('cases')
    _require(type(cases) is dict, 'invalid native observation cases')
    expected = {'sum_radii':value['native_sum_radii'], 'validity':float(value['native_validity']),
                'target_ratio':value['native_score'], 'combined_score':value['native_score']}
    if confirmation is not None:
        expected['n_trials'] = confirmation['trials']
    for key, amount in expected.items():
        actual = cases.get(key)
        _require(_finite_number(actual) and actual == amount,
                 'native observation case differs: ' + key)
    score = observation.get('score')
    _require(_finite_number(score), 'invalid outer observation score')
    if observation['ok']:
        _require(float(score) == value['native_score'], 'successful outer score differs from geometry')
    _require(not observation['confirmed'] or (observation['ok'] and confirmation is not None
                                             and confirmation['trials'] >= 2), 'invalid confirmed observation')
    if source_sha256 is not None:
        _hash(source_sha256, 'expected source_sha256')
        _require(value['source_sha256'] == source_sha256, 'native geometry source differs')
    if metric is not None:
        # Lazy import permits measurement's optional geometry projection without
        # creating a module initialization cycle.
        from ..evidence.measurement import validate_metric_context
        from ..engine.repeat_support import NATIVE_GEOMETRY_TASK_EVALUATORS
        descriptor = validate_metric_context(metric)
        _require(NATIVE_GEOMETRY_TASK_EVALUATORS.get(descriptor['task_hash']) == value['evaluator_sha256'],
                 'unregistered native evaluator/task tuple')
        semantics = descriptor['semantics']
        _require(descriptor['task_name'] == 'oe/circle_packing' and descriptor['primary_metric'] == 'combined_score'
                 and semantics.get('kind') == 'validity_gated_ratio'
                 and semantics.get('numerator_case') == 'sum_radii' and semantics.get('denominator') == 2.635
                 and semantics.get('validity_case') == 'validity' and semantics.get('valid_value') == 1.0
                 and semantics.get('invalid_score') == 0.0 and semantics.get('invalid_numerator_value') == 0.0,
                 'native geometry metric semantics differ')
    return value


def certify_artifact(geometry):
    value = validate_geometry(geometry)
    return {'schema':CERTIFICATE_SCHEMA, 'native_artifact_sha256':_sha(value),
            'source_sha256':value['source_sha256'], 'invocation_id':value['invocation_id'],
            'certificate':certify_packing(value['numeric'])}
