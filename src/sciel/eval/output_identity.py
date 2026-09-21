"""Evaluator-owned exact output witnesses, never score or case-vector equality.

The caller authenticates the evaluation receipt before calling ``adapt_output``.
Hashes bind that receipt's bytes; they do not establish producer authenticity on
their own. Unsupported, failed, or malformed evidence does not gate admission.
"""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path
import re

SCHEMA = 'sciel-output-identity/1'
CONFIG_SCHEMA = 'sciel-output-identity-adapter/1'
WITNESS_SCHEMA = 'sciel-full-output/1'
MAX_WITNESS_BYTES = 1024 * 1024


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    """Strict JSON; no tuples, non-string keys, NaN, or implicit coercions."""
    def check(item, depth=0):
        require(depth <= 64, 'output identity JSON is too deep')
        if item is None or type(item) in (str, bool, int):
            return
        if type(item) is float:
            require(math.isfinite(item), 'nonfinite output identity number')
        elif type(item) is list:
            for child in item:
                check(child, depth + 1)
        elif type(item) is dict:
            require(all(type(key) is str for key in item), 'non-string JSON key')
            for child in item.values():
                check(child, depth + 1)
        else:
            raise ValueError('non-JSON output identity value')
    check(value)
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=True, allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def require_hash(value, name):
    require(type(value) is str and re.fullmatch('[0-9a-f]{64}', value) is not None,
            'invalid ' + name)
    return value


def sealed(value):
    value = deepcopy(value)
    value['sha256'] = digest(value)
    return value


def check_seal(value):
    require(type(value) is dict, 'sealed evidence must be an object')
    require_hash(value.get('sha256'), 'evidence sha256')
    require(value['sha256'] == digest({k: v for k, v in value.items() if k != 'sha256'}),
            'output evidence digest differs')


def validate_binding(value):
    require(type(value) is dict and {'task_hash', 'source_sha256', 'island_id'} <= set(value),
            'missing evaluation binding')
    for key in ('task_hash', 'source_sha256'):
        require_hash(value[key], key)
    island = value['island_id']
    require(island is None or (type(island) is int and island >= 0)
            or (type(island) is str and 0 < len(island) <= 256), 'invalid island_id')
    for key in ('request_sha256', 'receipt_sha256'):
        if key in value:
            require_hash(value[key], key)
    require(len(canonical(value)) <= 8192, 'evaluation binding exceeds limit')
    return deepcopy(value)


def validate_output_identity_config(value):
    """Trusted task opt-in for a full JSON output artifact, not a digest claim."""
    if value is None:
        return None
    require(type(value) is dict and set(value) == {
        'schema', 'adapter', 'artifact', 'input_scope', 'seed_scope'},
        'invalid output_identity descriptor fields')
    require(value['schema'] == CONFIG_SCHEMA and value['adapter'] == 'canonical-json/1',
            'unsupported output identity adapter')
    require(type(value['artifact']) is str and re.fullmatch('[A-Za-z0-9_.-]{1,128}', value['artifact']),
            'invalid output identity artifact name')
    require(value['input_scope'] in ('task_fixed', 'artifact_sha256'), 'invalid input scope')
    require(value['seed_scope'] in ('unpaired', 'input_seed'), 'invalid seed scope')
    return deepcopy(value)


def _stage(reached, confirmed, confirmation):
    require(reached in ('smoke', 'full', 'confirm') and type(confirmed) is bool,
            'invalid selected output stage')
    if reached == 'confirm':
        require(type(confirmation) is dict and set(confirmation) == {
            'trials', 'selected_trial', 'selection'}, 'missing confirmation selection')
        require(type(confirmation['trials']) is int and confirmation['trials'] >= 1
                and type(confirmation['selected_trial']) is int
                and 0 <= confirmation['selected_trial'] < confirmation['trials']
                and type(confirmation['selection']) is str
                and 0 < len(confirmation['selection']) <= 128,
                'invalid confirmation selection')
        require(not confirmed or confirmation['trials'] >= 2, 'invalid confirmed trials')
        selected = {k: confirmation[k] for k in ('trials', 'selection')}
    else:
        require(confirmation is None and confirmed is False, 'unexpected confirmation selection')
        selected = None
    # Trial index identifies the observed draw, not the comparison population.
    # This matches only the selected output, never all confirmation trial outputs.
    return dict(reached=reached, confirmed=confirmed, confirmation=selected,
                selected_output_only=True)


def _scope(task_hash, evaluator_sha, adapter, stage, inputs, seeds, representation):
    return dict(task_hash=task_hash, evaluator_sha256=evaluator_sha, adapter=adapter,
                stage=stage, inputs=inputs, seeds=seeds, representation=representation)


def _confirmation_binding(task, outcome, confirmation):
    if outcome.reached != 'confirm':
        return
    # These cases are protocol metadata, never an output-equivalence shortcut.
    trials = outcome.result.cases.get('n_trials')
    required = getattr(task, 'confirm_n', None)
    require(type(required) is int and required >= 1, 'missing configured confirmation count')
    require(outcome.confirmed is True and outcome.result.confirmed_sampled is True,
            'confirmation did not authenticate repeated sampling')
    require(type(trials) in (int, float) and math.isfinite(trials)
            and trials == confirmation['trials'] and trials >= max(2, required),
            'confirmation witness differs from recorded/requested trial count')


def validate_identity(value):
    check_seal(value)
    require(set(value) == {'schema', 'scope', 'output_sha256', 'evidence', 'sha256'}
            and value['schema'] == SCHEMA, 'invalid output identity fields/schema')
    require_hash(value['output_sha256'], 'output_sha256')
    scope, evidence = value['scope'], value['evidence']
    require(type(scope) is dict and set(scope) == {'task_hash', 'evaluator_sha256', 'adapter',
        'stage', 'inputs', 'seeds', 'representation'}, 'invalid output identity scope')
    for key in ('task_hash', 'evaluator_sha256'):
        require_hash(scope[key], key)
    require(scope['adapter'] in ('canonical-json/1', 'native-circle-geometry/1'), 'unknown adapter')
    require(type(evidence) is dict and set(evidence) == {'source_sha256', 'artifact_sha256',
        'binding_sha256', 'invocation_id', 'selected_trial'}, 'invalid output evidence fields')
    for key in ('source_sha256', 'artifact_sha256', 'binding_sha256'):
        require_hash(evidence[key], key)
    require(type(evidence['invocation_id']) is str and 0 < len(evidence['invocation_id']) <= 128,
            'invalid output invocation_id')
    stage = scope['stage']
    require(type(stage) is dict and set(stage) == {'reached', 'confirmed', 'confirmation',
            'selected_output_only'} and stage['selected_output_only'] is True, 'invalid output stage scope')
    confirmation = stage['confirmation']
    if confirmation is not None:
        require(type(confirmation) is dict and set(confirmation) == {'trials', 'selection'},
                'invalid confirmation scope')
        confirmation = dict(confirmation, selected_trial=evidence['selected_trial'])
    else:
        require(evidence['selected_trial'] is None, 'unexpected selected trial')
    require(_stage(stage['reached'], stage['confirmed'], confirmation) == stage, 'stage scope differs')
    for name, kinds in (('inputs', ('task_fixed', 'artifact_sha256')),
                        ('seeds', ('unpaired', 'input_seed'))):
        item = scope[name]
        require(type(item) is dict and set(item) == {'kind', 'sha256'} and item['kind'] in kinds,
                'invalid ' + name + ' scope')
        if item['kind'] == 'unpaired':
            require(item['sha256'] is None, 'unpaired seed cannot claim a seed binding')
        else:
            require_hash(item['sha256'], name + ' sha256')
        if item['kind'] == 'task_fixed':
            require(item['sha256'] == scope['task_hash'], 'fixed input task differs')
    representation = scope['representation']
    if scope['adapter'] == 'canonical-json/1':
        require(representation == {'format': 'canonical-json/1'}, 'invalid JSON representation')
    else:
        require(type(representation) is dict and set(representation) == {
            'format', 'centers_dtype', 'radii_dtype'}
            and representation['format'] == 'centers26x2-radii26/1'
            and all(representation[k] in ('float16', 'float32', 'float64')
                    for k in ('centers_dtype', 'radii_dtype')), 'invalid geometry representation')
    require(len(canonical(value)) <= 16384, 'output identity exceeds limit')
    return deepcopy(value)


def _parse_full(text):
    require(type(text) is str and len(text.encode()) <= MAX_WITNESS_BYTES, 'missing/big full-output witness')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, 'duplicate full-output JSON key')
            result[key] = value
        return result
    value = json.loads(text, object_pairs_hook=unique)
    require(type(value) is dict and set(value) == {'schema', 'source_sha256', 'evaluator_sha256',
        'invocation_id', 'stage', 'confirmation', 'input_sha256', 'seed_sha256', 'output'}
        and value['schema'] == WITNESS_SCHEMA, 'invalid full-output witness fields/schema')
    require(text == canonical(value), 'full-output witness must be canonical JSON')
    return value


def adapt_output(task, source_sha256, outcome, evaluation_binding):
    """Return a sealed identity, or None when no supported authenticated output exists.

    ``evaluation_binding`` must come from the trusted, completed evaluation path;
    it is not accepted from candidate code. Receipt verification is caller-owned.
    The full witness remains in that immutable receipt, not duplicated in state.
    """
    try:
        binding = validate_binding(evaluation_binding)
        require_hash(source_sha256, 'source_sha256')
        require(binding['source_sha256'] == source_sha256
                and binding['task_hash'] == task.task_hash, 'task/source binding differs')
        result = outcome.result
        require(result.ok is True, 'failed evaluation is not eligible')
        evaluator_sha = hashlib.sha256(Path(task.evaluator_path).read_bytes()).hexdigest()
        config = validate_output_identity_config(getattr(task, 'output_identity', None))
        if config is not None:
            text = result.artifacts.get(config['artifact'])
            witness = _parse_full(text)
            require(witness['source_sha256'] == source_sha256
                    and witness['evaluator_sha256'] == evaluator_sha, 'witness source/evaluator differs')
            require(witness['stage'] == ('full' if outcome.reached == 'confirm' else outcome.reached),
                    'witness stage differs')
            stage = _stage(outcome.reached, outcome.confirmed, witness['confirmation'])
            inputs = dict(kind=config['input_scope'], sha256=(task.task_hash
                if config['input_scope'] == 'task_fixed' else witness['input_sha256']))
            require(config['input_scope'] != 'task_fixed' or witness['input_sha256'] is None,
                    'fixed task input must not claim varying input')
            seeds = dict(kind=config['seed_scope'], sha256=witness['seed_sha256'])
            require(config['seed_scope'] != 'unpaired' or witness['seed_sha256'] is None,
                    'unpaired seed must be unspecified')
            scope = _scope(task.task_hash, evaluator_sha, config['adapter'], stage, inputs, seeds,
                           {'format': 'canonical-json/1'})
            output_sha = digest(witness['output'])
        else:
            from .native_geometry import parse_artifact, validate_binding as native_binding
            from ..engine.repeat_support import NATIVE_GEOMETRY_TASK_EVALUATORS
            from ..hypothesis.measurement import metric_context
            require(NATIVE_GEOMETRY_TASK_EVALUATORS.get(task.task_hash) == evaluator_sha,
                    'no registered full-output adapter')
            text = result.artifacts.get('native_geometry')
            witness = parse_artifact(text)
            native_binding(witness, dict(ok=result.ok, score=result.score, cases=result.cases,
                reached=outcome.reached, confirmed=outcome.confirmed), source_sha256, metric_context(task))
            require(witness['native_validity'] is True, 'invalid geometry is not eligible')
            scope = _scope(task.task_hash, evaluator_sha, 'native-circle-geometry/1',
                _stage(outcome.reached, outcome.confirmed, witness['confirmation']),
                {'kind': 'task_fixed', 'sha256': task.task_hash}, {'kind': 'unpaired', 'sha256': None},
                dict(format='centers26x2-radii26/1', centers_dtype=witness['centers_dtype'],
                     radii_dtype=witness['radii_dtype']))
            output_sha = witness['numeric_sha256']
        identity = sealed(dict(schema=SCHEMA, scope=scope, output_sha256=output_sha, evidence=dict(
            source_sha256=source_sha256, artifact_sha256=hashlib.sha256(text.encode()).hexdigest(),
            binding_sha256=digest(binding), invocation_id=witness['invocation_id'],
            selected_trial=(witness['confirmation']['selected_trial'] if witness['confirmation'] else None))))
        _confirmation_binding(task, outcome, witness['confirmation'])
        return validate_identity(identity)
    except (ValueError, TypeError, KeyError, AttributeError, OSError, UnicodeError, RecursionError, OverflowError):
        # No weak fallback: a broken/missing witness cannot manufacture equality.
        return None


def adapt_near_output(task, source_sha256, outcome, evaluation_binding, *, identity=None, policy=None):
    """Bind complete output and canonical score from a returned evaluator receipt.

    The trusted caller authenticates ``evaluation_binding['receipt_sha256']``.
    A cold restore must reconstruct this value from that same immutable receipt,
    not trust a checkpoint's self-consistent payload and hashes. Bad/unsupported
    output yields None; an invalid trusted policy is a configuration error.
    """
    from .near_output import (MEASUREMENT_SCHEMA, policy_matches_identity,
                              validate_measurement, validate_policy)
    policy = validate_policy(policy)
    if policy is None:
        return None
    try:
        binding = validate_binding(evaluation_binding)
        require_hash(binding.get('receipt_sha256'), 'near-output receipt')
        exact = adapt_output(task, source_sha256, outcome, binding)
        if exact is None:
            return None
        if identity is not None:
            require(canonical(validate_identity(identity)) == canonical(exact),
                    'near-output supplied identity differs from returned output')
        identity = exact
        require(policy_matches_identity(policy, identity), 'near-output policy scope differs')
        if identity['scope']['adapter'] == 'canonical-json/1':
            config = validate_output_identity_config(task.output_identity)
            output = _parse_full(outcome.result.artifacts[config['artifact']])['output']
        else:
            from .native_geometry import parse_artifact
            output = parse_artifact(outcome.result.artifacts['native_geometry'])['numeric']
        value = sealed(dict(schema=MEASUREMENT_SCHEMA, policy_sha256=digest(policy),
            identity_sha256=identity['sha256'], receipt_sha256=binding['receipt_sha256'],
            score=outcome.result.score, output=output))
        return validate_measurement(value, identity=identity, binding=binding, policy=policy)
    except (ValueError, TypeError, KeyError, AttributeError, OSError, UnicodeError, RecursionError, OverflowError):
        return None
