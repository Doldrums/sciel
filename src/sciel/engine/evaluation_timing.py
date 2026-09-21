"""Source-bound wall timing with explicit evaluator-stage scope.

The subprocess runner measures a whole returned stage, including evaluator work.
An evaluator-reported trial count supports an arithmetic mean, not a measurement
of any one candidate invocation or its fraction of a per-run deadline.
"""
from copy import deepcopy
import math

from .validation import source_sha256


SCHEMA = 'sciel-evaluation-timing/1'
SCOPE = 'returned_evaluation_stage_including_evaluator_overhead'


def _duration(value):
    if type(value) not in (int, float):
        return None
    try:
        value = float(value)
    except OverflowError:
        return None
    return value if math.isfinite(value) and value >= 0 else None


def trial_count(value):
    """A reported positive integer is usable; absence never implies one run."""
    number = _duration(value)
    if number is None or number < 1 or not number.is_integer():
        return None
    return int(value)


def observation(source, result, *, evaluation_id=None, reached=None):
    result = getattr(result, 'result', result)
    cases = getattr(result, 'cases', None) or {}
    count = trial_count(cases.get('n_trials'))
    duration = _duration(getattr(result, 'timing', None))
    return dict(schema=SCHEMA, source_sha256=source_sha256(source),
        evaluation_id=evaluation_id, reached=reached,
        evaluation_stage_wall_seconds=duration,
        evaluator_reported_trials=count,
        trial_count_status=('reported_positive_integer' if count is not None else
                            'absent' if 'n_trials' not in cases else 'invalid'),
        mean_evaluation_stage_wall_seconds_per_reported_trial=(
            duration / count if duration is not None and count is not None else None),
        timing_scope=SCOPE)


def remember(engine, program, result):
    binding = getattr(engine, 'research_evaluation_bindings', {}).get(program.id) or {}
    meta = getattr(result, 'meta', None) or {}
    evaluation_id = (binding.get('evaluation_id') or getattr(result, 'evaluation_id', None)
                     or meta.get('evaluation_id'))
    row = observation(program.code, result, evaluation_id=evaluation_id,
                      reached=getattr(result, 'reached', None))
    if not hasattr(engine, 'evaluation_timings'):
        engine.evaluation_timings = {}
    if not hasattr(engine, 'eval_times'):
        engine.eval_times = {}
    engine.evaluation_timings[program.id] = row
    # Legacy cache remains an average with an explicit scope at every consumer.
    # Unknown count means unknown mean, including tasks that omit n_trials.
    engine.eval_times[program.id] = row['mean_evaluation_stage_wall_seconds_per_reported_trial']


def for_program(engine, program):
    """Do not reinterpret old unscoped cache values as measured per-run time."""
    row = getattr(engine, 'evaluation_timings', {}).get(program.id)
    if row is None:
        return None
    validate({program.id: row}, program_sources={program.id: program.code},
             program_cases=({program.id: program.cases} if hasattr(program, 'cases') else None),
             evaluation_bindings=getattr(engine, 'research_evaluation_bindings', {}))
    return deepcopy(row)


def validate(value, *, program_sources, program_cases=None, evaluation_bindings=None):
    if type(value) is not dict:
        raise ValueError('evaluation timing records must be a dictionary')
    fields = set(observation('', object()))
    for ident, row in value.items():
        if (type(ident) is not str or ident not in program_sources or type(row) is not dict
                or set(row) != fields or row['schema'] != SCHEMA or row['timing_scope'] != SCOPE
                or row['source_sha256'] != source_sha256(program_sources[ident])
                or row['reached'] not in (None, 'smoke', 'full', 'confirm')
                or (row['evaluation_id'] is not None and
                    (type(row['evaluation_id']) is not str or not row['evaluation_id']))):
            raise ValueError('invalid source-bound evaluation timing record')
        duration, count = row['evaluation_stage_wall_seconds'], row['evaluator_reported_trials']
        if duration is not None and _duration(duration) is None:
            raise ValueError('invalid evaluation stage wall duration')
        if count is not None and (type(count) is not int or trial_count(count) != count):
            raise ValueError('invalid evaluator-reported trial count')
        expected_status = ('reported_positive_integer',) if count is not None else ('absent', 'invalid')
        if row['trial_count_status'] not in expected_status:
            raise ValueError('evaluation trial count status differs')
        if program_cases is not None:
            cases = program_cases[ident]
            reported = trial_count(cases.get('n_trials'))
            status = ('reported_positive_integer' if reported is not None else
                      'absent' if 'n_trials' not in cases else 'invalid')
            if count != reported or row['trial_count_status'] != status:
                raise ValueError('evaluation timing count differs from saved evaluator cases')
        binding = (evaluation_bindings or {}).get(ident)
        if binding is not None and (row['evaluation_id'] != binding['evaluation_id']
                or row['source_sha256'] != binding['source_sha256']):
            raise ValueError('evaluation timing provenance differs from saved evaluation binding')
        expected = duration / count if duration is not None and count is not None else None
        mean = row['mean_evaluation_stage_wall_seconds_per_reported_trial']
        if mean != expected or (mean is not None and _duration(mean) is None):
            raise ValueError('evaluation stage mean differs from duration/count')
    return deepcopy(value)
