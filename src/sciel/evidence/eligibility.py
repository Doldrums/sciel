"""Admission eligibility is separate from the recorded numerical observation."""
from copy import deepcopy

VIEW_SCHEMA = 'sciel-output-admission-view/1'
REPEAT_STATUSES = ('exact_recorded_repeat', 'non_improving_near_repeat')


def validate_admission_view(value):
    from ..eval.output_identity import require, require_hash
    require(type(value) is dict and set(value) == {'schema', 'status', 'accept', 'evaluation_id',
        'verdict_sha256', 'scope_sha256', 'adapter', 'match'} and value['schema'] == VIEW_SCHEMA,
        'invalid output admission view')
    require(value['status'] in ('new_output', 'unsupported', *REPEAT_STATUSES)
        and type(value['accept']) is bool and value['accept'] is (value['status'] not in REPEAT_STATUSES),
        'invalid output admission disposition')
    require(type(value['evaluation_id']) is str and 0 < len(value['evaluation_id']) <= 512,
        'invalid output evaluation reference')
    require_hash(value['verdict_sha256'], 'verdict reference')
    if value['status'] == 'unsupported':
        require(value['scope_sha256'] is None and value['adapter'] is None and value['match'] is None,
            'unsupported admission cannot have output evidence')
    else:
        require_hash(value['scope_sha256'], 'scope reference')
        require(value['adapter'] in ('canonical-json/1', 'native-circle-geometry/1'), 'invalid output adapter')
    if value['status'] in REPEAT_STATUSES:
        match = value['match']
        require(type(match) is dict and set(match) == {'evaluation_id', 'source_sha256', 'island_id', 'program_id'},
            'invalid output match reference')
        require_hash(match['source_sha256'], 'matched source reference')
        island = match['island_id']
        require(island is None or (type(island) is int and island >= 0)
                or (type(island) is str and 0 < len(island) <= 256), 'invalid matched island')
        require(type(match['evaluation_id']) is str and 0 < len(match['evaluation_id']) <= 512
                and match['evaluation_id'] != value['evaluation_id'], 'invalid prior evaluation reference')
        require(match['program_id'] is None or (type(match['program_id']) is str
                and 0 < len(match['program_id']) <= 512), 'invalid prior program reference')
    else:
        require(value['match'] is None, 'nonrepeat cannot have a match')
    return deepcopy(value)


def admission_view(record):
    if 'output_novelty' not in record:
        return validate_admission_view(record['output_admission']) if 'output_admission' in record else None
    from ..engine.output_novelty import validate_verdict
    from ..eval.output_identity import digest
    verdict = validate_verdict(record['output_novelty'], source_sha256=record.get('source_sha256'))
    identity = verdict['identity']
    return validate_admission_view(dict(schema=VIEW_SCHEMA, status=verdict['status'], accept=verdict['accept'],
        evaluation_id=verdict['evaluation_id'], verdict_sha256=verdict['sha256'],
        scope_sha256=digest(identity['scope']) if identity is not None else None,
        adapter=identity['scope']['adapter'] if identity is not None else None, match=verdict['match']))


def validate_output_observation(value, *, source_sha256, metric, island, evaluation, status):
    from ..engine.output_novelty import validate_verdict
    verdict = validate_verdict(value, source_sha256=source_sha256)
    if source_sha256 is None or evaluation is None or status not in ('evaluated', 'eval_failed'):
        raise ValueError('output novelty requires a source-bound returned evaluation')
    binding = verdict['binding']
    if ((metric or {}).get('task_hash') not in (None, binding['task_hash'])
            or island not in (None, binding['island_id'])):
        raise ValueError('output verdict differs from research task/island')
    if verdict['identity'] is not None:
        stage = verdict['identity']['scope']['stage']
        if (not evaluation['ok'] or status != 'evaluated'
                or stage['reached'] != evaluation['reached']
                or stage['confirmed'] != evaluation['confirmed']):
            raise ValueError('output identity differs from successful selected evaluation')
        near = verdict.get('near_output')
        if near is not None and near['score'] != evaluation['score']:
            raise ValueError('near-output measurement differs from recorded evaluation score')
        geometry = evaluation.get('native_geometry')
        if geometry is not None and verdict['identity']['scope']['adapter'] == 'native-circle-geometry/1':
            from ..eval.output_identity import digest
            identity = verdict['identity']
            representation = identity['scope']['representation']
            confirmation = geometry['confirmation']
            if (identity['output_sha256'] != geometry['numeric_sha256']
                    or identity['scope']['evaluator_sha256'] != geometry['evaluator_sha256']
                    or identity['evidence']['invocation_id'] != geometry['invocation_id']
                    or identity['evidence']['artifact_sha256'] != digest(geometry)
                    or identity['evidence']['selected_trial'] != (confirmation['selected_trial'] if confirmation else None)
                    or stage['confirmation'] != ({key: confirmation[key] for key in ('trials', 'selection')}
                                                 if confirmation else None)
                    or any(representation[key] != geometry[key] for key in ('centers_dtype', 'radii_dtype'))):
                raise ValueError('output identity differs from recorded native geometry')
    return verdict


def output_repeat(record):
    value = record.get('output_novelty')
    if value is None:
        view = admission_view(record)
        return view is not None and view['status'] in REPEAT_STATUSES
    from ..engine.output_novelty import is_repeat
    return is_repeat(value)


def efficacy_eligible(record):
    return record.get('status') == 'evaluated' and not output_repeat(record)


def admission_facts(record):
    """An optional projection; historical records retain their original shape."""
    view = admission_view(record)
    if view is None:
        return {}
    return {'output_admission': view,
            'efficacy_eligible': efficacy_eligible(record)}
