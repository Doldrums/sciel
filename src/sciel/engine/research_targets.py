"""Frozen input identities for auxiliary research, separate from ancestry/scores.

References are available to a method, not evidence that its code used them.
The full source and saved measurement record travel in the immutable tool input.
"""
from copy import deepcopy
import hashlib
import math

from ..methods.models import content_hash, require_hash
from .observed_changes import _validate_record
from .research_artifacts import (saved_evaluation, saved_evaluation_for_program,
                                 unavailable, validate_saved_evaluation)

SCHEMA = 'sciel-research-targets/2'
FAILURE_SCHEMA = 'sciel-failed-research-target/1'


def source_sha256(source):
    return hashlib.sha256(source.encode('utf-8', errors='surrogatepass')).hexdigest()


def _require(condition, message):
    if not condition:
        raise ValueError('research targets: ' + message)


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def failed_provenance(result, *, iteration, B_evals):
    """A failed attempt has source/evaluation evidence, never an admitted ID."""
    body = dict(schema=FAILURE_SCHEMA, source_sha256=source_sha256(result.child_code or ''),
        iteration=iteration, B_evals=B_evals, evaluation_id=result.meta.get('evaluation_id'),
        reached=result.reached, confirmed=result.confirmed,
        artifacts=deepcopy(result.artifacts or {'error': 'eval_failed'}),
        artifacts_sha256=content_hash(result.artifacts or {'error': 'eval_failed'}))
    return {**body, 'sha256': content_hash(body)}


def validate_failure(value, source, artifacts=None):
    fields = {'schema','source_sha256','iteration','B_evals','evaluation_id','reached',
              'confirmed','artifacts','artifacts_sha256','sha256'}
    _require(type(value) is dict and set(value) == fields and value['schema'] == FAILURE_SCHEMA,
             'invalid failure provenance')
    _require(value['source_sha256'] == source_sha256(source)
             and value['sha256'] == content_hash({k:v for k,v in value.items() if k != 'sha256'}),
             'failure source/digest differs')
    require_hash(value['artifacts_sha256'], 'failed artifacts')
    _require(type(value['artifacts']) is dict
             and value['artifacts_sha256'] == content_hash(value['artifacts']), 'failure artifacts digest differs')
    _require(all(type(value[k]) is int and value[k] >= 0 for k in ('iteration','B_evals')),
             'invalid failure counters')
    _require(value['evaluation_id'] is None or type(value['evaluation_id']) is str,
             'invalid failed evaluation reference')
    _require(type(value['confirmed']) is bool and type(value['reached']) is str,
             'invalid failure stage')
    if artifacts is not None:
        _require(value['artifacts_sha256'] == content_hash(artifacts), 'failure artifacts differ')
    return deepcopy(value)


def program_reference(program, observed, *, engine=None):
    """Selection score is not silently promoted to a canonical measurement."""
    source_hash = source_sha256(program.code)
    record = observed._resolve_program(program.id) if observed is not None else None
    measurement = dict(status='unavailable', record=None, aliases=[])
    if record is not None and record['child_source_sha256'] == source_hash:
        aliases, ident = [], program.id
        while ident in observed._aliases:
            alias = deepcopy(observed._aliases[ident])
            aliases.append(alias)
            ident = alias['reference_program_id']
        measurement = dict(status='recorded', record=deepcopy(record), aliases=aliases)
    return dict(kind='program', program_id=program.id, source=program.code,
        source_sha256=source_hash, selection_score=program.score,
        measurement=measurement, failure=None,
        saved_evaluation=saved_evaluation_for_program(engine, program) if engine is not None
                         else unavailable('No run-bound evaluator artifact source was supplied.'))


def freeze_targets(engine, parent, source, *, failed=False, failure=None):
    if failed:
        failure = validate_failure(failure, source)
    selected = (dict(kind='failed_candidate', program_id=None, source=source,
        source_sha256=source_sha256(source), selection_score=None,
        measurement=dict(status='unavailable', record=None, aliases=[]),
        saved_evaluation=saved_evaluation(engine, source, failure.get('evaluation_id')),
        failure=validate_failure(failure, source)) if failed
        else program_reference(parent, engine.observed_changes, engine=engine))
    best_id = engine.db.pop_best_id
    best = engine.db.get(best_id) if best_id is not None else None
    body = dict(schema=SCHEMA, run_id=engine.log.run_id, task_hash=engine.task.task_hash,
        frozen_at=dict(iteration=engine.iteration_no, B_evals=engine.B_evals),
        ancestry_parent=dict(program_id=parent.id, source_sha256=source_sha256(parent.code)),
        selected=selected,
        population_best=program_reference(best, engine.observed_changes, engine=engine) if best is not None else None)
    # This describes the actual in-memory frozen state. It deliberately does not
    # label an older periodic checkpoint as containing these later observations.
    value = {**body, 'sha256': content_hash(body)}
    return validate_targets(value)


def _validate_reference(value):
    fields = {'kind','program_id','source','source_sha256','selection_score','measurement','failure','saved_evaluation'}
    _require(type(value) is dict and set(value) == fields and type(value['source']) is str,
             'invalid input source fields')
    _require(value['source_sha256'] == source_sha256(value['source']), 'input source hash differs')
    validate_saved_evaluation(value['saved_evaluation'], source_sha256=value['source_sha256'])
    measure = value['measurement']
    _require(type(measure) is dict and set(measure) == {'status','record','aliases'}
             and type(measure['aliases']) is list, 'invalid measurement fields')
    if value['kind'] == 'failed_candidate':
        _require(value['program_id'] is None and value['selection_score'] is None,
                 'failed source cannot claim an admitted program or score')
        validate_failure(value['failure'], value['source'])
        _require(measure == dict(status='unavailable', record=None, aliases=[]),
                 'failed source cannot borrow a canonical measurement')
        return
    _require(value['kind'] == 'program' and type(value['program_id']) is str and value['program_id']
             and _number(value['selection_score']) and value['failure'] is None, 'invalid admitted source')
    if measure['status'] == 'unavailable':
        _require(measure == dict(status='unavailable', record=None, aliases=[]),
                 'unavailable measurement contains evidence')
        return
    _require(measure['status'] == 'recorded', 'unknown measurement status')
    record = _validate_record(measure['record'])
    _require(record['scope'] == 'unpaired' and record['child_source_sha256'] == value['source_sha256'],
             'measurement source differs')
    ident, seen = value['program_id'], set()
    for alias in measure['aliases']:
        _require(type(alias) is dict and alias.get('program_id') == ident and ident not in seen
                 and alias.get('source_sha256') == value['source_sha256']
                 and alias.get('reference_record_sha256') == record['record_sha256']
                 and alias.get('score') == record['child_observation']['score'], 'measurement alias differs')
        cases = record['child_observation'].get('case_vector')
        _require(cases is None or alias.get('cases_sha256') == cases['cases_sha256'], 'alias cases differ')
        seen.add(ident)
        ident = alias.get('reference_program_id')
    _require(record['child_program_id'] == ident, 'measurement program differs')


def validate_targets(value, *, snapshot=None, task_hash=None):
    fields = {'schema','run_id','task_hash','frozen_at','ancestry_parent','selected','population_best','sha256'}
    _require(type(value) is dict and set(value) == fields and value['schema'] == SCHEMA,
             'missing/unsupported frozen input contract')
    _require(all(type(value[k]) is str and value[k] for k in ('run_id','task_hash')), 'missing run/task identity')
    _require(value['sha256'] == content_hash({k:v for k,v in value.items() if k != 'sha256'}), 'snapshot digest differs')
    frozen = value['frozen_at']
    _require(type(frozen) is dict and set(frozen) == {'iteration','B_evals'}
             and all(type(n) is int and n >= 0 for n in frozen.values()), 'invalid frozen counters')
    ancestry = value['ancestry_parent']
    _require(type(ancestry) is dict and set(ancestry) == {'program_id','source_sha256'}
             and type(ancestry['program_id']) is str and ancestry['program_id'], 'invalid ancestry reference')
    require_hash(ancestry['source_sha256'], 'ancestry source')
    _validate_reference(value['selected'])
    selected = value['selected']
    validate_saved_evaluation(selected['saved_evaluation'], source_sha256=selected['source_sha256'],
                             task_hash=value['task_hash'], run_id=value['run_id'])
    if selected['kind'] == 'program':
        _require(selected['program_id'] == ancestry['program_id']
                 and selected['source_sha256'] == ancestry['source_sha256'], 'selected ancestry differs')
    if value['population_best'] is not None:
        _validate_reference(value['population_best'])
        _require(value['population_best']['kind'] == 'program', 'champion is not an admitted program')
        validate_saved_evaluation(value['population_best']['saved_evaluation'],
            source_sha256=value['population_best']['source_sha256'], task_hash=value['task_hash'], run_id=value['run_id'])
    if task_hash is not None:
        _require(value['task_hash'] == task_hash, 'input belongs to another task')
    if snapshot is not None:
        _require(selected['source'] == snapshot.parent_code
                 and ancestry['program_id'] == snapshot.parent_id, 'input differs from generation snapshot')
    return deepcopy(value)


def input_binding(inputs):
    targets = validate_targets(inputs.get('research_targets'), task_hash=inputs['task'].get('task_hash'))
    selected, best = targets['selected'], targets['population_best']
    score = (selected['measurement']['record']['child_observation']['score']
             if selected['measurement']['status'] == 'recorded' else None)
    _require(inputs.get('parent_program_id') == selected['program_id']
             and inputs.get('parent_source') == selected['source']
             and inputs.get('parent_score') == score
             and inputs.get('ancestry_parent_id') == targets['ancestry_parent']['program_id'],
             'compatibility input aliases differ from frozen sources')
    return dict(candidate_id=selected['program_id'], target_kind=selected['kind'],
        target_source_sha256=selected['source_sha256'],
        ancestry_parent_id=targets['ancestry_parent']['program_id'], research_targets_sha256=targets['sha256'],
        population_best_id=best['program_id'] if best else None,
        population_best_source_sha256=best['source_sha256'] if best else None)


def tool_inputs(snapshot, task_identity, arguments):
    targets = validate_targets(snapshot.research_targets, snapshot=snapshot,
                               task_hash=task_identity.get('task_hash'))
    selected = targets['selected']
    measure = selected['measurement']
    return dict(task=deepcopy(task_identity), task_prompt=snapshot.task_prompt,
        parent_program_id=selected['program_id'], parent_source=selected['source'],
        parent_score=measure['record']['child_observation']['score'] if measure['status']=='recorded' else None,
        ancestry_parent_id=targets['ancestry_parent']['program_id'], research_targets=targets,
        island_id=snapshot.island, arguments=deepcopy(arguments),
        search_evidence=deepcopy(getattr(snapshot, 'research_search_evidence', None)))
