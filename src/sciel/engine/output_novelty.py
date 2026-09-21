"""Whole-run exact recorded-output history, separate from source novelty.

Observe on the coordinator after authenticated evaluation completion. Stable
coordinator order chooses the first witness; thread completion order must not.
No eviction or alias operation removes measured output history.
"""
from copy import deepcopy
from threading import RLock

from ..eval.output_identity import (canonical, check_seal, digest, require, require_hash,
                                    sealed, validate_binding, validate_identity)
from ..eval.near_output import (assessment, distance, validate_assessment,
                                validate_measurement, validate_policy)

SCHEMA = 'sciel-output-novelty/1'
VERDICT_SCHEMA = 'sciel-output-novelty-verdict/1'
NEAR_SCHEMA = 'sciel-output-novelty/2'
NEAR_VERDICT_SCHEMA = 'sciel-output-novelty-verdict/2'
REPEAT_STATUSES = frozenset(('exact_recorded_repeat', 'non_improving_near_repeat'))


def _id(value):
    require(type(value) is str and 0 < len(value) <= 512, 'invalid evaluation/program ID')
    return value


def _bound_identity(identity, binding):
    identity = validate_identity(identity)
    require(identity['scope']['task_hash'] == binding['task_hash']
            and identity['evidence']['source_sha256'] == binding['source_sha256']
            and identity['evidence']['binding_sha256'] == digest(binding), 'output/evaluation binding differs')
    return identity


def validate_verdict(value, *, source_sha256=None):
    check_seal(value)
    modern = value.get('schema') == NEAR_VERDICT_SCHEMA
    fields = {'schema', 'evaluation_id', 'accept', 'status', 'binding', 'identity', 'match', 'sha256'}
    require(set(value) == fields | ({'near_output'} if modern else set())
            and value['schema'] in (VERDICT_SCHEMA, NEAR_VERDICT_SCHEMA),
            'invalid output novelty verdict fields/schema')
    _id(value['evaluation_id'])
    binding = validate_binding(value['binding'])
    if source_sha256 is not None:
        require_hash(source_sha256, 'expected source_sha256')
        require(binding['source_sha256'] == source_sha256, 'verdict source differs')
    status = value['status']
    require(status in ('unsupported', 'new_output', 'exact_recorded_repeat')
            or (modern and status == 'non_improving_near_repeat'), 'invalid output verdict status')
    require(type(value['accept']) is bool and value['accept'] is (status not in REPEAT_STATUSES),
            'output admission differs from verdict')
    if status == 'unsupported':
        require(value['identity'] is None and value['match'] is None, 'unsupported evidence cannot match')
    else:
        _bound_identity(value['identity'], binding)
        if status == 'new_output':
            require(value['match'] is None, 'new output cannot have a match')
        else:
            match = value['match']
            require(type(match) is dict and set(match) == {'evaluation_id', 'source_sha256', 'island_id',
                'program_id'}, 'invalid output match fields')
            require(_id(match['evaluation_id']) != value['evaluation_id'], 'evaluation cannot repeat itself')
            validate_binding(dict(task_hash=binding['task_hash'], source_sha256=match['source_sha256'],
                                  island_id=match['island_id']))
            if match['program_id'] is not None:
                _id(match['program_id'])
    if modern:
        validate_assessment(value['near_output'], identity=value['identity'], status=status)
    return deepcopy(value)


def is_repeat(value):
    return value is not None and validate_verdict(value)['status'] in REPEAT_STATUSES


class OutputNoveltyRegistry:
    def __init__(self, identity, *, near_policy=None):
        require(type(identity) is dict, 'registry identity must be an object')
        require_hash(identity.get('task_hash'), 'registry task_hash')
        self._near_policy = validate_policy(near_policy)
        identity = deepcopy(identity)
        if 'near_output_policy' in identity:
            require(canonical(identity['near_output_policy']) == canonical(self._near_policy),
                    'registry near-output policy differs')
        if self._near_policy is not None:
            require(self._near_policy['task_hash'] == identity['task_hash'],
                    'registry near-output task differs')
            identity['near_output_policy'] = deepcopy(self._near_policy)
        require(len(canonical(identity)) <= 16384, 'registry identity exceeds limit')
        self._identity = deepcopy(identity)
        self._records, self._by_output, self._by_scope = {}, {}, {}
        self._lock = RLock()

    @property
    def identity(self):
        return deepcopy(self._identity)

    @property
    def near_policy(self):
        return deepcopy(self._near_policy)

    def __deepcopy__(self, memo):
        with self._lock:
            result = type(self)(self.identity, near_policy=self._near_policy)
            memo[id(self)] = result
            result._records = deepcopy(self._records, memo)
            result._by_output = deepcopy(self._by_output, memo)
            result._by_scope = deepcopy(self._by_scope, memo)
            return result

    @staticmethod
    def _key(identity):
        return digest(dict(scope=identity['scope'], output_sha256=identity['output_sha256']))

    def observe(self, evaluation_id, identity, binding, *, measurement=None):
        evaluation_id, binding = _id(evaluation_id), validate_binding(binding)
        require(binding['task_hash'] == self.identity['task_hash'], 'registry task binding differs')
        if identity is not None:
            try:
                identity = _bound_identity(identity, binding)
            except (ValueError, TypeError, KeyError, RecursionError, OverflowError):
                identity = None
        if measurement is not None:
            require(self._near_policy is not None and identity is not None,
                    'near-output measurement requires configured valid output identity')
            measurement = validate_measurement(measurement, identity=identity, binding=binding,
                                               policy=self._near_policy)
        with self._lock:
            old = self._records.get(evaluation_id)
            if old is not None:
                require(canonical(old['identity']) == canonical(identity)
                        and canonical(old['binding']) == canonical(binding)
                        and canonical(old.get('near_measurement')) == canonical(measurement),
                        'evaluation ID conflicts with previously observed evidence')
                return deepcopy(old['verdict'])
            match = None
            near_match = None
            if identity is None:
                status = 'unsupported'
            else:
                key = self._key(identity)
                previous_id = self._by_output.get(key)
                status = 'exact_recorded_repeat' if previous_id is not None else 'new_output'
                if previous_id is None and measurement is not None:
                    # Keep every observation: proximity is nontransitive, and a
                    # rejected observation can still be a later witness. Stable
                    # history order resolves ties after the strongest score.
                    best_score = None
                    for candidate_id in self._by_scope.get(digest(identity['scope']), ()):
                        candidate = self._records[candidate_id]['near_measurement']
                        if candidate['score'] < measurement['score']:
                            continue
                        delta = distance(measurement['output'], candidate['output'], self._near_policy)
                        if delta is not None and (best_score is None or candidate['score'] > best_score):
                            previous_id, best_score = candidate_id, candidate['score']
                            near_match = dict(measurement_sha256=candidate['sha256'],
                                              score=candidate['score'], max_abs_delta=delta)
                    if previous_id is not None:
                        status = 'non_improving_near_repeat'
                if previous_id is not None:
                    previous = self._records[previous_id]
                    match = dict(evaluation_id=previous_id, source_sha256=previous['binding']['source_sha256'],
                        island_id=previous['binding']['island_id'],
                        program_id=(previous['program_ids'][0] if previous['program_ids'] else None))
            modern = self._near_policy is not None
            verdict = dict(schema=NEAR_VERDICT_SCHEMA if modern else VERDICT_SCHEMA,
                evaluation_id=evaluation_id, accept=status not in REPEAT_STATUSES,
                status=status, binding=binding, identity=identity, match=match)
            if modern:
                verdict['near_output'] = assessment(measurement, self._near_policy, near_match)
            verdict = validate_verdict(sealed(verdict))
            self._records[evaluation_id] = dict(evaluation_id=evaluation_id, identity=identity,
                binding=binding, program_ids=[], verdict=verdict)
            if modern:
                self._records[evaluation_id]['near_measurement'] = deepcopy(measurement)
            if identity is not None:
                self._by_output.setdefault(self._key(identity), evaluation_id)
                if measurement is not None:
                    self._by_scope.setdefault(digest(identity['scope']), []).append(evaluation_id)
            return deepcopy(verdict)

    def bind_program(self, evaluation_id, program_id):
        evaluation_id, program_id = _id(evaluation_id), _id(program_id)
        with self._lock:
            require(evaluation_id in self._records, 'cannot bind an unobserved evaluation')
            aliases = self._records[evaluation_id]['program_ids']
            if program_id not in aliases:
                aliases.append(program_id)

    def state_dict(self):
        with self._lock:
            return sealed(dict(schema=NEAR_SCHEMA if self._near_policy is not None else SCHEMA, identity=self.identity,
                               records=deepcopy(list(self._records.values()))))

    def summary(self):
        with self._lock:
            counts = {key: 0 for key in ('new_output', 'exact_recorded_repeat', 'unsupported')}
            if self._near_policy is not None:
                counts['non_improving_near_repeat'] = 0
            for row in self._records.values():
                counts[row['verdict']['status']] += 1
            return dict(evaluations=len(self._records), distinct_outputs=len(self._by_output), **counts)

    def load_state_dict(self, state, *, measurement_validator=None):
        """Validate history atomically, reconstructing near data from receipts.

        For each record in a near-enabled checkpoint, the callback
        (evaluation_id, identity, binding) must return adapt_near_output's result
        from the independently authenticated immutable evaluator receipt. Even a
        stored None is rechecked, so deleting evidence is not a downgrade path.
        """
        check_seal(state)
        modern = self._near_policy is not None
        require(set(state) == {'schema', 'identity', 'records', 'sha256'}
                and state['schema'] == (NEAR_SCHEMA if modern else SCHEMA)
                and canonical(state['identity']) == canonical(self.identity) and type(state['records']) is list,
                'output novelty state identity/schema differs')
        restored = type(self)(self.identity, near_policy=self._near_policy)
        for row in state['records']:
            fields = {'evaluation_id', 'identity', 'binding', 'program_ids', 'verdict'}
            require(type(row) is dict and set(row) == fields | ({'near_measurement'} if modern else set()),
                    'invalid output history record')
            evaluation_id = _id(row['evaluation_id'])
            require(evaluation_id not in restored._records, 'duplicate evaluation ID in output history')
            binding = validate_binding(row['binding'])
            if row['identity'] is not None:
                _bound_identity(row['identity'], binding)  # strict on restore, never downgrade corruption
            measurement = row.get('near_measurement')
            if modern:
                require(callable(measurement_validator), 'near-output restore requires immutable receipt validator')
                actual = measurement_validator(evaluation_id, deepcopy(row['identity']), deepcopy(binding))
                require(canonical(actual) == canonical(measurement),
                        'near-output checkpoint measurement differs from immutable receipt')
            elif measurement is not None:
                raise ValueError('near-output checkpoint measurement lacks output identity')
            saved = validate_verdict(row['verdict'])
            expected = restored.observe(evaluation_id, row['identity'], binding, measurement=measurement)
            # An alias can be bound after a verdict was frozen. Null stays truthful
            # to that earlier instant; a recorded alias must exist in prior history.
            if saved['match'] is not None:
                require(expected['match'] is not None, 'recorded repeat has no previous matching output')
                previous = restored._records.get(saved['match']['evaluation_id'])
                require(previous is not None, 'output match has no previous evaluation')
                alias = saved['match']['program_id']
                require(alias is None or alias in previous['program_ids'], 'unknown matched program alias')
                expected['match']['program_id'] = alias
                expected = sealed({k: v for k, v in expected.items() if k != 'sha256'})
            require(canonical(saved) == canonical(expected), 'output history verdict differs from prior evidence')
            aliases = row['program_ids']
            require(type(aliases) is list and len(aliases) == len(set(_id(x) for x in aliases)),
                    'invalid duplicate output program aliases')
            restored._records[evaluation_id]['verdict'] = saved
            for program_id in aliases:
                restored.bind_program(evaluation_id, program_id)
        with self._lock:
            self._records, self._by_output, self._by_scope = restored._records, restored._by_output, restored._by_scope
