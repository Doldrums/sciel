"""Inert complete-output/receipt tests. No generated program is executed."""
from copy import deepcopy
import hashlib
import json
import math
from pathlib import Path

import pytest

from sciel.engine.output_novelty import OutputNoveltyRegistry, is_repeat, validate_verdict
from sciel.eval.near_output import (POLICY_SCHEMA, distance, validate_measurement,
                                    validate_policy)
from sciel.eval.output_identity import adapt_near_output, adapt_output, canonical, digest, sealed
from test_output_identity import fixture


def policy(task, tolerance=1e-6):
    return dict(schema=POLICY_SCHEMA, task_hash=task.task_hash,
        evaluator_sha256=hashlib.sha256(Path(task.evaluator_path).read_bytes()).hexdigest(),
        adapter='canonical-json/1', units='declared fixture output units',
        numeric_abs_tolerance=tolerance)


def reseal(value):
    return sealed({k: v for k, v in value.items() if k != 'sha256'})


class History:
    def __init__(self, tmp_path, tolerance=1e-6):
        self.path = tmp_path
        task, _, _, _ = fixture(tmp_path)
        self.policy = policy(task, tolerance)
        self.registry = OutputNoveltyRegistry({'task_hash': task.task_hash, 'engine_contract': 'test/2'},
                                             near_policy=self.policy)
        self.receipts = {}

    def add(self, ident, output, score, island=0, **kwargs):
        source = hashlib.sha256(ident.encode()).hexdigest()
        task, _, outcome, binding = fixture(self.path, source=source, output=output, **kwargs)
        outcome.result.score = score
        binding.update(receipt_sha256=digest(dict(id=ident, output=output, score=score)), island_id=island)
        identity = adapt_output(task, source, outcome, binding)
        measurement = adapt_near_output(task, source, outcome, binding, identity=identity, policy=self.policy)
        assert measurement is not None
        self.receipts[ident] = deepcopy((task, outcome, binding))
        return self.registry.observe(ident, identity, binding, measurement=measurement)

    def verify_receipt(self, ident, identity, binding):
        # This pure fixture represents an authenticated receipt loader, not a
        # comparison against the checkpoint's own saved measurement.
        task, outcome, expected = deepcopy(self.receipts[ident])
        assert canonical(binding) == canonical(expected)
        actual_identity = adapt_output(task, binding['source_sha256'], outcome, binding)
        assert canonical(actual_identity) == canonical(identity)
        return adapt_near_output(task, binding['source_sha256'], outcome, binding,
                                 identity=actual_identity, policy=self.policy)

    def restore(self, state=None):
        result = OutputNoveltyRegistry(self.registry.identity, near_policy=self.policy)
        result.load_state_dict(state or json.loads(json.dumps(self.registry.state_dict())),
                               measurement_validator=self.verify_receipt)
        return result


@pytest.mark.parametrize('change', [dict(schema='unknown'), dict(task_hash='bad'),
    dict(evaluator_sha256='bad'), dict(adapter='score-only'), dict(units=True),
    dict(units=''), dict(units='a\x00b'), dict(numeric_abs_tolerance=True),
    dict(numeric_abs_tolerance=0), dict(numeric_abs_tolerance=-1),
    dict(numeric_abs_tolerance=math.inf), dict(numeric_abs_tolerance=math.nan),
    dict(numeric_abs_tolerance=10**1000), dict(unexpected='field')])
def test_policy_requires_explicit_trusted_scope_units_and_finite_positive_resolution(tmp_path, change):
    task, _, _, _ = fixture(tmp_path)
    with pytest.raises(ValueError):
        validate_policy(dict(policy(task), **change))
    assert validate_policy(None) is None


def test_typed_ordered_json_and_exact_rational_boundary(tmp_path):
    task, _, _, _ = fixture(tmp_path)
    p = policy(task, .125)
    assert distance({'b': [1., 'x'], 'a': True}, {'a': True, 'b': [1.125, 'x']}, p) == .125
    assert distance([1.], [math.nextafter(1.125, math.inf)], p) is None
    assert distance([1.], [math.nextafter(1.125, -math.inf)], p) < .125
    for a, b in [(True, 1), (1, 1.), ([1., 2.], [2., 1.]), ([1.], [1., 2.]),
                 ({'x': 1.}, {'y': 1.}), ('x', 'y'), (None, False)]:
        assert distance(a, b, p) is None
    assert distance(-0., 0., p) == 0.
    assert distance(-1e308, 1e308, p) is None
    with pytest.raises(ValueError):
        distance([math.nan], [0.], p)


def test_strongest_nearby_historical_score_not_selected_parent_or_first_match(tmp_path):
    h = History(tmp_path, .125)
    h.add('weak', [1.], .2, island=0)
    h.add('strong', [1.0625], .9, island=1)
    h.registry.bind_program('strong', 'p-evicted')
    # Better than the first/weak parent; still dominated by the full history.
    v = h.add('candidate', [1.03125], .8, island=9)
    assert v['status'] == 'non_improving_near_repeat' and not v['accept'] and is_repeat(v)
    assert v['match']['evaluation_id'] == 'strong'
    assert v['match']['program_id'] == 'p-evicted'
    assert v['near_output']['score'] == .8
    assert v['near_output']['match']['score'] == .9
    assert v['near_output']['match']['max_abs_delta'] == .03125
    assert validate_verdict(v) == v
    assert h.restore().state_dict() == h.registry.state_dict()


def test_any_strict_score_gain_survives_even_one_representable_step(tmp_path):
    h = History(tmp_path)
    h.add('base', [1.], 1.)
    v = h.add('improved', [math.nextafter(1., math.inf)], math.nextafter(1., math.inf))
    assert v['status'] == 'new_output' and v['accept']
    assert v['near_output']['match'] is None
    far = h.add('exploration', [2.], -100.)
    assert far['status'] == 'new_output'  # Low score alone is never a repeat.
    exact = h.add('exact', [1.], 2.)
    assert exact['status'] == 'exact_recorded_repeat'  # Existing exact policy unchanged.


def test_rejected_history_is_retained_and_nearness_is_not_collapsed_transitively(tmp_path):
    h = History(tmp_path, 1.)
    h.add('a', [0.], 3.)
    b = h.add('b', [.75], 2.)
    assert b['match']['evaluation_id'] == 'a'
    # C is outside A's radius, but the nonadmitted B remains an exact witness.
    h.registry = h.restore()
    c = h.add('c', [1.5], 1.)
    assert c['status'] == 'non_improving_near_repeat'
    assert c['match']['evaluation_id'] == 'b' and c['match']['program_id'] is None
    assert h.registry.summary()['non_improving_near_repeat'] == 2


def test_equal_best_scores_choose_stable_first_history_order(tmp_path):
    h = History(tmp_path, 1.)
    h.add('a', [0.], 1.)
    h.add('b', [1.5], 1.)
    v = h.add('c', [.75], .9)
    assert v['match']['evaluation_id'] == 'a'
    h.registry.bind_program('a', 'later-alias')
    assert h.restore().state_dict() == h.registry.state_dict()


@pytest.mark.parametrize('kwargs', [dict(reached='smoke'),
    dict(input_scope='artifact_sha256', input_sha='4' * 64),
    dict(seed_scope='input_seed', seed_sha='5' * 64),
    dict(reached='confirm', confirmed=True,
         confirmation=dict(trials=2, selected_trial=0, selection='lowest'))])
def test_near_outputs_never_cross_input_seed_or_stage_scope(tmp_path, kwargs):
    h = History(tmp_path)
    h.add('base', [1.], 1.)
    assert h.add('different-scope', [1.0000001], .5, **kwargs)['status'] == 'new_output'


@pytest.mark.parametrize('damage', ['payload', 'score_bool', 'receipt', 'policy', 'identity'])
def test_measurement_structural_and_identity_joins_are_strict(tmp_path, damage):
    h = History(tmp_path)
    h.add('base', [1.], 1.)
    row = h.registry.state_dict()['records'][0]
    m = row['near_measurement']
    if damage == 'payload': m['output'] = [2.]
    elif damage == 'score_bool': m['score'] = True
    elif damage == 'receipt': m['receipt_sha256'] = 'd' * 64
    elif damage == 'policy': m['policy_sha256'] = 'd' * 64
    elif damage == 'identity': m['identity_sha256'] = 'd' * 64
    with pytest.raises(ValueError):
        validate_measurement(reseal(m), identity=row['identity'], binding=row['binding'], policy=h.policy)


@pytest.mark.parametrize('damage', ['score', 'remove_measurement', 'remove_identity', 'policy', 'verdict'])
def test_cold_restore_reconstructs_receipts_not_self_consistent_checkpoint_hashes(tmp_path, damage):
    h = History(tmp_path)
    h.add('a', [1.], 1.)
    h.add('b', [1.0000001], .5)
    before = h.registry.state_dict()
    state = deepcopy(before)
    row = state['records'][1]
    if damage == 'score':
        row['near_measurement']['score'] = 2.
        row['near_measurement'] = reseal(row['near_measurement'])
    elif damage == 'remove_measurement': row['near_measurement'] = None
    elif damage == 'remove_identity':
        row['near_measurement'] = row['identity'] = None
        row['verdict'].update(identity=None, status='unsupported', accept=True, match=None, near_output=None)
        row['verdict'] = reseal(row['verdict'])
    elif damage == 'policy': state['identity']['near_output_policy']['numeric_abs_tolerance'] *= 2
    elif damage == 'verdict':
        row['verdict']['near_output']['match']['score'] += 1.
        row['verdict'] = reseal(row['verdict'])
    with pytest.raises((ValueError, AssertionError)):
        h.registry.load_state_dict(reseal(state), measurement_validator=h.verify_receipt)
    assert h.registry.state_dict() == before


def test_receipt_validator_required_and_restore_is_detached(tmp_path):
    h = History(tmp_path)
    v = h.add('a', [1.], 1.)
    with pytest.raises(ValueError, match='immutable receipt validator'):
        h.registry.load_state_dict(h.registry.state_dict())
    copy = deepcopy(h.registry)
    copy.bind_program('a', 'copy-only')
    assert h.registry.state_dict()['records'][0]['program_ids'] == []
    v['near_output']['score'] = 99.
    h.registry = h.restore()
    row = h.registry.state_dict()['records'][0]
    assert h.registry.observe('a', row['identity'], row['binding'],
                              measurement=row['near_measurement'])['near_output']['score'] == 1.


def test_adapter_uses_receipt_output_and_score_and_never_a_claimed_payload(tmp_path):
    task, _, outcome, binding = fixture(tmp_path, output={'points': [[.1, .2]], 'valid': True})
    p = policy(task)
    identity = adapt_output(task, binding['source_sha256'], outcome, binding)
    m = adapt_near_output(task, binding['source_sha256'], outcome, binding, identity=identity, policy=p)
    assert m['output'] == {'points': [[.1, .2]], 'valid': True} and m['score'] == .8
    assert m['receipt_sha256'] == binding['receipt_sha256']
    wrong = reseal(dict(identity, output_sha256='f' * 64))
    assert adapt_near_output(task, binding['source_sha256'], outcome, binding, identity=wrong, policy=p) is None
    assert adapt_near_output(task, binding['source_sha256'], outcome, binding,
                             policy=dict(p, evaluator_sha256='e' * 64)) is None
    assert adapt_near_output(task, binding['source_sha256'], outcome,
                             {k: v for k, v in binding.items() if k != 'receipt_sha256'}, policy=p) is None
    assert adapt_near_output(task, binding['source_sha256'], outcome, binding, policy=None) is None
    outcome.result.score = True
    assert adapt_near_output(task, binding['source_sha256'], outcome, binding, policy=p) is None


@pytest.mark.parametrize('before,after,delta', [(1e308, -1e308, None), (1., .5, -.5)])
def test_retry_feedback_keeps_finite_scores_when_subtraction_overflows(tmp_path, before, after, delta):
    from sciel.engine.worker import candidate_retry_prompt
    h = History(tmp_path)
    h.add('base', [1.], before)
    verdict = h.add('retry', [1.0000001], after)
    assert verdict['status'] == 'non_improving_near_repeat'
    original = {'output_novelty': verdict, 'quality_measured': True}
    frozen = deepcopy(original)
    prompt = candidate_retry_prompt('original prompt', 'exact previous reply', measured_repeat=original)
    encoded = prompt.split('MEASURED_REPEAT_JSON:\n', 1)[1]
    shown, _ = json.JSONDecoder().raw_decode(encoded)
    assert shown['canonical_score_delta_from_match'] == delta
    assert shown['output_novelty']['near_output']['score'] == after
    assert shown['output_novelty']['near_output']['match']['score'] == before
    assert ('score_delta_unavailable_reason' in shown) is (delta is None)
    assert original == frozen
    assert 'exact previous reply' in prompt and 'does not improve its canonical score' in prompt


def test_actual_p28_and_three_shrunk_outputs_use_generic_json_resolution():
    from sciel.eval.cascade import CascadeOutcome
    from sciel.eval.facade import EvalResult
    from sciel.eval.registry import load_task
    root = Path(__file__).resolve().parents[2]
    directory = root / 'tests/fixtures/output_near_repeat'
    task = load_task(str(root / 'tasks/oe/circle_packing'))
    p = dict(policy(task), adapter='native-circle-geometry/1', units='unit-square coordinate length')
    r = OutputNoveltyRegistry({'task_hash': task.task_hash, 'engine_contract': 'inert-v11-witness'}, near_policy=p)
    for name in ('p28', 'p37', 'p38', 'p40'):
        intent = json.loads((directory / (name + '.evaluation-intent.json')).read_text())
        receipt = json.loads((directory / (name + '.evaluation-receipt.json')).read_text())
        payload = receipt['payload']
        # Physical WAL uses ensure_ascii=False, distinct from output identity hashing.
        wire = lambda value: hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
            ensure_ascii=False, allow_nan=False).encode()).hexdigest()
        assert receipt['sha256'] == wire(payload)
        assert intent['request_sha256'] == payload['request_sha256'] == wire({
            'kind': intent['kind'], 'request': intent['request']})
        raw = payload['value']
        outcome = CascadeOutcome(EvalResult(**raw['result']), raw_first_score=raw['raw_first_score'],
                                reached=raw['reached'], confirmed=raw['confirmed'])
        source = hashlib.sha256(intent['request']['source'].encode()).hexdigest()
        binding = dict(task_hash=task.task_hash, source_sha256=source, island_id=0,
                       receipt_sha256=receipt['sha256'], request_sha256=intent['request_sha256'])
        identity = adapt_output(task, source, outcome, binding)
        measurement = adapt_near_output(task, source, outcome, binding, identity=identity, policy=p)
        assert identity is not None and measurement is not None
        v = r.observe(name, identity, binding, measurement=measurement)
        if name == 'p28':
            assert v['status'] == 'new_output'
            r.bind_program(name, name)
        else:
            assert v['status'] == 'non_improving_near_repeat'
            assert v['match']['program_id'] == 'p28'
            assert v['near_output']['score'] < v['near_output']['match']['score']
