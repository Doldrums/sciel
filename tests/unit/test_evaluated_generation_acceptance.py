"""Pure externally pinned bindings; the tiny G object is deliberately synthetic."""
import copy
import hashlib

import pytest

from sciel.hypothesis import evaluated_generation_acceptance as acceptance
from sciel.knowledge_base.catalogs import canonical_sha


def example():
    protocol = acceptance.make_protocol('inert-evaluated-numeric', [17, 19, 23])
    code = 'SYNTHETIC DATA; not executable source'
    result = dict(schema=acceptance.RESULT_SCHEMA, eligible=True, status='eligible', code=code,
        source_sha256=hashlib.sha256(code.encode()).hexdigest(), proof={'synthetic': True},
        plan=dict(target={'synthetic': True}, policy_sha256='a'*64))
    decision = dict(schema=acceptance.SCHEMA, decision='accepted', generation_result_sha256=canonical_sha(result),
        generated_source_sha256=result['source_sha256'], source_proof_sha256=canonical_sha(result['proof']),
        target_sha256=canonical_sha(result['plan']['target']), policy_sha256='a'*64,
        audit_sha256='b'*64, criteria_sha256='c'*64, numeric_protocol_sha256=canonical_sha(protocol),
        issuer='SYNTHETIC UNIT AUTHORITY; no dispatch permitted')
    return result, protocol, decision


def test_whole_addendum_and_independent_decision_are_detached():
    result, protocol, decision = example()
    approved = acceptance.validate_decision(decision, generation_result=result,
        numeric_protocol=protocol, expected_sha256=canonical_sha(decision))
    assert approved == decision and approved is not decision
    assert protocol['schema'] == 'sciel-evaluated-target-numeric/1'
    assert protocol['schedule'] == [dict(index=i, seed=[17,19,23][i//2], role=role) for i,role in enumerate(
        ['comparator','challenger','challenger','comparator','comparator','challenger'])]
    assert protocol['max_candidate_calls'] == protocol['max_certifier_calls'] == 6
    assert protocol['provider_calls'] == 0 and protocol['no_retry'] is protocol['no_resume'] is True
    assert protocol['outer_series_wall_cap_seconds'] == 360
    approved['issuer'] = 'mutation'
    assert decision['issuer'] != 'mutation'


@pytest.mark.parametrize('key', ['generation_result_sha256','generated_source_sha256','source_proof_sha256',
                                'target_sha256','policy_sha256','numeric_protocol_sha256'])
def test_rehashed_wrong_binding_still_rejected(key):
    result, protocol, decision = example()
    decision[key] = 'f'*64
    with pytest.raises(ValueError, match='differs'):
        acceptance.validate_decision(decision, generation_result=result,
            numeric_protocol=protocol, expected_sha256=canonical_sha(decision))


def test_self_rehash_cannot_replace_independently_pinned_host_decision():
    result, protocol, decision = example()
    expected = canonical_sha(decision)
    decision['audit_sha256'] = 'd'*64
    with pytest.raises(ValueError, match='external decision digest'):
        acceptance.validate_decision(decision, generation_result=result,
            numeric_protocol=protocol, expected_sha256=expected)


@pytest.mark.parametrize('damage', ['rejected','extra','issuer','bool_digest','ineligible','wrong_family'])
def test_invalid_or_unaccepted_decision_and_G_shape(damage):
    result, protocol, decision = example()
    if damage == 'rejected': decision['decision'] = 'rejected'
    if damage == 'extra': decision['approval'] = True
    if damage == 'issuer': decision['issuer'] = ' '*129
    if damage == 'bool_digest': decision['audit_sha256'] = True
    if damage == 'ineligible': result['eligible'] = False
    if damage == 'wrong_family': result['schema'] = 'sciel-component-target-generation-result/1'
    with pytest.raises(ValueError):
        acceptance.validate_decision(decision, generation_result=result,
            numeric_protocol=protocol, expected_sha256=canonical_sha(decision))


@pytest.mark.parametrize('damage', ['duplicate_seed','bool_seed','large_seed','role','cap','limits','retry','slim'])
def test_rehashed_execution_contract_cannot_be_relaxed(damage):
    result, protocol, decision = example()
    if damage == 'duplicate_seed': protocol['seeds'][1] = protocol['seeds'][0]
    if damage == 'bool_seed': protocol['seeds'][0] = True
    if damage == 'large_seed': protocol['seeds'][0] = 2**32
    if damage == 'role': protocol['schedule'][0]['role'] = 'challenger'
    if damage == 'cap': protocol['outer_series_wall_cap_seconds'] = 361.
    if damage == 'limits': protocol['limits']['candidate_wall_seconds'] = 31.
    if damage == 'retry': protocol['no_retry'] = False
    if damage == 'slim': protocol = {k:protocol[k] for k in ('schema','study_id','seeds','limits','runtime','quality_gate')}
    decision['numeric_protocol_sha256'] = canonical_sha(protocol)
    with pytest.raises(ValueError):
        acceptance.validate_decision(decision, generation_result=result,
            numeric_protocol=protocol, expected_sha256=canonical_sha(decision))


def test_protocol_integer_float_identity_is_not_python_equality():
    _, protocol, _ = example()
    altered = copy.deepcopy(protocol)
    altered['no_retry'] = 1
    assert altered == protocol
    with pytest.raises(ValueError, match='fixed numerical protocol differs'):
        acceptance.validate_protocol(altered)
