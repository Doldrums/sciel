"""Audit declarations bind names and preserve ordinary generation validators."""
from copy import deepcopy
import hashlib
import json

import pytest

from sciel.hypothesis import parse_hypothesis
from sciel.hypothesis.intervention_claim import (
    SCHEMA, InterventionClaimError, extract, instruction, parse, validate,
)


CATALOG = [dict(id=ident, kind=kind, local_ast_sha256='a'*64, subtree_ast_sha256='b'*64)
           for ident, kind in [('<module>','module'), ('construct_packing','function'),
                               ('Solver','class'), ('Solver.polish','function'),
                               ('construct_packing.local','function'), ('helper#2','function')]]
HYPOTHESIS = dict(hypothesis='A declared change may improve quality.', predicted_low=-.1,
                  predicted_high=.1, reasoning='The mechanism remains an untested interpretation.', confidence=.4)
CODE = '```python\r\ndef construct_packing():\r\n    return "INTERVENTION_JSON: is source data"\r\n```\r\n'
ORDINARY = 'HYPOTHESIS_JSON:\n'+json.dumps(HYPOTHESIS)+'\n'+CODE


def claim():
    return dict(schema=SCHEMA, parent_unit_ids=['construct_packing'],
        changed_units=[dict(parent_unit_id='construct_packing', proposed_name='construct_packing', change='modify')],
        claim_type='quality', mechanism='Change the initial geometry while retaining the optimizer.',
        speed_used_for_search=False, search_effort_link=None,
        limitations=['The effect on repeated evaluations is unknown.'])


def response(value):
    return 'INTERVENTION_JSON:\n'+json.dumps(value)+'\n'+ORDINARY


def test_separate_block_preserves_hypothesis_validator_and_exact_code_bytes():
    raw = response(claim())
    assert parse_hypothesis(raw) is None, 'ordinary validator must remain strict'
    result = extract(raw, CATALOG)
    assert result['error'] is None and result['claim'] == claim()
    assert result['declared_targets'] == ['construct_packing']
    assert result['response_without_intervention'] == ORDINARY
    assert result['response_without_intervention'].endswith(CODE)
    assert parse_hypothesis(result['response_without_intervention']).text == HYPOTHESIS['hypothesis']
    block = raw[:raw.index('HYPOTHESIS_JSON:')]
    assert result['block_sha256'] == hashlib.sha256(block.encode()).hexdigest()
    result['claim']['limitations'].append('Detached')
    assert parse(raw, CATALOG)['claim'] == claim()


def test_block_after_hypothesis_and_before_code_is_supported_without_reordering():
    prefix = 'HYPOTHESIS_JSON:\n'+json.dumps(HYPOTHESIS)+'\n'
    raw = prefix+'  INTERVENTION_JSON: '+json.dumps(claim())+'\n'+CODE
    result = extract(raw, CATALOG)
    assert result['error'] is None
    assert result['response_without_intervention'] == ORDINARY


@pytest.mark.parametrize('body, code', [
    ('{"schema":', 'invalid_json'),
    ('{"schema":"x","schema":"y"}', 'invalid_json'),
    ('{"x":NaN}', 'invalid_json'),
    ('{"x":Infinity}', 'invalid_json'),
    (json.dumps(claim())+' trailing prose', 'invalid_json'),
    ('[]', 'invalid_fields'),
    ('x'*8193, 'block_too_long'),
    ('['*1100+']'*1100, 'invalid_json'),
])
def test_invalid_declaration_still_preserves_valid_hypothesis_and_candidate(body, code):
    raw = 'INTERVENTION_JSON:\n'+body+'\n'+ORDINARY
    result = extract(raw, CATALOG)
    assert result['error']['code'] == code
    assert result['claim'] is None and result['declared_targets'] is None
    assert result['response_without_intervention'] == ORDINARY
    assert parse_hypothesis(result['response_without_intervention']) is not None


def test_duplicate_blocks_are_not_accepted_but_code_is_not_excluded():
    block = 'INTERVENTION_JSON:\n'+json.dumps(claim())+'\n'
    result = extract(block+block+ORDINARY, CATALOG)
    assert result['error']['code'] == 'multiple_blocks'
    assert result['response_without_intervention'] == ORDINARY
    assert result['declared_targets'] is None


def test_no_block_is_inferred_from_candidate_code_or_prose():
    for raw in [ORDINARY, 'My INTERVENTION_JSON: '+json.dumps(claim())+'\n'+ORDINARY,
                ORDINARY+'INTERVENTION_JSON:\n'+json.dumps(claim()),
                '```python\n# kept\nINTERVENTION_JSON:\n'+json.dumps(claim())+'\n```']:
        result = extract(raw, CATALOG)
        assert result['error']['code'] == 'missing_block'
        assert result['response_without_intervention'] == raw


def test_code_marker_and_unrelated_protocol_blocks_are_never_removed():
    code = '```python\ns = """\nINTERVENTION_JSON:\nraw source\n"""\n```\n'
    raw = 'INTERVENTION_JSON:\n'+json.dumps(claim())+'\nOTHER_JSON: {}\n'+ORDINARY.replace(CODE,code)
    result = extract(raw, CATALOG)
    assert result['error'] is None
    assert result['response_without_intervention'].startswith('OTHER_JSON: {}\n')
    assert result['response_without_intervention'].endswith(code)
    assert parse_hypothesis(result['response_without_intervention']) is None


def test_exact_names_add_remove_and_rename_do_not_infer_semantic_identity():
    value = claim()
    value['parent_unit_ids'] = ['helper#2', 'Solver.polish']
    value['changed_units'] = [dict(parent_unit_id='helper#2',proposed_name='helper#2',change='modify'),
        dict(parent_unit_id='Solver.polish',proposed_name=None,change='remove'),
        dict(parent_unit_id=None,proposed_name='Solver.finish',change='add')]
    result = extract(response(value), CATALOG)
    assert result['error'] is None
    assert result['declared_targets'] == ['helper#2','Solver.polish','Solver.finish']
    value['parent_unit_ids'][0] = 'helper'
    with pytest.raises(InterventionClaimError) as caught:
        validate(value, CATALOG)
    assert caught.value.diagnostic['code'] == 'unknown_parent_unit'


def test_add_only_has_empty_parent_ids_but_not_empty_audit_targets():
    value = claim();value['parent_unit_ids'] = []
    value['changed_units'] = [dict(parent_unit_id=None,proposed_name='new_helper',change='add')]
    assert extract(response(value), CATALOG)['declared_targets'] == ['new_helper']


def test_sixteen_planned_units_are_allowed_without_restricting_actual_source_changes():
    catalog=[dict(id='unit'+str(i),kind='function') for i in range(17)]
    value=claim();value['parent_unit_ids']=[unit['id'] for unit in catalog[:16]]
    value['changed_units']=[dict(parent_unit_id=ident,proposed_name=ident,change='modify')
                            for ident in value['parent_unit_ids']]
    assert len(extract(response(value),catalog)['declared_targets'])==16
    text=instruction(catalog)
    assert 'MUST NOT constrain how many source units you may edit' in text
    assert 'Additional or unlisted source edits are audited' in text
    assert '350 characters or fewer, hard limit 600' in text
    value['changed_units'].append(dict(parent_unit_id=None,proposed_name='added',change='add'))
    invalid=extract(response(value),catalog)
    assert invalid['error']['code']=='invalid_changed_units'
    assert invalid['response_without_intervention']==ORDINARY


@pytest.mark.parametrize('mutate, expected', [
    (lambda v:v.update(extra=True), 'invalid_fields'),
    (lambda v:v.update(schema='other'), 'invalid_schema'),
    (lambda v:v.update(parent_unit_ids=['construct_packing','construct_packing']), 'duplicate_parent_unit'),
    (lambda v:v.update(parent_unit_ids=[]), 'parent_set_mismatch'),
    (lambda v:v.update(changed_units=[]), 'invalid_changed_units'),
    (lambda v:v.update(changed_units=v['changed_units']*2), 'duplicate_changed_unit'),
    (lambda v:v['changed_units'][0].update(change='rewrite'), 'invalid_change_kind'),
    (lambda v:v['changed_units'][0].update(proposed_name='renamed'), 'invalid_proposed_name'),
    (lambda v:v['changed_units'][0].update(extra='unknown'), 'invalid_change_fields'),
    (lambda v:v.update(claim_type='causally_proven'), 'invalid_claim_type'),
    (lambda v:v.update(mechanism='a'*601), 'invalid_mechanism'),
    (lambda v:v.update(mechanism=''), 'invalid_mechanism'),
    (lambda v:v.update(speed_used_for_search=1), 'invalid_speed_flag'),
    (lambda v:v.update(speed_used_for_search=True), 'invalid_speed_flag'),
    (lambda v:v.update(limitations=[]), 'invalid_limitations'),
    (lambda v:v.update(limitations=['a'*241]), 'invalid_limitations'),
    (lambda v:v.update(search_effort_link={}), 'unexpected_search_effort_link'),
])
def test_strict_bounded_claim_fields(mutate, expected):
    value = claim();mutate(value)
    result = extract(response(value), CATALOG)
    assert result['error']['code'] == expected
    assert result['response_without_intervention'] == ORDINARY


def test_speed_to_search_requires_explicit_link_not_keyword_classification():
    value = claim();value['claim_type'] = 'speed';value['speed_used_for_search'] = True
    assert extract(response(value), CATALOG)['error']['code'] == 'invalid_search_effort_link'
    value['search_effort_link'] = dict(resource_saved='Less time per radius update.',
        additional_work='More restarts only if the restart loop is deadline-bound.',
        budget_condition='A fixed restart count would prevent that increase.')
    assert validate(value, CATALOG) == value
    value['claim_type'] = 'search_effort';value['speed_used_for_search'] = False
    assert validate(value, CATALOG) == value
    # Words cannot silently turn an explicit quality claim into a speed gate.
    value = claim();value['mechanism'] = 'Faster speed enables more search and restarts.'
    assert validate(value, CATALOG) == value
    assert 'semantic' in instruction(CATALOG)


def test_catalog_identity_and_input_mutation_are_caller_owned_and_explicit():
    parent, value = deepcopy(CATALOG), claim()
    text = instruction(parent)
    assert all(item['id'] in text for item in parent)
    assert 'HYPOTHESIS_JSON' in text and 'before evaluation' in text
    assert 'invalid declaration' in text
    assert validate(value,parent) == value
    assert parent == CATALOG and value == claim()
    with pytest.raises(InterventionClaimError, match='unique'):
        instruction(parent+[parent[0]])
    bad = extract(response(value), [{'id':'wrong scope','kind':'function'}])
    assert bad['error']['code'] == 'invalid_parent_catalog'


@pytest.mark.parametrize('value, expected', [(None,'response_not_text'), ('x'*262145,'response_too_long')])
def test_response_limits_do_not_silently_rewrite_unknown_inputs(value, expected):
    result = extract(value,CATALOG)
    assert result['error']['code'] == expected
    assert result['response_without_intervention'] == value
