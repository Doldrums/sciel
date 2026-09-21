"""Task-independent response envelopes; never repair scientific proposal content."""
import json
from types import SimpleNamespace

import pytest

from sciel.methods.proposals import (ACTION_SCHEMA, PROCEDURE, ProposalFormatError,
    parse_proposal, prompt_for, proposal_diagnostic, response_record)


@pytest.mark.parametrize('wrap', [lambda s: s, lambda s: '```json\n' + s + '\n```',
    lambda s: '\n  ```JSON\r\n' + s + '\r\n```  \n', lambda s: '```\n' + s + '\n```'])
def test_single_json_envelope_preserves_proposal_exactly(wrap):
    value = dict(kind='method', files={'method.py': 'def run(value):\n    return value\n'},
                 manifest={'kind': 'tool', 'examples': [{'note': '```json remains literal source data'}]},
                 experiment={'inputs': [0, False, None, {'message': '\n'}]})
    raw = wrap(json.dumps(value))
    assert parse_proposal(raw) == value
    receipt = response_record(SimpleNamespace(text=raw, model='fixture', cost=.001,
                                             prompt_tokens=10, completion_tokens=20))
    assert receipt['text'] == raw  # The provider receipt is never rewritten.
    assert receipt['cost'] == .001


@pytest.mark.parametrize('raw', [
    'Here is the result:\n```json\n{"kind":"stop","reason":"done"}\n```',
    '```json\n{"kind":"stop","reason":"done"}\n```\nDone.',
    '```json\n{"kind":"stop","reason":"done"}\n```\n```json\n{}\n```',
    '```python\n{"kind":"stop","reason":"done"}\n```',
    '```json {"kind":"stop","reason":"done"}```',
    '```json\n{"kind":"stop","reason":"done"}',
    '{"kind":"stop","reason":"done"} {"kind":"stop","reason":"again"}',
])
def test_chatter_multiple_objects_or_nonjson_fences_are_refused(raw):
    with pytest.raises(ProposalFormatError):
        parse_proposal(raw)


@pytest.mark.parametrize('payload', [
    '{"kind":"stop","kind":"method","reason":"x"}',
    '{"kind":"method","files":{},"manifest":{"kind":"tool","kind":"search"}}',
    '{"kind":"method","files":{},"manifest":{"value":NaN}}',
    '{"kind":"method","files":{},"manifest":{"value":Infinity}}',
    '{"kind":"method","files":{},"manifest":{"value":-Infinity}}',
    '{"kind":"method","files":{},"manifest":{"value":1e999}}',
])
def test_fence_unwrapping_preserves_strict_json_checks(payload):
    for raw in (payload, '```json\n' + payload + '\n```'):
        with pytest.raises(ProposalFormatError):
            parse_proposal(raw)


def test_wrong_method_wrapper_is_not_silently_repaired():
    raw = json.dumps({'method': {'kind': 'tool', 'files': {}, 'manifest': {}}})
    with pytest.raises(ProposalFormatError) as caught:
        parse_proposal('```json\n' + raw + '\n```')
    diagnostic = proposal_diagnostic(caught.value)
    assert diagnostic['code'] == 'missing_action_kind'
    assert diagnostic['field'] == 'kind'
    assert 'top-level siblings' in diagnostic['message']
    assert 'manifest.kind' in diagnostic['message']


def test_format_diagnostic_does_not_copy_private_response_snippet():
    private = 'private_provider_text_not_for_error'
    with pytest.raises(ProposalFormatError) as caught:
        parse_proposal('{\n"kind":"stop",\n' + private + '}')
    diagnostic = proposal_diagnostic(caught.value)
    assert diagnostic['code'] == 'invalid_json'
    assert diagnostic['line'] == 3
    assert private not in json.dumps(diagnostic)
    assert private not in json.dumps(proposal_diagnostic(RuntimeError(private)))


def test_raw_byte_limit_includes_fence_and_multibyte_characters():
    raw = '```json\n' + json.dumps({'kind': 'stop', 'reason': 'é'}, ensure_ascii=False) + '\n```'
    assert parse_proposal(raw, max_bytes=len(raw.encode()))['reason'] == 'é'
    with pytest.raises(ProposalFormatError) as caught:
        parse_proposal(raw, max_bytes=len(raw.encode()) - 1)
    assert proposal_diagnostic(caught.value)['code'] == 'response_too_large'


def test_prompt_has_actual_action_schema_and_isolated_dependency_contract():
    context = {'lab_id': 'arbitrary-task', 'proposal_index': 1}
    prompt = prompt_for(context)
    body = prompt.split('\nACTION_SCHEMA_JSON\n', 1)[1]
    schema, saved_context = body.split('\nCONTEXT_JSON\n', 1)
    assert json.loads(schema) == ACTION_SCHEMA
    assert json.loads(saved_context) == context
    assert {row['properties']['kind']['const'] for row in ACTION_SCHEMA['oneOf']} == {'method', 'experiment', 'solution', 'stop'}
    assert all(row['additionalProperties'] is False for row in ACTION_SCHEMA['oneOf'])
    assert PROCEDURE['version'] == 2
    assert 'm_<hash>.invoke(JSONvalue)' in prompt and 'import m_<hash>' in prompt
    assert 'private helper filenames' in prompt
    assert 'draft alone is not experimental progress' in prompt
    assert 'context.solution_input_schema' in prompt
    assert 'treatment:["$new"]' in prompt and 'control:[]' in prompt
    assert 'exactly one input' in prompt
    assert 'Adjacent pipeline output_schema and input_schema must be exactly equal' in prompt
    assert 'including optional fields and array order' in prompt


@pytest.mark.parametrize('payload,code', [
    ({'kind':'tool','files':{},'manifest':{}}, 'unsupported_action_kind'),
    ({'kind':'method','manifest':{}}, 'missing_action_field'),
    ({'kind':'method','files':{},'manifest':{},'score':42}, 'unsupported_action_field'),
    ({'kind':'method','files':[],'manifest':{}}, 'invalid_method_shape'),
    ({'kind':'stop','reason':''}, 'invalid_stop_reason'),
])
def test_action_shape_refusals_have_stable_actionable_diagnostics(payload, code):
    with pytest.raises(ProposalFormatError) as caught:
        parse_proposal(json.dumps(payload))
    assert proposal_diagnostic(caught.value)['code'] == code
