"""Offline native-schema and honest no-experiment decision contracts.

Model messages and transport are inert fixtures; no provider or generated tool runs.
"""
from copy import deepcopy
import hashlib
import json
import re

import pytest

from sciel.research_tools.controller import (
    _compact_tool_instructions, parse_action, response_format,
)
from sciel.research_tools.research_steps import TEXT_FIELDS, proposal_fields
from test_codex_responses import MODEL, events, provider
from test_research_tool_controller import IO, LLM, prepare, snapshot
from test_sequential_research_episode import build, finish


def static_finish():
    return finish(
        reason='The required saved starting state is absent; no experiment was run.',
        question='Does the saved starting state permit a controlled continuation?',
        experiment='No experiment was run because the required starting state is unavailable.',
        decision_if_supported='If the state becomes available, compare a continuation with its control.',
        decision_if_refuted='If the state remains unavailable, leave this comparison unresolved.',
        next_decision='Close this question as inconclusive until the required state is available.',
        conclusion='inconclusive', code='', name='', description='', tool_id='',
    )


@pytest.mark.parametrize('background,independent', [(False, False), (True, False), (True, True)])
def test_native_nonempty_patterns_match_semantic_text_fields_only(background, independent):
    schema = response_format(background, sequential=True,
                             independent_research=independent)['json_schema']['schema']
    assert schema['additionalProperties'] is False
    assert set(TEXT_FIELDS) <= set(schema['required'])
    for field in TEXT_FIELDS:
        contract = schema['properties'][field]
        assert contract['type'] == 'string' and contract['pattern'] == r'\S'
        for value in ('', ' ', '\t\r\n', '\u2003\u00a0'):
            assert re.search(contract['pattern'], value) is None
        assert re.search(contract['pattern'], 'No experiment was run.')
        assert re.search(contract['pattern'], '\n未知\n')
    for field in ('input_receipt_id', 'candidate_receipt_id'):
        contract = schema['properties'][field]
        assert contract['type'] == 'string'
        assert 'pattern' not in contract and 'minLength' not in contract
    assert 'question' not in response_format()['json_schema']['schema']['properties']


def test_missing_blank_whitespace_and_nonstrings_are_reported_together_without_mutation():
    proposal = {'action': 'finish', 'decision_if_supported': '',
                'decision_if_refuted': '\t \n', 'experiment': None, 'next_decision': 3,
                'conclusion': 'inconclusive'}
    before = deepcopy(proposal)
    with pytest.raises(ValueError) as caught:
        proposal_fields(proposal)
    assert str(caught.value) == '; '.join(
        field + ' must be a nonempty concrete research field' for field in TEXT_FIELDS)
    assert proposal == before  # No missing-field autofill or invented experiment.


@pytest.mark.parametrize('field', TEXT_FIELDS)
@pytest.mark.parametrize('value', ['', '\t\n\u2003'])
def test_each_blank_field_is_still_semantically_rejected(tmp_path, field, value):
    tools, _, _, _ = build(tmp_path, [])
    proposal = static_finish()
    proposal[field] = value
    with pytest.raises(ValueError, match=field + ' must be a nonempty concrete research field'):
        parse_action(json.dumps(proposal), tools.config)


@pytest.mark.parametrize('public_task,minimal', [(False, False), (True, False), (False, True)])
def test_every_compact_contract_explains_nonempty_closure_without_invented_execution(public_task, minimal):
    text = _compact_tool_instructions(True, public_task=public_task, minimal=minimal)
    for field in TEXT_FIELDS:
        assert field in text
    assert 'nonempty text, even finish/skip' in text
    assert 'If no experiment ran, say so and why' in text
    assert 'Never invent results/execution' in text
    assert 'finish closes without executing next_decision advice' in text


@pytest.mark.parametrize('budget', [8000, 64000])
@pytest.mark.parametrize('independent', [False, True])
def test_blank_finish_repair_names_all_fields_and_accepts_honest_static_close(tmp_path, budget, independent):
    tools, backend, _, costs = build(tmp_path, [], max_prompt_chars=budget,
        independent_research=independent)
    current = snapshot()
    valid = static_finish()
    # Observed failure shape: a prose finish with no code/invocation and blank
    # structured decision fields. Claimed past tool execution is not accepted.
    invalid = {**valid, **{field: '' for field in TEXT_FIELDS},
               'reason': 'I already proposed a tool; no more action is needed.'}
    def repair(prompt):
        feedback = prompt.split('PREVIOUS ATTEMPT FEEDBACK:\n', 1)[1]
        for field in TEXT_FIELDS:
            assert field + ' must be a nonempty concrete research field' in feedback
        return valid
    llm = LLM([invalid, repair])
    plan = prepare(tools, current)
    if independent:
        from test_research_continuation_queue import lease
        assert plan is None
        _, current, plan = lease(tools)
    episode = tools.execute(plan, current, llm, io=IO())
    assert not backend.calls and not episode['receipt_ids'] and not episode['tool_ids']
    assert costs == [.03, .03] and episode['known_llm_cost'] == pytest.approx(.06)
    state = episode['research_state']
    assert state['status'] == 'closed' and state['conclusion'] == 'inconclusive'
    assert state['scientific_steps'] == 0
    assert [step['status'] for step in state['steps']] == ['proposal_invalid', 'closed']
    assert {field: state[field] for field in TEXT_FIELDS} == {field: valid[field] for field in TEXT_FIELDS}
    rejected = tools._read_artifact(state['steps'][0]['proposal_id'], 'proposal')
    assert json.loads(rejected['response']['text']) == invalid
    for prompt in llm.prompts:
        assert len(prompt) <= budget
        assert 'nonempty text, even finish/skip' in prompt
        assert 'If no experiment ran, say so and why' in prompt
        assert 'Never invent results/execution' in prompt
    assert all(card['observations']['status'] != 'success' for card in episode['cards'])
    tools.commit(episode)
    saved = deepcopy(tools.state())
    tools.load_state(saved)
    assert tools.state() == saved


def test_actual_codex_body_forwards_nonempty_native_schema_unchanged_and_binds_hash():
    contract = response_format(sequential=True)
    before = deepcopy(contract)
    payload = json.dumps(static_finish())
    llm, opener = provider(events(payload))  # In-memory SSE; no real credentials/network.
    response = llm.generate('Research contract fixture', MODEL, response_format=contract)
    assert response.text == payload and len(opener.calls) == 1
    request = opener.calls[0][0]
    body = json.loads(request.data)
    assert body['text']['format'] == {'type': 'json_schema', **contract['json_schema']}
    assert contract == before
    metadata = llm.request_metadata('Research contract fixture', MODEL, response_format=contract)
    assert metadata['body_sha256'] == hashlib.sha256(request.data).hexdigest()
    old = deepcopy(contract)
    for field in TEXT_FIELDS:
        old['json_schema']['schema']['properties'][field].pop('pattern')
    assert llm.request_metadata('Research contract fixture', MODEL,
        response_format=old)['body_sha256'] != metadata['body_sha256']
