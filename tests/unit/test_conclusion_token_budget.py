"""A C-only output budget does not change admission, replay or other call kinds."""
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest

from sciel.config import apply_defaults, load_config
from sciel.engine.research import ResearchController
from test_conclusion_format_repair import setup, response
from test_research_aux_failures import setup as plain_setup, response as plain_response


@pytest.mark.parametrize('value', [True, False, 0, -1, 1.5, '4096', [], {}])
def test_bad_conclusion_budget_rejected_by_config_and_direct_controller(value):
    with pytest.raises(ValueError, match='conclusion_max_tokens'):
        apply_defaults({'research': {'conclusion_max_tokens': value}})
    c, model, _, charges = setup([])
    invalid = {**c.config, 'conclusion_max_tokens': value}
    with pytest.raises(ValueError, match='conclusion_max_tokens'):
        ResearchController(invalid, model, c.log, charges.append)
    assert not model.calls and not charges


@pytest.mark.parametrize('options', [{}, {'conclusion_max_tokens': None}])
def test_omitted_or_null_budget_keeps_original_transport_default(options):
    c, model, events, _ = setup([response()], **options)
    c.complete_step()
    assert 'max_tokens' not in model.calls[0][1]
    assert 'max_tokens' not in next(e for e in events if e['phase'] == 'aux_start')
    assert c.config.get('conclusion_max_tokens') is None


def test_conclusion_and_single_repair_share_cap_and_reflection_does_not():
    c, model, events, charges = setup([response(bad_ref=True), response()], conclusion_max_tokens=4096)
    c.complete_step()
    assert [opts['max_tokens'] for _, opts in model.calls] == [4096, 4096]
    assert charges == [.03, .03] and c.conclusions['r000001a1']['status'] == 'parsed'
    assert [e['max_tokens'] for e in events if e['phase'] == 'aux_start'] == [4096, 4096]
    state = c.state()
    restored, _, _, _ = setup([], conclusion_max_tokens=4096)
    restored.queue = []
    restored.load_state(state)
    assert restored.state() == state
    reflection = c.prepare_aux('reflection', 'inert reflection', 1)
    assert 'max_tokens' not in reflection['options']


def test_cap_without_response_format_keeps_plain_receipt_semantics():
    c, model, rows, charges = plain_setup([])
    c.config['conclusion_max_tokens'] = 4096
    received = []
    def generate(prompt, model, **options):
        received.append(deepcopy(options))
        return plain_response()  # Legacy unstructured receipts may lack finish metadata.
    model.generate = generate
    assert c._aux('conclusion', 'fixture', 'r1') == '{}'
    assert received == [{'temperature': .2, 'max_tokens': 4096}]
    start, result = rows
    assert start['max_tokens'] == 4096 and 'response_format' not in start
    assert 'auxiliary_schema' not in start and 'structured_response_usable' not in result
    assert charges == [.03]


def test_known_length_receipt_is_charged_skipped_and_never_repaired():
    c, model, events, charges = setup([], conclusion_max_tokens=4096, incomplete_auxiliary='skip')
    calls = []
    def generate(*args, **options):
        calls.append(options)
        return NS(text=response(lessons=True), model='fixture', cost=.03,
            prompt_tokens=10, completion_tokens=4096, finish_reason='length', refusal=None)
    model.generate = generate
    c.complete_step()
    assert len(calls) == 1 and calls[0]['max_tokens'] == 4096
    assert charges == [.03] and c.conclusions['r000001a1']['status'] == 'skipped'
    assert not c.knowledge.lessons and not any(e['phase'] == 'conclusion_repair' for e in events)
    skip = next(e for e in events if e['phase'] == 'aux_skipped')
    assert skip['reason'] == 'response_length' and skip['current_call_cost_unknown'] is False
    restored, fresh, _, _ = setup([], conclusion_max_tokens=4096, incomplete_auxiliary='skip')
    restored.queue = []
    restored.load_state(c.state())
    restored.complete_step()
    assert fresh.calls == []


@pytest.mark.parametrize('name', ['circle_parallel_gemini', 'circle_parallel_gemini_rl_full',
    'circle_parallel_gemini_rl_presets_vq', 'circle_parallel_gemini_background_tools'])
def test_all_live_profiles_inherit_only_conclusion_cap(name):
    config = load_config('configs/' + name + '.yaml')
    assert config['research']['conclusion_max_tokens'] == 4096
    assert config['llm']['max_tokens'] == 32000
    assert 'conclusion_max_tokens' not in config.get('research_tools', {})
