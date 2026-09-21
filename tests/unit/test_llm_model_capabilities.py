"""Explicit transport capability policy; fake SDKs, no provider/model execution."""
from contextlib import nullcontext
import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sciel.config import apply_defaults
from sciel.campaign.budget import BudgetedLLM, CampaignStopped, CostLedger
from sciel.engine import checkpoint
from sciel.engine.factory import build_llm
from sciel.engine.llm.client import EnsembleLLM, LLMError, validate_model_capabilities
from sciel.engine.llm.trace import TracedLLM
from sciel.engine.parallel import WaveLLM
from sciel.engine.parallel_calls import CallLedger, CooperativeIO


MODEL = 'fixture/model'
CAPS = {MODEL: {'temperature': False}}
SCHEMA = {'type': 'json_schema', 'json_schema': {'name': 'exact', 'strict': True,
    'schema': {'type': 'object', 'properties': {'answer': {'type': 'string'}},
               'required': ['answer'], 'additionalProperties': False}}}
RATES = {'prompt': '.000001', 'completion': '.000002', 'request': '.000003'}


def provider(tmp_path, kind='ensemble', capabilities=None):
    sdk = Mock()
    sdk.chat.completions.create.return_value = SimpleNamespace(id='returned',
        choices=[SimpleNamespace(finish_reason='stop', message=SimpleNamespace(
            content='{"answer":"known"}', refusal=None))],
        usage=SimpleNamespace(cost=.0001, prompt_tokens=7, completion_tokens=9))
    cfg = {'models': [MODEL, 'other'], 'weights': [1, 1], 'max_tokens': 100,
           'api_base': 'https://openrouter.ai/api/v1', 'temperature': .7}
    if capabilities is not None:
        cfg['model_capabilities'] = capabilities
    if kind == 'ensemble':
        instance = EnsembleLLM.__new__(EnsembleLLM)
        instance.models, instance.weights = cfg['models'], cfg['weights']
        instance.temperature, instance.max_tokens, instance.max_retries = .7, 100, 1
        instance.api_base, instance._client = cfg['api_base'], sdk
        instance.model_capabilities = validate_model_capabilities(
            cfg.get('model_capabilities', {}), instance.models)
    else:
        instance = BudgetedLLM(cfg, {'models': [{'id': model, 'pricing': RATES}
            for model in cfg['models']]}, CostLedger(tmp_path / 'cost.json', 1),
            'run', 'unused', sdk)
    return instance, sdk


@pytest.mark.parametrize('kind', ['ensemble', 'budgeted'])
@pytest.mark.parametrize('role,temperature,structured', [
    ('G', .9, False), ('idea', .5, True), ('C', .2, True),
    ('R', .2, True), ('research_tool', .7, True)])
def test_all_roles_omit_only_temperature_and_trace_actual_wire(tmp_path, kind, role, temperature, structured):
    instance, sdk = provider(tmp_path, kind, CAPS)
    options = {'max_tokens': 64, **({'response_format': copy.deepcopy(SCHEMA)} if structured else {})}
    trace = TracedLLM(instance, tmp_path, lambda _: {'role': role})
    response = trace.generate('exact prompt', MODEL, temperature, **options)
    request, terminal = [json.loads(x) for x in (tmp_path / 'artifacts/llm_calls.jsonl').read_text().splitlines()]
    wire = sdk.chat.completions.create.call_args.kwargs
    assert 'temperature' not in wire and wire['max_tokens'] == 64
    assert wire['messages'] == [{'role': 'user', 'content': 'exact prompt'}]
    assert request['temperature'] == temperature
    assert request['transport_parameters'] == {'schema': 'sciel-llm-transport-parameters/1',
        'temperature': {'enabled': False, 'sent': False, 'value': None}}
    assert response.cost == terminal['cost'] == .0001
    assert terminal['current_call_cost_unknown'] is False
    assert sdk.chat.completions.create.call_count == 1
    if structured:
        assert wire['response_format'] == SCHEMA == options['response_format']
        assert wire['extra_body']['provider']['require_parameters'] is True
        assert wire['extra_body']['provider']['allow_fallbacks'] is False
    if kind == 'budgeted':
        assert len(json.loads((tmp_path / 'cost.json').read_text())['calls']) == 1


@pytest.mark.parametrize('kind', ['ensemble', 'budgeted'])
@pytest.mark.parametrize('caps', [None, {}, {MODEL: {'temperature': True}}, {'other': {'temperature': False}}])
@pytest.mark.parametrize('requested,expected', [(None, .7), (0., 0.), (.2, .2)])
def test_default_true_and_unlisted_models_preserve_temperature(tmp_path, kind, caps, requested, expected):
    instance, sdk = provider(tmp_path, kind, caps)
    TracedLLM(instance, tmp_path, lambda _: {}).generate('p', MODEL, requested)
    assert sdk.chat.completions.create.call_args.kwargs['temperature'] == expected
    request = json.loads((tmp_path / 'artifacts/llm_calls.jsonl').read_text().splitlines()[0])
    if not caps:
        assert 'transport_parameters' not in request
    else:
        assert request['transport_parameters']['temperature'] == {
            'enabled': True, 'sent': True, 'value': expected}


@pytest.mark.parametrize('bad', [None, [], False, {MODEL: {}}, {MODEL: {'temperatur': False}},
    {MODEL: {'temperature': 0}}, {MODEL: {'temperature': 'false'}},
    {MODEL: {'temperature': None}}, {MODEL: {'temperature': False, 'top_p': False}},
    {'typo/model': {'temperature': False}}])
def test_invalid_configuration_fails_before_provider_creation(bad, monkeypatch):
    sdk = Mock()
    monkeypatch.setattr('openai.OpenAI', sdk)
    with pytest.raises(ValueError, match='model_capabilities'):
        apply_defaults({'llm': {'models': [MODEL], 'model_capabilities': bad}})
    sdk.assert_not_called()


def test_partial_map_detached_factory_wiring_and_primary_model(monkeypatch):
    configured = copy.deepcopy(CAPS)
    cfg = apply_defaults({'llm': {'primary_model': MODEL, 'model_capabilities': configured}})
    configured[MODEL]['temperature'] = True
    assert cfg['llm']['model_capabilities'] == CAPS
    monkeypatch.setattr('sciel.engine.factory.load_env', lambda: None)
    monkeypatch.setenv('OPENROUTER_API_KEY', 'unused')
    sdk = Mock()
    monkeypatch.setattr('openai.OpenAI', sdk)
    actual = build_llm(cfg)
    assert actual.model_capabilities == CAPS and actual.models == [MODEL]
    assert sdk.call_args.kwargs['max_retries'] == 0


def test_model_id_string_cannot_be_used_as_substring_membership():
    with pytest.raises(ValueError, match='configured model IDs'):
        apply_defaults({'llm': {'models': MODEL, 'model_capabilities': CAPS}})


@pytest.mark.parametrize('kind', ['ensemble', 'budgeted'])
def test_failure_unknown_cost_no_retry_or_schema_relaxation(tmp_path, kind):
    instance, sdk = provider(tmp_path, kind, CAPS)
    sdk.chat.completions.create.side_effect = RuntimeError('unavailable endpoint')
    with pytest.raises((LLMError, CampaignStopped)):
        TracedLLM(instance, tmp_path, lambda _: {'role': 'C'}).generate(
            'p', MODEL, .2, response_format=SCHEMA)
    assert sdk.chat.completions.create.call_count == 1
    wire = sdk.chat.completions.create.call_args.kwargs
    assert wire['response_format'] == SCHEMA and 'temperature' not in wire
    row = json.loads((tmp_path / 'artifacts/llm_calls.jsonl').read_text().splitlines()[-1])
    assert row['event'] == 'error' and row['cost'] is None and row['current_call_cost_unknown']


def test_parallel_receipt_replays_same_policy_and_refuses_changed_effective_parameters(tmp_path):
    instance, sdk = provider(tmp_path, capabilities=CAPS)
    engine = SimpleNamespace(run_dir=tmp_path, _parallel_wave=1,
        _llm_trace_context=lambda _: {'role': 'C', 'reference': 'r1', 'experiment_id': None})
    capacity = SimpleNamespace(acquire=lambda *args, **kwargs: nullcontext())
    def run():
        io = CooperativeIO(CallLedger(tmp_path / 'wave', capacity, 'local'), 1)
        wave = WaveLLM(engine, instance, io)
        try:
            return io.run([('C:r1', lambda: wave.generate('p', MODEL, .2, response_format=SCHEMA))])
        finally:
            io.close()
    first, errors = run()
    assert not errors and first['C:r1'].cost == .0001
    replay, errors = run()
    assert not errors and replay == first and sdk.chat.completions.create.call_count == 1
    intent = json.loads(next((tmp_path / 'wave').rglob('intent.json')).read_text())
    trace = json.loads((tmp_path / 'artifacts/llm_calls.jsonl').read_text().splitlines()[0])
    assert intent['request']['transport_parameters'] == trace['transport_parameters']
    instance.model_capabilities[MODEL]['temperature'] = True
    changed, errors = run()
    assert not changed and errors and sdk.chat.completions.create.call_count == 1


def test_checkpoint_identity_pins_policy_and_actual_provider(tmp_path):
    instance, _ = provider(tmp_path, capabilities=CAPS)
    engine = SimpleNamespace(task=SimpleNamespace(qualified_name='fixture', task_hash='task'),
        llm=instance, cfg={'llm': {'models': instance.models, 'model_capabilities': copy.deepcopy(CAPS)}})
    saved = {'identity': checkpoint._identity(engine)}
    assert checkpoint.check_identity(engine, saved) == []
    engine.cfg['llm']['model_capabilities'][MODEL]['temperature'] = True
    with pytest.raises(checkpoint.IdentityMismatch, match='capabilities'):
        checkpoint.check_identity(engine, saved)
    instance.model_capabilities[MODEL]['temperature'] = True
    assert any('llm_transport' in p for p in checkpoint.check_identity(engine, saved))
    instance.model_capabilities = {}; engine.cfg['llm'].pop('model_capabilities')
    assert any('llm_transport' in p for p in checkpoint.check_identity(engine, saved))
