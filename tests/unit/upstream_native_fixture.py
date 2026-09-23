"""Inert SDK leaf; real local inference, renderer, counter and accounting."""
from types import SimpleNamespace
from unittest.mock import Mock
import pytest
from sciel.engine.factory import build_llm

@pytest.fixture
def provider(monkeypatch):
    monkeypatch.setattr('sciel.engine.factory.load_env', Mock(side_effect=AssertionError('dotenv')))
    sdk = Mock()
    constructor = Mock(return_value=sdk)
    monkeypatch.setattr('openai.OpenAI', constructor)
    config = {'llm': {'backend': 'local_openai', 'models': ['base-pinned'],
        'api_base': 'http://127.0.0.1:8000/v1', 'max_tokens': 512,
        'local_inference': {'base_model': 'Qwen/Qwen3-1.7B', 'base_revision': 'a'*40,
            'tokenizer_revision': 'b'*40, 'chat_template_sha256': 'c'*64,
            'base_alias': 'base-pinned', 'learner_by_island': {'0':'learner-0','1':'learner-1'},
            'adapters': {}, 'node_ledger_id': 'inert-node'}}}
    return build_llm(config), sdk, constructor
