"""Credential routing only: never construct a real provider or read auth files."""
import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def supervisor():
    path = Path(__file__).resolve().parents[2] / 'scripts/continue_live_run.py'
    spec = importlib.util.spec_from_file_location('codex_supervisor_fixture', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def test_explicit_auth_file_needs_neither_environment_nor_interactive_input(supervisor, monkeypatch):
    class NoEnvironment(dict):
        def get(self,key,*default):
            if key in ('CODEX_TOKEN','OPENROUTER_API_KEY'):
                pytest.fail('explicit file auth must not inspect credential environment')
            return super().get(key,*default)
    monkeypatch.setattr(supervisor.os,'environ',NoEnvironment())
    monkeypatch.setattr(supervisor.getpass,'getpass',lambda *a:pytest.fail('unexpected prompt'))
    supervisor.ensure_credentials({'llm':{'provider':'codex_responses','codex_auth_file':'/not-read/auth.json'}})


@pytest.mark.parametrize('llm,key,label', [
    ({'provider':'codex_responses'},'CODEX_TOKEN','Codex token'),
    ({},'OPENROUTER_API_KEY','OpenRouter API key'),
])
def test_environment_and_interactive_routes_name_the_correct_provider(supervisor,monkeypatch,llm,key,label):
    environment={};prompts=[]
    monkeypatch.setattr(supervisor.os,'environ',environment)
    monkeypatch.setattr(supervisor.getpass,'getpass',lambda prompt:prompts.append(prompt) or 'fixture')
    supervisor.ensure_credentials({'llm':llm})
    supervisor.ensure_credentials({'llm':llm})
    assert environment=={key:'fixture'} and prompts==[label+' (hidden): ']
