"""Regression for S05's adjacent JSON/Python blocks, without replaying a run."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from sciel.engine.prompt.parse import extract_code
from sciel.hypothesis import parse_hypothesis
from sciel.engine.checkpoint import check_identity


CODE = "def run_packing():\n    return 1\n"


@pytest.mark.parametrize('language', ['json', 'yaml', 'text'])
@pytest.mark.parametrize('separator', ['\n', '\n\n', '\nHere is the code:\n'])
def test_metadata_block_cannot_consume_following_python_opener(language, separator):
    response = f'```{language}\n' + ('metadata ' * 30) + '\n```' + separator + '```python\n' + CODE + '```'
    assert extract_code(response) == CODE


def test_s05_invalid_forecast_keeps_real_code_without_relaxing_forecast_contract():
    forecast = {'hypothesis': 'Use a new initialization.', 'predicted_low': .1,
                'predicted_high': .2, 'reasoning': 'Try a different layout.', 'confidence': .6}
    response = '```json\n' + json.dumps(forecast) + '\n```\n\n```python\n' + CODE + '```'
    assert parse_hypothesis(response) is None
    assert hashlib.sha256(extract_code(response).encode()).hexdigest() == hashlib.sha256(CODE.encode()).hexdigest()
    assert extract_code(response) != '\n'


def test_longest_eligible_block_preserved_while_other_languages_are_consumed():
    response = '```python\nx=1\n```\n```json\n' + ('a' * 1000) + '\n```\n```\n' + CODE + '```'
    assert extract_code(response) == CODE


@pytest.mark.parametrize('response', [CODE, '```json\n{"only_metadata":true}\n```'])
def test_unfenced_fallback_preserved_when_no_eligible_block_exists(response):
    assert extract_code(response) == response.strip() + '\n'


def test_python_block_can_follow_crlf_metadata_block():
    response = '```json\r\n{}\r\n```\r\n```python \r\n' + CODE + '```'
    assert extract_code(response) == CODE


def test_pre_fix_engine_checkpoint_cannot_silently_resume_with_new_extraction():
    engine = SimpleNamespace(task=SimpleNamespace(), cfg={}, descriptor_info={})
    mismatch = check_identity(engine, {'identity': {'engine_contract': 'espalier-3.1'}})
    assert any(item.startswith('engine_contract:') for item in mismatch)
