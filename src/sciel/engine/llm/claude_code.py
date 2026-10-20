"""Claude through the Claude Code CLI on a Claude subscription, behind the same generate() contract.

Each call is one `claude -p` with no tools, no settings, no session and a fixed system prompt, run in an empty
directory so no project memory enters the prompt. The subscription is used through the CLI login or
CLAUDE_CODE_OAUTH_TOKEN (from `claude setup-token`); API keys are removed from its environment so a call never
bills an API account. A usage limit comes back as HTTP 429 with a retry-after, which pauses the provider.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from types import SimpleNamespace

from .client import (COST_PROVENANCE_SCHEMA, CLAUDE_CODE_ENDPOINT_SHA256, LLMError, LLMResponse, _tokens,
                     validate_response_format)

SYSTEM = 'Follow the user task and requested output format exactly.'
EFFORTS = (None, 'low', 'medium', 'high', 'xhigh', 'max')
LIMIT_WAIT = 900.0
API_ENV = ('ANTHROPIC_API_KEY', 'ANTHROPIC_AUTH_TOKEN', 'ANTHROPIC_BASE_URL')
FINISH = {'end_turn': 'stop', 'stop_sequence': 'stop', 'max_tokens': 'length', 'refusal': 'stop'}


def validate_config(llm: dict) -> dict:
    models = llm.get('models') or []
    if not models or any(type(m) is not str or not m.startswith('claude-') for m in models):
        raise ValueError('llm.models must name Claude models such as claude-opus-5-5')
    if llm.get('reasoning_effort') not in EFFORTS:
        raise ValueError(f'llm.reasoning_effort must be one of {EFFORTS[1:]} or null')
    if 'model_capabilities' in llm:
        raise ValueError('llm.model_capabilities does not apply: the Claude Code CLI sends no sampling parameters')
    return llm


class _Status(Exception):
    """A CLI error carrying the API status it reported, read by the shared error details."""

    def __init__(self, status, retry_after=None):
        super().__init__(f'claude CLI reported status {status}')
        self.status_code = status
        self.response = SimpleNamespace(headers={} if retry_after is None else {'retry-after': str(int(retry_after))})


RateLimitError = type('RateLimitError', (_Status,), {})
APIStatusError = type('APIStatusError', (_Status,), {})


def _reset_seconds(text: str) -> float:
    """Seconds until the usage window resets, from the CLI's 'limit reached|<epoch>' form, else a fixed wait."""
    match = re.search(r'\|(\d{9,11})\b', text or '')
    if match:
        return max(60.0, float(match.group(1)) - time.time())
    return LIMIT_WAIT


class ClaudeCodeLLM:
    def __init__(self, cfg: dict):
        cfg = validate_config(cfg)
        self.models = list(cfg['models'])
        self.weights = cfg.get('weights') or [1.0] * len(self.models)
        self.max_tokens = cfg.get('max_tokens', 32000)
        self.temperature = None
        self.effort = cfg.get('reasoning_effort')
        self.timeout = float(cfg.get('timeout', 3600.0))
        self.binary = cfg.get('claude_bin') or shutil.which('claude') or 'claude'
        self.model_capabilities = {}
        self._env = {k: v for k, v in os.environ.items() if k not in API_ENV}

    def choose_model(self, rng) -> str:
        return rng.choices(self.models, weights=self.weights, k=1)[0]

    def generate(self, prompt, model, temperature=None, *, response_format=None, max_tokens=None) -> LLMResponse:
        output_tokens = self.max_tokens if max_tokens is None else max_tokens
        if type(output_tokens) is not int or output_tokens <= 0:
            raise ValueError('max_tokens must be a positive integer or null')
        command = [self.binary, '-p', '--model', model, '--output-format', 'json', '--tools', '',
                   '--system-prompt', SYSTEM, '--no-session-persistence', '--setting-sources', '',
                   '--strict-mcp-config']
        if self.effort is not None:
            command += ['--effort', self.effort]
        schema = validate_response_format(response_format)
        if schema is not None:
            command += ['--json-schema', json.dumps(schema['json_schema']['schema'])]
        env = {**self._env, 'CLAUDE_CODE_MAX_OUTPUT_TOKENS': str(output_tokens)}
        try:
            with tempfile.TemporaryDirectory(prefix='sciel-claude-') as cwd:
                done = subprocess.run(command, input=prompt, capture_output=True, text=True, cwd=cwd, env=env,
                                      timeout=self.timeout)
        except subprocess.TimeoutExpired as e:
            raise LLMError(cause=TimeoutError(str(e))) from None
        except OSError as e:
            raise LLMError(cause=e) from e
        try:
            out = json.loads(done.stdout.strip().splitlines()[-1])
        except (ValueError, IndexError):
            raise LLMError(stage='response', reason='malformed_response',
                           cause=RuntimeError((done.stderr or done.stdout)[-500:])) from None
        text = out.get('result') if type(out.get('result')) is str else ''
        if out.get('is_error'):
            status = out.get('api_error_status')
            if status == 429 or 'limit' in text.lower():
                raise LLMError(cause=RateLimitError(429, _reset_seconds(text)))
            if type(status) is int:
                raise LLMError(cause=APIStatusError(status))
            raise LLMError(cause=RuntimeError(text[-500:]))
        if schema is not None and out.get('structured_output') is not None:
            text = json.dumps(out['structured_output'])
            finish = 'stop'
        else:
            finish = FINISH.get(out.get('stop_reason'), out.get('stop_reason'))
        usage = out.get('usage') if type(out.get('usage')) is dict else {}
        prompt_tokens = sum(_tokens(usage.get(k)) or 0 for k in
                            ('input_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens'))
        provenance = {'schema': COST_PROVENANCE_SCHEMA, 'source': 'unavailable'}
        if finish == 'stop':
            provenance = {'schema': COST_PROVENANCE_SCHEMA, 'source': 'subscription_unpriced',
                          'response_status': 'completed', 'provider': 'claude_code',
                          'endpoint_sha256': CLAUDE_CODE_ENDPOINT_SHA256}
        refusal = text or 'refusal' if out.get('stop_reason') == 'refusal' else None
        return LLMResponse(text=text, model=model, cost=None, prompt_tokens=prompt_tokens,
                           completion_tokens=_tokens(usage.get('output_tokens')), finish_reason=finish,
                           refusal=refusal, cost_provenance=provenance)
