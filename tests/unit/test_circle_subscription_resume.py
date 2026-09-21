"""Opt-in generation replays the same pricing contract it uses at admission."""
import copy
import hashlib
import json

import pytest

from sciel.engine import checkpoint
from sciel.engine.llm.client import (LLMResponse, COST_PROVENANCE_SCHEMA,
                                    SUBSCRIPTION_ENDPOINT_SHA256)
from test_circle_generation_engine import build, response, rows


MARKER = dict(schema=COST_PROVENANCE_SCHEMA, source='subscription_unpriced',
              provider='codex_responses', response_status='completed',
              endpoint_sha256=SUBSCRIPTION_ENDPOINT_SHA256)


@pytest.mark.parametrize('unpriced', [False, True])
def test_completed_generation_cold_resume_preserves_accounting_without_calls(tmp_path, unpriced):
    reply = LLMResponse(response(), 'fixture', None if unpriced else .003, 17, 11,
                        finish_reason='stop', cost_provenance=MARKER if unpriced else None)
    engine = build(tmp_path / 'run', [reply])
    try:
        result = engine.run()
        assert result['B_evals'] == 1 and len(engine.llm.calls) == 1
        assert result['llm_unpriced_calls'] == int(unpriced)
        saved_state = engine.circle_generation.state()
        task = engine.task
    finally:
        engine.close()
    restored = build(tmp_path / 'run', [], task=task)
    try:
        checkpoint.load(restored, str(tmp_path / 'run/checkpoint.json'))
        assert restored.circle_generation.state() == saved_state
        assert restored.cost_total == (0.0 if unpriced else .003)
        assert restored.llm_unpriced_calls == int(unpriced)
        assert not restored.llm.calls and not restored.fixture_evaluated
    finally:
        restored.close()


@pytest.mark.parametrize('damage', ['missing_accounting', 'missing_provenance', 'unknown_endpoint',
                                   'known_cost', 'unpriced_count', 'boolean_count', 'unfinished'])
def test_unpriced_replay_rejects_invalid_or_missing_price_evidence(tmp_path, damage):
    reply = LLMResponse(response(), 'fixture', None, 17, 11,
                        finish_reason='stop', cost_provenance=MARKER)
    engine = build(tmp_path / 'run', [reply])
    try:
        engine.run()
        events = copy.deepcopy(rows(engine))
        result = next(e for e in events if e.get('phase') == 'circle_generation'
                      and e.get('action') == 'request_result')
        if damage == 'missing_accounting':
            result.pop('cost_accounting')
        elif damage == 'missing_provenance':
            result.pop('cost_provenance')
        elif damage == 'unknown_endpoint':
            result['cost_provenance']['endpoint_sha256'] = '0' * 64
        elif damage in ('unpriced_count', 'boolean_count'):
            result['cost_accounting']['unpriced_calls'] = 0 if damage == 'unpriced_count' else True
        elif damage == 'known_cost':
            result['cost_accounting']['known_usd'] = .01
        else:
            result['finish_reason'] = 'length'
        raw = ''.join(json.dumps(row) + '\n' for row in events).encode()
        damaged = tmp_path / 'damaged.jsonl'
        damaged.write_bytes(raw)
        with pytest.raises(ValueError):
            engine.circle_generation.assert_journal(damaged, dict(byte_offset=len(raw),
                prefix_sha256=hashlib.sha256(raw).hexdigest()))
    finally:
        engine.close()
