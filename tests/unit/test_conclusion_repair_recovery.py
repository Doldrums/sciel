"""Actual Engine recovery after a second C request; mock provider/evaluator only."""
import json

import pytest

from sciel.campaign.budget import CampaignStopped
from sciel.engine import checkpoint
from sciel.engine.llm.client import LLMError, LLMResponse
from sciel.engine.llm.mock import MockLLM
from sciel.engine.recovery import inspect_provider_stop, prepare_provider_recovery
from test_engine_recovery_review_fixes import construct, settings, outcome, rows


def generation(value, model):
    hypothesis = dict(hypothesis='Improve the returned fixture value.', predicted_low=0.,
                      predicted_high=1., reasoning='A bounded offline change.', confidence=.5)
    raw = 'HYPOTHESIS_JSON: ' + json.dumps(hypothesis) + (
        '\n```python\ndef run_search():\n    return ' + str(value) + '\n```')
    return LLMResponse(raw, model, .03, 10, 10)


@pytest.mark.parametrize('conclusion_tokens', [None, 4096])
def test_unknown_second_conclusion_recovers_to_new_generation_without_replay_or_lost_cost(tmp_path, conclusion_tokens):
    calls = []
    class InitialProvider(MockLLM):
        def generate(self, prompt, model, temperature=None, **options):
            if not options.get('response_format'):
                assert 'max_tokens' not in options
                calls.append(('generation', None))
                return generation(3, model)
            event = rows(tmp_path)[-1]
            assert options.get('max_tokens') == conclusion_tokens
            assert event['phase'] == 'aux_start' and event['kind'] == 'conclusion'
            calls.append(('conclusion', event['reference']))
            if event['call_index'] == 1:
                return LLMResponse('{invalid json', model, .07, 10, 10, finish_reason='stop')
            assert event['call_index'] == 2 and 'CONCLUSION FORMAT CORRECTION' in prompt
            raise LLMError(cause=ConnectionError())

    cfg = settings(research=dict(enabled=True, max_aux_calls=None, reflection_interval=0,
        structured_auxiliary=True, conclusion_source_refs=True, conclusion_repair_attempts=1,
        conclusion_max_tokens=conclusion_tokens))
    engine = construct(tmp_path, cfg, InitialProvider())
    initial_scores = iter([.1, .7])
    engine._evaluate = lambda _: outcome(next(initial_scores))
    with pytest.raises(CampaignStopped, match='research auxiliary request stopped'):
        engine.run()
    path = tmp_path / 'checkpoint.json'
    saved = json.loads(path.read_bytes())
    assert saved['iteration_no'] == saved['B_evals'] == 1
    assert saved['cost_total'] == pytest.approx(.10)
    assert saved['research']['aux_calls'] == 2 and saved['research']['aux_cost'] == pytest.approx(.07)
    assert saved['research']['queue'] == [] and saved['research']['pending'] == {}
    assert saved['research']['conclusions'] == {}
    assert calls == [('generation', None), ('conclusion', 'r000001a1'), ('conclusion', 'r000001a1')]
    initial_journal = rows(tmp_path)
    assert len([row for row in initial_journal if row.get('phase') == 'aux_result']) == 1
    rejected, = [row for row in initial_journal if row.get('phase') == 'conclusion_rejected']
    assert rejected['experiment_id'] == 'r000001a1'
    receipt = inspect_provider_stop(path, allow_legacy=False)
    assert receipt['auto_recoverable'] is True and receipt['unknown_cost'] is True
    assert receipt['request'] == dict(kind='conclusion', reference='r000001a1', call_index=2)
    assert receipt['known_cost_total'] == pytest.approx(.10)

    class ContinuedProvider(MockLLM):
        def generate(self, prompt, model, temperature=None, **options):
            if not options.get('response_format'):
                assert 'max_tokens' not in options
                calls.append(('generation', None))
                engine.request_stop()
                return generation(4, model)
            event = rows(tmp_path)[-1]
            assert options.get('max_tokens') == conclusion_tokens
            assert event['phase'] == 'aux_start' and event['reference'] == 'r000002a1'
            assert event['call_index'] == 3
            assert 'CONCLUSION FORMAT CORRECTION' not in prompt
            calls.append(('conclusion', event['reference']))
            result = dict(verdict='CONFIRMED', explanation='Observed native delta lies in the forecast.',
                lessons=[], mechanism_review=[dict(claim_ref=None, source_refs=[],
                    evidence_kind='forecast', assessment='directly_supported',
                    direct_effect='Only interval agreement is checked.',
                    untested_downstream='Mechanism and future quality remain unverified.')])
            return LLMResponse(json.dumps(result), model, .02, 10, 10, finish_reason='stop')

    engine = construct(tmp_path, saved['config'], ContinuedProvider())
    engine._evaluate = lambda _: outcome(.8)
    checkpoint.load(engine, str(path))
    prepare_provider_recovery(engine, path, allow_legacy=False)
    assert engine.research.queue == []
    assert engine.run()['reason'] == 'paused'
    final = json.loads(path.read_bytes())
    assert calls[3:] == [('generation', None), ('conclusion', 'r000002a1')]
    assert final['iteration_no'] == final['B_evals'] == 2
    assert final['cost_total'] == pytest.approx(.15)
    assert final['research']['aux_cost'] == pytest.approx(.09)
    assert final['research']['aux_calls'] == 3
    assert 'r000001a1' not in final['research']['conclusions']
    assert final['research']['conclusions']['r000002a1']['status'] == 'parsed'
    assert final['provider_recoveries'] == [receipt]
    assert final['provider_recoveries'][0]['unknown_cost'] is True
    assert sum(row.get('phase') == 'aux_result' for row in rows(tmp_path)) == 2
