"""Actual Engine/C/WAL source delivery with inert providers and measurements.

The large source fixture is parsed and saved, never executed. The late keyword
is ordinary fixture data, not a selection hint passed to the source projector.
"""
from copy import deepcopy
import hashlib
import json
import threading

import pytest

from sciel.engine import checkpoint
from sciel.engine.factory import build_engine
from sciel.engine.llm.client import LLMResponse
from sciel.engine.llm.mock import MockLLM
from sciel.hypothesis import conclusion_refs
from sciel.hypothesis.observed_facts import observed_comparison
from sciel.hypothesis.rendering import decode_evidence
from test_parallel_engine import config, evaluator, events, task_fixture
from test_parallel_provider_recovery import physical_files


LATE_SOURCE = "    return select_strategy(VALUE, method='SLSQP')\n"
MODEL_PROSE = 'UNVERIFIED fixture interpretation: this change guarantees future benefit.'


def long_source(value):
    return (f'VALUE = {value}\ndef run_search():\n'
            + ''.join(f'    padding_{i:03d} = {i}\n' for i in range(360))
            + LATE_SOURCE)


def evidence(prompt):
    value, _ = json.JSONDecoder().raw_decode(prompt.split('EVIDENCE_JSON:\n', 1)[1])
    return decode_evidence(value)


class SourceReviewProvider(MockLLM):
    def __init__(self, directory, workers):
        super().__init__()
        self.directory = directory
        self.workers = workers
        self.lock = threading.Lock()
        self.calls = []
        self.generated = []
        self.review_records = []
        self.stop_barrier = threading.Barrier(workers)
        self.first_wave = True

    def generate(self, prompt, model, temperature=None, **options):
        is_conclusion = 'Review this one experiment.' in prompt[:160]
        with self.lock:
            self.calls.append((prompt, deepcopy(options)))
        if not is_conclusion:
            with self.lock:
                source = long_source(len(self.generated) + 1)
                self.generated.append(source)
            hypothesis = dict(hypothesis='The final statement changes a strategy argument.',
                predicted_low=0., predicted_high=1.,
                reasoning='The effect of this fixture change is uncertain.', confidence=.5)
            text = ('HYPOTHESIS_JSON: ' + json.dumps(hypothesis)
                    + '\n```python\n' + source + '```')
            return LLMResponse(text, model, .03, 20, 20, finish_reason='stop')

        record = evidence(prompt)
        # This is the missing V39 evidence: a decisive late line, not merely the
        # hypothesis repeating its name or a saved-but-undelivered artifact.
        assert LATE_SOURCE.strip() in prompt
        assert 'conclusion_source' in record
        assert LATE_SOURCE.strip() not in record['change_excerpt']
        catalog = conclusion_refs.catalog(record)
        late = next(item for item in catalog['sources'] if LATE_SOURCE.strip() in item['text'])
        assert late['field'] != 'change_excerpt'
        with self.lock:
            self.review_records.append(deepcopy(record))
        text = json.dumps(dict(verdict='REFUTED', explanation=MODEL_PROSE, lessons=[],
            mechanism_review=[dict(claim_ref=catalog['claims'][0]['id'],
                source_refs=[late['id']], evidence_kind='source_effect', assessment='unknown',
                direct_effect='The late statement is visible; its benefit is unverified.',
                untested_downstream='No mechanism or future benefit is established.'),
                dict(claim_ref=None, source_refs=[], evidence_kind='forecast',
                    assessment='directly_supported', direct_effect='The delta is inside the interval.',
                    untested_downstream='Interval agreement does not prove a mechanism.')]))
        if self.first_wave:
            # All first-wave C transports have started before STOP. Their paid
            # responses must be receipted and later consumed without redispatch.
            self.stop_barrier.wait(timeout=5)
            (self.directory/'STOP').touch()
        return LLMResponse(text, model, .01, 20, 20, finish_reason='stop')


@pytest.mark.parametrize('workers', [1, 2])
def test_late_conclusion_source_survives_parallel_pause_without_paid_replay(tmp_path, monkeypatch, workers):
    tasks = tmp_path/'tasks'
    task = task_fixture(tasks)
    directory = tmp_path/'run'
    cfg = config(task, tmp_path/'capacity', workers=workers, num_islands=workers,
        max_iterations=workers, research=dict(enabled=True, reflection_interval=0,
            max_aux_calls=None, structured_auxiliary=True, conclusion_source_refs=True,
            conclusion_repair_attempts=1))
    provider = SourceReviewProvider(directory, workers)
    evaluation_calls, _ = evaluator(monkeypatch)

    def build(resume=False):
        engine = build_engine(cfg, str(tasks), str(directory), llm=provider)
        if resume:
            checkpoint.load(engine, str(directory/'checkpoint.json'))
        return engine

    first = build()
    assert first.run()['reason'] == 'paused'
    paused = json.loads((directory/'checkpoint.json').read_text())
    assert paused['iteration_no'] == paused['B_evals'] == paused['cost_total'] == 0
    assert len(provider.generated) == len(provider.review_records) == workers
    assert len(evaluation_calls) == workers + 1
    saved = physical_files(directory)
    calls = deepcopy(provider.calls)
    returned_records = deepcopy(provider.review_records)
    provider.first_wave = False

    resumed = build(resume=True)
    final = resumed.run()
    assert final['iteration_no'] == final['B_evals'] == workers
    assert final['cost_total'] == pytest.approx(workers*.04)
    assert provider.calls == calls and provider.review_records == returned_records
    assert len(evaluation_calls) == workers+1
    assert all(physical_files(directory)[name] == raw for name, raw in saved.items())
    assert not list((directory/'.parallel').rglob('abandonment.json'))

    # Large source context belongs to C, not to an ordinary growing lesson or
    # experiment record. Numerical facts do not inherit the fake model's prose.
    records = resumed.research.knowledge.experiments
    assert len(records) == workers and resumed.research.knowledge.lessons == []
    for record in records:
        assert 'conclusion_source' not in record
        assert len(record['change_excerpt']) <= 6000
        sent = next(item for item in returned_records if item['id'] == record['id'])
        assert sent['source_sha256'] == record['source_sha256']
        assert record['source_sha256'] in {
            hashlib.sha256(source.encode()).hexdigest() for source in provider.generated}
        result = resumed.research.conclusions[record['id']]
        assert result['status'] == 'parsed' and result['llm']['explanation'] == MODEL_PROSE
        assert result['llm']['verdict'] == 'REFUTED'
        assert result['observed_forecast_verdict'] == 'CONFIRMED'
        assert result['verdict_matches_observation'] is False
        assert result['observed_comparison'] == observed_comparison(record)
        assert result['observed_comparison']['forecast']['position'] == 'inside'
        child = next(p for p in resumed.db._all.values()
            if hashlib.sha256(p.code.encode()).hexdigest() == record['source_sha256'])
        context = resumed.research.context(resumed.db, child, child.island, workers)
        feedback = context['conclusion_feedback']
        card = next(c for c in feedback['cards'] if c['experiment_id'] == record['id'])
        assert card['unverified_interpretation']['explanation'] == MODEL_PROSE
        assert card['observed_comparison'] == observed_comparison(record)
        assert 'unverified' in feedback['interpretation_limits']
    assert not any(row.get('phase') == 'conclusion_rejected' for row in events(directory))
    before = deepcopy(resumed.research.state())
    cold = build(resume=True)
    assert cold.research.state() == before
    assert provider.calls == calls and len(evaluation_calls) == workers+1
    cold.log.close()
