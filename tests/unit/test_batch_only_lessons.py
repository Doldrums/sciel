"""Actual Engine/Research flow for batch-only creation, using inert I/O only.

The evaluator is never imported; source text supplies fixed numbers. No helper,
solver, candidate execution, network service or real monetary ledger is touched.
"""
import copy
import json
import re
import sys

import pytest

from sciel.campaign.budget import CampaignStopped
from sciel.config import apply_defaults
from sciel.engine import checkpoint
from sciel.engine.llm.client import LLMResponse
from sciel.engine.loop import Engine
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.hypothesis.conclusion_feedback import validate_conclusions
from sciel.hypothesis.rendering import decode_evidence
from _mechanism_fixtures import unknown_mechanism
from test_research_injection import fake_task, events


ABSENT = object()
POLICY_SCHEMA = 'sciel-conclusion-lesson-admission/1'


def decode(prompt, marker):
    raw, _ = json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1].lstrip())
    return decode_evidence(raw)


def proposal(record, phase):
    return dict(rule=f'{phase} fixture observation {record["id"]}.',
        polarity='positive' if record['actual_delta'] > 0 else 'negative',
        scope='branch', evidence_ids=[record['id']])


class Model:
    models = ['fake']
    weights = [1.]
    max_retries = 1

    def __init__(self, start=0, failure=None):
        self.step = start
        self.failure = failure
        self.calls = []

    def choose_model(self, rng):
        return 'fake'

    def generate(self, prompt, model, temperature=None):
        if 'Review this one experiment.' in prompt:
            kind = 'conclusion'
            record = decode(prompt, 'EVIDENCE_JSON:')
            value = dict(verdict=record['forecast_verdict'],
                explanation='Retained review of '+record['id'],
                lessons=[proposal(record, 'Individual')],
                mechanism_review=unknown_mechanism(record))
            if self.failure == 'mechanism':
                value['mechanism_review'][0]['claim_quote'] = 'NOT_THE_RECORDED_HYPOTHESIS'
            text = 'CONCLUSION_JSON: '+json.dumps(value)
        elif 'Review existing experimental evidence and lessons.' in prompt:
            kind = 'reflection'
            records = decode(prompt, 'EVIDENCE_JSON:')['observation_catalog']
            value = dict(lessons=[proposal(r, 'Reflection') for r in records.values()
                                 if r.get('status') == 'evaluated'], retire_ids=[])
            if self.failure == 'reflection_evidence':
                value['lessons'][0]['evidence_ids'] = ['missing-observation']
            text = 'REFLECTION_JSON: '+json.dumps(value)
        else:
            kind = 'generation'
            self.step += 1
            score = [.2, .05, .3][self.step-1]
            low, high = (-.5, -.001) if self.step == 2 else (.3, .4)
            value = dict(hypothesis=f'Inert fixed numeric change {self.step}.',
                predicted_low=low, predicted_high=high,
                reasoning='Only a fixture forecast; source is not executed.', confidence=.5)
            text = ('HYPOTHESIS_JSON: '+json.dumps(value)
                    +f'\n```python\ndef run_search():\n    return {score!r}\n```')
        self.calls.append(dict(kind=kind, prompt=prompt, text=text))
        if kind == 'reflection' and self.failure == 'provider_stop':
            raise CampaignStopped('inert unknown-cost reflection, no network')
        return LLMResponse(text, model, .01, 20, 20)


def build(task, directory, *, batch=ABSENT, start=0, interval=2, aux=8, failure=None):
    research = dict(enabled=True, reflection_interval=interval, max_aux_calls=aux)
    if batch is not ABSENT:
        research['batch_only_lessons'] = batch
    cfg = dict(task='test/injection', budget=3, max_iterations=3, dup_cap=1,
        random_seed=21, num_islands=1, pop_cap=1, grid_bins=1,
        schedule_policy='round_robin', milestone_fracs=[], op_fixed='full_rewrite',
        policy=dict(enabled=False, plateau_shift=False), llm={'mock': True},
        portfolio_seed=False, idea_island=False, dynamic_islands=False, crossover=False,
        embed_grid=False, embed_niche=False, three_head=False, confirm_records=False,
        research=research)
    model = Model(start, failure)
    engine = Engine(cfg, task, str(directory), model, python=sys.executable)

    def evaluate(code):
        score = float(re.search(r'return\s+([0-9.]+)', code).group(1))
        return CascadeOutcome(EvalResult(True, score, {'value': score}, timing=.01),
                              score, 'full', False)

    engine._evaluate = evaluate
    return engine, model


def admission(count=1):
    return dict(schema=POLICY_SCHEMA, policy='batch_only_lessons',
                status='suppressed', proposed_count=count)


@pytest.mark.parametrize('bad', [0, 1, None, 'false', [], {}])
def test_flag_is_strict_boolean(bad):
    with pytest.raises(ValueError, match='batch_only_lessons'):
        apply_defaults({'research': {'batch_only_lessons': bad}})


def test_absent_default_and_explicit_false_preserve_actual_behavior(fake_task, tmp_path):
    assert 'batch_only_lessons' not in apply_defaults({'research': {'enabled': True}})['research']
    absent, a = build(fake_task, tmp_path/'absent')
    disabled, b = build(fake_task, tmp_path/'false', batch=False)
    absent.run(); disabled.run()
    assert a.calls == b.calls
    assert absent.research.knowledge.state() == disabled.research.knowledge.state()
    assert absent.research.conclusions == disabled.research.conclusions
    assert all('lesson_admission' not in c for c in absent.research.conclusions.values())
    assert absent.research.summary()['presentation'] == disabled.research.summary()['presentation']


def test_suppressed_conclusion_keeps_review_facts_cost_and_next_feedback(fake_task, tmp_path):
    engine, model = build(fake_task, tmp_path/'suppressed', batch=True)
    engine._stop_at = 1
    engine.run()
    research = engine.research
    original = copy.deepcopy(research.knowledge.experiments[0])
    review = research.conclusions[original['id']]
    answer = json.loads(model.calls[-1]['text'].split('CONCLUSION_JSON:')[1])
    assert review['llm'] == answer
    assert review['lesson_admission'] == admission()
    assert review['accepted_lesson_ids'] == []
    assert review['lesson_error'] is review['lesson_error_detail'] is None
    assert research.knowledge.lessons == []
    assert research.knowledge.state()['next_lesson_id'] == 1
    assert original['actual_delta'] > 0 and original['forecast_verdict'] == 'REFUTED'
    assert research.aux_calls == 1 and engine.cost_total == pytest.approx(.02)
    assert research.knowledge.fragments  # Facts and successful code remain available.
    resumed, next_model = build(fake_task, tmp_path/'suppressed', batch=True, start=1)
    checkpoint.load(resumed, str(tmp_path/'suppressed/checkpoint.json'))
    resumed.run()
    research = resumed.research
    second = [c for c in next_model.calls if c['kind'] == 'generation'][0]
    wire = decode(second['prompt'], 'RESEARCH_CONTEXT_JSON:')
    card, = wire['conclusion_feedback']['cards']
    assert card['unverified_interpretation']['explanation'] == answer['explanation']
    assert card['lesson_admission']['policy_suppression'] == admission()
    assert card['lesson_admission']['accepted_lesson_ids'] == []
    assert 'RESEARCH_LESSON_JSON:' not in second['prompt']
    assert research.knowledge.experiments[0] == original
    assert research.knowledge.experiments[1]['injected_ids'] == []


def test_scheduled_reflection_creates_signed_lessons_used_only_afterward(fake_task, tmp_path):
    engine, model = build(fake_task, tmp_path/'cycle', batch=True)
    result = engine.run()
    records = engine.research.knowledge.experiments
    assert records[0]['actual_delta'] > 0 and records[0]['forecast_verdict'] == 'REFUTED'
    assert records[1]['actual_delta'] < 0 and records[1]['forecast_verdict'] == 'CONFIRMED'
    lessons = engine.research.knowledge.lessons
    assert len(lessons) == 2
    assert {(l['polarity'], tuple(l['source_support_ids'])) for l in lessons} == {
        ('positive', ('r000001a1',)), ('negative', ('r000002a1',))}
    assert all(l['exposure_ids'] == ['r000003a1'] for l in lessons)
    assert [c['kind'] for c in model.calls] == [
        'generation', 'conclusion', 'generation', 'conclusion', 'reflection', 'generation', 'conclusion']
    prompt = next(c['prompt'] for c in model.calls if c['kind'] == 'reflection')
    block = decode(prompt, 'REFLECTION_REVIEW_CONTEXT_JSON:')
    assert len(block['cards']) == 2
    assert all(c['lesson_admission']['policy_suppression'] == admission() for c in block['cards'])
    assert all(len(c['unverified_interpretation']['lessons']) == 1 for c in block['cards'])
    last = [c['prompt'] for c in model.calls if c['kind'] == 'generation'][-1]
    assert all(l['rule'] in last for l in lessons)
    assert all('Individual' not in l['rule'] for l in lessons)
    assert result['cost_total'] == pytest.approx(.07)
    assert engine.research.aux_calls == 4 and engine.research.reflections == 1
    assert engine.research.reflected_through == 2
    assert all(l['authoring']['catalog'][ref]['phase'] == 'reflection'
               for l in lessons for ref in l['authoring']['writer_refs'])


@pytest.mark.parametrize('stop', [1, 2])
def test_split_resume_preserves_full_prompts_writers_and_costs(fake_task, tmp_path, stop):
    whole, full_model = build(fake_task, tmp_path/'whole', batch=True)
    expected = whole.run()
    split, first_model = build(fake_task, tmp_path/'split', batch=True)
    split._stop_at = stop
    split.run()
    resumed, rest_model = build(fake_task, tmp_path/'split', batch=True, start=stop)
    checkpoint.load(resumed, str(tmp_path/'split/checkpoint.json'))
    assert resumed.run() == expected
    assert first_model.calls+rest_model.calls == full_model.calls
    assert resumed.research.state() == whole.research.state()


@pytest.mark.parametrize('interval,aux,expected_aux', [(0, 8, 3), (4, 8, 3), (2, 1, 1), (2, 0, 0)])
def test_no_extra_reflection_compensates_for_missing_lessons(fake_task, tmp_path, interval, aux, expected_aux):
    engine, model = build(fake_task, tmp_path/'no-extra', batch=True, interval=interval, aux=aux)
    engine.run()
    assert engine.research.knowledge.lessons == []
    assert engine.research.aux_calls == expected_aux
    assert len(model.calls) == 3+expected_aux
    assert not any(c['kind'] == 'reflection' for c in model.calls)
    assert engine.cost_total == pytest.approx(.01*(3+expected_aux))


@pytest.mark.parametrize('damage', ['count', 'bool_count', 'status', 'policy', 'schema', 'extra', 'accepted', 'error', 'unparsed', 'numeric_config'])
def test_forged_suppression_is_rejected_atomically(fake_task, tmp_path, damage):
    original, _ = build(fake_task, tmp_path/'original', batch=True)
    original._stop_at = 1; original.run()
    saved = original.research.state()
    review = saved['conclusions']['r000001a1']
    marker = review['lesson_admission']
    if damage == 'count': marker['proposed_count'] = 0
    elif damage == 'bool_count': marker['proposed_count'] = True
    elif damage == 'status': marker['status'] = 'accepted'
    elif damage == 'policy': marker['policy'] = 'unknown'
    elif damage == 'schema': marker['schema'] = 'unknown'
    elif damage == 'extra': marker['verified'] = True
    elif damage == 'accepted': review['accepted_lesson_ids'] = ['l000001']
    elif damage == 'error': review.update(lesson_error='ValueError', lesson_error_detail='forged')
    elif damage == 'unparsed': review.update(status='invalid', llm=None, verdict_matches_observation=None)
    else: saved['config']['batch_only_lessons'] = 1
    target, model = build(fake_task, tmp_path/'target', batch=True)
    before = target.research.state()
    with pytest.raises(ValueError):
        target.research.load_state(saved)
    assert target.research.state() == before and model.calls == []
    target.log.close()


@pytest.mark.parametrize('mode', [ABSENT, False])
def test_suppression_requires_explicit_mode_and_mode_drift_cannot_resume(fake_task, tmp_path, mode):
    original, _ = build(fake_task, tmp_path/'original', batch=True)
    original._stop_at = 1; original.run()
    state = original.research.state()
    ordinary = {r['id']: r for r in original.research.knowledge.experiments}
    with pytest.raises(ValueError):
        validate_conclusions(state['conclusions'], ordinary)
    target, model = build(fake_task, tmp_path/'target', batch=mode)
    before = target.research.state()
    with pytest.raises(checkpoint.IdentityMismatch):
        checkpoint.load(target, str(tmp_path/'original/checkpoint.json'))
    assert target.research.state() == before and model.calls == []
    target.log.close()


@pytest.mark.parametrize('failure', ['mechanism', 'reflection_evidence'])
def test_invalid_reviews_and_reflection_evidence_keep_distinct_failure_paths(fake_task, tmp_path, failure):
    engine, model = build(fake_task, tmp_path/'invalid', batch=True, failure=failure)
    engine.run()
    if failure == 'mechanism':
        assert all(c['status'] == 'invalid' and 'lesson_admission' not in c
                   for c in engine.research.conclusions.values())
        # Invalid prose does not invalidate the measured evidence used by reflection.
        assert len(engine.research.knowledge.lessons) == 2
    else:
        assert all(c['lesson_admission'] == admission() for c in engine.research.conclusions.values())
        assert engine.research.knowledge.lessons == []
        event = next(e for e in events(tmp_path/'invalid') if e.get('phase') == 'reflection')
        assert event['error'] == 'ValueError' and event['applied'] is None
    assert engine.research.aux_calls == 4 and len(model.calls) == 7


def test_stopped_reflection_does_not_retry_or_manufacture_lessons(fake_task, tmp_path):
    engine, model = build(fake_task, tmp_path/'stopped', batch=True, failure='provider_stop')
    with pytest.raises(CampaignStopped):
        engine.run()
    assert engine.research.knowledge.lessons == []
    assert engine.research.aux_calls == 3
    restored, later = build(fake_task, tmp_path/'stopped', batch=True, start=2)
    checkpoint.load(restored, str(tmp_path/'stopped/checkpoint.json'))
    with pytest.raises(CampaignStopped):
        restored.run()
    assert later.calls == []
    restored.log.close()


def test_preexisting_advice_survives_and_receives_real_exposure(fake_task, tmp_path):
    first, _ = build(fake_task, tmp_path/'prior', batch=True)
    first._stop_at = 1; first.run()
    engine, model = build(fake_task, tmp_path/'prior', batch=True, start=1)
    checkpoint.load(engine, str(tmp_path/'prior/checkpoint.json'))
    record = engine.research.knowledge.experiments[0]
    ident, = engine.research.knowledge.add_lessons([proposal(record, 'Previously admitted')],
                                               allowed_evidence_ids=[record['id']])
    engine.run()
    lesson = next(l for l in engine.research.knowledge.lessons if l['id'] == ident)
    assert 'r000002a1' in lesson['exposure_ids']
    assert lesson['rule'] in [c['prompt'] for c in model.calls if c['kind'] == 'generation'][0]
    assert all(c['accepted_lesson_ids'] == [] for c in engine.research.conclusions.values())


def test_empty_proposal_is_explicit_without_creating_any_credit(fake_task, tmp_path, monkeypatch):
    engine, model = build(fake_task, tmp_path/'empty', batch=True, interval=0)
    generate = model.generate

    def empty(prompt, *args, **kwargs):
        response = generate(prompt, *args, **kwargs)
        if 'Review this one experiment.' in prompt:
            parsed = json.loads(response.text.split('CONCLUSION_JSON:')[1])
            parsed['lessons'] = []
            response.text = 'CONCLUSION_JSON: '+json.dumps(parsed)
        return response

    monkeypatch.setattr(model, 'generate', empty)
    engine.run()
    assert all(c['lesson_admission'] == admission(0) for c in engine.research.conclusions.values())
    assert engine.research.knowledge.lessons == []
    assert engine.research.knowledge.state()['next_lesson_id'] == 1
    target, _ = build(fake_task, tmp_path/'restored', batch=True, interval=0)
    target.research.load_state(engine.research.state())
    assert target.research.state() == engine.research.state()
    target.log.close()


def test_legacy_unmarked_review_reads_without_fabricated_policy_history(fake_task, tmp_path):
    engine, _ = build(fake_task, tmp_path/'legacy')
    engine._stop_at = 1; engine.run()
    reviews = copy.deepcopy(engine.research.conclusions)
    reviews['r000001a1']['llm'].pop('mechanism_review')
    records = {r['id']: r for r in engine.research.knowledge.experiments}
    assert validate_conclusions(reviews, records, batch_only_lessons=True) == reviews
    assert validate_conclusions(reviews, records) == reviews
    forged = copy.deepcopy(reviews)
    forged['r000001a1']['accepted_lesson_ids'] = []
    forged['r000001a1']['lesson_admission'] = admission()
    with pytest.raises(ValueError, match='suppressed lesson admission'):
        validate_conclusions(forged, records, batch_only_lessons=True)
