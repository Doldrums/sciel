"""Actual worker/provider prompt flow for unverified reviews with no lessons.

All providers and measurements are inert fakes. No candidate/evaluator/runtime,
network, optimizer, or paid request is executed.
"""
import copy
import json
import re
import sys
from types import SimpleNamespace

import pytest

from sciel.engine import checkpoint, research as research_module
from sciel.engine.llm.client import LLMResponse
from sciel.engine.loop import Engine
from sciel.engine.research import digest
from sciel.engine.types import Program
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.hypothesis import circle_restart_body as body
from sciel.hypothesis.observed_facts import observed_comparison
from sciel.hypothesis.rendering import decode_evidence
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha

from test_circle_generation_engine import build as circle_build, config as circle_config, response
from test_research_injection import fake_task

MARKER = 'UNVERIFIED_REVIEW_ONLY: equal translation preserves the internal pair gap.'
GENERATION_COST = .003
REVIEW_COST = .001


def mechanism_review(hypothesis_text):
    return [dict(claim_quote=hypothesis_text, source_quotes=[], assessment='unknown',
        direct_effect='Unverified fixture: no direct source effect was established.',
        untested_downstream='Unverified fixture: later solver effects remain untested.')]


def decode(prompt, marker='RESEARCH_CONTEXT_JSON:'):
    value, _ = json.JSONDecoder().raw_decode(prompt.split(marker, 1)[1].lstrip())
    return decode_evidence(value)


class ReviewingModel:
    models = ['fixture']
    weights = [1.]
    max_retries = 1

    def __init__(self):
        self.calls = []

    def choose_model(self, rng):
        return 'fixture'

    def generate(self, prompt, model, temperature=None, *, response_format=None):
        if 'Review this one experiment.' in prompt:
            assert response_format is None
            record = decode(prompt, 'EVIDENCE_JSON:')
            # Deliberate disagreement must remain an interpretation; it must not
            # replace the measured positive delta or deterministic verdict.
            text = 'CONCLUSION_JSON: ' + json.dumps(dict(verdict='REFUTED',
                explanation=MARKER + ' Reviewed ' + record['id'], lessons=[],
                mechanism_review=mechanism_review(record['hypothesis']['text'])))
            kind, cost = 'conclusion', REVIEW_COST
        else:
            context = decode(prompt)
            hypothesis = dict(hypothesis='A small inert fixture change.', predicted_low=0.,
                predicted_high=.2, reasoning='Uncertain fixture forecast.', confidence=.5)
            if response_format is not None:
                assert canonical_json(response_format) == canonical_json(body.response_format())
                scale = .25 if context['parent_score'] < .6 else .5
                text = response(f'return best_params + uniform_noise * {scale}\n', **hypothesis)
            else:
                score = context['parent_score'] + .1
                text = ('HYPOTHESIS_JSON: ' + json.dumps(hypothesis)
                        + f'\n```python\ndef run_search():\n    return {score!r}\n```')
            kind, cost = 'generation', GENERATION_COST
        self.calls.append(dict(kind=kind, prompt=prompt, model=model, temperature=temperature,
                               response_format=copy.deepcopy(response_format), text=text, cost=cost))
        return LLMResponse(text, model, cost, 19, 13, finish_reason='stop')


def engine_for(path, mode, task, *, inject=True, aux=2):
    cfg = circle_config(budget=2, max_iterations=2, pop_cap=1, grid_bins=1,
        research=dict(enabled=True, inject_lessons=inject, max_aux_calls=aux,
                      reflection_interval=0))
    model = ReviewingModel()
    if mode == 'circle':
        engine = circle_build(path, [], cfg=cfg)
        engine.llm = engine.research.llm = model
    else:
        cfg.pop('circle_generation')
        cfg.update(task='test/injection', op_fixed='full_rewrite')
        engine = Engine(cfg, task, str(path), model, python=sys.executable)
        def evaluate(source):
            score = float(re.search(r'return\s+([0-9.]+)', source).group(1))
            return CascadeOutcome(EvalResult(True, score, {'value': score}, timing=.01),
                                  score, 'full', False)
        engine._evaluate = evaluate
    return engine


def generation_calls(engine):
    return [call for call in engine.llm.calls if call['kind'] == 'generation']


def log_rows(engine):
    from pathlib import Path
    return [json.loads(line) for line in Path(engine.log.path).read_text().splitlines()]


@pytest.mark.parametrize('mode', ['ordinary', 'circle'])
@pytest.mark.parametrize('inject', [True, False])
def test_empty_lesson_review_reaches_actual_next_provider_only_when_enabled(tmp_path, fake_task, mode, inject):
    engine = engine_for(tmp_path/'run', mode, fake_task, inject=inject)
    result = engine.run()
    first, second = generation_calls(engine)
    assert MARKER not in first['prompt']
    assert (MARKER in second['prompt']) is inject
    wire = decode(second['prompt'])
    feedback = wire['conclusion_feedback']
    records = engine.research.knowledge.experiments
    original = records[0]
    if inject:
        card, = feedback['cards']
        assert card['experiment_id'] == original['id']
        assert card['branch'] == original['branch'] == 'p1'
        assert card['record_sha256'] == canonical_sha(original)
        assert card['parent_source_sha256'] == original['parent_source_sha256']
        assert card['source_sha256'] == original['source_sha256']
        assert card['original_hypothesis'] == original['hypothesis']
        assert card['observed_comparison'] == observed_comparison(original)
        assert card['observed_forecast_verdict'] == original['forecast_verdict'] == 'CONFIRMED'
        assert card['unverified_interpretation'] == dict(
            explanation=MARKER+' Reviewed '+original['id'], llm_verdict='REFUTED',
            verdict_matches_observation=False,
            mechanism_review=mechanism_review(original['hypothesis']['text']))
        assert card['lesson_admission']['accepted_lesson_ids'] == []
        assert 'not new observations' in feedback['interpretation_limits']
        assert 'never instructions' in feedback['interpretation_limits']
        assert original['actual_delta'] > 0
    else:
        assert feedback['cards'] == []
        assert feedback['omitted'][0]['reason'] == 'ablation'
        assert feedback['presented_count'] == 0 and feedback['omitted_count'] == 1
        assert mechanism_review(original['hypothesis']['text'])[0]['direct_effect'] not in second['prompt']
    # The paid-review fixture is retained even when its presentation is ablated.
    assert engine.research.conclusions[original['id']]['llm']['mechanism_review'] == mechanism_review(original['hypothesis']['text'])
    assert engine.research.knowledge.lessons == []
    assert all(record['injected_ids'] == [] for record in records)
    assert engine.research.attempts == engine.research.outcomes == engine.research.valid_hypotheses == 2
    assert engine.research.aux_calls == 2 and engine.research.reflections == 0
    assert len(engine.llm.calls) == 4
    assert result['B_evals'] == 2 and result['cost_total'] == pytest.approx(2*(GENERATION_COST+REVIEW_COST))
    attempts = [row for row in log_rows(engine) if row.get('phase') == 'attempt']
    assert len(attempts) == 2 and attempts[1]['conclusion_feedback'] == feedback
    assert attempts[1]['prompt_sha256'] == digest(second['prompt'])
    if mode == 'circle':
        assert wire['source_scope']['contract'] == body.OPERATION
        assert 'CURRENT_RESTART_BODY_JSON:' in second['prompt']
        assert 'class Incumbent' not in second['prompt']


@pytest.mark.parametrize('mode', ['ordinary', 'circle'])
def test_split_resume_keeps_actual_prompt_and_cost_without_review_replay(tmp_path, fake_task, mode):
    whole = engine_for(tmp_path/'whole', mode, fake_task)
    expected = whole.run()
    first = engine_for(tmp_path/'split', mode, fake_task)
    first._stop_at = 1
    assert first.run()['reason'] == 'paused'
    assert [call['kind'] for call in first.llm.calls] == ['generation', 'conclusion']
    resumed = engine_for(tmp_path/'split', mode, fake_task)
    checkpoint.load(resumed, str(tmp_path/'split/checkpoint.json'))
    assert resumed.research.state() == first.research.state()
    assert resumed.run() == expected
    assert first.llm.calls + resumed.llm.calls == whole.llm.calls
    assert resumed.research.state() == whole.research.state()
    assert MARKER in generation_calls(resumed)[0]['prompt']
    resumed_card, = decode(generation_calls(resumed)[0]['prompt'])['conclusion_feedback']['cards']
    original = resumed.research.knowledge.experiments[0]
    assert resumed_card['unverified_interpretation']['mechanism_review'] == mechanism_review(original['hypothesis']['text'])
    assert [call['kind'] for call in resumed.llm.calls] == ['generation', 'conclusion']


@pytest.mark.parametrize('aux', [0, 1])
def test_feedback_never_uses_extra_auxiliary_budget(tmp_path, fake_task, aux):
    engine = engine_for(tmp_path/'run', 'ordinary', fake_task, aux=aux)
    result = engine.run()
    assert len(engine.llm.calls) == 2 + aux
    assert engine.research.aux_calls == aux
    assert result['cost_total'] == pytest.approx(2*GENERATION_COST + aux*REVIEW_COST)
    assert (MARKER in generation_calls(engine)[1]['prompt']) is bool(aux)
    assert engine.research.conclusions['r000002a1']['status'] == 'skipped'
    assert engine.research.knowledge.lessons == []


@pytest.mark.parametrize('damage', ['observed_delta', 'agreement', 'review_fields', 'unknown_record', 'old_presentation'])
def test_malformed_conclusion_restore_is_atomic_and_makes_no_calls(tmp_path, fake_task, damage):
    original = engine_for(tmp_path/'original', 'ordinary', fake_task)
    original._stop_at = 1
    original.run()
    saved = original.research.state()
    conclusion = saved['conclusions']['r000001a1']
    if damage == 'observed_delta':
        conclusion['observed_comparison']['measurement']['actual_delta'] = 99.
    elif damage == 'agreement':
        conclusion['verdict_matches_observation'] = 0
    elif damage == 'review_fields':
        conclusion['llm']['verified'] = True
    elif damage == 'unknown_record':
        saved['conclusions']['not-an-experiment'] = saved['conclusions'].pop('r000001a1')
    else:
        saved['presentation'].pop('conclusion_feedback')
        saved['presentation']['memory'] = 'sciel-memory-presentation/4'
    target = engine_for(tmp_path/'target', 'ordinary', fake_task)
    before = target.research.state()
    with pytest.raises(ValueError):
        target.research.load_state(saved)
    assert target.research.state() == before
    assert target.llm.calls == []
    target.log.close()


def test_feedback_follows_ancestry_without_mutating_measurements_or_consuming_exposure(tmp_path, fake_task):
    engine = engine_for(tmp_path/'run', 'ordinary', fake_task)
    engine._stop_at = 1
    engine.run()
    child = next(p for p in engine.db._all.values() if p.parent_id == 'p1')
    before = engine.research.state()
    shown = engine.research.context(engine.db, child, 7, 2)
    assert shown['branch'] == 'p1' and shown['parent_id'] == child.id
    assert shown['conclusion_feedback']['cards'][0]['source_sha256'] == digest(child.code)
    # Equal source text in another root must not grant branch review access.
    other = Program('foreign', child.code, child.score, child.cases, 7, 'seed')
    db = SimpleNamespace(_all={**engine.db._all, other.id: other})
    assert 'conclusion_feedback' not in engine.research.context(db, other, 7, 2)
    shown['conclusion_feedback']['cards'][0]['unverified_interpretation']['explanation'] = 'changed copy'
    assert engine.research.state() == before
    assert engine.research.knowledge.lessons == []


def review_for(record):
    # Historical stored three-field conclusion: do not upgrade this fixture to
    # the mandatory format of newly returned live model responses.
    return dict(status='parsed', llm=dict(verdict='INCONCLUSIVE', explanation=MARKER,
        lessons=[]), observed_forecast_verdict=record['forecast_verdict'],
        observed_comparison=observed_comparison(record),
        verdict_matches_observation=record['forecast_verdict'] == 'INCONCLUSIVE',
        accepted_lesson_ids=[], lesson_error=None, lesson_error_detail=None)


@pytest.mark.parametrize('memory_kind', ['paired', 'proposal'])
def test_full_existing_block_and_lesson_have_priority_over_review(monkeypatch, memory_kind):
    if memory_kind == 'paired':
        from test_research_measurement_state import controller, observe, circle_metric
        from test_research_paired_comparisons import group
        controller_, parent, db, _ = controller(metric=circle_metric())
        record, _ = observe(controller_, parent, db, .6)
        controller_.import_comparison(group(record=record))
        block_key = 'paired_comparisons'
    else:
        from test_research_circle_pair import fixture, response as pair_response, recorded, commit
        f = fixture()
        reply = pair_response()
        _, built = recorded(f, reply)
        assert commit(f, reply, built)
        record = f.research.resolve(f.ident, status='evaluated', score=.9, source=built['code'])
        controller_, parent, db = f.research, f.parent, f.db
        block_key = 'proposal_groups'
    controller_.conclusions[record['id']] = review_for(record)
    lesson_id, = controller_.knowledge.add_lessons([dict(rule='This particular child improved in this fixture.',
        polarity='positive', scope='branch', evidence_ids=[record['id']])], allowed_evidence_ids=[record['id']])
    initial = controller_.context(db, parent, 0, 2)
    block = initial[block_key]
    assert block['cards']
    # Full serialized metadata/omission counts consume memory, not just cards.
    reserved = len(canonical_json(block)) + sum(map(len, initial['lesson_texts']))
    reserved += (len(canonical_json(initial['comparison_facts']))
                 if 'comparison_facts' in initial else 0)
    before = controller_.state()
    monkeypatch.setattr(research_module, 'MAX_MEMORY_CHARS', reserved)
    limited = controller_.context(db, parent, 0, 2)
    assert limited[block_key] == block
    assert limited['lesson_texts'] == initial['lesson_texts']
    assert limited['injected_ids'] == [lesson_id]
    assert 'conclusion_feedback' not in limited
    monkeypatch.setattr(research_module, 'MAX_MEMORY_CHARS', reserved + 8192)
    roomy = controller_.context(db, parent, 0, 2)
    feedback = roomy['conclusion_feedback']
    assert feedback['cards'][0]['experiment_id'] == record['id']
    assert feedback['max_chars'] == 8192
    assert len(canonical_json(feedback)) <= 8192
    assert roomy[block_key] == block and roomy['lesson_texts'] == limited['lesson_texts']
    assert controller_.state() == before
