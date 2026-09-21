"""Real promotion route with inert model/evaluator/draws, never packing execution."""
import json
import sys

import pytest

from sciel.engine import checkpoint
from sciel.engine.llm.client import LLMResponse
from sciel.engine.loop import Engine
from sciel.campaign.budget import CampaignStopped
from sciel.hypothesis.rendering import decode_evidence
from test_repeat_engine_integration import (FakeLLM, FakeDraws, config, journal,
                                           search_observations, task)


class ReflectingModel(FakeLLM):
    def __init__(self, draws, start_step=0):
        super().__init__(start_step)
        self.draws = draws

    def generate(self, prompt, model, temperature=None):
        if 'Review existing experimental evidence and lessons.' not in prompt:
            return super().generate(prompt, model, temperature)
        raw = json.JSONDecoder().raw_decode(prompt.split('EVIDENCE_JSON:', 1)[1].lstrip())[0]
        records = decode_evidence(raw)['observation_catalog']
        groups = {ident: record for ident, record in records.items() if ident.startswith('comparison:')}
        proposals = []
        if groups:
            ident, group = max(groups.items(), key=lambda item: item[1]['batch_id'])
            outcome = group['derived']['outcome']
            if outcome in ('improved', 'regressed'):
                proposals = [dict(rule='Paired fixture result '+ident, scope='branch',
                    polarity='positive' if outcome == 'improved' else 'negative', evidence_ids=[ident])]
        text = json.dumps(dict(lessons=proposals, retire_ids=[]))
        self.calls.append(dict(kind='reflection', prompt=prompt, groups=groups,
            draw_count=len(self.draws.calls), text=text, model=model, temperature=temperature))
        return LLMResponse(text, model, .001, 10, 10)


def build(task, path, draws, *, budget=1, start_step=0, aux=None):
    cfg = config(budget=budget, max_iterations=budget)
    cfg['research'].update(reflection_interval=1, max_aux_calls=2*budget if aux is None else aux)
    model = ReflectingModel(draws, start_step)
    return Engine(cfg, task, str(path), model, python=sys.executable), model


def test_terminal_comparison_is_in_the_one_due_reflection(task, tmp_path, monkeypatch, search_observations):
    draws = FakeDraws(monkeypatch)
    engine, model = build(task, tmp_path/'terminal', draws)
    result = engine.run()
    reflected, = [c for c in model.calls if c['kind'] == 'reflection']
    group, = engine.research.knowledge.comparisons
    assert reflected['draw_count'] == 9
    assert group['id'] in reflected['groups'] and reflected['groups'][group['id']]['derived'] == group['derived']
    assert [c['kind'] for c in model.calls] == ['mutation', 'conclusion', 'reflection']
    assert result['B_evals'] == engine.research.outcomes == engine.research.reflected_through == 1
    assert engine.research.aux_calls == 2 and result['cost_total'] == pytest.approx(.003)
    assert any(lesson['source_support_ids'] == [group['id']] for lesson in engine.research.knowledge.lessons)


def test_final_batch_is_reviewed_and_paired_delivery_keeps_total_memory_limit(task, tmp_path, monkeypatch, search_observations):
    draws = FakeDraws(monkeypatch)
    engine, model = build(task, tmp_path/'two', draws, budget=2)
    engine.run()
    reflections = [c for c in model.calls if c['kind'] == 'reflection']
    groups = engine.research.knowledge.comparisons
    assert [c['draw_count'] for c in reflections] == [9, 15]
    assert all(g['id'] in c['groups'] for g, c in zip(groups, reflections, strict=True))
    generation = [c for c in model.calls if c['kind'] == 'mutation'][1]
    rule = 'Paired fixture result '+groups[0]['id']
    assert rule in generation['prompt']
    lesson = next(l for l in engine.research.knowledge.lessons if l['rule'] == rule)
    attempt = next(e for e in journal(tmp_path/'two')
                   if e.get('phase') == 'attempt' and e['experiment_id'] == 'r000002a1')
    presented, = attempt['lesson_presentation']['presented']
    assert presented['id'] == lesson['id'] and presented['chars'] <= 16000
    omission, = attempt['lesson_presentation']['omitted']
    omitted = next(l for l in engine.research.knowledge.lessons if l['id'] == omission['id'])
    # The existing rank order prefers the paired lesson. It now fits, while the
    # competing ordinary lesson still cannot fit the remaining total budget.
    assert omitted['source_support_ids'] == ['r000001a1']
    assert omission['reason'] == 'memory_char_limit' and omission['chars'] <= 16000
    assert (attempt['lesson_presentation']['rendered_chars'] + omission['chars'] +
            len(json.dumps(attempt['paired_comparisons'], sort_keys=True,
                           separators=(',', ':'), ensure_ascii=True)) > 32000)
    assert omitted['exposure_ids'] == []
    assert lesson['exposure_ids'] == ['r000002a1']
    assert groups[0]['id'] in [c['id'] for c in attempt['paired_comparisons']['cards']]
    assert any(l['source_support_ids'] == [groups[1]['id']] and l['polarity'] == 'negative'
               for l in engine.research.knowledge.lessons)
    assert engine.research.outcomes == 2 and engine.research.aux_calls == 4


def test_fitting_paired_lesson_reaches_next_attempt_without_extra_draws(task, tmp_path, monkeypatch, search_observations):
    draws = FakeDraws(monkeypatch)
    folder = tmp_path/'fitting'
    engine, model = build(task, folder, draws, budget=3)
    generate = model.generate

    def reflection_only_lessons(prompt, *args, **kwargs):
        # Two distinct sources, then a duplicate attempt: its actual generation
        # prompt can receive the second pair's lesson without another draw.
        # Individual conclusions propose no competing advice in this fixture.
        if 'RESEARCH_CONTEXT_JSON:' in prompt and model.mutation_count == 2:
            model.mutation_count = 1
        response = generate(prompt, *args, **kwargs)
        if 'Review this one experiment.' in prompt:
            parsed = json.loads(response.text.split('CONCLUSION_JSON:', 1)[1])
            parsed['lessons'] = []
            response.text = 'CONCLUSION_JSON: ' + json.dumps(parsed)
            model.calls[-1]['text'] = response.text
        return response

    monkeypatch.setattr(model, 'generate', reflection_only_lessons)
    engine.run()
    group = engine.research.knowledge.comparisons[1]
    rule = 'Paired fixture result '+group['id']
    generation = [c for c in model.calls if c['kind'] == 'mutation'][2]
    assert rule in generation['prompt']
    lesson = next(l for l in engine.research.knowledge.lessons if l['rule'] == rule)
    assert lesson['exposure_ids'] == ['r000003a1']
    assert len(draws.calls) == 15 and len(search_observations) == 3
    assert engine.research.outcomes == 3 and engine.research.reflected_through == 3


def test_unknown_repeat_stops_before_due_reflection(task, tmp_path, monkeypatch, search_observations):
    draws = FakeDraws(monkeypatch)
    draws.failure_at = 4
    draws.failure_status = 'infrastructure'
    engine, model = build(task, tmp_path/'stopped', draws)
    result = engine.run()
    assert result['reason'] == 'repeat_stopped'
    assert [c['kind'] for c in model.calls] == ['mutation', 'conclusion']
    assert engine.research.reflected_through == 0 and engine.research.outcomes == 1
    group, = engine.research.knowledge.comparisons
    assert group['derived']['outcome'] == 'incomplete'


def test_terminal_checkpoint_before_reflection_resumes_without_new_draws(task, tmp_path, monkeypatch, search_observations):
    draws = FakeDraws(monkeypatch)
    folder = tmp_path/'resume'
    engine, model = build(task, folder, draws)
    original = checkpoint.save

    def stop_after_terminal(engine, phase):
        path = original(engine, phase)
        if phase == 'repeat_after_draws' and engine.research.outcomes == 1:
            raise RuntimeError('inert crash after terminal checkpoint, before reflection')
        return path

    monkeypatch.setattr(checkpoint, 'save', stop_after_terminal)
    with pytest.raises(RuntimeError, match='inert crash'):
        engine.run()
    engine.log.close()
    assert [c['kind'] for c in model.calls] == ['mutation', 'conclusion']
    assert len(draws.calls) == 9
    monkeypatch.setattr(checkpoint, 'save', original)
    restored, next_model = build(task, folder, draws, start_step=1)
    checkpoint.load(restored, str(folder/'checkpoint.json'))
    restored.run()
    assert len(draws.calls) == 9
    assert [c['kind'] for c in next_model.calls] == ['reflection']
    assert restored.research.reflected_through == 1 and restored.research.aux_calls == 2
    # A terminal checkpoint after that reflection cannot dispatch it again.
    again, last_model = build(task, folder, draws, start_step=1)
    checkpoint.load(again, str(folder/'checkpoint.json'))
    again.run()
    assert last_model.calls == [] and len(draws.calls) == 9


def test_uncheckpointed_reflection_cannot_replay_on_resume(task, tmp_path, monkeypatch, search_observations):
    draws = FakeDraws(monkeypatch)
    folder = tmp_path/'aux_tail'
    engine, model = build(task, folder, draws)
    original = checkpoint.save

    def stop_before_aux_checkpoint(engine, phase):
        if phase == 'repeat_after_reflection':
            raise RuntimeError('inert crash after auxiliary result')
        return original(engine, phase)

    monkeypatch.setattr(checkpoint, 'save', stop_before_aux_checkpoint)
    with pytest.raises(RuntimeError, match='after auxiliary result'):
        engine.run()
    engine.log.close()
    assert [c['kind'] for c in model.calls] == ['mutation', 'conclusion', 'reflection']
    assert len(draws.calls) == 9
    monkeypatch.setattr(checkpoint, 'save', original)
    fresh, fresh_model = build(task, folder, draws, start_step=1)
    try:
        with pytest.raises(checkpoint.IdentityMismatch, match='uncheckpointed search activity|uncheckpointed research'):
            checkpoint.load(fresh, str(folder/'checkpoint.json'))
    finally:
        fresh.repeat_promotion.close()
        fresh.log.close()
    assert fresh_model.calls == [] and len(draws.calls) == 9


def test_aux_call_limit_skips_after_measurements_without_retry(task, tmp_path, monkeypatch, search_observations):
    draws = FakeDraws(monkeypatch)
    folder = tmp_path/'limit'
    engine, model = build(task, folder, draws, aux=1)
    engine.run()
    assert [c['kind'] for c in model.calls] == ['mutation', 'conclusion']
    assert len(draws.calls) == 9 and engine.research.reflected_through == 1
    events = journal(folder)
    context = next(e for e in events if e.get('phase') == 'reflection_context')
    assert engine.research.knowledge.comparisons[0]['id'] in context['experiment_ids']
    assert any(e.get('phase') == 'aux_skipped' and e.get('kind') == 'reflection' for e in events)
    restored, second_model = build(task, folder, draws, start_step=1, aux=1)
    checkpoint.load(restored, str(folder/'checkpoint.json'))
    restored.run()
    assert second_model.calls == [] and len(draws.calls) == 9


def test_provider_failure_after_draws_saves_stop_without_replay(task, tmp_path, monkeypatch, search_observations):
    draws = FakeDraws(monkeypatch)
    folder = tmp_path/'provider_stop'
    engine, model = build(task, folder, draws)
    original = model.generate

    def failed(prompt, *args, **kwargs):
        if 'Review existing experimental evidence and lessons.' in prompt:
            raise CampaignStopped('inert provider failure, no network')
        return original(prompt, *args, **kwargs)

    monkeypatch.setattr(model, 'generate', failed)
    with pytest.raises(CampaignStopped):
        engine.run()
    assert len(draws.calls) == 9 and engine.research.aux_calls == 2
    saved = json.loads((folder/'checkpoint.json').read_text())
    assert saved['research']['stopped_reason'] and len(saved['research']['knowledge']['paired_comparisons']['records']) == 1
    restored, second_model = build(task, folder, draws, start_step=1)
    try:
        checkpoint.load(restored, str(folder/'checkpoint.json'))
        with pytest.raises(CampaignStopped):
            restored.run()
    finally:
        restored.repeat_promotion.close()
        restored.log.close()
    assert second_model.calls == [] and len(draws.calls) == 9
