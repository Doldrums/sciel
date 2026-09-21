"""Actual Engine delivery with inert model/evaluator/draws; no numerical search."""
import copy
import hashlib
import json
import sys

import numpy as np
import pytest

from sciel.engine import checkpoint
from sciel.engine.loop import Engine
from sciel.engine.research import digest
from sciel.hypothesis.restart_trace import RestartTrace
from sciel.hypothesis.prompts import restart_block
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
from sciel.knowledge_base.restart_context import PREFIX, expand_context
from sciel.knowledge_base.memory_references import SCHEMA as SHOWN_SCHEMA, RULES as SHOWN_RULES, expand_comparisons
from sciel.knowledge_base.presentation import comparison_card
from test_repeat_engine_integration import FakeDraws, config, journal, task, search_observations
from test_repeat_reflection_lifecycle import ReflectingModel


def parse(prompt):
    return json.JSONDecoder().raw_decode(prompt.split(PREFIX, 1)[1].lstrip())[0]


class TracedDraws(FakeDraws):
    """Synthetic same-call records attached to inert numeric fixtures only."""
    identity = {'schema': 'fake-repeat-adapter/2', 'runtime': {'kind': 'inert-fixture'}}
    identity['identity_sha256'] = canonical_sha(identity)
    malformed = False

    def draw(self, source, **kwargs):
        receipt = super().draw(source, **kwargs)
        q = receipt['raw_sum']
        identity = dict(source_sha256=receipt['source_sha256'], seed=receipt['seed'],
            runtime_sha256=canonical_sha(self.identity['runtime']), scaffold_sha256='b'*64,
            invocation_id=hashlib.sha256((source+str(receipt['seed'])).encode()).hexdigest()[:32])
        trace = RestartTrace(identity)
        value = np.zeros(78); value[52:] = q/26
        args = {k: value.copy() for k in ('best', 'initial', 'noise', 'ordinary')}
        trace.begin_start(0, incumbent_sum=q, improvements=0)
        trace.begin_helper(args)
        trace.helper_end(raw_kind='none', post_inputs=args, effective=value,
            origin='saved_ordinary_fallback', incumbent_sum=q, improvements=0)
        trace.optimizer_enter(incumbent_sum=q, improvements=0)
        trace.optimizer_exit(incumbent_sum=q, improvements=0)
        trace.optimizer_result(value)
        trace.end_start('completed', incumbent_sum=q, improvements=0)
        captured = trace.finish('restart_limit', incumbent_sum=q, improvements=0)
        if self.malformed:
            captured['total_scheduled_calls'] += 1
        receipt['restart_trace'] = dict(status='captured', reason=None,
            sidecar_sha256='c'*64, trace=captured)
        return receipt


def build(task, folder, draws, *, start=0, inject=True):
    cfg = config(budget=2, max_iterations=2)
    cfg['research'].update(batch_only_lessons=True, reflection_interval=1, max_aux_calls=4,
                           inject_lessons=inject)
    model = ReflectingModel(draws, start_step=start)
    return Engine(cfg, task, str(folder), model, python=sys.executable), model


def test_trace_reaches_reflection_and_next_generation_after_resume_without_new_credit(
        task, tmp_path, monkeypatch, search_observations):
    draws = TracedDraws(monkeypatch)
    folder = tmp_path/'traced'
    first, model = build(task, folder, draws)
    first._stop_at = 1
    first.run()
    assert [call['kind'] for call in model.calls] == ['mutation', 'conclusion', 'reflection']
    assert PREFIX not in model.calls[0]['prompt'] and PREFIX not in model.calls[1]['prompt']
    reflected = expand_context(parse(model.calls[2]['prompt']))
    assert reflected['presented_count'] > 0
    assert all(row['observation_view']['observation']['status'] == 'available'
               for row in reflected['cards'])
    group, = first.research.knowledge.comparisons
    stored = canonical_json(group)
    assert reflected['groups'][0]['group_sha256'] == canonical_sha(group)
    lesson, = first.research.knowledge.lessons
    assert lesson['source_support_ids'] == [group['id']]
    assert lesson['exposure_ids'] == []
    assert len(draws.calls) == 9
    second, next_model = build(task, folder, draws, start=1)
    checkpoint.load(second, str(folder/'checkpoint.json'))
    assert canonical_json(second.research.knowledge.comparisons[0]) == stored
    second.run()
    generated = next(call for call in next_model.calls if call['kind'] == 'mutation')
    context = parse(generated['prompt'])
    assert context['presented_count'] > 0
    assert context['selected_count'] == context['presented_count'] + context['omitted_count'] == 6
    logs = journal(folder)
    event = next(row for row in logs if row.get('phase') == 'restart_observation_presentation'
                 and row['role'] == 'generation')
    assert event['prompt_sha256'] == digest(generated['prompt'])
    assert event['context'] == context and event['quality_changed'] is False
    assert event['delivery'] == 'prompt_prepared_not_model_use'
    attempt = next(row for row in logs if row.get('phase') == 'attempt'
                   and row['experiment_id'] == 'r000002a1')
    comparison_refs = attempt['paired_comparisons']
    assert comparison_refs['schema'] == SHOWN_SCHEMA
    lesson_text = second.research.lesson_text(lesson)
    restored = expand_comparisons(comparison_refs, {lesson['id']: lesson_text})
    assert restored['cards'] == [comparison_card(group)]
    assert SHOWN_RULES in generated['prompt'] and lesson_text in generated['prompt']
    assert (attempt['lesson_presentation']['rendered_chars']
            + len(canonical_json(comparison_refs)) + len(SHOWN_RULES) + len(restart_block(context))) <= 32000
    retained = next(item for item in second.research.knowledge.lessons if item['id'] == lesson['id'])
    assert retained['source_support_ids'] == [group['id']]
    assert retained['exposure_ids'] == ['r000002a1']
    assert len(draws.calls) == 15 and len(search_observations) == 3
    assert second.research.outcomes == 2 and second.research.aux_calls == 4
    assert canonical_json(second.research.knowledge.comparisons[0]) == stored
    before = canonical_json(second.research.knowledge.state())
    with pytest.raises(ValueError):
        second.research.knowledge.add_lessons([dict(rule='Invented trace support', polarity='positive',
            scope='branch', evidence_ids=['trace:'+context['context_sha256']])],
            allowed_evidence_ids=[group['id']])
    assert canonical_json(second.research.knowledge.state()) == before


def test_malformed_diagnostics_do_not_erase_quality_or_stop_engine(
        task, tmp_path, monkeypatch, search_observations):
    draws = TracedDraws(monkeypatch); draws.malformed = True
    engine, model = build(task, tmp_path/'malformed', draws)
    engine._stop_at = 1
    engine.run()
    reflected = expand_context(parse(next(c['prompt'] for c in model.calls if c['kind'] == 'reflection')))
    assert reflected['presented_count'] == 6
    assert all(c['unavailable_reason'] == 'invalid_diagnostic' for c in reflected['cards'])
    group, = engine.research.knowledge.comparisons
    assert group['derived']['outcome'] == 'improved'
    assert engine.research.outcomes == 1 and len(draws.calls) == 9


def test_generation_ablation_omits_trace_and_does_not_create_exposure(
        task, tmp_path, monkeypatch, search_observations):
    draws = TracedDraws(monkeypatch)
    engine, model = build(task, tmp_path/'ablation', draws, inject=False)
    engine.run()
    generated = [c for c in model.calls if c['kind'] == 'mutation']
    assert all(PREFIX not in c['prompt'] for c in generated)
    assert all(not lesson['exposure_ids'] for lesson in engine.research.knowledge.lessons)
    assert len(draws.calls) == 15


def test_context_capacity_omits_whole_diagnostics_and_keeps_full_kb(
        task, tmp_path, monkeypatch, search_observations):
    draws = TracedDraws(monkeypatch)
    engine, model = build(task, tmp_path/'capacity', draws)
    engine._stop_at = 1
    engine.run()
    group, = engine.research.knowledge.comparisons
    before = canonical_json(engine.research.knowledge.state())
    context, selected = engine.research._restart_context([group], 1)
    assert context is None and selected == [group['id']]
    assert canonical_json(engine.research.knowledge.state()) == before
    assert len(draws.calls) == 9


@pytest.mark.parametrize('field', ['restart_context', 'restart_observation',
    'restart_motion_observation', 'restart_motion_policy', 'restart_motion_sources',
    'shown_comparison_references'])
def test_old_presentation_cannot_resume_without_current_diagnostic_contract(
        task, tmp_path, monkeypatch, search_observations, field):
    draws = TracedDraws(monkeypatch)
    folder = tmp_path/field
    first, model = build(task, folder, draws)
    first._stop_at = 1
    first.run()
    state = json.loads((folder/'checkpoint.json').read_text())
    del state['identity']['research']['presentation'][field]
    old = folder/'old.json'; old.write_text(canonical_json(state))
    second, new_model = build(task, folder, draws, start=1)
    before = second.research.state()
    try:
        with pytest.raises(checkpoint.IdentityMismatch):
            checkpoint.load(second, str(old))
        assert second.research.state() == before and new_model.calls == []
        assert len(draws.calls) == 9
    finally:
        second.repeat_promotion.close()
        second.log.close()


@pytest.mark.parametrize('damage', ['changed_text', 'not_in_prompt'])
def test_reference_requires_actual_shown_text_before_attempt_or_exposure(
        task, tmp_path, monkeypatch, search_observations, damage):
    from types import SimpleNamespace
    draws = TracedDraws(monkeypatch)
    engine, model = build(task, tmp_path/damage, draws)
    engine._stop_at = 1
    engine.run()
    group, = engine.research.knowledge.comparisons
    parent = engine.db._all[group['branch']]
    ctx = engine.research.context(engine.db, parent, 0, 2)
    assert ctx['paired_comparisons']['schema'] == SHOWN_SCHEMA
    ctx['fragments'] = []
    texts = ctx['lesson_texts']
    prompt = '\n'.join(texts) if damage == 'changed_text' else 'No supplied lesson text.'
    shown = [text+' changed' for text in texts] if damage == 'changed_text' else texts
    snapshot = SimpleNamespace(research_context=ctx, kb_lessons=shown)
    before = canonical_json(engine.research.state())
    with pytest.raises(ValueError):
        engine.research.begin(snapshot, 0, 'full_rewrite', prompt)
    assert canonical_json(engine.research.state()) == before
    assert len(draws.calls) == 9
