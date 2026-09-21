"""Real controller R→memory→next prompt using inert model and numeric fixtures.

No candidate, solver, provider, network, or historical Engine is executed.
"""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from sciel.config import apply_defaults
from sciel.engine import research as research_module
from sciel.engine.research import ResearchController, digest
from sciel.engine.types import Program
from sciel.hypothesis import comparison_review
from sciel.hypothesis.auxiliary_schema import response_format
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
from test_research_measurement_state import begin_fixture, circle_metric, controller
from test_research_paired_comparisons import decoded_context, group, proposal as lesson_proposal


def review_proposal(g):
    return dict(comparison_id=g['id'], comparison_sha256=canonical_sha(g),
        interpretation='The measured mixed outcome does not justify advancement.',
        limitations='Three shared seeds do not establish an isolated causal mechanism.',
        next_hypothesis=dict(change='Change only the reinsertion order.',
            expected_observation='No paired raw or exact sum declines.',
            falsification='Any negative raw or exact paired delta.'))


def fixture(*, enabled=True, reflect=False, **settings):
    cfg = dict(comparison_reviews=enabled, structured_auxiliary=True,
        reflection_interval=1, max_aux_calls=4, batch_only_lessons=True)
    cfg.update(settings)
    c, parent, db, events = controller(metric=circle_metric(), **cfg)
    calls, charges = [], []
    f = NS(c=c, parent=parent, db=db, events=events, calls=calls, charges=charges,
           reply=None)

    def generate(prompt, model, **options):
        calls.append(dict(prompt=prompt, model=model, options=copy.deepcopy(options)))
        kind = options['response_format']['json_schema']['name']
        if kind == 'sciel_conclusion_v2':
            value = dict(verdict='CONFIRMED', explanation='The measured delta lies inside the forecast interval.',
                lessons=[], mechanism_review=[dict(evidence_kind='unavailable',
                    claim_quote='A small local improvement.', source_quotes=[], assessment='unknown', forecast_ref=None,
                    direct_effect='The fixture does not establish a causal mechanism.',
                    untested_downstream='Quality causality remains unverified.')])
        else:
            value = copy.deepcopy(f.reply)
        return NS(text=json.dumps(value), cost=0., model='fake', prompt_tokens=1,
                  completion_tokens=1, finish_reason='stop', refusal=None)

    c.llm = NS(models=['fake'], max_retries=1, generate=generate)
    c.charge = charges.append
    ident, _ = begin_fixture(c, c.context(db, parent, 0, 1))
    code = 'def run_search():\n    return .6\n'
    raw = 'HYPOTHESIS_JSON: ' + json.dumps(dict(hypothesis='A small local improvement.',
        predicted_low=.05, predicted_high=.2, reasoning='A bounded fixture prediction.', confidence=.5))
    raw += '\n```python\n' + code + '```'
    assert c.proposal(ident, NS(text=raw, cost=0., prompt_tokens=1, completion_tokens=1), code, NS(code=code))
    f.record = c.resolve(ident, status='evaluated', score=.6, source=code)
    c.complete_step(reflect=False)
    assert not c.queue and c.conclusions[f.record['id']]['status'] == 'parsed'
    f.group = group(record=f.record, radii=(.02, .005, .02))
    c.import_comparison(f.group)
    f.reply = dict(lessons=[], retire_ids=[])
    if enabled:
        f.reply['comparison_reviews'] = [review_proposal(f.group)]
    if reflect:
        c.complete_step()
    return f


def reflection_event(f):
    return next(row for row in reversed(f.events) if row['phase'] == 'reflection')


def restored(f, state=None):
    c = ResearchController(f.c.config, f.c.llm,
        NS(research=lambda **_: pytest.fail('checkpoint replayed a research event')),
        lambda _: pytest.fail('checkpoint incurred a charge'), metric=f.c.metric)
    c.load_state(copy.deepcopy(f.c.state() if state is None else state))
    return c


def test_mixed_reflection_checkpoint_and_next_generation_preserve_review_without_learning():
    f = fixture()
    before = f.c.knowledge.state()
    f.c.complete_step()
    assert f.c.knowledge.state() == before
    assert f.c.reflections == 1 and f.c.reflection_changes == 0
    assert f.c.state()['comparison_review_count'] == 1
    assert f.c.attempts == f.c.outcomes == 1 and f.c.aux_calls == 2
    assert f.c.knowledge.lessons == []
    card = f.c.state()['comparison_reviews'][f.group['id']]
    assert card['host_facts']['derived'] == f.group['derived']
    assert card['host_facts']['derived']['outcome'] == 'mixed'
    assert card['host_facts']['derived']['all_no_loss'] is False
    assert card['unverified_interpretation']['next_hypothesis'] == f.reply['comparison_reviews'][0]['next_hypothesis']
    assert card['writer'] == dict(phase='reflection', reference=1,
        raw_response_sha256=digest(json.dumps(f.reply)))
    assert reflection_event(f)['comparison_review_ids'] == [f.group['id']]
    assert reflection_event(f)['comparison_reviews_changed'] is True
    assert reflection_event(f)['changed'] is False and reflection_event(f)['error'] is None
    reflection_call = f.calls[-1]
    assert reflection_call['options']['response_format'] == response_format('reflection', comparison_reviews=True)
    assert canonical_sha(f.group) in reflection_call['prompt']
    assert 'COMPARISON_REVIEW_SOURCES_JSON:' in reflection_call['prompt']
    clone = restored(f)
    assert clone.state() == f.c.state()
    context = f.c.context(f.db, f.parent, 0, 2)
    assert clone.context(f.db, f.parent, 0, 2) == context
    assert context['injected_ids'] == [] and context['lesson_texts'] == []
    assert context['comparison_reviews']['cards'] == [card]
    ident, prompt = begin_fixture(f.c, context)
    sent = decoded_context(prompt)
    assert sent['comparison_reviews'] == context['comparison_reviews']
    attempt = next(row for row in f.events if row['phase'] == 'attempt' and row['experiment_id'] == ident)
    assert attempt['comparison_reviews'] == context['comparison_reviews']
    assert attempt['comparison_review_delivery'] == context['comparison_review_delivery']
    assert f.c.pending[ident]['injected_ids'] == []
    assert f.c.knowledge.state() == before
    assert len(f.calls) == 2  # Preparing generation is not a provider call.


@pytest.mark.parametrize('failure', ['digest', 'unknown_group', 'lesson', 'retirement'])
def test_reflection_review_lesson_and_retirement_admission_is_atomic(failure):
    f = fixture()
    old_id, = f.c.knowledge.add_lessons([lesson_proposal(f.record['id'])],
                                     allowed_evidence_ids=[f.record['id']])
    replacement = lesson_proposal(f.record['id'])
    replacement.update(rule='A replacement describing only the saved ordinary observation.', branch='p1')
    f.reply.update(lessons=[replacement], retire_ids=[old_id])
    if failure == 'digest':
        f.reply['comparison_reviews'][0]['comparison_sha256'] = '0' * 64
    elif failure == 'unknown_group':
        f.reply['comparison_reviews'][0]['comparison_id'] = 'comparison:missing'
    elif failure == 'lesson':
        f.reply['lessons'][0]['evidence_ids'] = ['missing']
    else:
        f.reply['retire_ids'] = ['missing']
    before = f.c.knowledge.state()
    f.c.complete_step()
    assert f.c.knowledge.state() == before
    assert f.c.state()['comparison_reviews'] == {}
    assert f.c.reflections == f.c.reflection_changes == 0
    event = reflection_event(f)
    assert event['status'] == 'parsed' and event['error'] == 'ValueError'
    assert event['comparison_review_ids'] == [] and not event['comparison_reviews_changed']


@pytest.mark.parametrize('failure', ['host_fact', 'prose', 'hash', 'drop_card',
    'drop_block', 'drop_delivery', 'boolean_cap'])
def test_forged_or_omitted_review_snapshot_fails_before_attempt(failure):
    f = fixture(reflect=True)
    ctx = f.c.context(f.db, f.parent, 0, 2)
    if failure == 'host_fact':
        ctx['comparison_reviews']['cards'][0]['host_facts']['derived']['all_no_loss'] = True
    elif failure == 'prose':
        ctx['comparison_reviews']['cards'][0]['unverified_interpretation']['interpretation'] = 'Invented verified success.'
    elif failure == 'hash':
        ctx['comparison_reviews']['cards'][0]['comparison_sha256'] = '0' * 64
    elif failure == 'drop_card':
        ctx['comparison_reviews']['cards'].clear()
    elif failure == 'drop_block':
        ctx.pop('comparison_reviews')
    elif failure == 'drop_delivery':
        ctx.pop('comparison_review_delivery')
    else:
        ctx['comparison_review_delivery']['max_chars'] = True
    before, events = f.c.state(), copy.deepcopy(f.events)
    with pytest.raises(ValueError, match='comparison review'):
        begin_fixture(f.c, ctx)
    assert f.c.state() == before and f.events == events


@pytest.mark.parametrize('damage', ['host_fact', 'missing_reviews', 'missing_count',
    'boolean_count', 'future_writer', 'unreflected_writer', 'zero_reflections', 'policy_identity'])
def test_checkpoint_rejects_corrupted_review_state_atomically(damage):
    f = fixture(reflect=True)
    broken = copy.deepcopy(f.c.state())
    if damage == 'host_fact':
        broken['comparison_reviews'][f.group['id']]['host_facts']['derived']['all_no_loss'] = True
    elif damage == 'missing_reviews': broken.pop('comparison_reviews')
    elif damage == 'missing_count': broken.pop('comparison_review_count')
    elif damage == 'boolean_count': broken['comparison_review_count'] = True
    elif damage == 'future_writer': broken['comparison_reviews'][f.group['id']]['writer']['reference'] = 2
    elif damage == 'unreflected_writer': broken['reflected_through'] = 0
    elif damage == 'zero_reflections': broken['reflections'] = 0
    else: broken['auxiliary_schema']['review_policy']['schema'] = 'changed'
    before = f.c.state()
    with pytest.raises(ValueError):
        f.c.load_state(broken)
    assert f.c.state() == before


def test_ablation_retains_reviews_but_never_exposes_their_prose_or_credits():
    f = fixture(reflect=True, inject_lessons=False)
    before = f.c.knowledge.state()
    ctx = f.c.context(f.db, f.parent, 0, 2)
    block = ctx['comparison_reviews']
    assert not block['enabled'] and block['cards'] == []
    assert block['retained_count'] == block['omitted_count'] == 1
    assert block['omitted'][0]['reason'] == 'disabled'
    ident, prompt = begin_fixture(f.c, ctx)
    assert f.reply['comparison_reviews'][0]['interpretation'] not in prompt
    assert f.reply['comparison_reviews'][0]['next_hypothesis']['falsification'] not in prompt
    assert f.c.pending[ident]['injected_ids'] == []
    assert f.c.knowledge.state() == before and f.c.state()['comparison_review_count'] == 1


def test_whole_review_is_omitted_when_remaining_memory_cannot_fit_it(monkeypatch):
    f = fixture(reflect=True)
    initial = f.c.context(f.db, f.parent, 0, 2)
    measured = sum(len(canonical_json(initial[key])) for key in ('paired_comparisons', 'comparison_facts'))
    monkeypatch.setattr(research_module, 'MAX_MEMORY_CHARS', measured + 1600)
    ctx = f.c.context(f.db, f.parent, 0, 2)
    assert ctx['comparison_reviews']['cards'] == []
    assert ctx['comparison_reviews']['omitted_count'] == 1
    assert ctx['comparison_reviews']['omitted'][0]['reason'] == 'memory_char_limit'
    assert ctx['comparison_reviews']['rendered_chars'] == len(canonical_json(ctx['comparison_reviews']))
    assert len(canonical_json(ctx['comparison_reviews'])) <= 1600
    _, prompt = begin_fixture(f.c, ctx)
    assert f.reply['comparison_reviews'][0]['interpretation'] not in prompt


def test_tiny_budget_keeps_explicit_delivery_omission_without_partial_card(monkeypatch):
    f = fixture(reflect=True)
    initial = f.c.context(f.db, f.parent, 0, 2)
    measured = sum(len(canonical_json(initial[key])) for key in ('paired_comparisons', 'comparison_facts'))
    monkeypatch.setattr(research_module, 'MAX_MEMORY_CHARS', measured + 1)
    ctx = f.c.context(f.db, f.parent, 0, 2)
    assert 'comparison_reviews' not in ctx
    assert ctx['comparison_review_delivery']['max_chars'] <= 1
    assert ctx['comparison_review_delivery']['omission'] == 'memory_char_limit'
    ident, prompt = begin_fixture(f.c, ctx)
    attempt = next(row for row in f.events if row['phase'] == 'attempt' and row['experiment_id'] == ident)
    assert attempt['comparison_review_delivery'] == ctx['comparison_review_delivery']
    assert attempt['comparison_reviews'] is None
    assert f.reply['comparison_reviews'][0]['interpretation'] not in prompt


@pytest.mark.parametrize('damage', ['inflated_review_allocation', 'other_optional_memory'])
def test_self_consistent_review_cannot_bypass_shared_memory_before_attempt(monkeypatch, damage):
    f = fixture(reflect=True)
    if damage == 'inflated_review_allocation':
        initial = f.c.context(f.db, f.parent, 0, 2)
        measured = sum(len(canonical_json(initial[key])) for key in ('paired_comparisons', 'comparison_facts'))
        monkeypatch.setattr(research_module, 'MAX_MEMORY_CHARS', measured + 1600)
        ctx = f.c.context(f.db, f.parent, 0, 2)
        assert ctx['comparison_reviews']['cards'] == []
        ctx['comparison_reviews'] = comparison_review.context(
            f.c.state()['comparison_reviews'], f.c._review_groups(), 'p1', 10000)
        assert len(ctx['comparison_reviews']['cards']) == 1
        ctx['comparison_review_delivery'] = dict(max_chars=10000, omission=None)
    else:
        ctx = f.c.context(f.db, f.parent, 0, 2)
        ctx['lesson_texts'] = ['X' * research_module.MAX_MEMORY_CHARS]
    before, events = f.c.state(), copy.deepcopy(f.events)
    with pytest.raises(ValueError, match='shared memory'):
        begin_fixture(f.c, ctx)
    assert f.c.state() == before and f.events == events


def test_review_does_not_keep_aged_out_group_alive():
    f = fixture(reflect=True, max_experiments=1)
    assert f.group['id'] in f.c.state()['comparison_reviews']
    replacement = group(batch='later', radii=(.005, .02, .005))
    f.c.import_comparison(replacement)
    assert f.group['id'] not in f.c.knowledge._evidence()
    assert f.c.state()['comparison_reviews'] == {}
    assert f.c.state()['comparison_review_count'] == 0
    assert f.c.attempts == f.c.outcomes == f.c.reflections == 1
    assert f.c.knowledge.lessons == []
    assert restored(f).state() == f.c.state()


def test_authored_mixed_group_review_fits_next_prompt_before_imported_conclusions(monkeypatch):
    from research_history_fixture import build_history
    from review_evidence_fixture import mixed_group
    from sciel.hypothesis import circle_optimizer_traced_body as builder
    _, state = build_history()
    source = builder.source_for()
    cfg = {**state['config'], 'model': None, 'comparison_reviews': True}
    events = []
    c = ResearchController(cfg, NS(models=['fixture']),
        NS(research=lambda **row: events.append(copy.deepcopy(row))),
        lambda _: pytest.fail('no paid request'), metric=state['metric'])
    c.knowledge.load_state(copy.deepcopy(state['knowledge']))
    c.conclusions = copy.deepcopy(state['conclusions'])
    c.attempts = c.outcomes = 2  # Avoid collision with the two borrowed ordinary IDs.
    c._external_history = {'fixture_only': 'exercise imported-conclusion presentation priority'}
    latest = state['knowledge']['experiments'][-1]
    paired = mixed_group(challenger=latest['source_sha256'], metric=state['metric'])
    c.import_comparison(paired)
    assert paired['derived']['all_no_loss'] is False and paired['derived']['outcome'] == 'mixed'
    proposed = review_proposal(paired)
    writer = dict(phase='reflection', reference=2,
                  raw_response_sha256=digest('explicit synthetic review fixture, not provider output'))
    c._comparison_reviews = comparison_review.build_cards([proposed], c._review_groups(), writer)
    latest = state['knowledge']['experiments'][-1]
    assert digest(source) == latest['parent_source_sha256']
    parent = Program('p1', source, latest['parent_score'], copy.deepcopy(latest['parent_cases']), 0, 'seed')
    db = NS(_all={'p1': parent})
    before = c.knowledge.state()
    ctx = c.context(db, parent, 0, 3)
    block = ctx['comparison_reviews']
    measurements = sum(len(canonical_json(ctx[key])) for key in ('paired_comparisons', 'comparison_facts'))
    # Reserve a complete review with little spare room: earlier conclusion
    # feedback would consume its slot if presentation priority regressed.
    memory_cap = measurements + len(canonical_json(block)) + 100
    monkeypatch.setattr(research_module, 'MAX_MEMORY_CHARS', memory_cap)
    ctx = c.context(db, parent, 0, 3)
    block = ctx['comparison_reviews']
    assert block['presented_count'] == 1 and block['omitted_count'] == 0
    assert block['cards'] == [c._comparison_reviews[paired['id']]]
    assert block['cards'][0]['host_facts']['derived'] == paired['derived']
    assert [(p['raw_physical_sum_direction'], p['exact_certified_sum_direction'])
        for p in block['cards'][0]['host_facts']['signs']['pairs']] == [
            ('negative', 'positive'), ('positive', 'negative'), ('negative', 'negative')]
    from sciel.hypothesis.conclusion_feedback import conclusion_feedback
    from sciel.hypothesis.prompts import initializer_block, restart_block
    from sciel.knowledge_base.restart_prediction_context import block as prediction_block
    measurements = sum(len(canonical_json(ctx[key])) for key in ('paired_comparisons', 'comparison_facts'))
    remaining_before_review = research_module.MAX_MEMORY_CHARS - measurements
    earlier_first = conclusion_feedback(c.conclusions,
        {r['id']: r for r in c.knowledge.experiments}, 'p1',
        max_chars=remaining_before_review, batch_only_lessons=True)
    assert earlier_first is not None
    lost_review = comparison_review.context(c._comparison_reviews, c._review_groups(), 'p1',
        max_chars=max(0, min(comparison_review.MAX_CONTEXT_CHARS,
            remaining_before_review - len(canonical_json(earlier_first)))))
    assert lost_review is None or lost_review['presented_count'] == 0
    actual_chars = (sum(len(text) for text in ctx['lesson_texts'])
        + sum(len(canonical_json(ctx[key])) for key in ('paired_comparisons', 'comparison_facts',
            'proposal_groups', 'conclusion_feedback', 'comparison_reviews') if key in ctx)
        + len(initializer_block(ctx.get('initializer_evidence')))
        + len(restart_block(ctx.get('restart_observations')))
        + len(prediction_block(ctx.get('restart_predictions'))))
    assert actual_chars <= research_module.MAX_MEMORY_CHARS == memory_cap
    ident, prompt = begin_fixture(c, ctx)
    assert ident == 'r000003a1'
    delivered = decoded_context(prompt)
    assert delivered['comparison_reviews'] == block
    assert proposed['interpretation'] in canonical_json(delivered['comparison_reviews'])
    assert delivered['comparison_reviews']['cards'][0]['unverified_interpretation']['next_hypothesis'] == proposed['next_hypothesis']
    assert c.pending[ident]['injected_ids'] == [] and c.knowledge.state() == before
    print(f'actual optional memory={actual_chars}/{memory_cap}; review={len(canonical_json(block))}; '
          f'conclusion={len(canonical_json(ctx.get("conclusion_feedback")))}; prompt={len(prompt)}')


def test_disabled_mode_keeps_legacy_reflection_and_no_review_state_or_delivery():
    f = fixture(enabled=False, reflect=True)
    assert f.c.reflections == 1
    assert 'comparison_reviews' not in f.c.state()
    assert 'comparison_review_count' not in f.c.state()
    assert f.calls[-1]['options']['response_format'] == response_format('reflection')
    assert 'COMPARISON_REVIEW_SOURCES_JSON:' not in f.calls[-1]['prompt']
    ctx = f.c.context(f.db, f.parent, 0, 2)
    assert 'comparison_reviews' not in ctx and 'comparison_review_delivery' not in ctx
    ctx['comparison_reviews'] = {}
    before = f.c.state()
    with pytest.raises(ValueError, match='disabled'):
        begin_fixture(f.c, ctx)
    assert f.c.state() == before


@pytest.mark.parametrize('value', [1, 0, None, 'true', {}, []])
def test_config_and_direct_controller_require_exact_boolean(value):
    with pytest.raises(ValueError, match='comparison_reviews'):
        apply_defaults({'research': {'enabled': True, 'structured_auxiliary': True,
                                    'comparison_reviews': value}})
    cfg = apply_defaults({'research': {'enabled': True, 'structured_auxiliary': True}})['research']
    cfg['comparison_reviews'] = value
    with pytest.raises(ValueError, match='comparison_reviews'):
        ResearchController(cfg, NS(models=['fake']), NS(), lambda _: None)


@pytest.mark.parametrize('structured', [False, None])
def test_review_mode_requires_explicit_structured_auxiliary(structured):
    research = dict(enabled=True, comparison_reviews=True)
    if structured is not None:
        research['structured_auxiliary'] = structured
    with pytest.raises(ValueError, match='requires structured_auxiliary'):
        apply_defaults({'research': research})
    cfg = apply_defaults({'research': {'enabled': True}})['research']
    cfg['comparison_reviews'] = True
    with pytest.raises(ValueError, match='requires structured_auxiliary'):
        ResearchController(cfg, NS(models=['fake']), NS(), lambda _: None)
