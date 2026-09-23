"""Complete R prompts obey token limits without losing atomic source evidence."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from sciel.config import apply_defaults
from sciel.engine.reflection_budget import ReflectionBudgetExceeded, contract
from sciel.engine.reflection_context import prepare_reflection_context
from sciel.engine.research import ResearchController
from sciel.hypothesis.conclusion_feedback import reflection_review_context
from sciel.hypothesis.prompts import reflection_prompt
from sciel.knowledge_base import KnowledgeBase
from sciel.knowledge_base.catalogs import canonical_json
from test_reflection_anchor_allocation import add, observe
from test_conclusion_feedback import inputs


def count(prompt):
    return len(prompt.encode('utf-8')) + 7  # Known fixture tokenizer + chat framing.


class Local:
    local_inference = True
    models = ['fixture']
    max_retries = 1
    max_tokens = 512
    prompt_tokenizer_identity = dict(schema='fixture/1', tokenizer_revision='a' * 40,
                                     chat_template_sha256='b' * 64, enable_thinking=False)
    count_prompt_tokens = staticmethod(count)

    def __init__(self):
        self.calls = []

    def generate(self, prompt, model, temperature=None, **options):
        self.calls.append(dict(prompt=prompt, model=model, temperature=temperature, **options))
        return SimpleNamespace(text='{"lessons":[],"retire_ids":[]}', model=model,
            cost=0.0, prompt_tokens=count(prompt), completion_tokens=10,
            finish_reason='stop', refusal=None, response_error=None)


def configuration(context=100000, **changes):
    cfg = apply_defaults({'llm': {'backend': 'local_openai'}, 'research': dict(
        enabled=True, reflection_interval=4, structured_auxiliary=True,
        reflection_context_tokens=context, **changes)})['research']
    return cfg


def prepare(kb, context=100000, conclusions=None, **changes):
    cfg = configuration(context, **changes)
    llm = Local()
    return prepare_reflection_context(kb, conclusions or {}, config=cfg,
        interval=4, through=4, selection_epoch=0,
        token_budget=contract(cfg, llm), count_prompt_tokens=llm.count_prompt_tokens)


def controller(context, kb=None):
    llm, events = Local(), []
    cfg = configuration(context)
    c = ResearchController(cfg, llm, SimpleNamespace(research=lambda **e: events.append(deepcopy(e))), lambda _: None)
    if kb is not None:
        # Match the controller's metric without inventing an outcome at runtime.
        for row in kb.experiments:
            c.knowledge.record(dict(row, metric=deepcopy(c.metric), parent_cases=None, evaluation=None))
        c.attempts = c.outcomes = 4
    return c, llm, events


@pytest.mark.parametrize('bad', [None, True, 0, -1, 8192.0, '8192'])
def test_invalid_context_limits_rejected(bad):
    with pytest.raises(ValueError, match='reflection_context_tokens'):
        configuration(bad)


def test_cap_is_local_only_and_absence_does_not_access_tokenizer():
    with pytest.raises(ValueError, match='local_openai'):
        apply_defaults({'research': {'reflection_context_tokens': 8192}})
    cfg = apply_defaults({'research': {'enabled': True}})['research']
    external = SimpleNamespace(models=['fixture'], max_retries=1)
    c = ResearchController(cfg, external, SimpleNamespace(), lambda _: None)
    assert c._reflection_token_budget is None
    assert 'reflection_token_budget' not in c.state()
    with pytest.raises(ValueError, match='pinned local'):
        ResearchController(configuration(), external, SimpleNamespace(), lambda _: None)


def test_exact_full_prompt_boundary_includes_unicode_framing_and_output():
    kb = KnowledgeBase()
    ident = observe(kb, 1)
    add(kb, 'Точная проверка 🌍' * 10, [ident])
    before = kb.state()
    wide = prepare(kb)
    exact = count(wide['prompt']) + Local.max_tokens
    bounded = prepare(kb, exact)
    assert bounded['prompt'] == wide['prompt']
    assert bounded['context_event']['token_budget']['total_reserved_tokens'] == exact
    smaller = prepare(kb, exact - 1)
    assert count(smaller['prompt']) + Local.max_tokens <= exact - 1
    assert kb.state() == before


def test_framing_overflow_precedes_counters_requests_and_knowledge_changes():
    kb = KnowledgeBase()
    observe(kb, 1)
    fixed = count(reflection_prompt([], [], structured=True)) + Local.max_tokens
    c, llm, events = controller(fixed - 1, kb)
    before = c.state()
    with pytest.raises(ReflectionBudgetExceeded):
        c.finish_conclusions()
    assert c.state() == before and not llm.calls
    assert not any(e['phase'] == 'aux_start' for e in events)
    rejected, = events
    assert rejected['phase'] == 'reflection_budget_rejected'
    assert rejected['generation_dispatch'] == 'not_dispatched'
    assert rejected['prompt_sha256'] == hashlib.sha256(rejected['prompt'].encode()).hexdigest()
    assert rejected['prompt_tokens'] == count(rejected['prompt'])


def test_no_whole_evidence_fits_is_durable_skip_without_dispatch_or_fake_reflection():
    kb = KnowledgeBase()
    ident = observe(kb, 1, pad=2000)
    fixed = count(reflection_prompt([], [], structured=True)) + Local.max_tokens
    c, llm, events = controller(fixed, kb)
    before = c.knowledge.state()
    c.finish_conclusions()
    assert not llm.calls and c.aux_calls == 0 and c.reflections == 0
    assert c.reflection_attempts == 1 and c.reflected_through == 4
    assert c.knowledge.state() == before
    context = next(e for e in events if e['phase'] == 'reflection_context')
    assert context['experiment_ids'] == []
    assert ident in context['token_budget']['omitted_observation_ids']
    skipped = next(e for e in events if e['phase'] == 'aux_skipped')
    assert skipped['reason'] == 'no_evidence_fits_token_budget' and skipped['cost'] == 0
    assert not any(e['phase'] == 'aux_start' for e in events)
    restored, resumed_llm, resumed_events = controller(fixed)
    restored.load_state(c.state())
    restored.finish_conclusions()
    assert restored.state() == c.state() and not resumed_llm.calls and not resumed_events


def test_atomic_lesson_keeps_sources_and_negative_exposure_or_is_wholly_omitted():
    kb = KnowledgeBase()
    source = observe(kb, 1)
    lesson = add(kb, 'Local association only.', [source])
    negative = observe(kb, 2, delta=-.1, injected=[lesson])
    rows, lessons = kb.reflection_context(4, selection_epoch=0)
    limit = count(reflection_prompt(rows, lessons, structured=True)) + Local.max_tokens
    recent = observe(kb, 3, pad=2000)
    before = kb.state()
    bounded = prepare(kb, limit)
    assert [l['id'] for l in bounded['lessons']] == [lesson]
    assert {r['id'] for r in bounded['records']} == {source, negative}
    assert recent in bounded['context_event']['token_budget']['omitted_observation_ids']
    smaller = prepare(kb, count(reflection_prompt([], [], structured=True)) + Local.max_tokens)
    assert smaller['lessons'] == [] and smaller['records'] == []
    assert kb.state() == before


def test_multiple_observations_fit_without_mutating_source_fields():
    kb = KnowledgeBase()
    for i in range(1, 5):
        observe(kb, i, delta=-.1 if i % 2 else .1)
    before = kb.state()
    prepared = prepare(kb)
    assert len(prepared['records']) == 4 and prepared['skip_reason'] is None
    assert prepared['context_event']['token_budget']['omitted_observation_ids'] == []
    assert kb.state() == before


def test_whole_reviews_use_remaining_full_prompt_budget_and_keep_omissions():
    reviews, records = inputs(2)
    kb = KnowledgeBase()
    for row in records.values():
        raw = deepcopy(row)
        raw.pop('code_fragment_sha256', None)
        kb.record(raw)
    before = kb.state()
    wide = prepare(kb, conclusions=reviews)
    # Independently choose a limit for source evidence with one complete review,
    # including framing/escaping; additional reviews must be omitted as units.
    rows = wide['records']
    one = reflection_review_context({rows[-1]['id']: reviews[rows[-1]['id']]},
        kb._evidence(), [rows[-1]['id']])
    limit = count(reflection_prompt(rows, [], structured=True, review_context=one)) + Local.max_tokens + 600
    bounded = prepare(kb, limit, conclusions=reviews)
    assert len(bounded['records']) == 2
    assert count(bounded['prompt']) + Local.max_tokens <= limit
    assert bounded['context_event']['review_presented_count'] < 2
    assert bounded['context_event']['review_omitted_count'] > 0
    assert kb.state() == before
    # The review selector also handles a budget that fits no wrapper/cards.
    assert reflection_review_context(reviews, records, list(records), accepts=lambda _: False) is None


def test_exact_reservation_is_bound_even_if_provider_default_changes_and_resume_is_stable():
    kb = KnowledgeBase()
    observe(kb, 1)
    c, llm, events = controller(100000, kb)
    before = c.state()
    restored, second, second_events = controller(100000)
    restored.load_state(before)
    llm.max_tokens = 4096
    c.finish_conclusions()
    restored.finish_conclusions()
    assert llm.calls[0]['max_tokens'] == 512  # Frozen into options, not a heuristic.
    assert llm.calls == second.calls
    context = next(e for e in events if e['phase'] == 'reflection_context')
    assert context == next(e for e in second_events if e['phase'] == 'reflection_context')
    assert c.state() == restored.state()
    state = c.state()
    state['reflection_token_budget']['tokenizer']['tokenizer_revision'] = 'c' * 40
    with pytest.raises(ValueError, match='token budget mismatch'):
        restored.load_state(state)


def test_model_context_and_output_reservation_must_be_compatible():
    with pytest.raises(ValueError, match='exceed its reserved output'):
        contract(configuration(512), Local())




def test_fixed_record_dependent_review_framing_is_counted_not_only_evidence(monkeypatch):
    import sciel.engine.reflection_context as preparation
    actual = preparation.reflection_prompt
    def large_source_block(records, lessons, **kwargs):
        return actual(records, lessons, **kwargs) + ('📚' * 1500 if records else '')
    monkeypatch.setattr(preparation, 'reflection_prompt', large_source_block)
    kb = KnowledgeBase()
    observe(kb, 1)
    context = count(actual([], [], structured=True)) + Local.max_tokens + 3000
    bounded = prepare(kb, context)
    assert bounded['skip_reason'] == 'no_evidence_fits_token_budget'
    assert not bounded['records']


def test_declared_budget_change_or_missing_checkpoint_identity_rejects_resume():
    c, _, _ = controller(100000)
    state = c.state()
    different, _, _ = controller(99999)
    with pytest.raises(ValueError, match='configuration mismatch'):
        different.load_state(state)
    state.pop('reflection_token_budget')
    with pytest.raises(ValueError, match='token budget mismatch'):
        c.load_state(state)
