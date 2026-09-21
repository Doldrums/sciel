"""Source-diverse retrieval against real KB observations; all sources are inert.

These are mechanism/provenance tests, not tests of semantic equivalence, model
adherence, causal effects, or packing quality. No evaluator or model is called.
"""
import copy
import json
import random
from types import SimpleNamespace

import pytest

from sciel.config import apply_defaults
from sciel.engine.prompt.build import build
from sciel.engine.research import ResearchController, digest
from sciel.engine.types import Program, Snapshot
from sciel.hypothesis.measurement import metric_context
from sciel.hypothesis.rendering import decode_evidence
from sciel.knowledge_base import KnowledgeBase


def record(kb, number, *, delta=.125, branch='p1', injected=()):
    ident = f'r{number:06d}a1'
    code = f'# inert observation {ident}\n'
    measured = delta is not None
    score = .5 + delta if measured else None
    kb.record(dict(id=ident, branch=branch, parent_id=branch, parent_score=.5,
        parent_source_sha256=digest('# inert measured parent\n'),
        status='evaluated' if measured else 'eval_failed', score=score,
        actual_delta=score - .5 if measured else None,
        source_sha256=digest(code), proposed_sha256=digest(code),
        injected_ids=list(injected), metric=metric_context(), parent_cases={},
        evaluation=dict(ok=measured, score=score if measured else 0., cases={},
                        reached='full', confirmed=False,
                        raw_first_score=score if measured else 0.)))
    return ident


def add(kb, rule, evidence, *, polarity='positive', scope='branch'):
    return kb.add_lessons([dict(rule=rule, polarity=polarity, scope=scope,
        evidence_ids=list(evidence))], allowed_evidence_ids=list(evidence))[0]


def fixture(kb=None, *, branch='p1', **settings):
    kb = KnowledgeBase() if kb is None else kb
    config = apply_defaults({'research': {'enabled': True,
        'max_lessons': kb.max_lessons, 'max_experiments': kb.max_experiments,
        'max_fragments': kb.max_fragments, 'max_aux_calls': 0,
        'reflection_interval': 0, **settings}})['research']
    events = []
    llm = SimpleNamespace(models=['fixture'], max_retries=1,
        generate=lambda *a, **kw: pytest.fail('retrieval must not call a model'))
    controller = ResearchController(config, llm,
        SimpleNamespace(research=lambda **kw: events.append(copy.deepcopy(kw))),
        lambda cost: pytest.fail('retrieval must not charge a model call'))
    controller.knowledge = kb
    controller.attempts = controller.outcomes = len(kb.experiments)
    parent = Program(branch, '# packing\n', .5, {}, 0, 'seed')
    return controller, SimpleNamespace(_all={branch: parent}), parent, events


def context(controller, db, parent):
    return controller.context(db, parent, 0, controller.attempts + 1)


def begin(controller, ctx):
    snapshot = Snapshot(ctx['island'], 'diff', 'fixture', ctx['iteration'] - 1,
        ctx['parent_id'], ctx['parent_code'], ctx['parent_score'],
        research_context=ctx, kb_lessons=ctx['lesson_texts'])
    return controller.begin(snapshot, 0, 'diff', build('diff', snapshot))


def cards_in(prompt):
    result = []
    for suffix in prompt.split('RESEARCH_LESSON_JSON:\n')[1:]:
        raw, _ = json.JSONDecoder().raw_decode(suffix)
        result.append(decode_evidence(raw))
    return result


def repeated_sources():
    kb = KnowledgeBase()
    first, second = record(kb, 1), record(kb, 2)
    a = add(kb, 'packing: spread the circle centers', [first])
    b = add(kb, 'packing: distribute circle positions more widely', [first])
    c = add(kb, 'packing: preserve radii during refinement', [second])
    return kb, (a, b, c), (first, second)


def legacy_admission(controller, db, parent):
    """Independent pre-K07 ranking/admission oracle, including whole-card caps."""
    from sciel.engine.research import MAX_CARD_CHARS, MAX_MEMORY_CHARS
    candidates = [l for l in controller.knowledge.lessons
                  if l['scope'] == 'global' or l['branch'] == parent.id]
    # Fixtures have exactly one shared query word and no matching identifiers;
    # confidence and stable IDs therefore resolve the original lexical ties.
    candidates.sort(key=lambda l: (-l['confidence'], l['id']))
    chosen, texts, shown, omitted = [], [], [], []
    used = 0
    for lesson in candidates:
        if not controller.config.get('inject_lessons', True):
            omitted.append(dict(id=lesson['id'], reason='ablation',
                                chars=len(controller.lesson_text(lesson))))
            continue
        if len(chosen) >= controller.config['retrieve_k']:
            break
        text = controller.lesson_text(lesson)
        reason = ('card_char_limit' if len(text) > MAX_CARD_CHARS else
                  'memory_char_limit' if used + len(text) > MAX_MEMORY_CHARS else None)
        if reason:
            omitted.append(dict(id=lesson['id'], reason=reason, chars=len(text)))
            continue
        chosen.append(lesson)
        texts.append(text)
        shown.append(dict(id=lesson['id'], sha256=digest(text), chars=len(text)))
        used += len(text)
    return chosen, texts, dict(schema='sciel-memory-presentation/5', presented=shown,
        omitted=omitted, rendered_chars=used, ranked_count=len(candidates),
        unshown_count=len(candidates) - len(chosen), max_card_chars=MAX_CARD_CHARS,
        max_memory_chars=MAX_MEMORY_CHARS)


@pytest.mark.parametrize('inject,k', [(True, 0), (True, 1), (True, 3), (False, 3)])
def test_legacy_default_matches_independent_pre_k07_card_admission(inject, k):
    kb, _, _ = repeated_sources()
    controller, db, parent, events = fixture(kb, inject_lessons=inject, retrieve_k=k)
    before = controller.state()
    rng = random.getstate()
    expected_lessons, expected_texts, expected_presentation = legacy_admission(controller, db, parent)
    actual = context(controller, db, parent)
    assert actual['lessons'] == expected_lessons
    assert actual['lesson_texts'] == expected_texts
    assert actual['lesson_presentation'] == expected_presentation
    assert actual['injected_ids'] == [l['id'] for l in expected_lessons]
    assert controller.state() == before and random.getstate() == rng and events == []


DIVERSE = {'mode': 'source_diverse', 'diversity_weight': .5}


def selected(kb, *, k=2, **settings):
    controller, db, parent, events = fixture(kb, retrieve_k=k,
        lesson_selection=DIVERSE, **settings)
    return context(controller, db, parent), controller, db, parent, events


def decision(ctx, ident):
    return next(d for d in ctx['lesson_presentation']['selection']['decisions']
                if d['id'] == ident)


def test_paraphrases_share_support_but_independent_observation_gets_second_slot():
    kb, (a, b, c), (e1, e2) = repeated_sources()
    before = kb.state()
    ctx, controller, db, parent, events = selected(kb)
    assert ctx['injected_ids'] == [a, c]
    assert kb.state() == before and events == []
    assert [l['id'] for l in kb.lessons] == [a, b, c]  # No semantic merge/deletion.
    trace = ctx['lesson_presentation']['selection']
    assert trace['schema'] == 'sciel-lesson-selection/1'
    assert trace['mode'] == 'source_diverse' and trace['diversity_weight'] == .5
    assert trace['base_candidates'] == [
        dict(id=a, base_rank=0, base_utility=1., source_support_ids=[e1]),
        dict(id=b, base_rank=1, base_utility=.5, source_support_ids=[e1]),
        dict(id=c, base_rank=2, base_utility=1 / 3, source_support_ids=[e2])]
    assert decision(ctx, a)['against_ids'] == []
    assert decision(ctx, c)['against_ids'] == [a]
    assert decision(ctx, c)['redundancy'] == decision(ctx, c)['penalty'] == 0
    assert trace['unshown_ids'] == trace['remaining_ids'] == [b]
    assert context(controller, db, parent) == ctx


def test_soft_penalty_preserves_all_same_source_cards_when_more_slots_available():
    kb = KnowledgeBase()
    eid = record(kb, 1)
    ids = [add(kb, f'packing observation wording {i}', [eid]) for i in range(3)]
    ctx, *_ = selected(kb, k=3)
    assert ctx['injected_ids'] == ids
    last = decision(ctx, ids[-1])
    assert last['against_ids'] == ids[:-1]
    assert last['redundancy'] == 1 and last['penalty'] == .5
    assert last['score'] == pytest.approx(1 / 3 - .5)
    assert last['admitted'] is True and last['reason'] is None


def test_partial_support_overlap_is_max_jaccard_not_an_exposure_or_union_count():
    kb = KnowledgeBase()
    e1, e2, e3 = [record(kb, n) for n in (1, 2, 3)]
    a = add(kb, 'packing first observation', [e1, e2])
    b = add(kb, 'packing partly shared observation', [e2, e3])
    c = add(kb, 'packing fully shared observation', [e1, e2])
    ctx, *_ = selected(kb, k=3)
    assert ctx['injected_ids'] == [a, b, c]
    assert decision(ctx, b)['redundancy'] == pytest.approx(1 / 3)
    assert decision(ctx, b)['penalty'] == pytest.approx(1 / 6)
    assert decision(ctx, c)['redundancy'] == 1


def test_opposite_conditions_negations_and_conflicting_measurements_are_not_merged():
    kb = KnowledgeBase()
    positive, negative = record(kb, 1), record(kb, 2, delta=-.125)
    rules = ['packing: use x < 0', 'packing: use x > 0',
             'packing: do not use x < 0', "packing: match 'a  b'"]
    ids = [add(kb, rules[0], [positive]), add(kb, rules[1], [positive]),
           add(kb, rules[2], [negative], polarity='negative'),
           add(kb, rules[3], [positive])]
    before = kb.state()
    ctx, *_ = selected(kb, k=4)
    assert ctx['injected_ids'][:2] == [ids[0], ids[2]]
    assert set(ctx['injected_ids']) == set(ids)
    assert kb.state() == before
    cards = [decode_evidence(json.loads(t.split('RESEARCH_LESSON_JSON:\n', 1)[1]))
             for t in ctx['lesson_texts']]
    assert {c['unverified_interpretation'] for c in cards} == set(rules)
    warning = next(c for c in cards if c['lesson_id'] == ids[2])
    assert warning['source_ids'] == [negative]
    assert warning['source_counts']['negative'] == 1
    assert warning['source_observations'][0]['actual_delta'] == -.125
    assert warning['advice_exposures']['retained_unique_counts']['total'] == 0


def test_oversized_first_card_never_penalizes_its_fitting_source_sibling(monkeypatch):
    import sciel.engine.research as module
    kb = KnowledgeBase()
    e1, e2 = record(kb, 1), record(kb, 2)
    large = add(kb, 'packing ' + 'long ' * 300, [e1])
    sibling = add(kb, 'packing fitting paraphrase', [e1])
    independent = add(kb, 'packing distinct source', [e2])
    texts = {l['id']: ResearchController.lesson_text(l) for l in kb.lessons}
    monkeypatch.setattr(module, 'MAX_CARD_CHARS', len(texts[large]) - 1)
    ctx, *_ = selected(kb, k=1)
    assert ctx['injected_ids'] == [sibling]
    rejected, accepted = ctx['lesson_presentation']['selection']['decisions']
    assert rejected['id'] == large and rejected['admitted'] is False
    assert rejected['reason'] == 'card_char_limit'
    assert accepted['id'] == sibling and accepted['against_ids'] == []
    assert accepted['redundancy'] == 0 and accepted['score'] == .5
    assert ctx['lesson_presentation']['selection']['unshown_ids'] == [large, independent]


def test_memory_overflow_omission_cannot_penalize_later_fitting_card(monkeypatch):
    import sciel.engine.research as module
    kb = KnowledgeBase()
    e1, e2 = record(kb, 1), record(kb, 2)
    a = add(kb, 'packing first small', [e1])
    large = add(kb, 'packing ' + 'oversized ' * 160, [e2])
    fitting = add(kb, 'packing fitting independent', [e2])
    duplicate = add(kb, 'packing small repeated source', [e1])
    texts = {l['id']: ResearchController.lesson_text(l) for l in kb.lessons}
    monkeypatch.setattr(module, 'MAX_CARD_CHARS', max(map(len, texts.values())) + 1)
    monkeypatch.setattr(module, 'MAX_MEMORY_CHARS', len(texts[a]) + len(texts[fitting]))
    ctx, *_ = selected(kb)
    assert ctx['injected_ids'] == [a, fitting]
    assert decision(ctx, large)['reason'] == 'memory_char_limit'
    assert decision(ctx, large)['admitted'] is False
    admitted = decision(ctx, fitting)
    assert admitted['against_ids'] == [a] and admitted['redundancy'] == 0
    assert ctx['lesson_presentation']['rendered_chars'] == len(texts[a]) + len(texts[fitting])
    assert duplicate in ctx['lesson_presentation']['selection']['remaining_ids']


@pytest.mark.parametrize('inherit', [False, True])
def test_shared_direct_or_inherited_exposures_do_not_become_source_redundancy(inherit):
    kb, (a, b, c), (_, e2) = repeated_sources()
    shared = record(kb, 3, delta=0., injected=[a, c])
    if inherit:
        c, = kb.reflect([dict(rule='packing replacement for the independent observation',
            polarity='positive', scope='branch', evidence_ids=[e2])], [c],
            allowed_evidence_ids=[e2])['lesson_ids']
    record(kb, 4, delta=None, injected=[a, c])
    before = kb.state()
    ctx, *_ = selected(kb)
    assert ctx['injected_ids'] == [a, c]
    assert decision(ctx, c)['redundancy'] == 0
    assert kb.state() == before
    candidate = next(d for d in ctx['lesson_presentation']['selection']['base_candidates']
                     if d['id'] == c)
    assert candidate['source_support_ids'] == [e2] and shared not in candidate['source_support_ids']
    card = next(ca for ca in [decode_evidence(json.loads(t.split('RESEARCH_LESSON_JSON:\n', 1)[1]))
                             for t in ctx['lesson_texts']] if ca['lesson_id'] == c)
    assert card['advice_exposures']['retained_unique_counts'] == {
        'total': 2, 'positive': 0, 'negative': 0, 'zero': 1, 'unmeasured': 1}
    assert card['ranking_statistic']['value'] == .5
    assert card['advice_exposures']['inherited_unique_counts']['total'] == int(inherit)


def test_branch_filtering_precedes_diversity_and_global_evidence_remains_bound():
    kb = KnowledgeBase()
    e1, e2 = record(kb, 1), record(kb, 2, branch='other')
    foreign = add(kb, 'packing foreign branch observation', [e2])
    local = add(kb, 'packing local observation', [e1])
    glob = add(kb, 'packing evidence spanning branches', [e1, e2], scope='global')
    ctx, controller, db, parent, _ = selected(kb, k=3)
    assert ctx['injected_ids'] == [local, glob]
    trace = ctx['lesson_presentation']['selection']
    assert [x['id'] for x in trace['base_candidates']] == [local, glob]
    assert foreign not in trace['unshown_ids']
    assert next(l for l in ctx['lessons'] if l['id'] == glob)['source_support_ids'] == [e1, e2]
    # Exact-code migrants follow original branch, not their new ID or island.
    migrant = Program('migrated', parent.code, parent.score, {}, 1, 'migrant', parent_id=parent.id)
    db._all[migrant.id] = migrant
    migrated = controller.context(db, migrant, 1, ctx['iteration'])
    assert migrated['branch'] == 'p1' and migrated['injected_ids'] == [local, glob]


def test_actual_prompt_contains_only_whole_selected_cards_and_only_they_gain_exposure():
    kb, (a, b, c), _ = repeated_sources()
    ctx, controller, db, parent, events = selected(kb)
    ident, prompt = begin(controller, ctx)
    cards = cards_in(prompt)
    assert [card['lesson_id'] for card in cards] == [a, c]
    assert len(cards) == len(ctx['lesson_texts']) == 2
    assert next(l['rule'] for l in kb.lessons if l['id'] == b) not in prompt
    for text, shown in zip(ctx['lesson_texts'], ctx['lesson_presentation']['presented']):
        assert prompt.count(text) == 1
        assert shown['sha256'] == digest(text) and shown['chars'] == len(text)
    assert events[0]['lesson_presentation'] == ctx['lesson_presentation']
    assert events[0]['evidence']['injected_ids'] == [a, c]
    assert events[0]['prompt_sha256'] == digest(prompt)
    controller.resolve(ident, status='generation_stopped', failure='inert fixture')
    for lesson in kb.lessons:
        assert lesson['exposure_ids'] == ([ident] if lesson['id'] in (a, c) else [])
        assert lesson['support_ids'] == lesson['refute_ids'] == []
    assert next(l for l in kb.lessons if l['id'] == b)['source_support_ids']
    assert controller.aux_calls == 0 and controller.aux_cost == 0


def test_explicit_ranked_matches_missing_config_context_prompt_and_no_selection_identity():
    kb, _, _ = repeated_sources()
    legacy, db, parent, events = fixture(kb)
    ranked, db2, parent2, events2 = fixture(kb,
        lesson_selection={'mode': 'ranked', 'diversity_weight': .9})
    a, b = context(legacy, db, parent), context(ranked, db2, parent2)
    assert a == b
    assert begin(legacy, a) == begin(ranked, b)
    assert events == events2
    assert 'lesson_selection' not in legacy.summary() and 'lesson_selection' not in ranked.summary()


def test_zero_weight_has_identical_cards_and_prompt_with_only_additive_selection_trace():
    kb, ids, _ = repeated_sources()
    legacy, db, parent, _ = fixture(kb, retrieve_k=2)
    zero, db2, parent2, _ = fixture(kb, retrieve_k=2,
        lesson_selection={'mode': 'source_diverse', 'diversity_weight': 0.})
    a, b = context(legacy, db, parent), context(zero, db2, parent2)
    assert a['injected_ids'] == b['injected_ids'] == list(ids[:2])
    comparison = copy.deepcopy(b)
    trace = comparison['lesson_presentation'].pop('selection')
    assert a == comparison and all(d['penalty'] == 0 for d in trace['decisions'])
    assert begin(legacy, a) == begin(zero, b)


@pytest.mark.parametrize('k', [0, 2])
def test_ablation_keeps_exact_legacy_path_without_diversity_trace(k):
    kb, _, _ = repeated_sources()
    legacy, db, parent, _ = fixture(kb, inject_lessons=False, retrieve_k=k)
    diverse, db2, parent2, _ = fixture(kb, inject_lessons=False, retrieve_k=k,
                                    lesson_selection=DIVERSE)
    a, b = context(legacy, db, parent), context(diverse, db2, parent2)
    assert a == b and a['injected_ids'] == []
    assert 'selection' not in b['lesson_presentation']
    assert begin(legacy, a) == begin(diverse, b)


def test_zero_slots_does_not_manufacture_admitted_set_or_retrieval_side_effects():
    kb, ids, _ = repeated_sources()
    before = kb.state()
    ctx, *_ = selected(kb, k=0)
    assert ctx['lessons'] == ctx['lesson_texts'] == ctx['injected_ids'] == []
    trace = ctx['lesson_presentation']['selection']
    assert trace['decisions'] == [] and trace['remaining_ids'] == list(ids)
    assert trace['unshown_ids'] == list(ids) and kb.state() == before


def test_controller_json_resume_preserves_next_prompt_decisions_and_exposure_continuation():
    kb, _, _ = repeated_sources()
    first, db, parent, events = fixture(kb, retrieve_k=2, lesson_selection=DIVERSE)
    saved = json.loads(json.dumps(first.state(), allow_nan=False))
    restored, db2, parent2, events2 = fixture(retrieve_k=2, lesson_selection=DIVERSE)
    restored.load_state(saved)
    rng = random.getstate()
    a, b = context(first, db, parent), context(restored, db2, parent2)
    assert a == b and random.getstate() == rng
    id1, prompt1 = begin(first, a)
    id2, prompt2 = begin(restored, b)
    assert (id1, prompt1) == (id2, prompt2)
    for instance, ident in [(first, id1), (restored, id2)]:
        instance.resolve(ident, status='generation_stopped', failure='same inert continuation')
        instance.complete_step()
    assert first.state() == restored.state() and events == events2
    assert context(first, db, parent) == context(restored, db2, parent2)


@pytest.mark.parametrize('corruption', ['missing_identity', 'foreign_identity', 'weight'])
def test_resume_rejects_policy_identity_drift_atomically(corruption):
    kb, _, _ = repeated_sources()
    source, *_ = fixture(kb, lesson_selection=DIVERSE)
    saved = source.state()
    if corruption == 'missing_identity':
        saved.pop('lesson_selection')
    elif corruption == 'foreign_identity':
        saved['lesson_selection']['schema'] = 'foreign-selection/999'
    else:
        saved['config']['lesson_selection']['diversity_weight'] = .75
    target, *_ = fixture(lesson_selection=DIVERSE)
    before = target.state()
    with pytest.raises(ValueError):
        target.load_state(saved)
    assert target.state() == before


@pytest.mark.parametrize('weight', [0., 1.])
@pytest.mark.parametrize('location', ['config', 'identity'])
def test_resume_rejects_boolean_weight_even_when_python_compares_it_equal(weight, location):
    policy = dict(mode='source_diverse', diversity_weight=weight)
    source, *_ = fixture(lesson_selection=policy)
    saved = json.loads(json.dumps(source.state()))
    if location == 'config':
        saved['config']['lesson_selection']['diversity_weight'] = bool(weight)
    else:
        saved['lesson_selection']['diversity_weight'] = bool(weight)
    target, *_ = fixture(lesson_selection=policy)
    before = target.state()
    with pytest.raises(ValueError):
        target.load_state(saved)
    assert target.state() == before
