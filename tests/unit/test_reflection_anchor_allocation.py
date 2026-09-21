"""Evidence-complete R allocation under real serializer and finite budgets."""
import copy
import hashlib

import pytest

from sciel.hypothesis import rendering
from sciel.knowledge_base import KnowledgeBase


def observe(kb, number, *, branch='generic', base=.5, delta=.1, injected=(), pad=0):
    source = hashlib.sha256(f'{branch}-{number}'.encode()).hexdigest()
    score = base + delta if delta is not None else None
    row = dict(id=f'r{number:06d}a1', branch=branch,
        status='evaluated' if score is not None else 'eval_failed', parent_score=base,
        score=score, actual_delta=score-base if score is not None else None,
        source_sha256=source if score is not None else None,
        proposed_sha256=source if score is not None else None,
        iteration=number, injected_ids=list(injected))
    if pad:
        row.update(change_excerpt='x'*6000, hypothesis=dict(text='Recent candidate',
            reasoning='r'*pad, predicted_low=0., predicted_high=.2, confidence=.5))
    kb.record(row)
    return row['id']


def add(kb, rule, ids):
    return kb.add_lessons([dict(rule=rule, polarity='positive', scope='branch', evidence_ids=ids)],
        allowed_evidence_ids=ids)[0]


def assert_context(kb, records, summaries, *, max_chars=90000, max_records=64, max_lessons=8):
    assert len(rendering.serialize_reflection(records, summaries)) <= max_chars
    assert len(records) <= max_records and len(summaries) <= max_lessons
    ids = {r['id'] for r in records}
    for lesson in summaries:
        assert set(lesson['evidence_ids']) <= ids
        assert set(lesson['observed_exposures']['shown_ids']) <= ids
    restored = KnowledgeBase(max_lessons=kb.max_lessons, max_experiments=kb.max_experiments,
        max_fragments=kb.max_fragments)
    restored.load_state(copy.deepcopy(kb.state()))
    assert restored.reflection_context(10, max_chars=max_chars, max_experiments=max_records,
        max_lessons=max_lessons) == (records, summaries)


@pytest.mark.parametrize('branch,base', [('routing', 120.), ('symbolic_error', -.7), ('compiler_size', -4096.)])
def test_large_recent_history_cannot_starve_feasible_pair_on_its_anchor_turn(branch, base):
    kb = KnowledgeBase()
    source = observe(kb, 1, branch=branch, base=base)
    pair = [add(kb, f'Unverified interpretation {i}', [source]) for i in range(2)]
    negative = observe(kb, 2, branch=branch, base=base, delta=-.1, injected=pair)
    for number in range(3, 21):
        observe(kb, number, branch=branch, base=base, pad=1500)
    before = kb.state()
    records, lessons = kb.reflection_context(10)
    assert [l['id'] for l in lessons if l['focus_for_consolidation']] == pair
    assert all(l['observed_exposures']['negative'] == 1 for l in lessons)
    assert {source, negative} <= {r['id'] for r in records}
    assert len({r['id'] for r in records} & {e['id'] for e in kb.experiments[-10:]}) >= 8
    assert_context(kb, records, lessons)
    assert kb.state() == before


def test_tiny_always_admissible_lesson_does_not_starve_larger_units():
    kb = KnowledgeBase()
    cheap_source = observe(kb, 0)
    source = observe(kb, 1)
    pair = [add(kb, f'An older interpretation {i}', [source]) for i in range(2)]
    observe(kb, 2, delta=-.1, injected=pair)
    for number in range(3, 13):
        observe(kb, number, pad=1250)
    cheap = add(kb, 'Cheap interpretation', [cheap_source])
    anchors, focused = set(), set()
    for step in range(8):
        if step:
            for number in range(3+10*step, 13+10*step):
                observe(kb, number, pad=1250)
        before = kb.state()
        records, lessons = kb.reflection_context(10)
        assert lessons
        anchors.add(lessons[0]['id'])
        focused.update(l['id'] for l in lessons if l['focus_for_consolidation'])
        assert_context(kb, records, lessons)
        assert kb.state() == before
    assert anchors == {*pair, cheap}
    assert focused == set(pair)


def test_anchor_keeps_inherited_negative_and_direct_unmeasured_representatives():
    kb = KnowledgeBase()
    source = observe(kb, 1)
    old = add(kb, 'Previous wording', [source])
    negative = observe(kb, 2, delta=-.1, injected=[old])
    replacement = kb.reflect([dict(rule='Replacement wording', polarity='positive', scope='branch',
        evidence_ids=[source])], [old], allowed_evidence_ids=[source])['lesson_ids'][0]
    partner = add(kb, 'Alternative wording', [source])
    failed = observe(kb, 3, delta=None, injected=[replacement, partner])
    for number in range(4, 21):
        observe(kb, number, pad=1500)
    before = kb.state()
    records, lessons = kb.reflection_context(10)
    assert {source, negative, failed} <= {r['id'] for r in records}
    inherited = next(l for l in lessons if l['id'] == replacement)['observed_exposures']
    assert inherited['negative'] == inherited['unmeasured'] == 1
    attribution = next(x for x in inherited['representatives'] if x['experiment_id'] == negative)['attribution']
    assert attribution == [dict(kind='inherited', from_lesson_id=old, from_rule='Previous wording')]
    assert_context(kb, records, lessons)
    assert kb.state() == before


def test_infeasible_units_leave_recent_evidence_and_do_not_partially_drop_sources():
    kb = KnowledgeBase()
    sources = [observe(kb, n) for n in range(1, 4)]
    add(kb, 'Requires all three source observations', sources)
    observe(kb, 4)
    saved = kb.state()
    records, lessons = kb.reflection_context(10, max_experiments=1)
    assert lessons == [] and [r['id'] for r in records] == ['r000004a1']
    assert_context(kb, records, lessons, max_records=1)
    assert kb.state() == saved


def test_record_and_lesson_limits_allow_complete_singleton_without_partial_pair_flags():
    kb = KnowledgeBase()
    source = observe(kb, 1)
    for rule in ('Left interpretation', 'Right interpretation'):
        add(kb, rule, [source])
    for number in range(2, 21):
        observe(kb, number)
    records, lessons = kb.reflection_context(10, max_experiments=1, max_lessons=1)
    assert len(lessons) == 1 and not lessons[0]['focus_for_consolidation']
    assert [r['id'] for r in records] == [source]
    assert_context(kb, records, lessons, max_records=1, max_lessons=1)


def test_pair_lookup_and_cached_units_do_not_serialize_quadratic_pair_inventory(monkeypatch):
    kb = KnowledgeBase(max_lessons=80)
    source = observe(kb, 1)
    for number in range(60):
        add(kb, f'Interpretation {number} '+('w'*3000), [source])
    observe(kb, 2)
    calls = []
    real = rendering.serialize_reflection
    def counted(*args):
        calls.append(1)
        return real(*args)
    monkeypatch.setattr(rendering, 'serialize_reflection', counted)
    records, lessons = kb.reflection_context(1, max_chars=7000)
    assert len(lessons) == 1 and not lessons[0]['focus_for_consolidation']
    assert len(calls) < 3 * len(kb.lessons) + 10
    assert len(real(records, lessons)) <= 7000


def test_even_empty_context_must_fit_the_declared_serialized_budget():
    with pytest.raises(ValueError, match='serialization overhead'):
        KnowledgeBase().reflection_context(10, max_chars=1)
