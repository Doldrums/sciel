"""Actual R scheduling rotates on attempts, not batch size or accepted output."""
import copy
import json
from types import SimpleNamespace as NS

import pytest

from sciel.engine.research import ResearchController
from sciel.engine.reflection_context import ALLOCATION_POLICY, prepare_reflection_context
from test_research_measurement_state import controller
from test_reflection_anchor_allocation import add


def append_wave(c, count=4):
    import hashlib
    for _ in range(count):
        c.attempts += 1
        c.outcomes += 1
        source = hashlib.sha256(str(c.outcomes).encode()).hexdigest()
        c.knowledge.record(dict(id=f'r{c.outcomes:06d}a1', branch='generic',
            status='evaluated', parent_score=.5, score=.6, actual_delta=.6-.5,
            source_sha256=source, proposed_sha256=source, iteration=c.outcomes,
            injected_ids=[], metric=c.metric, parent_cases=None, evaluation=None))


def scheduled(*, cap=None):
    c, _, _, events = controller(reflection_interval=10, max_aux_calls=cap,
        max_experiments=200)
    calls, charges = [], []
    def generate(prompt, model, **options):
        calls.append(prompt)
        # Alternate a typed admission failure with a malformed response. Neither
        # succeeds, yet both have consumed this scheduled reflection's turn.
        text = json.dumps(dict(lessons=[], retire_ids=['not_supplied'])) if len(calls) % 2 else 'invalid'
        return NS(text=text, cost=.01, prompt_tokens=1, completion_tokens=1,
            finish_reason='stop', refusal=None, model='fake')
    c.llm = NS(models=['fake'], max_retries=1, generate=generate)
    c.charge = charges.append
    return c, events, calls, charges


def clone(c, events, charges):
    restored = ResearchController(c.config, c.llm,
        NS(research=lambda **kw: events.append(copy.deepcopy(kw))), charges.append, metric=c.metric)
    state = c.state()
    restored.load_state(copy.deepcopy(state))
    assert restored.state() == state
    return restored


def test_interval_ten_four_job_waves_visit_every_lane_even_after_rejected_R_and_resume():
    c, events, calls, charges = scheduled()
    append_wave(c)
    source = c.knowledge.experiments[0]['id']
    ids = [add(c.knowledge, f'Unverified generic interpretation {n}', [source]) for n in range(3)]
    original_lessons = c.knowledge.lessons
    for turn in range(6):
        while c.outcomes < 12*(turn+1):
            append_wave(c)
            # This is the actual post-wave scheduler boundary, not every record.
            c.finish_conclusions()
        assert c.reflection_attempts == turn+1
        assert c.reflected_through == 12*(turn+1)
        assert c.reflections == 0 and c.knowledge.lessons == original_lessons
        c = clone(c, events, charges)
    contexts = [e for e in events if e['phase'] == 'reflection_context']
    assert [e['selection_epoch'] for e in contexts] == list(range(6))
    assert all(e['allocation_policy'] == ALLOCATION_POLICY for e in contexts)
    assert [e['lesson_ids'][0] for e in contexts[1::2]] == ids
    assert [e['lesson_ids'][:2] for e in contexts[::2]] == [[ids[0], ids[1]], [ids[0], ids[2]], [ids[1], ids[2]]]
    terminal = [e for e in events if e['phase'] == 'reflection']
    assert [e['status'] for e in terminal] == ['parsed', 'invalid']*3
    assert [e['error'] for e in terminal] == ['ValueError', None]*3
    assert all('supplied in the reflection context' in e['error_detail'] for e in terminal[::2])
    assert all(e['applied'] is None and e['changed'] is False for e in terminal)
    assert len(calls) == 6 and charges == [.01]*6 and c.aux_calls == 6
    assert c.aux_cost == pytest.approx(.06)


def test_budget_skipped_R_still_advances_and_resumes_without_dispatch():
    c, events, calls, charges = scheduled(cap=0)
    for _ in range(6):
        append_wave(c)
        c.finish_conclusions()
    assert c.reflection_attempts == 2 and c.reflected_through == 24
    assert c.reflections == 0 and c.aux_calls == 0 and calls == charges == []
    assert [e['selection_epoch'] for e in events if e['phase'] == 'reflection_context'] == [0, 1]
    clone(c, events, charges)


@pytest.mark.parametrize('bad', [True, -1, 1., '0'])
def test_noncanonical_explicit_epoch_rejected_before_preparation(bad):
    c, _, _, _ = scheduled()
    with pytest.raises(ValueError, match='epoch'):
        c.knowledge.reflection_context(10, selection_epoch=bad)
    with pytest.raises(ValueError, match='epoch'):
        prepare_reflection_context(c.knowledge, {}, config=c.config,
            interval=10, through=1, selection_epoch=bad)


@pytest.mark.parametrize('patch', [
    {'reflection_attempts': True}, {'reflection_attempts': -1},
    {'reflection_attempts': 1}, {'reflection_allocation': 'unknown'},
])
def test_counter_and_policy_tampering_reject_atomically(patch):
    c, _, _, _ = scheduled()
    before = c.state()
    state = copy.deepcopy(before)
    state.update(patch)
    with pytest.raises(ValueError, match='(counter|accounting|allocation)'):
        c.load_state(state)
    assert c.state() == before


def test_pre_policy_pristine_state_only_can_use_explicit_zero_fallback():
    c, events, calls, charges = scheduled()
    pristine = c.state()
    del pristine['reflection_allocation'], pristine['reflection_attempts']
    c.load_state(pristine)
    assert c.reflection_attempts == 0 and c.state()['reflection_allocation'] == ALLOCATION_POLICY
    for _ in range(3):
        append_wave(c)
        c.finish_conclusions()
    state = c.state()
    del state['reflection_allocation'], state['reflection_attempts']
    before = c.state()
    with pytest.raises(ValueError, match='missing reflection allocation history'):
        c.load_state(state)
    assert c.state() == before


def test_attempt_count_cannot_exceed_cadence_or_exist_when_reflection_disabled():
    c, _, _, _ = scheduled()
    for _ in range(3):
        append_wave(c)
        c.finish_conclusions()
    state = c.state()
    state['reflection_attempts'] = 12
    before = c.state()
    with pytest.raises(ValueError, match='accounting'):
        c.load_state(state)
    assert c.state() == before
    disabled, _, _, _ = controller(reflection_interval=0)
    append_wave(disabled, 1)
    state = disabled.state()
    state.update(reflection_attempts=1, reflected_through=1)
    with pytest.raises(ValueError, match='accounting'):
        disabled.load_state(state)


def test_current_reflection_policy_reconstructs_through_historical_reducer():
    from sciel.engine.external_research_v2 import _reflection
    from sciel.knowledge_base import KnowledgeBase
    from test_comparison_review_flow import fixture
    f = fixture()
    before = f.c.knowledge.state()
    conclusions = copy.deepcopy(f.c.conclusions)
    f.c.complete_step()
    rows = [dict(event='research', event_seq=i+1, **e) for i, e in enumerate(f.events)]
    terminal = next(e for e in reversed(rows) if e['phase'] == 'reflection')
    def replay(replay_rows):
        kb = KnowledgeBase(max_lessons=f.c.knowledge.max_lessons,
            max_experiments=f.c.knowledge.max_experiments, max_fragments=f.c.knowledge.max_fragments)
        kb.load_state(before)
        _reflection(kb, {}, replay_rows, terminal, interval=1,
            config=f.c.config, conclusions=conclusions)
        assert kb.state() == f.c.knowledge.state()
    replay(rows)
    for patch in ({'allocation_policy': 'unknown'}, {'selection_epoch': True},
                  {'selection_epoch': -1}, {'selection_epoch': 1}):
        changed = copy.deepcopy(rows)
        context = next(e for e in changed if e['phase'] == 'reflection_context')
        context.update(patch)
        with pytest.raises(ValueError, match='allocation policy or epoch'):
            replay(changed)
    for missing in ('allocation_policy', 'selection_epoch'):
        changed = copy.deepcopy(rows)
        del next(e for e in changed if e['phase'] == 'reflection_context')[missing]
        with pytest.raises(ValueError, match='allocation policy or epoch'):
            replay(changed)
