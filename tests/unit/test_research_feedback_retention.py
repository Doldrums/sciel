"""Bounded research memory remains valid across retries; no code/API execution."""
import copy

import pytest

from sciel.engine.prompt.build import build
from sciel.engine.types import Result, Snapshot
from sciel.engine.worker import run_worker
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult

from test_conclusion_feedback_flow import review_for
from test_research_measurement_state import controller, lesson, observe
from test_research_retry_feedback import Model, context, text


def full_window(cap):
    research, parent, db, events = controller(max_experiments=cap)
    first, _ = observe(research, parent, db, .6)
    lesson_id = lesson(research, first, 'An inert retained measurement.')
    research.conclusions[first['id']] = review_for(first)
    for number in range(2, cap + 1):
        row = copy.deepcopy(first)
        row['id'] = f'r{number:06d}a1'
        research.knowledge.record(row)
    research.attempts = research.outcomes = cap
    ctx = research.context(db, parent, 0, cap + 1)
    snapshot = Snapshot(0, 'full_rewrite', 'fake', cap, parent.id,
        parent.code, parent.score, research_context=ctx,
        kb_lessons=ctx['lesson_texts'], required_entrypoints=['run_search'])
    return research, parent, db, events, snapshot, first['id'], lesson_id


@pytest.mark.parametrize('cap', [2, 200])
def test_retry_keeps_bound_feedback_after_evidence_eviction_and_can_resume(cap):
    research, parent, db, events, snapshot, first_id, lesson_id = full_window(cap)
    original = copy.deepcopy(snapshot)
    lessons_before = research.knowledge.lessons
    model = Model([text('def run_search(:\n    return 0.4\n'), text()])
    evaluated = []

    def evaluate(source):
        evaluated.append(source)
        return CascadeOutcome(EvalResult(True, .6, {}), .6, 'full')

    result = run_worker('full_rewrite', snapshot, model, evaluate,
                        dup_cap=2, research=research)
    assert isinstance(result, Result) and result.ok
    assert result.cost == pytest.approx(.02)
    assert len(model.calls) == 2 and len(evaluated) == 1
    first_prompt, retry_prompt = [context(call[0]) for call in model.calls]
    assert first_prompt['conclusion_feedback'] == retry_prompt['conclusion_feedback']
    assert retry_prompt['conclusion_feedback']['cards'][0]['experiment_id'] == first_id
    assert snapshot == original
    assert first_id not in {row['id'] for row in research.knowledge.experiments}
    assert first_id not in research.conclusions
    assert len(research.knowledge.experiments) == cap
    # Review retirement must not delete lesson support or misattribute its use.
    retained, = research.knowledge.lessons
    assert retained['id'] == lesson_id
    assert retained['evidence_snapshots'] == lessons_before[0]['evidence_snapshots']
    new_outcomes = [row['evidence'] for row in events if row['phase'] == 'outcome'][-2:]
    assert all(row['injected_ids'] == [lesson_id] for row in new_outcomes)
    research.complete_step()
    assert research._feedback_delivery is None
    saved = research.state()
    restored, _, _, _ = controller(max_experiments=cap)
    restored.load_state(saved)
    assert restored.state() == saved
    assert first_id not in restored.conclusions
    assert 'conclusion_feedback' not in restored.context(db, parent, 0, cap + 3)


@pytest.mark.parametrize('damage', ['text', 'record_hash', 'unknown_id', 'branch'])
def test_retry_does_not_accept_changed_feedback_after_retirement(damage):
    research, _, _, _, snapshot, _, _ = full_window(2)
    ident, _ = research.begin(snapshot, 0, 'full_rewrite', build('full_rewrite', snapshot))
    research.resolve(ident, status='no_child')
    card = snapshot.research_context['conclusion_feedback']['cards'][0]
    if damage == 'text':
        card['unverified_interpretation']['explanation'] = 'Fabricated replacement.'
    elif damage == 'record_hash':
        card['record_sha256'] = '0' * 64
    elif damage == 'unknown_id':
        card['experiment_id'] = 'invented'
    else:
        snapshot.research_context['branch'] = 'other'
    attempts = research.attempts
    with pytest.raises(ValueError, match='conclusion feedback does not match'):
        research.begin(snapshot, 1, 'full_rewrite', build('full_rewrite', snapshot))
    assert research.attempts == attempts
    assert research.pending == {}


def test_fresh_snapshot_cannot_reuse_retired_feedback():
    research, _, _, _, snapshot, _, _ = full_window(2)
    ident, _ = research.begin(snapshot, 0, 'full_rewrite', build('full_rewrite', snapshot))
    research.resolve(ident, status='no_child')
    fresh = copy.deepcopy(snapshot)
    with pytest.raises(ValueError, match='conclusion feedback does not match'):
        research.begin(fresh, 0, 'full_rewrite', build('full_rewrite', fresh))
