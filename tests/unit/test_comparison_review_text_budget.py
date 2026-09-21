"""Dispatch cannot spend space reserved for withheld comparison interpretation.

Fixture responses and observations are inert; no provider or evaluator runs.
"""
import copy

import pytest

from sciel.engine import research
from sciel.hypothesis.prompts import initializer_block, restart_block
from sciel.knowledge_base.catalogs import canonical_json
from sciel.knowledge_base.restart_prediction_context import block as prediction_block
from test_comparison_review_flow import fixture
from test_research_measurement_state import begin_fixture


def memory_counts(context):
    actual = (sum(map(len, context['lesson_texts']))
        + sum(len(canonical_json(context[key])) for key in (
            'paired_comparisons', 'comparison_facts', 'proposal_groups',
            'conclusion_feedback', 'comparison_reviews') if key in context)
        + len(initializer_block(context.get('initializer_evidence')))
        + len(restart_block(context.get('restart_observations')))
        + len(prediction_block(context.get('restart_predictions'))))
    reviews = context['comparison_reviews']
    return actual, actual - reviews['rendered_chars'] + reviews['reserved_chars']


def test_shorter_off_text_cannot_pass_using_only_actual_character_budget(monkeypatch):
    f = fixture(reflect=True, comparison_review_text=False)
    context = f.c.context(f.db, f.parent, 0, 2)
    actual, allocated = memory_counts(context)
    assert actual < allocated
    before, events = copy.deepcopy(f.c.state()), copy.deepcopy(f.events)
    # Recheck an already prepared snapshot against a dispatch budget that fits
    # its actual text but not the promised full ON allocation.
    monkeypatch.setattr(research, 'MAX_MEMORY_CHARS', actual)
    with pytest.raises(ValueError, match='shared memory character limit'):
        begin_fixture(f.c, context)
    assert f.c.state() == before and f.events == events


@pytest.mark.parametrize('with_review', [False, True])
@pytest.mark.parametrize('spare', [-1, 0])
def test_dispatch_boundary_is_equal_for_on_off_and_empty_channels(monkeypatch, with_review, spare):
    arms = [fixture(reflect=with_review, comparison_review_text=mode) for mode in (True, False)]
    contexts = [f.c.context(f.db, f.parent, 0, 2) for f in arms]
    counted = [memory_counts(context) for context in contexts]
    assert counted[0][1] == counted[1][1]
    cap = counted[0][1] + spare
    monkeypatch.setattr(research, 'MAX_MEMORY_CHARS', cap)
    for f, context in zip(arms, contexts):
        before = f.c.attempts
        if spare < 0:
            with pytest.raises(ValueError, match='shared memory character limit'):
                begin_fixture(f.c, context)
            assert f.c.attempts == before
        else:
            begin_fixture(f.c, context)
            assert f.c.attempts == before + 1
        # The only synthetic replies were fixture C and optional fixture R;
        # begin prepares a prompt and never calls the model.
        assert len(f.calls) == 1 + int(with_review)
