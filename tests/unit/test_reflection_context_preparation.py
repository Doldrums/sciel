"""Pure preparation preserves the saved wire and the live scheduler projection."""
import copy
import hashlib

import pytest

from sciel.engine.reflection_context import prepare_reflection_context
from sciel.knowledge_base import KnowledgeBase
from test_comparison_review_flow import fixture
from test_external_history_v2_actual import actual_history


def test_preparation_is_pure_and_matches_live_scheduler():
    f = fixture()
    before = copy.deepcopy(f.c.state())
    events = copy.deepcopy(f.events)
    prepared = prepare_reflection_context(f.c.knowledge, f.c.conclusions,
        config=f.c.config, interval=1, through=f.c.outcomes,
        selection_epoch=f.c.reflection_attempts)
    assert f.c.state() == before and f.events == events
    assert len(f.calls) == 1  # The fixture's earlier ordinary conclusion only.
    f.c.complete_step()
    assert f.calls[-1]['prompt'] == prepared['prompt']
    assert f.calls[-1]['options']['response_format'] == prepared['response_format']
    context = next(e for e in reversed(f.events) if e['phase'] == 'reflection_context')
    assert {k: context[k] for k in prepared['context_event']} == prepared['context_event']
    prepared['records'].clear()
    prepared['comparison_review_groups'][0]['branch'] = 'mutated'
    assert f.c.knowledge.comparisons[0]['branch'] == 'p1'


@pytest.mark.parametrize('enabled,template_sha,wire_sha', [
    (False, 'a974a198863503133cfff6c0b108a3cb559a5fc575a0a73b4973bc6fb359497e',
        'e2d06308d0c4bef0348035f5bc6d13a30b9f6954d3e5dbe90ac55f5705347e16'),
    (True, '856ef8cd14dd0e5529da6c5b53e622fb6cafb28ae71cd51ccb0b3712b77ccb45',
        '83a1737bd7822c1b4b441a821df3d387e8cd6f86eaad52ff41be37eb2524e334'),
])
def test_authored_prompt_matches_explicit_evidence_and_pinned_wire(
        actual_history, enabled, template_sha, wire_sha):
    from sciel.hypothesis.auxiliary_schema import response_format
    from sciel.hypothesis.conclusion_feedback import reflection_review_context
    from sciel.hypothesis.measurement import observation_view
    from sciel.hypothesis.prompts import reflection_prompt
    from sciel.knowledge_base.catalogs import canonical_sha
    from sciel.knowledge_base.presentation import comparison_card
    from sciel.knowledge_base.restart_prediction_context import build_context as prediction_context

    _, final = actual_history
    cfg = {**final['config'], 'comparison_reviews': enabled}
    kb = KnowledgeBase(**final['knowledge']['limits'])
    kb.load_state(final['knowledge'])
    before = kb.state()
    # Pin the existing empty instruction/response contracts independently of
    # authored observations. These golden hashes predate this fixture rewrite.
    options = dict(structured=True, comparison_reviews=enabled,
        comparison_review_groups=[] if enabled else None)
    template = reflection_prompt([], [], **options)
    assert hashlib.sha256(template.encode()).hexdigest() == template_sha
    wire = response_format('reflection', comparison_reviews=enabled)
    assert canonical_sha(wire) == wire_sha

    # Interval one selects the latest ordinary observation and whole paired
    # group. Construct those inputs explicitly, without using preparation's
    # selector or any fields returned by the function under test.
    latest, group = kb.experiments[-1], kb.comparisons[-1]
    assert latest['id'] == 'r000002a1' and not kb.lessons
    assert 'proposal_group' not in latest
    ordinary = observation_view({key: copy.deepcopy(value) for key, value in latest.items()
        if key not in ('code_fragment', 'code_fragment_sha256')})
    records = [ordinary, comparison_card(group)]
    ordinary_records = {record['id']: record for record in kb.experiments}
    review = reflection_review_context(final['conclusions'],
        ordinary_records, [latest['id']],
        max_chars=32000, batch_only_lessons=True)
    assert review is not None
    predictions = prediction_context([group], ordinary_records, max_chars=6000)
    expected = reflection_prompt(records, [], review_context=review, prediction_context=predictions,
        structured=True, comparison_reviews=enabled,
        comparison_review_groups=[group] if enabled else None)
    prepared = prepare_reflection_context(kb, final['conclusions'], config=cfg, interval=1, through=2)
    assert prepared['records'] == records and prepared['lessons'] == []
    assert prepared['prompt'].encode('utf-8') == expected.encode('utf-8')
    assert prepared['response_format'] == wire
    assert prepared['context_event']['experiment_ids'] == [latest['id'], group['id']]
    assert kb.state() == before


@pytest.mark.parametrize('interval,through', [(True, 1), (1, True), (0, 1), (1, 0), (1., 2)])
def test_preparation_rejects_noncanonical_schedule_coordinates(interval, through):
    f = fixture()
    with pytest.raises(ValueError):
        prepare_reflection_context(f.c.knowledge, f.c.conclusions,
            config=f.c.config, interval=interval, through=through)
