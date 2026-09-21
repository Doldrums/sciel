"""The text treatment changes interpretation only, including at real memory pressure."""
import copy
from types import SimpleNamespace as NS

import pytest

from sciel.engine.research import ResearchController
from sciel.hypothesis.comparison_review import build_cards
from sciel.knowledge_base.catalogs import canonical_json
from test_comparison_review_flow import fixture, review_proposal
from test_external_history_v2_actual import actual_history, construct, import_history
from test_research_measurement_state import begin_fixture
from test_research_paired_comparisons import decoded_context


def common_review(block):
    value = copy.deepcopy(block)
    value.pop('text_enabled')
    value.pop('rendered_chars')
    for card in value['cards']:
        card.pop('unverified_interpretation')
        card.pop('interpretation_omission')
    return value


def test_text_only_delivery_keeps_facts_knowledge_and_nonreview_prompt():
    arms = [fixture(reflect=True, comparison_review_text=mode) for mode in (True, False)]
    contexts, sent = [], []
    for f in arms:
        context = f.c.context(f.db, f.parent, 0, 2)
        _, prompt = begin_fixture(f.c, context)
        contexts.append(context)
        sent.append(decoded_context(prompt))
    on, off = contexts
    assert arms[0].c.knowledge.state() == arms[1].c.knowledge.state()
    assert on['comparison_reviews']['cards'][0]['unverified_interpretation'] is not None
    assert off['comparison_reviews']['cards'][0]['unverified_interpretation'] is None
    assert common_review(on['comparison_reviews']) == common_review(off['comparison_reviews'])
    assert on['comparison_reviews']['reserved_chars'] == off['comparison_reviews']['reserved_chars']
    for values in (contexts, sent):
        a, b = copy.deepcopy(values)
        a.pop('comparison_reviews'); b.pop('comparison_reviews')
        assert canonical_json(a) == canonical_json(b)
    # No model call beyond the fixture's same ordinary C and reflection R.
    assert len(arms[0].calls) == len(arms[1].calls) == 2


def test_authored_history_does_not_fill_withheld_text_with_another_conclusion(actual_history, tmp_path):
    from review_evidence_fixture import mixed_group
    capsule, final = actual_history
    contexts = []
    for mode in (True, False):
        f = construct(capsule, final, tmp_path / str(mode))
        import_history(f, capsule)
        # Add bounded authored restart evidence after importing the immutable
        # history, so this test exercises allocation beside both memory channels.
        record = f.c.knowledge.experiments[-1]
        group = mixed_group(origin=record['id'], challenger=record['source_sha256'],
            comparator=record['parent_source_sha256'], metric=f.c.metric)
        f.c.import_comparison(group)
        f.c._comparison_reviews = build_cards([review_proposal(group)], [group],
            {'phase': 'reflection', 'reference': 2, 'raw_response_sha256': 'd' * 64})
        f.c.config['comparison_review_text'] = mode
        contexts.append(f.c.context(NS(_all={'p1': f.parent}), f.parent, 0, 1))
    a, b = copy.deepcopy(contexts)
    assert a['comparison_reviews']['presented_count'] == b['comparison_reviews']['presented_count'] == 1
    on, off = a.pop('comparison_reviews'), b.pop('comparison_reviews')
    assert off['reserved_chars'] == on['reserved_chars']
    assert off['rendered_chars'] < off['reserved_chars']
    assert common_review(on) == common_review(off)
    assert canonical_json(a) == canonical_json(b)
    assert 'conclusion_feedback' in a and 'restart_observations' in a


@pytest.mark.parametrize('damage', ['switch', 'reserve', 'restore_text', 'erase_facts'])
def test_disabled_text_forgery_is_rejected_before_attempt(damage):
    f = fixture(reflect=True, comparison_review_text=False)
    context = f.c.context(f.db, f.parent, 0, 2)
    block = context['comparison_reviews']
    if damage == 'switch':
        block['text_enabled'] = True
    elif damage == 'reserve':
        block['reserved_chars'] = block['rendered_chars']
    elif damage == 'restore_text':
        block['cards'][0]['unverified_interpretation'] = {'interpretation': 'invented'}
    else:
        block['cards'][0]['host_facts'] = {}
    before = f.c.state()
    with pytest.raises(ValueError, match='comparison review'):
        begin_fixture(f.c, context)
    assert f.c.state() == before


@pytest.mark.parametrize('value', [None, 0, 1, 'false'])
def test_direct_constructor_rejects_invalid_text_switch(value):
    with pytest.raises(ValueError, match='comparison_review_text'):
        fixture(comparison_review_text=value)


def test_checkpoint_rejects_bool_alias_in_text_switch():
    f = fixture(reflect=True, comparison_review_text=True)
    saved = f.c.state()
    saved['config']['comparison_review_text'] = 1
    clone = ResearchController(f.c.config, f.c.llm, NS(research=lambda **_:None), lambda _:None,
                               metric=f.c.metric)
    with pytest.raises(ValueError, match='comparison_review_text'):
        clone.load_state(saved)
