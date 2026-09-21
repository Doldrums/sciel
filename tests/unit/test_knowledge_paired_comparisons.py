"""Paired evidence has one source credit and no invented forecast or exposure."""
import copy
import json

import pytest

from sciel.knowledge_base import KnowledgeBase
from sciel.knowledge_base.catalogs import canonical_json
from sciel.knowledge_base.paired_comparisons import build_comparison
from sciel.knowledge_base.presentation import (comparison_card, observation_card,
    reflection_payload, reflection_inline_payload, expand_reflection_payload)
from sciel.hypothesis.rendering import decode_evidence, serialize_reflection
from test_paired_comparisons import inputs
from test_research_knowledge import experiment, lesson


def group(n=1, *, new=(.11, .11, .11), branch='p1'):
    data = inputs(new=new)
    data.update(batch_id=f'b{n:06d}', branch=branch)
    return build_comparison(**data)


def test_opposite_repeats_leave_original_forecast_attempt_and_positive_lesson_intact():
    kb = KnowledgeBase()
    kb.record(experiment(1, branch='p1'))
    lesson(kb, 'r000001a1')
    before = kb.state()
    g = group(new=(.09, .09, .09))
    assert kb.record_comparison(g) is True
    for field in ('seen_experiments', 'experiments', 'lessons', 'fragments'):
        assert kb.state()[field] == before[field]
    assert len(kb.comparisons) == 1
    assert kb.comparisons[0]['forecast'] == 'not_scored'
    assert kb.experiments[0]['forecast_verdict'] == 'CONFIRMED'
    context = kb.comparison_context('p1')
    assert context['cards'][0]['derived']['outcome'] == 'regressed'
    assert context['rendered_chars'] == len(canonical_json(context)) <= 12000
    assert kb.comparison_context('foreign')['cards'] == []
    records, lessons = kb.reflection_context(20)
    wire = decode_evidence(json.loads(serialize_reflection(records, lessons)))
    assert wire['observation_catalog']['r000001a1']['actual_delta'] > 0
    assert wire['observation_catalog'][g['id']]['derived']['outcome'] == 'regressed'
    assert len(records) == 2


def test_replay_conflict_and_returned_mutations_are_atomic_even_after_eviction():
    kb = KnowledgeBase(max_experiments=1)
    first = group(); kb.record_comparison(first)
    kb.record_comparison(group(2))
    assert kb.record_comparison(first) is False
    before = kb.state()
    with pytest.raises(ValueError, match='conflict'):
        kb.record_comparison(group(new=(.09, .09, .09)))
    assert kb.state() == before
    clone = kb.comparisons; clone[0]['branch'] = 'changed'
    assert kb.state() == before
    restored = KnowledgeBase(max_experiments=1); restored.load_state(before)
    assert restored.state() == before and restored.record_comparison(first) is False


@pytest.mark.parametrize('new,polarity', [((.11,.11,.11),'positive'), ((.09,.09,.09),'negative')])
def test_group_lesson_is_one_source_zero_exposures_and_survives_eviction(new,polarity):
    kb = KnowledgeBase(max_experiments=1)
    g=group(new=new); kb.record_comparison(g)
    lid=lesson(kb,g['id'],polarity=polarity)
    kb.record_comparison(group(2))
    saved=kb.state(); restored=KnowledgeBase(max_experiments=1); restored.load_state(saved)
    assert restored.state()==saved
    l=restored.lessons[0]
    assert l['source_support_ids']==[g['id']] and l['exposure_ids']==[] and l['confidence']==.5
    assert not restored.experiments and not restored.fragments and not restored._seen
    card=observation_card(l)
    assert card['source_counts']['total']==1
    assert card['source_counts'][polarity]==1
    assert card['advice_exposures']['retained_unique_counts']['total']==0
    assert card['source_observations'][0]['id']==g['id']
    records,lessons=restored.reflection_context(20)
    assert g['id'] in [r['id'] for r in records]
    assert expand_reflection_payload(reflection_payload(records,lessons))==reflection_inline_payload(records,lessons)
    restored.record(experiment(1,branch='p1',injected=[lid]))
    assert restored.lessons[0]['exposure_ids']==['r000001a1']
    assert restored.lessons[0]['source_support_ids']==[g['id']]


@pytest.mark.parametrize('new',[(.11,.09,.1),(.1,.1,.1)])
@pytest.mark.parametrize('polarity',['positive','negative'])
def test_mixed_and_tie_are_facts_not_polarity_support(new,polarity):
    kb=KnowledgeBase(); g=group(new=new); kb.record_comparison(g); before=kb.state()
    with pytest.raises(ValueError,match='polarity'):
        lesson(kb,g['id'],polarity=polarity)
    assert kb.state()==before
    assert kb.comparison_context('p1')['cards'][0]['derived']==g['derived']


def test_incomplete_groups_and_views_cannot_be_used_as_quality_evidence():
    data=inputs(); data['pairs'][1]['challenger']=None
    g=build_comparison(**data); kb=KnowledgeBase();kb.record_comparison(g)
    for polarity in ('positive','negative'):
        with pytest.raises(ValueError,match='polarity'):
            lesson(kb,g['id'],polarity=polarity)
    with pytest.raises(ValueError):kb.record_comparison(comparison_card(g))


@pytest.mark.parametrize('mutation',['derived','index','exposure','collision','duplicate'])
def test_saved_group_tampering_cannot_replace_live_state(mutation):
    kb=KnowledgeBase();g=group();kb.record_comparison(g);lesson(kb,g['id'])
    before=kb.state(); changed=copy.deepcopy(before)
    if mutation=='derived':changed['paired_comparisons']['records'][0]['derived']['outcome']='regressed'
    elif mutation=='index':changed['paired_comparisons']['seen'][g['id']]='f'*64
    elif mutation=='exposure':changed['lessons'][0]['exposure_snapshots']=[g]
    elif mutation=='collision':changed['seen_experiments'][g['id']]='f'*64
    elif mutation=='duplicate':changed['paired_comparisons']['records'].append(g)
    with pytest.raises(ValueError):kb.load_state(changed)
    assert kb.state()==before


def test_whole_card_admission_and_bounded_omission_metadata():
    kb=KnowledgeBase()
    for n in range(1,24):kb.record_comparison(group(n))
    view=kb.comparison_context('p1',max_items=0)
    assert view['cards']==[] and view['omitted_count']==23
    assert len(view['omitted_ids'])==16 and view['omitted_ids_truncated'] is True
    assert view['rendered_chars']==len(canonical_json(view))<=12000
    one=kb.comparison_context('p1',max_items=1)
    assert len(one['cards'])==1 and one['cards'][0]['batch_id']=='b000023'
    assert len(one['cards'][0]['derived']['pairs'])==3
    limited=kb.comparison_context('p1',max_chars=view['rendered_chars'])
    assert limited['cards']==[] and limited['omitted_count']==23


def test_global_rule_still_requires_two_distinct_branches_not_three_seeds():
    kb=KnowledgeBase();a=group();kb.record_comparison(a)
    with pytest.raises(ValueError,match='two branches'):lesson(kb,a['id'],scope='global')
    b=group(2,branch='p2');kb.record_comparison(b)
    lid=lesson(kb,a['id'],b['id'],scope='global')
    assert len(kb.lessons[0]['source_support_ids'])==2
    assert kb.retrieve(branch='p3',context='steps')[0]['id']==lid


def test_legacy_state_shape_is_unchanged_and_can_replace_nonempty_comparison_state():
    legacy=KnowledgeBase().state(); assert 'paired_comparisons' not in legacy
    kb=KnowledgeBase();kb.record_comparison(group());kb.load_state(legacy)
    assert kb.state()==legacy and kb.comparisons==[]


def test_recent_attempt_window_cannot_starve_paired_reflection_card():
    kb=KnowledgeBase();g=group();kb.record_comparison(g)
    for n in range(1,5):kb.record(experiment(n,branch='p1'))
    records,_=kb.reflection_context(4,max_experiments=4)
    assert len(records)==4 and g['id'] in [r['id'] for r in records]
    assert 'r000001a1' not in [r['id'] for r in records]
    assert len(kb.experiments)==4
