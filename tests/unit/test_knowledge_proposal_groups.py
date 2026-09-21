"""One proposal group remains one KB observation and one advice exposure."""
import copy
import json

import pytest

from sciel.hypothesis import circle_proposal_group as groups
from sciel.hypothesis.rendering import serialize_reflection
from sciel.knowledge_base import KnowledgeBase
from sciel.knowledge_base.catalogs import canonical_json
from test_circle_proposal_group import FIRST, SECOND, make, record, wire
from test_research_knowledge import experiment, lesson, proposal


def test_two_alternatives_create_one_exposure_and_only_selected_quality_evidence():
    kb=KnowledgeBase();kb.record(experiment(90,branch='p1'));lid=lesson(kb,'r000090a1')
    group=make(selected=0);e=record(group);e['injected_ids']=[lid]
    kb.record(e)
    assert len(kb.experiments)==2 and kb.experiments[-1]['proposal_group']==group
    assert kb.lessons[0]['exposure_ids']==[e['id']]
    assert kb.lessons[0]['support_ids']==[e['id']]
    assert kb.lessons[0]['confidence']==2/3
    assert set(kb._evidence())=={'r000090a1',e['id']}
    assert len(kb.fragments)==1  # Only the original fixture; group alternatives minted none.
    with pytest.raises(ValueError):
        kb.add_lessons([proposal(e['id']+'/c1')],allowed_evidence_ids=[e['id']+'/c1'])
    lid2=lesson(kb,e['id'],rule='The selected measured initializer improved in this observation')
    assert kb.lessons[-1]['id']==lid2 and kb.lessons[-1]['evidence_ids']==[e['id']]
    restored=KnowledgeBase();restored.load_state(kb.state())
    assert canonical_json(restored.state())==canonical_json(kb.state())


def test_no_selection_keeps_both_forecasts_without_outcome_or_quality_credit():
    kb=KnowledgeBase();group=make(gates=[groups.gate('rejected',reason='known_source')]*2)
    e=record(group,status='no_child');kb.record(e)
    assert kb.experiments[0]['hypothesis'] is None
    assert all(x['hypothesis'] is not None for x in group['entries'])
    assert not kb.lessons and not kb.fragments
    assert all(kb.experiments[0][key] is None for key in ('score','actual_delta','source_sha256','proposed_sha256'))
    with pytest.raises(ValueError):lesson(kb,e['id'],polarity='negative')
    ctx=kb.proposal_group_context('p1')
    assert ctx['retained_count']==1 and ctx['presented_ids']==[e['id']]
    assert ctx['cards'][0]['selected_index'] is None
    assert 'unmeasured' in ctx['cards'][0]['interpretation']
    assert kb.proposal_group_context('foreign')['retained_count']==0


def test_context_admits_whole_cards_and_discloses_all_omissions_and_limits():
    kb=KnowledgeBase()
    for i in range(1,21):
        group=make(request_id=f'r{i:06d}a1',selected=0)
        kb.record(record(group,status='eval_failed'))
    ctx=kb.proposal_group_context('p1',max_items=1,max_chars=12000)
    assert ctx['presented_ids']==['r000020a1'] and ctx['omitted_count']==19
    assert len(ctx['omitted_ids'])==16 and ctx['omitted_ids_truncated']
    assert ctx['rendered_chars']==len(canonical_json(ctx))<=12000
    none=kb.proposal_group_context('p1',max_items=0,max_chars=2000)
    assert none['cards']==[] and none['omitted_count']==20
    small=kb.proposal_group_context('p1',max_items=3,max_chars=2000)
    assert small['cards']==[] and small['omitted_count']==20
    with pytest.raises(ValueError):kb.proposal_group_context('p1',max_chars=1)
    with pytest.raises(ValueError):kb.proposal_group_context('p1',max_items=True)
    before=kb.state();ctx['cards'][0]['entries'][1]['disposition_at_original_request']='selected'
    assert kb.state()==before


def test_reflection_uses_bounded_group_card_not_large_raw_bodies_or_extra_observations():
    data=json.loads(wire())
    # Keep the existing selected-source proof size fixed; only the unused
    # alternative grows. Its raw text must not inflate reflection capacity.
    data['candidates'][0]['body']=FIRST
    data['candidates'][1]['body']='#'+'b'*7000+'\n'+SECOND
    raw=json.dumps(data);group=make(raw,selected=0)
    kb=KnowledgeBase();kb.record(record(group))
    records,lessons=kb.reflection_context(1,max_chars=20000)
    assert len(records)==1 and not lessons
    view=records[0]
    assert 'proposal_group' not in view and 'proposal_group_card' in view
    assert view['proposal_group_card']==groups.group_card(group)
    assert len(serialize_reflection(records,lessons))<20000
    assert kb.experiments[0]['proposal_group']['entries'][1]['body']==data['candidates'][1]['body']


@pytest.mark.parametrize('mutation',[
    lambda e:e.update(hypothesis=e['proposal_group']['entries'][1]['hypothesis']),
    lambda e:e['proposal_group'].update(selected_index=True),
    lambda e:e['proposal_group']['entries'][1].update(actual_delta=0),
    lambda e:e['proposal_group'].update(selected_proof_sha256='0'*64),
    lambda e:e['proposal_group'].update(parent_source_sha256='0'*64),
])
def test_group_binding_rejection_is_atomic_and_state_load_cannot_accept_tampering(mutation):
    kb=KnowledgeBase();e=record(make(selected=0));mutation(e)
    before=kb.state()
    with pytest.raises(ValueError):kb.record(e)
    assert kb.state()==before
    good=record(make(selected=0));kb.record(good);state=kb.state();mutation(state['experiments'][0])
    live=kb.state()
    with pytest.raises(ValueError):kb.load_state(state)
    assert kb.state()==live


def test_legacy_state_load_and_reflection_shapes_have_no_new_empty_fields():
    kb=KnowledgeBase();kb.record(experiment(1))
    old=kb.state();other=KnowledgeBase();other.load_state(old)
    assert canonical_json(other.state())==canonical_json(old)
    assert 'proposal_group' not in old['experiments'][0]
    records,_=other.reflection_context(1)
    assert 'proposal_group_card' not in records[0]
    assert other.proposal_group_context('seed0')['retained_count']==0


def test_group_survives_old_experiment_eviction_as_one_lesson_source_snapshot():
    kb=KnowledgeBase(max_experiments=1);e=record(make(selected=0));kb.record(e);lesson(kb,e['id'])
    kb.record(experiment(99,branch='p1'))
    assert e['id'] not in [r['id'] for r in kb.experiments]
    assert kb._evidence()[e['id']]['proposal_group']==e['proposal_group']
    other=KnowledgeBase(max_experiments=1);other.load_state(kb.state())
    assert canonical_json(other.state())==canonical_json(kb.state())
    records,_=other.reflection_context(1,max_experiments=3)
    assert any(r['id']==e['id'] and 'proposal_group_card' in r for r in records)
