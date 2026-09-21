"""Pure synthetic receipts and KB operations; no helper/model/solver execution."""
from copy import deepcopy
import hashlib

import pytest

from sciel.hypothesis import initializer_evidence as evidence_api
from sciel.knowledge_base import KnowledgeBase
from sciel.knowledge_base.base import _digest
from sciel.knowledge_base.catalogs import canonical_json
from sciel.knowledge_base.presentation import validate_initializer_context
from test_initializer_evidence import synthetic_receipt, ROLES


@pytest.fixture(scope='module')
def evidence():
    return evidence_api.build_evidence(synthetic_receipt(),roles=ROLES)


@pytest.fixture(scope='module')
def different():
    return evidence_api.build_evidence(synthetic_receipt(child_shift=.01),roles=ROLES)


def record(ident='r1',source='1'*64,parent='2'*64):
    return dict(id=ident,branch='p1',status='evaluated',parent_score=1.,score=.75,actual_delta=-.25,
                source_sha256=source,proposed_sha256=source,parent_source_sha256=parent,hypothesis=None)


def kb(count=1,**kwargs):
    result=KnowledgeBase(**kwargs)
    for i in range(count):result.record(record('r'+str(i+1)))
    return result


def with_prior(value):
    # Explicit compact synthetic evidence, not claimed to be an executed receipt.
    value=deepcopy(value);value['roles']['prior']='prior'
    value['sources']['prior']=deepcopy(value['sources']['candidate'])
    value['derived']=evidence_api._derive(value,evidence_api._fixtures())
    value['plan_sha256']=evidence_api.core._hash(evidence_api._plan(value,evidence_api._fixtures()))
    value['evidence_sha256']=evidence_api.core._hash({k:v for k,v in value.items() if k!='evidence_sha256'})
    return evidence_api.validate_evidence(value)


def rehash(binding):
    binding['binding_sha256']=_digest({k:v for k,v in binding.items() if k!='binding_sha256'})


def test_registration_separate_legacy_shape_idempotent_and_no_alias(evidence):
    k=kb();before=k.state();fresh=KnowledgeBase();fresh.load_state(before);assert fresh.state()==before
    assert 'initializer_evidence' not in before
    supplied=deepcopy(evidence);assert k.register_initializer_evidence('r1',supplied) is True
    after=k.state();assert {key:value for key,value in after.items() if key!='initializer_evidence'}==before
    assert k._evidence()==fresh._evidence() and k._seen==fresh._seen
    assert k.register_initializer_evidence('r1',evidence) is False and k.state()==after
    supplied['sources'].clear();returned=k.initializer_evidence_for('r1');returned['evidence']['sources'].clear()
    assert k.state()==after
    fresh.load_state(after);assert fresh.state()==after
    proposal=dict(rule='Measured negative result.',polarity='negative',scope='branch',evidence_ids=['r1'])
    fresh.add_lessons([proposal],allowed_evidence_ids=['r1'])
    assert fresh.lessons[0]['source_support_ids']==['r1'] and fresh.lessons[0]['exposure_ids']==[]
    assert fresh.lessons[0]['confidence']==.5


def test_changed_evidence_same_owner_rejected_without_mutation(evidence,different):
    k=kb();k.register_initializer_evidence('r1',evidence);before=k.state()
    with pytest.raises(ValueError,match='conflicting'):k.register_initializer_evidence('r1',different)
    assert k.state()==before


def test_failed_ordinary_keeps_unknown_quality_when_initializer_evidence_exists(evidence):
    k=KnowledgeBase();failed=record();failed.update(status='eval_failed',score=None,actual_delta=None)
    k.record(failed);before=k.experiments
    k.register_initializer_evidence('r1',evidence)
    assert k.experiments==before and k.experiments[0]['actual_delta'] is None
    with pytest.raises(ValueError,match='measured quality'):
        k.add_lessons([dict(rule='A quality claim.',polarity='negative',scope='branch',evidence_ids=['r1'])],
                      allowed_evidence_ids=['r1'])
    assert k.lessons==[]


@pytest.mark.parametrize('damage',['unknown','source','parent','unmeasured','prior_missing','prior_source','prior_same','prior_role_absent','prior_id_absent'])
def test_exact_source_and_prior_joins_fail_before_registration(evidence,damage):
    k=KnowledgeBase();owner=record();value=evidence;ident='r1';prior=None
    if damage=='unknown':ident='absent'
    elif damage=='source':owner['source_sha256']=owner['proposed_sha256']='9'*64
    elif damage=='parent':owner['parent_source_sha256']='9'*64
    elif damage=='unmeasured':owner.update(status='no_child',source_sha256=None,proposed_sha256=None,score=None,actual_delta=None)
    if damage.startswith('prior_'):
        if damage!='prior_role_absent':value=with_prior(evidence)
        if damage!='prior_id_absent':prior='r0'
        if damage=='prior_same':prior='r1'
        elif damage not in ('prior_missing','prior_id_absent'):
            k.record(record('r0',source='9'*64 if damage=='prior_source' else '1'*64))
    k.record(owner);before=k.state()
    with pytest.raises(ValueError):k.register_initializer_evidence(ident,value,prior)
    assert k.state()==before


@pytest.mark.parametrize('damage',['wrapper','observation','source','seen','duplicate','capacity_bool','view'])
def test_restore_tampering_is_atomic(evidence,damage):
    k=kb();k.register_initializer_evidence('r1',evidence);before=k.state();state=deepcopy(before)
    extension=state['initializer_evidence'];binding=extension['records'][0]
    if damage=='wrapper':binding['binding_sha256']='f'*64
    elif damage=='observation':binding['observation_sha256']='f'*64;rehash(binding)
    elif damage=='source':binding['evidence']['sources']['candidate']['source_sha256']='f'*64;rehash(binding)
    elif damage=='seen':extension['seen']['r1']='f'*64
    elif damage=='duplicate':extension['records'].append(deepcopy(binding))
    elif damage=='capacity_bool':extension['seen']=True
    elif damage=='view':binding['evidence']=evidence_api.evidence_view(evidence);rehash(binding)
    with pytest.raises(ValueError):k.load_state(state)
    assert k.state()==before


def test_optional_prior_aging_keeps_bounded_tombstone_and_does_not_resurrect(evidence):
    k=KnowledgeBase(max_experiments=2);k.record(record('r0'));k.record(record('r1'))
    k.register_initializer_evidence('r1',with_prior(evidence),'r0')
    k.record(record('r2'));assert k.initializer_evidence_for('r1') is None
    state=k.state();assert set(state['initializer_evidence']['seen'])=={'r1'}
    assert state['initializer_evidence']['records']==[]
    fresh=KnowledgeBase(max_experiments=2);fresh.load_state(state);assert fresh.state()==state
    context=fresh.initializer_context(['r1']);assert context['omitted']==[dict(observation_id='r1',reason='evidence_aged_out')]
    with pytest.raises(ValueError):fresh.register_initializer_evidence('r1',evidence)
    assert fresh.state()==state
    fresh.record(record('r3'));assert 'initializer_evidence' not in fresh.state()
    with pytest.raises(ValueError,match='unavailable'):fresh.register_initializer_evidence('r1',evidence)


def test_bounded_full_fifo_keeps_owner_tombstone_without_stopping_new_registration(evidence,different):
    k=kb(max_experiments=1)
    k.add_lessons([dict(rule='Observed regression.',polarity='negative',scope='branch',evidence_ids=['r1'])],allowed_evidence_ids=['r1'])
    k.register_initializer_evidence('r1',evidence);k.record(record('r2'))
    assert k.initializer_evidence_for('r1') is not None
    ordinary=k.experiments;lessons=k.lessons;seen=deepcopy(k._seen)
    assert k.register_initializer_evidence('r2',evidence) is True
    assert k.initializer_evidence_for('r1') is None and k.initializer_evidence_for('r2') is not None
    assert set(k.state()['initializer_evidence']['seen'])=={'r1','r2'}
    assert k.experiments==ordinary and k.lessons==lessons and k._seen==seen
    assert k.initializer_context(['r1'])['omitted'][0]['reason']=='evidence_aged_out'
    before=k.state()
    for item in (evidence,different):
        with pytest.raises(ValueError):k.register_initializer_evidence('r1',item)
        assert k.state()==before
    fresh=KnowledgeBase(max_experiments=1);fresh.load_state(before);assert fresh.state()==before


def test_fragment_only_retention_does_not_resurrect_initializer_registry(evidence):
    code='fixture_value = 1\n';source=hashlib.sha256(code.encode()).hexdigest()
    value=deepcopy(evidence);value['sources']['candidate']['source_sha256']=source
    value['plan_sha256']=evidence_api.core._hash(evidence_api._plan(value,evidence_api._fixtures()))
    value['evidence_sha256']=evidence_api.core._hash({k:v for k,v in value.items() if k!='evidence_sha256'})
    k=KnowledgeBase(max_experiments=1);ordinary=record(source=source)
    ordinary.update(score=1.25,actual_delta=.25,code_fragment=code)
    k.record(ordinary);k.register_initializer_evidence('r1',value);k.record(record('r2'))
    assert k.fragments[0]['experiment_id']=='r1' and 'r1' in k._evidence()
    assert k.initializer_evidence_for('r1') is None and 'initializer_evidence' not in k.state()
    with pytest.raises(ValueError,match='unavailable'):k.register_initializer_evidence('r1',value)


def test_whole_context_and_derived_comparisons_are_separate_and_source_bound(evidence,different):
    k=kb(3)
    for ident,item in [('r1',evidence),('r2',different),('r3',evidence)]:k.register_initializer_evidence(ident,item)
    before=k.state();comparisons=k.initializer_comparisons_for('r1')
    assert [item['other_observation_id'] for item in comparisons]==['r3','r2']
    assert [item['comparison']['status'] for item in comparisons]==['same_on_witnesses','different_on_witness']
    context=k.initializer_context(['r1'],max_chars=50000)
    assert context['selected_ids']==context['presented_ids']==['r1']
    assert context['compared_count']==context['presented_comparison_count']==2
    assert context['cards'][0]['observation_sha256']==before['seen_experiments']['r1']
    assert validate_initializer_context(context)==context and k.state()==before
    context['cards'][0]['role_sources'].clear();assert k.state()==before


def test_budget_omits_whole_cards_and_reports_counts(evidence):
    k=kb();k.register_initializer_evidence('r1',evidence);before=k.state()
    full=k.initializer_context(['r1'],max_chars=50000)
    assert k.initializer_context(['r1'],max_chars=full['rendered_chars'])==full
    budget=full['rendered_chars']-1
    small=k.initializer_context(['r1'],max_chars=budget)
    assert small['cards']==[] and small['presented_count']==0 and small['omitted_count']==1
    assert small['omitted'][0]['reason']=='presentation_budget' and small['rendered_chars']<=budget
    assert k.initializer_context(['r1'],max_items=0)['omitted'][0]['reason']=='item_limit'
    assert k.initializer_context(['r1'],max_chars=0) is None and k.initializer_context([]) is None
    assert k.initializer_context(['r1'],max_chars=1) is None
    with pytest.raises(ValueError):k.initializer_context(['r'+str(i) for i in range(65)])
    assert k.state()==before


def test_comparison_computation_only_for_admitted_owners(evidence,monkeypatch):
    k=kb(4)
    for i in range(4):k.register_initializer_evidence('r'+str(i+1),evidence)
    called=[];original=k.initializer_comparisons_for
    def capture(ident):called.append(ident);return original(ident)
    monkeypatch.setattr(k,'initializer_comparisons_for',capture)
    value=k.initializer_context(['r1','r2','r3','r4'],max_items=1,max_chars=50000)
    assert called==value['comparison_owner_ids']==['r1']
    assert value['comparison_owner_omitted_count']==3 and value['not_computed_comparison_count']==9
    assert value['compared_count']==3


@pytest.mark.parametrize('damage',['view','source','comparison','branch','count'])
def test_context_validation_rejects_rehashed_internal_mismatch(evidence,damage):
    k=kb(2)
    for ident in ('r1','r2'):k.register_initializer_evidence(ident,evidence)
    value=k.initializer_context(['r1'],max_chars=50000)
    if damage=='view':value['cards'][0]['evidence']['candidate_cases'][0][0]='unknown'
    elif damage=='source':value['cards'][0]['role_sources']['candidate']='f'*64
    elif damage=='comparison':value['comparisons'][0]['observation_sha256']='f'*64
    elif damage=='branch':value['comparisons'][0]['branch']='wrong_branch'
    else:value['omitted_count']=1
    value['context_sha256']='0'*64
    while value['rendered_chars']!=len(canonical_json(value)):value['rendered_chars']=len(canonical_json(value))
    value['context_sha256']=_digest({k:v for k,v in value.items() if k!='context_sha256'})
    with pytest.raises(ValueError):validate_initializer_context(value)


def test_cross_branch_comparison_keeps_explicit_origins_without_quality_transfer(evidence):
    k=KnowledgeBase();k.record(record('r1'));other=record('r2');other['branch']='p2';k.record(other)
    for ident in ('r1','r2'):k.register_initializer_evidence(ident,evidence)
    before=k.state();view=k.initializer_context(['r1'],max_chars=50000)
    card,=view['cards'];comparison,=view['comparisons']
    assert card['branch']=='p1' and comparison['branch']=='p1' and comparison['other_branch']=='p2'
    assert comparison['other_observation_sha256']==before['seen_experiments']['r2']
    assert comparison['comparison']['status']=='same_on_witnesses'
    assert 'not transferable quality, lesson support' in comparison['scope']
    assert validate_initializer_context(view)==view and k.state()==before
    assert k.lessons==[] and k.comparisons==[] and len(k.experiments)==2
    both=k.initializer_context(['r1','r2'],max_chars=50000)
    assert {item['observation_id']:item['branch'] for item in both['cards']}=={'r1':'p1','r2':'p2'}
