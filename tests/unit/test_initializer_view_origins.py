"""Pure saved-array provenance projections; never execute a candidate or helper."""
from copy import deepcopy
from functools import lru_cache
import itertools
import json

import pytest
from sciel.hypothesis import circle_initializer_witness as core
from sciel.hypothesis import initializer_evidence as evidence

ROLES={'candidate':'candidate','parent':'parent'}


@lru_cache(maxsize=8)
def _synthetic(profile,mode):
    fixtures=core.make_fixture_bundle(profile)
    identity={k:'d'*64 for k in evidence._IDENTITY if k.endswith('_sha256')}
    identity.update(builder_files={k:'e'*64 for k in ('circle_interventions.py','circle_proposals.py',
        'circle_restart_body.py','circle_restart_scales.py')},builder_contract='circle_restart_body/1',
        python='synthetic Python',platform='synthetic platform',numpy_version='synthetic version')
    pins=[dict(label=label,source_sha256=char*64,body_sha256='b'*64,helper_sha256='c'*64)
          for label,char in [('candidate','1'),('parent','2')]]
    plan=dict(schema=core.SCHEMA,identity=identity,fixtures=fixtures,sources=pins,limits=core._limits(5.),scope=core.SCOPE)
    plan['plan_sha256']=core._hash(plan);sources=[]
    for pin in pins:
        candidate=pin['label']=='candidate'
        events=[dict(kind='header',plan_sha256=plan['plan_sha256'],label=pin['label'],identity=identity)]
        for index,case in enumerate(fixtures['cases']):
            events.append(dict(kind='intent',case_id=case['id']))
            if candidate and mode=='mixed' and index==5:break
            selected=('fallback','ordinary','best','failed','failed_value')[index] if candidate and mode=='mixed' else mode if candidate else 'fallback'
            broken=selected in ('failed','failed_value')
            value=selected in ('ordinary','best','failed_value')
            output=core._decode_array(case['inputs']['best_params' if selected=='best' else 'ordinary_params'])
            raw_array=core._array([1.]) if selected=='failed_value' else core._array(output) if value else None
            raw=None if selected=='failed' else dict(kind='value' if value else 'none',array=raw_array,
                python_type='ndarray' if value else 'NoneType',shares_input_storage=[False]*4 if value else None)
            events.append(dict(kind='observation',case_id=case['id'],status='failed' if broken else 'observed',
                raw_return=raw,post_inputs=deepcopy(case['inputs']),effective_output=None if broken else core._array(output),
                error_code='ZeroDivisionError' if selected=='failed' else 'ValueError' if broken else None,
                input_storage_detached=True,handoff_counts=dict(proposal_attempts=1,proposal_accepted=int(value and not broken),
                    proposal_none=int(not value and not broken),proposal_errors=int(broken))))
        partial=candidate and mode=='mixed'
        if not partial:events.append(dict(kind='terminal',completed=True))
        process=dict(returncode=-9 if partial else 0,failure='timeout' if partial else None,elapsed_seconds=0.,
            stdout=b''.join(core._canonical(v).encode()+b'\n' for v in events),stderr=b'')
        sources.append({**pin,**core._consume(process,plan,pin['label'])})
    compact={s['label']:{'complete':s['complete'],'observations':[dict(status=o['status'],
        effective_output=evidence._compact(o['effective_output']) if o['status']=='observed' else None) for o in s['observations']]} for s in sources}
    comparisons=[dict(left=a,right=b,**evidence._compare_sources(compact[a],compact[b],[c['id'] for c in fixtures['cases']]))
        for a,b in itertools.combinations(compact,2)]
    receipt=dict(schema=core.SCHEMA,plan=plan,sources=sources,comparisons=comparisons,
        status='observed' if all(s['status']=='observed' for s in sources) else 'incomplete',scope=core.SCOPE,
        candidate_rejection_authorized=False,semantic_equivalence_proven=False,new_model_calls=0,new_solver_calls=0,new_native_evaluations=0)
    receipt['receipt_sha256']=core._hash(receipt)
    return evidence.build_evidence(receipt,roles=ROLES)


def synthetic_value(profile='legacy_v1',mode='fallback'):
    """Detached real validated evidence for prompt tests; pure fake streams only."""
    assert mode in ('fallback','ordinary','best','mixed')
    return deepcopy(_synthetic(profile,mode))


def view(mode='fallback',profile='legacy_v1'):
    return evidence.evidence_view(synthetic_value(profile,mode),max_chars=20000)


def rehash(card):card['view_sha256']=core._hash({k:v for k,v in card.items() if k!='view_sha256'})


def rows(card):return evidence.evidence_case_views(card)


def handoff(card,index=0):return card['handoffs'][card['candidate_cases'][index][2]]


def origin_view(value):
    """Construct the prior wire shape directly from validated stored observations."""
    card=evidence.evidence_view(value,max_chars=20000)
    card['schema']=evidence.ORIGIN_VIEW_SCHEMA;card['scope']=evidence.ORIGIN_VIEW_SCOPE
    card['columns'].pop('case');card.pop('handoffs')
    card['candidate_cases']=[evidence._origin_case_view(row,obs) for row,obs in zip(
        value['derived']['candidate_cases'],value['sources'][value['roles']['candidate']]['observations'],strict=True)]
    rehash(card)
    return card


@pytest.mark.parametrize('profile',core.FIXTURE_PROFILES)
def test_explicit_provenance_distinguishes_equal_effective_arrays(profile,monkeypatch):
    for name in ('prepare_witness','identity','_launch','witness_sources'):
        monkeypatch.setattr(core,name,lambda *a,**k:pytest.fail('projection must remain pure'))
    fallback=view('fallback',profile);returned=view('ordinary',profile);best=view('best',profile)
    assert fallback['schema']==evidence.VIEW_SCHEMA=='sciel-initializer-evidence-view/3'
    assert fallback['columns']['handoff']==['attempts','accepted','none','errors']
    assert fallback['columns']['case']==['status','facts','handoff_ref']
    for a,b,c in zip(rows(fallback),rows(returned),rows(best),strict=True):
        assert a['facts']==b['facts']  # array equality cannot choose provenance
        assert (a['origin'],a['return_kind'],a['handoff'],a['error'])==('saved_ordinary','none',[1,0,1,0],None)
        assert (b['origin'],b['return_kind'],b['handoff'],b['error'])==('return','value',[1,1,0,0],None)
        assert (c['origin'],c['return_kind'],c['handoff'],c['error'])==('return','value',[1,1,0,0],None)
        assert c['facts']['changes'][:3]==[0,0,0]  # accepted value need not edit best
    for card in (fallback,returned,best):assert evidence.validate_evidence_view(card)==card
    assert fallback['scope']==evidence.VIEW_SCOPE and 'not necessarily helper edits' in fallback['scope']


def test_legacy_w05_changed_radii_are_explicitly_saved_ordinary_not_helper_edits():
    card=view();row=rows(card)[5]
    assert row['case_id']=='w05' and row['facts']['changes'][2]==26
    assert row['origin']=='saved_ordinary' and row['return_kind']=='none' and row['handoff']==[1,0,1,0]


@pytest.mark.parametrize('profile',core.FIXTURE_PROFILES)
def test_failure_unknown_and_not_dispatched_have_distinct_exact_metadata(profile):
    card=view('mixed',profile);expanded=rows(card)
    assert expanded[3]['status']=='failed' and expanded[3]['origin'] is None and expanded[3]['return_kind'] is None
    assert expanded[3]['handoff']==[1,0,0,1] and expanded[3]['error']=='ZeroDivisionError' and expanded[3]['facts'] is None
    assert expanded[4]['status']=='failed' and expanded[4]['return_kind']=='value' and expanded[4]['origin'] is None
    assert expanded[4]['handoff']==[1,0,0,1] and expanded[4]['error']=='ValueError'
    assert expanded[5]['status']=='unknown'
    for row in expanded[5:]:
        assert row['origin'] is row['return_kind'] is row['handoff'] is row['facts'] is None
        assert row['error']=='timeout'
    assert all(row['status']=='not_dispatched' for row in expanded[6:])
    assert evidence.validate_evidence_view(card)==card


@pytest.mark.parametrize('change',[
    {'origin':'return'}, {'return_kind':'value'}, {'handoff':[1,1,0,0]},
    {'handoff':[True,0,1,0]}, {'error':'ValueError'}, {'origin':None},
])
def test_rehashed_contradictory_observed_provenance_rejected(change):
    card=view();handoff(card).update(change);rehash(card)
    with pytest.raises(ValueError):evidence.validate_evidence_view(card)


def test_rehashed_fallback_facts_cannot_claim_a_helper_edit():
    card=view();card['candidate_cases'][5][1]['changes'][3]=1;rehash(card)
    with pytest.raises(ValueError,match='saved ordinary view facts'):evidence.validate_evidence_view(card)


@pytest.mark.parametrize('index,change',[(3,{'origin':'return'}),(3,{'error':None}),
    (3,{'handoff':[1,1,0,0]}),(5,{'return_kind':'none'}),(5,{'handoff':[0,0,0,0]}),
    (5,{'error':None}),(6,{'origin':'saved_ordinary'})])
def test_rehashed_failed_or_missing_provenance_cannot_invent_an_output(index,change):
    card=view('mixed');handoff(card,index).update(change);rehash(card)
    with pytest.raises(ValueError):evidence.validate_evidence_view(card)


@pytest.mark.parametrize('profile',core.FIXTURE_PROFILES)
def test_legacy_view_parser_returns_original_shape_without_inferred_origins(profile):
    old=origin_view(synthetic_value(profile));old['schema']=evidence.LEGACY_VIEW_SCHEMA;old['scope']=evidence.SCOPE
    old['columns'].pop('handoff')
    for row in old['candidate_cases']:
        for key in ('origin','return_kind','handoff','error'):row.pop(key)
    rehash(old);raw=core._canonical(old)
    restored=evidence.validate_evidence_view(old)
    assert core._canonical(restored)==raw
    assert all('origin' not in row for row in restored['candidate_cases'])
    assert rows(restored)==restored['candidate_cases']
    old['candidate_cases'][0]['origin']=None;rehash(old)
    with pytest.raises(ValueError):evidence.validate_evidence_view(old)


def test_view_scope_does_not_restamp_stored_evidence_or_comparisons():
    value=synthetic_value();before=core._canonical(value);comparison=evidence.compare_evidence(value,value)
    card=evidence.evidence_view(value,max_chars=20000)
    assert core._canonical(value)==before and evidence.validate_evidence(value)==value
    assert value['schema']=='sciel-initializer-evidence/1' and value['scope']==evidence.SCOPE
    assert evidence.compare_evidence(value,value)==comparison
    assert card['scope']!=value['scope']
    handoff(card)['handoff'][0]=99
    assert rows(evidence.evidence_view(value,max_chars=20000))[0]['handoff']==[1,0,1,0]


@pytest.mark.parametrize('profile',core.FIXTURE_PROFILES)
def test_whole_card_budget_is_measured_and_omission_is_not_silent_truncation(profile):
    value=synthetic_value(profile);whole=evidence.evidence_view(value,max_chars=20000)
    size=len(core._canonical(whole));assert size<20000
    assert evidence.evidence_view(value,max_chars=size)==whole
    omitted=evidence.evidence_view(value,max_chars=size-1)
    assert omitted['status']=='omitted' and omitted['card_chars']==size and 'candidate_cases' not in omitted
    assert evidence.validate_evidence_view(omitted)==omitted
    assert rows(omitted)==[]
    tiny=evidence.evidence_view(value,max_chars=1)
    assert tiny['status']=='omitted' and tiny['fits_budget'] is False
    assert evidence.validate_evidence_view(tiny)==tiny
    default=evidence.evidence_view(value)
    assert default==whole if size<=8192 else default['status']=='omitted'
    assert default['evidence_sha256']==value['evidence_sha256']


@pytest.mark.parametrize('profile',core.FIXTURE_PROFILES)
@pytest.mark.parametrize('mode',['fallback','ordinary','best','mixed'])
def test_compaction_is_lossless_and_prior_origin_view_is_exactly_readable(profile,mode):
    value=synthetic_value(profile,mode);old=origin_view(value);new=view(mode,profile)
    before=core._canonical(old)
    assert evidence.validate_evidence_view(old)==old and core._canonical(old)==before
    assert rows(new)==rows(old)==old['candidate_cases']
    assert old['schema']=='sciel-initializer-evidence-view/2'
    assert new['scope']==old['scope']+' Case rows align with case_ids; handoff_ref indexes handoffs.'
    assert new['comparisons']==old['comparisons']
    assert len(core._canonical(new))<len(before)
    assert view(mode,profile)==new  # stable catalog numbering and digest
    assert all(type(row) is list and len(row)==3 for row in new['candidate_cases'])
    assert new['case_ids']==old['case_ids']
    metadata=[]
    for row in old['candidate_cases']:
        item={key:row[key] for key in ('origin','return_kind','handoff','error')}
        if item not in metadata:metadata.append(item)
    assert new['handoffs']==metadata


def test_resolver_detaches_every_row_including_shared_catalog_entries():
    card=view();original=deepcopy(card);expanded=rows(card)
    assert len(card['handoffs'])==1
    expanded[0]['handoff'][0]=99;expanded[0]['facts']['changes'][0]=99
    assert expanded[1]['handoff']==[1,0,1,0]
    assert card==original and rows(card)[0]['handoff']==[1,0,1,0]


@pytest.mark.parametrize('ref',[True,False,-1,1,0.0,'0',None])
def test_catalog_references_are_exact_bounded_integers(ref):
    card=view();card['candidate_cases'][0][2]=ref;rehash(card)
    with pytest.raises(ValueError,match='handoff reference'):rows(card)


@pytest.mark.parametrize('mutation',[
    lambda c:c['handoffs'].append(deepcopy(c['handoffs'][0])),
    lambda c:c['handoffs'].append({'origin':'return','return_kind':'value','handoff':[1,1,0,0],'error':None}),
    lambda c:c['handoffs'].clear(),
    lambda c:c['handoffs'][0].__setitem__('invented',None),
    lambda c:c['candidate_cases'][0].append(None),
    lambda c:c['candidate_cases'].pop(),
    lambda c:c['case_ids'].__setitem__(0,c['case_ids'][1]),
])
def test_rehashed_catalog_and_case_shape_tamper_rejected(mutation):
    card=view();mutation(card);rehash(card)
    with pytest.raises(ValueError):rows(card)


def test_relabelled_catalog_cannot_create_noncanonical_first_use_order():
    card=view('mixed');card['handoffs'][0],card['handoffs'][1]=card['handoffs'][1],card['handoffs'][0]
    for row in card['candidate_cases']:
        if row[2] in (0,1):row[2]=1-row[2]
    rehash(card)
    with pytest.raises(ValueError,match='first-use order'):rows(card)


@pytest.mark.parametrize('schema',[evidence.LEGACY_VIEW_SCHEMA,evidence.ORIGIN_VIEW_SCHEMA])
def test_legacy_omitted_view_does_not_invent_case_rows(schema):
    old=evidence.evidence_view(synthetic_value(),max_chars=1);old['schema']=schema;rehash(old)
    original=core._canonical(old)
    assert evidence.validate_evidence_view(old)==old and rows(old)==[]
    assert core._canonical(old)==original
