"""Pure authoring provenance, inert auxiliary provider, no evaluator or model I/O."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from sciel.config import apply_defaults
from sciel.engine.llm.client import LLMResponse
from sciel.engine.research import ResearchController, digest
from sciel.hypothesis.rendering import presentation_identity, serialize_reflection
from sciel.knowledge_base import KnowledgeBase
from sciel.knowledge_base import authoring
from sciel.knowledge_base.catalogs import canonical_sha
from sciel.knowledge_base.presentation import observation_card, reflection_payload, expand_reflection_payload


def record(ident='r000001a1',branch='p1'):
    source=hashlib.sha256(b'value = 1\n').hexdigest()
    return dict(id=ident,branch=branch,status='evaluated',parent_score=1.,score=.75,actual_delta=-.25,
        source_sha256=source,proposed_sha256=source,parent_source_sha256='a'*64,
        change_excerpt='value = 1\n',hypothesis=dict(text='This change will improve score.',
        reasoning='The effect is untested.',predicted_low=.1,predicted_high=.2,confidence=.5))


def kb():
    k=KnowledgeBase();k.record(record());return k


def proposal(rule='The measured child regressed.'):
    return dict(rule=rule,polarity='negative',scope='branch',evidence_ids=['r000001a1'])


def writer(tag='a',phase='conclusion'):
    return dict(phase=phase,reference='r000001a1' if phase=='conclusion' else 1,raw_response_sha256=tag*64)


def add(k,rule='The measured child regressed.',context=None):
    return k.add_lessons([proposal(rule)],allowed_evidence_ids=['r000001a1'],writer_context=context)[0]


def rewrite(k,old,rule,tag='b'):
    return k.reflect([proposal(rule)],[old],allowed_evidence_ids=['r000001a1'],
                     writer_context=writer(tag,'reflection'))['lesson_ids'][0]


def nodes(lesson,kind):return [n for n in lesson['authoring']['catalog'].values() if n['kind']==kind]


def test_legacy_direct_api_and_restore_keep_exact_shape_and_unknown_authorship():
    k=kb();old=add(k);saved=k.state()
    assert 'authoring' not in k.lessons[0] and 'authoring_facts' not in observation_card(k.lessons[0])
    fresh=KnowledgeBase();fresh.load_state(saved);assert fresh.state()==saved
    fresh.reflect([proposal('Another wording.')],[old],allowed_evidence_ids=['r000001a1'])
    assert 'authoring' not in fresh.lessons[0]


def test_writer_binds_literal_unicode_and_stored_trim_without_model_owned_metadata():
    k=kb();raw=' \nObserved α e\u0301 regression.\t';ctx=writer();saved=deepcopy(ctx)
    ident=add(k,raw,ctx);l=k.lessons[0];n,=nodes(l,'writer')
    assert l['rule']==raw.strip() and n['raw_rule']==raw
    assert n['raw_rule_sha256']==authoring.text_sha(raw) and n['stored_rule_sha256']==authoring.text_sha(raw.strip())
    assert n['proposal_index']==0 and n['reference']=='r000001a1' and ctx==saved
    assert l['authoring']['prior_writers_unknown'] is False
    bad={**proposal(),'authoring':l['authoring']};before=k.state()
    with pytest.raises(ValueError,match='proposal schema'):k.add_lessons([bad],allowed_evidence_ids=['r000001a1'])
    assert k.state()==before and ident=='l000001'
    exported=k.state();exported['lessons'][0]['authoring']['catalog'].clear()
    assert nodes(k.lessons[0],'writer')==[n]


def test_same_writer_idempotent_multiple_writers_dedup_without_new_observations():
    k=kb();ident=add(k,context=writer());before=k.state()
    assert add(k,context=writer())==ident and k.state()==before
    assert add(k,context=writer('b'))==ident
    l=k.lessons[0];assert len(nodes(l,'writer'))==2
    assert k.state()['next_lesson_id']==2 and k.experiments==before['experiments']
    for key in ('evidence_ids','source_support_ids','exposure_ids','support_ids','refute_ids','confidence'):
        assert l[key]==before['lessons'][0][key]


def test_zero_exposure_consolidation_retains_all_ancestors_and_writer_refs():
    k=kb();one=add(k,'First wording.',writer());two=rewrite(k,one,'Second wording.')
    three=rewrite(k,two,'Third wording.','c');l=k.lessons[0]
    assert l['id']==three and l['exposure_ids']==[] and l['inherited_exposures']==[]
    ancestors=nodes(l,'retired_lesson')
    assert {n['lesson_id'] for n in ancestors}=={one,two} and len(nodes(l,'writer'))==3
    assert all(n['polarity']=='negative' and n['branch']=='p1' and n['evidence_ids']==['r000001a1'] for n in ancestors)
    assert {n['rule_sha256'] for n in ancestors}=={authoring.text_sha('First wording.'),authoring.text_sha('Second wording.')}
    fresh=KnowledgeBase();fresh.load_state(k.state());assert fresh.state()==k.state()


def test_rewriting_legacy_does_not_assign_new_writer_to_old_wording():
    k=kb();one=add(k,'Legacy wording.');rewrite(k,one,'New wording.')
    l=k.lessons[0];old,=nodes(l,'retired_lesson');new,=nodes(l,'writer')
    assert old['prior_writers_unknown'] is True and old['writer_refs']==[]
    assert old['rule']=='Legacy wording.' and old['rule_sha256']==authoring.text_sha(old['rule'])
    assert new['raw_rule']=='New wording.' and l['exposure_ids']==[] and l['inherited_exposures']==[]
    assert l['authoring']['prior_writers_unknown'] is False
    k2=kb();add(k2,'Legacy wording.');add(k2,'Legacy wording.',writer())
    assert k2.lessons[0]['authoring']['prior_writers_unknown'] is True


def test_catalog_or_serialized_capacity_failure_is_atomic(monkeypatch):
    k=kb();one=add(k,'First wording.',writer());saved=k.state()
    monkeypatch.setattr(authoring,'MAX_NODES',2)
    with pytest.raises(ValueError,match='capacity'):rewrite(k,one,'Replacement.')
    assert k.state()==saved
    monkeypatch.setattr(authoring,'MAX_NODES',64);monkeypatch.setattr(authoring,'MAX_CHARS',100)
    with pytest.raises(ValueError,match='capacity'):add(k,context=writer('c'))
    assert k.state()==saved


@pytest.mark.parametrize('damage',['digest','dangling','wrong_rule','raw_rule','unknown_flag','orphan','ancestor_scope','ancestor_counter','ancestor_literal','writer_reference'])
def test_restore_rejects_malformed_provenance_atomically(damage):
    k=kb();one=add(k,'First.',writer());rewrite(k,one,'Second.')
    state=k.state();a=state['lessons'][0]['authoring'];catalog=a['catalog']
    current=a['writer_refs'][0];revision=a['revision_refs'][0]
    if damage=='digest':catalog[current]['proposal_index']=3
    elif damage=='dangling':a['writer_refs']=['f'*64]
    elif damage=='unknown_flag':a['prior_writers_unknown']=1
    elif damage=='orphan':
        node=deepcopy(catalog[current]);node['proposal_index']=7;catalog[canonical_sha(node)]=node
    else:
        ref=revision if damage.startswith('ancestor_') else current
        node=catalog.pop(ref)
        if damage=='wrong_rule':node.update(raw_rule='Different.',raw_rule_sha256=authoring.text_sha('Different.'),stored_rule_sha256=authoring.text_sha('Different.'))
        elif damage=='raw_rule':node['raw_rule']='altered without updating literal hash'
        elif damage=='ancestor_scope':node['branch']='p2'
        elif damage=='ancestor_literal':node['rule']='Tampered old wording.'
        elif damage=='ancestor_counter':node['lesson_id']='l999999'
        elif damage=='writer_reference':node['reference']='r000999a1'
        new=canonical_sha(node);catalog[new]=node
        key='revision_refs' if ref==revision else 'writer_refs';a[key]=[new if r==ref else r for r in a[key]]
    before=k.state()
    with pytest.raises(ValueError):k.load_state(state)
    assert k.state()==before


def test_authoring_context_and_prompt_roundtrip_separate_text_and_observation_origins():
    k=kb();one=add(k,'First wording.',writer());rewrite(k,one,'Second wording.')
    l=k.lessons[0];card=observation_card(l)
    assert card['authoring_facts']['provenance']==l['authoring']
    assert card['source_ids']==['r000001a1'] and card['source_counts']['total']==1
    assert 'not observations' in card['authoring_facts']['interpretation_limits']
    records,lessons=k.reflection_context(1)
    payload=reflection_payload(records,lessons);expanded=expand_reflection_payload(payload)
    assert expanded['lessons'][0]['authoring_facts']==card['authoring_facts']
    assert 'sciel-lesson-authoring/1' in serialize_reflection(records,lessons)
    assert presentation_identity()['lesson_authoring']==authoring.SCHEMA


def live_controller(responses):
    class Fake:
        models=['fixture'];max_retries=1
        def __init__(self):self.prompts=[]
        def generate(self,prompt,model,temperature):
            self.prompts.append(prompt)
            return LLMResponse(next(responses),model,.001,10,10)
    model=Fake();events=[]
    cfg=apply_defaults({'research':{'enabled':True,'model':'fixture','max_aux_calls':2,'reflection_interval':1}})['research']
    c=ResearchController(cfg,model,SimpleNamespace(research=lambda **x:events.append(x)),lambda _:None)
    r=record();r.update(metric=c.metric,parent_cases={},evaluation=dict(ok=True,score=.75,cases={},reached='full',confirmed=False,raw_first_score=.75))
    c.knowledge.record(r);c.attempts=c.outcomes=c.valid_hypotheses=1;c.queue=deepcopy(c.knowledge.experiments)
    return c,model,events


def review():
    return dict(verdict='REFUTED',explanation='The measured score decreased.',lessons=[proposal(' First live wording. ')],
        mechanism_review=[dict(claim_quote='This change will improve score.',source_quotes=[],assessment='unknown',
                               direct_effect='Source behavior is unverified.',untested_downstream='Not established.')])


def test_production_conclusion_reflection_stamps_raw_responses_and_restore_identity():
    first=json.dumps(review());second=json.dumps(dict(lessons=[proposal('Consolidated live wording.')],retire_ids=['l000001']))
    c,model,_=live_controller(iter([first,second]));c.complete_step()
    l=c.knowledge.lessons[0];writers=nodes(l,'writer')
    assert {(n['phase'],n['raw_response_sha256']) for n in writers}=={('conclusion',digest(first)),('reflection',digest(second))}
    assert c.outcomes==1 and c.aux_calls==2 and len(c.knowledge.experiments)==1
    assert 'authoring_facts' in model.prompts[1]
    state=c.state();fresh=ResearchController(c.config,c.llm,SimpleNamespace(research=lambda **_:None),lambda _:None,metric=c.metric)
    fresh.load_state(state);assert fresh.state()==state
    damaged=deepcopy(state);damaged['presentation'].pop('lesson_authoring')
    with pytest.raises(ValueError,match='presentation mismatch'):fresh.load_state(damaged)
    assert fresh.state()==state


def test_invalid_review_is_never_writer_of_an_existing_lesson():
    c,_,_=live_controller(iter(['not a conclusion']));c.config['reflection_interval']=0
    add(c.knowledge,'Legacy rule.');before=c.knowledge.state()
    c.complete_step()
    assert c.conclusions['r000001a1']['status']=='invalid' and c.knowledge.state()==before


def test_circle_identity_pins_authoring_module_drift_without_touching_disk(monkeypatch):
    from pathlib import Path
    from sciel.engine.circle_generation import _integration_identity, CircleGeneration
    before=_integration_identity();pair_before=CircleGeneration._pair_identity()
    key='knowledge_base/authoring.py';assert key in before and key in pair_before['integration_sha256']
    original=Path.read_bytes
    def changed(path):
        data=original(path)
        return data+b'\n# fixture drift' if str(path).endswith('/'+key) else data
    monkeypatch.setattr(Path,'read_bytes',changed)
    assert _integration_identity()[key]!=before[key]
    assert CircleGeneration._pair_identity()['integration_sha256'][key]!=pair_before['integration_sha256'][key]


def test_bad_writer_reference_and_conflicting_same_origin_are_atomic_at_admission():
    k=kb();before=k.state();context=writer();context['reference']='r000999a1'
    with pytest.raises(ValueError,match='writer observation reference'):add(k,context=context)
    assert k.state()==before
    add(k,' Original. ',writer());before=k.state()
    with pytest.raises(ValueError,match='conflicting writer'):add(k,'Original.',writer())
    assert k.state()==before


def test_literal_legacy_ancestry_counts_toward_whole_reflection_card_budget():
    k=kb();old=add(k,'Legacy source wording. '+('Detailed condition. '*40).strip())
    rewrite(k,old,'Current concise wording.')
    before=k.state();records,lessons=k.reflection_context(1)
    payload=reflection_payload(records,lessons)
    ancestor=next(node for node in payload['lessons'][0]['authoring_facts']['provenance']['catalog'].values()
                  if node['kind']=='retired_lesson')
    assert ancestor['rule'].startswith('Legacy source wording.') and ancestor['prior_writers_unknown'] is True
    full_size=len(serialize_reflection(records,lessons))
    bare_size=len(serialize_reflection(records,[]))
    assert full_size>bare_size+len(ancestor['rule'])
    limited_records,limited_lessons=k.reflection_context(1,max_chars=full_size-1)
    assert limited_records==records and limited_lessons==[] and k.state()==before
    assert k.reflection_context(1,max_chars=full_size)==(records,lessons)


def test_conflicting_writer_across_live_lessons_rejects_atomically():
    k=kb();add(k,'First rule.',writer());before=k.state()
    with pytest.raises(ValueError,match='across lessons'):add(k,'Different rule.',writer())
    assert k.state()==before


@pytest.mark.parametrize('damage',['writer_origin','retired_is_current','retired_conflict'])
def test_cross_catalog_restore_rejects_locally_valid_but_inconsistent_nodes(damage):
    k=kb()
    if damage=='writer_origin':
        add(k,'First.',writer());add(k,'Second.',writer('b'))
    else:
        first=add(k,'First.',writer());rewrite(k,first,'Second.','b')
        third=add(k,'Third.',writer('c'))
        if damage=='retired_conflict':rewrite(k,third,'Fourth.','d')
    saved=k.state();state=deepcopy(saved)
    lesson=state['lessons'][1] if damage in ('writer_origin','retired_conflict') else state['lessons'][0]
    data=lesson['authoring'];key='writer_refs' if damage=='writer_origin' else 'revision_refs'
    old=data[key][0];node=data['catalog'].pop(old)
    if damage=='writer_origin':node['raw_response_sha256']='a'*64
    else:node['lesson_id']=third if damage=='retired_is_current' else first
    new=canonical_sha(node);data['catalog'][new]=node;data[key]=[new]
    # The counterexample passes each catalog's local bindings and hashes.
    authoring.validate(data,rule=lesson['rule'],lesson_id=lesson['id'],
                       evidence={e['id']:e for e in lesson['evidence_snapshots']},
                       next_lesson_id=state['next_lesson_id'],polarity=lesson['polarity'])
    with pytest.raises(ValueError,match='across lessons|still current|across catalogs'):k.load_state(state)
    assert k.state()==saved


def test_shared_identical_retired_ancestry_is_consistent_without_aliases():
    k=kb();old=add(k,'Legacy.');rewrite(k,old,'First current.','a')
    add(k,'Second current.',writer('b'));state=k.state()
    first,second=state['lessons'];old_ref=first['authoring']['revision_refs'][0]
    second['authoring']['revision_refs']=[old_ref]
    second['authoring']['catalog'][old_ref]=deepcopy(first['authoring']['catalog'][old_ref])
    k.load_state(state);assert k.state()==state
    second['authoring']['catalog'][old_ref]['rule']='Changed external alias.'
    assert k.state()!=state and nodes(k.lessons[0],'retired_lesson')[0]['rule']=='Legacy.'


def test_retired_literal_must_match_existing_inherited_exposure_wording():
    k=kb();old=add(k,'Legacy rule.');exposure=record('r000002a1');exposure['injected_ids']=[old]
    k.record(exposure);rewrite(k,old,'Current rule.');saved=k.state();state=deepcopy(saved)
    lesson=state['lessons'][0];data=lesson['authoring'];ref=data['revision_refs'][0]
    assert lesson['inherited_exposures'][0]['from_rule']=='Legacy rule.'
    node=data['catalog'].pop(ref);node['rule']='Conflicting historical text.'
    node['rule_sha256']=authoring.text_sha(node['rule']);new=canonical_sha(node)
    data['catalog'][new]=node;data['revision_refs']=[new]
    authoring.validate(data,rule=lesson['rule'],lesson_id=lesson['id'])
    with pytest.raises(ValueError,match='inherited exposure origin'):k.load_state(state)
    assert k.state()==saved
