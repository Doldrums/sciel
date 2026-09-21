"""Literal bindings for model reviews, with no candidate or provider execution."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from sciel.config import apply_defaults
from sciel.engine.llm.client import LLMResponse
from sciel.engine.research import ResearchController
from sciel.hypothesis.conclusion_feedback import conclusion_feedback, validate_conclusions
from sciel.hypothesis.mechanism_review import (MAX_REVIEW_CHARS, SCHEMA, source_fields,
                                               validate_binding, validate_review)
from sciel.hypothesis.models import Conclusion
from sciel.hypothesis.parsing import parse_conclusion
from sciel.hypothesis.prompts import conclusion_prompt
from sciel.hypothesis.rendering import decode_evidence, presentation_identity, serialize_context
from sciel.knowledge_base.catalogs import canonical_json

CLAIM = 'Adding a step will improve the result.'
SOURCE = 'value = base + step\n'


def entry(**changes):
    result = dict(claim_quote=CLAIM, source_quotes=[dict(field='change_excerpt', text=SOURCE)],
                  assessment='directly_supported', direct_effect='The expression adds step to base.',
                  untested_downstream='Its effect on subsequent search and quality is unknown.')
    result.update(changes)
    return result


def payload(**changes):
    result = dict(verdict='REFUTED', explanation='The measured result regressed outside the forecast.',
                  lessons=[], mechanism_review=[entry()])
    result.update(changes)
    return result


def wire(value): return json.dumps(value, ensure_ascii=False)


def evidence():
    return dict(hypothesis=dict(text=CLAIM, reasoning='A smaller residual might help.'),
                change_excerpt=SOURCE, code_fragment='SECRET_UNSHOWN_SOURCE')


def controller(response=None):
    class Fake:
        models, max_retries = ['fixture'], 1
        def __init__(self): self.calls=[]
        def generate(self, prompt, model, temperature):
            self.calls.append(prompt)
            return LLMResponse(text=wire(payload() if response is None else response), model=model,
                               cost=.001, prompt_tokens=10, completion_tokens=20)
    llm=Fake(); events=[]; charges=[]
    config=apply_defaults({'research':{'enabled':True,'model':'fixture','max_aux_calls':1,'reflection_interval':0}})['research']
    c=ResearchController(config,llm,SimpleNamespace(research=lambda **x:events.append(x)),charges.append)
    source=hashlib.sha256(SOURCE.encode()).hexdigest()
    c.knowledge.record(dict(id='r000001a1',branch='p1',status='evaluated',parent_score=1.,score=.75,actual_delta=-.25,
        metric=c.metric,parent_cases={},evaluation=dict(ok=True,score=.75,cases={},reached='full',confirmed=False,raw_first_score=.75),
        source_sha256=source,proposed_sha256=source,parent_source_sha256='a'*64,change_excerpt=SOURCE,
        hypothesis=dict(text=CLAIM,predicted_low=.1,predicted_high=.2,reasoning='A smaller residual might help.',confidence=.7)))
    c.attempts=c.outcomes=c.valid_hypotheses=1
    c.queue=deepcopy(c.knowledge.experiments)
    return c,llm,events,charges


def test_legacy_serialization_is_exact_and_live_shape_is_required_explicitly():
    legacy=dict(verdict='REFUTED',explanation='Old review.',lessons=[])
    assert Conclusion(**legacy).to_dict()==legacy
    assert parse_conclusion(wire(legacy)).to_dict()==legacy
    assert parse_conclusion(wire(legacy),require_mechanism=True) is None
    assert parse_conclusion(wire({**legacy,'mechanism_review':None})) is None
    new=parse_conclusion(wire(payload()),require_mechanism=True)
    assert new.to_dict()==payload()
    detached=new.to_dict();detached['mechanism_review'][0]['source_quotes'][0]['text']='changed'
    assert new.to_dict()==payload()


@pytest.mark.parametrize('change',[
    {'mechanism_review':[]}, {'mechanism_review':[entry()]*5},
    {'mechanism_review':[entry(assessment='verified')]},
    {'mechanism_review':[entry(source_quotes=[])]},
    {'mechanism_review':[entry(source_quotes=[dict(field='change_excerpt',text=SOURCE)]*3)]},
    {'mechanism_review':[entry(source_quotes=[dict(field='code_fragment',text='secret')])]},
    {'mechanism_review':[entry(source_quotes=[dict(field='controlled_change.target_body',text=' ')])]},
    {'mechanism_review':[entry(source_quotes=[dict(field='change_excerpt',text=SOURCE,extra=True)])]},
    {'mechanism_review':[entry(claim_quote=None)]},
    {'mechanism_review':[entry(claim_quote='x'*513)]},
    {'mechanism_review':[entry(direct_effect='')]},
    {'mechanism_review':[entry(untested_downstream='x'*513)]},
    {'mechanism_review':[entry(extra='opaque')]},
])
def test_live_parser_rejects_malformed_typed_block_without_fallback(change):
    assert parse_conclusion(wire(payload(**change)),require_mechanism=True) is None


def test_atomic_duplicate_nonfinite_and_escaped_unicode_total_limits():
    raw=wire(payload())
    assert parse_conclusion(raw.replace('"assessment": "directly_supported"','"assessment": "unknown", "assessment": "directly_supported"')) is None
    assert parse_conclusion(raw.replace('"direct_effect": "The expression adds step to base."','"direct_effect": NaN')) is None
    large=[entry(claim_quote='界'*400,direct_effect='界'*400)]
    assert len(canonical_json(large))>MAX_REVIEW_CHARS
    with pytest.raises(ValueError,match='total character'):validate_review(large)
    assert parse_conclusion(wire(payload(mechanism_review=large))) is None


def test_exact_multiline_crlf_whitespace_unicode_quotes_are_preserved():
    claim='  Café e\u0301\r\n可能性  '
    source='\tvalue = (\r\n    "α" + "β"\r\n)\r\n'
    data=dict(hypothesis=dict(text=claim,reasoning='Other claim'),controlled_change={'target_body':source})
    review=[entry(claim_quote=claim,source_quotes=[dict(field='controlled_change.target_body',text=source)])]
    parsed=parse_conclusion(wire(payload(mechanism_review=review)),require_mechanism=True)
    assert parsed.mechanism_review==review and validate_binding(review,data)==review
    for wrong in (source.replace('\r\n','\n'),source.replace('    ','  ')):
        with pytest.raises(ValueError,match='source quote'):validate_binding([entry(claim_quote=claim,source_quotes=[dict(field='controlled_change.target_body',text=wrong)])],data)
    with pytest.raises(ValueError,match='claim quote'):
        validate_binding([entry(claim_quote=claim.replace('e\u0301','é'),source_quotes=review[0]['source_quotes'])],data)


@pytest.mark.parametrize('path',[
    ('controlled_change','parent_body'),('controlled_change','target_body'),
    ('controlled_change','before','text'),('controlled_change','after','text'),('change_excerpt',),
])
def test_each_declared_source_path_is_bound_without_aliasing(path):
    data=evidence();node=data
    for component in path[:-1]:node=node.setdefault(component,{})
    node[path[-1]]=SOURCE
    typed=[entry(source_quotes=[dict(field='.'.join(path),text=SOURCE)])]
    result=validate_binding(typed,data)
    assert result==typed
    result[0]['source_quotes'][0]['text']='changed'
    assert node[path[-1]]==SOURCE and typed[0]['source_quotes'][0]['text']==SOURCE


def test_no_cross_field_join_parent_as_child_or_opaque_lookalike_binding():
    data=evidence();data['controlled_change']={'parent_body':'old = value\n','target_body':'new = value\n'}
    data['opaque']={'controlled_change':{'target_body':'hidden'}}
    assert set(source_fields(data))=={'change_excerpt','controlled_change.parent_body','controlled_change.target_body'}
    for text in ('old = value\n','old = value\nnew = value\n','hidden'):
        with pytest.raises(ValueError,match='source quote'):
            validate_binding([entry(source_quotes=[dict(field='controlled_change.target_body',text=text)])],data)
    with pytest.raises(ValueError,match='field unavailable'):
        validate_binding([entry(source_quotes=[dict(field='controlled_change.before.text',text=SOURCE)])],data)
    with pytest.raises(ValueError,match='claim quote'):
        validate_binding([entry(claim_quote=CLAIM+'A smaller residual might help.')],data)


def test_unknown_with_no_source_and_null_claim_only_if_no_original_claim():
    unknown=[entry(source_quotes=[],assessment='unknown')]
    assert validate_binding(unknown,{'hypothesis':{'text':CLAIM}})==unknown
    empty=[entry(claim_quote=None,source_quotes=[],assessment='unknown')]
    assert validate_binding(empty,{'hypothesis':None})==empty
    assert validate_binding(empty,{'hypothesis':{'text':' ','reasoning':None}})==empty
    with pytest.raises(ValueError,match='claim quote'):validate_binding(empty,evidence())
    with pytest.raises(ValueError,match='claim unavailable'):validate_binding(unknown,{'hypothesis':None})


@pytest.mark.parametrize('failure',['missing_block','bad_claim','bad_source'])
def test_live_binding_rejection_precedes_lessons_keeps_paid_raw_and_original_forecast(failure):
    raw=payload(lessons=[dict(rule='This one source change regressed.',polarity='negative',scope='branch',branch='p1',evidence_ids=['r000001a1'])])
    if failure=='missing_block':del raw['mechanism_review']
    elif failure=='bad_claim':raw['mechanism_review'][0]['claim_quote']='invented original claim'
    else:raw['mechanism_review'][0]['source_quotes'][0]['text']='invented code'
    c,llm,events,charges=controller(raw);before=c.knowledge.state()
    c.complete_step()
    saved=c.conclusions['r000001a1']
    assert saved['status']=='invalid' and saved['llm'] is None and saved['observed_forecast_verdict']=='REFUTED'
    assert saved['accepted_lesson_ids']==[] and c.knowledge.state()==before
    assert c.aux_calls==1 and c.aux_cost==.001 and charges==[.001]
    result=next(e for e in events if e['phase']=='aux_result');assert result['response']==wire(raw)
    rejected=next(e for e in events if e['phase']=='conclusion_rejected')
    assert rejected['reason']==('invalid_conclusion_structure' if failure=='missing_block' else 'mechanism_binding_invalid')
    assert events.index(result)<events.index(rejected)
    c.complete_step();assert len(llm.calls)==1


def test_actual_live_prompt_and_whole_feedback_preserve_false_assessment_as_unverified():
    # This is intentionally false: a real quote does not make a semantic proof.
    raw=payload(mechanism_review=[entry(direct_effect='This addition squares base and proves every result optimal.')])
    c,llm,events,charges=controller(raw);before=c.knowledge.state();c.complete_step()
    assert c.knowledge.state()==before and c.conclusions['r000001a1']['status']=='parsed'
    assert c.conclusions['r000001a1']['llm']['mechanism_review']==raw['mechanism_review']
    prompt=llm.calls[0]
    assert 'mechanism_review: 1 to 4 entries' in prompt and 'canonical ASCII-escaped JSON' in prompt
    assert 'same translation' not in prompt and 'circle' not in prompt.lower()
    shown=decode_evidence(json.loads(prompt.split('EVIDENCE_JSON:\n',1)[1]))
    assert 'code_fragment' not in shown and shown['change_excerpt']==SOURCE
    feedback=conclusion_feedback(c.conclusions,c.knowledge._evidence(),'p1')
    card=feedback['cards'][0]
    assert card['unverified_interpretation']['mechanism_review']==raw['mechanism_review']
    assert card['observed_forecast_verdict']=='REFUTED' and 'not semantic truth' in feedback['interpretation_limits']
    assert 'verified' not in card and 'verified' not in card['unverified_interpretation']
    assert decode_evidence(json.loads(serialize_context({'conclusion_feedback':feedback})))['conclusion_feedback']==feedback
    limited=conclusion_feedback(c.conclusions,c.knowledge._evidence(),'p1',max_chars=1200)
    assert limited['cards']==[] and limited['omitted_count']==1
    assert c.conclusions['r000001a1']['llm']['mechanism_review']==raw['mechanism_review']


def test_valid_structured_review_can_use_existing_lesson_admission_without_new_quality():
    raw=payload(lessons=[dict(rule='The one measured addition had a negative score delta.',polarity='negative',scope='branch',branch='p1',evidence_ids=['r000001a1'])])
    c,llm,events,charges=controller(raw);old=c.knowledge.experiments;c.complete_step()
    assert len(c.knowledge.lessons)==1 and c.conclusions['r000001a1']['accepted_lesson_ids']
    assert c.knowledge.experiments==old and c.attempts==c.outcomes==1


def test_restore_accepts_legacy_and_new_but_rechecks_every_present_binding_atomically():
    c,llm,events,charges=controller();c.complete_step();saved=c.state()
    target,_,_,_=controller();target.queue=[]
    target.load_state(saved);assert target.state()==saved
    baseline=target.state()
    for damage in ('claim','source','path'):
        altered=deepcopy(saved);item=altered['conclusions']['r000001a1']['llm']['mechanism_review'][0]
        if damage=='claim':item['claim_quote']='other hypothesis'
        elif damage=='source':item['source_quotes'][0]['text']='other source'
        else:item['source_quotes'][0]['field']='controlled_change.target_body'
        with pytest.raises(ValueError):target.load_state(altered)
        assert target.state()==baseline
    legacy=deepcopy(saved);del legacy['conclusions']['r000001a1']['llm']['mechanism_review']
    target.load_state(legacy);assert target.state()==legacy
    view=conclusion_feedback(target.conclusions,target.knowledge._evidence(),'p1')['cards'][0]
    assert 'mechanism_review' not in view['unverified_interpretation']
    bad=deepcopy(legacy);bad['presentation'].pop('mechanism_review')
    with pytest.raises(ValueError,match='presentation mismatch'):target.load_state(bad)
    assert target.state()==legacy and presentation_identity()['mechanism_review']==SCHEMA
    assert presentation_identity()['conclusion_feedback']=='sciel-conclusion-feedback/2'
