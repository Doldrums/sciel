"""One typed study R: recorded bytes only, never a provider or numeric run."""
import copy
import json

import pytest

from sciel.engine.evaluated_study_reflection import EvaluatedStudyReflectionJob, validate_sidecar, INSTRUCTION
from sciel.engine.external_reflection import ExternalReflectionRejected
from sciel.engine.llm.client import response_format_bytes
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
from sciel.hypothesis import circle_optimizer_traced_body as builder

TRANSPORT = dict(reasoning={'effort':'none'}, provider=dict(allow_fallbacks=False,
    require_parameters=True, max_price={'prompt':3.0,'completion':15.0,'request':0.0}))


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail('Pure study R cannot dispatch a provider, runtime, optimizer or candidate.')
    for name in ('sciel.eval.repeat_draw.draw','sciel.eval.repeat_draw.adapter_identity',
                 'sciel.engine.llm.client.EnsembleLLM.generate','sciel.campaign.budget.BudgetedLLM.generate'):
        monkeypatch.setattr(name,forbidden)


@pytest.fixture(scope='module')
def bundle():
    from evaluated_study_fixture import build
    artifact=build()
    return {k:artifact[k] for k in ('capsule','author_dependencies','author_authority')}


def config_for(bundle):
    return dict(external_evaluated_study_sha256=canonical_sha(bundle['capsule']),
        external_evaluated_study_authority_sha256=canonical_sha(bundle['author_authority']),
        model='anthropic/claude-sonnet-4.5',max_aux_calls=1,inject_lessons=True,
        restart_predictions=True,structured_auxiliary=True,comparison_reviews=True,
        comparison_review_version=3,batch_only_lessons=True,comparison_review_text=True,
        comparison_review_text_policy={'schema':'sciel-review-text-policy/1','decisions':[]})


def make_job(bundle,config=None):
    config=config_for(bundle) if config is None else config
    return EvaluatedStudyReflectionJob(bundle,expected_sha256=canonical_sha(bundle['capsule']),
        expected_authority_sha256=canonical_sha(bundle['author_authority']),config=config)


def reply(bundle, *, target=False):
    value=dict(claims=[dict(fact_id='quality:verdict',value='REFUTED'),
                       dict(fact_id='author:kind',value='HOST')],
        conjecture='This synthetic review makes no causal inference from the known paired loss.',
        limitations='The unissued third pair remains unknown; retained geometry does not cover omitted calls.',
        next_hypothesis_reason='This unit fixture has no independently justified new intervention to propose.',
        evidence_refs={key:['quality:verdict'] for key in
            ('conjecture','limitations','next_hypothesis_reason')},next_hypothesis=None)
    if target:
        value['next_hypothesis_reason']='Synthetic prospective change only; the observed loss does not predict its quality.'
        value['next_hypothesis']=dict(base_source_sha256=bundle['capsule']['author']['candidate_source_sha256'],
            comparator_source_sha256=bundle['capsule']['author']['comparator_source_sha256'],
            change='Replace one initializer branch in this synthetic prospective unit fixture.',
            diagnostics=[],motivating_evidence=['quality:verdict'])
    return value


def captures(plan, proposal, *, raw=None):
    text=canonical_json(proposal) if raw is None else raw
    response=dict(text=text,model=plan['model'],cost=.012,prompt_tokens=10,completion_tokens=20,
                  finish_reason='stop',refusal=None)
    provider=dict(id='inert-study-review-response',model=plan['model'],
        choices=[dict(index=0,finish_reason='stop',message=dict(role='assistant',content=text,refusal=None))],
        usage=dict(cost=.012,prompt_tokens=10,completion_tokens=20))
    return response,copy.deepcopy(plan['request']),provider


def plan_for(job):
    return job.prepare(run_id='new-inert-study-r',segment_id='unit',transport_settings=TRANSPORT,max_tokens=4096)


def admit(job,plan,values):
    response,request,provider=values
    return job.admit(plan,response,manifest_sha256='a'*64,journal_prefix=plan['journal_prefix'],
                     provider_request=request,provider_response=provider)


@pytest.fixture(scope='module')
def prepared(bundle):
    job=make_job(bundle)
    plan=plan_for(job)
    return job,plan


def test_preparation_has_static_small_wire_and_all_sources_without_native_work(bundle,prepared):
    from sciel.hypothesis import evaluated_study_review as wire
    from sciel.hypothesis.evaluated_study_targets import instruction
    job,plan=prepared
    assert plan['response_format']==wire.response_format()
    assert len(response_format_bytes(plan['response_format'])) <= 4096
    assert plan['request']['extra_body']==TRANSPORT
    assert plan['request']['messages']==[{'role':'user','content':plan['prompt']}]
    assert plan['request_sha256']==canonical_sha(plan['request'])
    assert INSTRUCTION in plan['prompt'] and instruction() in plan['prompt']
    for source in bundle['capsule']['source_registry'].values():
        # Each body is complete in the literal authoring JSON, not a clipped excerpt.
        assert json.dumps(builder.body_of(source),ensure_ascii=False)[1:-1] in plan['prompt']
    assert plan['new_ordinary_outcomes']==plan['new_search_attempts']==0
    assert plan_for(job)==plan
    assert 'without a previous gain' not in plan['prompt'] or 'test' in plan['prompt']


@pytest.mark.parametrize('target',[False,True])
def test_actual_study_accepts_synthetic_reasoned_decision_without_G_or_credit(bundle,prepared,target):
    job,plan=prepared
    proposal=reply(bundle,target=target)
    result=admit(job,plan,captures(plan,proposal))
    sidecar=result['extended_capsule']; card=sidecar['review_card']
    assert card['unverified_interpretation']==proposal
    assert (card['resolved_target'] is not None)==target
    assert result['new_ordinary_outcomes']==result['new_search_attempts']==0
    assert result['new_lesson_ids']==result['retired_lesson_ids']==[]
    assert result['generation_allowed'] is result['execution_performed'] is result['provider_dispatch_verified'] is False
    assert result['recorded_provider_responses']==1 and result['known_cost']==.012
    assert validate_sidecar(sidecar,bundle=bundle,expected_sha256=canonical_sha(sidecar),config=config_for(bundle))==sidecar
    assert result==admit(job,plan,captures(plan,proposal))  # Pure replay is not request authorization.


@pytest.mark.parametrize('damage',['request_price','request_prompt','request_wire','provider_text',
    'provider_model','provider_id','provider_cost','provider_tokens','provider_choices',
    'response_cost_bool','response_refusal','response_finish'])
def test_actual_transport_mismatch_is_journalled_and_rejected(bundle,prepared,damage):
    job,plan=prepared
    response,request,provider=captures(plan,reply(bundle))
    if damage=='request_price': request['extra_body']['provider']['max_price']['prompt']=30.0
    elif damage=='request_prompt': request['messages'][0]['content']+=' forged'
    elif damage=='request_wire': request['response_format']['json_schema']['name']='foreign'
    elif damage=='provider_text': provider['choices'][0]['message']['content']+=' '
    elif damage=='provider_model': provider['model']='different-model'
    elif damage=='provider_id': provider['id']=''
    elif damage=='provider_cost': provider['usage']['cost']=.013
    elif damage=='provider_tokens': provider['usage']['prompt_tokens']=True
    elif damage=='provider_choices': provider['choices'].append(copy.deepcopy(provider['choices'][0]))
    elif damage=='response_cost_bool': response['cost']=True
    elif damage=='response_refusal': response['refusal']='refused'
    else: response['finish_reason']='length'
    with pytest.raises(ExternalReflectionRejected) as exc:
        admit(job,plan,(response,request,provider))
    journal=[json.loads(row) for row in exc.value.journal.splitlines()]
    assert journal[-2]['phase']=='returned' and journal[-2]['response']==response['text']
    assert journal[-1]['phase']=='admission' and journal[-1]['status']=='invalid'
    assert journal[-1]['generation_allowed'] is False


@pytest.mark.parametrize('damage',['metadata_claim','wrong_fact','unknown_source','unbounded_reason',
                                  'extra_lesson','duplicate_json','empty'])
def test_invalid_model_response_preserves_raw_without_repair(bundle,prepared,damage):
    job,plan=prepared; proposal=reply(bundle,target=True); raw=None
    if damage=='metadata_claim': proposal['claims'][0]['fact_id']='comparison_id'
    elif damage=='wrong_fact': proposal['claims'][0]['value']='CONFIRMED'
    elif damage=='unknown_source': proposal['next_hypothesis']['base_source_sha256']='0'*64
    elif damage=='unbounded_reason': proposal['next_hypothesis_reason']='x'*1025
    elif damage=='extra_lesson': proposal['lessons']=[{'rule':'fake lesson'}]
    elif damage=='duplicate_json': raw='{"claims":[],"claims":[]}'
    else: raw=''
    response,request,provider=captures(plan,proposal,raw=raw)
    with pytest.raises(ExternalReflectionRejected) as exc:
        admit(job,plan,(response,request,provider))
    rows=[json.loads(row) for row in exc.value.journal.splitlines()]
    assert rows[-2]['response']==response['text'] and rows[-1]['status']=='invalid'
    assert rows[-1]['new_lesson_ids']==[]


@pytest.mark.parametrize('damage',['plan','prefix','config','card_reason','writer','compiled_target'])
def test_sidecar_cannot_rehash_its_way_past_external_config_or_raw_writer(bundle,prepared,damage):
    job,plan=prepared
    sidecar=admit(job,plan,captures(plan,reply(bundle,target=True)))['extended_capsule']
    if damage=='plan':
        sidecar['plan']['request']['messages'][0]['content']+=' forged'
        sidecar['plan']['request_sha256']=canonical_sha(sidecar['plan']['request'])
    elif damage=='prefix': sidecar['plan']['journal_prefix']+='\n'
    elif damage=='config': sidecar['config']['model']='forged-model'
    elif damage=='card_reason': sidecar['review_card']['unverified_interpretation']['next_hypothesis_reason']='Forged new reason.'
    elif damage=='writer': sidecar['review_card']['writer']['raw_response_sha256']='b'*64
    else: sidecar['review_card']['resolved_target']['change']='Different prospective source edit.'
    with pytest.raises(ValueError):
        validate_sidecar(sidecar,bundle=bundle,expected_sha256=canonical_sha(sidecar),config=config_for(bundle))


@pytest.mark.parametrize('damage',['wrong_authority','old_component','more_aux','unreviewed_policy'])
def test_R_requires_external_input_authority_and_explicit_review_only_config(bundle,damage):
    config=config_for(bundle)
    if damage=='wrong_authority': config['external_evaluated_study_authority_sha256']='0'*64
    elif damage=='old_component': config['external_component_sha256']='a'*64
    elif damage=='more_aux': config['max_aux_calls']=2
    else: del config['comparison_review_text_policy']
    with pytest.raises(ValueError): make_job(bundle,config)
