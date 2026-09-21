"""Scoped review targets use authored immutable sources and inert SDK replies.

No model, optimizer or candidate executes; local ledgers exercise accounting.
"""
import copy
from dataclasses import asdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

import pytest

from sciel.engine import component_target_generation as generation
from sciel.hypothesis import circle_optimizer_traced_body as builder
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
from sciel.knowledge_base.generated_paired_context import expand_context
from test_component_generation_hypothesis import payload
from test_generated_paired_integration import pilot_artifacts, component
from test_scoped_paired_reflection import saved
from test_generated_paired_review_delivery import policy_for


TRANSPORT=dict(reasoning={'effort':'none'},provider=dict(allow_fallbacks=False,
    require_parameters=True,max_price=dict(prompt=3.0,completion=15.0,request=0.0)))


@pytest.fixture(scope='module',autouse=True)
def no_execution():
    def forbidden(*_args,**_kwargs):pytest.fail('Pure target generation attempted execution or real transport.')
    with pytest.MonkeyPatch.context() as patch:
        for name in ('sciel.eval.repeat_draw.draw','sciel.eval.repeat_draw.adapter_identity',
                     'sciel.engine.llm.client.EnsembleLLM.generate','subprocess.Popen'):
            patch.setattr(name,forbidden)
        yield


def make_job(saved, *, cfg=None, sidecar=None, card_sha=None, task='Construct a packing for 26 circles in the unit square.'):
    sidecar=saved.sidecar if sidecar is None else sidecar
    config=copy.deepcopy(saved.config)
    config.update(external_component_review_sha256=canonical_sha(sidecar),
                  comparison_review_text_policy=policy_for(sidecar,'accepted'))
    if cfg is not None:config.update(cfg)
    card=next(iter(sidecar['review_cards'].values()))
    return generation.ComponentTargetGenerationJob(saved.component,sidecar,
        expected_component_sha256=canonical_sha(saved.component),expected_sidecar_sha256=canonical_sha(sidecar),
        comparison_id=card['comparison_id'],expected_card_sha256=card_sha or canonical_sha(card),
        config=config,task_prompt=task),config


@pytest.fixture(scope='module')
def prepared(saved):
    job,config=make_job(saved)
    plan=job.prepare(run_id='synthetic-component-target-G',segment_id='synthetic-G',transport_settings=TRANSPORT)
    base=plan['source_registry'][plan['target']['base_source_sha256']]['body']
    # The authored OFF helper has this gate; the synthetic R names exactly this edit.
    assert 'restart_index % 8' in base
    body=base.replace('restart_index % 8','restart_index % 16',1)
    value=payload(body)
    value['target_sha256']=plan['target_sha256']
    value['hypothesis']['paired_forecast']['target_sha256']=plan['target_sha256']
    value['hypothesis']['text']='SYNTHETIC G: reduce OFF activation to every 16th restart.'
    return NS(job=job,config=config,plan=plan,value=value,saved=saved)


def response_for(value,model):
    return dict(text=json.dumps(value),model=model,cost=.001,prompt_tokens=20,completion_tokens=30,
                finish_reason='stop',refusal=None)


def provider_for(response,ident='synthetic-G-provider'):
    return dict(id=ident,model=response.get('model'),choices=[dict(index=0,finish_reason=response.get('finish_reason'),
        message=dict(role='assistant',content=response.get('text'),refusal=response.get('refusal')))],
        usage={key:response.get(key) for key in ('cost','prompt_tokens','completion_tokens')})


def admit(prepared, *, value=None, response=None, provider=None, request=None, plan=None, prefix=None):
    plan=prepared.plan if plan is None else plan
    response=response_for(prepared.value if value is None else value,plan['request']['model']) if response is None else response
    return prepared.job.admit(plan,response,manifest_sha256='d'*64,
        journal_prefix=plan['journal_prefix'] if prefix is None else prefix,
        provider_request=plan['request'] if request is None else request,
        provider_response=provider_for(response) if provider is None else provider)


def test_declared_OFF_base_control_comparator_full_prompt_and_unmeasured_result(prepared):
    p=prepared.plan;before=canonical_sha(prepared.saved.component)
    result=admit(prepared)
    assert result['status']=='eligible' and result['eligible'] is True
    target=p['target'];assert target['base_source_sha256']!=target['comparator_source_sha256']
    assert target['comparator_source_sha256']==prepared.saved.component['root_parent']['source_sha256']
    assert result['proof']['parent_source_sha256']==target['base_source_sha256']
    assert result['proof']['child_source_sha256']==result['source_sha256']
    assert result['code']==builder.apply(p['source_registry'][target['base_source_sha256']]['source'],result['body'])['code']
    assert result['proof']['unchanged_outside_helper'] is True
    context,_=json.JSONDecoder().raw_decode(p['prompt'].split('COMPONENT_TARGET_CONTEXT_JSON:\n',1)[1])
    for role,key in [('edit_base','base_source_sha256'),('paired_comparator','comparator_source_sha256')]:
        delivered=context['source_bodies'][role]
        assert delivered['body']==p['source_registry'][target[key]]['body']
        assert builder.source_for(delivered['body'])==p['source_registry'][target[key]]['source']
    packed,_=json.JSONDecoder().raw_decode(p['prompt'].split('GENERATED_PAIRED_HISTORY_JSON:\n',1)[1])
    expanded=expand_context(packed)
    assert packed['presented_count']==1 and 'REFUTED' in canonical_json(expanded) and 'INCONCLUSIVE' in canonical_json(expanded)
    assert len(prepared.saved.component['comparisons'])==3
    assert context['selected_review']==next(iter(prepared.saved.sidecar['review_cards'].values()))
    assert 'parent_score' not in context and 'parent_cases' not in context
    assert result['native_observation']['status']=='UNMEASURED' and result['native_observation']['score'] is None
    assert result['paired_observation']['status']=='UNMEASURED'
    assert result['new_ordinary_outcomes']==result['new_search_attempts']==0
    assert result['provider_dispatch_verified'] is result['execution_performed'] is False
    assert result['new_lesson_ids']==result['retired_lesson_ids']==[]
    assert canonical_sha(prepared.saved.component)==before
    assert result==admit(prepared)  # Pure re-admission has no dispatch side effect.
    assert generation.validate_result(result,component=prepared.saved.component,review_sidecar=prepared.saved.sidecar,
        expected_sha256=canonical_sha(result),config=prepared.config)==result


@pytest.mark.parametrize('status',['pending','rejected','stale'])
def test_fresh_host_acceptance_is_required_even_for_authentic_saved_card(saved,status):
    with pytest.raises(ValueError,match='host acceptance'):
        make_job(saved,cfg={'comparison_review_text_policy':policy_for(saved.sidecar,status)})


@pytest.mark.parametrize('cfg',[{'comparison_review_text':False},{'generated_paired_review_version':1},
    {'generated_paired_review_version':True},{'external_component_sha256':'0'*64},
    {'external_history_sha256':'0'*64},{'batch_only_lessons':False}])
def test_no_legacy_downgrade_hidden_text_or_other_history(saved,cfg):
    with pytest.raises(ValueError):make_job(saved,cfg=cfg)


def test_raw_R_and_host_card_cannot_be_replaced_by_rehashing(saved):
    forged=copy.deepcopy(saved.sidecar)
    card=next(iter(forged['review_cards'].values()))
    card['unverified_interpretation']['next_hypothesis']['change']='Fabricated target'
    with pytest.raises(ValueError):make_job(saved,sidecar=forged)
    with pytest.raises(ValueError,match='exact selected'):
        make_job(saved,card_sha='f'*64)


def test_no_target_does_not_get_invented_from_descriptive_review(saved):
    proposal=copy.deepcopy(saved.proposal);proposal['next_hypothesis']=None
    raw=dict(lessons=[],retire_ids=[],comparison_reviews=[proposal])
    returned=response_for(raw,saved.config['model'])
    sidecar=saved.job.admit(saved.plan,returned,manifest_sha256='e'*64,
                           journal_prefix=saved.plan['journal_prefix'])['extended_capsule']
    with pytest.raises(ValueError,match='no prospective target'):make_job(saved,sidecar=sidecar)


@pytest.mark.parametrize('mutation',['prompt','target','base','journal','temperature','max_tokens','routing'])
def test_changed_plan_and_prefix_reject_before_response(prepared,mutation):
    plan=copy.deepcopy(prepared.plan)
    if mutation=='prompt':plan['prompt']+='change'
    elif mutation=='target':plan['target']['change']='change'
    elif mutation=='base':plan['source_registry'][plan['target']['base_source_sha256']]['source']+='\n'
    elif mutation=='journal':plan['journal_prefix']+='{}\n'
    elif mutation=='temperature':plan['request']['temperature']=.3
    elif mutation=='max_tokens':plan['max_tokens']=True
    else:plan['transport_settings']['provider']['allow_fallbacks']=True
    with pytest.raises(ValueError):admit(prepared,plan=plan)
    with pytest.raises(ValueError,match='prefix'):admit(prepared,prefix=prepared.plan['journal_prefix']+'{}\n')


def test_abstention_invalid_bodies_and_duplicates_retain_raw_without_phantom_candidate(prepared):
    cases=[]
    value=copy.deepcopy(prepared.value);value.update(body=None,hypothesis=None,abstention_reason='Synthetic explicit abstention.')
    cases.append((value,'abstained'))
    value=copy.deepcopy(prepared.value);value['body']='import os\nreturn None\n';cases.append((value,'invalid_response'))
    value=copy.deepcopy(prepared.value);value['target_sha256']='a'*64;cases.append((value,'invalid_response'))
    for source in prepared.plan['source_registry'].values():
        value=copy.deepcopy(prepared.value);value['body']=source['body'];cases.append((value,'duplicate'))
    value=copy.deepcopy(prepared.value)
    value['body']=prepared.plan['source_registry'][prepared.plan['target']['base_source_sha256']]['body']+'\n'
    cases.append((value,'duplicate'))  # Exact bytes differ; normalized source is still historical.
    for value,status in cases:
        result=admit(prepared,value=value)
        assert result['status']==status and not result['eligible']
        assert result['code'] is result['source_sha256'] is result['proof'] is None
        rows=[json.loads(line) for line in result['journal'].splitlines()]
        assert rows[-2]['response']==result['response']['text'] and rows[-1]['status']==status


@pytest.mark.parametrize('damage',['missing_cost','cost_bool','cost_huge','tokens_bool','finish','refusal','raw',
    'provider_content','provider_cost','provider_cost_bool','provider_tokens_bool',
    'provider_id_replay','request_prompt','request_schema','request_rate'])
def test_invalid_sdk_metadata_and_transport_binding_fail_closed_with_raw_journal(prepared,damage):
    response=response_for(prepared.value,prepared.plan['request']['model'])
    provider=provider_for(response);request=copy.deepcopy(prepared.plan['request'])
    if damage=='missing_cost':response.pop('cost')
    elif damage=='cost_bool':response['cost']=True
    elif damage=='cost_huge':response['cost']=10**1000
    elif damage=='tokens_bool':response['prompt_tokens']=True
    elif damage=='finish':response['finish_reason']='length'
    elif damage=='refusal':response['refusal']='Synthetic refusal'
    elif damage=='raw':response['text']='```json\n'+response['text']+'\n```';provider=provider_for(response)
    elif damage=='provider_content':provider['choices'][0]['message']['content']='{}'
    elif damage=='provider_cost':provider['usage']['cost']=.002
    elif damage=='provider_cost_bool':response['cost']=1.0;provider['usage']['cost']=True
    elif damage=='provider_tokens_bool':response['prompt_tokens']=1;provider['usage']['prompt_tokens']=True
    elif damage=='provider_id_replay':provider['id']=prepared.saved.component['proposals'][0]['provider_response']['id']
    elif damage=='request_prompt':request['messages'][0]['content']+='tampered'
    elif damage=='request_schema':request['response_format']['json_schema']['name']='other'
    else:request['extra_body']['provider']['max_price']['request']=1
    result=admit(prepared,response=response,provider=provider,request=request)
    assert result['status']=='invalid_response' and result['code'] is result['proof'] is None
    rows=[json.loads(line) for line in result['journal'].splitlines()]
    assert rows[-2]['response']==response['text'] and result['error'] is not None


def test_repair_or_graft_cannot_silently_change_the_candidate(prepared,monkeypatch):
    def repair(code,**_kwargs):return NS(code=code+'\n',grafted=['run_packing'],violation=None)
    monkeypatch.setattr(generation,'validate_candidate',repair)
    result=admit(prepared)
    assert result['status']=='validation_rejected' and result['code'] is result['proof'] is None


def test_invalid_missing_or_oversized_capture_keeps_bounded_terminal_journal(prepared):
    missing=admit(prepared,response={'text':'retained raw with missing SDK fields'})
    assert missing['status']=='invalid_response' and missing['code'] is None
    assert json.loads(missing['journal'].splitlines()[-2])['response']=='retained raw with missing SDK fields'
    response=response_for(prepared.value,prepared.plan['request']['model'])
    response['text']='x'*(generation.MAX_CAPTURE_BYTES+1)
    with pytest.raises(generation.ComponentTargetGenerationRejected) as captured:
        admit(prepared,response=response)
    rows=[json.loads(line) for line in captured.value.journal.splitlines()]
    assert rows[-1]['phase']=='invalid_capture' and rows[-1]['response_truncated'] is True
    assert len(rows[-1]['response'])==32768 and len(captured.value.journal)<40000
    assert rows[-1]['response_sha256']==hashlib.sha256(response['text'].encode()).hexdigest()


def test_preparation_cannot_silently_use_changed_input_or_reused_historical_run(prepared):
    with pytest.raises(ValueError,match='fresh'):
        prepared.job.prepare(run_id=prepared.saved.component['proposals'][0]['provenance']['run_id'],
            segment_id='synthetic-G',transport_settings=TRANSPORT)
    original=copy.deepcopy(prepared.job._config)
    try:
        prepared.job._config['comparison_review_text_policy']['decisions']=[]
        with pytest.raises(ValueError,match='frozen input'):
            prepared.job.prepare(run_id='synthetic-fresh',segment_id='synthetic-G',transport_settings=TRANSPORT)
    finally:
        prepared.job._config=original


def test_generation_identity_namespaces_same_raw_by_manifest_run_and_target(prepared):
    original=admit(prepared)
    other=prepared.job.prepare(run_id='synthetic-second-independent-stage',segment_id='synthetic-G',transport_settings=TRANSPORT)
    result=admit(prepared,plan=other)
    assert original['generation_id']!=result['generation_id']
    assert original['source_sha256']==result['source_sha256']
    assert 'r000003a1' not in result['generation_id']


def test_rehashed_result_cannot_graft_source_measurement_or_journal(prepared):
    result=admit(prepared)
    for damage in ('code','proof','journal','native','approval'):
        forged=copy.deepcopy(result)
        if damage=='code':forged['code']+='\n# forged\n'
        elif damage=='proof':forged['proof']['parent_source_sha256']=prepared.plan['target']['comparator_source_sha256']
        elif damage=='journal':forged['journal']+='{}\n';forged['journal_sha256']=hashlib.sha256(forged['journal'].encode()).hexdigest()
        elif damage=='native':forged['native_observation']['score']=2.6
        else:forged['config']['comparison_review_text_policy']['decisions'][0]['audit_sha256']='e'*64
        with pytest.raises(ValueError):generation.validate_result(forged,component=prepared.saved.component,
            review_sidecar=prepared.saved.sidecar,expected_sha256=canonical_sha(forged),config=prepared.config)


@pytest.mark.parametrize('prices,cost',[
    (dict(prompt=3.0,completion=15.0,request=0.0),.001),
    (dict(prompt=3,completion=15,request=0),0),
    (dict(prompt=0,completion=0,request=0),0)])
def test_real_BudgetedLLM_request_builder_matches_frozen_transport_without_network(prepared,prices,cost):
    from sciel.campaign.budget import BudgetedLLM
    transport=copy.deepcopy(TRANSPORT);transport['provider']['max_price']=prices
    plan=prepared.job.prepare(run_id=prepared.plan['run_id'],segment_id=prepared.plan['segment_id'],transport_settings=transport)
    assert all(type(v) is float for v in plan['request']['extra_body']['provider']['max_price'].values())
    assert transport['provider']['max_price']==prices  # No caller-input mutation.
    response=response_for(prepared.value,plan['request']['model']);response['cost']=cost;provider=provider_for(response)
    choice=NS(index=0,finish_reason='stop',message=NS(content=response['text'],refusal=None))
    client=Mock();client.chat.completions.create.return_value=NS(id=provider['id'],choices=[choice],
        usage=NS(cost=response['cost'],prompt_tokens=response['prompt_tokens'],completion_tokens=response['completion_tokens']))
    # Exercise the real request construction/return codec with an inert ledger and SDK.
    llm=BudgetedLLM.__new__(BudgetedLLM)
    llm.models=[response['model']];llm.prices={response['model']:{key:str(Decimal(str(value))/(1000000 if key!='request' else 1))
        for key,value in prices.items()}}
    llm.temperature=.2;llm.max_tokens=4096;llm.run_id=prepared.plan['run_id'];llm.client=client
    llm.ledger=NS(reserve=Mock(return_value='synthetic-only'),settle=Mock(return_value=None))
    returned=llm.generate(plan['prompt'],response['model'],response_format=plan['response_format'])
    actual=client.chat.completions.create.call_args.kwargs
    captured=asdict(returned)
    assert actual==plan['request']
    assert captured['response_error'] is None
    assert captured['cost_provenance']==dict(schema='sciel-llm-cost-provenance/1',
        source='provider_reported',reported_cost=float(cost))
    # The historical admission wire has seven fields; diagnostics remain separate,
    # as in the production evaluated-target runner's explicit receipt projection.
    receipt={key:captured[key] for key in
        ('text','model','cost','prompt_tokens','completion_tokens','finish_reason','refusal')}
    assert receipt==response
    result=admit(prepared,plan=plan,response=receipt,request=actual,provider=provider)
    assert result['eligible'] is True
    assert type(result['response']['cost']) is float
    assert type(result['provider_response']['usage']['cost']) is type(cost)
    assert client.chat.completions.create.call_count==1


def test_provider_HTTP400_keeps_unknown_charge_and_cannot_supply_a_review(saved, tmp_path):
    from sciel.campaign.budget import CampaignStopped
    from test_campaign_budget import llm
    wrapped, client, ledger = llm(tmp_path)
    class RejectedRequest(Exception):
        status_code = 400
    client.chat.completions.create.side_effect = RejectedRequest('Authored rejected request')
    with pytest.raises(CampaignStopped, match='unknown'):
        wrapped.generate('Authored reflection request', 'm')
    state = ledger.snapshot()
    assert state['calls'] == 1 and state['halted']
    assert Decimal(state['known_cost_usd']) == 0
    assert Decimal(state['reserved_or_unknown_usd']) > 0
    with pytest.raises(CampaignStopped):
        wrapped.generate('No automatic followup', 'm')
    assert client.chat.completions.create.call_count == 1
    error_receipt = {'error_status_code': 400, 'review_cards': {}}
    card = next(iter(saved.sidecar['review_cards'].values()))
    config = dict(saved.config, external_component_review_sha256=canonical_sha(error_receipt),
                  comparison_review_text_policy=policy_for(saved.sidecar, 'accepted'))
    with pytest.raises(ValueError):
        generation.ComponentTargetGenerationJob(saved.component, error_receipt,
            expected_component_sha256=canonical_sha(saved.component),
            expected_sidecar_sha256=canonical_sha(error_receipt),
            comparison_id=card['comparison_id'], expected_card_sha256=canonical_sha(card),
            config=config, task_prompt='Authored task')
    assert ledger.snapshot() == state
