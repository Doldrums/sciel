"""Finite-probe controls with inert provider/evaluator fixtures; no paid calls."""
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sciel.campaign.budget import BudgetedLLM, CampaignStopped, RequestNotDispatched
from sciel.diagnostics import directive_probe as probe
from sciel.hypothesis.intervention_claim import SCHEMA as CLAIM_SCHEMA


def read(path):
    return json.loads(Path(path).read_text())


@pytest.fixture
def prepared(tmp_path, monkeypatch):
    root, history, output = (tmp_path/name for name in ('checkout','history','probe'))
    for directory in (root/'src', root/'configs', root/'tasks/oe/circle_packing', history):
        directory.mkdir(parents=True)
    (root/'src/fixture.py').write_text('# frozen implementation fixture\n')
    (root/'pyproject.toml').write_text('[project]\nname="fixture"\nversion="0"\n')
    config = dict(research=dict(enabled=True, compact_lesson_memory=True),
        llm=dict(models=[probe.MODEL],weights=[1],api_base='https://openrouter.ai/api/v1',max_tokens=32,max_retries=1))
    (root/'configs/circle_live_gemini.yaml').write_text(json.dumps(config))
    task = root/'tasks/oe/circle_packing'
    (task/'task.yaml').write_text(json.dumps(dict(suite='oe',name='circle_packing',origin='oe',
        primary_metric='combined_score',smoke_timeout=30,full_timeout=30,stochastic=False)))
    (task/'evaluator.py').write_text('# inert evaluator; draw_fn supplies receipts\n')
    (task/'initial_program.py').write_text('def run_packing():\n    return 0\n')
    programs = [dict(id='p'+str(i),code=f'def run_packing():\n    return {i}\n',score=.3+i/100,
                     cases={},island=i-1,origin='fixture') for i in range(1,4)]
    knowledge = dict(limits=dict(max_lessons=48,max_experiments=100,max_fragments=8), lessons=[])
    (history/'checkpoint.json').write_text(json.dumps(dict(research=dict(knowledge=knowledge),
        db=dict(programs=programs),iteration_no=99,eval_times={p['id']:1.5 for p in programs})))
    (history/'sciel_events.jsonl').write_text('{"event":"run_end","reason":"paused"}\n')

    class FrozenResearch:
        def __init__(self, config, *args, **kwargs):
            self.config = config
            self.knowledge = SimpleNamespace(load_state=lambda state:setattr(self,'loaded',deepcopy(state)))

        def context(self, db, parent, island, iteration):
            assert self.config['max_lessons'] == self.loaded['limits']['max_lessons']
            return dict(lesson_texts=['Frozen evidence for '+parent.id],injected_ids=['lesson-'+parent.id])

    monkeypatch.setattr(probe,'ROOT',root)
    monkeypatch.setattr(probe,'ResearchController',FrozenResearch)
    monkeypatch.setattr(probe,'runtime_identity',lambda:dict(python='fixture',dependencies={}))
    monkeypatch.setattr(probe,'source_identity',lambda root:dict(revision='fixture',dirty=False,
        files={'src/fixture.py':probe.file_sha(root/'src/fixture.py')},
        project_files={'pyproject.toml':probe.file_sha(root/'pyproject.toml')}))
    pricing = dict(models=[dict(id=probe.MODEL,pricing=dict(prompt='0.00000001',completion='0.00000002'))])
    manifest = probe.prepare(history,output,pricing,parent_ids=['p1','p2','p3'],root=root)
    return SimpleNamespace(root=root,history=history,output=output,manifest=manifest,pricing=pricing)


def generated(number, *, bad_claim=False, code=None):
    declaration = dict(schema=CLAIM_SCHEMA,parent_unit_ids=['run_packing'],
        changed_units=[dict(parent_unit_id='run_packing',proposed_name='run_packing',change='modify')],
        claim_type='quality',mechanism='Change a return constant.',speed_used_for_search=False,
        search_effort_link=None,limitations=['No causal effect is established by this declaration.'])
    if bad_claim:
        declaration['parent_unit_ids'] = ['nonexistent']
    hypothesis = dict(hypothesis='Changing the return value may improve the score.',
        predicted_low=-.1,predicted_high=.1,reasoning='Fixture interpretation.',confidence=.2)
    candidate = code if code is not None else f'def run_packing():\n    return {100+number}\n'
    return ('INTERVENTION_JSON:\n'+json.dumps(declaration)+'\nHYPOTHESIS_JSON:\n'+
            json.dumps(hypothesis)+'\n```python\n'+candidate+'```\n')


class Client:
    def __init__(self, behavior=None):
        self.requests = []
        self.behavior = behavior
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **request):
        self.requests.append(deepcopy(request))
        number = len(self.requests)
        result = self.behavior(number,request) if self.behavior else {}
        result = result or {}
        return SimpleNamespace(id='fixture-'+str(number),usage=SimpleNamespace(
            cost=result.get('cost',.00001),prompt_tokens=10,completion_tokens=10),
            choices=[SimpleNamespace(finish_reason=result.get('finish_reason','stop'),
                message=SimpleNamespace(content=result.get('text',generated(number)),refusal=None))])


def install_client(monkeypatch, client):
    # Exercise the actual reservation/settlement wrapper with an inert transport.
    monkeypatch.setattr(probe,'BudgetedLLM',lambda cfg,pricing,ledger,run_id,key:
                        BudgetedLLM(cfg,pricing,ledger,run_id,key,client=client))


class Draws:
    def __init__(self, hook=None):
        self.calls = []
        self.hook = hook

    def __call__(self, code, task, seed, **options):
        self.calls.append(dict(code=code,seed=seed,task=task,**options))
        if self.hook:
            replacement = self.hook(self.calls[-1],len(self.calls))
            if replacement is not None:
                return replacement
        return dict(ok=True,valid=True,score=.5,seed=seed,stage_invocations=1)


def test_prepare_has_exact_balanced_18_slots_and_no_history_mutation(prepared):
    original = {name:probe.file_sha(prepared.history/name) for name in ('checkpoint.json','sciel_events.jsonl')}
    manifest = probe.verify(prepared.output)
    assert len(manifest['schedule']) == manifest['max_calls'] == 18
    assert manifest['operator'] == 'diff'
    assert manifest['learning'] is manifest['conclusions'] is manifest['resume'] is False
    assert manifest['resamples'] == 0
    for parent in manifest['parent_ids']:
        for arm in probe.ARMS:
            assert sum(row['parent_id']==parent and row['arm']==arm for row in manifest['schedule']) == 2
    assert manifest['history_pins'] == original
    assert not (prepared.output/'STARTED.json').exists()
    assert not (prepared.output/'cost_ledger.json').exists()


def test_all_arm_prompts_share_parent_memory_baseline_and_operator(prepared):
    parent = read(prepared.output/'frozen/parents.json')[0]
    task = probe.load_task(str(prepared.output/'frozen/task'))
    prompts = {arm:probe.prompt_for(parent,dict(score=.5),task,arm) for arm in probe.ARMS}
    def without_directive(prompt):
        before,body = prompt.split('DIRECTIVE:\n',1)
        directive,after = body.split('\n\nLESSONS:',1)
        return before+'DIRECTIVE:\n<TREATMENT>\n\nLESSONS:'+after
    assert len({without_directive(text) for text in prompts.values()}) == 1
    assert 'REFINEMENT:' not in prompts['neutral'] and 'STAGNATION:' not in prompts['neutral']
    assert 'KEEP this construction' in prompts['refine']
    assert 'STRUCTURAL change' in prompts['escape']
    assert all('Follow the requested search scope in DIRECTIVE' in text for text in prompts.values())
    assert all('Frozen evidence for p1' in text for text in prompts.values())
    assert all('parent_score": 0.5' in text for text in prompts.values())
    assert prompts['refine'] == probe.prompt_for(parent,dict(score=.5),task,'refine')


def test_complete_probe_has_18_budgeted_calls_63_inert_draws_and_no_replay(prepared,monkeypatch):
    client,draws = Client(),Draws();install_client(monkeypatch,client)
    summary = probe.execute(prepared.output,api_key='inert-fixture',draw_fn=draws)
    assert summary['status'] == 'complete'
    assert len(client.requests) == 18 and len(draws.calls) == 63
    ledger = read(prepared.output/'cost_ledger.json')
    assert len(ledger['calls']) == 18
    assert sum(Decimal(row['cost_usd']) for row in ledger['calls'].values()) == Decimal('.00018')
    assert all(row['status']=='settled' for row in ledger['calls'].values())
    assert all(arm['planned']==arm['returned']==arm['evaluated']==6 for arm in summary['arms'].values())
    plan = read(prepared.output/'execution_plan.json')['cells']
    for parent in prepared.manifest['parent_ids']:
        cells = [cell for cell in plan if cell['parent_id']==parent]
        assert len({cell['memory_sha256'] for cell in cells}) == 1
        for arm in probe.ARMS:
            assert len({cell['prompt_sha256'] for cell in cells if cell['arm']==arm}) == 1
    with pytest.raises(ValueError,match='already started'):
        probe.execute(prepared.output,api_key='inert-fixture',draw_fn=draws)
    assert len(client.requests)==18 and len(draws.calls)==63


def test_invalid_claim_is_audited_but_valid_code_still_evaluated(prepared,monkeypatch):
    client=Client(lambda n,r:dict(text=generated(n,bad_claim=n==1)))
    draws=Draws();install_client(monkeypatch,client)
    probe.execute(prepared.output,api_key='inert-fixture',draw_fn=draws)
    receipt=read(prepared.output/'candidates/g01/receipt.json')
    assert receipt['status']=='evaluated' and len(receipt['draws'])==3
    assert receipt['declaration']['error']['code']=='unknown_parent_unit'
    assert receipt['declaration']['declared_targets'] is None
    audit=read(prepared.output/'candidates/g01/change_audit.json')
    assert audit['declaration']['available'] is False
    assert len(client.requests)==18


def test_invalid_python_is_a_failed_slot_not_end_of_diagnostic(prepared,monkeypatch):
    client=Client(lambda n,r:dict(text=generated(n,code='def run_packing(:\n' if n==1 else None)))
    draws=Draws();install_client(monkeypatch,client)
    result=probe.execute(prepared.output,api_key='inert-fixture',draw_fn=draws)
    receipt=read(prepared.output/'candidates/g01/receipt.json')
    assert receipt['status']=='invalid_candidate' and receipt['draws']==[]
    assert result['status']=='complete' and len(client.requests)==18 and len(draws.calls)==60
    audit=read(prepared.output/'candidates/g01/change_audit.json')
    assert audit['candidate']['parse_error'] is not None


def test_unknown_second_charge_stops_preserves_known_cost_and_forbids_replay(prepared,monkeypatch):
    def behavior(number,request):
        if number==2:
            raise TimeoutError('inert unknown transport')
    client=Client(behavior);draws=Draws();install_client(monkeypatch,client)
    with pytest.raises(CampaignStopped,match='unknown'):
        probe.execute(prepared.output,api_key='inert-fixture',draw_fn=draws)
    ledger=read(prepared.output/'cost_ledger.json');calls=list(ledger['calls'].values())
    assert len(calls)==2
    assert calls[0]['status']=='settled' and Decimal(calls[0]['cost_usd'])==Decimal('.00001')
    assert calls[1]['status']=='unknown' and calls[1]['cost_usd'] is None
    assert Decimal(calls[1]['reservation_usd'])>0 and ledger['halted']
    summary=read(prepared.output/'summary.json')
    assert summary['status']=='stopped'
    assert summary['started_cells']==['g01','g02']
    assert summary['without_returned_response']==['g02']
    with pytest.raises(ValueError,match='already started'):
        probe.execute(prepared.output,api_key='inert-fixture',draw_fn=draws)
    assert len(client.requests)==2


def test_missing_returned_charge_stops_without_evaluation_or_retry(prepared,monkeypatch):
    client=Client(lambda n,r:dict(cost=None));draws=Draws();install_client(monkeypatch,client)
    with pytest.raises(CampaignStopped,match='actual charge'):
        probe.execute(prepared.output,api_key='inert-fixture',draw_fn=draws)
    assert len(client.requests)==1 and len(draws.calls)==9
    call=next(iter(read(prepared.output/'cost_ledger.json')['calls'].values()))
    assert call['status']=='unknown' and call['cost_usd'] is None


def test_incomplete_return_is_charged_saved_and_not_retried(prepared,monkeypatch):
    client=Client(lambda n,r:dict(finish_reason='length') if n==1 else {})
    draws=Draws();install_client(monkeypatch,client)
    result=probe.execute(prepared.output,api_key='inert-fixture',draw_fn=draws)
    receipt=read(prepared.output/'candidates/g01/receipt.json')
    assert receipt['status']=='incomplete_response' and receipt['draws']==[]
    assert receipt['cost']==.00001
    assert result['status']=='complete' and len(client.requests)==18 and len(draws.calls)==60


def test_identical_cross_arm_sources_link_observations_without_claiming_new_draws(prepared,monkeypatch):
    client=Client(lambda n,r:dict(text=generated(1)))
    draws=Draws();install_client(monkeypatch,client)
    result=probe.execute(prepared.output,api_key='inert-fixture',draw_fn=draws)
    assert len(client.requests)==18 and len(draws.calls)==12
    assert sum(arm['evaluated'] for arm in result['arms'].values())==1
    assert sum(arm['duplicates'] for arm in result['arms'].values())==17
    original=read(prepared.output/'candidates/g01/receipt.json')
    later=read(prepared.output/'candidates/g02/receipt.json')
    assert later['status']=='duplicate' and later['draws']==[]
    assert later['shared_observation_id']=='g01'
    assert later['reused_draws']==original['draws']
    row=next(row for arm in result['arms'].values() for row in arm['candidates'] if row['id']=='g02')
    assert row['paired_native_score_deltas']==[] and row['successful_draws']==0
    assert len(row['paired_reused_reference_deltas'])==3


@pytest.mark.parametrize('target',['frozen','history','source'])
def test_changed_input_prevents_any_evaluation_or_provider_dispatch(prepared,monkeypatch,target):
    path={'frozen':prepared.output/'frozen/parents.json','history':prepared.history/'checkpoint.json',
          'source':prepared.root/'src/fixture.py'}[target]
    path.write_text(path.read_text()+'\n')
    client=Client();draws=Draws();install_client(monkeypatch,client)
    with pytest.raises(ValueError,match='identity drift'):
        probe.execute(prepared.output,api_key='inert-fixture',draw_fn=draws)
    assert not client.requests and not draws.calls
    assert not (prepared.output/'STARTED.json').exists()


@pytest.mark.parametrize('target',['frozen','source'])
def test_new_unpinned_input_file_prevents_dispatch(prepared,monkeypatch,target):
    path=(prepared.output/'frozen/task/new_asset.py' if target=='frozen' else
          prepared.root/'src/new_module.py')
    path.write_text('# unexpected input\n')
    client=Client();draws=Draws();install_client(monkeypatch,client)
    with pytest.raises(ValueError,match='inventory changed|file set changed'):
        probe.execute(prepared.output,api_key='inert-fixture',draw_fn=draws)
    assert not client.requests and not draws.calls


def test_stop_after_one_returned_call_preserves_charged_receipt_and_does_not_dispatch_next(prepared,monkeypatch):
    def behavior(number,request):
        (prepared.output/'STOP').touch()
    client=Client(behavior);draws=Draws();install_client(monkeypatch,client)
    with pytest.raises(CampaignStopped,match='STOP'):
        probe.execute(prepared.output,api_key='inert-fixture',draw_fn=draws)
    assert len(client.requests)==1 and len(draws.calls)==9
    summary=read(prepared.output/'summary.json')
    assert sum(arm['returned'] for arm in summary['arms'].values())==1
    assert len(summary['cost_ledger']['calls'])==1
    assert (prepared.output/'candidates/g01/pre_evaluation.json').is_file()


def test_stop_before_start_has_no_ledger_or_evaluations(prepared,monkeypatch):
    (prepared.output/'STOP').touch()
    client=Client();draws=Draws();install_client(monkeypatch,client)
    with pytest.raises(ValueError,match='stopped before dispatch'):
        probe.execute(prepared.output,api_key='inert-fixture',draw_fn=draws)
    assert not client.requests and not draws.calls
    assert not (prepared.output/'STARTED.json').exists()
    assert not (prepared.output/'cost_ledger.json').exists()


def test_frozen_memory_modified_during_calibration_stops_before_generation(prepared,monkeypatch):
    def modify(call,number):
        if number==9:
            path=prepared.output/'frozen/parents.json'
            path.write_text(path.read_text()+'\n')
    client=Client();draws=Draws(modify);install_client(monkeypatch,client)
    with pytest.raises(ValueError,match='identity drift'):
        probe.execute(prepared.output,api_key='inert-fixture',draw_fn=draws)
    assert not client.requests and len(draws.calls)==9
    assert not read(prepared.output/'cost_ledger.json')['calls']


def test_failed_reference_calibration_prevents_paid_generation(prepared,monkeypatch):
    client=Client();draws=Draws(lambda call,n:dict(ok=True,valid=False,score=0) if n==1 else None)
    install_client(monkeypatch,client)
    with pytest.raises(CampaignStopped,match='reference parent'):
        probe.execute(prepared.output,api_key='inert-fixture',draw_fn=draws)
    assert not client.requests
    assert not read(prepared.output/'cost_ledger.json')['calls']


def test_durable_ledger_enforces_18_reservations_even_after_reload(tmp_path):
    ledger=probe.ProbeLedger(tmp_path/'cost.json','3.00')
    for _ in range(18):
        ident=ledger.reserve('.001',run_id='fixture',model=probe.MODEL)
        ledger.settle(ident,0)
    ledger=probe.ProbeLedger(tmp_path/'cost.json','3.00')
    with pytest.raises(RequestNotDispatched,match='18-request'):
        ledger.reserve('.001',run_id='fixture',model=probe.MODEL)
    assert len(read(tmp_path/'cost.json')['calls'])==18
