"""Mock physical containers only; immutable stage attachments remain untouched."""
import json
import sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/unit'))
from test_candidate_backend import setup_backend, evaluate_once
from sciel.engine.parallel_calls import digest
from sciel.eval.facade import EvaluationInfrastructureError

@pytest.mark.parametrize('mutation', ['score', 'confirm', 'empty_receipts', 'cpu', 'nonterminal_status'])
def test_physical_evidence_must_authorize_replayed_science(tmp_path,monkeypatch,mutation):
    cfg,task,executor=setup_backend(tmp_path,monkeypatch)
    ledger,directory,request,receipt=evaluate_once(tmp_path,cfg,task,executor)
    original={p.name:p.read_bytes() for p in directory.glob('*') if p.name != 'receipt.json'}
    path=directory/'receipt.json'
    saved=json.loads(path.read_bytes())
    value=saved['payload']['value']
    if mutation=='score': value['result']['score']=123.0; value['raw_first_score']=123.0
    elif mutation=='confirm': value['confirmed']=True; value['result']['confirmed_sampled']=True
    elif mutation=='empty_receipts': value['execution']['receipts']=[]
    elif mutation=='cpu':
        value['execution']['resources']['cpu_seconds']=0.0
        value['execution']['resources']['known_cpu_seconds']=0.0
    else: value['execution']['status']='unknown'
    saved['sha256']=digest(saved['payload'])
    path.write_text(json.dumps(saved))
    calls=len(executor.calls)
    with pytest.raises(EvaluationInfrastructureError):
        _,_,cached=ledger.prepare('G:1:0','evaluation',request)
        ledger.result('evaluation',cached)
    assert len(executor.calls)==calls
    assert {p.name:p.read_bytes() for p in directory.glob('*') if p.name != 'receipt.json'}==original

def test_raw_first_score_remains_full_before_lower_confirmation(tmp_path,monkeypatch):
    cfg,task,executor=setup_backend(tmp_path,monkeypatch,'heil')
    def shrink_confirm(kind,request,raw,transport):
        if kind=='candidate' and len(executor.calls)>=5:
            output=json.loads(raw['output'])
            output['values']['points']=[[x/2,y/2] for x,y in output['values']['points']]
            raw['output']=json.dumps(output)
    executor.change=shrink_confirm
    ledger,directory,request,receipt=evaluate_once(tmp_path,cfg,task,executor,confirm=True)
    outcome=ledger.result('evaluation',receipt)
    full=json.loads((directory/'full-0.receipt.json').read_bytes())['scorer_receipt']['output']['result']['score']
    assert outcome.result.score < full
    assert outcome.raw_first_score == full

from sciel.methods.models import content_hash

def reseal_attachment(path, value):
    value['sha256']=content_hash({k:v for k,v in value.items() if k!='sha256'})
    path.write_text(json.dumps(value))
    return value['sha256']

@pytest.mark.parametrize('mutation', ['omit_last_trial_ref','reverse_trials','omit_confirm_stage',
    'duplicate_stage','stage_declares_two_trials','stage_selects_wrong_result',
    'stage_omits_receipt_with_complete_claim','erase_cpu_complete'])
def test_replay_inventory_and_stage_plan_cannot_override_physical_truth(tmp_path,monkeypatch,mutation):
    cfg,task,executor=setup_backend(tmp_path,monkeypatch,'heil')
    ledger,directory,request,receipt=evaluate_once(tmp_path,cfg,task,executor,confirm=True)
    original={p.name:p.read_bytes() for p in directory.glob('*')
              if p.name.endswith('.receipt.json') or p.name.endswith('.intent.json')}
    path=directory/'receipt.json';saved=json.loads(path.read_bytes())
    value=saved['payload']['value'];execution=value['execution']
    if mutation=='omit_last_trial_ref': execution['receipts']=execution['receipts'][:-1]
    elif mutation=='reverse_trials': execution['receipts'][-3:]=reversed(execution['receipts'][-3:])
    elif mutation=='omit_confirm_stage': execution['stages']=execution['stages'][:-1]
    elif mutation=='duplicate_stage': execution['stages'][-1]=dict(execution['stages'][-2])
    elif mutation=='erase_cpu_complete':
        execution['resources']['cpu_complete']=False
        execution['resources']['cpu_seconds']=None
    else:
        summary_path=directory/'confirm.stage.json';summary=json.loads(summary_path.read_bytes())
        if mutation=='stage_declares_two_trials': summary['complete']=2
        elif mutation=='stage_selects_wrong_result':
            summary['result']['score']=999.0
            value['result']['score']=999.0
        else:
            execution['receipts']=execution['receipts'][:-1]
            summary['cpu_seconds']-=.25
            execution['resources']['known_cpu_seconds']-=.25
            execution['resources']['cpu_seconds']-=.25
        new_hash=reseal_attachment(summary_path,summary)
        execution['stages'][-1]['sha256']=new_hash
    saved['sha256']=digest(saved['payload']);path.write_text(json.dumps(saved))
    calls=len(executor.calls)
    with pytest.raises(EvaluationInfrastructureError):
        _,_,cached=ledger.prepare('G:1:0','evaluation',request)
        ledger.result('evaluation',cached)
    assert len(executor.calls)==calls
    assert {p.name:p.read_bytes() for p in directory.glob('*')
              if p.name.endswith('.receipt.json') or p.name.endswith('.intent.json')}==original

def test_partial_confirm_known_budget_replays_exact_cpu_once(tmp_path,monkeypatch):
    cfg,task,executor=setup_backend(tmp_path,monkeypatch,'heil')
    cfg['evaluation']['cpu_seconds']=.5
    ledger,directory,request,receipt=evaluate_once(tmp_path,cfg,task,executor,confirm=True)
    outcome=ledger.result('evaluation',receipt)
    assert outcome.reached=='confirm' and not outcome.confirmed and not outcome.result.ok
    assert len(executor.calls)==8  # smoke/full and exactly two confirm trials, each candidate+scorer
    assert outcome.execution['resources']['cpu_seconds']==1.0
    assert outcome.execution['resources']['known_cpu_seconds']==1.0
    assert outcome.execution['resources']['cpu_complete'] is True
    summary=json.loads((directory/'confirm.stage.json').read_bytes())
    assert summary['resource_exhausted'] is True and summary['complete'] is False
    assert summary['cpu_seconds']==.5
    assert len([r for r in outcome.execution['receipts'] if r['name'].startswith('confirm-')])==2
    for _ in range(3):
        _,_,cached=ledger.prepare('G:1:0','evaluation',request)
        replayed=ledger.result('evaluation',cached)
        assert replayed.execution==outcome.execution
        assert replayed.result==outcome.result
    assert len(executor.calls)==8
