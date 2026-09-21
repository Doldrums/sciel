"""Resealed attachments must still obey the owned method receipt contract."""
import json,sys
from pathlib import Path
import pytest
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tests/unit'))
from test_candidate_backend import setup_backend,evaluate_once
from sciel.engine.parallel_calls import digest
from sciel.methods.models import content_hash
from sciel.eval.facade import EvaluationInfrastructureError

def seal(value,key='sha256'):
    value[key]=content_hash({k:v for k,v in value.items() if k!=key})
    return value[key]

@pytest.mark.parametrize('mutation',['cpu_authority','output_hash','candidate_scorer_input_link'])
def test_replay_revalidates_physical_scorer_contract(tmp_path,monkeypatch,mutation):
    cfg,task,executor=setup_backend(tmp_path,monkeypatch)
    ledger,directory,request,receipt=evaluate_once(tmp_path,cfg,task,executor)
    physical_path=directory/'full-0.receipt.json'
    physical=json.loads(physical_path.read_bytes())
    scorer=physical['method_receipt' if mutation=='candidate_scorer_input_link' else 'scorer_receipt']
    if mutation=='cpu_authority': scorer['resource_measurement']['cpu']='child_claim'
    elif mutation=='output_hash': scorer['output_sha256']='0'*64
    else:
        scorer['output']='{}'
        scorer['output_sha256']=content_hash(scorer['output'])
    seal(scorer,'receipt_sha256');seal(physical)
    physical_path.write_text(json.dumps(physical))
    path=directory/'receipt.json';saved=json.loads(path.read_bytes())
    refs=saved['payload']['value']['execution']['receipts']
    next(ref for ref in refs if ref['name']==physical_path.name)['sha256']=physical['sha256']
    saved['sha256']=digest(saved['payload']);path.write_text(json.dumps(saved))
    calls=len(executor.calls)
    with pytest.raises(EvaluationInfrastructureError):
        _,_,cached=ledger.prepare('G:1:0','evaluation',request)
        ledger.result('evaluation',cached)
    assert len(executor.calls)==calls
