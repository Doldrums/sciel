"""Explicit opt-in fixtures: authored constant data, no models or scientific search."""
import hashlib
import json
import os
from pathlib import Path

import pytest

from sciel.config import apply_defaults
from sciel.eval.candidate_backend import CandidateBackend
from sciel.eval.registry import load_task
from sciel.eval.trusted_scoring.adapter import HostBinding, score_payload
from sciel.eval.trusted_scoring.transport import SCHEMA
from sciel.methods.executor import DockerExecutor
from test_candidate_backend import IMAGE, POLICY, ROOT, TASKS, evaluate_once
from test_parallel_engine import config

pytestmark=pytest.mark.skipif(os.environ.get('SCIEL_CANDIDATE_FIXTURE_IMAGE')!=IMAGE,
    reason='explicit existing immutable image required for inert Docker fixtures')


def fixture_config(tmp_path,kind):
    cfg=config(TASKS[kind],tmp_path/'capacity',workers=1,num_islands=1,max_iterations=1,evaluation=dict(POLICY))
    return cfg,load_task(str(ROOT/'tasks'/TASKS[kind]))


def audit_owned_receipts(directory,receipt):
    refs=receipt['value']['execution']['receipts']
    actual=[]
    for ref in refs:
        row=json.loads((directory/ref['name']).read_text())
        for name in ('method_receipt','scorer_receipt'):
            if row[name] is None:continue
            physical=row[name]
            assert physical['container']['removed'] is True
            assert physical['container']['image_id']==IMAGE
            assert physical['resource_measurement']['cpu']=='trusted_cgroup_observation'
            assert physical['resources']['cpu_seconds']>0
            actual.append(physical['container']['name'])
        if row['scorer_receipt'] is not None:
            runtime=row['scorer_receipt']['output']['runtime']
            assert set(runtime)=={'python','numpy'} and all(runtime.values())
    assert len(actual)==len(set(actual))
    return actual


@pytest.mark.parametrize('kind,count',[('circle',4),('heil',10),('beat',4)])
def test_real_candidate_and_separate_scorer_match_inert_host_arithmetic(tmp_path,kind,count):
    cfg,task=fixture_config(tmp_path,kind)
    if kind=='circle':
        centers=[[((i%6)+.5)/6,((i//6)+.5)/5] for i in range(26)]
        values={'centers':centers,'radii':[.03]*26,'centers_dtype':'float64','radii_dtype':'float64','reported_sum':'-99.000000'}
        source='def run_packing():\n return '+repr((centers,[.03]*26,-99.))+'\n'
    elif kind=='heil':
        points=[[.1+.08*i,.03+.15*(.1+.08*i)**2] for i in range(11)]
        values={'points':points}
        source='def heilbronn_triangle11():\n return '+repr(points)+'\n'
    else:
        values={'support':[0.,1.],'weights':[1.,1.]}
        source='def beat_the_average_measure():\n return [[0,1]],[[1,1]]\n'
    # Authored fixture checks that no scorer/task/source checkout is visible.
    source='import os\nassert sorted(os.listdir("/bundle")) == ["candidate.py", "worker.py"]\nassert os.geteuid() == 10001\n'+source
    executor=DockerExecutor(IMAGE)
    ledger,directory,request,receipt=evaluate_once(tmp_path,cfg,task,executor,source.encode(),confirm=True)
    assert receipt['status']=='returned',receipt
    outcome=ledger.result('evaluation',receipt)
    assert outcome.result.ok
    assert len(audit_owned_receipts(directory,receipt))==count
    first=json.loads((directory/'full-0.receipt.json').read_text())
    physical=first['method_receipt']
    expected=score_payload(kind,json.dumps({'schema':SCHEMA,'kind':kind,'values':values}).encode(),
        HostBinding(hashlib.sha256(source.encode()).hexdigest(),outcome.execution['identity']['task']['evaluator_sha256'],
            physical['invocation_id'],'full',float(physical['resources']['method_wall_seconds'])))
    assert first['scorer_receipt']['output']['result']==expected
    assert outcome.raw_first_score==expected['score']
    assert outcome.confirmed==(kind=='heil')
    _,_,cached=ledger.prepare('G:1:0','evaluation',request)
    assert cached==receipt and ledger.result('evaluation',cached).execution==outcome.execution


@pytest.mark.parametrize('attack',['score','hash','shape','duplicate'])
def test_real_candidate_stdout_cannot_forge_scientific_authority(tmp_path,attack):
    cfg,task=fixture_config(tmp_path,'circle')
    value={'schema':SCHEMA,'kind':'circle','values':{'centers':[[.5,.5]]*26,'radii':[0.]*26,
        'centers_dtype':'float64','radii_dtype':'float64','reported_sum':'0.000000'}}
    if attack=='score':value['score']=999999.
    if attack=='hash':value['source_sha256']='f'*64
    if attack=='shape':value['values']['radii'].pop()
    inner=json.dumps(value,separators=(',',':'))
    if attack=='duplicate':inner=inner.replace('"radii_dtype":"float64"','"radii_dtype":"float64","radii_dtype":"float32"')
    encoded=json.dumps(inner).encode()
    source=('import os\ndef run_packing():\n os.write(1,'+repr(encoded)+')\n os._exit(0)\n').encode()
    ledger,directory,request,receipt=evaluate_once(tmp_path,cfg,task,DockerExecutor(IMAGE),source)
    assert receipt['status']=='returned',receipt
    outcome=ledger.result('evaluation',receipt)
    assert not outcome.result.ok and outcome.result.score==0 and outcome.reached=='smoke'
    assert len(audit_owned_receipts(directory,receipt))==2
    physical=json.loads((directory/'smoke-0.receipt.json').read_text())
    assert physical['method_receipt']['output']==inner
    assert outcome.execution['resources']['cpu_seconds']>0
    _,_,cached=ledger.prepare('G:1:0','evaluation',request)
    assert ledger.result('evaluation',cached).result.score==0
