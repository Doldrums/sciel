"""Host-only protocol replay and explicit split-helper source identity."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest
from sciel.eval import oracle_evidence as O, oracle_runtime_profiles as profiles

SOURCE=b'# synthetic public candidate; never executed\n'
SOURCE_SHA=hashlib.sha256(SOURCE).hexdigest()
TASK=O.task_module()
path=Path(__file__).parents[1]/'fixtures/upstream_oracle_trial.py'
spec=importlib.util.spec_from_file_location('split_public_trial',path)
fixture=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture)


def report(module, stage, status):
    p=module.panels
    panel=p.panel('train','b'*64,[p.digest([i]) for i in range(80)],0)
    trials,journals=[],[]
    for i in range(1 if stage=='smoke' else p.PANEL_SIZE):
        value,raw=fixture.synthetic_trial(module,i,status=status if i==0 else 'ok',
            source=SOURCE,name=f'sciel-oracle-{i:032x}',selected=panel)
        trials.append(value);journals.append(raw)
    binding=dict(image_id='sha256:'+'a'*64,assets_sha256=module.runtime.runtime_assets_sha256(),
        base_image=module.runtime.BASE_IMAGE)
    expected=dict(expected_stage=stage,source_sha256=SOURCE_SHA,panel=panel,binding=binding,
        task_hash='c'*64,version='d'*64)
    value=module.receipts.build(stage=stage,source_sha256=SOURCE_SHA,panel=panel,binding=binding,
        task_hash='c'*64,version='d'*64,trials=trials,journals=journals)
    return value,journals,expected


@pytest.mark.parametrize('stage',['smoke','full','confirm'])
@pytest.mark.parametrize('status',['ok','protocol_error','infrastructure_error','timeout'])
def test_split_replays_whole_stage_and_rejects_rehashed_frame_tamper(stage,status):
    value,journals,expected=report(TASK,stage,status)
    assert TASK.receipts.verify(value,journals=journals,**expected)==value
    raw=[json.loads(line) for line in journals[0].splitlines()]
    raw[0]['algorithm_seed']+=1
    changed=b''.join(TASK.panels.canonical(row)+b'\n' for row in raw)
    forged=deepcopy(value);forged['trials'][0]['journal_sha256']=hashlib.sha256(changed).hexdigest()
    forged['sha256']=TASK.panels.digest({k:v for k,v in forged.items() if k!='sha256'})
    with pytest.raises(ValueError):TASK.receipts.verify(forged,journals=[changed,*journals[1:]],**expected)


@pytest.mark.parametrize('helper',['assets/oracle_protocol.py','assets/receipt_trials.py'])
def test_new_host_helpers_are_hash_bound_without_changing_container_assets(tmp_path,helper):
    root=tmp_path/'task';shutil.copytree(TASK.ROOT,root,ignore=shutil.ignore_patterns('__pycache__'))
    binding=dict(image_id='sha256:'+'a'*64,assets_sha256=TASK.runtime.runtime_assets_sha256(),
        base_image=TASK.runtime.BASE_IMAGE)
    before=TASK.receipts.evaluator_version(root,binding)
    profiles.validate_historical_retained(root,profiles.SCIENTIFIC_RUNTIME,profiles.SCIENTIFIC_EVALUATOR_VERSION)
    entry=root/helper;entry.write_bytes(entry.read_bytes()+b'\n# changed helper\n')
    assert TASK.receipts.evaluator_version(root,binding)!=before
    with pytest.raises(ValueError,match='replay implementation differs'):
        profiles.validate_historical_retained(root,profiles.SCIENTIFIC_RUNTIME,profiles.SCIENTIFIC_EVALUATOR_VERSION)
    # Those new modules stay on the trusted host, outside the image entrypoint.
    digest=hashlib.sha256()
    for name in ('Dockerfile','container_supervisor.py','oracle_client.py','candidate_runner.py'):
        digest.update(name.encode()+b'\0'+(root/'assets'/name).read_bytes()+b'\0')
    assert digest.hexdigest()==binding['assets_sha256']


def test_generator_binds_protocol_helper_before_creating_or_accepting_panels(tmp_path, monkeypatch):
    root=tmp_path/'assets';root.mkdir()
    for name in ('oracle_runtime.py','oracle_protocol.py'):
        shutil.copyfile(TASK.ROOT/'assets'/name,root/name)
    monkeypatch.setattr(TASK.runtime, '__file__', str(root/'oracle_runtime.py'))
    before=TASK.panels.generator_sha256()
    panel=TASK.panels.panel('train','b'*64,[TASK.panels.digest([i]) for i in range(80)],0)
    assert panel['generator_sha256']==before
    protocol=root/'oracle_protocol.py'
    protocol.write_bytes(protocol.read_bytes()+b'\n# changed objective/protocol source\n')
    assert TASK.panels.generator_sha256()!=before
    with pytest.raises(ValueError): TASK.panels.validate_panel(panel,expected_split='train')
