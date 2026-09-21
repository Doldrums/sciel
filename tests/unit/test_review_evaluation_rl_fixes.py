"""Confirmed review regressions; mocked search and tiny evaluator wire fixtures only."""
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

import pytest

from sciel.components.policy.mode_features import mode_features
from sciel.engine.factory import build_engine
from sciel.engine.db import Database
from sciel.engine import checkpoint
from sciel.eval.facade import EvalResult, MAX_RESULT_BYTES, MAX_ARTIFACT_BYTES
from sciel.eval.sandbox import run_stage, _read_result, ResultTooLarge
from sciel.stats import PlateauMonitor
from test_parallel_engine import config, task_fixture, evaluator, Provider, events

ROOT=Path(__file__).resolve().parents[2]


def script(name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/f'{name}.py')
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result);return result


def test_parallel_training_features_are_frozen_before_any_wave_commit(tmp_path,monkeypatch):
    task=task_fixture(tmp_path/'tasks');evaluator(monkeypatch)
    cfg=config(task,tmp_path/'capacity',workers=2,budget=2,max_iterations=2,
        policy={'enabled':True,'mode_bandit':True,'plateau_shift':False})
    engine=build_engine(cfg,str(tmp_path/'tasks'),str(tmp_path/'run'),llm=Provider('run_search',concurrent=2))
    selected=[];pick=engine.controller.pick_mode
    def capture(island,t,rng,view=None,phase=None,features=None):
        selected.append(mode_features(phase,engine._progress(),engine.db.pop_best,engine.db.island_best(island)))
        return pick(island,t,rng,view,phase,features)
    engine.controller.pick_mode=capture
    try:
        engine.run()
        samples,_=script('dt_dataloader').build_samples(tmp_path/'run/sciel_events.jsonl')
        assert [row['features'] for row in samples]==selected
        iterations=[row for row in events(tmp_path/'run') if row.get('child_id')]
        assert iterations[1]['pop_best_pre']>iterations[1]['decision_context']['pop_best']
        intents=[job for path in (tmp_path/'run/.parallel').rglob('jobs.json')
                 for job in json.loads(path.read_text())['jobs'] if job.get('attribution',{}).get('decision_context')]
        assert len(intents)==2
        assert [job['attribution']['decision_context']['B_evals'] for job in intents]==[0,0]
    finally:engine.close()


def test_successful_remeasurement_is_bound_but_not_a_training_mutation(tmp_path,monkeypatch):
    task=task_fixture(tmp_path/'tasks');evaluator(monkeypatch,fail_first_candidate=True)
    cfg=config(task,tmp_path/'capacity',workers=1,budget=1,max_iterations=2)
    engine=build_engine(cfg,str(tmp_path/'tasks'),str(tmp_path/'run'),llm=Provider('run_search',same=True))
    try:
        engine.run()
        journal=script('journal_data')
        rows,end=journal.completed_events(tmp_path/'run/sciel_events.jsonl')
        measured=[row for row in rows if row.get('reason')=='remeasurement']
        assert len(measured)==1 and measured[0]['measurement_ok'] is True
        assert measured[0].get('child_id') is None and end['B_evals']==1
        assert script('dt_dataloader').build_samples(tmp_path/'run/sciel_events.jsonl')[0]==[]
        assert script('lineage_labels').labels_for(tmp_path/'run/sciel_events.jsonl')==[]
    finally:engine.close()


@pytest.mark.parametrize('field,value',[('measurement_program_id','p1'),('measurement_ok','true')])
def test_remeasurement_reader_rejects_unbound_or_nonboolean_claim(tmp_path,monkeypatch,field,value):
    task=task_fixture(tmp_path/'tasks');evaluator(monkeypatch,fail_first_candidate=True)
    cfg=config(task,tmp_path/'capacity',workers=1,budget=1,max_iterations=2)
    engine=build_engine(cfg,str(tmp_path/'tasks'),str(tmp_path/'run'),llm=Provider('run_search',same=True))
    log_type=type(engine.log); original=log_type.iteration
    def corrupt(self, **event):
        if event.get('reason')=='remeasurement':event[field]=value
        return original(self, **event)
    # Waves replace the log with a staged RunLog, so patch its class boundary.
    monkeypatch.setattr(log_type,'iteration',corrupt)
    try:
        engine.run()
        module=script('journal_data')
        with pytest.raises(module.DirtyRun,match='remeasurement'):
            module.completed_events(tmp_path/'run/sciel_events.jsonl')
    finally:engine.close()


def test_historical_parallel_features_without_decision_evidence_are_rejected(monkeypatch):
    loader=script('dt_dataloader')
    start={'event':'run_start','schema_version':1,'budget':10,
           'config':{'budget':10,'parallel_search':{'enabled':True}}}
    step={'event':'iteration','island':0,'child_id':'p2','child_score':.5,'mode':'base',
          'pop_best_pre':.2,'island_best_pre':.1,'B_evals':2}
    monkeypatch.setattr(loader,'completed_events',lambda path:([start,step],{'pop_best':.5}))
    with pytest.raises(loader.DirtyRun,match='decision-time'):
        loader.build_samples('fixture')


def migration_fixture(tmp_path):
    db=Database(3)
    seeds=[db.add(f'seed{i}',.1,{},i,'seed') for i in range(3)]
    first=db.add('first improvement',.3,{},0,'gen',parent_id=seeds[0].id)
    db.migrate_unique_elites(0,[1])
    one=next(p for p in db.islands[1].programs.values() if p.parent_id==first.id)
    db.migrate_unique_elites(1,[2])
    two=next(p for p in db.islands[2].programs.values() if p.parent_id==one.id)
    last=db.add('later improvement',.9,{},2,'gen',parent_id=two.id)
    rows=[dict(event='iteration',child_id=p.id,parent_id=p.parent_id,child_score=p.score,
              island=p.island,B_evals=i+1) for i,p in enumerate((first,last))]
    state={'provenance':{'schema':'sciel-checkpoint-provenance/1','checkpoint_id':'fixed'},'db':db.state()}
    raw=json.dumps(state).encode();(tmp_path/'checkpoint.json').write_bytes(raw)
    end={'segment_id':'selected','checkpoint_id':'fixed','checkpoint_sha256':hashlib.sha256(raw).hexdigest(),
         'pop_best':.9}
    return rows,end,first,last,one,two


@pytest.mark.parametrize('with_code',[False,True])
def test_repeated_migration_bridges_lineage_without_new_training_examples(tmp_path,with_code):
    rows,end,first,last,one,two=migration_fixture(tmp_path)
    labels=script('lineage_labels').labels_for(tmp_path/'sciel_events.jsonl',with_code=with_code,trajectory=(rows,end))
    by_id={row['id']:row for row in labels}
    assert set(by_id)=={first.id,last.id}
    assert by_id[first.id]['subtree_max']==.9
    assert by_id[first.id]['lineage_gain']==pytest.approx(.6)
    assert by_id[last.id]['parent']==two.id and by_id[last.id]['gen']==1


def test_unbound_checkpoint_cannot_introduce_migration_ancestry(tmp_path):
    rows,_,first,*_=migration_fixture(tmp_path)
    labels=script('lineage_labels').labels_for(tmp_path/'sciel_events.jsonl',trajectory=(rows,{'pop_best':.9}))
    assert next(row for row in labels if row['id']==first.id)['subtree_max']==.3


def test_long_plateau_recovers_and_rearms_with_finite_round_trip_and_legacy_load():
    monitor=PlateauMonitor()
    for _ in range(30000):monitor.observe(False,phase=-1)
    assert monitor.newly_declared()
    assert monitor.log_wealth()>710
    saved=json.loads(json.dumps(monitor.state(),allow_nan=False))
    restored=PlateauMonitor();restored.load_state(saved)
    assert restored.state()==monitor.state()
    for _ in range(2000):
        restored.observe(True,phase=-1);restored.newly_declared()
    assert not restored.is_plateau() and restored.log_wealth()==0
    for _ in range(150):restored.observe(False,phase=-1)
    assert restored.newly_declared()
    legacy={k:v for k,v in saved.items() if k not in ('schema','log_wealth')}
    legacy['wealth']=3.5
    restored.load_state(legacy)
    assert restored.log_wealth()==pytest.approx(math.log(3.5))
    with pytest.raises(ValueError,match='lost history'):
        restored.load_state(dict(legacy,wealth=float('inf')))


def test_engine_checkpoint_roundtrip_of_unrepresentable_wealth_is_finite(tmp_path):
    task=task_fixture(tmp_path/'tasks')
    cfg=config(task,tmp_path/'capacity',workers=1,budget=1,num_islands=1,plateau_shadow=True)
    cfg['candidate_novelty']={'enabled':False}
    cfg['parallel_search']={'enabled':False}
    first=build_engine(cfg,str(tmp_path/'tasks'),str(tmp_path/'first'),llm=Provider('run_search'))
    second=None
    try:
        first.db.add('VALUE=0\ndef run_search(): return VALUE\n',0.,{},0,'seed')
        monitor=first.plateau_monitors[0]
        for _ in range(30000):monitor.observe(False,phase=-1)
        saved=checkpoint.save(first,'test-long-plateau')
        raw=Path(saved).read_text()
        json.loads(raw,parse_constant=lambda value:pytest.fail('nonfinite checkpoint '+value))
        second=build_engine(cfg,str(tmp_path/'tasks'),str(tmp_path/'second'),llm=Provider('run_search'))
        checkpoint.load(second,saved)
        assert second.plateau_monitors[0].state()==monitor.state()
    finally:
        first.close()
        if second is not None:second.close()


def test_malformed_bound_migration_cannot_fabricate_descendant_credit(tmp_path):
    rows,end,_,_,one,_=migration_fixture(tmp_path)
    state=json.loads((tmp_path/'checkpoint.json').read_text())
    next(row for row in state['db']['programs'] if row['id']==one.id)['code']='different source'
    raw=json.dumps(state).encode();(tmp_path/'checkpoint.json').write_bytes(raw)
    end['checkpoint_sha256']=hashlib.sha256(raw).hexdigest()
    assert script('lineage_labels').labels_for(tmp_path/'sciel_events.jsonl',trajectory=(rows,end))==[]


def test_file_bound_rejects_large_ordinary_artifact_without_loading_it(tmp_path):
    task=tmp_path/'evaluator.py';candidate=tmp_path/'candidate.py';candidate.write_text('')
    task.write_text('def full(path):\n return {"ok":True,"score":1.,"artifacts":{"huge":"x"*8388608}}\n')
    result=run_stage(str(task),'full',str(candidate),timeout=8,python=sys.executable)
    assert not result.ok and result.artifacts['error_type']=='ResultTooLarge'
    path=tmp_path/'oversize.json'
    with path.open('wb') as stream:stream.truncate(MAX_RESULT_BYTES+1)
    with pytest.raises(ResultTooLarge):_read_result(str(path))


@pytest.mark.parametrize('wire',[
    '{"ok":true,"score":NaN}', '{"ok":true,"score":"not-a-number"}',
    '{"ok":"yes","score":1}', '{"ok":true,"score":1,"cases":{"x":Infinity}}',
    '{"ok":true,"score":1,"cost":-1}', '{"ok":true,"score":1,"score":2}',
    '[]', '{"ok":true,"score":1,"artifacts":{"x":123}}'])
def test_corrupt_raw_evaluator_wire_returns_failed_result(tmp_path,wire):
    task=tmp_path/'evaluator.py';candidate=tmp_path/'candidate.py';candidate.write_text('')
    task.write_text('import os,sys\ndef full(path):\n'
                    f' open(sys.argv[4],"w").write({wire!r})\n os._exit(0)\n')
    result=run_stage(str(task),'full',str(candidate),timeout=8,python=sys.executable)
    assert not result.ok and math.isfinite(result.score)
    assert result.artifacts['error_type']=='ResultCorrupt'


def test_artifact_bounds_and_native_geometry_roundtrip():
    with pytest.raises(ValueError,match='artifact value'):
        EvalResult.from_json({'ok':True,'score':1.,'artifacts':{'large':'x'*(MAX_ARTIFACT_BYTES+1)}})
    result=EvalResult(True,.7,{'sum_radii':1.8445},artifacts={'native_geometry':'{"fixture":true}'},timing=.1)
    assert EvalResult.from_json(result.to_json())==result
