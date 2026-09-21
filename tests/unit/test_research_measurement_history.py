"""Inert, immutable evaluation records and mocked Engine transports only."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS
import sys

import pytest

from sciel.engine import checkpoint
from sciel.engine.evaluation_timing import remember
from sciel.engine.factory import build_engine
from sciel.engine.novelty import CandidateNoveltyRegistry
from sciel.engine.observed_changes import ObservedChanges
from sciel.engine.research_artifacts import bind_evaluation_program, capture_evaluation
from sciel.engine.research_search_evidence import ALIAS_LIMIT, search_evidence
from sciel.engine.types import Program
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from test_parallel_engine import Provider, config, evaluator, task_fixture


def history(tmp_path, *, novelty=True):
    return NS(run_dir=tmp_path, task=NS(task_hash='inert-history-task'), cfg={},
        log=NS(run_id='inert-history-run'), observed_changes=ObservedChanges(),
        candidate_novelty=CandidateNoveltyRegistry() if novelty else None,
        db=NS(_all={}, pop_best=.5), iteration_no=3, B_evals=3, _per_run_budget=lambda:28)


def add(engine, ident, evaluation_id, *, code=None, island=0, origin='gen', bind=True):
    code = code or 'def solve(): return 0\n'
    p = Program(ident, code, .5, {'quality':.5, 'n_trials':3.}, island, origin)
    result = CascadeOutcome(EvalResult(True,.5,deepcopy(p.cases),timing=6.,
        artifacts={'state':[1,2,3]}),.5,'confirm',True)
    result.evaluation_id = evaluation_id
    if bind:
        capture_evaluation(engine,code,result,evaluation_id)
        bind_evaluation_program(engine,p,evaluation_id)
        if engine.candidate_novelty is not None:
            engine.candidate_novelty.register(code,evaluation_id,evaluation_ok=True)
            engine.candidate_novelty.bind_program(code,p.id,evaluation_id=evaluation_id)
    engine.db._all[p.id] = p
    engine.observed_changes.observe(p,dict(ok=True,score=.5,cases=p.cases,artifacts={},
                                          reached='confirm',confirmed=True))
    remember(engine,p,result)
    return p


def migrate(engine, parent, ident, island):
    p = Program(ident,parent.code,parent.score,deepcopy(parent.cases),island,'migrant',parent_id=parent.id)
    engine.db._all[p.id] = p
    engine.observed_changes.bind_alias(p,parent)
    return p


@pytest.mark.parametrize('novelty',[False,True])
def test_real_snapshot_shape_twelve_rows_are_four_measurements(tmp_path,novelty):
    engine = history(tmp_path,novelty=novelty)
    # V34 snapshot p16..p24 were nine recent placements of one canonical eval;
    # the actual run retained all24 seed placements plus three admitted children.
    for i in range(1,25):
        seed = add(engine,f'p{i}','canonical-seed',island=i-1,origin='seed')
    for i in range(25,28):
        child = add(engine,f'p{i}',f'eval-{i}',code=f'def solve(): return {i}\n',island=i-25)
    evidence = search_evidence(engine,child)
    rows = evidence['recent_evaluations']
    assert evidence['schema']=='sciel-search-evidence/3' and 'recent_admissions' not in evidence
    assert [r['evaluation_id'] for r in rows]==['canonical-seed','eval-25','eval-26','eval-27']
    assert len(rows)==4 and all(r['identity_status']=='authenticated' for r in rows)
    assert rows[0]['program_id']=='p1' and rows[0]['admissions_count']==24
    assert len(rows[0]['aliases'])==ALIAS_LIMIT and rows[0]['aliases_omitted']==23-ALIAS_LIMIT
    assert evidence['admission_stats']['observed_admissions_total']==27
    assert evidence['admission_stats']['bound_evaluation_identities_total']==4
    assert evidence['unverified_admissions']==[]
    assert all(r['saved_evaluation_sha256'] and r['observation_sha256'] for r in rows)
    assert all(r['evaluation_stage_wall_seconds']==6. for r in rows)
    assert evidence['recent_evaluations_omitted']==0


def test_same_source_and_identical_outputs_with_distinct_ids_are_two_measurements(tmp_path):
    engine = history(tmp_path)
    first = add(engine,'p1','eval-first')
    later = add(engine,'p2','eval-again',code=first.code,origin='remeasurement')
    rows = search_evidence(engine,later)['recent_evaluations']
    assert len(rows)==2 and rows[0]['score']==rows[1]['score']
    assert rows[0]['source_sha256']==rows[1]['source_sha256']
    assert [r['evaluation_id'] for r in rows]==['eval-first','eval-again']
    assert rows[0]['saved_evaluation_sha256']!=rows[1]['saved_evaluation_sha256']


def test_migration_aliases_do_not_refresh_measurement_recency_or_evict_new_results(tmp_path):
    engine = history(tmp_path)
    first = add(engine,'p1','eval-1')
    for i in (2,3):
        last = add(engine,f'p{i}',f'eval-{i}',code=f'def solve(): return {i}\n')
    alias = migrate(engine,first,'p4',9)
    migrate(engine,alias,'p5',10)
    rows = search_evidence(engine,last,limit=2)['recent_evaluations']
    assert [r['evaluation_id'] for r in rows]==['eval-2','eval-3']
    evidence = search_evidence(engine,last)
    assert evidence['admission_stats']['explicit_migration_aliases_total']==2
    row = evidence['recent_evaluations'][0]
    assert row['admissions_count']==3 and [a['program_id'] for a in row['aliases']]==['p4','p5']
    assert all(a['origin']=='migrant' for a in row['aliases'])


def test_unknown_identity_is_explicit_and_never_grouped_by_source(tmp_path):
    engine = history(tmp_path)
    for i in range(1,4):
        p = add(engine,f'p{i}','unattested-same-id',bind=False)
    evidence = search_evidence(engine,p)
    assert evidence['recent_evaluations']==[]
    assert [r['program_id'] for r in evidence['unverified_admissions']]==['p1','p2','p3']
    assert all(r['identity_status']=='unknown' and r['evaluation_id'] is None
               for r in evidence['unverified_admissions'])


def test_missing_immutable_artifact_does_not_turn_declared_id_into_measurement(tmp_path):
    engine = history(tmp_path)
    first = add(engine,'p1','declared-eval')
    second = add(engine,'p2','declared-eval')
    root=tmp_path/'research_evidence'/'evaluations'
    (root/('evaluation-'+hashlib.sha256(b'declared-eval').hexdigest()+'.json')).unlink()
    evidence = search_evidence(engine,second)
    assert evidence['recent_evaluations']==[]
    assert [r['program_id'] for r in evidence['unverified_admissions']]==[first.id,second.id]
    assert all(r['evaluation_id']=='declared-eval' and r['identity_status']=='unknown'
               for r in evidence['unverified_admissions'])


@pytest.mark.parametrize('tamper',['source','evaluation_id','artifact','alias_source','timing'])
def test_changed_measurement_binding_is_rejected(tmp_path,tamper):
    engine = history(tmp_path)
    p = add(engine,'p1','eval-1')
    if tamper=='source':
        engine.research_evaluation_bindings[p.id]['source_sha256']='0'*64
    elif tamper=='evaluation_id':
        engine.research_evaluation_bindings[p.id]['evaluation_id']='not-the-registry-id'
    elif tamper=='artifact':
        index = tmp_path/'research_evidence'/'evaluations'/('evaluation-'+hashlib.sha256(b'eval-1').hexdigest()+'.json')
        value=json.loads(index.read_text()); value['source_sha256']='0'*64
        index.write_text(json.dumps(value))
    elif tamper=='alias_source':
        alias=migrate(engine,p,'p2',5)
        engine.observed_changes._aliases[alias.id]['source_sha256']='0'*64
    else:
        engine.evaluation_timings[p.id]['evaluation_id']='not-the-measured-id'
    with pytest.raises(ValueError,match='source|binding|identity|index|provenance'):
        search_evidence(engine,p)


def test_cyclic_migration_binding_is_rejected(tmp_path):
    from sciel.engine.research_artifacts import evaluation_id_for_program
    engine=history(tmp_path)
    first=add(engine,'p1','eval-1')
    alias=migrate(engine,first,'p2',4)
    engine.observed_changes._aliases[first.id]={**engine.observed_changes._aliases[alias.id],
                                               'reference_program_id':alias.id}
    with pytest.raises(ValueError,match='cyclic'):
        evaluation_id_for_program(engine,alias)


@pytest.mark.parametrize('budget',[8000,64000])
@pytest.mark.parametrize('background',[False,True])
def test_schema_three_history_reaches_planner_or_explicit_omission_with_repairs(tmp_path,budget,background):
    from sciel.engine.research_targets import tool_inputs
    from sciel.research_tools.research_steps import new_state
    from test_research_tool_controller import snapshot, prepare
    from test_sequential_research_episode import build
    engine=history(tmp_path)
    for i in range(24):
        add(engine,f'p{i}','seed-measurement',origin='seed',island=i)
    for i in range(24,27):
        parent=add(engine,f'p{i}',f'measurement-{i}',code=f'def solve(): return {i}\n')
    current=snapshot()
    current.research_search_evidence=search_evidence(engine,parent)
    before=deepcopy(current.research_search_evidence)
    tools,_,_,_=build(tmp_path,[],max_prompt_chars=budget,background=background)
    plan=prepare(tools,current)
    state=new_state()
    state['question']='Do independent measurements support an additional controlled trial?'
    state['next_decision']='Inspect the measured differences before allocating compute.'
    feedback=[{'tool_id':'a'*64,'receipt_id':'b'*64,
        'result':{'status':'tool_error','error':'KeyError: missing_expected_field'},
        'stderr':'Traceback: KeyError: missing_expected_field'}]
    prompt=tools._prompt(plan,current,[],feedback,1,research_state=state)
    assert len(prompt)<=budget and 'missing_expected_field' in prompt
    view=json.JSONDecoder().raw_decode(prompt.split('OBSERVED SEARCH LIMITS JSON:\n',1)[1])[0]
    assert view['schema']=='sciel-search-evidence/3' and 'recent_admissions' not in view
    assert len(view.get('recent_evaluations',[]))+view.get('recent_evaluations_omitted',0)==4
    if budget==64000:
        assert [r['evaluation_id'] for r in view['recent_evaluations']]==[
            'seed-measurement','measurement-24','measurement-25','measurement-26']
        assert view['admission_stats']['observed_admissions_total']==27
    assert tool_inputs(current,tools.task_identity,{})['search_evidence']==before
    assert current.research_search_evidence==before


@pytest.mark.parametrize('parallel',[False,True])
def test_real_engine_seed_and_ordinary_resume_retain_measurements_not_placements(tmp_path,monkeypatch,parallel):
    tasks=tmp_path/'tasks'
    task=task_fixture(tasks)
    calls,_=evaluator(monkeypatch)
    cfg=config(task,tmp_path/'capacity',workers=2,num_islands=4,max_iterations=2)
    cfg['parallel_search']['enabled']=parallel
    provider=Provider('run_search')
    engine=build_engine(cfg,str(tasks),str(tmp_path/'run'),python=sys.executable,llm=provider)
    finish=engine._finish
    def with_retained_eviction_and_migration(reason):
        best=engine.db.get(engine.db.pop_best_id)
        initial=search_evidence(engine,best)
        seed=engine.db.get(initial['recent_evaluations'][0]['program_id'])
        for island in engine.db.islands:
            island.programs.pop(seed.id,None)
        assert search_evidence(engine,best)==initial
        alias=engine.db.add(seed.code,seed.score,deepcopy(seed.cases),best.island,
                           origin='migrant',parent_id=seed.id)
        engine.observed_changes.bind_alias(alias,seed)
        return finish(reason)
    monkeypatch.setattr(engine,'_finish',with_retained_eviction_and_migration)
    engine.run()
    best=engine.db.get(engine.db.pop_best_id)
    expected=search_evidence(engine,best)
    assert len(calls)==3 and len(expected['recent_evaluations'])==3
    assert expected['recent_evaluations'][0]['admissions_count']==5
    path=str(Path(engine.run_dir)/'checkpoint.json')
    restored=build_engine(cfg,str(tasks),engine.run_dir,python=sys.executable,llm=provider)
    checkpoint.load(restored,path)
    assert search_evidence(restored,restored.db.get(best.id))==expected
    assert len(calls)==3
