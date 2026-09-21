"""Authored Circle history, inert import/generation and journal restoration."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from sciel.engine import external_research as external
from sciel.engine.circle_generation import CircleGeneration
from sciel.engine.research import ResearchController
from sciel.engine.llm.client import LLMResponse
from sciel.engine.types import Snapshot, NoChild
from sciel.hypothesis import circle_optimizer_traced_body as builder
from sciel.knowledge_base.catalogs import canonical_sha, canonical_json
from sciel.run_log import RunLog

ROOT=Path(__file__).resolve().parents[2]


@pytest.fixture(scope='module')
def capsule():
    from legacy_history_fixture import build_legacy_history
    return build_legacy_history()


class Model:
    models=['anthropic/claude-sonnet-4.5'];max_retries=1
    def __init__(self,text=None):self.text=text;self.calls=[]
    def generate(self,prompt,model,temperature=None,*,response_format=None):
        assert self.text is not None,'unexpected model dispatch'
        self.calls.append(dict(prompt=prompt,response_format=response_format))
        return LLMResponse(self.text,model,.001,10,10,finish_reason='stop')


def construct(capsule,path,*,text=None):
    cfg=copy.deepcopy(capsule['conclusion']['state']['config']);cfg[external.CONFIG_KEY]=canonical_sha(capsule)
    model=Model(text);model.models=[cfg['model']];log=RunLog(str(path/'events.jsonl'));log.run_start(scope='inert external-history test')
    charges=[];c=ResearchController(cfg,model,log,charges.append,metric=capsule['source']['state']['metric'])
    worker=CircleGeneration(c,execution_profile=builder.EXECUTION_PROFILE)
    record=capsule['source']['state']['knowledge']['experiments'][0]
    parent=NS(id='p1',parent_id=None,island=0,code=builder.source_for(),score=record['parent_score'],
              cases=record['parent_cases'],card='')
    return NS(c=c,worker=worker,parent=parent,model=model,log=log,charges=charges,cfg=cfg)


def import_(f,capsule):
    return f.worker.import_external_history(capsule,expected_sha256=canonical_sha(capsule),parent=f.parent)


def rows(path):return [json.loads(line) for line in Path(path).read_text().splitlines()]


def rewrite(capsule,block,mutator):
    value=copy.deepcopy(capsule);events=[json.loads(line) for line in value[block]['journal'].splitlines()]
    mutator(events);value[block]['journal']=''.join(json.dumps(e)+'\n' for e in events);return value


def snapshot(f):
    context=f.c.context(NS(_all={'p1':f.parent}),f.parent,0,1)
    return Snapshot(island=0,op='circle_generate',model=f.model.models[0],iteration_no=1,
        parent_id='p1',parent_code=f.parent.code,parent_score=f.parent.score,
        task_prompt='oe/circle_packing',research_context=context,kb_lessons=context['lesson_texts'])


def test_actual_import_preserves_exact_knowledge_review_and_current_only_counters(capsule,tmp_path):
    f=construct(capsule,tmp_path);before=copy.deepcopy(capsule)
    assert import_(f,capsule) is True
    assert f.c.knowledge.state()==capsule['reflection']['state']['knowledge']
    assert f.c.conclusions==capsule['reflection']['state']['conclusions']
    assert all(getattr(f.c,k)==0 for k in ['attempts','outcomes','valid_hypotheses','aux_calls','aux_cost','reflections','reflected_through'])
    assert f.worker.seen_proofs==[] and f.worker.calls_started==0 and f.charges==[] and f.model.calls==[]
    assert [e.get('phase') for e in rows(f.log.path)]==[None,external.EVENT]
    assert import_(f,capsule) is False and len(rows(f.log.path))==2
    assert capsule==before
    f.worker.assert_journal(f.log.path,f.log.anchor())
    restored=construct(capsule,tmp_path/'restored');restored.c.load_state(f.c.state());restored.worker.load_state(f.worker.state())
    restored.worker.assert_journal(f.log.path,f.log.anchor())
    assert restored.c.state()==f.c.state()


@pytest.mark.parametrize('memory_limit,presented,omitted', [(32000, 2, 4), (22000, 0, 6)])
def test_actual_context_allocates_new_id_and_delivers_whole_review_within_memory_limit(
        capsule,tmp_path,monkeypatch,memory_limit,presented,omitted):
    from sciel.engine import research
    monkeypatch.setattr(research, 'MAX_MEMORY_CHARS', memory_limit)
    f=construct(capsule,tmp_path);import_(f,capsule);snap=snapshot(f)
    context=snap.research_context;feedback=context.get('conclusion_feedback')
    assert feedback is not None and feedback['retained_count']==1
    assert feedback['presented_count']==1 and feedback['omitted_count']==0
    card,=feedback['cards']
    assert card['experiment_id']=='r000001a1'
    assert card['record_sha256']==canonical_sha(f.c.knowledge.experiments[0])
    assert card['unverified_interpretation']['mechanism_review']==f.c.conclusions['r000001a1']['llm']['mechanism_review']
    from sciel.hypothesis.prompts import restart_block
    from sciel.knowledge_base.restart_prediction_context import block as prediction_block
    memory_chars=sum(len(canonical_json(context[key])) for key in (
        'paired_comparisons','comparison_facts','proposal_groups','conclusion_feedback',
        'initializer_evidence') if key in context)
    memory_chars+=sum(map(len,context['lesson_texts']))
    memory_chars+=len(restart_block(context.get('restart_observations')))
    memory_chars+=len(prediction_block(context.get('restart_predictions')))
    assert memory_chars<=context['lesson_presentation']['max_memory_chars']==memory_limit
    detail=context['restart_observations']
    assert detail['presented_count']==presented and detail['omitted_count']==omitted
    assert len(detail['omissions'])==omitted
    ident,prompt=f.c.begin(snap,0,'circle_generate',f.worker.prepare_prompt(snap),context_scope='circle_restart_body/1')
    assert ident=='r000002a1' and f.c.attempts==1 and f.c.outcomes==0 and f.model.calls==[]
    event=rows(f.log.path)[-1];assert event['conclusion_feedback']==feedback
    assert 'conclusion_feedback' in prompt


@pytest.mark.parametrize('damage',['raw','proof','hypothesis','conclusion_finish','reflection_status','reflection_index'])
def test_tampered_foreign_provenance_rejected_before_mutation_even_with_new_digest(capsule,tmp_path,damage):
    block='source' if damage in ('raw','proof','hypothesis') else 'conclusion' if damage=='conclusion_finish' else 'reflection'
    def mutate(events):
        if damage=='raw':next(e for e in events if e.get('action')=='request_result')['response']+=' '
        if damage=='proof':next(e for e in events if e.get('action')=='source_binding')['proof']['target_body']+='\n'
        if damage=='hypothesis':next(e for e in events if e.get('phase')=='hypothesis')['evidence']['restart_prediction_binding']['response_sha256']='0'*64
        if damage=='conclusion_finish':next(e for e in events if e.get('phase')=='aux_result')['finish_reason']='length'
        if damage=='reflection_status':next(e for e in events if e.get('phase')=='reflection')['status']='invalid'
        if damage=='reflection_index':next(e for e in events if e.get('phase')=='aux_start')['call_index']=1
    changed=rewrite(capsule,block,mutate);f=construct(changed,tmp_path);before=f.c.state()
    with pytest.raises(ValueError):import_(f,changed)
    assert f.c.state()==before and len(rows(f.log.path))==1 and f.model.calls==[]


def test_declared_digest_parent_and_empty_lessons_gate_are_fail_closed(capsule,tmp_path):
    f=construct(capsule,tmp_path/'wrongdigest');changed=copy.deepcopy(capsule);changed['source']['manifest_sha256']='0'*64
    with pytest.raises(ValueError,match='declared digest'):import_(f,changed)
    f.parent.code=builder.source_for('return best_params.copy()\n')
    with pytest.raises(ValueError,match='parent binding'):import_(f,capsule)
    changed=copy.deepcopy(capsule)
    for block in ('source','conclusion','reflection'):changed[block]['state']['knowledge']['lessons']=[{}]
    g=construct(changed,tmp_path/'lessons')
    with pytest.raises(ValueError):import_(g,changed)
    assert g.c.knowledge.experiments==[]


def test_oversize_capsule_rejected_before_canonical_serializer(capsule,monkeypatch):
    def forbidden(*a):pytest.fail('oversized source reached canonical serialization')
    monkeypatch.setattr(external,'canonical_json',forbidden)
    with pytest.raises(ValueError,match='8 MiB'):external._bounded({'journal':'x'*(external.MAX_BYTES+1)})


def test_import_log_failure_and_evicted_seen_state_do_not_publish(capsule,tmp_path,monkeypatch):
    f=construct(capsule,tmp_path/'logfail');before=f.c.state()
    def fail(**kwargs):raise OSError('inert disk failure')
    monkeypatch.setattr(f.log,'research',fail)
    with pytest.raises(OSError):import_(f,capsule)
    assert f.c.state()==before and f.worker.seen_proofs==[]
    g=construct(capsule,tmp_path/'seen');g.c.knowledge._seen['old']='0'*64
    with pytest.raises(ValueError,match='fresh research'):import_(g,capsule)
    assert g.c.knowledge._seen=={'old':'0'*64}


@pytest.mark.parametrize('damage',['null','missing','floor','review'])
def test_registry_restore_mutation_rejected_atomically(capsule,tmp_path,damage):
    f=construct(capsule,tmp_path/'source');import_(f,capsule);saved=f.c.state()
    if damage=='null':saved['external_history']=None
    if damage=='missing':del saved['external_history']
    if damage=='floor':saved['external_history']['attempt_id_floor']=0
    if damage=='review':saved['conclusions']={}
    target=construct(capsule,tmp_path/'target');before=target.c.state()
    with pytest.raises(ValueError):target.c.load_state(saved)
    assert target.c.state()==before


def test_missing_import_event_and_foreign_result_for_new_id_rejected(capsule,tmp_path):
    f=construct(capsule,tmp_path/'source');import_(f,capsule);events=rows(f.log.path)
    with pytest.raises(ValueError,match='durable import'):external.assert_import_journal(f.c,events[:1])
    imported=copy.deepcopy(f.c.knowledge.experiments[0]);imported['id']='r000002a1'
    f.c.knowledge.record(imported);f.c.attempts=f.c.outcomes=f.c.valid_hypotheses=1
    with pytest.raises(ValueError,match='raw request journal'):f.worker.assert_journal(f.log.path,f.log.anchor())


def test_journal_physical_order_allows_new_resumed_segment_but_not_old_id(capsule,tmp_path):
    f=construct(capsule,tmp_path);import_(f,capsule);events=rows(f.log.path)
    events.extend([dict(event='run_start',event_seq=1,segment_id='new',resumed=True),
        dict(event='research',phase='attempt',event_seq=2,segment_id='new',experiment_id='r000002a1')])
    external.assert_import_journal(f.c,events)
    events[-1]['experiment_id']='r000001a1'
    with pytest.raises(ValueError,match='collision'):external.assert_import_journal(f.c,events)


@pytest.mark.parametrize('normalized',[False,True])
def test_imported_child_repetition_rejected_by_real_worker_before_evaluator(capsule,tmp_path,normalized):
    raw=next(e['response'] for e in map(json.loads,capsule['source']['journal'].splitlines()) if e.get('action')=='request_result')
    if normalized:
        payload=json.loads(raw);payload['body']=payload['body'].replace('\n','\n\n',1);raw=json.dumps(payload)
    f=construct(capsule,tmp_path,text=raw);import_(f,capsule)
    def forbidden(code):pytest.fail('repeated historical source reached evaluator')
    result=f.worker.run(snapshot(f),f.model,forbidden,base_temp=.2)
    assert isinstance(result,NoChild) and result.reason=='generation_repeated' and len(f.model.calls)==1
    assert f.worker.evaluations==0 and f.worker.seen_proofs==[] and f.worker.repeated_proposals==1
    assert f.c.attempts==f.c.outcomes==1 and f.c.knowledge.experiments[-1]['id']=='r000002a1'
    f.worker.assert_journal(f.log.path,f.log.anchor())
    restored=construct(capsule,tmp_path/'restored');restored.c.load_state(f.c.state());restored.worker.load_state(f.worker.state())
    restored.worker.assert_journal(f.log.path,f.log.anchor())


def test_trimmed_observation_keeps_id_floor_and_source_provenance(capsule,tmp_path):
    f=construct(capsule,tmp_path);import_(f,capsule)
    # Model the bounded KB's ordinary-window trimming; immutable seen identity and
    # the fragment snapshot still retain the old observation's provenance.
    f.c.knowledge._experiments=[];f.c.conclusions={}
    restored=external.validate_loaded(external.state(f.c),config=f.cfg,metric=f.c.metric,
        knowledge=f.c.knowledge,conclusions=f.c.conclusions)
    assert restored['attempt_id_floor']==1 and external.experiment_number(f.c,1)==2
    f.worker.assert_journal(f.log.path,f.log.anchor())


def test_policy_drift_or_missing_import_refuses_new_id_without_counter_increment(capsule,tmp_path,monkeypatch):
    f=construct(capsule,tmp_path/'missing')
    with pytest.raises(ValueError,match='not been imported'):external.experiment_number(f.c,1)
    assert f.c.attempts==0
    g=construct(capsule,tmp_path/'drift');import_(g,capsule)
    monkeypatch.setattr(external,'identity',lambda:{'changed':True})
    with pytest.raises(ValueError,match='policy changed'):external.experiment_number(g.c,1)
    assert g.c.attempts==0
