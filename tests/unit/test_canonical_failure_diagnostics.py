"""Evaluator diagnostics stay source-bound and actionable under real prompt limits."""
from copy import deepcopy
import json
from unittest.mock import Mock

import pytest

from sciel.engine.research_candidate import evaluate_submission
from sciel.engine.types import Snapshot
from sciel.engine.validation import source_sha256
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.methods.models import content_hash
from sciel.research_tools import canonical_diagnostics as diagnostics, shared_evidence
from test_research_canonical_citations import SOURCE, designed, persistent
from test_research_tool_controller import IO, LLM, prepare, snapshot
from test_research_tool_result_flow import result
from test_sequential_research_episode import build, current_receipt, decision, finish

ERROR = 'RuntimeError: required callable missing from candidate'


def failed(stderr='', artifact=None):
    value = EvalResult(False, 0., {'validity':0., 'queries':0., 'native_metric':7.},
        stderr=stderr, artifacts={'trials':json.dumps(artifact or [{'stderr':ERROR}])}, timing=.25)
    outcome = CascadeOutcome(value,0.,'smoke',False)
    outcome.evaluation_id = 'evaluation:fixture-run:wave:1:research:0:0'
    return outcome


def evaluate(source, provenance, outcome):
    snap = Snapshot(0,'full_rewrite','mock',1,'p0','def run_search():\n    return 0\n',0.,
                    required_entrypoints=['run_search'])
    return evaluate_submission(source, provenance, snap, Mock(return_value=outcome))[1]


@pytest.mark.parametrize('budget',[8000,64000])
@pytest.mark.parametrize('long_unicode',[False,True])
def test_failure_from_exact_evaluator_to_following_and_restored_planner(tmp_path,budget,long_unicode):
    trace = ('Traceback:\n' + ('雪☃\n' * 3000) + ERROR) if long_unicode else ERROR
    returned = failed('No valid trials; inspect diagnostics.',[{'stderr':trace,'error':None}])
    returned.result.artifacts['error'] = 'candidate failed'
    original = deepcopy(returned.result.to_json())
    tools,backend,events,costs = build(tmp_path,[result({'candidate_source':SOURCE})],
        experimental_protocol=True,shared_observations=True,max_prompt_chars=budget)
    snap=snapshot(); ids={}; prompts=[]
    def submit(prompt):
        ids['tool']=current_receipt(prompt)
        return designed(decision(action='submit_candidate',candidate_receipt_id=ids['tool']),
            kind='closure',budget_action='none',prior_receipt_ids=[ids['tool']])
    def interpret(prompt):
        prompts.append(prompt); assert len(prompt)<=budget
        summary=persistent(prompt)['canonical_evaluations'][-1]
        ids['canonical']=summary['artifact_id']
        assert summary['status']=='eval_failed' and summary['efficacy']=='unknown'
        assert ERROR in json.dumps(summary,ensure_ascii=False)
        assert summary['cases']=={'validity':0.,'queries':0.,'native_metric':7.}
        feedback=prompt.split('PREVIOUS ATTEMPT FEEDBACK:\n',1)[1].split('\n\n',1)[0]
        assert 'Status: success' not in feedback
        return designed(finish(conclusion='inconclusive'),kind='closure',budget_action='none',
            prior_receipt_ids=[ids['tool'],ids['canonical']])
    calls=[]
    def callback(source,provenance):
        calls.append(source);return evaluate(source,provenance,returned)
    episode=tools.execute(prepare(tools,snap),snap,LLM([designed(decision()),submit,interpret]),
        io=IO(),candidate_evaluator=callback)
    assert calls==[SOURCE] and len(backend.calls)==1 and costs==[.03]*3
    assert returned.result.to_json()==original
    saved=tools._read_artifact(ids['canonical'],'candidate_evaluation')['evaluation']
    raw=saved['failure_diagnostics']; assert raw['stderr']==original['stderr']
    assert raw['artifacts']==original['artifacts'] and raw['source_sha256']==source_sha256(SOURCE)
    assert raw['artifacts_sha256']==saved['artifacts_sha256']==content_hash(original['artifacts'])
    tools.commit(episode); checkpoint=deepcopy(tools.state());tools.load_state(checkpoint,expected_run_id='fixture-run')
    cell=episode['plan']['cell_id'];state=tools.state()['lifecycle']['research_state'][cell]
    again=tools._prompt(episode['plan'],snap,[],[],0,research_state=state)
    restored_view = persistent(again)
    assert len(again)<=budget and ERROR in json.dumps(restored_view['canonical_evaluations'],ensure_ascii=False)
    assert restored_view['canonical_evaluations'][0]['artifact_id'] == ids['canonical']
    assert restored_view['canonical_evaluations'][0]['cases'] == saved['cases']
    if budget == 8000 and long_unicode:
        assert 'canonical_evaluations_omitted' not in restored_view
        assert restored_view['other_state_fields_omitted'] is True
    shared=shared_evidence.collect(tools.lifecycle,tools.library,1,tools.task_identity)
    assert ERROR in json.dumps(shared['canonical_evaluations'],ensure_ascii=False)
    assert calls==[SOURCE] and costs==[.03]*3
    forged=deepcopy(raw);forged['artifacts']['trials']=json.dumps([{'stderr':'different reason'}])
    with pytest.raises(ValueError,match='diagnostics differ'):
        diagnostics.validate(forged,source_sha256=source_sha256(SOURCE),evaluation_id=returned.evaluation_id,
                             artifacts_sha256=saved['artifacts_sha256'])


def test_generic_nested_json_errors_unicode_excerpts_and_explicit_omissions():
    outcome=failed('', {'diagnostics':{'stderr':'雪'*4000+'\n'+ERROR},
                         'unrelated_arrays':list(range(20000))})
    block=diagnostics.capture(outcome.result,source_sha256='a'*64,evaluation_id='evaluation:x')
    projected=diagnostics.project(block,500)
    assert len(json.dumps(projected,ensure_ascii=True,separators=(',',':')))<=500
    assert ERROR in json.dumps(projected,ensure_ascii=False)
    entry=projected['entries'][0]
    assert entry['path']=='/artifacts/trials::json/diagnostics/stderr'
    assert entry['omitted_chars']>0 and entry['sha256']==diagnostics._sha('雪'*4000+'\n'+ERROR)
    assert projected['uninspected_values']>0
    assert block['artifacts']==outcome.result.artifacts


def test_empty_or_nonjson_diagnostics_are_not_invented():
    block=diagnostics.capture(EvalResult(False,0.,artifacts={'diagnostics':'{"broken"'}),
        source_sha256='b'*64,evaluation_id=None)
    view=diagnostics.project(block)
    assert view['entries'][0]['text']=='{"broken"' and view['uninspected_values']==1
    empty=diagnostics.capture(EvalResult(False,0.),source_sha256='b'*64,evaluation_id=None)
    assert diagnostics.project(empty)['entries']==[]


def test_fatal_evaluator_exception_is_not_a_diagnostic_or_retry():
    source=SOURCE;prov={'candidate_source_sha256':source_sha256(source)}
    snap=Snapshot(0,'full_rewrite','mock',1,'p0','def run_search():\n    return 0\n',0.,required_entrypoints=['run_search'])
    callback=Mock(side_effect=RuntimeError('unreconciled evaluator transport'))
    with pytest.raises(RuntimeError,match='unreconciled'):
        evaluate_submission(source,prov,snap,callback)
    callback.assert_called_once_with(source)


def test_generic_banner_and_large_observations_do_not_hide_nested_terminal_cause():
    artifact = {'error':'candidate failed','trials':[{
        'observations':list(range(20000)), 'stderr':'Traceback:\n'+('frame\n'*1000)+ERROR}]}
    block=diagnostics.capture(failed('No valid trials.',artifact).result,
        source_sha256='a'*64,evaluation_id='evaluation:x')
    projected=diagnostics.project(block,500)
    assert ERROR in json.dumps(projected,ensure_ascii=False)
    assert projected['entries'][0]['path']=='/artifacts/trials::json/trials/0/stderr'
    assert projected['omitted_entries']>=1 and projected['uninspected_values']>0
    with pytest.raises(ValueError,match='budget cannot fit'):
        diagnostics.project(block,50)
    for budget in (40,80,150,350,700):
        text=diagnostics.feedback({'status':'eval_failed','failure_diagnostics':block},'b'*64,budget)
        assert len(text)<=budget and json.loads(text)['status']=='eval_failed'


def test_real_saved_evaluation_index_cannot_be_bypassed_by_rehashed_diagnostics(tmp_path):
    from types import SimpleNamespace
    from sciel.engine.research_artifacts import capture_evaluation
    engine=SimpleNamespace(run_dir=tmp_path,log=SimpleNamespace(run_id='fixture-run'),
        task=SimpleNamespace(task_hash='fixture-task'),cfg={})
    outcome=failed('Aggregate failure.')
    reference=capture_evaluation(engine,SOURCE,outcome,outcome.evaluation_id)
    raw=diagnostics.capture(outcome.result,source_sha256=source_sha256(SOURCE),
        evaluation_id=outcome.evaluation_id,record_reference=reference)
    library=SimpleNamespace(run_dir=tmp_path)
    def authenticate(block):
        diagnostics.validate(block,source_sha256=source_sha256(SOURCE),
            evaluation_id=outcome.evaluation_id,artifacts_sha256=block['artifacts_sha256'],
            library=library,run_id='fixture-run',task_hash='fixture-task')
    authenticate(raw)
    for change in ('remove_reference','record_hash','stderr','artifacts'):
        forged=deepcopy(raw)
        if change=='remove_reference':forged['record_reference']=None
        elif change=='record_hash':forged['record_reference']['sha256']='f'*64
        elif change=='stderr':
            forged['stderr']='Changed outer cause.'
            forged['stderr_sha256']=diagnostics._sha(forged['stderr'])
        else:
            forged['artifacts']['trials']=json.dumps([{'stderr':'Changed terminal cause.'}])
            forged['artifacts_sha256']=content_hash(forged['artifacts'])
        with pytest.raises(ValueError,match='binding differs|saved evaluator record'):
            authenticate(forged)
    # Loss of the original record is unavailable evidence, not permission to
    # trust a self-consistent rewrite of its diagnostic projection.
    (tmp_path/'research_evidence'/'evaluations'/(reference['sha256']+'.json')).unlink()
    with pytest.raises(ValueError,match='record is unavailable'):
        authenticate(raw)


@pytest.mark.parametrize('budget',[8000,64000])
@pytest.mark.parametrize('kind',['sha256','ast','output_repeat'])
def test_next_planner_keeps_exact_novelty_refusal_and_match_ids(tmp_path,budget,kind):
    tools,backend,_,costs=build(tmp_path,[result({'candidate_source':SOURCE})],max_prompt_chars=budget)
    current=snapshot(); ids={}; calls=[]
    reason='exact_source_duplicate' if kind=='sha256' else 'structural_ast_duplicate'
    value={'status':'rejected','matched_id':'p_existing','reason':reason,'efficacy':'unknown',
        'novelty':{'accept':False,'match_kind':kind,'matched_program_id':'p_existing',
            'matched_evaluation_id':'evaluation:fixture-run:old','reason':reason,
            'neighbors':[{'program_id':'other','distance':.1}]*1000}}
    if kind=='output_repeat':
        value={'status':'output_repeat_rejected','score':.7,'cases':{'native':7.},'efficacy':'unknown',
            'evaluation_id':'evaluation:fixture-run:new',
            'output_novelty':{'accept':False,'status':'exact_recorded_repeat',
                'evaluation_id':'evaluation:fixture-run:new','identity':{'large':list(range(1000))},
                'match':{'program_id':'p_existing','evaluation_id':'evaluation:fixture-run:old',
                    'source_sha256':'a'*64,'island_id':2}}}
        reason='exact_recorded_repeat'
    original=deepcopy(value)
    def submit(prompt):
        ids['tool']=current_receipt(prompt)
        return decision(action='submit_candidate',candidate_receipt_id=ids['tool'])
    def inspect(prompt):
        assert len(prompt)<=budget
        feedback=prompt.split('PREVIOUS ATTEMPT FEEDBACK:\n',1)[1]
        assert 'p_existing' in feedback and 'evaluation:fixture-run:old' in feedback
        assert reason in feedback
        assert value['status'] in feedback
        return finish(conclusion='inconclusive')
    def evaluate(source,provenance):
        calls.append(source);return deepcopy(value)
    tools.execute(prepare(tools,current),current,LLM([decision(),submit,inspect]),
        io=IO(),candidate_evaluator=evaluate)
    assert calls==[SOURCE] and len(backend.calls)==1 and costs==[.03]*3
    assert value==original


def test_canonical_metadata_is_whole_when_it_fits_and_omissions_stay_explicit():
    value={'status':'evaluated','score':.75,'cases':{'native':7.,'n_trials':3},
        'reached':'full','timing_seconds':.125,'efficacy':'unknown',
        'output_novelty':{'status':'new_output','match':None}}
    assert json.loads(diagnostics.feedback(value,'b'*64,2000))=={**value,'receipt_id':'b'*64}
    value={'status':'rejected','matched_id':'p_existing','reason':'exact_source_duplicate',
        'novelty':{'match_kind':'sha256','matched_program_id':'p_existing',
            'matched_evaluation_id':'evaluation:old','neighbors':list(range(10000))}}
    projected=json.loads(diagnostics.feedback(value,'b'*64,550))
    assert projected['matched_id']=='p_existing'
    assert projected['novelty']['matched_evaluation_id']=='evaluation:old'
    assert projected['novelty']['other_fields_omitted'] is True
    assert projected['reason']=='exact_source_duplicate'
    tiny=json.loads(diagnostics.feedback(value,'b'*64,80))
    assert tiny['status']=='rejected' and tiny['details_in']=='canonical_evaluations'


def test_canonical_refusal_feedback_respects_every_small_budget_boundary():
    value={'status':'rejected','matched_id':'p_existing','reason':'exact source repeat '*100,
        'novelty':{'accept':False,'match_kind':'sha256','matched_program_id':'p_existing',
            'matched_evaluation_id':'evaluation:fixture-run:old','reason':'exact_source_duplicate',
            'neighbors':list(range(1000))},'cases':{'native':7.}}
    original=deepcopy(value)
    for budget in range(80,1500):
        text=diagnostics.feedback(value,'b'*64,budget)
        assert len(text)<=budget, (budget,len(text))
        assert json.loads(text)['status']=='rejected'
    assert value==original


def test_oversized_canonical_projection_cannot_fall_back_to_tool_success(monkeypatch):
    import sciel.research_tools.prompt_sections as module
    monkeypatch.setattr(module,'canonical_feedback',lambda *args:'x'*2000)
    item={'status':'canonical_evaluation','receipt_id':'b'*64,
        'result':{'status':'success','host_status':'canonical_evaluator'},
        'output':{'observations':{'canonical_evaluation':{'status':'eval_failed'}}}}
    for budget in (80,141,350,1000):
        text=module._feedback_card([item],budget)
        assert len(text)<=budget and json.loads(text)['status']=='eval_failed'
        assert 'success' not in text


def test_compact_failure_prompt_keeps_positive_canonical_omission_count(tmp_path, monkeypatch):
    # Force the legitimate state-prompt minimum that reports an unavailable
    # canonical body. The compact formatter may omit a zero, never a positive
    # omission count. No canonical result or measurement is invented.
    from sciel.research_tools import prompt_builder as controller_module
    from sciel.research_tools.research_steps import new_state
    original = controller_module.state_prompt
    def with_omission(*args, **kwargs):
        text = original(*args, **kwargs)
        marker, encoded = text.split('\n', 1)
        value = json.loads(encoded)
        value['canonical_evaluations_omitted'] = 1
        return marker + '\n' + json.dumps(value, sort_keys=True, separators=(',', ':'))
    monkeypatch.setattr(controller_module, 'state_prompt', with_omission)
    tools, _, _, _ = build(tmp_path, [], max_prompt_chars=8000,
                           experimental_protocol=True, shared_observations=True)
    current = snapshot()
    prompt = tools._prompt(prepare(tools, current), current, [], [], 0, research_state=new_state())
    view = persistent(prompt)
    assert len(prompt) <= 8000 and view['canonical_evaluations_omitted'] == 1
    assert 'other_state_fields_omitted' in view  # The compact branch actually ran.
