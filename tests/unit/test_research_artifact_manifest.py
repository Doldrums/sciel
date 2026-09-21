"""Authoring metadata uses saved data only; no generated code or leaves run."""
from copy import deepcopy
import hashlib
import json
from types import SimpleNamespace

import pytest

from sciel.engine.research_artifacts import capture_evaluation, saved_evaluation
from sciel.engine.research_targets import tool_inputs
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.methods.models import canonical_bytes, content_hash
from sciel.research_tools.artifact_manifest import artifact_manifest, MAX_PARSE_BYTES
from sciel.research_tools.research_steps import new_state
from test_research_tool_controller import snapshot, prepare, LLM, IO
from test_sequential_research_episode import build, decision, finish
from test_research_tool_result_contract import result
from test_tool_input_projection import rendered_view, pointer_value


def saved(artifacts):
    return {'status':'available','reference':{'sha256':'a'*64},
            'record':{'result':{'artifacts':artifacts}}}


def entries(value, budget=20000):
    manifest = artifact_manifest(saved(value), '/fixture/artifacts', budget)
    assert len(json.dumps(manifest, ensure_ascii=True, sort_keys=True,
                          separators=(',', ':'))) <= budget
    return {row['key']:row for row in manifest['entries']}


def bind_artifacts(tmp_path, current, artifacts, best_artifacts=None):
    targets = current.research_targets
    engine = SimpleNamespace(run_dir=tmp_path, task=SimpleNamespace(task_hash=targets['task_hash']),
        cfg={},log=SimpleNamespace(run_id=targets['run_id']))
    for role in ('selected','population_best'):
        target = targets[role]
        values = best_artifacts if role == 'population_best' and best_artifacts is not None else artifacts
        outcome = CascadeOutcome(EvalResult(True,.7,{'quality':.7},artifacts=deepcopy(values)),
                                 raw_first_score=.7,reached='full')
        capture_evaluation(engine,target['source'],outcome,'saved-'+role)
        target['saved_evaluation'] = saved_evaluation(engine,target['source'],'saved-'+role)
    targets['sha256'] = content_hash({k:v for k,v in targets.items() if k!='sha256'})
    return current


def test_generic_nested_json_is_discoverable_without_copying_observations():
    text = json.dumps({'schema':'generic-saved/7','state':{'vectors':[[123.456,9],[8,7]],
                         'weights':[11,22]},'success':True,'message':'DO_NOT_COPY_OBSERVATION'})
    row = entries({'arbitrary_dataset':text})['arbitrary_dataset']
    assert row['outer_type']=='string' and row['json_syntax']=='valid'
    assert row['declared_schema']=='generic-saved/7'
    assert row['structure_scope']=='json_decoded_value'
    fields = row['structure']['fields']
    assert fields['state']['fields']['vectors']['shape']==[2,2]
    assert fields['state']['fields']['weights']['shape']==[2]
    assert fields['success']=={'type':'boolean'}
    assert '123.456' not in json.dumps(row) and 'DO_NOT_COPY_OBSERVATION' not in json.dumps(row)
    assert row['sha256']==hashlib.sha256(text.encode()).hexdigest()


@pytest.mark.parametrize('value,status,kind', [
    ('ordinary prose','not_strict_json',None), ('{broken','not_strict_json',None),
    ('NaN','not_strict_json',None), ('1e999','not_strict_json',None),
    ('{"x":1,"x":2}','not_strict_json',None), ('null','valid','null'),
    ('true','valid','boolean'), ('42','valid','integer'), ('3.5','valid','number'),
    ('"a JSON string"','valid','string'),
    ('['*30+'0'+']'*30,'omitted_depth_limit',None),
    ('x'*(MAX_PARSE_BYTES+1),'omitted_byte_limit',None),
])
def test_strict_syntax_probe_is_bounded_and_never_invents_a_format(value,status,kind):
    row = entries({'value':value})['value']
    assert row['json_syntax']==status
    if kind:
        assert row['structure']['type']==kind
    else:
        assert 'structure' not in row


@pytest.mark.parametrize('value,shape,status', [
    ([[1,2],[3,4]],[2,2],'verified'), ([[],[]],[2,0],'verified'),
    ([[1],[2,3]],None,'ragged_or_mixed'), ([1,[2]],None,'ragged_or_mixed'),
    (list(range(1000)),None,'omitted_node_limit'),
])
def test_shape_is_only_reported_after_full_verification(value,shape,status):
    row = entries({'data':value})['data']
    assert row['json_syntax']=='not_a_string' and row['structure_scope']=='outer_value'
    assert row['structure']['shape_status']==status
    assert row['structure'].get('shape')==shape
    assert row['structure']['length']==len(value)


def test_exact_keys_pointer_escaping_and_deterministic_order():
    values={'雪~/🙂':json.dumps({'a/b':{'~value':[1,2]}}),'other':'text'}
    a=artifact_manifest(saved(values),'/fixture/artifacts',20000)
    b=artifact_manifest(saved(dict(reversed(list(values.items())))),'/fixture/artifacts',20000)
    assert a==b
    for row in a['entries']:
        assert pointer_value({'fixture':{'artifacts':values}},row['pointer'])==values[row['key']]
    assert a['entries'][1]['structure']['fields']['a/b']['fields']['~value']['shape']==[2]


def test_small_budget_omits_whole_fields_and_never_modifies_raw_data():
    values={'a':'[1,2,3]','b':'x'*(MAX_PARSE_BYTES+100),'雪🙂'*300:'{"nested":1}'}
    original=deepcopy(values)
    for budget in (0,500,900,1400):
        manifest=artifact_manifest(saved(values),'/fixture/artifacts',budget)
        if budget:
            assert len(json.dumps(manifest,ensure_ascii=True,sort_keys=True,separators=(',',':'))) <= budget
        assert manifest['entries_omitted']==len(values)-len(manifest['entries'])
        assert all(row['key'] in values for row in manifest['entries'])
        assert manifest['record_sha256']=='a'*64
    assert values==original


def test_shallow_fallback_keeps_root_keys_without_claiming_nested_detail():
    value=json.dumps({'schema':'generic/1','a':{'nested':{'array':[[1,2],[3,4]]}},
                      'b':{'nested':{'array':[[1,2],[3,4]]}}})
    found=False
    for budget in range(500,1400,25):
        manifest=artifact_manifest(saved({'blob':value}),'/x',budget)
        row=manifest['entries'][0] if manifest['entries'] else {}
        structure=row.get('structure',{})
        if structure.get('nested_details_omitted'):
            assert structure['keys']==['a','b','schema'] and structure['fields_omitted']==0
            assert 'shape' not in json.dumps(structure)
            found=True
            break
    assert found


def test_unavailable_and_empty_artifacts_are_distinct():
    absent=artifact_manifest({'status':'unavailable','reference':None,'record':None},'/x',1000)
    empty=artifact_manifest(saved({}),'/x',1000)
    assert absent['status']=='unavailable' and absent['record_sha256'] is None
    assert empty['status']=='available' and empty['entries']==[] and empty['entries_omitted']==0


def test_first_and_repair_prompts_show_formats_while_inputs_remain_exact(tmp_path):
    current=bind_artifacts(tmp_path,snapshot(),{
        'unfamiliar_blob':json.dumps({'schema':'any-domain/3','state':{'points':[[1,2],[3,4]]}}),
        'log':'opaque text'})
    before=deepcopy(current.research_targets)
    tools,backend,_,_=build(tmp_path,[result(status='tool_error',error='missing expected field'),result({'checked':True})])
    llm=LLM([decision(),decision(code=decision()['code']+'# repair\n'),finish()])
    episode=tools.execute(prepare(tools,current),current,llm,io=IO())
    for prompt in llm.prompts:
        view=rendered_view(prompt)
        for reference in view['references']:
            manifest=reference['artifact_manifest']
            rows={x['key']:x for x in manifest['entries']}
            assert rows['unfamiliar_blob']['declared_schema']=='any-domain/3'
            assert rows['unfamiliar_blob']['structure']['fields']['state']['fields']['points']['shape']==[2,2]
            assert reference['saved_evaluation_sha256']==current.research_targets[
                reference['role']]['saved_evaluation']['reference']['sha256']
    assert 'missing expected field' in llm.prompts[1]
    assert current.research_targets==before
    for call in backend.calls:
        assert call['inputs']==tool_inputs(current,tools.task_identity,{})
        assert call['inputs']['research_targets']==before
    for ident in episode['receipt_ids']:
        receipt=tools._read_artifact(ident,'invocation')
        assert receipt['receipt']['input_sha256']==content_hash(receipt['inputs'])
    tools.commit(episode);tools.load_state(deepcopy(tools.state()))


@pytest.mark.parametrize('background',[False,True])
def test_saturated_8k_prompt_keeps_both_artifact_roles_refs_and_repair_cause(tmp_path,background):
    current=bind_artifacts(tmp_path,snapshot(),{'alpha':'{"nested":{"array":[[1,2],[3,4]]}}'},
        {'beta':'{"schema":"other-task/1","matrix":[[5],[6]]}','oversized':'x'*20000})
    current.task_prompt='long task '*10000
    current.research_search_evidence={'schema':'sciel-search-evidence/1',
        'selected_runtime':{'observed_wall_seconds_per_trial':1.5},
        'recent_admissions':[{'program_id':str(i),'source_sha256':str(i).zfill(64)} for i in range(12)]}
    tools,_,_,_=build(tmp_path,[],max_prompt_chars=8000,background=background)
    plan=prepare(tools,current);state=new_state()
    for key in ('question','decision_if_supported','decision_if_refuted','experiment','next_decision'):
        state[key]='界🙂'*700
    state['current_experiment']=state['experiment']
    state['steps']=[{'action':'use','status':'success','receipt_id':format(i+1,'064x'),'tool_id':'a'*64} for i in range(32)]
    for feedback in ([],[{'tool_id':'a'*64,'receipt_id':'b'*64,'result':{'status':'tool_error',
        'error':'KeyError: missing_expected_field'},'stderr':'Traceback: KeyError: missing_expected_field'}]):
        prompt=tools._prompt(plan,current,[],feedback,31,research_state=state,scientific_step=1)
        assert len(prompt)<=8000
        view=rendered_view(prompt)
        assert len(view['references'])==2
        for row in view['references']:
            manifest=row['artifact_manifest']
            assert row['saved_evaluation_status']=='available'
            assert row['saved_evaluation_sha256']==current.research_targets[
                row['role']]['saved_evaluation']['reference']['sha256']
            pointer=view['access_paths'][row['role']]+view['access_paths']['fields']['saved_artifacts']
            assert pointer_value(tool_inputs(current,tools.task_identity,{}),pointer)
            assert manifest['entries'], 'minimum prompt must deliver artifact keys/types, not just omissions'
            assert manifest['entries_omitted']>=0
        if feedback:
            assert 'missing_expected_field' in prompt
    assert canonical_bytes(current.research_targets)==canonical_bytes(tool_inputs(current,tools.task_identity,{})['research_targets'])


@pytest.mark.parametrize('independent', [False, True])
@pytest.mark.parametrize('sequential,yield_after_success',[(False,False),(True,False),(True,True)])
def test_combined_minimum_context_preserves_artifact_keys_and_exact_repair_receipt(tmp_path,monkeypatch,sequential,yield_after_success,independent):
    from sciel.engine.research_cells_context import compact_context
    from sciel.research_tools.followups import build_followup
    from test_method_followups import ledger, record
    from test_replay_feedback_contract import feedback
    from test_tool_prior_results_prompt import known_jobs, prior_card, target_fixture
    current=bind_artifacts(tmp_path,target_fixture('failed',huge=True),
        {'generic_blob':'{"schema":"generic/1","matrix":[[1,2],[3,4]]}'})
    synthetic_prior = prior_card()
    current.research_cells_context=compact_context([synthetic_prior])
    tools,_,_,_=build(tmp_path,[],max_prompt_chars=8000,background=True,
        yield_after_success=yield_after_success,independent_research=independent)
    # This direct formatter case deliberately invents prior memory, jobs and
    # replay state to saturate the minimum prompt. It does not claim artifact
    # provenance. Bypass only this complete synthetic card on this instance;
    # any different or actually invoked card keeps production authentication.
    authenticate = tools.lifecycle._delivery_measurement
    def render_prior(card, episode=None):
        return card if card == synthetic_prior else authenticate(card, episode)
    monkeypatch.setattr(tools.lifecycle, '_delivery_measurement', render_prior)
    plan=prepare(tools,current)
    if independent:
        from test_research_continuation_queue import lease
        assert plan is None
        _,current,plan=lease(tools)
        current.research_cells_context=compact_context([synthetic_prior])
    plan['visible_jobs']=known_jobs()
    plan['author_replay_feedback']=[feedback(result({'probe':1}),result({'probe':2}))]
    history=ledger();history.add(build_followup(record(1)))
    plan['method_followups']=history.planner_view(['c'*64],max_chars=8000)
    item={'tool_id':'a'*64,'receipt_id':'b'*64,
        'result':{'status':'tool_error','host_status':'ok','error':'terminal repair cause'+' long diagnostic'*1000},
        'stderr':'Traceback\n'+'middle stack detail\n'*1000+"KeyError: 'measurement_status'"}
    state=new_state() if sequential else None
    if state is not None:
        state['question']='Does the saved state improve?'
        state['next_decision']='Compare a controlled continuation.'
    original=deepcopy(item)
    prompt=tools._prompt(plan,current,[],[item],1,research_state=state)
    assert item==original
    assert len(prompt)<=8000
    assert 'terminal repair cause' in prompt and "KeyError: 'measurement_status'" in prompt
    assert 'Receipt: '+item['receipt_id'] in prompt
    if state is not None:
        assert state['question'] in prompt and state['next_decision'] in prompt
        assert 'finish closes without executing next_decision advice' in prompt
    if yield_after_success:
        assert 'YIELD:' in prompt and 'question stays open' in prompt
    view=rendered_view(prompt)
    for row in view['references']:
        assert row['saved_evaluation_sha256']==current.research_targets[row['role']]['saved_evaluation']['reference']['sha256']
        assert any(entry['key']=='generic_blob' and entry['outer_type']=='string'
                   for entry in row['artifact_manifest']['entries'])
    for marker in ('RESEARCH_CHECK_RESULTS_VIEW_JSON:\n','REPLAY_FEEDBACK_JSON:\n',
                   'KNOWN JOBS (compact last committed observations):\n','METHOD_FOLLOWUPS_VIEW_JSON:\n'):
        assert isinstance(json.JSONDecoder().raw_decode(prompt.split(marker,1)[1])[0],dict)
    if sequential:
        from sciel.research_tools.research_steps import TEXT_FIELDS, proposal_fields
        with pytest.raises(ValueError) as caught:
            proposal_fields({'action':'finish', 'conclusion':'inconclusive'})
        invalid = {'status':'proposal_invalid', 'reason':str(caught.value)}
        repair = tools._prompt(plan,current,[],[invalid],1,research_state=state)
        assert len(repair) <= 8000
        for field in TEXT_FIELDS:
            assert field + ' must be a nonempty concrete research field' in repair
        assert 'nonempty text, even finish/skip' in repair
        assert 'If no experiment ran, say so and why' in repair
        assert rendered_view(repair)['access_paths'] == view['access_paths']
