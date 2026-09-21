"""Committed cross-island evidence, not publication/claims, controls visibility."""
from copy import deepcopy
import json

import pytest

from sciel.research_tools import shared_evidence
from sciel.research_tools.research_steps import new_state
from test_research_tool_controller import IO, LLM, prepare, snapshot
from test_research_tool_result_flow import result
from test_sequential_research_episode import build, decision, finish


def collect(tools,island=1):
    return tools.lifecycle.shared_observations(tools.library,island,tools.task_identity)


def test_unpublished_negative_experiment_shared_after_commit_and_resume(tmp_path):
    tools,backend,events,costs=build(tmp_path,[result({'gain':0,'seed':17})],shared_observations=True)
    current=snapshot()
    episode=tools.execute(prepare(tools,current),current,LLM([decision(),finish(reason='No gain at the measured budget.')]),io=IO())
    assert collect(tools)['entries']==[]  # Saved but uncommitted artifacts are not shared.
    tools.commit(episode)
    before=collect(tools)
    assert before['total_checks']==1 and before['entries_omitted']==0
    row=before['entries'][0]
    assert row['receipt_id']==episode['receipt_ids'][0]
    assert row['island_id']==0 and row['candidate_id']=='p0'
    assert row['publication']=='locally_tested' and tools.state()['published']==[]
    assert row['measurement']['output']['observations']=={'gain':0,'seed':17}
    assert row['decision']['conclusion']=='refuted'
    assert row['decision']['interpretation']=='No gain at the measured budget.'
    assert row['task']==tools.task_identity
    assert tools.context(snapshot(1))[0]['source']['receipt_id']==row['receipt_id']
    tools.load_state(deepcopy(tools.state()))
    assert collect(tools)==before
    assert len(backend.calls)==1 and costs==[.03,.03]


def test_failed_tool_result_is_shared_as_failure_not_negative_measurement(tmp_path):
    tools,_,_,_=build(tmp_path,[result(status='tool_error',error='wrong shape')],shared_observations=True)
    current=snapshot()
    episode=tools.execute(prepare(tools,current),current,LLM([decision(),finish(conclusion='inconclusive')]),io=IO())
    tools.commit(episode)
    row=collect(tools)['entries'][0]
    assert row['status']=='tool_error' and row['publication']=='unpublished'
    assert row['measurement']['error']=='wrong shape'
    assert row['decision']['conclusion']=='inconclusive'


def test_all_history_survives_retained_card_replacement_and_task_filter(tmp_path):
    tools,_,_,_=build(tmp_path,[result({'gain':0}),result({'gain':1})],shared_observations=True,yield_after_success=True)
    for i in range(2):
        current=snapshot(iteration=i)
        episode=tools.execute(prepare(tools,current),current,LLM([decision(code=decision()['code']+f'# version {i}\n')]),io=IO())
        tools.commit(episode)
    assert len(tools.lifecycle._state['cards'])==1
    view=collect(tools)
    assert view['total_checks']==2 and len(view['check_index'])==2
    assert {r['measurement']['output']['observations']['gain'] for r in view['entries']}=={0,1}
    other=tools.lifecycle.shared_observations(tools.library,1,{'task':'other','task_hash':'other'})
    assert other['total_checks']==0


def test_tampered_receipt_cannot_become_shared_evidence(tmp_path):
    tools,_,_,_=build(tmp_path,[result({'gain':0})],shared_observations=True,yield_after_success=True)
    current=snapshot();episode=tools.execute(prepare(tools,current),current,LLM([decision()]),io=IO());tools.commit(episode)
    path=tools.path/(episode['receipt_ids'][0]+'.json')
    data=json.loads(path.read_text());data['payload']['receipt']['output']['observations']['gain']=999
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError,match='invalid immutable'):
        collect(tools)


@pytest.mark.parametrize('budget',[8000,64000])
def test_shared_prompt_delivery_retains_exact_receipt_and_bounds(tmp_path,budget):
    tools,_,events,_=build(tmp_path,[result({'gain':0,'seed':9})],shared_observations=True,max_prompt_chars=budget,yield_after_success=True)
    current=snapshot();episode=tools.execute(prepare(tools,current),current,LLM([decision()]),io=IO());tools.commit(episode)
    foreign=snapshot(1,2)
    plan=prepare(tools,foreign)
    plan['shared_observations']=collect(tools)
    prompt=tools._prompt(plan,foreign,[],[],0,research_state=new_state())
    assert len(prompt)<=budget
    view=json.JSONDecoder().raw_decode(prompt.split(shared_evidence.MARKER,1)[1])[0]
    assert view['total_checks']==1
    assert view['entries'] and view['entries'][0]['receipt_id']==episode['receipt_ids'][0]
    delivered=[event for event in events if event['phase']=='shared_observations_delivered'][-1]
    assert delivered['delivered'][0]['receipt_id']==episode['receipt_ids'][0]
    assert delivered['recipient_candidate_id']=='p1'
    assert len(delivered['prompt_sha256'])==64
    assert 'actual use or adoption is not established' in delivered['scope']


def test_small_shared_projection_is_whole_and_honest(tmp_path):
    tools,_,_,_=build(tmp_path,[result({'large':[7]*5000,'gain':0})],shared_observations=True,yield_after_success=True)
    current=snapshot();episode=tools.execute(prepare(tools,current),current,LLM([decision()]),io=IO());tools.commit(episode)
    view=collect(tools)
    text=shared_evidence.prompt(view,450)
    decoded=json.JSONDecoder().raw_decode(text.split(shared_evidence.MARKER,1)[1])[0]
    assert len(text)<=450 and decoded['entries']==[] and decoded['omitted_count']==1


def test_unchanged_heads_reuse_authenticated_view_without_rereading_receipts(tmp_path,monkeypatch):
    tools,_,_,_=build(tmp_path,[result({'gain':0})],shared_observations=True,yield_after_success=True)
    current=snapshot();episode=tools.execute(prepare(tools,current),current,LLM([decision()]),io=IO());tools.commit(episode)
    calls=[]; original=tools.lifecycle._receipt
    def counted(*args):calls.append(args[0]);return original(*args)
    monkeypatch.setattr(tools.lifecycle,'_receipt',counted)
    first=collect(tools);second=collect(tools,island=2)
    assert len(calls)==1 and first['entries']==second['entries']
    assert first['recipient_island_id']==1 and second['recipient_island_id']==2
    path=tools.path/(episode['receipt_ids'][0]+'.json')
    path.write_bytes(path.read_bytes()+b' ')
    with pytest.raises(ValueError,match='immutable shared evidence file changed'):
        collect(tools)


def test_complete_nested_v32_measurements_fit_shared_prompt_without_inner_clipping(tmp_path):
    from test_research_result_delivery_budget import saved_output
    output=saved_output();frozen=deepcopy(output)
    tools,_,events,_=build(tmp_path,[output],shared_observations=True,max_prompt_chars=64000,yield_after_success=True)
    current=snapshot();episode=tools.execute(prepare(tools,current),current,LLM([decision()]),io=IO());tools.commit(episode)
    foreign=snapshot(1,2);plan=prepare(tools,foreign);plan['shared_observations']=collect(tools)
    prompt=tools._prompt(plan,foreign,[],[],0,research_state=new_state())
    view=json.JSONDecoder().raw_decode(prompt.split(shared_evidence.MARKER,1)[1])[0]
    row=view['entries'][0]
    assert row['measurement']['output']['observations']==frozen['observations']
    for role in ('selected','population_best'):
        got=row['measurement']['output']['observations']['roles'][role]
        assert got['lp_at_saved_centers']==frozen['observations']['roles'][role]['lp_at_saved_centers']
        assert got['polish']==frozen['observations']['roles'][role]['polish']
    assert not row.get('detail_omitted')
    assert len(prompt)<=64000 and output==frozen


def test_eight_k_prompt_with_protocol_shared_result_and_repair_still_fits(tmp_path):
    from test_research_experimental_protocol import protocol
    tools,_,_,_=build(tmp_path,[result({'gain':0,'seed':9})],shared_observations=True,
        experimental_protocol=True,max_prompt_chars=8000,yield_after_success=True)
    current=snapshot();episode=tools.execute(prepare(tools,current),current,LLM([decision(experimental_protocol=protocol())]),io=IO());tools.commit(episode)
    foreign=snapshot(1,2);plan=prepare(tools,foreign);plan['shared_observations']=collect(tools)
    prompt=tools._prompt(plan,foreign,[],[{'status':'proposal_invalid','reason':'REPAIR_EXACT_CAUSE'}],1,research_state=new_state())
    assert len(prompt)<=8000 and 'REPAIR_EXACT_CAUSE' in prompt
    view=json.JSONDecoder().raw_decode(prompt.split(shared_evidence.MARKER,1)[1])[0]
    assert view['total_checks']==1
    assert len(view['entries'])+view['omitted_count']==1
    assert 'EXPERIMENT PROTOCOL' in prompt
