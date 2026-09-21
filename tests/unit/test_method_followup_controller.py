"""Actual controller artifact/restore and frozen prompt integration, no provider."""
from copy import deepcopy
import json

import pytest

from sciel.engine.research_cells_context import compact_context
from sciel.methods.models import content_hash
from sciel.research_tools.followups import capture_exposures
from test_method_followups import record, observed, sha, seal
from test_research_tool_controller import setup, snapshot, prepare, LLM, IO, create


def completed(tmp_path, **config):
    tools, backend, events, costs = setup(tmp_path, **config)
    source = snapshot()
    episode = tools.execute(prepare(tools, source), source, LLM([create()]), io=IO())
    tools.commit(episode)
    return tools, source, episode, events


def evidence(card, number=1, **options):
    item = record(number, **options)
    item['method_exposures'] = capture_exposures(compact_context([card]), prompt_sha256=sha('actual prompt'))
    return item


@pytest.mark.parametrize('retained', [False, True])
def test_actual_fresh_and_retained_card_join_restore_idempotently(tmp_path, retained):
    tools, source, episode, events = completed(tmp_path)
    card = tools.context(source)[0] if retained else episode['cards'][-1]
    if retained:
        assert card['revision_id'] != episode['cards'][-1]['revision_id']
    item = evidence(card)
    assert tools.record_followup(item, child_program_id='p2', observed_record=observed())
    assert not tools.record_followup(item, child_program_id='p2', observed_record=observed())
    saved = tools.state()
    row = saved['followups']['records'][0]
    artifact = tools._read_artifact(row['evidence_artifact_id'], 'method_followup_evidence')
    assert artifact['research_record'] == item
    assert artifact['observed_record'] == observed()
    restored, _, _, _ = setup(tmp_path)
    restored.load_state(saved)
    assert restored.state() == saved
    assert restored.followups.planner_view(episode['tool_ids'], max_chars=8000)['records'][0]['child_program_id'] == 'p2'
    assert len([e for e in events if e['phase'] == 'method_followup']) == 1
    assert row['model_use'] == row['causal_benefit'] == 'unknown'


@pytest.mark.parametrize('status,dispatch', [('no_child', 'returned'), ('eval_failed', 'returned'),
    ('llm_error', 'unknown'), ('llm_error', 'not_dispatched')])
def test_no_child_failure_and_unknown_outcomes_survive_without_new_episode(tmp_path, status, dispatch):
    tools, source, _, _ = completed(tmp_path)
    item = evidence(tools.context(source)[0], status=status, dispatch=dispatch)
    assert tools.record_followup(item)
    saved = tools.state()
    tools.load_state(saved)
    row = saved['followups']['records'][0]
    assert row['outcome']['child_program_id'] is None
    assert row['outcome']['score'] is None
    assert row['generation']['dispatch'] == dispatch


def test_omitted_card_does_not_create_followup_or_artifact(tmp_path):
    tools, _, _, _ = completed(tmp_path)
    item = record()
    item['method_exposures'] = capture_exposures({'schema': 'sciel-research-cells-context/1',
        'cards': [], 'omitted_count': 1, 'omission_reasons': {'prompt_budget': 1}}, prompt_sha256=sha('p'))
    before = set(tools.path.iterdir())
    assert not tools.record_followup(item)
    assert set(tools.path.iterdir()) == before


@pytest.mark.parametrize('field', ['status', 'artifact', 'source', 'revision'])
def test_changed_receipt_reference_is_rejected_before_followup_commit(tmp_path, field):
    tools, source, _, _ = completed(tmp_path)
    item = evidence(tools.context(source)[0])
    card = item['method_exposures']['cards'][0]
    if field == 'status':
        card['observation_status'] = 'tool_error'
    elif field == 'source':
        card['source']['target_source_sha256'] = '1' * 64
    elif field == 'revision':
        card['revision_id'] = '1' * 64
    else:
        path = tools.path / (card['source']['receipt_id'] + '.json')
        body = json.loads(path.read_text())
        body['payload']['receipt']['output'] = {'changed': True}
        path.write_text(json.dumps(body))
    item['method_exposures'] = seal(item['method_exposures'])
    with pytest.raises((ValueError, FileNotFoundError)):
        tools.record_followup(item)
    assert tools.state()['followups']['records'] == []


@pytest.mark.parametrize('change', ['failure', 'child', 'evidence'])
def test_resealed_followup_cannot_replace_immutable_evidence_and_restore_is_atomic(tmp_path, change):
    tools, source, _, _ = completed(tmp_path)
    tools.record_followup(evidence(tools.context(source)[0]), child_program_id='p2', observed_record=observed())
    original = tools.state()
    bad = deepcopy(original)
    row = bad['followups']['records'][0]
    if change == 'failure':
        row['outcome']['failure'] = 'Invented later prose'
    elif change == 'child':
        row['outcome']['child_program_id'] = None
        row['outcome']['observed_record'] = None
    else:
        path = tools.path / (row['evidence_artifact_id'] + '.json')
        payload = json.loads(path.read_text())
        payload['payload']['research_record']['failure'] = 'modified'
        path.write_text(json.dumps(payload))
    bad['followups']['records'][0] = seal(row)
    bad['followups'] = seal(bad['followups'])
    with pytest.raises(ValueError):
        tools.load_state(bad)
    assert tools.state() == original


@pytest.mark.parametrize('budget', [8000, 64000])
def test_frozen_planner_history_has_whole_rows_or_explicit_omissions(tmp_path, budget):
    tools, source, _, _ = completed(tmp_path, max_prompt_chars=budget)
    tools.record_followup(evidence(tools.context(source)[0]))
    next_source = snapshot(iteration=3)
    next_source.research_cells_context = tools.context(source)
    plan = prepare(tools, next_source)
    prompt = tools._prompt(plan, next_source, plan['visible_tool_ids'], [], 0)
    tools.record_followup(evidence(tools.context(source)[0], 2, status='no_child'))
    assert tools._prompt(plan, next_source, plan['visible_tool_ids'], [], 0) == prompt
    assert len(prompt) <= budget
    view = json.JSONDecoder().raw_decode(prompt.split('METHOD_FOLLOWUPS_VIEW_JSON:\n')[1])[0]
    assert view['eligible_count'] == 1
    assert len(view['records']) + view['omitted_count'] == 1
    assert 'causal benefit are unknown' in view['scope']
    inputs = json.JSONDecoder().raw_decode(prompt.split('FROZEN_INPUT_VIEW_JSON:\n')[1])[0]
    assert inputs['access_paths']['fields']['score'] == '/measurement/record/child_observation/score'
    if budget == 64000:
        assert len(view['records']) == 1
        assert view['records'][0]['canonical_evaluation']['score'] == 0.0
    assert content_hash({k: v for k, v in plan.items() if k != 'action_id'}) == plan['action_id']
