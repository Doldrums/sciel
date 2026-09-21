"""Sharing queue isolation, fairness and cold authentication; inert fake leaves."""
from copy import deepcopy

import pytest

from sciel.research_tools.controller_jobs import JobRefused
from test_research_continuation_queue import queued, lease
from test_research_tool_controller import IO, LLM, snapshot
from test_research_tool_result_flow import result, collect
from test_sequential_research_episode import build, decision, finish


def prepared(tmp_path, *, background=False, open_author=False):
    tools, backend, events, costs = build(tmp_path, [result({'gain': -1})] * 12,
        independent_research=True, share_between_islands=True,
        yield_after_success=True, background=background)
    author = queued(tools, snapshot())
    _, current, plan = lease(tools)
    tools.commit(tools.execute(plan, current, LLM([decision()]), io=IO()))
    if not open_author:
        _, current, plan = lease(tools)
        tools.commit(tools.execute(plan, current, LLM([finish()]), io=IO()))
    return tools, backend, events, costs, author


def commit_next(tools):
    _, current, plan = lease(tools)
    answer = LLM([finish(conclusion='inconclusive')]) if plan['work_kind'] == 'question' else LLM([])
    outcome = tools.execute(plan, current, answer, io=IO())
    tools.commit(outcome)
    return plan, outcome


def test_no_recipient_is_retained_then_full_pipeline_preserves_both_questions(tmp_path):
    tools, backend, events, costs, author = prepared(tmp_path)
    tool_id = tools.state()['registered'][0]
    before = tools.state()
    assert before['sharing']['waiting_methods'] == {tool_id: 'no_recipient_yet'}
    assert before['sharing']['items'] == {}
    assert tools.reserve_continuations(lambda saved, state, **work: saved) == []
    tools.load_state(deepcopy(before))
    recipient = queued(tools, snapshot(1))
    plans = [commit_next(tools)[0] for _ in range(4)]
    assert [p['stage'] for p in plans] == ['independent_replay', 'research', 'adopt_trial', 'use_again']
    after = tools.state()
    assert after['lifecycle']['research_state'][author] == before['lifecycle']['research_state'][author]
    assert after['lifecycle']['cards'][author] == before['lifecycle']['cards'][author]
    assert after['lifecycle']['cards'][recipient]['source']['tool_id'] is None
    assert all(s['status'] == 'closed' for s in after['lifecycle']['research_state'].values())
    assert all(s['status'] == 'completed' for s in after['sharing']['items'].values())
    assert not after['sharing']['waiting_methods']
    assert len(backend.calls) == 4 and costs == [.03] * 3
    tools.load_state(deepcopy(after))
    assert tools.state() == after


def test_question_and_sharing_alternate_even_when_author_keeps_question_open(tmp_path):
    tools, backend, _, _, author = prepared(tmp_path, open_author=True)
    queued(tools, snapshot(1))
    kinds = []
    for _ in range(6):
        _, current, plan = lease(tools)
        kinds.append(plan['work_kind'])
        if plan['work_kind'] == 'question' and plan['cell_id'] == author:
            answer = decision(action='use', tool_id=tools.state()['registered'][0])
        else:
            answer = finish(conclusion='inconclusive')
        tools.commit(tools.execute(plan, current, LLM([answer]) if plan['work_kind']=='question' else LLM([]), io=IO()))
    assert kinds[:4] == ['sharing', 'question', 'sharing', 'question']
    assert any(m['used_again'] for m in tools.state()['lifecycle']['methods'].values())
    assert tools.state()['lifecycle']['research_state'][author]['status'] == 'open'


@pytest.mark.parametrize('field,value', [
    ('receipt_id', None), ('reference_receipt_id', 'a'*64),
    ('snapshot_artifact_id', 'b'*64), ('action_id', 'c'*64),
    ('status', 'ready'), ('reason', 'stage_superseded'),
    ('completed_iteration', None), ('queue_sequence', True),
])
def test_completed_sharing_tampering_is_rejected_without_state_mutation(tmp_path, field, value):
    tools, _, _, _, _ = prepared(tmp_path)
    queued(tools, snapshot(1))
    commit_next(tools)
    saved = tools.state()
    ident = next(k for k, row in saved['sharing']['items'].items() if row['status']=='completed')
    bad = deepcopy(saved)
    bad['sharing']['items'][ident][field] = value
    with pytest.raises((ValueError, FileNotFoundError)):
        tools.load_state(bad)
    assert tools.state() == saved


def test_background_refusal_requeues_without_new_scientific_turn(tmp_path, monkeypatch):
    tools, backend, _, costs, _ = prepared(tmp_path, background=True)
    queued(tools, snapshot(1))
    original = tools._submit_job
    monkeypatch.setattr(tools, '_submit_job', lambda *args, **kwargs: (_ for _ in ()).throw(JobRefused('full')))
    plan, outcome = commit_next(tools)
    assert outcome['deferred'] and plan['work_kind'] == 'sharing'
    row = tools.state()['sharing']['items'][plan['sharing_work_id']]
    assert row['status']=='ready' and row['reason']=='submission_deferred'
    tools.load_state(deepcopy(tools.state()))
    monkeypatch.setattr(tools, '_submit_job', original)
    assert commit_next(tools)[0]['work_kind'] == 'question'
    plan, outcome = commit_next(tools)
    assert len(outcome['submitted_jobs']) == 1
    assert tools.state()['sharing']['items'][plan['sharing_work_id']]['status']=='waiting'
    assert len(backend.calls)==1 and costs==[.03]*3
    tools.load_state(deepcopy(tools.state()))


@pytest.mark.parametrize('failure', ['unknown', 'cancel', 'tool_error'])
def test_background_failure_is_terminal_without_retry_or_question_overwrite(tmp_path, failure):
    from sciel.research_tools.jobs import write_json
    from sciel.research_tools.job_worker import run_job
    tools, backend, events, costs, _ = prepared(tmp_path, background=True)
    queued(tools, snapshot(1))
    plan, episode = commit_next(tools)
    job = episode['submitted_jobs'][0]['job_id']
    before = deepcopy(tools.state()['lifecycle']['research_state'])
    if failure == 'unknown':
        write_json(tools.jobs.root/job/'dispatch.json',
            {'job_id': job, 'request_hash': tools.jobs.spec(job)['request_hash']}, immutable=True)
    elif failure == 'cancel':
        tools.jobs.cancel(job)
    else:
        backend.outputs[1] = result(status='tool_error', error='bounded fake failure')
    run_job(tools.lifecycle.run_dir, job, tools.library)
    tools.collect_background(IO())
    saved = tools.state()
    row = saved['sharing']['items'][plan['sharing_work_id']]
    assert row['status'] == 'completed'
    assert row['reason'] == {'unknown':'unknown','cancel':'host_failed','tool_error':'tool_error'}[failure]
    assert saved['lifecycle']['research_state'] == before
    assert saved['published'] == []
    if failure == 'unknown':
        assert saved['collected_jobs'][job]['resources']['cpu_seconds'] is None
        assert row['receipt_id'] is None and len(backend.calls) == 1
    tools.load_state(deepcopy(saved))
    commit_next(tools)  # Recipient's original question still closes normally.
    assert tools.reserve_continuations(lambda current, state, **work: current) == []
    assert not any(e['phase'] in ('published','adopted') for e in events)
    assert costs == [.03] * 3
