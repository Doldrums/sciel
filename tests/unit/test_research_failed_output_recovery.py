"""Malformed terminal results remain immutable failures, not host exceptions.

Only saved fake model/executor leaves run; submitted Python is never executed.
"""
from copy import deepcopy
import json

import pytest

from sciel.methods.models import content_hash
from sciel.research_tools.job_worker import run_job
from test_research_experimental_protocol import comparison
from test_research_tool_controller import IO, LLM, prepare, snapshot
from test_research_tool_result_flow import result
from test_sequential_research_episode import build, decision, finish


def failed_outputs(backend, count):
    execute = backend.execute

    def invoke(*args, **kwargs):
        receipt = execute(*args, **kwargs)
        if len(backend.calls) <= count:
            receipt.update(status='failed', reason='output_schema_invalid')
            receipt.pop('receipt_sha256')
            receipt['receipt_sha256'] = content_hash(receipt)
            backend.receipts[-1] = deepcopy(receipt)
        return receipt

    backend.execute = invoke


def proposal(enabled, **changes):
    return decision(**({'experimental_protocol': comparison()} if enabled else {}), **changes)


def check_failure(tools, receipt_id, output):
    record = tools._read_artifact(receipt_id, 'invocation')
    assert record['receipt']['output'] == output
    assert record['receipt']['output_sha256'] == content_hash(output)
    assert record['receipt']['status'] == 'failed'
    assert record['receipt']['reason'] == 'output_schema_invalid'
    assert record['result']['status'] == 'host_failed'
    assert record['receipt']['resources'] == {'cpu_seconds': .01, 'wall_seconds': .02}
    assert not tools.state()['published']
    assert not tools.state()['lifecycle']['methods']
    return record


@pytest.mark.parametrize('output', [[1], 'invalid JSON envelope', 1, True, {'observations': [1]}])
@pytest.mark.parametrize('enabled', [False, True])
@pytest.mark.parametrize('background', [False, True])
def test_failed_output_commits_and_cold_loads_without_reexecution(tmp_path, output, enabled, background):
    options = dict(max_requests=1, background=background,
                   experimental_protocol=enabled, shared_observations=enabled)
    tools, backend, _, costs = build(tmp_path, [output], **options)
    failed_outputs(backend, 1)
    current = snapshot()
    llm = LLM([proposal(enabled, **({'execution': 'background'} if background else {}))])
    episode = tools.execute(prepare(tools, current), current, llm, io=IO())
    tools.commit(episode)
    if background:
        job = episode['submitted_jobs'][0]['job_id']
        run_job(tools.lifecycle.run_dir, job, tools.library)
        io = IO()
        tools.collect_background(io)
        done = tools.state()['collected_jobs'][job]
        receipt_id = done['invocation_artifact_id']
        assert done['tool_result_status'] == 'host_failed'
        assert done['continuation_available'] is False
        if enabled:
            assert done['comparison_report']['report_shape'] == 'invalid'
    else:
        receipt_id = episode['invocations'][0]['receipt_id']
        assert episode['research_state']['status'] == 'repair_exhausted'
        assert episode['research_state']['scientific_steps'] == 0
        if enabled:
            assessment = episode['research_state']['steps'][0]['comparison_report_assessment']
            assert assessment['report_shape'] == 'invalid'
            assert assessment['adherence'] == 'tool_reported_unverified'
    record = check_failure(tools, receipt_id, output)
    assert costs == [.03] and len(backend.calls) == 1
    state = json.loads(json.dumps(tools.state()))
    cold, cold_backend, _, cold_costs = build(tmp_path, [], **options)
    cold.load_state(state)
    assert cold.state() == state
    assert check_failure(cold, receipt_id, output) == record
    if background:
        cold.collect_background(IO())
        assert cold.state() == state
    assert cold_backend.calls == [] and cold_costs == []
    if enabled:
        tampered = deepcopy(state)
        if background:
            report = tampered['collected_jobs'][job]['comparison_report']
        else:
            report = tampered['lifecycle']['research_state'][episode['plan']['cell_id']]['steps'][0][
                'comparison_report_assessment']
        report['report_shape'] = 'complete'
        report['problems'] = []
        with pytest.raises(ValueError, match='comparison assessment differs'):
            cold.load_state(tampered)
        assert cold_backend.calls == [] and cold_costs == []


def test_failed_raw_output_reaches_repair_and_keeps_both_receipts(tmp_path):
    tools, backend, _, costs = build(tmp_path, [[1], result({'gain': 0})], max_requests=2,
                                   experimental_protocol=True, shared_observations=True)
    failed_outputs(backend, 1)

    def repair(prompt):
        assert 'output_schema_invalid' in prompt
        assert 'comparison_results missing or not an array' in prompt
        previous = next(tools.library.objects.glob('*.json')).stem
        return proposal(True, code=decision()['code'] + '# repaired envelope\n', parents=[previous])

    close = finish(experimental_protocol={**comparison(), 'kind': 'closure', 'arms': [],
        'matched_seeds': [], 'cpu_seconds_per_arm': None, 'budget_action': 'none'})
    current = snapshot()
    episode = tools.execute(prepare(tools, current), current,
                            LLM([proposal(True), repair, close]), io=IO())
    tools.commit(episode)
    assert [row['result']['status'] for row in episode['invocations']] == ['host_failed', 'success']
    assert len(backend.calls) == 2 and costs == [.03] * 3
    assert episode['research_state']['scientific_steps'] == 1
    assert episode['research_state']['resources']['cpu_seconds'] == .02
    assert episode['registered_ids'][0] not in tools.state()['lifecycle']['methods']
    assert not tools.state()['published']
    tools.load_state(json.loads(json.dumps(tools.state())))
