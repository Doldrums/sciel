"""Public controller, background collection and lifecycle result regressions.

Only deterministic fake executors/model responses are used. Submitted Python
source and the frozen historical tools are never imported or executed.
"""
from copy import deepcopy
from research_target_fixtures import bind_snapshot
import json

import pytest

from sciel.methods.models import content_hash
from sciel.research_tools.config import validate_config, library_limits
from sciel.research_tools.controller import ResearchTools
from sciel.research_tools.library import ToolLibrary
from sciel.research_tools.jobs import BackgroundJobs
from sciel.research_tools.job_worker import run_job
from test_research_tool_controller import IMAGE, IO, LLM, snapshot, prepare
from test_research_tool_result_contract import result, saved_rows

CODE = ('def run(payload):\n    return {"schema": "sciel-tool-result/1", "status": "success", '
        '"observations": {"probe": 1}, "diagnostics": {}, "error": None}\n')


class SavedExecutor:
    identity = {'image_id': IMAGE, 'schema': 'result-regression-fixture/1'}

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls, self.receipts = [], []

    def execute(self, bundle, inputs, **kwargs):
        self.calls.append({'bundle': deepcopy(bundle), 'inputs': deepcopy(inputs),
                           'kwargs': {k: deepcopy(v) for k, v in kwargs.items() if k != 'cancel_requested'}})
        output = deepcopy(self.outputs[len(self.calls) - 1])
        receipt = dict(schema='sciel-method-execution/1', method_id=bundle['id'],
            invocation_id=kwargs['invocation_id'], plan_id=kwargs['plan_id'],
            reservation_id=kwargs['reservation_id'], input_sha256=content_hash(inputs),
            source_sha256=content_hash(bundle['files']), runtime_sha256=content_hash(self.identity),
            request_sha256=content_hash({'invocation': kwargs['invocation_id']}),
            limits=deepcopy(kwargs['limits']), dependency_ids=sorted(b['id'] for b in kwargs['dependency_bundles']),
            status='ok', dispatch_status='invoked', output=output, output_sha256=content_hash(output),
            reason=None, stderr='', resources={'cpu_seconds': .01, 'wall_seconds': .02})
        receipt['receipt_sha256'] = content_hash(receipt)
        self.receipts.append(deepcopy(receipt))
        return receipt


def action(*, policy='exact_observations', revision=False, parents=(), background=False):
    value = dict(action='create', name='diagnostic', description='A bounded research check.',
        code=CODE + ('# explicit repair revision\n' if revision else ''), parents=list(parents),
        dependencies=[], arguments_json='{}', replay_policy=policy)
    if background:
        value.update(execution='background', cpu_seconds=2, wall_seconds=5)
    return value


def controller(tmp_path, outputs, *, background=False, max_requests=2, **config_changes):
    config_changes.setdefault('max_scientific_steps',1)
    config = validate_config(dict(enabled=True, image_id=IMAGE, interval=1,
        max_requests=max_requests, **({'jobs': {'enabled': True}} if background else {}), **config_changes))
    backend = SavedExecutor(outputs)
    run = tmp_path/'run'
    library = ToolLibrary(run, {'image_id': IMAGE}, library_limits(config), executor=backend)
    jobs = BackgroundJobs(run, library, config['jobs'], launch=lambda: None) if background else None
    events, costs = [], []
    tools = ResearchTools(config, run, task_identity={'task': 'generic-fixture'},
        emit=lambda **event: events.append(event), charge=costs.append, library=library, jobs=jobs)
    return tools, backend, events, costs


def execute_create(tools, current, *, policy='exact_observations', background=False, llm=None, io=None):
    return tools.execute(prepare(tools, current), current,
        llm or LLM([action(policy=policy, background=background)]), io=io or IO())


def collect(tools, job_id):
    run_job(tools.lifecycle.run_dir, job_id, tools.library)
    tools.collect_background(IO())


def assessment(row):
    return row['result']['status']


def test_caught_namespace_error_gets_bounded_repair_and_cannot_advance_or_share(tmp_path):
    error = next(x['receipt']['output']['error'] for x in saved_rows() if x['tool_id'].startswith('ab57c'))
    failed = result(status='tool_error', error=error)
    tools, backend, events, costs = controller(tmp_path, [failed, result({'boundary_check': 'negative_finding'})])
    def repair(prompt):
        assert error in prompt
        original = next(tools.library.objects.glob('*.json')).stem
        return action(revision=True, parents=[original])
    llm = LLM([action(), repair]); episode = execute_create(tools, snapshot(), llm=llm)
    tools.commit(episode)
    old, fixed = episode['registered_ids']
    assert len(backend.calls) == len(llm.prompts) == 2
    assert costs == [.03, .03] and episode['known_llm_cost'] == pytest.approx(.06)
    assert [assessment(row) for row in episode['invocations']] == ['tool_error', 'success']
    assert old not in tools.state()['lifecycle']['methods']
    assert tools.lifecycle.method(fixed)['stage'] == 'locally_tested'
    assert tools.state()['published'] == []
    assert tools.library.get(fixed)['manifest']['parents'] == [old]
    # The semantic assessment is separate: do not rewrite host process evidence.
    artifact = json.loads((tools.path/(episode['receipt_ids'][0]+'.json')).read_text())
    assert artifact['payload']['receipt'] == backend.receipts[0]
    assert artifact['payload']['receipt']['status'] == 'ok'
    assert artifact['payload']['receipt']['output'] == failed
    assert not any(e['phase'] in ('published', 'adopted') for e in events)
    saved = tools.state(); tools.load_state(saved); assert tools.state() == saved


@pytest.mark.parametrize('bad', [{}, {'error': 'legacy arbitrary JSON error'},
    {'schema': 'sciel-tool-result/1', 'status': 'success', 'observations': [], 'diagnostics': {}, 'error': None}])
def test_malformed_output_obeys_request_budget_and_replay_never_redispatches(tmp_path, bad):
    tools, backend, _, costs = controller(tmp_path, [bad, bad], max_requests=2)
    current = snapshot(); plan = prepare(tools, current); io = IO()
    episode = tools.execute(plan, current, LLM([action(), action(revision=True)]), io=io)
    assert len(backend.calls) == 2 and costs == [.03, .03]
    assert [assessment(x) for x in episode['invocations']] == ['invalid_output', 'invalid_output']
    tools.commit(episode)
    assert tools.state()['published'] == [] and tools.state()['lifecycle']['methods'] == {}
    # Same durable keys reuse physical receipts and charge IDs even on replay.
    again = tools.execute(plan, current, LLM([action(), action(revision=True)]), io=io)
    assert len(backend.calls) == 2 and costs == [.03, .03]
    assert again['receipt_ids'] == episode['receipt_ids']
    saved = tools.state(); tools.load_state(saved); assert tools.state() == saved


def test_successful_negative_finding_can_replay_without_becoming_benefit_claim(tmp_path):
    negative = result({'hypothesis_supported': False, 'violation_count': 4}, {'elapsed_seconds': .1})
    later = result(negative['observations'], {'elapsed_seconds': .8})
    tools, backend, _, costs = controller(tmp_path, [negative, later])
    local = execute_create(tools, snapshot()); tools.commit(local)
    tool_id = local['registered_ids'][0]
    other = snapshot(1, 2); replay = tools.execute(prepare(tools, other), other, LLM([]), io=IO()); tools.commit(replay)
    assert replay['replay_matched'] is True
    assert replay['replay_evidence']['measurement_reproduced'] is True
    assert tools.state()['published'] == [tool_id]
    assert backend.calls[1]['inputs'] == backend.calls[0]['inputs']
    assert costs == [.03]
    saved = tools.state(); tools.load_state(saved); assert tools.state() == saved


@pytest.mark.parametrize('policy,released,reproduced', [
    ('exact_observations', False, False), ('independent_execution', True, False)])
def test_declared_replay_policy_controls_changed_observations_without_fallback(tmp_path, policy, released, reproduced):
    tools, backend, _, costs = controller(tmp_path, [result({'sample': 1}), result({'sample': 2})])
    local = execute_create(tools, snapshot(), policy=policy); tools.commit(local)
    ident = local['registered_ids'][0]
    assert tools.library.get(ident)['manifest']['protocol']['replay_policy'] == policy
    foreign = snapshot(1, 2); plan = prepare(tools, foreign)
    replay = tools.execute(plan, foreign, LLM([]), io=IO()); tools.commit(replay)
    assert replay['replay_matched'] is released
    assert replay['replay_evidence']['policy'] == policy
    assert replay['replay_evidence']['measurement_reproduced'] is reproduced
    assert bool(tools.state()['published']) is released
    assert len(backend.calls) == 2 and costs == [.03]
    assert backend.calls[0]['inputs'] == backend.calls[1]['inputs']
    if not released:
        assert prepare(tools, snapshot(1, 3))['stage'] == 'research'
        assert tools.library.get(ident)['manifest']['protocol']['replay_policy'] == 'exact_observations'
    saved = tools.state(); tools.load_state(saved); assert tools.state() == saved


def test_replay_policy_is_immutable_content_addressed_version_data(tmp_path):
    tools, _, _, _ = controller(tmp_path, [])
    kwargs = dict(name='same', description='same source, different declared policy', code=CODE, author_island=0)
    exact = tools.library.register(**kwargs, replay_policy='exact_observations')
    independent = tools.library.register(**kwargs, replay_policy='independent_execution')
    assert exact['id'] != independent['id']
    assert exact['files'] == independent['files']
    assert tools.library.get(exact['id'])['manifest']['protocol']['replay_policy'] == 'exact_observations'
    with pytest.raises(ValueError):
        tools.library.register(**kwargs, replay_policy='automatic_fallback')


@pytest.mark.parametrize('failure_stage', ['adopt_trial', 'use_again'])
def test_semantic_tool_error_cannot_adopt_and_later_invocation_keeps_failure_status(tmp_path, failure_stage):
    good = result({'finding': 1}); error = result(status='tool_error', error='caught dependency namespace error')
    outputs = [good, good] + ([error] if failure_stage == 'adopt_trial' else [good, error])
    tools, backend, events, _ = controller(tmp_path, outputs)
    local = execute_create(tools, snapshot()); tools.commit(local); ident = local['registered_ids'][0]
    foreign = snapshot(1, 2); replay = tools.execute(prepare(tools, foreign), foreign, LLM([]), io=IO()); tools.commit(replay)
    plan = prepare(tools, foreign)
    if failure_stage == 'use_again':
        assert plan['stage'] == 'adopt_trial'
        trial = tools.execute(plan, foreign, LLM([]), io=IO()); tools.commit(trial)
        plan = prepare(tools, foreign)
    assert plan['stage'] == failure_stage
    failed = tools.execute(plan, foreign, LLM([]), io=IO()); tools.commit(failed)
    assert assessment(failed['invocations'][0]) == 'tool_error'
    method = tools.lifecycle.method(ident)
    if failure_stage == 'adopt_trial':
        assert method['adoptions'] == method['used_again'] == {}
    else:
        assert method['used_again'][plan['cell_id']] == failed['receipt_ids'][0]
        use_event = next(e for e in reversed(events) if e['phase'] == 'post_adoption_use')
        assert use_event['tool_result_status'] == 'tool_error'
        assert use_event['tool_result_error'] == error['error']
    assert tools.state()['published'] == [ident]
    assert len(backend.calls) == len(outputs)
    saved = tools.state(); tools.load_state(saved); assert tools.state() == saved


@pytest.mark.parametrize('semantic', ['success', 'tool_error', 'invalid_output'])
def test_background_collection_matches_foreground_semantics_and_survives_resume(tmp_path, semantic):
    output = (result({'constraint_violated': True}) if semantic == 'success' else
              result(status='tool_error', error='caught namespace error') if semantic == 'tool_error' else {'error': 'legacy'})
    tools, backend, _, costs = controller(tmp_path, [output], background=True, max_requests=1)
    local = execute_create(tools, snapshot(), background=True); tools.commit(local)
    job_id = local['submitted_jobs'][0]['job_id']; ident = local['registered_ids'][0]
    assert backend.calls == []
    tools.load_state(tools.state())
    collect(tools, job_id)
    assert len(backend.calls) == 1 and costs == [.03]
    assert (ident in tools.state()['lifecycle']['methods']) is (semantic == 'success')
    assert tools.state()['published'] == []
    context = tools.context(snapshot(iteration=1))
    assert semantic in json.dumps(context)
    saved = tools.state(); tools.load_state(saved); tools.collect_background(IO())
    assert len(backend.calls) == 1 and costs == [.03]
    assert tools.state() == saved
    assert tools.jobs.result(job_id) == backend.receipts[0]


@pytest.mark.parametrize('policy,replay_sample,release', [
    ('exact_observations', 1, True), ('exact_observations', 2, False), ('independent_execution', 2, True)])
def test_background_replay_requires_actual_completion_and_declared_observation_policy(tmp_path, policy, replay_sample, release):
    tools, backend, _, costs = controller(tmp_path,
        [result({'sample': 1}, {'elapsed': .1}), result({'sample': replay_sample}, {'elapsed': .2})], background=True)
    local = execute_create(tools, snapshot(), policy=policy, background=True); tools.commit(local)
    collect(tools, local['submitted_jobs'][0]['job_id']); ident = local['registered_ids'][0]
    foreign = snapshot(1, 2); replay = tools.execute(prepare(tools, foreign), foreign, LLM([]), io=IO()); tools.commit(replay)
    assert replay['plan']['stage'] == 'independent_replay' and tools.state()['published'] == []
    tools.load_state(tools.state())
    collect(tools, replay['submitted_jobs'][0]['job_id'])
    assert bool(tools.state()['published']) is release
    assert backend.calls[1]['inputs'] == backend.calls[0]['inputs']
    if release:
        record = tools.lifecycle.method(ident)['release_record']
        assert record['policy'] == policy
        assert record['measurement_reproduced'] is (policy == 'exact_observations')
    saved = tools.state(); tools.load_state(saved); assert tools.state() == saved
    assert len(backend.calls) == 2 and costs == [.03]


def test_minimum_prompt_budget_preserves_actual_tool_error_before_large_context(tmp_path):
    error = "Error calling construct_packing: name 'compute_max_radii' is not defined"
    broken = result({'large_partial_output': 'x' * 18000}, status='tool_error', error=error)
    tools, backend, _, costs = controller(tmp_path, [broken, result({'fixed': True})], max_prompt_chars=8000)
    current = snapshot()
    current.task_prompt = 'Long task description ' * 2000
    current.parent_code += '# large candidate source' * 2000
    bind_snapshot(current)
    current.research_cells_context = 'Retained contextual evidence ' * 2000
    llm = LLM([action(), action(revision=True)])
    episode = execute_create(tools, current, llm=llm)
    assert len(llm.prompts) == len(backend.calls) == 2
    assert all(len(prompt) <= 8000 for prompt in llm.prompts)
    assert error in llm.prompts[1] and 'tool_error' in llm.prompts[1]
    assert episode['invocations'][1]['result']['status'] == 'success'
    assert costs == [.03, .03]


@pytest.mark.parametrize('field,value', [('tool_result_status', 'success'), ('continuation_available', True)])
def test_background_resume_cannot_relabel_tool_error_or_invent_usable_continuation(tmp_path, field, value):
    tools, backend, _, costs = controller(tmp_path, [result(
        {'continuation_state': {'partial': 1}}, status='tool_error', error='caught namespace failure')],
        background=True, max_requests=1)
    local = execute_create(tools, snapshot(), background=True); tools.commit(local)
    job_id = local['submitted_jobs'][0]['job_id']; collect(tools, job_id)
    saved = tools.state()
    assert saved['collected_jobs'][job_id]['tool_result_status'] == 'tool_error'
    assert saved['collected_jobs'][job_id]['continuation_available'] is False
    with pytest.raises(ValueError):
        tools.jobs.continuation_inputs(job_id)
    bad = deepcopy(saved); bad['collected_jobs'][job_id][field] = value
    with pytest.raises(ValueError):
        tools.load_state(bad)
    assert tools.state() == saved
    assert len(backend.calls) == 1 and costs == [.03]
