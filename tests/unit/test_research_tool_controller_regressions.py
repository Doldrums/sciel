"""Evidence-bound tool controller regressions, with no provider/Docker calls."""
from copy import deepcopy
from types import SimpleNamespace
from research_target_fixtures import bind_snapshot
import json

import pytest

from sciel.campaign.budget import CampaignStopped
from sciel.methods.models import content_hash
from sciel.research_tools.config import validate_config
from sciel.research_tools.controller import ResearchTools, parse_action
from sciel.research_tools.library import ToolLibrary
from test_research_tool_result_contract import result as tool_result

IMAGE = 'sha256:' + '7' * 64
CODE = 'def run(payload):\n    return {"observation": 1}\n'


def create(name='probe'):
    return dict(action='create', name=name, description='A bounded diagnostic.', code=CODE,
        parents=[], dependencies=[], arguments_json='{}', reason='Inspect the selected input.')


def signals():
    return {'schema': 'sciel-stagnation-signals/1',
        'population_stall': {'threshold_reached': True, 'last_record_B_evals': 0},
        'controller_plateau': {'enabled': False, 'active': False}}


def prepare(tools, current):
    return tools.prepare(current, signals=signals())


def snapshot(island=0):
    return bind_snapshot(SimpleNamespace(island=island, iteration_no=0, parent_id='p0', parent_code='def solve(): return 1\n',
        parent_score=.5, model='fixture', op='full_rewrite', task_prompt='Return a correct program.',
        artifacts={}, directive='', kb_lessons=[]))


class IO:
    def __init__(self):
        self.calls = []

    def call(self, kind, request, callback):
        self.calls.append(deepcopy((kind, request)))
        return callback()


class LLM:
    def __init__(self, actions):
        self.actions, self.calls = list(actions), []

    def generate(self, prompt, model, *args, **kwargs):
        self.calls.append(prompt)
        action = self.actions[len(self.calls) - 1]
        return SimpleNamespace(text=action if isinstance(action, str) else json.dumps(action), model=model,
            cost=.03, prompt_tokens=100, completion_tokens=80, finish_reason='stop', refusal=None)


class Executor:
    identity = {'schema':'regression-executor/1', 'image_id':IMAGE}

    def __init__(self, mutate=None, first_failure=False):
        self.mutate, self.first_failure, self.calls = mutate, first_failure, []

    def execute(self, bundle, inputs, **kwargs):
        self.calls.append(deepcopy((bundle, inputs, kwargs)))
        failed = self.first_failure and len(self.calls) == 1
        output = None if failed else tool_result({'useful_observation':'LATEST_SUCCESS_OBSERVATION'})
        receipt = dict(schema='sciel-method-execution/1', method_id=bundle['id'],
            invocation_id=kwargs['invocation_id'], plan_id=kwargs['plan_id'], reservation_id=kwargs['reservation_id'],
            source_sha256=content_hash(bundle['files']), runtime_sha256=content_hash(self.identity),
            input_sha256=content_hash(inputs), request_sha256=content_hash({'fixture':kwargs['invocation_id']}),
            status='failed' if failed else 'ok', reason='failed diagnostic' if failed else None,
            dispatch_status='invoked', output=output, output_sha256=content_hash(output) if output is not None else None,
            limits=deepcopy(kwargs['limits']), runtime=deepcopy(self.identity),
            dependency_ids=sorted(row['id'] for row in kwargs['dependency_bundles']), dependency_use_attested=False,
            stderr='OLD_FAILURE_LOG_' + 'x'*12000 if failed else '',
            resources=dict(cpu_seconds=.1, wall_seconds=.2, memory_peak_bytes=1000, output_bytes=60, stderr_bytes=12016 if failed else 0))
        if self.mutate:
            self.mutate(receipt)
        receipt['receipt_sha256'] = content_hash(receipt)
        return receipt


def controller(tmp_path, executor=None, **changes):
    config = validate_config({**dict(enabled=True, image_id=IMAGE, interval=1,
        max_requests=1, max_scientific_steps=1), **changes})
    executor = executor or Executor()
    library = ToolLibrary(tmp_path, {'image_id':IMAGE}, config['limits'], executor=executor)
    events, charges = [], []
    tools = ResearchTools(config, tmp_path, task_identity={'schema':'task/1'},
        emit=lambda **event: events.append(event), charge=charges.append, library=library)
    return tools, executor, events, charges


@pytest.mark.parametrize('field,value', [
    ('reservation_id','another-reservation'), ('source_sha256','0'*64), ('runtime_sha256','1'*64),
    ('limits',{'wall_seconds':1}), ('dependency_ids',['2'*64]),
])
def test_self_hashed_but_wrong_bound_execution_cannot_publish(tmp_path, field, value):
    tools, backend, _, _ = controller(tmp_path, Executor(mutate=lambda receipt:receipt.update({field:value})))
    current = snapshot()
    with pytest.raises(CampaignStopped):
        tools.execute(prepare(tools, current), current, LLM([create()]), io=IO())
    assert len(backend.calls) == 1
    assert tools.state()['published'] == []


@pytest.mark.parametrize('field,value', [('cpu_seconds',None), ('wall_seconds',None),
                                        ('cpu_seconds',-1), ('wall_seconds','0.2')])
def test_known_invocation_requires_known_finite_nonnegative_resource_measurements(tmp_path, field, value):
    def corrupt(receipt):
        receipt['resources'][field] = value
    tools, backend, _, _ = controller(tmp_path, Executor(mutate=corrupt))
    current = snapshot()
    with pytest.raises(CampaignStopped):
        tools.execute(prepare(tools, current), current, LLM([create()]), io=IO())
    assert len(backend.calls) == 1 and tools.state()['published'] == []


@pytest.mark.parametrize('text', [
    '{"action":"create","action":"skip","reason":"ambiguous action"}',
    json.dumps({**create(), 'arguments_json':'{"seed":1,"seed":2}'}),
    json.dumps({**create(), 'arguments_json':'{"nested":{"x":1,"x":2}}'}),
])
def test_duplicate_action_and_nested_argument_keys_are_refused(text):
    with pytest.raises(ValueError):
        parse_action(text, validate_config())


def test_latest_success_observation_survives_long_earlier_failure_in_bounded_context(tmp_path):
    tools, backend, _, charges = controller(tmp_path, Executor(first_failure=True),
        max_requests=2, max_context_chars=4096, limits={'max_stderr_bytes':16384})
    current = snapshot()
    result = tools.execute(prepare(tools, current), current, LLM([create('failed'),create('repair')]), io=IO())
    assert len(backend.calls) == 2 and len(charges) == 2
    assert result['publish_ids'] == []
    tools.commit(result)
    assert tools.state()['published'] == []
    assert len(result['context']) <= 4096
    assert 'LATEST_SUCCESS_OBSERVATION' in result['context']


def test_frozen_sibling_requires_independent_release_then_recipient_trial_before_general_access(tmp_path):
    tools, backend, _, charges = controller(tmp_path)
    owner, sibling, reviewer = snapshot(0), snapshot(1), snapshot(2)
    owner_plan, sibling_plan = prepare(tools, owner), prepare(tools, sibling)
    result = tools.execute(owner_plan, owner, LLM([create()]), io=IO())
    tools.commit(result)
    tool_id = result['tool_ids'][0]
    assert tools.state()['published'] == []
    assert tools.lifecycle.method(tool_id)['stage'] == 'locally_tested'
    replay_plan = prepare(tools, reviewer)
    assert replay_plan['stage'] == 'independent_replay'
    replay_llm = LLM([])
    replay = tools.execute(replay_plan, reviewer, replay_llm, io=IO())
    tools.commit(replay)
    assert replay_llm.calls == []
    assert tools.state()['published'] == [tool_id]
    assert backend.calls[1][1] == backend.calls[0][1]  # exact creator input
    use = dict(action='use', tool_id=tool_id, arguments_json='{}', reason='Try a foreign diagnostic.')
    # The sibling's already-frozen research plan never gains the released tool.
    blocked = tools.execute(sibling_plan, sibling, LLM([use]), io=IO())
    assert blocked['invocations'] == [] and len(backend.calls) == 2
    tools.commit(blocked)
    next_plan = prepare(tools, sibling)
    assert next_plan['stage'] == 'adopt_trial'
    assert next_plan['assigned_tool_id'] == tool_id
    assert tools.lifecycle.adopted_tool_ids(next_plan['cell_id']) == []
    trial = tools.execute(next_plan, sibling, LLM([]), io=IO())
    tools.commit(trial)
    assert tools.lifecycle.adopted_tool_ids(next_plan['cell_id']) == [tool_id]
    use_again = prepare(tools, sibling)
    assert use_again['stage'] == 'use_again'
    later = tools.execute(use_again, sibling, LLM([]), io=IO())
    tools.commit(later)
    ordinary = prepare(tools, sibling)
    assert ordinary['stage'] == 'research' and tool_id in ordinary['visible_tool_ids']
    allowed = tools.execute(ordinary, sibling, LLM([use]), io=IO())
    tools.commit(allowed)
    assert len(backend.calls) == 5 and charges == [.03, .03, .03]
    # Both independent lineages and their different receipt roles survive resume.
    saved = tools.state()
    tools.load_state(saved)
    assert tools.state() == saved


def test_small_context_keeps_card_question_bound_to_saved_invocation_on_resume(tmp_path):
    def long_observation(receipt):
        receipt['output'] = tool_result({'observation': 'LARGE_OUTPUT' * 300,
            'interpretation': 'An interpretation to test. ' * 35,
            'next_check': 'Run an independent experiment. ' * 35})
        receipt['output_sha256'] = content_hash(receipt['output'])
    tools, _, _, _ = controller(tmp_path, Executor(mutate=long_observation), max_context_chars=4096)
    current = snapshot()
    action = create(); action['reason'] = 'A detailed experimental question with a precise condition. ' * 15
    result = tools.execute(prepare(tools, current), current, LLM([action]), io=IO())
    assert len(result['cards']) == 1
    tools.commit(result)
    saved = tools.state()
    tools.load_state(saved)
    assert tools.state() == saved
    assert len(result['context']) <= tools.config['max_context_chars']


@pytest.mark.parametrize('mismatch', ['published_without_replay', 'unregistered_method'])
def test_checkpoint_outer_publication_matches_validated_method_lifecycle(tmp_path, mismatch):
    tools, _, _, _ = controller(tmp_path)
    current = snapshot()
    result = tools.execute(prepare(tools, current), current, LLM([create()]), io=IO())
    tools.commit(result)
    good = tools.state()
    bad = deepcopy(good)
    if mismatch == 'published_without_replay':
        bad['published'] = result['tool_ids']
    else:
        bad['registered'] = []
        bad['uses'] = {}
    with pytest.raises(ValueError):
        tools.load_state(bad)
    assert tools.state() == good


@pytest.mark.parametrize('value_length,field_length', [(300, 120), (10, 300)])
def test_formatted_context_including_header_obeys_declared_character_limit(tmp_path, value_length, field_length):
    def normal_observation(receipt):
        receipt['output'] = tool_result({'value': 'x' * value_length,
            'interpretation': 'x' * field_length, 'next_check': 'x' * field_length})
        receipt['output_sha256'] = content_hash(receipt['output'])
    tools, _, _, _ = controller(tmp_path, Executor(mutate=normal_observation), max_context_chars=4096)
    current = snapshot()
    action = create(); action['reason'] = 'q' * 120
    result = tools.execute(prepare(tools, current), current, LLM([action]), io=IO())
    assert len(result['cards']) == 1
    assert len(result['context']) <= tools.config['max_context_chars']
    assert result['cards'][0]['source']['receipt_id'] == result['receipt_ids'][0]


def test_unicode_questions_and_large_observations_keep_a_bound_resumable_card(tmp_path):
    def unicode_observation(receipt):
        receipt['output'] = tool_result({'observation': 'Результат измерения Δ 中文🙂' * 500,
            'interpretation': 'Гипотеза, которую ещё нужно проверить. 中文' * 100,
            'next_check': 'Проверить на независимом входе. 中文' * 100})
        receipt['output_sha256'] = content_hash(receipt['output'])
    tools, _, _, _ = controller(tmp_path, Executor(mutate=unicode_observation), max_context_chars=4096)
    current = snapshot()
    action = create(); action['reason'] = 'Проверяем воспроизводимость наблюдения 中文🙂. ' * 80
    result = tools.execute(prepare(tools, current), current, LLM([action]), io=IO())
    assert len(result['cards']) == 1
    assert len(result['context']) <= tools.config['max_context_chars']
    assert result['cards'][0]['question']
    assert result['cards'][0]['observations']['output_excerpt']
    tools.commit(result)
    saved = tools.state()
    tools.load_state(saved)
    assert tools.state() == saved
