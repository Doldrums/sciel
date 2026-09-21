"""Tool creation, execution, evidence, sharing and costs without provider calls."""
from copy import deepcopy
import json

import pytest

from sciel.campaign.budget import CampaignStopped
from sciel.engine.llm.client import LLMResponse
from sciel.engine.types import Snapshot
from research_target_fixtures import bind_snapshot
from sciel.methods.executor import ExecutionRefused
from sciel.methods.models import content_hash
from sciel.research_tools.config import validate_config
from sciel.research_tools.controller import ResearchTools, parse_action, PROMPT_MARKER
from sciel.research_tools.library import ToolLibrary
from test_research_tool_result_contract import result as tool_result

IMAGE = 'sha256:' + 'a' * 64
CODE = 'def run(payload):\n    return {"finding": "inspect parent"}\n'


class Executor:
    identity = {'image_id': IMAGE, 'schema': 'controller-fixture/1'}

    def __init__(self, statuses=('ok',), output=None, refuse=False):
        self.statuses, self.output, self.refuse = list(statuses), tool_result(
            {'finding': 'observed'} if output is None else output), refuse
        self.calls = []

    def execute(self, bundle, inputs, **kwargs):
        self.calls.append((deepcopy(bundle), deepcopy(inputs), deepcopy(kwargs)))
        if self.refuse:
            raise ExecutionRefused('fixture refused before dispatch')
        status = self.statuses[min(len(self.calls) - 1, len(self.statuses) - 1)]
        output = deepcopy(self.output) if status == 'ok' else None
        receipt = dict(schema='fixture/1', method_id=bundle['id'],
            invocation_id=kwargs['invocation_id'], plan_id=kwargs['plan_id'],
            reservation_id=kwargs['reservation_id'], input_sha256=content_hash(inputs),
            source_sha256=content_hash(bundle['files']), runtime_sha256=content_hash(self.identity),
            limits=kwargs['limits'], dependency_ids=sorted(b['id'] for b in kwargs['dependency_bundles']),
            status=status, dispatch_status='unknown' if status == 'unknown' else 'invoked',
            output=output, output_sha256=content_hash(output) if output is not None else None,
            reason=None if status == 'ok' else 'fixture failure', stderr='',
            resources={'cpu_seconds': None if status == 'unknown' else .01, 'wall_seconds': .02})
        receipt['receipt_sha256'] = content_hash(receipt)
        return receipt


class IO:
    def __init__(self):
        self.calls = []
        self.results = {}

    def call(self, kind, request, function):
        self.calls.append((kind, deepcopy(request)))
        key = content_hash({'kind': kind, 'request': request})
        if key not in self.results:
            self.results[key] = function()
        return deepcopy(self.results[key])


class LLM:
    def __init__(self, actions, cost=.03):
        self.actions, self.prompts, self.cost = list(actions), [], cost

    def generate(self, prompt, model, temperature=None, **options):
        self.prompts.append(prompt)
        value = self.actions.pop(0)
        if callable(value):
            value = value(prompt)
        assert options['response_format']['json_schema']['strict'] is True
        return LLMResponse(value if isinstance(value, str) else json.dumps(value), model, self.cost, 12, 8)


def create(**fields):
    return dict(action='create', name='Inspect', description='Research fixture', code=CODE,
                parents=[], dependencies=[], arguments_json='{"probe": 3}', **fields)


def snapshot(island=0, iteration=0):
    return bind_snapshot(Snapshot(island=island, op='full_rewrite', model='fixture', iteration_no=iteration,
        parent_id=f'p{island}', parent_code='def run_search():\n    return 0\n', parent_score=.1,
        task_prompt='Generic fixture: improve the program', kb_lessons=['Keep the measured interface']))


def setup(tmp_path, *, executor=None, interval=1, **config):
    config.setdefault('max_scientific_steps',1)  # These fixtures exercise one legacy repair episode.
    cfg = validate_config(dict(enabled=True, image_id=IMAGE, interval=interval, **config))
    backend = executor or Executor()
    library = ToolLibrary(tmp_path / 'run', {'image_id': IMAGE}, cfg['limits'], executor=backend)
    events, costs = [], []
    controller = ResearchTools(cfg, tmp_path / 'run', task_identity={'task': 'fixture/any', 'task_hash': 'fixture'},
        emit=lambda **event: events.append(event), charge=costs.append, library=library)
    return controller, backend, events, costs


def prepare(tools, source):
    return tools.prepare(source, signals={'schema':'sciel-stagnation-signals/1',
        'population_stall':{'threshold_reached':True, 'last_record_B_evals':0}})


def test_create_isolated_tool_then_share_on_next_wave_and_bind_actual_child(tmp_path):
    tools, executor, events, costs = setup(tmp_path)
    first = snapshot()
    plan = prepare(tools, first)
    foreign_prepared_before_commit = prepare(tools, snapshot(1))
    episode = tools.execute(plan, first, LLM([create()]), io=IO())
    assert len(executor.calls) == 1 and executor.calls[0][1]['parent_source'] == first.parent_code
    assert executor.calls[0][1]['arguments'] == {'probe': 3}
    assert tools.state()['published'] == []
    assert foreign_prepared_before_commit['visible_tool_ids'] == []
    tools.commit(episode, child_program_id='p_real')
    ident = episode['tool_ids'][0]
    # A sibling frozen before commit sees no tool and finishes independently.
    sibling = tools.execute(foreign_prepared_before_commit, snapshot(1), LLM([{'action':'skip'}]), io=IO())
    tools.commit(sibling)
    foreign = snapshot(1, 2)
    foreign_plan = prepare(tools, foreign)
    assert ident in foreign_plan['visible_tool_ids']
    assert foreign_plan['stage'] == 'independent_replay'
    reused = tools.execute(foreign_plan, foreign, LLM([]), io=IO())
    tools.commit(reused, child_program_id='p_foreign')
    assert tools.state()['uses'][ident] == {'0': 1, '1': 1}
    assert costs == [.03, .03]
    links = [event for event in events if event['phase'] == 'child_link']
    assert [event['child_program_id'] for event in links] == ['p_real', 'p_foreign']
    assert links[1]['receipt_ids'] == reused['receipt_ids']
    assert any(event['phase'] == 'invocation_result' and event['island_id'] == 1
               and event['author_island_id'] == 0 for event in events)


def test_failure_repairs_new_version_with_exact_parent_without_premature_release(tmp_path):
    tools, executor, events, costs = setup(tmp_path, executor=Executor(('failed','ok')))
    def repair(prompt):
        assert 'fixture failure' in prompt
        original = next(tools.library.objects.glob('*.json')).stem
        value = create(); value.update(name='Repaired', code=CODE+'# fixed\n', parents=[original])
        return value
    source = snapshot()
    episode = tools.execute(prepare(tools, source), source, LLM([create(), repair]), io=IO())
    tools.commit(episode)
    assert len(episode['registered_ids']) == 2 and len(executor.calls) == 2
    assert tools.state()['published'] == []
    assert tools.lifecycle.method(episode['registered_ids'][1])['stage'] == 'locally_tested'
    assert tools.library.get(episode['registered_ids'][1])['manifest']['parents'] == [episode['registered_ids'][0]]
    assert sum(costs) == pytest.approx(.06)
    assert not any(event['phase'] == 'child_link' for event in events)


def test_two_malformed_requests_are_charged_and_return_diagnostics_without_fake_tool(tmp_path):
    tools, executor, events, costs = setup(tmp_path)
    source = snapshot()
    llm = LLM(['bad JSON', '{"action":"unrecognized"}'])
    result = tools.execute(prepare(tools, source), source, llm, io=IO())
    tools.commit(result)
    assert len(llm.prompts) == 2 and sum(costs) == pytest.approx(.06)
    assert executor.calls == [] and tools.state()['published'] == []
    assert result['receipt_ids'] == result['tool_ids'] == []
    assert 'proposal_invalid' in result['context']


def test_known_predispatch_refusal_continues_with_diagnostics_and_no_actual_invocation(tmp_path):
    tools, executor, events, costs = setup(tmp_path, executor=Executor(refuse=True), max_requests=1)
    source = snapshot()
    result = tools.execute(prepare(tools, source), source, LLM([create()]), io=IO())
    tools.commit(result)
    assert costs == [.03] and result['tool_ids'] == result['receipt_ids'] == []
    assert result['invocations'][0]['dispatch_status'] == 'not_dispatched'
    assert tools.state()['published'] == []
    assert 'execution_refused_before_dispatch' in result['context']
    assert 'fixture refused before dispatch' in result['context']


def test_unknown_execution_pauses_without_publishing_or_fabricating_measurement(tmp_path):
    tools, executor, events, costs = setup(tmp_path, executor=Executor(('unknown',)))
    source = snapshot()
    with pytest.raises(CampaignStopped, match='outcome unknown'):
        tools.execute(prepare(tools, source), source, LLM([create()]), io=IO())
    assert len(executor.calls) == 1 and costs == [.03]
    assert tools.state()['published'] == []
    result = next(event for event in events if event['phase'] == 'invocation_result')
    assert result['resources']['cpu_seconds'] is None


def test_charge_ids_and_commits_are_idempotent_and_checkpoint_controls_visibility(tmp_path):
    tools, executor, events, costs = setup(tmp_path)
    source, io = snapshot(), IO()
    baseline = tools.state()
    plan = prepare(tools, source)
    episode = tools.execute(plan, source, LLM([create()]), io=io)
    tools.execute(plan, source, LLM([create()]), io=io)
    assert len(executor.calls) == 1 and costs == [.03]
    tools.commit(episode, child_program_id='p1'); tools.commit(episode, child_program_id='p1')
    assert len([e for e in events if e['phase']=='child_link']) == 1
    saved = tools.state()
    tools.load_state(baseline)
    assert prepare(tools, snapshot(2))['visible_tool_ids'] == []
    assert list(tools.library.objects.glob('*.json'))  # uncommitted artifacts confer no visibility
    tools.load_state(saved)
    assigned = prepare(tools, snapshot(2))
    assert assigned['stage'] == 'independent_replay'
    assert assigned['visible_tool_ids'] == episode['tool_ids']
    bad = deepcopy(saved); bad['published'] = ['b'*64]
    with pytest.raises(ValueError): tools.load_state(bad)
    assert tools.state()['published'] == saved['published']


def test_private_mode_and_unauthorized_reference_cannot_import_foreign_tools(tmp_path):
    tools, executor, events, costs = setup(tmp_path, share_between_islands=False, max_requests=1)
    source = snapshot()
    episode = tools.execute(prepare(tools, source), source, LLM([create()]), io=IO())
    tools.commit(episode)
    other = snapshot(1, 1); plan = prepare(tools, other)
    assert plan['visible_tool_ids'] == []
    denied = tools.execute(plan, other, LLM([{'action':'use','tool_id':episode['tool_ids'][0]}]), io=IO())
    assert len(executor.calls) == 1 and denied['tool_ids'] == []
    assert 'not visible' in denied['context']


def test_full_receipt_retained_while_generator_context_and_prompt_are_bounded(tmp_path):
    output = {'finding': 'x'*18000}
    tools, executor, events, costs = setup(tmp_path, executor=Executor(output=output), max_context_chars=4096)
    source = snapshot(); source.parent_code += '#' + 'long source'*20000
    bind_snapshot(source)
    llm = LLM([create()])
    result = tools.execute(prepare(tools, source), source, llm, io=IO())
    assert len(json.dumps(result['cards'], sort_keys=True, separators=(',', ':'))) <= 4096
    assert result['cards']
    assert len(llm.prompts[0]) <= tools.config['max_prompt_chars']
    assert llm.prompts[0].startswith(PROMPT_MARKER)
    ident = result['receipt_ids'][0]
    artifact = json.loads((tools.path / (ident+'.json')).read_text())
    assert content_hash(artifact) == ident
    assert artifact['payload']['receipt']['output'] == tool_result(output)
    assert artifact['payload']['inputs']['parent_source'] == source.parent_code


def test_frozen_input_and_non_durable_dispatch_are_rejected_before_cost(tmp_path):
    tools, executor, events, costs = setup(tmp_path)
    source = snapshot(); plan = prepare(tools, source)
    source.parent_code += '# changed'
    with pytest.raises(ValueError, match='frozen plan'):
        tools.execute(plan, source, LLM([create()]), io=IO())
    with pytest.raises(ValueError, match='durable wave'):
        tools.execute(plan, source, LLM([create()]), io=None)
    assert costs == executor.calls == []


@pytest.mark.parametrize('patch', [{'enabled':'yes'}, {'image_id':'latest'}, {'interval':0},
    {'limits':{'cpu_seconds':float('inf')}}, {'max_requests':4}, {'unknown':True}])
def test_invalid_configs_are_not_silently_enabled_or_unbounded(patch):
    with pytest.raises(ValueError): validate_config({'enabled':True, 'image_id':IMAGE, **patch})


def test_skip_is_robust_to_unused_fields_but_code_and_inputs_must_be_valid():
    config = validate_config({'max_scientific_steps':1})
    assert parse_action('{"action":"skip"}', config)['action'] == 'skip'
    with pytest.raises(ValueError): parse_action('{"action":"use","tool_id":"bad"}', config)
    with pytest.raises(ValueError): parse_action(json.dumps(dict(create(), arguments_json='[1]')), config)


def test_next_tool_planner_receives_its_cells_previous_observation(tmp_path):
    tools, executor, _, _ = setup(tmp_path, executor=Executor(output={'finding':'unique_previous_research_check'}))
    first = snapshot()
    episode = tools.execute(prepare(tools, first), first, LLM([create()]), io=IO())
    tools.commit(episode)
    next_snapshot = snapshot(iteration=1)
    next_snapshot.research_cells_context = tools.context(next_snapshot)
    llm = LLM([{'action':'skip','reason':'Use the already recorded observation.'}])
    tools.execute(prepare(tools, next_snapshot), next_snapshot, llm, io=IO())
    assert 'unique_previous_research_check' in llm.prompts[0]
    assert episode['receipt_ids'][0] in llm.prompts[0]
    assert len(executor.calls) == 1
