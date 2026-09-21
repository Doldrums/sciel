"""CLI protocol tests use prescribed outcomes, never host execution or providers."""
from copy import deepcopy
import json

import pytest

from sciel.methods import __main__ as cli
from sciel.methods.models import content_hash
from sciel.methods.proposals import ScriptedProposer
from sciel.methods.registry import Registry
from sciel.methods.technical import BUDGET, RESOURCES, sort_bundle, technical, experiment

IMAGE = 'sha256:' + 'a' * 64


class PrescribedExecutor:
    """Known fixture oracle; no generated code is executed or interpreted."""
    identity = {'schema': 'scripted-cli-test/1', 'image_id': IMAGE}
    overhead_wall_bound = 12

    def __init__(self):
        self.calls = []

    def execute(self, bundle, value, *, invocation_id, plan_id, reservation_id, limits, dependency_bundles=()):
        self.calls.append(invocation_id)
        known_defect = bundle['manifest']['protocol'].get('defect') == 'deliberate removal of duplicates'
        output = sorted(set(value)) if known_defect else sorted(value)
        return dict(schema='prescribed-cli-test-receipt/1', status='ok', dispatch_status='invoked',
                    method_id=bundle['id'], invocation_id=invocation_id, plan_id=plan_id,
                    reservation_id=reservation_id, input_sha256=content_hash(value),
                    source_sha256=content_hash(bundle['files']), runtime_sha256=content_hash(self.identity),
                    output=output, output_sha256=content_hash(output),
                    resources={'cpu_seconds': .001, 'wall_seconds': .002}, stderr='')


def config(mode='A'):
    budget = dict(BUDGET, dollars=5, steps=30)
    return dict(schema='sciel-method-run/1', mode=mode, task='sort', image_id=IMAGE,
                llm=dict(models=['fixture/model'], weights=[1], max_tokens=1000,
                         api_base='https://openrouter.ai/api/v1', max_retries=1),
                pricing={'models': [{'id': 'fixture/model', 'pricing': {'prompt': '.000001', 'completion': '.000002'}}]},
                model='fixture/model', budget=budget, resources=deepcopy(RESOURCES),
                max_steps=3, max_cases=4, calendar_seconds=1800, storage_bytes=100_000_000,
                labs=[dict(id=name, agenda='Technical fixture', account_id=name, limits=deepcopy(budget))
                      for name in ('author', 'reviewer', 'recipient')], common_toolbox='sort')


def test_technical_cycle_and_second_command_reuse_exact_receipts(tmp_path):
    executor = PrescribedExecutor()
    result = technical(tmp_path, IMAGE, executor=executor)
    assert result['status'] == 'passed'
    assert result['technical_only'] and result['llm_provider_calls'] == 0
    assert result['next_action_invoked_method'] and result['next_proposal_received_method']
    assert result['repaired_vs_control_effect'] == result['transfer_vs_control_effect'] == 0
    calls, ledger = list(executor.calls), result['resource_ledger']
    second = technical(tmp_path, IMAGE, executor=executor)
    assert executor.calls == calls
    assert second['resource_ledger'] == ledger
    assert second['experiments'] == result['experiments']
    assert second['final_revision'] == result['final_revision']


@pytest.mark.parametrize('change, match', [
    ({'budget': {'dollars': 5}}, 'six'),
    ({'shared_planner': True}, 'unknown'),
    ({'claim': 'campaign_comparison'}, 'scientific'),
    ({'unknown': True}, 'unknown'),
    ({'mode': 'C', 'claim': 'bounded_arm'}, 'routing'),
    ({'upfront_costs': [{}]}, 'upfront'),
])
def test_config_rejects_silent_feature_or_budget_gaps(change, match):
    value = config(); value.update(change)
    with pytest.raises((ValueError, NotImplementedError), match=match):
        cli.validate_config(value)


def test_credentials_are_never_frozen_and_key_required_before_output(tmp_path, monkeypatch):
    value = config(); value['llm']['api_key'] = 'test-secret'
    with pytest.raises(ValueError, match='credentials'):
        cli.validate_config(value)
    monkeypatch.delenv('OPENROUTER_API_KEY', raising=False)
    output = tmp_path/'absent'
    with pytest.raises(ValueError, match='environment'):
        cli.run_config(config(), output)
    assert not output.exists()


def mocked_provider(monkeypatch, responses):
    import sciel.campaign.budget as money
    execution = PrescribedExecutor()
    proposers = []
    monkeypatch.setenv('OPENROUTER_API_KEY', 'offline-test-credential')
    monkeypatch.setattr(cli, 'DockerExecutor', lambda image: execution)
    monkeypatch.setattr(cli, 'shared_capacity', lambda provider: None)
    monkeypatch.setattr(money, 'BudgetedLLM', lambda *args, **kwargs: object())
    def proposer(*args, **kwargs):
        instance = ScriptedProposer(responses)
        proposers.append(instance)
        return instance
    monkeypatch.setattr(cli, 'BudgetedProposer', proposer)
    return execution, proposers


def test_run_resume_continues_after_existing_first_step_without_new_setup_storage_charge(tmp_path, monkeypatch):
    value = config()
    common = sort_bundle(IMAGE, RESOURCES)
    responses = {name: [dict(kind='experiment', experiment=experiment(common['id'], common['id'], [[2, 1, 2]])),
                        dict(kind='stop', reason='Fixture complete')]
                 for name in ('author', 'reviewer', 'recipient')}
    execution, proposers = mocked_provider(monkeypatch, responses)
    real_step = cli.Coordinator.step
    calls = 0
    def crash_second(self, lab):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError('synthetic process boundary')
        return real_step(self, lab)
    monkeypatch.setattr(cli.Coordinator, 'step', crash_second)
    with pytest.raises(RuntimeError, match='synthetic'):
        cli.run_config(value, tmp_path)
    first_registry_bytes = sum(p.stat().st_size for p in (tmp_path/'registry').rglob('*') if p.is_file())
    monkeypatch.setattr(cli.Coordinator, 'step', real_step)
    result = cli.run_config(value, tmp_path)
    assert result['status'] == 'completed'
    assert len(result['outcomes']) == 6
    assert sum(len(p.calls) for p in proposers) == 6
    assert all(result['outcomes'].count(row) == 1 for row in result['outcomes'])
    setups = [r for r in result['resource_ledger']['reservations'] if r['category'] == 'bootstrap']
    assert len(setups) == 2
    assert setups[1]['actual']['storage_bytes'] < first_registry_bytes
    assert 'offline-test-credential' not in (tmp_path/'run-config.json').read_text()
    call_count = len(execution.calls)
    third = cli.run_config(value, tmp_path)
    assert third['steps_executed_this_invocation'] == 0
    assert len(execution.calls) == call_count
    assert third['outcomes'] == result['outcomes']


@pytest.mark.parametrize('mode', ['B', 'C'])
def test_b_adopts_locally_c_also_transfers_then_uses_next_action(tmp_path, monkeypatch, mode):
    value = config(mode)
    common = sort_bundle(IMAGE, RESOURCES)
    new = sort_bundle(IMAGE, RESOURCES, author='author')
    stop = dict(kind='stop', reason='Fixture complete')
    responses = {'author': [dict(kind='method', files=new['files'], manifest=new['manifest'],
                                 experiment=experiment('$new', common['id'], [[2, 1, 2]])),
                             dict(kind='experiment', experiment=experiment(new['id'], common['id'], [[3, 2, 3]])), stop],
                 'reviewer': [stop],
                 'recipient': [dict(kind='experiment', experiment=experiment(new['id'] if mode == 'C' else common['id'],
                                                                           common['id'], [[4, 2, 4]])), stop]}
    value.update(claim='bounded_arm', upfront_costs=[],
                 routing=dict(author='author', reviewer='reviewer', recipient='recipient', control=[common['id']],
                              transfer_inputs=[[5, 1, 5]], max_transfers=1))
    execution, proposers = mocked_provider(monkeypatch, responses)
    result = cli.run_config(value, tmp_path)
    assert result['status'] == 'completed'
    assert len(result['routing']) == 1
    routed = result['routing'][0]
    assert routed['status'] == 'completed'
    assert routed['author_adoption_id']
    registry = Registry(tmp_path/'registry')
    author = registry.revision(registry.lab('author')['active_revision'])
    recipient = registry.revision(registry.lab('recipient')['active_revision'])
    assert new['id'] in author['active_methods']
    assert (new['id'] in recipient['active_methods']) == (mode == 'C')
    assert ('recipient_adoption_id' in routed) == (mode == 'C')
    if mode == 'C':
        adoption = registry.get_artifact('adoption', routed['recipient_adoption_id'])
        step = next(row for row in result['outcomes'] if row['lab_id'] == 'recipient' and row['index'] == 0)
        plan = registry.get_artifact('experiment_plan', step['plan_id'])
        assert plan['lab_revision_id'] == adoption['new_revision']
        assert plan['treatment'] == [new['id']]


def test_routing_resume_after_adoption_uses_frozen_trial_not_new_revision(tmp_path, monkeypatch):
    value = config('C')
    common = sort_bundle(IMAGE, RESOURCES)
    new = sort_bundle(IMAGE, RESOURCES, author='author')
    stop = dict(kind='stop', reason='Fixture complete')
    responses = {'author': [dict(kind='method', files=new['files'], manifest=new['manifest'],
                                 experiment=experiment('$new', common['id'], [[2, 1, 2]])), stop],
                 'reviewer': [stop],
                 'recipient': [dict(kind='experiment', experiment=experiment(new['id'], common['id'], [[4, 2, 4]])), stop]}
    value['routing'] = dict(author='author', reviewer='reviewer', recipient='recipient', control=[common['id']],
                            transfer_inputs=[[5, 1, 5]], max_transfers=1)
    executor, _ = mocked_provider(monkeypatch, responses)
    real_adopt = cli.Coordinator.adopt
    interrupted = False
    def after_local_adoption(self, lab, *args, **kwargs):
        nonlocal interrupted
        result = real_adopt(self, lab, *args, **kwargs)
        if lab == 'author' and not interrupted:
            interrupted = True
            raise RuntimeError('crash after committed author adoption')
        return result
    monkeypatch.setattr(cli.Coordinator, 'adopt', after_local_adoption)
    with pytest.raises(RuntimeError, match='committed author'):
        cli.run_config(value, tmp_path)
    prefix = list(executor.calls)
    monkeypatch.setattr(cli.Coordinator, 'adopt', real_adopt)
    result = cli.run_config(value, tmp_path)
    assert result['routing'][0]['status'] == 'completed'
    assert len(set(executor.calls)) == len(executor.calls)
    assert executor.calls[:len(prefix)] == prefix
    registry = Registry(tmp_path/'registry')
    assigned = [row for _, row in cli._artifact_rows(registry, 'experiment_plan')
                if row['proposal_id'] == 'cli-own-adoption:' + new['id']]
    assert len(assigned) == 1
    assert sum(1 for _, row in cli._artifact_rows(registry, 'adoption') if row['lab_id'] == 'author') == 1


def test_solution_only_route_delivers_checked_output_without_code_or_method_adoption(tmp_path, monkeypatch):
    value = config('C')
    common = sort_bundle(IMAGE, RESOURCES)
    stop = dict(kind='stop', reason='Fixture complete')
    responses = {'author': [dict(kind='solution', code='def solve(values):\n    return sorted(values)\n',
                                 experiment=experiment('$new', common['id'], [[2, 1, 2]])), stop],
                 'reviewer': [stop], 'recipient': [stop]}
    value.update(transfer_policy='solution_only', claim='bounded_arm', upfront_costs=[],
                 routing=dict(author='author', reviewer='reviewer', recipient='recipient', control=[common['id']],
                              transfer_inputs=[[5, 1, 5]], max_transfers=1))
    executor, proposers = mocked_provider(monkeypatch, responses)
    result = cli.run_config(value, tmp_path)
    assert result['status'] == 'completed'
    route, = result['routing']
    assert route['status'] == 'completed' and route['kind'] == 'solution_only'
    assert 'recipient_adoption_id' not in route and 'method_id' not in route
    contexts = [json.loads(prompt.rsplit('\nCONTEXT_JSON\n', 1)[1]) for p in proposers for prompt in p.calls]
    recipient = next(context for context in contexts if context['lab_id'] == 'recipient')
    solution, = recipient['previous_solutions']
    assert solution['id'] == route['solution_id']
    assert solution['transfer_kind'] == 'solution_only'
    assert solution['payload'] == {'input': [2, 1, 2], 'output': [1, 2, 2]}
    assert content_hash(solution['payload']) == solution['payload_sha256']
    assert solution['source_sha256'] == route['source_sha256']
    assert not {'code', 'manifest', 'files'}.intersection(solution)
    assert recipient['available_library'] == []
    assert recipient['revision']['active_methods'] == [common['id']]
    assert recipient['revision']['active_adapters'] == []
