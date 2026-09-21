"""Real Engine integration, with local provider and isolated-tool stand-ins."""
from copy import deepcopy
import json
from pathlib import Path
import sys
from types import ModuleType

import pytest

from sciel.config import apply_defaults
from sciel.engine import checkpoint
from sciel.engine.factory import build_engine
from sciel.engine.loop import Engine
from sciel.engine.types import NoChild
from sciel.methods.models import canonical_bytes, content_hash
from sciel.research_tools.lifecycle import CellLifecycle
from sciel.research_tools.library import ToolLibrary
from sciel.research_tools.result_contract import assess_receipt
from sciel.engine.research_targets import input_binding, tool_inputs
from sciel.research_tools.followups import build_followup, validate_followup

from test_parallel_engine import Provider, ResearchProvider, config, evaluator, events, task_fixture


TOOL_CONFIG = {'enabled': True, 'image_id': 'sha256:' + '1' * 64}


class FakeTools:
    """Match the controller contract; the actual library has independent tests."""
    invocations = []
    execution_status = 'completed'

    def __init__(self, config, run_dir, *, task_identity, emit, charge):
        self.identity = {'schema': 'fixture-tools/1', 'task_identity': task_identity}
        self.emit, self.charge = emit, charge
        self.cursor = 0
        self.charged, self.visible, self.committed = [], [], []
        self.followups = []
        self.config, self.task_identity = config, task_identity
        # The controller/provider boundary is fake, but observation delivery
        # and cold provenance validation use the production immutable readers.
        self.lifecycle = CellLifecycle(run_dir, config, lambda **_: None)
        self.library = ToolLibrary(run_dir, {'image_id': config['image_id']}, config['limits'])

    def saved_observation(self, snapshot, *, label, observations, commit=False, status='completed'):
        """Save inert measured evidence; never import or run the package source."""
        from test_research_tool_result_contract import result as tool_result
        question = 'Inspect the fixture ' + label + '.'
        cell = self.lifecycle.cells.ensure_cell(snapshot.island, question,
            {'kind': 'fixture', 'evidence': label})
        if cell['head_revision'] is None:
            cell = self.lifecycle.cells.revise(cell['id'], phase='proposed')
        action_id = content_hash({'fixture_action': label})
        plan = dict(action_id=action_id, cell_id=cell['id'], cell_revision_id=cell['head_revision'],
                    island_id=snapshot.island, parent_program_id=snapshot.parent_id,
                    iteration=snapshot.iteration_no, question=question, stage='research')
        bundle = self.library.register('fixture-' + label, question,
            'def run(payload):\n    return {}\n', snapshot.island)
        tool_id = bundle['id']
        invocation_id = content_hash({'fixture_invocation': label, 'tool_id': tool_id})
        inputs = tool_inputs(snapshot, self.task_identity, {})
        bound = input_binding(inputs)
        output = tool_result(observations) if status == 'completed' else None
        receipt = dict(schema='authenticated-upstream-fixture/1', method_id=tool_id,
            invocation_id=invocation_id, plan_id=action_id, reservation_id=invocation_id,
            input_sha256=content_hash(inputs), output=output,
            output_sha256=content_hash(output) if output is not None else None,
            runtime_sha256=content_hash({'image_id': self.config['image_id'], 'schema':'fixture'}),
            limits=self.config['limits'], dependency_ids=[], resources={'cpu_seconds':.1, 'wall_seconds':.2},
            source_sha256=content_hash(bundle['files']), status='ok' if output is not None else 'failed',
            dispatch_status='invoked')
        receipt['receipt_sha256'] = content_hash(receipt)
        assessed = assess_receipt(receipt)
        payload = dict(action_id=action_id, island_id=snapshot.island,
            parent_program_id=snapshot.parent_id, iteration=snapshot.iteration_no,
            tool_id=tool_id, inputs=inputs, receipt=receipt, cell_id=cell['id'],
            cell_revision_id=cell['head_revision'], stage='research', question=question,
            input_program_id=bound['candidate_id'], target_binding=bound, result=assessed,
            replay_policy=bundle['manifest']['protocol']['replay_policy'])
        artifact = dict(schema='sciel-research-tool-artifact/1', kind='invocation', payload=payload)
        receipt_id = content_hash(artifact)
        directory = self.library.path / 'artifacts'
        directory.mkdir(exist_ok=True)
        (directory / (receipt_id + '.json')).write_bytes(canonical_bytes(artifact))
        card = {'schema':'sciel-research-check/3', 'replay_comparison':None,
            'cell_id':cell['id'], 'revision_id':cell['head_revision'], 'question':question,
            'source':dict(tool_id=tool_id, invocation_id=invocation_id, receipt_id=receipt_id, **bound),
            'observations':dict(status=assessed['status'], output_excerpt=json.dumps(observations),
                                resources=deepcopy(receipt['resources'])),
            'interpretation':'Auxiliary fixture observation.', 'next_check':'Evaluate the next candidate.',
            'limitations':['Not a canonical evaluation.']}
        row = dict(tool_id=tool_id, invocation_id=invocation_id, receipt_id=receipt_id,
            status=receipt['status'], dispatch_status='invoked', result=assessed,
            replay_policy=payload['replay_policy'], input_sha256=receipt['input_sha256'],
            output_sha256=receipt['output_sha256'], resources=receipt['resources'],
            target_binding=bound, author_island_id=snapshot.island)
        episode = dict(plan=plan, invocations=[row], cards=[card], receipt_ids=[receipt_id],
                       tool_ids=[tool_id], action_id=label)
        if commit:
            self.lifecycle.commit(episode)
            episode['cards'] = [deepcopy(self.lifecycle.state()['cards'][cell['id']])]
        return episode

    def prepare(self, snapshot, *, signals):
        self.cursor += 1
        return {'action_id': str(self.cursor), 'island_id': snapshot.island,
                'parent_program_id': snapshot.parent_id, 'iteration': snapshot.iteration_no,
                'attempt': 1, 'visible': list(self.visible), 'signals': deepcopy(signals)}

    def context(self, snapshot):
        return []

    def format_context(self, cards):
        return '\n\nRESEARCH CELL OBSERVATIONS\n' + json.dumps(cards)

    def execute(self, plan, snapshot, llm, *, io, candidate_evaluator=None):
        assert io is not None
        response = llm.generate('SCIEL_RESEARCH_TOOL_EPISODE\n' + json.dumps(plan) + '\nInspect.',
                                snapshot.model)
        if plan['action_id'] not in self.charged:
            self.charge(response.cost)
            self.charged.append(plan['action_id'])
        def leaf():
            self.invocations.append(plan['action_id'])
            return {'status': self.execution_status, 'output': {'probe': 999},
                    'cpu_seconds': .1, 'wall_seconds': .2}
        result = io.call('tool_execution', {'action_id': plan['action_id']}, leaf)
        self.emit(phase='execution', action_id=plan['action_id'], receipt=result)
        episode = self.saved_observation(snapshot, label=plan['action_id'],
            observations=result['output'], status=result['status'])
        episode['context'] = 'Fixture tool observation: ' + json.dumps(result)
        return episode

    def candidate_binding(self, episode, *, source, provenance, evaluation_id, score, cases):
        # This Engine fixture checks the controller boundary, not immutable
        # artifact authentication (covered by the real-controller/WAL tests).
        from sciel.engine.validation import source_sha256
        saved = episode['candidate_submission']
        assert source == saved['source'] and provenance == saved['provenance']
        assert provenance['candidate_source_sha256'] == source_sha256(source)
        producer = episode['cards'][0]['source']
        for key in ('tool_id', 'receipt_id', 'invocation_id'):
            assert provenance[key] == producer[key]
        value = saved['evaluation']
        assert value['status'] == 'evaluated'
        assert value['source_sha256'] == source_sha256(source)
        assert value['evaluation_id'] == evaluation_id
        assert value['score'] == score and value['cases'] == cases
        return {'link_basis': 'canonical_candidate_source',
            'tool_ids': [provenance['tool_id']], 'receipt_ids': [provenance['receipt_id']],
            'candidate_source_sha256': source_sha256(source), 'evaluation_id': evaluation_id,
            'canonical_evaluation_id': content_hash({'fixture_candidate_evaluation': saved}),
            'candidate_invocation_id': provenance['invocation_id']}

    def commit(self, episode, *, child_program_id=None, link_kind='mutation',
               candidate_binding=None, child_source=None):
        extra = {}
        if child_program_id is not None and episode.get('candidate_submission'):
            saved = episode['candidate_submission']
            value = saved['evaluation']
            assert candidate_binding == self.candidate_binding(episode, source=child_source,
                provenance=saved['provenance'], evaluation_id=value['evaluation_id'],
                score=value['score'], cases=value['cases'])
            extra = {'candidate_binding': deepcopy(candidate_binding), 'child_source': child_source}
        else:
            assert candidate_binding is None and child_source is None
        self.lifecycle.commit(episode)
        self.visible.extend(episode['tool_ids'])
        self.committed.append({'action_id': episode['action_id'], 'child': child_program_id,
                               'link_kind': link_kind, **extra})

    def record_followup(self, record, *, child_program_id=None, link_kind='mutation', observed_record=None):
        # The pure production join validates research/measurement semantics.
        # Real immutable artifact authentication has separate controller tests.
        row = build_followup(record, child_program_id=child_program_id,
                             link_kind=link_kind, observed_record=observed_record)
        if row is None:
            return False
        old = next((x for x in self.followups if x['experiment_id'] == row['experiment_id']), None)
        if old is not None:
            assert old == row
            return False
        self.followups.append(deepcopy(row))
        self.emit(phase='method_followup', experiment_id=row['experiment_id'],
                  followup_sha256=row['sha256'], child_program_id=child_program_id)
        return True

    def state(self):
        return deepcopy({'cursor': self.cursor, 'charged': self.charged,
                         'visible': self.visible, 'committed': self.committed,
                         'followups': self.followups, 'lifecycle':self.lifecycle.state()})

    def load_state(self, value, *, expected_run_id=None):
        if set(value) != {'cursor', 'charged', 'visible', 'committed', 'followups', 'lifecycle'}:
            raise ValueError('invalid fixture tools state')
        for row in value['followups']:
            validate_followup(row)
        self.lifecycle.load_state(value['lifecycle'])
        self.__dict__.update(deepcopy({k:v for k,v in value.items() if k != 'lifecycle'}))


def setup_tools(tmp_path, monkeypatch, *, workers=2, enabled=True, max_iterations=2,
                entry='run_search', metric='combined_score', provider=None, **overrides):
    module = ModuleType('sciel.research_tools.controller')
    module.ResearchTools = FakeTools
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(FakeTools, 'invocations', [])
    monkeypatch.setattr(FakeTools, 'execution_status', 'completed')
    tasks = tmp_path / 'tasks'
    task = task_fixture(tasks, entry, metric)
    cfg = config(task, tmp_path / 'capacity', workers=workers, max_iterations=max_iterations,
                 research_tools=TOOL_CONFIG if enabled else {'enabled': False}, **overrides)
    provider = provider or Provider(entry)
    evaluations, _ = evaluator(monkeypatch)
    directory = tmp_path / 'run'

    def build(*, resume=False):
        engine = build_engine(cfg, str(tasks), str(directory), python=sys.executable, llm=provider)
        if resume:
            checkpoint.load(engine, str(directory / 'checkpoint.json'))
        return engine
    return build, provider, evaluations, directory


@pytest.mark.parametrize('workers', [1, 2])
@pytest.mark.parametrize('entry,metric', [('run_search', 'combined_score'), ('construct_solution', 'quality')])
def test_tool_observations_feed_normal_search_and_do_not_inflate_reward_or_cost(tmp_path, monkeypatch, workers, entry, metric):
    build, provider, evaluations, directory = setup_tools(tmp_path, monkeypatch, workers=workers,
                                                        entry=entry, metric=metric)
    engine = build()
    result = engine.run()
    assert result['iteration_no'] == result['B_evals'] == 2
    assert len(evaluations) == 3 and len(FakeTools.invocations) == 2
    assert len(provider.calls) == 4 and result['cost_total'] == pytest.approx(.12)
    assert engine.db.pop_best < 1  # Tool output 999 never enters reward/population.
    assert all('RESEARCH CELL OBSERVATIONS' in prompt for prompt in provider.calls
               if not prompt.startswith('SCIEL_RESEARCH_TOOL_EPISODE'))
    for prompt in provider.calls:
        if prompt.startswith('SCIEL_RESEARCH_TOOL_EPISODE'):
            continue
        delivered = json.JSONDecoder().raw_decode(prompt.split('RESEARCH CELL OBSERVATIONS\n', 1)[1])[0][0]
        assert delivered['schema'] == 'sciel-research-check/5'
        assert delivered['comparison_assessment'] == {
            'schema':'sciel-research-comparison-assessment/1',
            'receipt_id':delivered['source']['receipt_id'], 'proposal_id':None,
            'status':'unavailable', 'assessment':None}
        saved = engine.research_tools.lifecycle._receipt(delivered['source']['receipt_id'],
            delivered['source']['tool_id'], engine.research_tools.library)
        assert json.loads(delivered['observations']['output_excerpt']) == saved['receipt']['output']['observations']
        assert delivered['observations']['resources'] == saved['receipt']['resources']
    assert all(row['child'] in engine.db._all for row in engine.research_tools.committed)
    rows = events(directory)
    assert len([row for row in rows if row['event'] == 'research_tool']) == 2
    assert sum(row.get('tool_executions', 0) for row in rows if row.get('action') == 'wave_complete') == 2
    intents = [job for path in (directory / '.parallel').rglob('jobs.json')
               for job in json.loads(path.read_text())['jobs'] if job['job_id'].startswith('G:')]
    assert all('tool_episode_plan' in job['snapshot'] for job in intents)
    assert all(job['snapshot']['tool_episode_plan']['signals']['schema'] == 'sciel-stagnation-signals/1'
               for job in intents)
    assert all('RESEARCH CELL OBSERVATIONS' not in job['snapshot']['directive'] for job in intents)
    if workers == 2:
        assert all(job['snapshot']['tool_episode_plan']['visible'] == [] for job in intents)
    traces = [json.loads(line) for line in (directory / 'artifacts/llm_calls.jsonl').read_text().splitlines()]
    tool_requests = [row for row in traces if row['event'] == 'request' and row['role'] == 'research_tool']
    assert len(tool_requests) == 2 and {row['action_id'] for row in tool_requests} == {'1', '2'}
    assert json.loads((directory / 'checkpoint.json').read_text())['research_tools'] == engine.research_tools.state()


def test_known_tool_failure_keeps_ordinary_candidate_search(tmp_path, monkeypatch):
    build, _, evaluations, _ = setup_tools(tmp_path, monkeypatch, workers=1, max_iterations=1)
    monkeypatch.setattr(FakeTools, 'execution_status', 'failed')
    result = build().run()
    assert result['B_evals'] == 1 and len(evaluations) == 2


def test_tools_preserve_hypotheses_conclusions_and_lessons_in_existing_research_loop(tmp_path, monkeypatch):
    provider = ResearchProvider(concurrent=2)
    build, _, evaluations, directory = setup_tools(tmp_path, monkeypatch, provider=provider,
        research={'enabled': True, 'reflection_interval': 2, 'max_aux_calls': None})
    engine = build()
    result = engine.run()
    assert result['B_evals'] == 2 and len(evaluations) == 3
    assert provider.roles.count('G') == 4 and provider.roles.count('C') == 2
    assert provider.roles.count('R') == 1 and result['cost_total'] == pytest.approx(.16)
    rows = events(directory)
    assert len([row for row in rows if row['event'] == 'research' and row.get('phase') == 'hypothesis']) == 2
    assert len([row for row in rows if row['event'] == 'research' and row.get('phase') == 'conclusion']) == 2
    assert engine.research.knowledge.lessons
    from sciel.hypothesis.rendering import decode_evidence
    generation_contexts = [decode_evidence(json.JSONDecoder().raw_decode(
        prompt.split('RESEARCH_CONTEXT_JSON:\n', 1)[1])[0]) for prompt in provider.calls
        if 'RESEARCH_CONTEXT_JSON:\n' in prompt and not prompt.startswith('SCIEL_RESEARCH_TOOL_EPISODE')]
    assert len(generation_contexts) == 2
    assert all(context['research_cells_context']['cards'][0]['observations']['output_excerpt'] == '{"probe": 999}'
               for context in generation_contexts)
    assert all(context['parent_score'] == .1 for context in generation_contexts)
    assert all(context['research_cells_reservation']['reserved_chars'] == 10000 for context in generation_contexts)
    outcomes = [row['evidence'] for row in rows if row['event'] == 'research' and row.get('phase') == 'outcome']
    assert all('research_cells_context' not in outcome for outcome in outcomes)
    assert all(outcome['score'] < 1 for outcome in outcomes)
    preparations = [row for row in rows if row['event'] == 'research' and row.get('phase') == 'context_prepared']
    assert len(preparations) == 2 and all(row['research_cells_context']['card_sha256'] for row in preparations)
    followups = engine.research_tools.followups
    assert len(followups) == 2
    by_id = {record['id']: record for record in outcomes}
    from sciel.engine.research import digest
    prompts = [p for p in provider.calls if 'RESEARCH_CONTEXT_JSON:\n' in p]
    for row in followups:
        record = by_id[row['experiment_id']]
        assert row['evidence_sha256'] == content_hash(record)
        assert row['exposures'] == record['method_exposures']
        assert row['exposures']['prompt_sha256'] in {digest(p) for p in prompts}
        assert row['generation']['dispatch'] == 'returned'
        assert row['generation']['response_sha256'] == record['generation_response_sha256']
        assert row['outcome']['evaluation'] == record['evaluation']
        child = engine.db.get(row['outcome']['child_program_id'])
        assert row['proposal']['source_sha256'] == digest(child.code)
        assert row['outcome']['observed_record'] == engine.observed_changes.record_for_program(child.id)
        assert row['model_use'] == row['causal_benefit'] == 'unknown'
    before_calls, before_evaluations = len(provider.calls), len(evaluations)
    restored = build(resume=True)
    try:
        assert restored.research_tools.followups == followups
        assert restored.research.knowledge.state() == engine.research.knowledge.state()
        assert len(provider.calls) == before_calls and len(evaluations) == before_evaluations
    finally:
        restored.close()


def test_retained_card_without_new_episode_gets_real_generation_followup(tmp_path, monkeypatch):
    provider = ResearchProvider(concurrent=1)
    build, _, evaluations, _ = setup_tools(tmp_path, monkeypatch, workers=1,
        max_iterations=1, provider=provider,
        research={'enabled': True, 'reflection_interval': 0, 'max_aux_calls': None})
    monkeypatch.setattr(FakeTools, 'prepare', lambda *args, **kwargs: None)
    saved = []
    def retained(self, snapshot):
        if not saved:
            saved.extend(self.saved_observation(snapshot, label='retained',
                observations={'finding':'A retained method observation.'}, commit=True)['cards'])
        return deepcopy(saved)
    monkeypatch.setattr(FakeTools, 'context', retained)
    engine = build()
    result = engine.run()
    assert FakeTools.invocations == [] and engine.research_tools.committed == []
    assert provider.roles == ['G', 'C'] and result['cost_total'] == pytest.approx(.04)
    assert len(evaluations) == 2 and len(engine.research_tools.followups) == 1
    row = engine.research_tools.followups[0]
    assert row['outcome']['child_program_id'] in engine.db._all
    assert row['exposures']['cards'][0]['source'] == saved[0]['source']
    assert row['outcome']['score'] < 1 and row['model_use'] == row['causal_benefit'] == 'unknown'


def test_no_child_still_commits_tools_without_program_link(tmp_path, monkeypatch):
    build, _, evaluations, _ = setup_tools(tmp_path, monkeypatch, workers=1, max_iterations=1)
    def no_child(op, snapshot, *args, **kwargs):
        assert 'RESEARCH CELL OBSERVATIONS' in snapshot.directive
        return NoChild(snapshot.island, 'fixture', {})
    monkeypatch.setattr('sciel.engine.loop.run_worker', no_child)
    engine = build()
    result = engine.run()
    assert result['B_evals'] == 0 and len(evaluations) == 1
    assert result['cost_total'] == pytest.approx(.03)
    assert engine.research_tools.committed == [{'action_id': '1', 'child': None, 'link_kind': 'mutation'}]


def test_tool_link_distinguishes_remeasurement_without_mutation_reward(tmp_path, monkeypatch):
    build, _, _, _ = setup_tools(tmp_path, monkeypatch, workers=1,
                                 provider=Provider('run_search', same=True))
    evaluations, _ = evaluator(monkeypatch, fail_first_candidate=True)
    engine = build()
    result = engine.run()
    assert len(evaluations) == 3 and result['B_evals'] == 1
    assert engine.n_remeasurements == engine.n_remeasurement_successes == 1
    assert engine.n_records == engine.n_events == 0 and engine.reward.state()['jumps'] == []
    assert engine.iteration_no == 2 and engine._last_pop_improve_iteration == 0
    first, second = engine.research_tools.committed
    assert first['child'] is None and second['link_kind'] == 'remeasurement'
    program = engine.db.get(second['child'])
    assert program.origin == 'remeasurement' and program.parent_id is None


def test_disabled_feature_has_no_tool_state_calls_or_changed_intent(tmp_path, monkeypatch):
    build, provider, evaluations, directory = setup_tools(tmp_path, monkeypatch, enabled=False)
    engine = build()
    result = engine.run()
    assert engine.research_tools is None and FakeTools.invocations == []
    assert len(provider.calls) == 2 and len(evaluations) == 3
    assert result['cost_total'] == pytest.approx(.06)
    state = json.loads((directory / 'checkpoint.json').read_text())
    assert 'research_tools' not in state and 'research_tools' not in state['identity']
    assert all('tool_episode_plan' not in job.get('snapshot', {})
               for path in (directory / '.parallel').rglob('jobs.json')
               for job in json.loads(path.read_text())['jobs'])
    assert not any(row['event'] == 'research_tool' for row in events(directory))
    assert 'research_tools' not in apply_defaults({})


@pytest.mark.parametrize('change', ['missing_state', 'identity', 'invalid_state'])
def test_tool_checkpoint_cannot_be_forced_to_drop_or_change_history(tmp_path, monkeypatch, change):
    build, _, _, directory = setup_tools(tmp_path, monkeypatch)
    build().run()
    path = directory / 'checkpoint.json'
    state = json.loads(path.read_text())
    if change == 'missing_state':
        state.pop('research_tools')
    elif change == 'identity':
        state['identity']['research_tools']['identity']['schema'] = 'changed'
    else:
        state['research_tools'].pop('charged')
    path.write_text(json.dumps(state))
    engine = build()
    with pytest.raises(checkpoint.IdentityMismatch, match='research tools|candidate novelty'):
        checkpoint.load(engine, str(path), allow_identity_mismatch=True)


def test_tools_require_durable_parallel_execution_without_auto_enabling_it():
    with pytest.raises(ValueError, match='research_tools requires parallel_search'):
        apply_defaults({'research_tools': TOOL_CONFIG})
