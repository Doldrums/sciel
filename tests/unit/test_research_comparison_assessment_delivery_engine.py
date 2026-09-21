"""Known comparison defects must accompany measurements into real ordinary G.

This reproduces the recorded V10 Luna mechanism without its task or source:
all four arm/seed rows exist, execution succeeds, but every row's conditions
disagree with the saved design. A generic unverified-benefit caveat is not the
host's known, receipt-bound invalid-condition assessment. Engine, WAL, planner,
retrieval and artifact authentication are real; all execution leaves are inert.
"""
from copy import deepcopy
import hashlib
import json
import sys

import pytest

from sciel.engine import checkpoint
from sciel.engine.factory import build_engine
from sciel.methods.models import content_hash
from sciel.research_tools.observation_cards import format_context
from sciel.research_tools.result_projection import PROJECTION_SCHEMAS, verify_projection
from test_independent_research_engine import designed
from test_parallel_engine import config, evaluator, task_fixture
from test_research_experimental_protocol import comparison
from test_research_terminal_delivery_engine import delivered_cards
from test_research_tool_controller import IMAGE
from test_research_tool_result_flow import SavedExecutor, result
from test_research_yield_engine_integration import YieldProvider, generation_prompts
from test_sequential_research_episode import current_receipt, decision, finish


ROWS = [dict(arm_id=arm, seed=seed, cpu_seconds=.25,
             best_result={'value': 1.125 if arm == 'a' else 1.25},
             conditions={}, saved_state={'seed': seed, 'state': 'x' * 3000})
        for arm in ('a', 'b') for seed in (11, 17)]
OBSERVATIONS = {'comparison_results': ROWS, 'measured_marker': 'FOUR_EXACT_UNMATCHED_ROWS'}
INTERPRETATION = 'These four measured rows do not establish a controlled comparison.'
NEXT = 'A later experiment needs the declared common conditions in each arm.'


def _physical(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (directory / '.parallel').rglob('receipt.json')}


def _bound_warning(card, record, receipt_id, proposal_id):
    assert card['source']['receipt_id'] == receipt_id
    assert card['source']['invocation_id'] == record['receipt']['invocation_id']
    assert card['observations']['status'] == 'success'  # Execution, not experimental validity.
    warning = card.get('comparison_assessment')
    assert warning is not None, 'Known invalid conditions disappeared from the actual G card'
    assert warning['schema'] == 'sciel-research-comparison-assessment/1'
    assert warning['status'] == 'assessed'
    assert warning['receipt_id'] == receipt_id and warning['proposal_id'] == proposal_id
    assert warning['assessment']['report_shape'] == 'invalid'
    assert warning['assessment']['problems'] == ['reported conditions differ']
    assert warning['assessment']['adherence'] == 'tool_reported_unverified'
    assert card['schema'] == 'sciel-research-check/5'
    original = {'output': record['receipt']['output'], 'result': record['result'],
                'replay_policy': record['replay_policy'],
                'reason': record['receipt'].get('reason'),
                'diagnostic': record['receipt'].get('diagnostic')}
    projection = json.loads(card['observations']['output_excerpt'])
    assert projection['schema'] in PROJECTION_SCHEMAS
    verify_projection(projection, original)
    groups = projection['comparison_projection']['groups']
    observed = [row['values'] for group in groups for row in group['rows']]
    expected = [{key: row[key] for key in ('arm_id', 'seed', 'cpu_seconds', 'best_result', 'conditions')}
                for row in ROWS]
    assert sorted(observed, key=lambda row: (row['arm_id'], row['seed'])) == expected
    return warning


@pytest.mark.parametrize('research_enabled', [False, True])
@pytest.mark.parametrize('independent', [False, True])
@pytest.mark.parametrize('budget', [8000, 10000])
def test_invalid_comparison_is_bound_in_actual_generation_after_close_and_cold_wal(
        tmp_path, monkeypatch, research_enabled, independent, budget):
    tasks = tmp_path / 'tasks'
    task = task_fixture(tasks)
    evaluations, _ = evaluator(monkeypatch)
    backend = SavedExecutor([result(OBSERVATIONS)])
    monkeypatch.setattr('sciel.research_tools.library.DockerExecutor', lambda image: backend)
    directory = tmp_path / 'run'
    identity = {}

    def close(prompt):
        assert current_receipt(prompt) == identity['receipt_id']
        assert 'reported conditions differ' in prompt
        # A returned paid closure is durable while its wave is still uncommitted.
        (directory / 'STOP').touch()
        return designed(finish(conclusion='inconclusive', reason=INTERPRETATION,
                               next_decision=NEXT), kind='closure', budget_action='none',
                        prior_receipt_ids=[identity['receipt_id']])

    provider = YieldProvider([decision(experimental_protocol=comparison()), close])
    cfg = config(task, tmp_path / 'capacity', workers=1, num_islands=1,
        pop_cap=1, max_iterations=1,
        research={'enabled': research_enabled, 'reflection_interval': 0, 'max_aux_calls': None},
        research_tools={'enabled': True, 'image_id': IMAGE, 'interval': 1,
            'proactive_every_attempts': 1000, 'max_scientific_steps': 3,
            'yield_after_success': True, 'independent_research': independent,
            'research_jobs_per_wave': 1, 'experimental_protocol': True,
            'shared_observations': True, 'max_requests': 2,
            'max_context_chars': budget, 'share_between_islands': False})

    def build(iterations, *, resume=False):
        engine = build_engine({**cfg, 'max_iterations': iterations}, str(tasks),
                              str(directory), python=sys.executable, llm=provider)
        if resume:
            checkpoint.load(engine, str(directory / 'checkpoint.json'))
        return engine

    first = build(1)
    assert first.run()['iteration_no'] == 1
    saved = deepcopy(first.research_tools.state())
    cell_id, scientific = next(iter(saved['lifecycle']['research_state'].items()))
    step = scientific['steps'][0]
    assert step['comparison_report_assessment']['report_shape'] == 'invalid'
    assert step['comparison_report_assessment']['problems'] == ['reported conditions differ']
    receipt_id, proposal_id = step['receipt_id'], step['proposal_id']
    identity['receipt_id'] = receipt_id
    receipt_path = first.research_tools.path / (receipt_id + '.json')
    receipt_bytes = receipt_path.read_bytes()
    record = json.loads(receipt_bytes)['payload']
    assert record['receipt'] == backend.receipts[0]
    assert record['receipt']['output']['observations'] == OBSERVATIONS
    first_cards = delivered_cards(generation_prompts(provider)[0], research_enabled) if not independent else []
    if first_cards:
        _bound_warning(first_cards[0], record, receipt_id, proposal_id)

    second = build(2, resume=True)
    assert second.research_tools.state() == saved
    paused = second.run()
    assert paused['reason'] == 'paused' and paused['iteration_no'] == 1
    assert len(provider.planner_prompts) == 2 and len(backend.calls) == 1
    physical = _physical(directory)
    charged_calls = deepcopy(provider.planner_prompts), deepcopy(backend.receipts)
    restored = build(2, resume=True)
    assert restored.run()['iteration_no'] == 2
    assert (provider.planner_prompts, backend.receipts) == charged_calls
    assert all(_physical(directory).get(key) == digest for key, digest in physical.items())
    assert receipt_path.read_bytes() == receipt_bytes
    assert restored.research_tools.state()['lifecycle']['research_state'][cell_id]['status'] == 'closed'

    third = build(3, resume=True)
    monkeypatch.setattr(third.research_tools, 'prepare', lambda *args, **kwargs: None)
    done = third.run()
    prompts = generation_prompts(provider)
    assert len(prompts) == 3 and len(provider.planner_prompts) == 2 and len(backend.calls) == 1
    assert len(evaluations) == len(set(evaluations)) == 4
    assert done['cost_total'] == pytest.approx(.18 if research_enabled else .15)
    cards = delivered_cards(prompts[-1], research_enabled)
    assert len(format_context(cards)) <= budget
    trace = [json.loads(line) for line in (directory / 'artifacts/llm_calls.jsonl').read_text().splitlines()]
    prompt_sha = hashlib.sha256(prompts[-1].encode()).hexdigest()
    request = [row for row in trace if row.get('event') == 'request'
               and row.get('prompt_sha256') == prompt_sha]
    assert len(request) == 1
    wire = directory / 'artifacts/llm_calls' / (request[0]['call_id'] + '.prompt.txt')
    assert wire.read_bytes() == prompts[-1].encode()
    measured = [card for card in cards if card['source']['receipt_id'] == receipt_id]
    terminal = [card for card in cards if card.get('decision', {}).get('interpretation') == INTERPRETATION]
    assert len(measured) == len(terminal) == 1
    warning = _bound_warning(measured[0], record, receipt_id, proposal_id)
    assert 'reported conditions differ' in prompts[-1]
    assert measured[0]['observations']['resources'] == backend.receipts[0]['resources']
    assert terminal[0]['source']['receipt_id'] is None and terminal[0]['source']['tool_id'] is None
    assert terminal[0]['decision']['receipt_ids'] == [receipt_id]
    assert terminal[0]['decision']['next_decision'] == NEXT
    assert not terminal[0].get('comparison_assessment')  # No fictitious terminal invocation.
    assert receipt_path.read_bytes() == receipt_bytes
    assert third.research_tools.state()['lifecycle']['research_state'][cell_id]['known_llm_cost'] == pytest.approx(.06)
    assert third.research_tools.state()['lifecycle']['research_state'][cell_id]['resources']['cpu_seconds'] == .01
    assert not any(program.origin == 'research_tool' for program in third.db._all.values())

    if research_enabled:
        followups = [row for row in third.research_tools.state()['followups']['records']
                     if row['exposures']['prompt_sha256'] == prompt_sha]
        assert len(followups) == 1
        linked = [card for card in followups[0]['exposures']['cards']
                  if card['source']['receipt_id'] == receipt_id]
        assert len(linked) == 1 and linked[0]['card_sha256'] == content_hash(measured[0])
        assert followups[0]['causal_benefit'] == followups[0]['model_use'] == 'unknown'
    frozen = deepcopy(third.research_tools.state())
    final = build(3, resume=True)
    assert final.research_tools.state() == frozen
    assert warning == measured[0]['comparison_assessment']
    assert (provider.planner_prompts, backend.receipts) == charged_calls


def test_automatic_sharing_keeps_original_invalidity_and_does_not_invent_new_design(
        tmp_path, monkeypatch):
    from test_research_independent_sharing_engine import _sharing_run

    original_build, provider, backend, evaluations, directory, _ = _sharing_run(
        tmp_path, monkeypatch, pause_after_replay=True, iterations=10)
    backend.outputs = [result(OBSERVATIONS) for _ in backend.outputs]
    original_actions = list(provider.actions)

    def controlled(action):
        def respond(prompt):
            value = action(prompt)
            if value['action'] == 'create':
                value['experimental_protocol'] = comparison()
            return value
        return respond

    provider.actions = [controlled(action) for action in original_actions]
    episodes = []

    def build(*, resume=False):
        engine = original_build(resume=resume)
        execute = engine.research_tools.execute

        def capture(*args, **kwargs):
            episode = execute(*args, **kwargs)
            episodes.append(deepcopy(episode))
            return episode

        monkeypatch.setattr(engine.research_tools, 'execute', capture)
        return engine

    engine = build()
    assert engine.run()['reason'] == 'paused'
    assert len(backend.calls) == 2  # Original experiment and already paid replay.
    physical = _physical(directory)
    original_receipts = deepcopy(backend.receipts)
    engine = build(resume=True)
    done = engine.run()
    assert done['iteration_no'] == 10 and done['cost_total'] == pytest.approx(.39)
    assert backend.receipts[:2] == original_receipts
    assert len(backend.calls) == 4 and len(provider.planner_prompts) == 3
    assert len(evaluations) == len(set(evaluations)) == 11
    assert all(_physical(directory).get(key) == digest for key, digest in physical.items())
    state = engine.research_tools.state()
    method = next(iter(state['lifecycle']['methods'].values()))
    assert method['stage'] == 'released'
    assert len(method['adoptions']) == len(method['used_again']) == 1
    assert method['release_record']['measurement_reproduced'] is True
    original_id = method['local_receipt_id']
    original_record = engine.research_tools._read_artifact(original_id, 'invocation')
    original_source = original_record['target_binding']
    cards = []
    for prompt in generation_prompts(provider):
        if 'RESEARCH_CHECK_RESULTS_JSON:\n' in prompt:
            cards.extend(delivered_cards(prompt, False))
    original_cards = [card for card in cards if card['source']['receipt_id'] == original_id]
    assert original_cards, 'Sharing must not erase the original comparison defect from G'
    for card in original_cards:
        assert card['comparison_assessment']['status'] == 'assessed'
        assert card['comparison_assessment']['receipt_id'] == original_id
        assert card['comparison_assessment']['assessment']['problems'] == ['reported conditions differ']
        assert {key: card['source'][key] for key in original_source} == original_source

    # An automatic replay/trial is a different physical invocation without a
    # new authored comparison design. It must neither clear old invalidity nor
    # borrow the old proposal as if it authored this recipient's experiment.
    by_stage = {episode['plan']['stage']: episode for episode in episodes
                if episode['plan'].get('sharing_work_id')}
    assert set(by_stage) == {'independent_replay', 'adopt_trial', 'use_again'}
    sharing_receipts = set()
    for stage, episode in by_stage.items():
        assert len(episode['cards']) == 1
        card = episode['cards'][0]
        receipt_id = episode['receipt_ids'][0]
        sharing_receipts.add(receipt_id)
        record = engine.research_tools._read_artifact(receipt_id, 'invocation')
        assert card['observations']['status'] == 'success'
        assert card['source']['receipt_id'] == receipt_id != original_id
        assert card['source']['invocation_id'] == record['receipt']['invocation_id']
        assert card['comparison_assessment'] == dict(
            schema='sciel-research-comparison-assessment/1', receipt_id=receipt_id,
            proposal_id=None, status='unavailable', assessment=None)
        if stage == 'independent_replay':
            assert record['inputs'] == original_record['inputs']
            assert card['replay_comparison']['matched'] is True
        else:
            assert record['target_binding']['candidate_id'] != original_source['candidate_id']
    assert len(sharing_receipts) == 3
    assert all(row['status'] == 'closed' for row in state['lifecycle']['research_state'].values())
    calls = deepcopy(backend.receipts), deepcopy(provider.planner_prompts), list(evaluations)
    cold = build(resume=True)
    assert cold.research_tools.state() == state
    assert (backend.receipts, provider.planner_prompts, evaluations) == calls
