"""A later closure must not erase a real earlier measurement from ordinary G.

Real Engine/controller/queue/WAL and artifact authentication; provider, evaluator
and isolated execution are fixed fakes. Authored source is never executed.
"""
from copy import deepcopy
import hashlib
import json
import sys

import pytest

from sciel.engine import checkpoint
from sciel.engine.factory import build_engine
from sciel.methods.models import content_hash
from test_independent_research_engine import designed
from test_parallel_engine import config, evaluator, task_fixture
from test_research_experimental_protocol import comparison
from test_research_terminal_delivery_engine import delivered_cards
from test_research_tool_controller import IMAGE
from test_research_tool_result_flow import SavedExecutor, result
from test_research_yield_engine_integration import YieldProvider, generation_prompts
from test_sequential_research_episode import current_receipt, decision, finish


ROWS = [dict(arm_id=arm, seed=seed, cpu_seconds=.002,
             best_result={'value': 0.125 if arm == 'a' else 0.0625, 'valid': True},
             conditions={'input': 'same saved fixture'}, saved_state={'seed': seed})
        for arm in ('a', 'b') for seed in (11, 17)]
MEASURED = {'comparison_results': ROWS, 'measurement_marker': 'RAW_PAIRED_MEASUREMENT'}
INTERPRETATION = 'No measured advantage in this fixture; close this one question.'
NEXT = 'A different question would require a separate measured experiment.'


def _physical(directory):
    return {str(p.relative_to(directory)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (directory / '.parallel').rglob('receipt.json')}


@pytest.mark.parametrize('independent', [False, True])
def test_later_closure_preserves_original_measurement_and_exposure_after_wal_resume(
        tmp_path, monkeypatch, independent):
    tasks = tmp_path / 'tasks'
    task = task_fixture(tasks)
    evaluations, _ = evaluator(monkeypatch)
    backend = SavedExecutor([result(MEASURED)])
    monkeypatch.setattr('sciel.research_tools.library.DockerExecutor', lambda image: backend)
    directory = tmp_path / 'run'
    original = {}

    def close(prompt):
        assert current_receipt(prompt) == original['receipt_id']
        assert 'RAW_PAIRED_MEASUREMENT' in prompt
        saved = json.loads((directory / 'checkpoint.json').read_text())
        assert saved['iteration_no'] == 1
        # Pause after this response is physically returned, before ordered wave
        # commit. Recovery must consume the same charged receipt exactly once.
        (directory / 'STOP').touch()
        return designed(finish(conclusion='refuted', reason=INTERPRETATION,
                               next_decision=NEXT),
                        kind='closure', budget_action='none',
                        prior_receipt_ids=[original['receipt_id']])

    provider = YieldProvider([decision(experimental_protocol=comparison()), close])
    cfg = config(task, tmp_path / 'capacity', workers=1, num_islands=1,
        pop_cap=1, max_iterations=1,
        research={'enabled': True, 'reflection_interval': 0, 'max_aux_calls': None},
        research_tools={'enabled': True, 'image_id': IMAGE, 'interval': 1,
            'proactive_every_attempts': 1000, 'max_scientific_steps': 3,
            'yield_after_success': True, 'independent_research': independent,
            'research_jobs_per_wave': 1, 'experimental_protocol': True,
            'shared_observations': True, 'max_requests': 2,
            'max_context_chars': 10000, 'share_between_islands': False})

    def build(iterations, *, resume=False):
        engine = build_engine({**cfg, 'max_iterations': iterations}, str(tasks),
                              str(directory), python=sys.executable, llm=provider)
        if resume:
            checkpoint.load(engine, str(directory / 'checkpoint.json'))
        return engine

    first = build(1)
    first_result = first.run()
    assert first_result['iteration_no'] == 1
    assert len(provider.planner_prompts) == len(backend.calls) == 1
    saved = first.research_tools.state()
    original_card = deepcopy(next(iter(saved['lifecycle']['cards'].values())))
    original.update(original_card['source'])
    assert original['receipt_id'] and original['tool_id']
    assert original_card['observations']['resources'] == {'cpu_seconds': .01, 'wall_seconds': .02}
    assert 'RAW_PAIRED_MEASUREMENT' in original_card['observations']['output_excerpt']
    assert json.loads(original_card['observations']['output_excerpt'])['output']['observations'] == MEASURED
    assert not original_card.get('decision')
    first_card_hash = content_hash(original_card)

    second = build(2, resume=True)
    assert second.research_tools.state() == saved
    paused = second.run()
    assert paused['reason'] == 'paused' and paused['iteration_no'] == 1
    assert len(provider.planner_prompts) == 2 and len(backend.calls) == 1
    assert not (directory / 'STOP').exists()
    physical_before = _physical(directory)
    calls_before = (deepcopy(provider.planner_prompts), deepcopy(backend.receipts))
    restored = build(2, resume=True)
    done = restored.run()
    assert done['iteration_no'] == 2
    assert (provider.planner_prompts, backend.receipts) == calls_before
    assert all(_physical(directory).get(k) == v for k, v in physical_before.items())
    assert len(backend.calls) == 1
    cell = original_card['cell_id']
    state = restored.research_tools.state()['lifecycle']['research_state'][cell]
    assert state['status'] == 'closed' and state['scientific_steps'] == 1
    assert state['known_llm_cost'] == pytest.approx(.06)
    assert state['resources']['cpu_seconds'] == .01

    third = build(3, resume=True)
    assert third.research_tools.state() == restored.research_tools.state()
    # Only ordinary retrieval now: a new question is outside this regression.
    monkeypatch.setattr(third.research_tools, 'prepare', lambda *args, **kwargs: None)
    completed = third.run()
    prompts = generation_prompts(provider)
    assert len(prompts) == 3 and len(provider.planner_prompts) == 2
    assert len(backend.calls) == 1 and len(evaluations) == len(set(evaluations)) == 4
    assert completed['iteration_no'] == completed['B_evals'] == 3
    assert completed['cost_total'] == pytest.approx(.18)  # Two planners + three G/C pairs.
    cards = delivered_cards(prompts[-1], True)
    measured = [c for c in cards if c['source']['receipt_id'] == original['receipt_id']]
    closures = [c for c in cards if c.get('decision', {}).get('interpretation') == INTERPRETATION]
    assert len(measured) == len(closures) == 1
    measured, closure = measured[0], closures[0]
    assert measured is not closure
    assert measured['source'] == original
    assert measured['observations'] == original_card['observations']
    assert measured['question'] == original_card['question']
    assert measured['revision_id'] == original_card['revision_id']
    assert measured['cell_id'] == closure['cell_id'] == cell
    assert content_hash(original_card) == first_card_hash
    # A historical measurement is not a newly invoked terminal episode.
    assert closure['source']['receipt_id'] is None and closure['source']['tool_id'] is None
    assert closure['observations'] == dict(status='not_invoked_this_episode', output_excerpt='', resources={})
    assert closure['decision']['receipt_ids'] == [original['receipt_id']]
    assert closure['decision']['next_decision'] == NEXT
    assert 'RAW_PAIRED_MEASUREMENT' in prompts[-1] and INTERPRETATION in prompts[-1]
    assert not any(p.origin == 'research_tool' for p in third.db._all.values())

    # The exposure is tied to the actually delivered measured card and prompt,
    # without attributing causal benefit or claiming the model used the method.
    prompt_sha = hashlib.sha256(prompts[-1].encode()).hexdigest()
    followups = [r for r in third.research_tools.state()['followups']['records']
                 if r['exposures']['prompt_sha256'] == prompt_sha]
    assert len(followups) == 1
    assert followups[0]['parent']['program_id'] != original['candidate_id']
    linked = [c for c in followups[0]['exposures']['cards']
              if c['source']['receipt_id'] == original['receipt_id']]
    assert len(linked) == 1
    assert linked[0]['source'] == original
    assert linked[0]['card_sha256'] == content_hash(measured)
    assert linked[0]['observations_sha256'] == content_hash(original_card['observations'])
    assert followups[0]['model_use'] == followups[0]['causal_benefit'] == 'unknown'
    final_physical = _physical(directory)
    final = build(3, resume=True)
    assert final.research_tools.state() == third.research_tools.state()
    assert _physical(directory) == final_physical
    assert (provider.planner_prompts, backend.receipts) == calls_before
