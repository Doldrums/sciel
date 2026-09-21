"""Real Engine/H15/KB wiring with inert provider and fake geometry receipts.

No candidate, evaluator, optimizer, subprocess, runtime probe or API executes.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from sciel.engine import checkpoint
from sciel.engine.repeat_promotion import RepeatPromotionError
from test_repeat_engine_integration import (batches, config, construct, draws,
    fixture_code, journal, search_observations, sha, task)


def groups(engine):
    return engine.research.knowledge.comparisons


def generation_prompts(llm):
    return [c['prompt'] for c in llm.calls if c['kind'] == 'mutation']


def test_terminal_batches_reach_next_prompt_without_new_learning_calls(
        task, tmp_path, search_observations, draws):
    engine, llm = construct(task, tmp_path / 'next-prompt')
    result = engine.run()
    saved = groups(engine)
    assert len(saved) == 2 and len(draws.calls) == 15
    assert [g['derived']['outcome'] for g in saved] == ['improved', 'regressed']
    assert [g['origin_experiment_id'] for g in saved] == ['r000001a1', 'r000002a1']
    assert all(g['task_metric'] == engine.research.metric for g in saved)
    assert all(g['forecast'] == 'not_scored' for g in saved)
    assert saved[0]['comparator_source_sha256'] == sha(Path(task.initial_program_path).read_bytes())
    assert saved[1]['comparator_source_sha256'] == sha(fixture_code(1))
    assert saved[1]['challenger_source_sha256'] == sha(fixture_code(2))
    first, second = generation_prompts(llm)
    assert 'paired_comparisons' not in first
    assert saved[0]['id'] in second and saved[1]['id'] not in second
    assert len(llm.calls) == 4 and engine.research.attempts == engine.research.outcomes == 2
    assert engine.research.aux_calls == 2 and len(engine.research.knowledge.experiments) == 2
    assert not any(e.get('phase') == 'comparison_import' for e in journal(engine.run_dir))
    assert all(not any(i.startswith('comparison:') for i in r['injected_ids'])
               for r in engine.research.knowledge.experiments)
    assert result['B_evals'] == 2 and engine.db.pop_best == 1.1 / 2.635
    for group, batch in zip(saved, batches(engine)[1:]):
        p = group['provenance']
        assert p['plan'] == batch['plan'] and p['decision'] == batch['decision']
        assert len(p['ordered_draws']) == 6
        assert [d['draw_id'] for d in p['ordered_draws']] == [d['draw_id'] for d in batch['plan']['draws']]
        for pair in group['pairs']:
            assert pair['comparator']['seed'] == pair['challenger']['seed'] == pair['seed']
    assert batches(engine)[0]['plan']['kind'] == 'bootstrap'
    assert all(g['batch_id'] != 'b000000' for g in saved)


def test_confirmed_comparator_is_not_substituted_with_generation_parent(
        task, tmp_path, monkeypatch, search_observations, draws):
    engine, llm = construct(task, tmp_path / 'distinct-parent')
    # The second generation deliberately rewrites the original seed, while H15
    # has already promoted child1. These are different real source identities.
    monkeypatch.setattr(engine.selection, 'select', lambda view, **kw:
        SimpleNamespace(parent_id=view.ids[0], propensity=1.))
    engine.run()
    record = engine.research.knowledge._evidence()['r000002a1']
    group = groups(engine)[1]
    initial_sha = sha(Path(task.initial_program_path).read_bytes())
    assert record['parent_source_sha256'] == initial_sha
    assert group['comparator_source_sha256'] == sha(fixture_code(1)) != initial_sha
    assert record['actual_delta'] > 0 and record['forecast_verdict'] == 'CONFIRMED'
    assert group['derived']['outcome'] == 'regressed' and group['forecast'] == 'not_scored'
    assert group['origin_experiment_id'] == record['id']


def test_completed_import_survives_checkpoint_and_repeated_sync_is_noop(
        task, tmp_path, search_observations, draws):
    directory = tmp_path / 'resume'
    first, _ = construct(task, directory)
    first._stop_at = 1
    first.run()
    original = copy.deepcopy(groups(first))
    restored, llm = construct(task, directory, start_step=1)
    checkpoint.load(restored, str(directory / 'checkpoint.json'))
    assert groups(restored) == original
    before = restored.research.state()
    assert restored._sync_repeat_comparisons() == restored._sync_repeat_comparisons() == 0
    assert restored.research.state() == before and len(draws.calls) == 9 and not llm.calls
    restored.run()
    assert len(draws.calls) == 15 and len(groups(restored)) == 2
    assert groups(restored)[0] == original[0]
    assert original[0]['id'] in generation_prompts(llm)[0]


@pytest.mark.parametrize('after_import', [False, True])
def test_crash_between_durable_decision_and_checkpoint_reimports_without_dispatch(
        task, tmp_path, monkeypatch, search_observations, draws, after_import):
    directory = tmp_path / str(after_import)
    first, _ = construct(task, directory)
    real_import = first.research.import_comparison
    def crash(group):
        if after_import:
            real_import(group)
        raise KeyboardInterrupt('fixture crash at pure comparison import')
    monkeypatch.setattr(first.research, 'import_comparison', crash)
    with pytest.raises(KeyboardInterrupt):
        first.run()
    first.log.close()
    assert len(draws.calls) == 9
    checkpoint_before = json.loads((directory / 'checkpoint.json').read_text())
    assert checkpoint_before['milestone'] == 'repeat_before_draws'
    assert first.repeat_promotion.terminal_comparisons()[0]['provenance']['decision']['result']['status'] == 'promote'
    restored, llm = construct(task, directory, start_step=1)
    checkpoint.load(restored, str(directory / 'checkpoint.json'))
    assert not groups(restored)
    restored._stop_at = 1
    restored.run()
    assert len(draws.calls) == 9 and not llm.calls
    assert len(groups(restored)) == 1 and groups(restored)[0]['derived']['outcome'] == 'improved'
    again, later = construct(task, directory, start_step=1)
    checkpoint.load(again, str(directory / 'checkpoint.json'))
    assert again._sync_repeat_comparisons() == 0
    again.run()
    assert len(draws.calls) == 15 and len(groups(again)) == 2
    assert groups(again)[0]['id'] in generation_prompts(later)[0]


def test_infrastructure_stop_preserves_unstarted_pairs_in_one_group(
        task, tmp_path, search_observations, draws):
    draws.failure_at = 4
    draws.failure_status = 'infrastructure'
    engine, llm = construct(task, tmp_path / 'infrastructure')
    assert engine.run()['reason'] == 'repeat_stopped'
    group, = groups(engine)
    assert group['derived']['outcome'] == 'incomplete' and not group['derived']['all_valid']
    receipts = [p[k] for p in group['pairs'] for k in ('comparator', 'challenger')]
    assert sum(r is None for r in receipts) == 5
    assert [r['status'] for r in receipts if r is not None] == ['infrastructure']
    assert all(p['raw_delta'] is p['certified_delta'] is None for p in group['derived']['pairs'])
    assert len(group['provenance']['ordered_draws']) == 6 and len(draws.calls) == 4
    assert len(llm.calls) == 2 and engine.research.outcomes == 1


def test_unknown_dispatch_is_distinct_from_five_unstarted_draws_after_resume(
        task, tmp_path, search_observations, draws):
    directory = tmp_path / 'unknown'
    first, _ = construct(task, directory)
    def interrupt(call):
        if len(draws.calls) == 4:
            raise KeyboardInterrupt('fixture unknown intent')
    draws.on_draw = interrupt
    with pytest.raises(KeyboardInterrupt):
        first.run()
    first.log.close(); draws.on_draw = None
    resumed, llm = construct(task, directory, start_step=1)
    checkpoint.load(resumed, str(directory / 'checkpoint.json'))
    assert resumed.run()['reason'] == 'repeat_stopped'
    group, = groups(resumed)
    receipts = [p[k] for p in group['pairs'] for k in ('comparator', 'challenger')]
    assert sum(r is None for r in receipts) == 5
    assert [r['status'] for r in receipts if r is not None] == ['unknown']
    assert group['derived']['outcome'] == 'incomplete' and len(draws.calls) == 4 and not llm.calls


def test_missing_explicit_origin_is_not_guessed_from_matching_source(
        task, tmp_path, monkeypatch, search_observations, draws):
    import sciel.engine.loop as loop
    original = loop.run_worker
    def without_origin(*args, **kwargs):
        result = original(*args, **kwargs)
        result.meta.pop('research_id', None)
        return result
    monkeypatch.setattr(loop, 'run_worker', without_origin)
    engine, _ = construct(task, tmp_path / 'unknown-origin', settings=config(budget=1, max_iterations=1))
    engine.run()
    group, = groups(engine)
    assert group['origin_experiment_id'] is None
    assert group['branch'] == engine.research.knowledge.experiments[0]['branch']
    assert engine.research.knowledge.experiments[0]['source_sha256'] == group['challenger_source_sha256']


def test_research_off_keeps_existing_repeat_plan_shape_and_no_import(
        task, tmp_path, search_observations, draws):
    engine, llm = construct(task, tmp_path / 'off', settings=config(research=False, budget=1, max_iterations=1))
    engine.run()
    assert engine.research is None and engine._sync_repeat_comparisons() == 0
    assert all('research_context' not in b['plan'] for b in batches(engine))
    assert len(llm.calls) == 1 and len(draws.calls) == 9


def test_source_mismatched_research_context_fails_before_plan_or_draw(
        task, tmp_path, search_observations, draws):
    directory = tmp_path / 'invalid-binding'
    first, _ = construct(task, directory)
    first._stop_at = 0; first.run()
    restored, _ = construct(task, directory)
    checkpoint.load(restored, str(directory / 'checkpoint.json'))
    before = restored.repeat_promotion.state()
    try:
        with pytest.raises(RepeatPromotionError, match='source-bound'):
            restored.repeat_promotion.plan_candidate(fixture_code(1), {'validity':1.,'sum_radii':1.}, 'p2',
                research_context={'branch':'p1','origin_experiment_id':None,'source_sha256':'0'*64})
        assert restored.repeat_promotion.state() == before and len(draws.calls) == 3
    finally:
        restored.repeat_promotion.close();restored.log.close()
