"""H15 Engine integration with inert source text and fake independent draws.

The canonical Circle task is loaded only as metadata. Neither its evaluator nor
any candidate, optimizer, subprocess or model provider is executed.
"""
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import pytest
from sciel.config import apply_defaults
from sciel.engine import checkpoint
from sciel.engine.llm.client import LLMResponse
from sciel.engine.loop import Engine
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.eval.registry import load_task
from sciel.hypothesis.rendering import decode_evidence
from repeat_engine_fixtures import (
    ROOT, sha, journal, fixture_code,
    FakeLLM, task, config, construct,
    search_observations, semantic_search_state, FakeDraws, draws,
    batches, _split_observed_feedback, _observed_feedback, _comparison_normalized_calls,
)


@pytest.mark.parametrize('setting', [None, False])
def test_disabled_has_no_adapter_probe_draw_or_repeat_artifacts(
        task, tmp_path, monkeypatch, search_observations, setting):
    def forbidden(*args, **kwargs):
        raise AssertionError('disabled repeat path touched adapter')

    monkeypatch.setattr('sciel.eval.repeat_draw.adapter_identity', forbidden)
    monkeypatch.setattr('sciel.eval.repeat_draw.draw', forbidden)
    engine, llm = construct(task, tmp_path / str(setting), settings=config(setting))
    assert engine.repeat_promotion is None
    result = engine.run()
    assert 'repeat_promotion' not in result
    assert not list(Path(engine.run_dir).glob('*repeat*'))
    assert not (Path(engine.run_dir) / 'confirmed_program.py').exists()
    saved = json.loads((Path(engine.run_dir) / 'checkpoint.json').read_text())
    assert 'repeat_promotion' not in saved
    assert 'repeat_promotion' not in saved['identity']
    assert [call['kind'] for call in llm.calls] == ['mutation', 'conclusion'] * 2
    assert len(search_observations) == 3


@pytest.mark.parametrize('conflict', [
    {'confirm_records': True}, {'portfolio_seed': True}, {'dynamic_islands': True},
    {'inflight_cap': 2}, {'inflight_cap': True}, {'task': 'oe/function_minimization'},
])
def test_unsupported_engine_modes_are_rejected_before_initialization(conflict):
    with pytest.raises(ValueError, match='repeat_promotion'):
        apply_defaults(config(**conflict))


@pytest.mark.parametrize('field,bad', [
    ('draws_per_source', 2), ('seed_master', True), ('max_comparisons', -1),
    ('max_draw_attempts', False), ('max_wall_seconds', float('nan')),
    ('timeout_seconds', 0), ('cpu_seconds', .5), ('mem_bytes', True),
    ('certification_seconds', float('inf')),
])
def test_repeat_config_rejects_ambiguous_or_unbounded_budgets(field, bad):
    settings = config()
    settings['repeat_promotion'][field] = bad
    with pytest.raises(ValueError):
        apply_defaults(settings)


def test_docker_runtime_is_forwarded_to_identity_and_every_draw_without_execution(
        task, tmp_path, search_observations, draws):
    runtime = {'backend': 'docker', 'image': 'sha256:' + 'a' * 64}
    draws.expected_runtime = copy.deepcopy(runtime)
    settings = config(research=False, budget=1, max_iterations=1)
    settings['repeat_promotion']['runtime'] = copy.deepcopy(runtime)

    engine, llm = construct(task, tmp_path / 'docker-routing', settings=settings)
    assert draws.identity_calls == [{'python': sys.executable, 'runtime': runtime}]
    result = engine.run()

    assert engine.research is None
    assert result['B_evals'] == engine.iteration_no == 1
    assert [call['kind'] for call in llm.calls] == ['mutation']
    assert len(search_observations) == 2  # The seed is outside B_evals.
    assert len(draws.calls) == 9  # Bootstrap plus one fresh paired comparison.
    assert all(call['runtime'] == runtime for call in draws.calls)
    assert [batch['decision']['result']['status'] for batch in batches(engine)] == [
        'confirmed', 'promote']
    saved = json.loads((Path(engine.run_dir) / 'checkpoint.json').read_text())
    assert saved['identity']['repeat_promotion']['config']['runtime'] == runtime


def test_draws_follow_fsynced_checkpoint_and_completed_research(
        task, tmp_path, monkeypatch, search_observations, draws):
    engine, llm = construct(task, tmp_path / 'durable')
    fsynced = []
    original_fsync = os.fsync

    def tracked_fsync(fd):
        original_fsync(fd)
        stat = os.fstat(fd)
        fsynced.append((stat.st_dev, stat.st_ino))

    monkeypatch.setattr(os, 'fsync', tracked_fsync)
    observations = []

    def inspect_safe_point(call):
        folder = Path(engine.run_dir)
        path = folder / 'checkpoint.json'
        state = json.loads(path.read_text())
        stat, directory_stat = path.stat(), folder.stat()
        assert (stat.st_dev, stat.st_ino) in fsynced
        assert (directory_stat.st_dev, directory_stat.st_ino) in fsynced
        journal_stat = (folder / 'sciel_events.jsonl').stat()
        assert (journal_stat.st_dev, journal_stat.st_ino) in fsynced
        assert state['milestone'] == 'repeat_before_draws'
        assert not state['research']['pending'] and not state['research']['queue']
        b = state['repeat_promotion']['data']['batches'][-1]
        assert b['decision'] is None
        assert call['source_sha256'] in {d['source_sha256'] for d in b['plan']['draws']}
        assert state['B_evals'] == state['research']['outcomes']
        events = journal(folder)
        assert events[-1]['event'] == 'checkpoint'
        assert events[-1]['milestone'] == 'repeat_before_draws'
        if state['B_evals']:
            conclusions = [e for e in events if e.get('phase') == 'conclusion']
            assert len(conclusions) == state['B_evals']
        observations.append(state['B_evals'])

    draws.on_draw = inspect_safe_point
    result = engine.run()
    assert observations == [0] * 3 + [1] * 6 + [2] * 6
    assert len(search_observations) == 3
    assert [c['kind'] for c in llm.calls] == ['mutation', 'conclusion'] * 2
    assert result['B_evals'] == 2 and result['cost_total'] == pytest.approx(.004)
    assert engine.research.outcomes == 2
    assert [e['score'] for e in engine.research.knowledge.experiments] == [1. / 2.635, 1.1 / 2.635]
    counters = result['repeat_promotion']['repeat_summary']['counters']
    assert counters['draw_attempts'] == 15 and counters['comparisons'] == 2
    assert counters['completed'] == counters['certified_draws'] == 15
    assert counters['failed'] == counters['unknown'] == 0


def test_both_sources_get_fresh_shared_seeds_not_reused_bootstrap_samples(
        task, tmp_path, search_observations, draws):
    engine, _ = construct(task, tmp_path / 'fresh')
    engine.run()
    plans = [b['plan'] for b in batches(engine)]
    bootstrap_seeds = {d['seed'] for d in plans[0]['draws']}
    previous_seeds = bootstrap_seeds
    for plan in plans[1:]:
        assert len(plan['draws']) == 6
        seeds = {d['seed'] for d in plan['draws']}
        assert len(seeds) == 3 and seeds.isdisjoint(previous_seeds)
        for index in range(3):
            pair = [d for d in plan['draws'] if d['draw_index'] == index]
            assert len({d['seed'] for d in pair}) == 1
            assert {d['role'] for d in pair} == {'incumbent', 'challenger'}
        assert [d['role'] for d in plan['draws'][:2]] != [d['role'] for d in plan['draws'][2:4]]
        previous_seeds |= seeds
    assert [b['decision']['result']['status'] for b in batches(engine)] == ['confirmed', 'promote', 'reject']
    assert len(draws.calls) == 15


def test_repeat_does_not_change_search_rng_rewards_program_scores_or_lessons(
        task, tmp_path, search_observations, draws):
    plain, plain_llm = construct(task, tmp_path / 'plain', settings=config(None))
    plain.run()
    enabled, enabled_llm = construct(task, tmp_path / 'enabled')
    enabled.run()
    state = semantic_search_state(enabled)
    extension = state['knowledge'].pop('paired_comparisons')
    assert state == semantic_search_state(plain)
    assert len(extension['records']) == len(extension['seen']) == 2
    assert [g['derived']['outcome'] for g in extension['records']] == ['improved', 'regressed']
    # The new evidence is intentionally visible in generation2. Preserve exact
    # comparisons of every pre-existing context field and all other call data.
    assert _comparison_normalized_calls(enabled_llm.calls) == _comparison_normalized_calls(plain_llm.calls)
    shown = _observed_feedback(enabled_llm.calls, enabled.observed_changes.state()['records'],
                               extension['records'], enabled.db._all)
    plain_shown = _observed_feedback(plain_llm.calls, plain.observed_changes.state()['records'],
                                     [], plain.db._all)
    assert any(record.get('pairs') for rows in shown for record in rows)
    assert all(record['scope'] == 'unpaired' for rows in plain_shown for record in rows)
    assert len(draws.calls) == 15


def test_confirmed_output_survives_active_eviction_and_is_not_single_score_best(
        task, tmp_path, search_observations, draws):
    settings = config(container={'kind': 'pool'}, pop_cap=1)
    engine, _ = construct(task, tmp_path / 'evict', settings=settings)
    result = engine.run()
    active = list(engine.db.islands[0].programs.values())
    assert len(active) == 1 and active[0].code == fixture_code(2)
    assert all(p.code != fixture_code(1) for p in active)
    exported = result['repeat_promotion']
    assert exported['observed_search_best']['source_sha256'] == sha(fixture_code(2))
    assert exported['confirmed_output_incumbent']['source_sha256'] == sha(fixture_code(1))
    assert Path(exported['confirmed_output_incumbent']['path']).read_text() == fixture_code(1)
    assert json.loads((Path(engine.run_dir) / 'repeat_promotion.json').read_text()) == exported
    assert engine.db.pop_best == 1.1 / 2.635


def test_continuous_and_resumed_search_and_repeat_decisions_match_without_replay(
        task, tmp_path, search_observations, draws):
    whole, whole_llm = construct(task, tmp_path / 'whole')
    whole.run()
    whole_draws = copy.deepcopy(draws.calls)
    draws.calls.clear()
    first, first_llm = construct(task, tmp_path / 'split')
    first._stop_at = 1
    assert first.run()['reason'] == 'paused'
    assert len(draws.calls) == 9
    # The completed Engine must release its owned coordinator lock, even while
    # callers retain the Engine object for metrics. No manual close hides a leak.
    resumed, resumed_llm = construct(task, tmp_path / 'split', start_step=1)
    checkpoint.load(resumed, str(tmp_path / 'split/checkpoint.json'))
    resumed.run()
    assert len(draws.calls) == 15 and draws.calls == whole_draws
    left, right = semantic_search_state(resumed), semantic_search_state(whole)
    left_groups = left['knowledge'].pop('paired_comparisons')['records']
    right_groups = right['knowledge'].pop('paired_comparisons')['records']
    assert left == right
    assert len(left_groups) == len(right_groups) == 2
    for a, b in zip(left_groups, right_groups):
        a, b = copy.deepcopy(a), copy.deepcopy(b)
        # Distinct runs have distinct evidence IDs and journal-chain roots.
        # Keep every source, seed, slot order, receipt, metric and decision.
        for group in (a, b):
            group.pop('id'); group.pop('namespace')
            group['provenance'].pop('run_id')
            group['provenance']['decision_anchor'].pop('sha256')
        assert a == b
    assert _comparison_normalized_calls(first_llm.calls + resumed_llm.calls, left_groups) == \
           _comparison_normalized_calls(whole_llm.calls, right_groups)
    feedback = _observed_feedback(first_llm.calls + resumed_llm.calls,
        resumed.observed_changes.state()['records'], left_groups, resumed.db._all)
    assert feedback == _observed_feedback(whole_llm.calls, whole.observed_changes.state()['records'],
                                          right_groups, whole.db._all)
    assert any(record.get('pairs') for rows in feedback for record in rows)
    assert resumed.repeat_promotion.summary() == whole.repeat_promotion.summary()
    assert batches(resumed) == batches(whole)


@pytest.mark.parametrize('direction', ['disable', 'enable', 'config', 'identity', 'missing_state'])
def test_force_cannot_mix_repeat_contract_or_state(
        task, tmp_path, search_observations, draws, direction):
    first, _ = construct(task, tmp_path / 'bound', settings=config(None if direction == 'enable' else True))
    first._stop_at = 1
    first.run()
    path = tmp_path / 'bound/checkpoint.json'
    before_draws = len(draws.calls)
    state = json.loads(path.read_text())
    settings = config(False if direction == 'disable' else True)
    if direction == 'config':
        settings['repeat_promotion']['seed_master'] += 1
    elif direction == 'identity':
        state['identity']['repeat_promotion']['identity']['contract'] = 'foreign-contract'
    elif direction == 'missing_state':
        del state['repeat_promotion']
    path.write_text(json.dumps(state))
    resumed, _ = construct(task, tmp_path / 'bound', settings=settings, start_step=1)
    with pytest.raises(checkpoint.IdentityMismatch):
        checkpoint.load(resumed, str(path), allow_identity_mismatch=True)
    assert len(draws.calls) == before_draws
    assert resumed.B_evals == 0 and not resumed.db._all
    resumed.log.close()


@pytest.mark.parametrize('failure', ['research_presentation', 'later_component'])
def test_failed_restore_releases_repeat_lock_before_a_fresh_restore(
        task, tmp_path, monkeypatch, search_observations, draws, failure):
    directory = tmp_path / 'failed_restore'
    first, _ = construct(task, directory)
    first._stop_at = 1
    first.run()
    path = directory / 'checkpoint.json'
    original = path.read_bytes()
    saved = json.loads(original)
    prior_draws = len(draws.calls)
    rejected, rejected_llm = construct(task, directory, start_step=1)
    if failure == 'research_presentation':
        changed = copy.deepcopy(saved)
        changed['research']['presentation']['observed_comparison'] = 'invalid-presentation'
        path.write_text(json.dumps(changed))
        expected = checkpoint.IdentityMismatch
        message = 'research checkpoint presentation mismatch'
    else:
        def component_failure(_state):
            raise RuntimeError('fixture later component restore failure')

        monkeypatch.setattr(rejected.db, 'load_state', component_failure)
        expected = RuntimeError
        message = 'later component restore failure'
    try:
        with pytest.raises(expected, match=message):
            checkpoint.load(rejected, str(path))
        # The H15 restore happened before either injected failure. Its durable
        # state remains available for inspection, but it must no longer own the
        # coordinator lock when a fresh Engine loads the original checkpoint.
        assert rejected.repeat_promotion.state() == saved['repeat_promotion']
        assert len(draws.calls) == prior_draws and not rejected_llm.calls
        path.write_bytes(original)
        restored, restored_llm = construct(task, directory, start_step=1)
        try:
            checkpoint.load(restored, str(path))
            assert semantic_search_state(restored) == semantic_search_state(first)
            assert restored.repeat_promotion.state() == saved['repeat_promotion']
            assert len(draws.calls) == prior_draws and not restored_llm.calls
        finally:
            restored.repeat_promotion.close()
            restored.log.close()
    finally:
        rejected.log.close()


def test_rollback_with_later_search_activity_is_rejected_before_any_replay(
        task, tmp_path, search_observations, draws):
    directory = tmp_path / 'rollback'
    first, _ = construct(task, directory)
    first._stop_at = 1
    first.run()
    old_checkpoint = (directory / 'checkpoint.json').read_bytes()
    resumed, _ = construct(task, directory, start_step=1)
    checkpoint.load(resumed, str(directory / 'checkpoint.json'))
    resumed.run()
    prior_calls = len(draws.calls)
    (directory / 'checkpoint.json').write_bytes(old_checkpoint)
    rollback, _ = construct(task, directory, start_step=1)
    with pytest.raises(checkpoint.IdentityMismatch, match='uncheckpointed search activity'):
        checkpoint.load(rollback, str(directory / 'checkpoint.json'), allow_identity_mismatch=True)
    assert len(draws.calls) == prior_calls
    rollback.log.close()


def test_uncheckpointed_repeat_plan_rejects_even_when_search_tail_is_clean(
        task, tmp_path, search_observations, draws):
    directory = tmp_path / 'orphan'
    first, _ = construct(task, directory)
    first._stop_at = 1
    first.run()
    resumed, _ = construct(task, directory, start_step=1)
    path = directory / 'checkpoint.json'
    checkpoint.load(resumed, str(path))
    assert resumed.repeat_promotion.plan_candidate(fixture_code(2),
        {'validity': 1., 'sum_radii': 1.1}, 'uncommitted-child')
    resumed.close()  # Simulate process exit, including directory ownership.
    prior_calls = len(draws.calls)
    rolled_back, _ = construct(task, directory, start_step=1)
    with pytest.raises(checkpoint.IdentityMismatch, match='uncheckpointed repeat plan'):
        checkpoint.load(rolled_back, str(path), allow_identity_mismatch=True)
    assert len(draws.calls) == prior_calls
    rolled_back.log.close()


@pytest.mark.parametrize('status,expected_calls,expected_reason', [
    ('candidate_failure', 15, 'budget'), ('infrastructure', 4, 'repeat_stopped'),
])
def test_draw_failures_keep_original_search_outcome_and_honest_null_metrics(
        task, tmp_path, search_observations, draws, status, expected_calls, expected_reason):
    draws.failure_at = 4  # A comparison draw, after a confirmed bootstrap.
    draws.failure_status = status
    engine, _ = construct(task, tmp_path / status)
    result = engine.run()
    assert result['reason'] == expected_reason
    assert len(draws.calls) == expected_calls
    records = engine.research.knowledge.experiments
    assert records[0]['status'] == 'evaluated'
    assert records[0]['score'] == 1. / 2.635
    assert records[0]['actual_delta'] == 1. / 2.635 - .5 / 2.635
    failed_batch = batches(engine)[1]
    failed_receipt = next(r for r in failed_batch['receipts'].values() if r['status'] == status)
    assert failed_receipt['raw_sum'] is None and failed_receipt['certified_sum'] is None
    assert failed_batch['decision']['result']['aggregates'] is None
    assert failed_batch['decision']['result']['raw_deltas'] is None
    assert failed_batch['decision']['result']['status'] == 'incomplete'
    if status == 'infrastructure':
        assert engine.B_evals == 1
        assert result['repeat_promotion']['confirmed_output_incumbent']['source_sha256'] == sha(
            Path(task.initial_program_path).read_bytes())


def test_bootstrap_failure_exports_uninitialized_not_observed_seed_as_confirmed(
        task, tmp_path, search_observations, draws):
    draws.failure_at = 2
    engine, llm = construct(task, tmp_path / 'failed_bootstrap')
    result = engine.run()
    assert len(draws.calls) == 3 and not llm.calls
    assert result['reason'] == 'repeat_stopped' and engine.B_evals == 0
    assert result['repeat_promotion']['confirmed_output_incumbent'] is None
    assert result['repeat_promotion']['observed_search_best']['score'] == .5 / 2.635
    assert not (Path(engine.run_dir) / 'confirmed_program.py').exists()


def test_reservation_budget_must_fit_whole_batch_before_first_comparison_draw(
        task, tmp_path, search_observations, draws):
    settings = config()
    settings['repeat_promotion']['max_draw_attempts'] = 8  # 3 fits, next 6 does not.
    engine, llm = construct(task, tmp_path / 'budget', settings=settings)
    result = engine.run()
    assert len(draws.calls) == 3
    summary = result['repeat_promotion']['repeat_summary']
    assert summary['counters']['reserved_draw_attempts'] == 3
    assert summary['budget_unavailable'] and not summary['coverage_complete']
    assert engine.B_evals == 2 and len(llm.calls) == 4
    assert result['repeat_promotion']['confirmed_output_incumbent']['source_sha256'] == sha(
        Path(task.initial_program_path).read_bytes())


def test_interrupted_dispatch_resumes_as_unknown_without_repeating_draw_or_paid_work(
        task, tmp_path, search_observations, draws):
    directory = tmp_path / 'interrupted'
    first, first_llm = construct(task, directory)

    def interrupt_after_intent(call):
        if len(draws.calls) == 4:
            raise KeyboardInterrupt('inert crash after durable dispatch intent')

    draws.on_draw = interrupt_after_intent
    with pytest.raises(KeyboardInterrupt):
        first.run()
    # Flush/close the search stream like process teardown, but do not close the
    # coordinator here: Engine must already have released its own lock.
    first.log.close()
    assert len(draws.calls) == 4 and len(first_llm.calls) == 2
    draws.on_draw = None
    resumed, resumed_llm = construct(task, directory, start_step=1)
    checkpoint.load(resumed, str(directory / 'checkpoint.json'))
    result = resumed.run()
    assert result['reason'] == 'repeat_stopped'
    assert len(draws.calls) == 4 and not resumed_llm.calls
    assert resumed.B_evals == 1 and resumed.research.outcomes == 1
    counters = result['repeat_promotion']['repeat_summary']['counters']
    assert counters['unknown'] == 1 and counters['completed'] == 3
    assert counters['held_wall_seconds'] == 6 * (30. + 5.)
    assert result['repeat_promotion']['confirmed_output_incumbent']['source_sha256'] == sha(
        Path(task.initial_program_path).read_bytes())


def test_checkpoint_fsync_failure_prevents_dispatch_and_releases_coordinator(
        task, tmp_path, monkeypatch, search_observations, draws):
    directory = tmp_path / 'fsync_failure'
    engine, llm = construct(task, directory)
    actual_fsync = os.fsync
    failed = []

    def fail_only_checkpoint_bytes(fd):
        temporary = directory / 'checkpoint.json.tmp'
        descriptor = os.fstat(fd)
        if temporary.exists():
            stat = temporary.stat()
            if (descriptor.st_dev, descriptor.st_ino) == (stat.st_dev, stat.st_ino):
                failed.append(True)
                raise OSError('fixture durable checkpoint write failed')
        actual_fsync(fd)

    monkeypatch.setattr(os, 'fsync', fail_only_checkpoint_bytes)
    with pytest.raises(OSError, match='durable checkpoint'):
        engine.run()
    assert failed == [True] and not draws.calls and not llm.calls
    assert engine.repeat_promotion.summary()['counters']['draw_attempts'] == 0
    assert engine.repeat_promotion.summary()['counters']['reserved_draw_attempts'] == 3
    assert not (directory / 'checkpoint.json').exists()
    # Acquire the OS lock from another descriptor to prove Engine cleanup,
    # without looking at a private coordinator field or triggering a draw.
    import fcntl
    with (directory / 'repeat_promotion/coordinator.lock').open('r+b') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    engine.log.close()
