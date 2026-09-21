"""Timing scope regressions with inert source and fake evaluation transports only."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace as NS
import sys

import pytest

from sciel.engine import checkpoint
from sciel.engine.evaluation_timing import observation, remember, for_program, validate
from sciel.engine.factory import build_engine
from sciel.engine.prompt.build import compose_directive
from sciel.engine.research_search_evidence import search_evidence
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from test_parallel_engine import Provider, config, number, task_fixture


@pytest.mark.parametrize('stage_seconds,trials,cap,expected', [
    (68.03561210632324, 3, 30, 22.678537368774414),  # saved Heilbronn confirmation shape
    (8.885784149169922, 20, 5, .4442892074584961),  # saved oracle confirmation shape
    (72., 3, 28, 24.),  # packing-style repeated confirmation
])
def test_returned_stage_is_not_a_candidate_time(stage_seconds, trials, cap, expected):
    parent = NS(id='p25', code='inert source', parent_id=None, island=0)
    outcome = CascadeOutcome(EvalResult(True, .8, {'n_trials':float(trials)},
        timing=stage_seconds, artifacts={'eval_time':'0.001', 'candidate_seconds':'0.001'}),
        .9, 'confirm', True)
    outcome.evaluation_id = 'measurement-25'
    engine = NS(eval_times={}, evaluation_timings={}, db=NS(_all={'p25':parent}, pop_best=.8),
        observed_changes=None, iteration_no=0, B_evals=0, _per_run_budget=lambda:cap)
    remember(engine, parent, outcome)
    row = for_program(engine, parent)
    assert row['evaluation_stage_wall_seconds'] == stage_seconds
    assert row['evaluator_reported_trials'] == trials
    assert row['mean_evaluation_stage_wall_seconds_per_reported_trial'] == expected
    assert engine.eval_times['p25'] == expected
    evidence = search_evidence(engine, parent)['selected_runtime']
    assert evidence['fraction_of_wall_limit'] is None
    assert evidence['evaluation_id'] == 'measurement-25'
    assert 'overhead' in evidence['timing_scope']
    assert 'observed_wall_seconds_per_trial' not in evidence
    directive = compose_directive(cap, engine.eval_times['p25'], False,
                                  parent_timing_scope='evaluation_stage_mean')
    assert 'evaluation-stage average' in directive and 'including evaluator overhead' in directive
    assert 'recorded parent run used' not in directive
    assert 'nearly all' not in directive and 'under a quarter' not in directive


@pytest.mark.parametrize('count', [None, True, False, 0, -1, 1.5, '3', [], {},
                                  float('nan'), float('inf'), 10**400])
def test_missing_or_invalid_count_never_implies_one_candidate_run(count):
    cases = {} if count is None else {'n_trials':count}
    row = observation('inert', NS(timing=12., cases=cases))
    assert row['evaluation_stage_wall_seconds'] == 12.
    assert row['evaluator_reported_trials'] is None
    assert row['mean_evaluation_stage_wall_seconds_per_reported_trial'] is None
    assert row['trial_count_status'] == ('absent' if count is None else 'invalid')


def test_legacy_unscoped_cache_is_not_relabelled_and_source_binding_is_required():
    program = NS(id='p1', code='inert', parent_id=None, island=0)
    engine = NS(eval_times={'p1':68.}, db=NS(_all={'p1':program},pop_best=1),
                observed_changes=None, iteration_no=0, B_evals=0, _per_run_budget=lambda:30)
    assert for_program(engine, program) is None
    assert search_evidence(engine, program)['selected_runtime']['fraction_of_wall_limit'] is None
    remember(engine, program, NS(timing=68., cases={'n_trials':3}, reached='confirm'))
    changed = NS(id='p1', code='different source')
    with pytest.raises(ValueError, match='source-bound'):
        for_program(engine, changed)
    state = deepcopy(engine.evaluation_timings)
    state['p1']['mean_evaluation_stage_wall_seconds_per_reported_trial'] = 68.
    with pytest.raises(ValueError, match='mean differs'):
        validate(state, program_sources={'p1':'inert'})


def make_engine(tmp_path, monkeypatch, *, parallel, trials, portfolio=True):
    tasks = tmp_path/'tasks'
    task = task_fixture(tasks)
    calls = []
    def evaluate(task, path, **options):
        value = number(Path(path).read_text())
        calls.append(value)
        score = .1 + value / 20
        cases = {'combined_score':score}
        if trials is not None:
            cases['n_trials'] = float(trials)
        return CascadeOutcome(EvalResult(True, score, cases, timing=60.), score,
                              'confirm' if trials is not None else 'full', trials is not None)
    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate', evaluate)
    cfg = config(task, tmp_path/'capacity', workers=1, num_islands=1, max_iterations=1,
                 portfolio_seed=portfolio)
    cfg['parallel_search']['enabled'] = parallel
    provider = Provider('run_search')
    engine = build_engine(cfg, str(tasks), str(tmp_path/'run'),
                          python=sys.executable, llm=provider)
    return engine, cfg, tasks, provider, calls


@pytest.mark.parametrize('parallel', [False, True])
@pytest.mark.parametrize('trials', [3, 20, None])
def test_seed_portfolio_ordinary_and_resume_use_identical_timing_scope(tmp_path, monkeypatch, parallel, trials):
    engine, cfg, tasks, provider, calls = make_engine(tmp_path, monkeypatch, parallel=parallel, trials=trials)
    engine.run()
    assert len(calls) == 3  # initial seed, portfolio direction, ordinary mutation
    assert {p.origin for p in engine.db._all.values()} >= {'seed', 'portfolio', 'gen'}
    expected = 60. / trials if trials is not None else None
    for p in engine.db._all.values():
        assert engine.eval_times[p.id] == expected
        row = for_program(engine, p)
        assert row['evaluation_stage_wall_seconds'] == 60.
        assert row['evaluator_reported_trials'] == trials
    path = str(Path(engine.run_dir)/'checkpoint.json')
    restored = build_engine(cfg, str(tasks), engine.run_dir,
                            python=sys.executable, llm=provider)
    checkpoint.load(restored, path)
    assert restored.evaluation_timings == engine.evaluation_timings
    assert restored.eval_times == engine.eval_times
    assert len(calls) == 3  # restore performs no evaluator call
    selected = engine.db.get(engine.db.pop_best_id)
    restored_selected = restored.db.get(selected.id)
    assert search_evidence(restored, restored_selected) == search_evidence(engine, selected)
    prompt = provider.calls[-1]
    assert 'recorded parent run used' not in prompt
    if trials is not None:
        assert 'evaluation-stage average' in prompt


def test_checkpoint_rejects_changed_mean_or_source(tmp_path, monkeypatch):
    engine, cfg, tasks, provider, _ = make_engine(tmp_path, monkeypatch, parallel=True, trials=3)
    engine.run()
    path = Path(engine.run_dir)/'checkpoint.json'
    state = json.loads(path.read_text())
    ident = next(iter(state['evaluation_timings']))
    for field, bad in [('source_sha256','0'*64),
                       ('mean_evaluation_stage_wall_seconds_per_reported_trial',60.)]:
        tampered = deepcopy(state)
        tampered['evaluation_timings'][ident][field] = bad
        target = tmp_path/('bad-'+field+'.json')
        target.write_text(json.dumps(tampered))
        fresh = build_engine(cfg, str(tasks), str(tmp_path/('restore-'+field)),
                             python=sys.executable, llm=provider)
        with pytest.raises(checkpoint.IdentityMismatch, match='timing|mean differs'):
            checkpoint.load(fresh, str(target))


def test_successful_remeasurement_uses_its_own_stage_and_count(tmp_path, monkeypatch):
    from test_candidate_novelty_engine import Provider as RetryProvider, construct, fake_evaluator, outcome, settings
    provider = RetryProvider('def run_search():\n    return 312.0\n')
    engine = construct(tmp_path, settings(max_iterations=2), provider)
    measured = [outcome(), outcome(False, 0), outcome(True, .8)]
    for result, count in zip(measured, (3, 3, 20)):
        result.result.timing = 60.
        result.result.cases['n_trials'] = float(count)
    calls = fake_evaluator(monkeypatch, measured)
    engine.run()
    assert len(calls) == 3 and engine.n_remeasurement_successes == 1
    admitted = engine.db.get('p2')
    assert admitted.origin == 'remeasurement'
    row = for_program(engine, admitted)
    assert row['evaluator_reported_trials'] == 20
    assert row['evaluation_stage_wall_seconds'] == 60.
    assert engine.eval_times[admitted.id] == 3.
    assert engine.eval_times['p1'] == 20.
    assert row['evaluation_id'] != for_program(engine, engine.db.get('p1'))['evaluation_id']
