"""End-to-end output admission using real adapters and offline transports.

Candidate text is inspected as AST only; these tests never execute it.
"""
import hashlib
import json
from pathlib import Path
import sys
import threading

import pytest
import yaml

from sciel.campaign.budget import CampaignStopped
from sciel.config import apply_defaults
from sciel.engine import checkpoint
from sciel.engine.factory import build_engine
from sciel.engine.llm.client import LLMResponse
from sciel.engine.llm.mock import MockLLM
from sciel.engine.loop import Engine
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.eval.output_identity import canonical
from sciel.eval.registry import load_task

from test_parallel_engine import config, events, number, source, task_fixture


class Provider(MockLLM):
    def __init__(self, values):
        super().__init__()
        self.values, self.calls, self.lock = iter(values), [], threading.Lock()

    def generate(self, prompt, model, temperature=None, **options):
        with self.lock:
            value = next(self.values)
            self.calls.append(prompt)
        return LLMResponse('```python\n'+source('run_search', value)+'```', model, .03, 20, 20)


def setup(tmp_path, monkeypatch, values, outputs, *, workers=1, iterations=1,
          islands=1, portfolio=False, witness=True, failures=(), parallel=True,
          resolution=None, scores=None, research=None):
    tasks = tmp_path/'tasks'
    task = task_fixture(tasks)
    task_path = tasks/task/'task.yaml'
    spec = yaml.safe_load(task_path.read_text())
    spec['output_identity'] = dict(schema='sciel-output-identity-adapter/1',
        adapter='canonical-json/1', artifact='output_identity', input_scope='task_fixed',
        seed_scope='unpaired')
    task_path.write_text(yaml.safe_dump(spec))
    cfg = config(task, tmp_path/'capacity', workers=workers, num_islands=islands,
        max_iterations=iterations, portfolio_seed=portfolio,
        candidate_novelty=dict(enabled=True, max_resamples=2, neighbors=3, output_repeats=True))
    if resolution is not None:
        loaded=load_task(str(task_path.parent))
        cfg['candidate_novelty']['output_resolution']=dict(schema='sciel-near-output-policy/1',
            task_hash=loaded.task_hash,
            evaluator_sha256=hashlib.sha256(Path(loaded.evaluator_path).read_bytes()).hexdigest(),
            adapter='canonical-json/1', units='fixture output units', numeric_abs_tolerance=resolution)
    if research is not None:
        cfg['research']=research
    if not parallel:
        cfg['parallel_search']['enabled'] = False
    provider, evaluations, lock = Provider(values), [], threading.Lock()

    def evaluate(task, path, **options):
        code = Path(path).read_text()
        value = number(code)
        with lock:
            evaluations.append(value)
            invocation = str(len(evaluations))
        artifact = canonical(dict(schema='sciel-full-output/1',
            source_sha256=hashlib.sha256(code.encode()).hexdigest(),
            evaluator_sha256=hashlib.sha256(Path(task.evaluator_path).read_bytes()).hexdigest(),
            invocation_id=invocation, stage='full', confirmation=None, input_sha256=None,
            seed_sha256=None, output=outputs.get(value, [value])))
        score=(scores or {}).get(value,.5)
        result = EvalResult(value not in failures, score, {'quality': score}, timing=.01,
            artifacts={'output_identity':artifact} if witness else {})
        return CascadeOutcome(result, score, 'full', False)

    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate', evaluate)
    directory = tmp_path/'run'

    def build(resume=False):
        engine = build_engine(cfg, str(tasks), str(directory), python=sys.executable, llm=provider)
        if resume:
            checkpoint.load(engine, str(directory/'checkpoint.json'))
        return engine

    return build, provider, evaluations, directory


def test_repeat_then_distinct_output_with_equal_scores_admits_only_second(tmp_path, monkeypatch):
    build, provider, evaluations, directory = setup(tmp_path, monkeypatch, [1, 2], {1:[0]})
    engine = build(); result = engine.run()
    assert evaluations == [0, 1, 2] and len(provider.calls) == 2
    assert result['cost_total'] == pytest.approx(.06)
    assert result['B_evals'] == 1 and result['n_evaluation_results'] == 3
    assert result['n_output_repeats'] == 1
    assert [number(p.code) for p in engine.db._all.values()] == [0, 2]
    assert len(engine.output_novelty.state_dict()['records']) == 3
    row = next(r for r in events(directory) if r.get('policy') == 'output_novelty'
               and r['status'] == 'exact_recorded_repeat')
    assert row['match']['program_id'] == 'p1'
    assert row['binding']['island_id'] == 0
    assert row['evaluation_id'] in provider.calls[1]


def test_source_and_output_refusals_share_three_generation_cap(tmp_path, monkeypatch):
    build, provider, evaluations, directory = setup(tmp_path, monkeypatch, [0, 1, 2], {1:[0]})
    result = build().run()
    assert evaluations == [0, 1, 2] and len(provider.calls) == 3
    assert result['cost_total'] == pytest.approx(.09) and result['B_evals'] == 1


def test_serial_search_uses_same_gate_and_accounting(tmp_path, monkeypatch):
    build, provider, evaluations, directory = setup(tmp_path, monkeypatch, [1, 2],
                                                    {1:[0]}, parallel=False)
    result = build().run()
    assert evaluations == [0, 1, 2] and len(provider.calls) == 2
    assert result['B_evals'] == 1 and result['n_output_repeats'] == 1
    assert result['cost_total'] == pytest.approx(.06)


def test_three_output_repeats_end_without_candidate_but_keep_paid_measurements(tmp_path, monkeypatch):
    build, provider, evaluations, directory = setup(tmp_path, monkeypatch, [1, 2, 3],
                                                    {1:[0], 2:[0], 3:[0]})
    engine = build(); result = engine.run()
    assert evaluations == [0, 1, 2, 3] and len(provider.calls) == 3
    assert result['B_evals'] == 0 and result['cost_total'] == pytest.approx(.09)
    assert result['n_evaluation_results'] == 4 and result['n_output_repeats'] == 3
    assert len(engine.db._all) == 1
    iteration = next(r for r in events(directory) if r['event'] == 'iteration')
    assert iteration['reason'] == 'output_repeat_cap'
    assert len(iteration['meta']['evaluation_ids']) == 3
    assert len(engine.output_novelty.state_dict()['records']) == 4
    restored = build(resume=True)
    try:
        assert restored.output_novelty.state_dict() == engine.output_novelty.state_dict()
        assert restored.n_evaluation_results == 4
    finally:
        restored.close()


def test_missing_output_witness_keeps_score_ties_eligible(tmp_path, monkeypatch):
    build, provider, evaluations, directory = setup(tmp_path, monkeypatch, [1], {1:[0]}, witness=False)
    result = build().run()
    assert result['B_evals'] == 1 and result['n_output_repeats'] == 0
    assert len(provider.calls) == 1


def test_parallel_siblings_compare_before_admission_across_islands(tmp_path, monkeypatch):
    build, provider, evaluations, directory = setup(tmp_path, monkeypatch, [1, 2, 3],
        {1:[1], 2:[1]}, workers=2, iterations=2, islands=2)
    engine = build(); result = engine.run()
    assert len(provider.calls) == 3 and len(evaluations) == 4
    assert result['B_evals'] == 2 and result['n_output_repeats'] == 1
    assert result['cost_total'] == pytest.approx(.09)
    repeat = next(r for r in events(directory) if r.get('policy') == 'output_novelty'
                  and r['status'] == 'exact_recorded_repeat')
    assert repeat['match']['island_id'] != repeat['binding']['island_id']
    assert repeat['match']['evaluation_id']
    assert repeat['match']['program_id'] is None  # sibling not committed yet


def test_replay_after_all_retries_reuses_receipts_and_costs(tmp_path, monkeypatch):
    build, provider, evaluations, directory = setup(tmp_path, monkeypatch, [1, 2], {1:[0]})
    commit = Engine._commit_step
    crash = True

    def fail_once(self, *args, **kwargs):
        nonlocal crash
        if crash:
            crash = False
            raise RuntimeError('output retry admission crash')
        return commit(self, *args, **kwargs)

    monkeypatch.setattr(Engine, '_commit_step', fail_once)
    with pytest.raises(CampaignStopped, match='admission crash'):
        build().run()
    assert evaluations == [0, 1, 2] and len(provider.calls) == 2
    result = build(resume=True).run()
    assert evaluations == [0, 1, 2] and len(provider.calls) == 2
    assert result['cost_total'] == pytest.approx(.06)
    assert result['n_output_repeats'] == 1 and result['n_evaluation_results'] == 3


def test_crash_after_repeat_evaluation_replays_without_remeasuring(tmp_path, monkeypatch):
    build, provider, evaluations, directory = setup(tmp_path, monkeypatch, [1, 2], {1:[0]})
    evaluate = Engine._evaluate
    crash = True

    def fail_once(self, *args, **kwargs):
        nonlocal crash
        outcome = evaluate(self, *args, **kwargs)
        if crash and outcome.output_novelty['status'] == 'exact_recorded_repeat':
            crash = False
            raise RuntimeError('after repeat result, before regeneration')
        return outcome

    monkeypatch.setattr(Engine, '_evaluate', fail_once)
    with pytest.raises(CampaignStopped, match='Parallel wave is durably stopped'):
        build().run()
    assert evaluations == [0, 1] and len(provider.calls) == 1
    result = build(resume=True).run()
    assert evaluations == [0, 1, 2] and len(provider.calls) == 2
    assert result['cost_total'] == pytest.approx(.06) and result['B_evals'] == 1
    assert result['n_output_repeats'] == 1


def test_cannot_force_resume_without_complete_output_registry(tmp_path, monkeypatch):
    build, provider, evaluations, directory = setup(tmp_path, monkeypatch, [1], {})
    build().run()
    original = directory/'checkpoint.json'
    value = json.loads(original.read_text())
    value.pop('output_novelty')
    corrupt = directory/'missing-output-history.json'
    corrupt.write_text(json.dumps(value))
    engine = build()
    try:
        with pytest.raises(checkpoint.IdentityMismatch, match='output novelty history'):
            checkpoint.load(engine, str(corrupt), allow_identity_mismatch=True)
    finally:
        engine.close()


def test_bootstrap_repeats_do_not_create_portfolio_programs(tmp_path, monkeypatch):
    build, provider, evaluations, directory = setup(tmp_path, monkeypatch, [1, 2, 3],
        {1:[0], 2:[0], 3:[0]}, portfolio=True)
    engine = build(); engine._stop_at = 0; result = engine.run()
    assert len(engine.db._all) == 1 and result['B_evals'] == 0
    assert len(provider.calls) == 3 and len(evaluations) == 4
    assert result['cost_total'] == pytest.approx(.09)


def test_output_repeat_policy_requires_source_registry():
    with pytest.raises(ValueError, match='requires candidate_novelty'):
        apply_defaults({'candidate_novelty':dict(enabled=False, output_repeats=True)})
