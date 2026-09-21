"""Real coordinator integration with local transports and unevaluated source fixtures."""
from collections import Counter
import ast
import hashlib
import json
from pathlib import Path
import re
import sys
import threading
import time

import pytest
import yaml

from sciel.engine.factory import build_engine
from sciel.engine import checkpoint
from sciel.campaign.budget import CampaignStopped
from sciel.engine.llm.client import LLMResponse
from sciel.engine.llm.mock import MockLLM
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.hypothesis.rendering import decode_evidence
from sciel.run_log import RunLog


def source(entry, number, comment=''):
    return f'{comment}VALUE = {number}\ndef {entry}():\n    return VALUE\n'


def number(code):
    tree = ast.parse(code)
    return next(node.value.value for node in tree.body
                if isinstance(node, ast.Assign) and node.targets[0].id == 'VALUE')


def task_fixture(root, entry='run_search', metric='combined_score'):
    directory = root / 'fixture' / entry
    directory.mkdir(parents=True)
    (directory / 'task.yaml').write_text(yaml.safe_dump(dict(suite='fixture', name=entry,
        origin='oe' if entry == 'run_search' else 'math', primary_metric=metric,
        case_metrics=[metric], tiers=['smoke', 'full'], stochastic=False,
        confirm_n=1, smoke_timeout=1, full_timeout=1)))
    (directory / 'initial_program.py').write_text(source(entry, 0))
    # Read solely for task identity/entrypoint extraction, never executed.
    (directory / 'evaluator.py').write_text(
        f'def evaluate(program):\n    assert hasattr(program, "{entry}")\n')
    return 'fixture/' + entry


def config(task, capacity, workers=4, **overrides):
    value = dict(task=task, budget=None, max_iterations=4, schedule_horizon=80,
        random_seed=23, num_islands=4, inflight_cap=1, dup_cap=4,
        schedule_policy='round_robin', milestone_fracs=[], portfolio_seed=False,
        dynamic_islands=False, embed_grid=False, embed_niche=False,
        idea_island=False, crossover=False, op_fixed='full_rewrite',
        policy={'enabled': False}, llm={'mock': True},
        candidate_novelty={'enabled': True, 'max_resamples': 2, 'neighbors': 3},
        parallel_search={'enabled': True, 'workers': workers,
            'capacity_dir': str(capacity), 'host_llm_slots': 4, 'provider_slots': 4,
            'host_evaluator_slots': 2, 'max_inflight_per_island': 1,
            'max_inflight_per_parent': 1})
    value.update(overrides)
    return value


class Provider(MockLLM):
    def __init__(self, entry, *, concurrent=1, same=False, ast_variants=False):
        super().__init__()
        self.entry, self.same, self.ast_variants = entry, same, ast_variants
        self.lock = threading.Lock()
        self.calls, self.active, self.peak, self.return_order = [], 0, 0, []
        self.barrier = threading.Barrier(concurrent) if concurrent > 1 else None
        self.concurrent = concurrent

    def generate(self, prompt, model, temperature=None, **options):
        with self.lock:
            index = len(self.calls) + 1
            self.calls.append(prompt)
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if self.barrier is not None and index <= self.concurrent:
                self.barrier.wait(timeout=3)
            # Exercise transport completion out of logical order.
            time.sleep(.004 * (5 - min(index, 4)))
            value = 1 if self.same else index
            code = source(self.entry, value, f'# variant {index}\n' if self.ast_variants else '')
            return LLMResponse('```python\n' + code + '```', model, .03, 20, 20)
        finally:
            with self.lock:
                self.active -= 1
                self.return_order.append(index)


def evaluator(monkeypatch, *, fail_first_candidate=False):
    calls, threads = [], []
    lock = threading.Lock()

    def evaluate(task, path, **options):
        code = Path(path).read_text()
        value = number(code)
        with lock:
            calls.append(code)
            threads.append(threading.get_ident())
            occurrence = sum(number(prior) == value for prior in calls)
        # Proves the immutable task/source contract reaches every isolated call.
        assert task.name in {node.name for node in ast.parse(code).body if isinstance(node, ast.FunctionDef)}
        ok = not (fail_first_candidate and value and occurrence == 1)
        score = .1 + value / 20 if ok else 0.
        result = EvalResult(ok, score, {task.primary_metric: score}, timing=.01)
        return CascadeOutcome(result, score, 'full', False)

    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate', evaluate)
    return calls, threads


def events(directory):
    return [json.loads(line) for line in (directory / 'sciel_events.jsonl').read_text().splitlines()]


def run_engine(tmp_path, monkeypatch, *, entry='run_search', metric='combined_score',
               workers=4, provider=None, evaluator_options=None, **overrides):
    tasks = tmp_path / 'tasks'
    task = task_fixture(tasks, entry, metric)
    provider = provider or Provider(entry, concurrent=workers)
    calls, threads = evaluator(monkeypatch, **(evaluator_options or {}))
    coordinator, emit = threading.get_ident(), RunLog.emit
    def checked_emit(log, *args, **kwargs):
        assert threading.get_ident() == coordinator, 'scientific journal mutated by I/O worker'
        return emit(log, *args, **kwargs)
    monkeypatch.setattr(RunLog, 'emit', checked_emit)
    directory = tmp_path / 'run'
    engine = build_engine(config(task, tmp_path / 'capacity', workers, **overrides),
        str(tasks), str(directory), python=sys.executable, llm=provider)
    result = engine.run()
    return engine, provider, calls, threads, result, directory


@pytest.mark.parametrize('entry,metric', [('run_search', 'combined_score'), ('construct_solution', 'quality')])
@pytest.mark.parametrize('workers', [1, 4])
def test_fresh_parallel_engine_has_real_bounded_transport_and_ordered_commits(tmp_path, monkeypatch, entry, metric, workers):
    coordinator = threading.get_ident()
    engine, provider, calls, threads, result, directory = run_engine(
        tmp_path, monkeypatch, entry=entry, metric=metric, workers=workers)
    assert provider.peak == workers and len(provider.calls) == 4
    assert len(calls) == 5 and coordinator not in threads
    assert result['B_evals'] == result['iteration_no'] == 4
    assert result['cost_total'] == pytest.approx(.12)
    committed = [row['job_id'] for row in events(directory)
                 if row.get('policy') == 'parallel_search' and row.get('action') == 'job_committed']
    assert committed == ['G:1', 'G:2', 'G:3', 'G:4']
    if workers == 4:
        assert provider.return_order != [1, 2, 3, 4]
    receipts = {}
    for path in (directory / '.parallel').rglob('result-G-*.json'):
        receipts[int(path.stem.rsplit('-', 1)[1])] = json.loads(path.read_text())['result']
    children = [program for program in engine.db._all.values() if program.origin != 'seed']
    assert len(children) == 4
    assert [child.code for child in children] == [receipts[i]['child_code'] for i in range(1, 5)]
    assert all(child.parent_id in {'p1', 'p2', 'p3', 'p4'} for child in children)
    assert all(engine.candidate_novelty.inspect(child.code)['matched_program_id'] == child.id for child in children)


@pytest.mark.parametrize('ast_variants', [False, True])
def test_simultaneous_duplicates_get_one_evaluation_and_bounded_paid_retries(tmp_path, monkeypatch, ast_variants):
    provider = Provider('run_search', concurrent=4, same=True, ast_variants=ast_variants)
    engine, provider, calls, _, result, directory = run_engine(tmp_path, monkeypatch, provider=provider)
    assert len(calls) == 2  # One canonical seed, one unique admitted candidate.
    assert len(provider.calls) == 10  # First job succeeds; three jobs pay at most three attempts each.
    assert result['B_evals'] == 1 and result['iteration_no'] == 4
    assert result['cost_total'] == pytest.approx(.3)
    rejected = [row for row in events(directory) if row.get('policy') == 'candidate_novelty'
                and row.get('action') == 'reject']
    assert rejected and all(row['match_kind'] == ('ast' if ast_variants else 'sha256') for row in rejected)
    assert len(engine.candidate_novelty.state()['records']) == 2


def test_failed_evaluation_remains_retryable_without_new_mutation_credit(tmp_path, monkeypatch):
    provider = Provider('run_search', same=True)
    engine, provider, calls, _, result, _ = run_engine(tmp_path, monkeypatch, provider=provider,
        workers=1, num_islands=1, max_iterations=2, evaluator_options={'fail_first_candidate': True})
    assert len(calls) == 3 and len(provider.calls) == 2
    assert result['cost_total'] == pytest.approx(.06)
    assert engine.n_remeasurements == engine.n_remeasurement_successes == 1
    assert engine.n_events == engine.n_records == engine.n_postmortems == 0
    child = engine.db.get('p2')
    assert child.origin == 'remeasurement' and child.parent_id is None


def test_frozen_parent_survives_eviction_before_later_job_commit(tmp_path, monkeypatch):
    tasks = tmp_path / 'tasks'
    task = task_fixture(tasks)
    calls, _ = evaluator(monkeypatch)
    cfg = config(task, tmp_path / 'capacity', 2, num_islands=1, pop_cap=1, max_iterations=2)
    cfg['parallel_search'].update(max_inflight_per_island=2, max_inflight_per_parent=2)
    provider = Provider('run_search', concurrent=2)
    engine = build_engine(cfg, str(tasks), str(tmp_path / 'run'), llm=provider)
    result = engine.run()
    assert result['B_evals'] == 2 and len(calls) == 3
    assert 'p1' not in engine.db.islands[0].programs
    assert engine.db.get('p1').code == source('run_search', 0)
    assert [engine.db.get(pid).parent_id for pid in ('p2', 'p3')] == ['p1', 'p1']
    intents = [json.loads(path.read_text()) for path in (tmp_path / 'run' / '.parallel').rglob('jobs.json')]
    jobs = [job for record in intents for job in record['jobs'] if job['job_id'].startswith('G:')]
    assert len(jobs) == 2
    expected = hashlib.sha256(source('run_search', 0).encode()).hexdigest()
    assert all(job['parent_source_sha256'] == expected for job in jobs)
    assert all(job['snapshot']['parent_code'] == source('run_search', 0) for job in jobs)


def test_parent_capacity_masks_selection_before_any_job_is_dispatched(tmp_path, monkeypatch):
    parallel = config('unused', tmp_path / 'capacity')['parallel_search']
    parallel['max_inflight_per_island'] = 4
    _, _, calls, _, result, directory = run_engine(tmp_path, monkeypatch,
        provider=Provider('run_search'), num_islands=1, parallel_search=parallel)
    assert result['iteration_no'] == 4 and len(calls) == 5
    waves = [json.loads(path.read_text())['jobs'] for path in (directory / '.parallel').rglob('jobs.json')]
    ordinary = [[job for job in wave if job['job_id'].startswith('G:')] for wave in waves]
    ordinary = [jobs for jobs in ordinary if jobs]
    assert min(len(jobs) for jobs in ordinary) == 1
    for jobs in ordinary:
        parents = [job['snapshot']['parent_id'] for job in jobs]
        assert len(parents) == len(set(parents))
        assert all(job['selection']['propensity'] > 0 for job in jobs)


class ResearchProvider(Provider):
    def __init__(self, *, concurrent=4):
        super().__init__('run_search', concurrent=concurrent)
        self.roles = []

    def generate(self, prompt, model, temperature=None, **options):
        role = ('I' if prompt.startswith('IDEA STAGE 1:') else
                'C' if 'Review this one experiment.' in prompt[:160] else
                'R' if 'Review existing experimental evidence and lessons.' in prompt[:180] else 'G')
        with self.lock:
            index = len(self.calls) + 1
            self.calls.append(prompt)
            self.roles.append(role)
            self.active += 1
            self.peak = max(self.peak, self.active)
        try:
            if self.barrier is not None and index <= self.concurrent:
                self.barrier.wait(timeout=3)
            time.sleep(.001 * (5 - min(index, 4)))
            if role == 'I':
                text = 'IDEA_CARD_JSON: ' + json.dumps(dict(family='fixture constants',
                    strategy='Choose a new fixed output.', mechanism='Change the VALUE constant.'))
                cost = .02
            elif role == 'C':
                raw = json.JSONDecoder().raw_decode(prompt.rsplit('EVIDENCE_JSON:\n', 1)[1].lstrip())[0]
                evidence = decode_evidence(raw)
                review = dict(claim_quote=evidence['hypothesis']['text'], source_quotes=[],
                    assessment='unknown', direct_effect='A fixture constant was changed.',
                    untested_downstream='This local fixture does not establish a causal mechanism.')
                if 'QUOTE_CATALOG_JSON:\n' in prompt:
                    from sciel.hypothesis.conclusion_refs import catalog
                    review = dict(claim_ref=catalog(evidence)['claims'][0]['id'], source_refs=[],
                        assessment='unknown', direct_effect='A fixture constant was changed.',
                        untested_downstream='No causal mechanism has been established.', evidence_kind='unavailable')
                lesson = dict(rule='Observed fixture score ' + str(evidence['score']), polarity='positive',
                    scope='branch', branch=evidence['branch'], evidence_ids=[evidence['id']])
                text = json.dumps(dict(verdict=evidence['forecast_verdict'], explanation='A measured local fixture outcome.',
                    lessons=[lesson], mechanism_review=[review]))
                cost = .01
            elif role == 'R':
                text, cost = json.dumps(dict(lessons=[], retire_ids=[])), .02
            else:
                hypothesis = dict(hypothesis='Change the fixture constant.', predicted_low=0.,
                    predicted_high=1., reasoning='Test a different fixture output.', confidence=.5)
                text = 'HYPOTHESIS_JSON: ' + json.dumps(hypothesis) + '\n```python\n' + source('run_search', index) + '```'
                cost = .03
            return LLMResponse(text, model, cost, 20, 20, finish_reason='stop')
        finally:
            with self.lock:
                self.active -= 1
                self.return_order.append(index)


@pytest.mark.parametrize('structured', [False, True])
def test_research_concludes_each_job_then_reflects_before_next_wave(tmp_path, monkeypatch, structured):
    provider = ResearchProvider()
    research = {'enabled': True, 'reflection_interval': 4, 'max_aux_calls': None}
    if structured:
        research.update(structured_auxiliary=True, conclusion_source_refs=True, conclusion_repair_attempts=1)
    engine, provider, calls, _, result, directory = run_engine(tmp_path, monkeypatch, provider=provider,
        max_iterations=8, research=research)
    assert Counter(provider.roles) == {'G': 8, 'C': 8, 'R': 2}
    assert provider.roles == (['G'] * 4 + ['C'] * 4 + ['R']) * 2
    assert len(calls) == 9 and result['cost_total'] == pytest.approx(.36)
    rows = events(directory)
    outcomes = [row for row in rows if row.get('phase') == 'outcome']
    conclusions = [row for row in rows if row.get('phase') == 'conclusion']
    assert len(outcomes) == len(conclusions) == 8
    assert all(row['status'] == 'parsed' and row['lesson_error'] is None for row in conclusions)
    assert [row['experiment_id'] for row in conclusions] == [row['experiment_id'] for row in outcomes]
    assert len(engine.research.knowledge.lessons) == 8
    second_prompts = [prompt for role, prompt in zip(provider.roles[9:], provider.calls[9:]) if role == 'G']
    assert all('Observed fixture score' in prompt for prompt in second_prompts)
    assert engine.research.pending == {} and engine.research.queue == []
    saved = json.loads((directory / 'checkpoint.json').read_text())
    assert saved['research'] == engine.research.state()


def test_two_stage_idea_is_not_dropped_from_parallel_research(tmp_path, monkeypatch):
    monkeypatch.setattr('sciel.engine.loop.OpBandit.pick',
                        lambda *args, **kwargs: ('full_rewrite', 1., {}))
    provider = ResearchProvider()
    engine, provider, calls, _, result, directory = run_engine(tmp_path, monkeypatch, provider=provider,
        op_fixed=None, idea_island=True, idea_two_stage={'enabled': True, 'card_repair_attempts': 1},
        research={'enabled': True, 'reflection_interval': 4, 'max_aux_calls': None})
    assert Counter(provider.roles) == {'G': 4, 'I': 1, 'C': 4, 'R': 1}
    assert len(calls) == 5 and result['cost_total'] == pytest.approx(.20)
    assert engine.idea.attempts == engine.idea.finished == engine.idea.evaluations == 1
    assert engine.idea.calls_started == engine.idea.calls_returned == 2
    assert engine.idea.known_cost == pytest.approx(.05)
    idea_rows = [row for row in events(directory) if row.get('phase') == 'idea']
    assert any(row.get('action') == 'source_binding' for row in idea_rows)
    assert engine.research.valid_hypotheses == 4 and len(engine.research.conclusions) == 4


def test_crash_after_duplicate_retry_receipts_replays_without_provider_or_evaluator(tmp_path, monkeypatch):
    from sciel.engine.parallel_store import WaveTransaction
    tasks = tmp_path / 'tasks'
    task = task_fixture(tasks)
    cfg = config(task, tmp_path / 'capacity')
    directory = tmp_path / 'run'
    provider = Provider('run_search', concurrent=4, same=True, ast_variants=True)
    calls, _ = evaluator(monkeypatch)
    first = build_engine(cfg, str(tasks), str(directory), llm=provider)
    original_commit = WaveTransaction.commit

    def fail_before_wave_commit(transaction):
        if transaction.wave_id == '1':
            raise RuntimeError('offline simulated crash before publication')
        return original_commit(transaction)

    with monkeypatch.context() as stopped:
        stopped.setattr(WaveTransaction, 'commit', fail_before_wave_commit)
        with pytest.raises(CampaignStopped, match='simulated crash'):
            first.run()
    assert len(provider.calls) == 10 and len(calls) == 2

    class ReplayOnly(MockLLM):
        def generate(self, *args, **kwargs):
            pytest.fail('a fully receipted wave must not dispatch another provider call')

    def no_evaluation(*args, **kwargs):
        pytest.fail('a fully receipted wave must not re-evaluate a candidate')

    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate', no_evaluation)
    resumed = build_engine(cfg, str(tasks), str(directory), llm=ReplayOnly())
    checkpoint.load(resumed, str(directory / 'checkpoint.json'))
    result = resumed.run()
    assert result['B_evals'] == 1 and result['iteration_no'] == 4
    assert result['cost_total'] == pytest.approx(.3)
