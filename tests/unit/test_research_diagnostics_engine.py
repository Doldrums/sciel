"""Saved canonical failures reach research through real Engine/WAL replay.

All provider, evaluator and tool execution leaves are fixture implementations.
Authored source strings are parsed/saved only; none is imported or executed.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import monotonic, sleep

import pytest

from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.research_tools import shared_evidence
from test_independent_research_engine import designed, setup
from test_parallel_engine import events, source
from test_research_canonical_citations import persistent
from test_research_tool_result_flow import result
from test_sequential_research_episode import current_receipt, decision, finish


ERROR = "AttributeError: module 'candidate_module' has no attribute 'run_search'"


@pytest.mark.parametrize('workers', [1, 2])
def test_nested_canonical_failure_survives_pilot_prompt_pause_and_wal_replay(
        tmp_path, monkeypatch, workers):
    candidate = source('run_search', 19)
    trial_trace = 'Traceback (most recent call last):\n' + ('  fixture frame\n' * 400) + ERROR
    trial_artifact = json.dumps({'trials': [{'seed': 37, 'stderr': trial_trace}]})
    cited = {}

    def submit(prompt):
        cited['tool'] = current_receipt(prompt)
        return designed(decision(action='submit_candidate', candidate_receipt_id=cited['tool']),
            kind='closure', budget_action='none', prior_receipt_ids=[cited['tool']])

    def interpret_and_pause(prompt):
        assert len(prompt) <= 64000
        measured = persistent(prompt)['canonical_evaluations'][-1]
        cited['canonical'] = measured['artifact_id']
        assert measured['status'] == 'eval_failed'
        assert measured['score'] is None and measured['efficacy'] == 'unknown'
        # The generic outer failure is not enough: the actionable nested trial
        # exception must survive the persistent, bounded typed-evidence path.
        assert ERROR in json.dumps(measured, ensure_ascii=False)
        assert cited['canonical'] != cited['tool'] == current_receipt(prompt)
        deadline = monotonic() + 3
        while len(evaluations) < workers + 2 and monotonic() < deadline:
            sleep(.001)
        assert len(evaluations) == workers + 2
        (directory / 'STOP').touch()
        return designed(finish(conclusion='inconclusive',
            reason='The canonical trial failed; its diagnostic does not measure method benefit.'),
            kind='closure', budget_action='none',
            prior_receipt_ids=[cited['tool'], cited['canonical']])

    build, provider, backend, evaluations, directory = setup(tmp_path, monkeypatch,
        [designed(decision()), submit, interpret_and_pause],
        [result({'candidate_source': candidate, 'fixture_observation': 'saved candidate'})],
        iterations=workers, workers=workers,
        tools_overrides={'yield_after_success': False, 'max_prompt_chars': 64000})
    from sciel.engine import loop
    original_evaluate = loop.cascade_evaluate

    def failing_evaluate(task, path, **options):
        ordinary = original_evaluate(task, path, **options)
        if Path(path).read_text() != candidate:
            return ordinary
        failure = EvalResult(False, 0., {task.primary_metric: 0., 'n_trials': 1.},
            stderr='No valid trials; inspect the per-trial diagnostic.', timing=.025,
            artifacts={'trial_results': trial_artifact})
        return CascadeOutcome(failure, 0., 'full', False)

    monkeypatch.setattr(loop, 'cascade_evaluate', failing_evaluate)
    first = build()
    paused = first.run()
    assert paused['reason'] == 'paused'
    assert paused['iteration_no'] == paused['B_evals'] == paused['cost_total'] == 0
    assert len(provider.planner_prompts) == 3
    assert len(backend.calls) == evaluations.count(candidate) == 1
    assert first.research_tools.state()['lifecycle']['research_state'] == {}
    saved_checkpoint = json.loads((directory / 'checkpoint.json').read_text())
    assert saved_checkpoint['research_tools']['lifecycle']['research_state'] == {}
    physical = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (directory / '.parallel').rglob('receipt.json')}
    before = (deepcopy(provider.planner_prompts), list(provider.calls),
              deepcopy(backend.receipts), list(evaluations))

    resumed = build(resume=True)
    completed = resumed.run()
    # The failed research evaluation is measured/costed but creates no program
    # admission and does not consume an ordinary iteration.
    assert completed['iteration_no'] == completed['B_evals'] == workers
    assert resumed.n_eval_fail == 1
    assert resumed.n_evaluation_results == workers + 2  # seed + ordinary + failed research
    assert completed['cost_total'] == pytest.approx(.03 * (workers + 3))
    assert (provider.planner_prompts, provider.calls, backend.receipts, evaluations) == before
    assert evaluations.count(candidate) == len(backend.calls) == 1
    assert physical == {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (directory / '.parallel').rglob('receipt.json')}

    tools = resumed.research_tools
    cell, state = next(iter(tools.state()['lifecycle']['research_state'].items()))
    assert state['status'] == 'closed' and state['conclusion'] == 'inconclusive'
    assert [step['status'] for step in state['steps']] == ['success', 'eval_failed', 'closed']
    assert state['known_llm_cost'] == pytest.approx(.09)
    assert state['resources']['cpu_seconds'] == .01
    assert state['pending_method_trial'] is None
    canonical = tools._read_artifact(cited['canonical'], 'candidate_evaluation')
    assert canonical['source'] == candidate
    assert canonical['provenance']['cell_id'] == cell
    assert canonical['provenance']['receipt_id'] == cited['tool']
    assert canonical['evaluation']['status'] == 'eval_failed'
    assert ERROR in json.dumps(canonical, ensure_ascii=False)
    assert not [p for p in resumed.db._all.values() if p.origin == 'research_tool']
    assert not any(row.get('phase') == 'rejected' for row in events(directory))
    for marker in (directory / '.parallel').rglob('dispatched.json'):
        assert marker.with_name('receipt.json').is_file()

    # A cold checkpoint load reauthenticates the failure and reproduces its
    # shared projection, without dispatching anything or changing its status.
    restored = build(resume=True)
    assert restored.research_tools.state() == tools.state()
    shared = shared_evidence.collect(restored.research_tools.lifecycle,
        restored.research_tools.library, 1, restored.research_tools.task_identity)
    row = next(v for v in shared['canonical_evaluations']
               if v['artifact_id'] == cited['canonical'])
    assert row['status'] == 'eval_failed' and ERROR in json.dumps(row, ensure_ascii=False)
    assert cited['canonical'] not in {v['receipt_id'] for v in shared['check_index']}
    assert (provider.planner_prompts, provider.calls, backend.receipts, evaluations) == before
