"""Canonical citations survive an interrupted real Engine/CooperativeIO wave.

Only fake provider, evaluator and executor leaves run. Candidate and tool source
are saved strings; neither is imported or executed.
"""
from copy import deepcopy
import hashlib
import json
from time import monotonic, sleep
from pathlib import Path

import pytest

from test_independent_research_engine import designed, setup
from test_parallel_engine import events, source
from test_research_canonical_citations import persistent
from test_research_tool_result_flow import result
from test_sequential_research_episode import current_receipt, decision, finish


@pytest.mark.parametrize('workers', [1, 2])
def test_saved_finish_cites_tool_and_canonical_evidence_and_replays_without_paid_calls(
        tmp_path, monkeypatch, workers):
    candidate = source('run_search', 19)
    cited = {}

    def submit(prompt):
        cited['tool'] = current_receipt(prompt)
        return designed(decision(action='submit_candidate', candidate_receipt_id=cited['tool']),
                        kind='closure', budget_action='none', prior_receipt_ids=[cited['tool']])

    def close_and_pause(prompt):
        measured = persistent(prompt)['canonical_evaluations'][-1]
        cited['canonical'] = measured['artifact_id']
        assert measured['status'] == 'evaluated'
        assert measured['score'] == pytest.approx(1.05)
        assert measured['efficacy'] == 'unknown'
        assert cited['tool'] == current_receipt(prompt)
        assert cited['canonical'] != cited['tool']
        # The physical response is returned and durably recorded after this
        # marker; the coordinator then pauses before committing the wave.
        # Ready-completion scheduling allows this research chain to finish
        # before ordinary siblings. This fixture intentionally pauses after
        # every first-wave evaluator has physically run, so its replay assertion
        # covers only paid receipts, not legitimate previously queued work.
        deadline = monotonic() + 3
        while len(evaluations) < workers + 2 and monotonic() < deadline:
            sleep(.001)
        assert len(evaluations) == workers + 2
        (directory / 'STOP').touch()
        return designed(finish(conclusion='inconclusive',
            reason='The candidate was evaluated, but method transfer benefit remains unmeasured.'),
            kind='closure', budget_action='none',
            prior_receipt_ids=[cited['tool'], cited['canonical']])

    build, provider, backend, evaluations, directory = setup(tmp_path, monkeypatch,
        [designed(decision()), submit, close_and_pause],
        [result({'candidate_source': candidate, 'measured_gain': 0.125})],
        iterations=workers, workers=workers, tools_overrides={'yield_after_success': False})
    # A fresh evaluator returns insertion-ordered dictionaries; durable JSON
    # sorts them. Exercise the actual canonical-feedback boundary that used to
    # change the next planner prompt even though all measurements were equal.
    from sciel.engine import loop
    evaluate = loop.cascade_evaluate
    warm_cases = []

    def unordered_evaluation(*args, **kwargs):
        outcome = evaluate(*args, **kwargs)
        outcome.result.cases = {'z_measurement': 3.0, **outcome.result.cases,
                                'a_measurement': 1.0}
        outcome.result.artifacts = {'z_diagnostic': {'z': 2, 'a': 1},
                                    'a_diagnostic': [3, 1, 2]}
        if Path(args[1]).read_text() == candidate:
            warm_cases.append(list(outcome.result.cases))
        return outcome

    monkeypatch.setattr(loop, 'cascade_evaluate', unordered_evaluation)
    first = build()
    paused = first.run()
    assert paused['reason'] == 'paused'
    assert paused['iteration_no'] == paused['B_evals'] == paused['cost_total'] == 0
    assert len(provider.planner_prompts) == 3
    assert len(backend.calls) == evaluations.count(candidate) == 1
    assert first.research_tools.state()['lifecycle']['research_state'] == {}
    assert not (directory / 'STOP').exists()
    saved = json.loads((directory / 'checkpoint.json').read_text())
    assert saved['research_tools']['lifecycle']['research_state'] == {}
    assert saved['cost_total'] == 0  # Physical charges remain in the unsettled wave's WAL.
    physical_receipts = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (directory / '.parallel').rglob('receipt.json')}
    assert physical_receipts
    assert warm_cases == [['z_measurement', 'combined_score', 'a_measurement']]
    evaluation_receipts = [json.loads(path.read_bytes())
        for path in (directory / '.parallel').rglob('receipt.json')]
    assert any(list(row.get('payload', {}).get('value', {}).get('result', {}).get('cases', {})) ==
               ['a_measurement', 'combined_score', 'z_measurement']
               for row in evaluation_receipts)
    before = (deepcopy(provider.planner_prompts), deepcopy(backend.receipts), list(evaluations))
    before_provider = list(provider.calls)

    resumed = build(resume=True)
    complete = resumed.run()
    assert complete['iteration_no'] == workers and complete['B_evals'] == workers + 1
    assert complete['cost_total'] == pytest.approx(.03 * (workers + 3))
    assert provider.planner_prompts == before[0]
    assert backend.receipts == before[1] and len(backend.calls) == 1
    assert evaluations == before[2] and len(evaluations) == workers + 2
    assert evaluations.count(candidate) == 1
    assert provider.calls == before_provider
    assert physical_receipts == {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (directory / '.parallel').rglob('receipt.json')}

    tools = resumed.research_tools
    cell, state = next(iter(tools.state()['lifecycle']['research_state'].items()))
    assert state['status'] == 'closed' and state['conclusion'] == 'inconclusive'
    assert [step['status'] for step in state['steps']] == ['success', 'evaluated', 'closed']
    assert state['scientific_steps'] == 1 and state['known_llm_cost'] == pytest.approx(.09)
    assert state['resources']['cpu_seconds'] == .01
    assert state['unknown_llm_calls'] == 0
    assert state['pending_method_trial']  # Closing this question does not fabricate transfer success.
    final_proposal = tools._read_artifact(state['steps'][-1]['proposal_id'], 'proposal')
    assert final_proposal['canonical_evaluation_ids'] == [cited['canonical']]
    assert final_proposal['evidence_receipt_ids'] == [cited['tool']]
    response = json.loads(final_proposal['response']['text'])
    assert response['experimental_protocol']['prior_receipt_ids'] == [cited['tool'], cited['canonical']]
    canonical = tools._read_artifact(cited['canonical'], 'candidate_evaluation')
    assert canonical['source'] == candidate and canonical['provenance']['cell_id'] == cell
    assert canonical['provenance']['receipt_id'] == cited['tool']
    assert canonical['evaluation']['status'] == 'evaluated'
    assert canonical['evaluation']['efficacy'] == 'unknown'
    children = [program for program in resumed.db._all.values() if program.origin == 'research_tool']
    assert len(children) == 1 and children[0].code == candidate
    journal = events(directory)
    assert not any(row.get('phase') == 'rejected' for row in journal)
    assert len([row for row in journal if row.get('action') == 'research_candidate']) == 1
    for marker in (directory / '.parallel').rglob('dispatched.json'):
        assert marker.with_name('receipt.json').is_file()
    restored = build(resume=True)
    assert restored.research_tools.state() == tools.state()
    assert evaluations == before[2] and provider.planner_prompts == before[0]
