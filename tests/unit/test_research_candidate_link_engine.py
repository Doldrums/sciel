"""Canonical child provenance survives later experiments and whole-wave replay.

Real Engine, queue and WAL; provider, evaluator and tool execution are fake leaves.
Candidate/tool source strings are saved and inspected, never executed.
"""
from copy import deepcopy
import hashlib
import json
from time import monotonic, sleep

import pytest

from test_independent_research_engine import designed, setup
from test_parallel_engine import events, source
from test_research_canonical_citations import persistent
from test_research_tool_result_flow import result
from test_sequential_research_episode import current_receipt, decision


@pytest.mark.parametrize('workers', [1, 2])
def test_historical_candidate_producer_not_later_tool_survives_stop_and_replay(
        tmp_path, monkeypatch, workers):
    candidate = source('run_search', 19)
    cited = {}

    def submit(prompt):
        cited['producer_receipt'] = current_receipt(prompt)
        return designed(decision(action='submit_candidate',
            candidate_receipt_id=cited['producer_receipt']),
            kind='closure', budget_action='none',
            prior_receipt_ids=[cited['producer_receipt']])

    def later_experiment(prompt):
        measured = persistent(prompt)['canonical_evaluations'][-1]
        assert measured['status'] == 'evaluated'
        cited['canonical'] = measured['artifact_id']
        assert measured['efficacy'] == 'unknown'
        return designed(decision(
            code=decision()['code'] + '# A later check, not the candidate producer.\n',
            question='Does an independent perturbation change the measured outcome?',
            experiment='Measure perturbation sensitivity after the candidate was evaluated.'),
            question_key='later_perturbation_check', kind='follow_up',
            new_uncertainty='The canonical score does not measure perturbation sensitivity.',
            budget_action='reallocate', budget_reason='Test sensitivity after evaluation.',
            prior_receipt_ids=[cited['producer_receipt'], cited['canonical']])

    build, provider, backend, evaluations, directory = setup(tmp_path, monkeypatch,
        [designed(decision()), submit, later_experiment],
        [result({'candidate_source': candidate, 'saved_seed': 17}),
         result({'later_check': 'no additional gain', 'measured_gain': 0})],
        iterations=2 * workers, workers=workers,
        tools_overrides={'yield_after_success': True})
    execute = backend.execute

    def stop_after_later_tool(*args, **kwargs):
        returned = execute(*args, **kwargs)
        if len(backend.calls) == 2:
            # Drain the fixture's ordinary measurements before STOP, so replay
            # is compared only with work that already has physical receipts.
            deadline = monotonic() + 3
            while len(evaluations) < 2 * workers + 2 and monotonic() < deadline:
                sleep(.001)
            assert len(evaluations) == 2 * workers + 2
            (directory / 'STOP').touch()
        return returned

    monkeypatch.setattr(backend, 'execute', stop_after_later_tool)
    first = build()
    paused = first.run()
    assert paused['reason'] == 'paused'
    assert paused['iteration_no'] == paused['B_evals'] == workers
    assert evaluations.count(candidate) == 1 and len(backend.calls) == 2
    assert len(provider.planner_prompts) == 3
    checkpoint = json.loads((directory / 'checkpoint.json').read_text())
    saved_state = next(iter(checkpoint['research_tools']['lifecycle']['research_state'].values()))
    assert [step['status'] for step in saved_state['steps']] == ['success']
    assert not [p for p in first.db._all.values() if p.origin == 'research_tool']
    physical = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (directory / '.parallel').rglob('receipt.json')}
    before = (deepcopy(provider.planner_prompts), list(provider.calls),
              deepcopy(backend.receipts), list(evaluations))

    resumed = build(resume=True)
    completed = resumed.run()
    assert completed['iteration_no'] == 2 * workers
    assert completed['B_evals'] == 2 * workers + 1
    assert completed['cost_total'] == pytest.approx(.03 * (2 * workers + 3))
    assert (provider.planner_prompts, provider.calls, backend.receipts, evaluations) == before
    assert len(backend.calls) == 2 and evaluations.count(candidate) == 1
    assert physical == {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (directory / '.parallel').rglob('receipt.json')}

    tools = resumed.research_tools
    state = next(iter(tools.state()['lifecycle']['research_state'].values()))
    assert [step['status'] for step in state['steps']] == ['success', 'evaluated', 'success']
    assert state['status'] == 'open' and state['scientific_steps'] == 2
    producer = tools._read_artifact(cited['producer_receipt'], 'invocation')
    later = state['steps'][-1]
    assert later['receipt_id'] != cited['producer_receipt']
    assert later['tool_id'] != producer['tool_id']
    canonical = tools._read_artifact(cited['canonical'], 'candidate_evaluation')
    assert canonical['source'] == producer['receipt']['output']['observations']['candidate_source'] == candidate
    assert canonical['provenance']['receipt_id'] == cited['producer_receipt']
    children = [p for p in resumed.db._all.values() if p.origin == 'research_tool']
    assert len(children) == 1 and children[0].code == candidate
    links = [row for row in events(directory) if row.get('phase') == 'child_link'
             and row.get('child_program_id') == children[0].id]
    assert len(links) == 1
    link = links[0]
    assert link['link_kind'] == 'mutation'
    assert link['link_basis'] == 'canonical_candidate_source'
    assert link['receipt_ids'] == [cited['producer_receipt']]
    assert link['tool_ids'] == [producer['tool_id']]
    assert link['canonical_evaluation_id'] == cited['canonical']
    assert link['evaluation_id'] == canonical['evaluation']['evaluation_id']
    assert link['candidate_invocation_id'] == producer['receipt']['invocation_id']
    assert link['candidate_source_sha256'] == hashlib.sha256(candidate.encode()).hexdigest()
    assert 'method benefit' in link['scope'].lower() and 'not established' in link['scope'].lower()
    # A later question does not replace the submitted method's branch obligation.
    assert state['pending_method_trial']['submission_receipt_id'] == cited['producer_receipt']
    restored = build(resume=True)
    assert restored.research_tools.state() == tools.state()
    assert (provider.planner_prompts, provider.calls, backend.receipts, evaluations) == before
    assert [row for row in events(directory) if row.get('phase') == 'child_link'
            and row.get('child_program_id') == children[0].id] == links
