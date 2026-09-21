"""An open question and its deferred method transfers have independent progress.

The Engine, controller, checkpoint and cooperative WAL are real. Provider,
evaluator and isolated tool execution are fixed fake leaves. Saved candidate
and tool source is never executed or imported.
"""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from time import monotonic, sleep
import traceback

import pytest

from sciel.engine import loop
from test_independent_research_engine import designed, setup
from test_parallel_engine import events, number, source
from test_research_canonical_citations import persistent
from test_research_tool_result_flow import result
from test_sequential_research_episode import current_receipt, decision, finish


def _state(engine):
    states = engine.research_tools.state()['lifecycle']['research_state']
    assert len(states) == 1
    return next(iter(states.values()))


@pytest.mark.parametrize('pause_after_second_tool', [False, True])
def test_open_question_progresses_and_two_transfers_survive_fifo_and_wal(
        tmp_path, monkeypatch, pause_after_second_tool):
    first_candidate, second_candidate = [source('run_search', n) for n in (19, 21)]
    first_transfer, second_transfer = [source('run_search', n) for n in (31, 32)]
    captured = {}

    def submit_first(prompt):
        captured['first_receipt'] = current_receipt(prompt)
        return designed(decision(action='submit_candidate',
            candidate_receipt_id=captured['first_receipt']),
            kind='closure', budget_action='none',
            prior_receipt_ids=[captured['first_receipt']])

    def create_second(prompt):
        measured = persistent(prompt)['canonical_evaluations'][-1]
        assert measured['status'] == 'evaluated'
        captured['first_canonical'] = measured['artifact_id']
        return designed(decision(code=decision()['code'] + '# distinct second method\n'),
            kind='follow_up', question_key='second_candidate_same_research_question',
            new_uncertainty='Does a different bounded method produce a better candidate?',
            budget_action='reallocate', budget_reason='Check another method after the first measurement.',
            prior_receipt_ids=[captured['first_receipt'], captured['first_canonical']])

    def submit_second(prompt):
        # This planner must run even though the first canonical candidate left a
        # pending transfer and no independent branch yet exists in the database.
        saved = json.loads((directory / 'checkpoint.json').read_text())
        state = next(iter(saved['research_tools']['lifecycle']['research_state'].values()))
        assert saved['iteration_no'] == 1
        assert state['status'] == 'open'
        assert state['pending_method_trial']['submission_receipt_id'] == captured['first_receipt']
        assert not [p for p in saved['db']['programs'] if p['origin'] == 'gen']
        captured['second_receipt'] = current_receipt(prompt)
        assert captured['second_receipt'] != captured['first_receipt']
        assert 'second ready candidate' in prompt
        return designed(decision(action='submit_candidate',
            candidate_receipt_id=captured['second_receipt']),
            kind='closure', budget_action='none',
            prior_receipt_ids=[captured['second_receipt']])

    def close_question(prompt):
        measured = next(row for row in persistent(prompt)['canonical_evaluations']
                        if row['source_sha256'] == hashlib.sha256(second_candidate.encode()).hexdigest())
        assert measured['status'] == 'evaluated'
        captured['second_canonical'] = measured['artifact_id']
        assert captured['second_canonical'] != captured['first_canonical']
        return designed(finish(conclusion='inconclusive',
            reason='Both candidates were evaluated; transfer to another branch is still unmeasured.'),
            kind='closure', budget_action='none')

    def review_transfer(prompt):
        # Transfer execution does not permanently retarget the scientist's
        # question to the recipient branch. Its next interpretation returns to
        # the original saved source, with the new canonical receipt available.
        view = json.JSONDecoder().raw_decode(prompt.split('FROZEN_INPUT_VIEW_JSON:\n', 1)[1])[0]
        selected = next(row for row in view['references'] if row['role'] == 'selected')
        assert selected['source_sha256'] == hashlib.sha256(
            backend.calls[0]['inputs']['parent_source'].encode()).hexdigest()
        captured.setdefault('review_targets', []).append(selected['program_id'])
        assert 'canonical_evaluations' in persistent(prompt)
        return designed(finish(conclusion='inconclusive',
            reason='The other branch has a measured candidate; a matched benefit remains untested.'),
            kind='closure', budget_action='none')

    build, provider, backend, evaluations, directory = setup(tmp_path, monkeypatch,
        [designed(decision()), submit_first, create_second, submit_second, close_question,
         review_transfer, review_transfer],
        [result({'candidate_source': first_candidate, 'seed': 101}),
         result({'candidate_source': second_candidate, 'seed': 202,
                 'marker': 'second ready candidate'}),
         result({'candidate_source': first_transfer, 'seed': 101}),
         result({'candidate_source': second_transfer, 'seed': 202})],
        iterations=5, interleaved=True, tools_overrides={'yield_after_success': True})

    # The first portfolio candidate is the one researched branch. Until the
    # question closes, the next three ordinary/portfolio attempts are invalid.
    # The fifth G call creates the independent branch from another seed alias.
    evaluate = loop.cascade_evaluate

    def controlled_evaluation(task, path, **options):
        outcome = evaluate(task, path, **options)
        value = number(Path(path).read_text())
        if 2 <= value <= 4:
            return replace(outcome,
                result=replace(outcome.result, ok=False, score=0.,
                    artifacts={'error': 'fixed invalid ordinary fixture'}),
                raw_first_score=0.)
        return outcome

    monkeypatch.setattr(loop, 'cascade_evaluate', controlled_evaluation)
    generate = provider.generate

    def observe_closed_checkpoint(prompt, *args, **kwargs):
        if not prompt.startswith('SCIEL_RESEARCH_TOOL_EPISODE') and len(provider.calls) == 4:
            saved = json.loads((directory / 'checkpoint.json').read_text())
            state = next(iter(saved['research_tools']['lifecycle']['research_state'].values()))
            assert saved['iteration_no'] == 2 and state['status'] == 'closed'
            assert state['pending_method_trial']['submission_receipt_id'] == captured['first_receipt']
            assert [item['submission_receipt_id'] for item in state['pending_method_trial_backlog']] == [
                captured['second_receipt']]
            captured['closed_checkpoint'] = deepcopy(saved['research_tools'])
        try:
            return generate(prompt, *args, **kwargs)
        except (AssertionError, KeyError, IndexError):
            captured['provider_fixture_error'] = traceback.format_exc()
            raise

    monkeypatch.setattr(provider, 'generate', observe_closed_checkpoint)
    if pause_after_second_tool:
        execute = backend.execute

        def pause_after_paid_result(*args, **kwargs):
            receipt = execute(*args, **kwargs)
            if len(backend.calls) == 2:
                deadline = monotonic() + 3
                while len(evaluations) < 5 and monotonic() < deadline:
                    sleep(.001)
                assert len(evaluations) == 5  # Seed, three G attempts, first candidate.
                (directory / 'STOP').touch()
            return receipt

        monkeypatch.setattr(backend, 'execute', pause_after_paid_result)

    def run(engine):
        try:
            return engine.run()
        except Exception:
            # The real transport correctly seals an unknown/error receipt;
            # preserve a readable assertion from our fake leaf for diagnosis.
            if captured.get('provider_fixture_error'):
                pytest.fail(captured['provider_fixture_error'])
            raise

    engine = build()
    completed = run(engine)
    physical_before = {}
    if pause_after_second_tool:
        assert completed['reason'] == 'paused' and completed['iteration_no'] == 1
        assert len(provider.planner_prompts) == 3 and len(backend.calls) == 2
        assert evaluations.count(first_candidate) == 1
        physical_before = {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (directory / '.parallel').rglob('receipt.json')}
        engine = build(resume=True)
        completed = run(engine)

    assert completed['iteration_no'] == 5
    assert len(provider.planner_prompts) == 7
    assert len(backend.calls) == 4
    assert len(evaluations) == 14  # Seed, nine G attempts, four canonical submissions.
    assert completed['cost_total'] == pytest.approx(.48)  # Nine G calls + seven research proposals.
    assert all(evaluations.count(code) == 1 for code in (
        first_candidate, second_candidate, first_transfer, second_transfer))
    assert captured['closed_checkpoint']
    assert captured['review_targets'] == [backend.calls[0]['inputs']['parent_program_id']] * 2
    assert all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest
               for path, digest in physical_before.items())

    tools = engine.research_tools
    first_record = tools._read_artifact(captured['first_receipt'], 'invocation')
    second_record = tools._read_artifact(captured['second_receipt'], 'invocation')
    assert [call['bundle']['id'] for call in backend.calls] == [
        first_record['tool_id'], second_record['tool_id'],
        first_record['tool_id'], second_record['tool_id']]
    for original, transfer in zip(backend.calls[:2], backend.calls[2:]):
        assert transfer['inputs']['parent_source'] != original['inputs']['parent_source']
        assert transfer['inputs']['arguments'] == original['inputs']['arguments']
        assert transfer['bundle']['files'] == original['bundle']['files']
        assert 'research_stage' not in transfer['inputs']
        assert 'continuation_state' not in transfer['inputs']
        assert original['inputs']['parent_program_id'] not in transfer['inputs'][
            'search_evidence']['selected_lineage']['program_ids']
    state = _state(engine)
    assert state['status'] == 'closed' and state['conclusion'] == 'inconclusive'
    assert state['pending_method_trial'] is None
    assert state['pending_method_trial_backlog'] == []
    branch_steps = [step for step in state['steps'] if step['action'] == 'branch_trial']
    assert [step['reference_receipt_id'] for step in branch_steps] == [
        captured['first_receipt'], captured['second_receipt']]
    assert all(step['benefit'] == 'not_established' for step in branch_steps)

    links = [row for row in events(directory) if row.get('phase') == 'child_link'
             and row.get('link_basis') == 'canonical_candidate_source']
    assert len(links) == 4
    for link in links:
        child = engine.db._all[link['child_program_id']]
        producer = tools._read_artifact(link['receipt_ids'][0], 'invocation')
        canonical = tools._read_artifact(link['canonical_evaluation_id'], 'candidate_evaluation')
        assert child.code == canonical['source'] == producer['receipt']['output']['observations']['candidate_source']
        assert canonical['provenance']['receipt_id'] == link['receipt_ids'][0]
    before = (deepcopy(provider.planner_prompts), deepcopy(backend.receipts), list(evaluations))
    restored = build(resume=True)
    assert restored.research_tools.state() == tools.state()
    assert (provider.planner_prompts, backend.receipts, evaluations) == before
