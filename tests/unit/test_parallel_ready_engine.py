"""Real Engine novelty races survive ready-order replay; all leaves are fixtures.

Candidate/tool strings are saved but never imported or executed. Thread events
select the first novelty reservation without relying on sleep/completion timing.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import threading

import pytest

from sciel.campaign.budget import CampaignStopped
from sciel.engine import loop
from sciel.engine.parallel_schedule import ResumeOrder
from test_independent_research_engine import designed, setup
from test_parallel_engine import events, number, source
from test_research_canonical_citations import persistent
from test_research_tool_result_flow import result
from test_sequential_research_episode import current_receipt, decision, finish


def _physical(directory):
    return {str(path.relative_to(directory)): hashlib.sha256(path.read_bytes()).hexdigest()
            for pattern in ('intent.json', 'dispatched.json', 'receipt.json')
            for path in (directory / '.parallel').rglob(pattern)}


@pytest.mark.parametrize('winner', ['ordinary', 'research'])
@pytest.mark.parametrize('match_kind', ['sha256', 'ast'])
@pytest.mark.parametrize('interruption', ['stop', 'head_write'])
def test_research_and_ordinary_novelty_winner_replays_exactly(
        tmp_path, monkeypatch, winner, match_kind, interruption):
    ordinary_source = source('run_search', 1)
    research_source = source('run_search', 1, '# research-only comment\n' if match_kind == 'ast' else '')
    winner_evaluated = threading.Event()
    ordinary_calls_finished = threading.Event()
    finish_returning = threading.Event()
    ordinary_attempts = 3 if winner == 'research' else 1

    def create(prompt):
        if winner == 'ordinary':
            assert winner_evaluated.wait(5), 'Ordinary G must reach its evaluator while research waits.'
        return designed(decision())

    def submit(prompt):
        return designed(decision(action='submit_candidate', candidate_receipt_id=current_receipt(prompt)),
                        kind='closure', budget_action='none')

    def close(prompt):
        measured = persistent(prompt)['canonical_evaluations'][-1]
        assert measured['status'] == ('evaluated' if winner == 'research' else 'rejected')
        assert ordinary_calls_finished.wait(5), 'Pause only after all expected ordinary calls physically return.'
        if interruption == 'stop':
            (directory / 'STOP').touch()
        finish_returning.set()
        return designed(finish(conclusion='inconclusive', reason='Only one source was canonically measured.'),
                        kind='closure', budget_action='none',
                        prior_receipt_ids=[current_receipt(prompt), measured['artifact_id']])

    build, provider, backend, evaluations, directory = setup(tmp_path, monkeypatch,
        [create, submit, close], [result({'candidate_source': research_source})],
        iterations=1, workers=1, tools_overrides={'yield_after_success': False})
    original_generate = provider.generate
    ordinary_prompts = []

    def generate(prompt, *args, **kwargs):
        if prompt.startswith('SCIEL_RESEARCH_TOOL_EPISODE'):
            return original_generate(prompt, *args, **kwargs)
        if winner == 'research':
            assert winner_evaluated.wait(5), 'Research must reach its evaluator while ordinary G waits.'
        ordinary_prompts.append(prompt)
        response = original_generate(prompt, *args, **kwargs)
        # The same candidate is returned on every retry to exercise the exact
        # additional-request cap and its paid refusal feedback.
        response.text = '```python\n' + ordinary_source + '```'
        if len(ordinary_prompts) == ordinary_attempts:
            ordinary_calls_finished.set()
        return response

    provider.generate = generate
    original_evaluate = loop.cascade_evaluate

    def evaluate(task, path, **kwargs):
        answer = original_evaluate(task, path, **kwargs)
        if number(Path(path).read_text()) == 1:
            winner_evaluated.set()
        return answer

    monkeypatch.setattr(loop, 'cascade_evaluate', evaluate)
    original_save_head = ResumeOrder._save_head
    interrupted = []

    def save_head(order):
        if (interruption == 'head_write' and finish_returning.is_set() and not interrupted
                and order.turns and order.turns[-1]['job_id'].startswith('research:')
                and order.turns[-1]['kind'] == 'llm'):
            interrupted.append(order.turns[-1]['call_key'])
            raise OSError('fixture ready-order head failure before final planner consumption')
        return original_save_head(order)

    monkeypatch.setattr(ResumeOrder, '_save_head', save_head)
    first = build()
    if interruption == 'stop':
        paused = first.run()
        assert paused['reason'] == 'paused'
    else:
        with pytest.raises(CampaignStopped, match='ready-order head failure'):
            first.run()
        assert len(interrupted) == 1
    checkpoint = json.loads((directory / 'checkpoint.json').read_text())
    assert checkpoint['iteration_no'] == checkpoint['B_evals'] == checkpoint['cost_total'] == 0
    assert len(provider.planner_prompts) == 3
    assert len(ordinary_prompts) == ordinary_attempts
    assert len(backend.calls) == 1 and len(evaluations) == 2
    assert number(evaluations[0]) == 0
    assert evaluations[1] == (research_source if winner == 'research' else ordinary_source)
    assert not (directory / 'STOP').exists()
    before = (deepcopy(provider.planner_prompts), list(ordinary_prompts), list(provider.calls),
              deepcopy(backend.receipts), list(evaluations), _physical(directory))
    first_order = {str(path.relative_to(directory)): path.read_bytes()
                   for path in (directory / '.parallel').rglob('turn-*.json')}

    # On replay every physical response is available immediately. In particular
    # the ordinary job comes first in the frozen job list, but must not steal a
    # research reservation that originally happened while ordinary G was slow.
    resumed = build(resume=True)
    complete = resumed.run()
    assert complete['iteration_no'] == complete['B_evals'] == 1
    assert complete['cost_total'] == pytest.approx(.03 * (ordinary_attempts + 3))
    assert before == (provider.planner_prompts, ordinary_prompts, provider.calls,
                      backend.receipts, evaluations, _physical(directory))
    final_order = {str(path.relative_to(directory)): path.read_bytes()
                   for path in (directory / '.parallel').rglob('turn-*.json')}
    # A paid sibling may have returned but not yet been consumed when a head
    # write fails. Its new turn is legitimate; every existing turn is immutable.
    assert all(final_order[path] == data for path, data in first_order.items())
    assert len(final_order) == len(list((directory / '.parallel').rglob('receipt.json')))
    children = [program for program in resumed.db._all.values() if program.origin != 'seed']
    assert len(children) == 1
    assert children[0].code == evaluations[1]
    assert (children[0].origin == 'research_tool') == (winner == 'research')
    assert resumed.candidate_novelty.inspect(ordinary_source)['matched_program_id'] == children[0].id
    journal = events(directory)
    committed = [row['job_id'] for row in journal if row.get('action') == 'job_committed']
    assert committed[0] == 'G:1' and len(committed) == 2 and committed[1].startswith('research:')
    refusals = [row for row in journal if row.get('policy') == 'candidate_novelty'
                and row.get('action') == 'reject']
    assert len(refusals) == (ordinary_attempts if winner == 'research' else 1)
    assert {row['match_kind'] for row in refusals} == {match_kind}
    state = next(iter(resumed.research_tools.state()['lifecycle']['research_state'].values()))
    assert state['status'] == 'closed' and state['known_llm_cost'] == pytest.approx(.09)
    for path in (directory / '.parallel').rglob('dispatched.json'):
        assert path.with_name('receipt.json').is_file()
    # Cold checkpoint restoration validates both evidence ownership and queue
    # closure, without issuing a second transport/executor/evaluator invocation.
    assert build(resume=True).research_tools.state() == resumed.research_tools.state()
