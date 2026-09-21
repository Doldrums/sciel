"""Bounded failure delivery, including the saved v5 G19 serialization failure."""
from copy import deepcopy
from research_target_fixtures import bind_snapshot, binding
from pathlib import Path

import pytest

from sciel.research_tools.controller import ResearchTools, _feedback_card, _head_tail
from test_research_tool_controller_regressions import (
    Executor, IO, LLM, controller, create, prepare, snapshot)

STDERR = (Path(__file__).parents[1] / 'fixtures/research_tool_v5_g19_stderr.txt').read_text()
CAUSE = 'TypeError: Object of type bool is not JSON serializable'


def failure(stderr=STDERR):
    return dict(status='failed', reason='method_failed', diagnostic=None, stderr=stderr,
                output=None, tool_id='a' * 64, receipt_id='b' * 64,
                invocation_id='c' * 64, candidate_id='p0', target_binding=binding(snapshot()), question='Inspect the saved program.',
                replay_policy='exact_observations', resources={'cpu_seconds': .1, 'wall_seconds': .2},
                result={'status': 'host_failed', 'host_status': 'failed', 'error': 'method_failed'})


def test_saved_g19_traceback_preserves_context_and_terminal_cause_in_small_feedback():
    assert CAUSE not in STDERR[:500]  # The previous head-only excerpt lost the cause.
    item = failure(); before = deepcopy(item)
    rendered = _feedback_card([item], 750)
    assert len(rendered) <= 750
    assert 'Traceback (most recent call last)' in rendered and CAUSE in rendered
    assert '[Excerpt: middle omitted.]' in rendered
    assert rendered.endswith('Receipt: ' + item['receipt_id'])
    assert item == before


@pytest.mark.parametrize('limit', [0, 5, 64, 512])
def test_head_tail_has_a_hard_bound_and_marks_omission(limit):
    rendered = _head_tail('FRAME\n' + 'x' * 10000 + '\nFINAL_CAUSE', limit)
    assert len(rendered) <= limit
    if limit >= 64:
        assert rendered.startswith('FRAME') and rendered.endswith('FINAL_CAUSE')
        assert 'middle omitted' in rendered


def test_long_semantic_error_cause_survives_large_partial_output():
    item = failure('')
    item.update(status='ok', reason=None, output={'partial': 'x' * 20000})
    item['result'] = dict(status='tool_error', host_status='ok',
                         error='Starting check\n' + 'nested diagnostic\n' * 500 + '\nFINAL_DATA_ERROR')
    rendered = _feedback_card([item], 1000)
    assert len(rendered) <= 1000 and 'FINAL_DATA_ERROR' in rendered
    assert 'Status: tool_error' in rendered and 'Host execution: ok' in rendered


def test_actual_second_prompt_keeps_g19_cause_at_minimum_prompt_budget(tmp_path):
    def replace_stderr(receipt):
        if receipt['status'] == 'failed':
            receipt['stderr'] = STDERR
    tools, backend, _, costs = controller(tmp_path,
        Executor(first_failure=True, mutate=replace_stderr), max_requests=2,
        max_prompt_chars=8000, max_context_chars=4096)
    current = snapshot(); current.parent_code += '# very long source\n' * 3000
    bind_snapshot(current)
    current.artifacts = {'old_failure': 'x' * 30000}
    llm = LLM([create('failed'), create('repair')])
    episode = tools.execute(prepare(tools, current), current, llm, io=IO())
    assert len(llm.calls) == len(backend.calls) == 2 and costs == [.03, .03]
    assert len(llm.calls[1]) <= 8000 and CAUSE in llm.calls[1]
    assert 'Traceback (most recent call last)' in llm.calls[1]
    assert episode['invocations'][0]['result']['status'] == 'host_failed'
    assert episode['invocations'][1]['result']['status'] == 'success'


def test_retained_failure_card_keeps_terminal_cause_after_escaped_size_compaction():
    tools = ResearchTools.__new__(ResearchTools)
    tools.config = {'max_context_chars': 4096}
    item = failure('Traceback (most recent call last):\n' + 'кадр\n' * 4000 + CAUSE)
    before = deepcopy(item)
    plan = {'cell_id': 'd' * 64, 'cell_revision_id': 'e' * 64,
            'stage': 'research', 'question': item['question']}
    card = tools._card(plan, item)
    assert len(tools.format_context([card])) <= 4096
    assert CAUSE in card['observations']['output_excerpt']
    assert card['observations']['status'] == 'host_failed'
    assert card['source']['receipt_id'] == item['receipt_id']
    assert item == before
