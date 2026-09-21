"""Author feedback stays parseable at the minimum planner budget."""
from copy import deepcopy
import json

from sciel.research_tools.replay_feedback import is_projection
from test_replay_feedback_contract import feedback, result


def test_author_block_is_whole_json_with_explicit_omission_under_heavy_prior_feedback(tmp_path):
    from test_replay_feedback_delivery import completed_replay
    from test_research_tool_controller import snapshot, prepare
    tools, *_ = completed_replay(tmp_path, max_context_chars=4096, max_prompt_chars=8000)
    current = snapshot(0, 3)
    plan = prepare(tools, current)
    # Exercise the formatting bound independently of source registration count.
    value = feedback(result({str(i): 'x'*100 for i in range(8)}),
                     result({str(i): 'y'*100 for i in range(8)}))
    plan['author_replay_feedback'] = [deepcopy(value) for _ in range(8)]
    prompt = tools._prompt(plan, current, plan['visible_tool_ids'], [
        {'result': {'status': 'tool_error', 'error': 'prior error '*3000},
         'reason': 'prior diagnostic '*3000}], 1)
    assert len(prompt) <= 8000
    block, _ = json.JSONDecoder().raw_decode(prompt.split('REPLAY_FEEDBACK_JSON:\n', 1)[1])
    assert block['role'] == 'original_author'
    assert block['comparisons'] and block['comparisons_omitted'] > 0
    assert len(block['comparisons']) + block['comparisons_omitted'] == 8
    assert all(is_projection(row, value) for row in block['comparisons'])
