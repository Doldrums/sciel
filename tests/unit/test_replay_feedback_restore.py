"""A valid-looking comparison cannot replace its immutable replay evidence."""
from copy import deepcopy

import pytest

from test_replay_feedback_delivery import completed_replay
from test_research_tool_result_flow import controller


@pytest.mark.parametrize('tamper', ['changed_value', 'reference', 'policy'])
def test_restore_rejects_replay_feedback_not_supported_by_receipts(tmp_path, tamper):
    tools, *_ = completed_replay(tmp_path)
    saved = tools.state()
    altered = deepcopy(saved)
    comparison = next(card['replay_comparison']
                      for card in altered['lifecycle']['cards'].values()
                      if card['replay_comparison'] is not None)
    if tamper == 'changed_value':
        comparison['changed_paths'][0]['replay'] = '999'
    elif tamper == 'reference':
        comparison['reference_receipt_id'] = 'f' * 64
    else:
        comparison.update(policy='independent_execution', matched=True,
                          status='independently_executed',
                          reason='independent_execution_passed')
    restored, backend, _, costs = controller(tmp_path, [], max_requests=1)
    before = restored.state()
    with pytest.raises(ValueError, match='replay card comparison'):
        restored.load_state(altered)
    assert restored.state() == before
    assert backend.calls == [] and costs == []
    assert tools.state() == saved
