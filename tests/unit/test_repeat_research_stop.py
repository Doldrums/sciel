"""A stopped post-evaluation conclusion must not claim complete repeat coverage.

Uses the existing inert Engine/provider/adapter fixtures; no candidate or
evaluator is imported or run, and no model API is used.
"""
import copy
import json
from pathlib import Path

import pytest

from sciel.campaign.budget import CampaignStopped, RequestNotDispatched
from sciel.engine.repeat_promotion import RepeatPromotion
from test_repeat_engine_integration import (config, construct, draws, fixture_code,
    journal, search_observations, semantic_search_state, sha, task)


@pytest.mark.parametrize('error', [RequestNotDispatched, CampaignStopped, RuntimeError])
def test_post_evaluation_research_stop_retains_unexamined_candidate_and_search_state(
        task, tmp_path, search_observations, draws, error):
    def stopped_engine(repeat, name):
        engine, llm = construct(task, tmp_path / name, settings=config(repeat))
        original, attempts = llm.generate, []

        def generate(prompt, model, temperature=None):
            is_conclusion = 'Review this one experiment.' in prompt
            attempts.append('conclusion' if is_conclusion else 'mutation')
            if is_conclusion:
                raise error('private provider detail must not enter H15 evidence')
            return original(prompt, model, temperature)

        llm.generate = generate
        with pytest.raises(CampaignStopped):
            engine.run()
        assert attempts == ['mutation', 'conclusion']
        return engine

    enabled = stopped_engine(True, 'enabled')
    directory = Path(enabled.run_dir)
    saved = json.loads((directory / 'checkpoint.json').read_text())
    exported = json.loads((directory / 'repeat_promotion.json').read_text())
    summary = exported['repeat_summary']
    assert not summary['coverage_complete'] and summary['status'] == 'search_step_incomplete'
    assert len(draws.calls) == 3  # Bootstrap only; never bypass unfinished research.
    assert summary['counters']['comparisons'] == 0
    assert summary['counters']['reserved_draw_attempts'] == summary['counters']['draw_attempts'] == 3
    event, = summary['unexamined_candidates']
    candidate = enabled.db.get(enabled.db.pop_best_id)
    assert event['program_id'] == candidate.id and event['source_sha256'] == sha(fixture_code(1))
    expected_error = 'RequestNotDispatched' if error is RequestNotDispatched else 'CampaignStopped'
    assert event['reason'] == 'post_evaluation_research_stopped:' + expected_error
    assert event['screening'] == {'sum_radii': 1., 'validity': 1.}
    assert not {'raw_sum', 'certified_sum', 'metrics'} & event.keys()
    assert (directory / 'repeat_promotion/sources' / (event['source_sha256'] + '.py')).read_text() == candidate.code
    initial_sha = sha(Path(task.initial_program_path).read_bytes())
    assert exported['confirmed_output_incumbent']['source_sha256'] == initial_sha
    assert exported['observed_search_best']['source_sha256'] == sha(candidate.code)
    assert saved['repeat_promotion']['data']['unexamined_candidates'] == [event]
    assert saved['repeat_promotion']['data']['stopped_reason'] == 'search_step_incomplete'
    assert journal(directory)[-1]['reason'] == 'research_stopped'
    assert enabled.B_evals == enabled.research.outcomes == 1
    assert enabled.cost_total == pytest.approx(.001)
    assert enabled.research.aux_cost == 0 and not enabled.research.knowledge.lessons
    record, = enabled.research.knowledge.experiments
    assert record['status'] == 'evaluated' and record['score'] == 1. / 2.635
    assert record['actual_delta'] == (1. / 2.635) - (.5 / 2.635)

    # The new durable marker is recoverable without permitting any repeat call.
    original = enabled.repeat_promotion
    restored = RepeatPromotion(directory, copy.deepcopy(original.config),
        copy.deepcopy(original.identity), lambda *a, **kw: pytest.fail('no repeat after research stop'))
    try:
        restored.load_state(saved['repeat_promotion'])
        restored.run_pending()
        assert restored.state() == saved['repeat_promotion']
        assert not restored.summary()['coverage_complete']
    finally:
        restored.close()
    disabled = stopped_engine(False, 'disabled')
    assert semantic_search_state(enabled) == semantic_search_state(disabled)
    assert len(draws.calls) == 3
