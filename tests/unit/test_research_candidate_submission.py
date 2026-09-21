"""Exact numerical artifacts use canonical gates; all evaluators are fakes."""
from unittest.mock import Mock

import pytest

from sciel.engine.research_candidate import evaluate_submission
from sciel.engine.types import Snapshot, Result, NoChild
from sciel.engine.validation import source_sha256
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult


SOURCE = 'def solve():\n    return [0.125, 0.75]\n'


def snapshot():
    return Snapshot(0, 'full_rewrite', 'mock', 3, 'p0', 'def solve():\n    return [0.0]\n',
                    0.0, required_entrypoints=['solve'])


def provenance(source=SOURCE):
    return dict(candidate_source_sha256=source_sha256(source), cell_id='cell',
                receipt_id='a'*64, tool_id='b'*64, discovery_resources={'cpu_seconds': 80.5})


def outcome(ok=True):
    value = CascadeOutcome(EvalResult(ok=ok, score=.875 if ok else 0.0,
        cases={'native': .875}, artifacts={'values': [.125,.75]}, timing=.2),
        raw_first_score=.875, reached='full')
    value.evaluation_id = 'evaluation:1'
    return value


def test_exact_code_and_provenance_reach_evaluator_without_generation():
    evaluate = Mock(return_value=outcome())
    result, feedback = evaluate_submission(SOURCE, provenance(), snapshot(), evaluate)
    evaluate.assert_called_once_with(SOURCE)
    assert isinstance(result, Result) and result.child_code == SOURCE
    assert result.origin == 'research_tool' and result.meta['op_on_policy'] is False
    assert result.cost == 0  # tool planner charges are owned by the cell ledger
    assert result.meta['research_candidate']['discovery_resources']['cpu_seconds'] == 80.5
    assert feedback['status'] == 'evaluated' and feedback['efficacy'] == 'unknown'
    assert result.meta['evaluation_count'] == 1


def test_a_saved_source_needing_graft_is_rejected_not_silently_changed():
    source = 'def another_function():\n    return 4\n'
    evaluate = Mock()
    result, feedback = evaluate_submission(source, provenance(source), snapshot(), evaluate)
    assert isinstance(result, NoChild) and feedback['status'] == 'rejected'
    assert 'repair/graft' in feedback['reason']
    evaluate.assert_not_called()


def test_digest_mismatch_cannot_dispatch():
    evaluate = Mock()
    with pytest.raises(ValueError, match='immutable receipt'):
        evaluate_submission(SOURCE+'\n', provenance(), snapshot(), evaluate)
    evaluate.assert_not_called()


def test_global_novelty_refusal_is_returned_before_evaluation():
    check = Mock(return_value=dict(accept=False, match_kind='ast',
        matched_program_id='evicted-other-island', reason='same AST'))
    check.candidate_registry = True
    evaluate = Mock()
    result, feedback = evaluate_submission(SOURCE, provenance(), snapshot(), evaluate, check)
    evaluate.assert_not_called()
    assert isinstance(result, NoChild)
    assert feedback['novelty']['matched_program_id'] == 'evicted-other-island'
    assert result.meta['evaluation_count'] == 0


def test_failed_evaluator_is_a_recorded_failure_without_efficacy_claim():
    result, feedback = evaluate_submission(SOURCE, provenance(), snapshot(), Mock(return_value=outcome(False)))
    assert isinstance(result, Result) and result.ok is False
    assert result.meta['evaluations'][0]['admission'] == 'eval_failed'
    assert feedback['status'] == 'eval_failed' and feedback['efficacy'] == 'unknown'


def test_exact_recorded_output_repeat_is_measured_but_not_admitted(tmp_path):
    from test_output_repeat_worker import evaluator
    evaluate, calls, _ = evaluator(tmp_path, [{'value':0}])
    result, feedback = evaluate_submission(SOURCE, provenance(), snapshot(), evaluate)
    assert calls == [SOURCE] and isinstance(result, NoChild)
    assert feedback['status'] == 'output_repeat_rejected'
    assert result.meta['evaluation_count'] == result.meta['output_repeat_rejections'] == 1
    assert feedback['efficacy'] == 'unknown'


def test_delayed_submission_preserves_original_studied_ancestry():
    saved = provenance()
    saved['target_binding'] = {'ancestry_parent_id':'p-previous-branch'}
    result, _ = evaluate_submission(SOURCE, saved, snapshot(), Mock(return_value=outcome()))
    assert result.parent_id == 'p-previous-branch'


def submit_fixture_candidate(episode, source, candidate_evaluator):
    """Bind the synthetic candidate to this fixture's producing observation."""
    saved = provenance(source)
    producer = episode['cards'][0]['source']
    saved.update({key: producer[key] for key in ('tool_id', 'receipt_id', 'invocation_id')})
    saved['cell_id'] = episode['cards'][0]['cell_id']
    evidence = candidate_evaluator(source, saved)
    episode['candidate_submission'] = dict(source=source, provenance=saved, evaluation=evidence)


@pytest.mark.parametrize('workers', [1, 2])
def test_engine_commits_tool_candidate_without_generator_or_operator_credit(tmp_path, monkeypatch, workers):
    from test_research_tool_engine import FakeTools, setup_tools
    original = FakeTools.execute
    def execute(self, plan, snap, llm, *, io, candidate_evaluator):
        episode = original(self, plan, snap, llm, io=io)
        source = f'VALUE = {int(plan["action_id"]) + 10}\ndef run_search():\n    return VALUE\n'
        submit_fixture_candidate(episode, source, candidate_evaluator)
        return episode
    monkeypatch.setattr(FakeTools, 'execute', execute)
    build, provider, evaluations, _ = setup_tools(tmp_path, monkeypatch, workers=workers)
    engine = build()
    observe = Mock(wraps=engine.op_bandit.observe)
    engine.op_bandit.observe = observe
    engine.run()
    assert len(evaluations) == 3  # canonical seed and two submitted artifacts
    children = [p for p in engine.db._all.values() if p.origin == 'research_tool']
    assert len(children) == 2
    assert all(p.parent_id is not None for p in children)
    assert len(engine.research_tools.committed) == len(children)
    for record in engine.research_tools.committed:
        child = engine.db._all[record['child']]
        assert record['child_source'] == child.code
        assert record['candidate_binding']['candidate_source_sha256'] == source_sha256(child.code)
        assert record['candidate_binding']['link_basis'] == 'canonical_candidate_source'
    observe.assert_not_called()
    assert len(provider.calls) == 2  # cell requests only, no extra G rewrite
    assert all('SCIEL_RESEARCH_TOOL_EPISODE' in prompt for prompt in provider.calls)


@pytest.mark.parametrize('policy', ['mode_bandit', 'adaptive_presets'])
@pytest.mark.parametrize('failed', [False, True])
def test_tool_candidate_does_not_train_sampled_parent_mode_or_preset(tmp_path, monkeypatch, policy, failed):
    from copy import deepcopy
    from test_parallel_engine import evaluator
    from test_research_tool_engine import FakeTools, setup_tools
    original = FakeTools.execute

    def execute(self, plan, snap, llm, *, io, candidate_evaluator):
        episode = original(self, plan, snap, llm, io=io)
        source = f'VALUE = {int(plan["action_id"]) + 20}\ndef run_search():\n    return VALUE\n'
        submit_fixture_candidate(episode, source, candidate_evaluator)
        return episode

    monkeypatch.setattr(FakeTools, 'execute', execute)
    options = ({'policy': {'enabled': True, 'mode_bandit': True}}
               if policy == 'mode_bandit' else {'preset_selection': True, 'preset_adaptive': True})
    build, provider, _, _ = setup_tools(tmp_path, monkeypatch, workers=2, **options)
    evaluations, _ = evaluator(monkeypatch, fail_first_candidate=failed)
    engine = build()
    engine.selection.observe = Mock(wraps=engine.selection.observe)
    engine.op_bandit.observe = Mock(wraps=engine.op_bandit.observe)
    engine.controller.observe = Mock(wraps=engine.controller.observe)
    if policy == 'mode_bandit':
        weights_before = deepcopy(engine.controller.mode_bandit.state())
        engine.controller.mode_bandit.observe = Mock(wraps=engine.controller.mode_bandit.observe)
    else:
        weights_before = deepcopy(engine.preset_weights.state())
        engine.preset_weights.update = Mock(wraps=engine.preset_weights.update)
    result = engine.run()
    assert len(evaluations) == 3  # Saved canonical seed plus each submitted program once.
    assert len(provider.calls) == 2 and result['cost_total'] == pytest.approx(.06)
    assert result['B_evals'] == (0 if failed else 2)
    assert len(engine.research_tools.committed) == 2
    assert sum('candidate_binding' in row for row in engine.research_tools.committed) == (0 if failed else 2)
    engine.selection.observe.assert_not_called()
    engine.op_bandit.observe.assert_not_called()
    assert engine.controller.observe.call_count == 2
    assert all(call.kwargs['on_policy'] is False for call in engine.controller.observe.call_args_list)
    if policy == 'mode_bandit':
        engine.controller.mode_bandit.observe.assert_not_called()
        weights_after = engine.controller.mode_bandit.state()
        # Reading a distribution lazily creates empty phase/arm buffers.
        # Those empty buffers are not reward observations or learned weights.
        assert not any(weights_after['events'].values())
        assert {k:v for k,v in weights_after.items() if k != 'events'} == {
            k:v for k,v in weights_before.items() if k != 'events'}
    else:
        engine.preset_weights.update.assert_not_called()
        assert engine.preset_weights.state() == weights_before
