"""Fresh mapping order and sorted WAL decoding render identical research input.

Only inert JSON/source strings are inspected. Lists and source bytes are data;
canonical ordering applies to object keys, never to observations or source.
"""
from copy import deepcopy
import json

import pytest

from sciel.engine.parallel_calls import CallLedger
from sciel.eval.facade import EvalResult
from sciel.methods.models import canonical_bytes
from sciel.research_tools import canonical_diagnostics as diagnostics
from sciel.research_tools.artifact_manifest import artifact_manifest
from sciel.research_tools.controller import _research_results_prompt
from sciel.research_tools.observation_cards import _clip_json, _failure_context, format_context
from sciel.research_tools.research_steps import new_state, state_prompt
from sciel.research_tools.result_projection import success_context, verify_projection
from test_parallel_calls import Capacity
from test_research_tool_controller import prepare
from test_research_tool_result_flow import result
from test_sequential_research_episode import build
from test_tool_input_projection import target_fixture
from test_tool_prior_results_prompt import prior_card


def reordered(value):
    if type(value) is dict:
        return {key: reordered(value[key]) for key in reversed(list(value))}
    if type(value) is list:
        return [reordered(item) for item in value]
    return value


@pytest.mark.parametrize('budget', [8000, 14000, 64000])
@pytest.mark.parametrize('large', [False, True])
def test_whole_planner_prompt_survives_nested_mapping_reorder(tmp_path, budget, large):
    tools, backend, _, costs = build(tmp_path, [], max_prompt_chars=budget,
        experimental_protocol=True, shared_observations=True, background=True)
    snap = target_fixture('failed', huge=large)
    snap.artifacts = {'z': {'stderr': 'same failure', 'detail': [3, 1, 2]},
                      'a': {'context': 'quoted "雪"' * (1000 if large else 1)}}
    snap.research_search_evidence = {'schema': 'sciel-search-evidence/3',
        'global_best_score': .75, 'recent_evaluations': [
            {'evaluation_id': f'evaluation:fixture:{i}',
             'cases': {'z': i, 'a': -i}, 'conditions': {'z': 2, 'a': 1}}
            for i in range(32 if large else 1)]}
    plan = prepare(tools, snap)
    state = new_state()
    state.update(question='Test an exact bound.', experiment='Compare the saved observations.',
                 next_decision='Use the measured result.',
                 decision_if_supported='Continue.', decision_if_refuted='Close.')
    feedback = [{'status': 'canonical_evaluation', 'receipt_id': 'a' * 64,
        'output': {'observations': {'canonical_evaluation': {
            'status': 'evaluated', 'cases': {'z_measurement': 3., 'score': .75, 'a_measurement': 1.},
            'timing_seconds': .1, 'details': {'z': [3, 1, 2], 'a': {'z': 2, 'a': 1}}}}}}]
    before = deepcopy((plan, vars(snap), state, feedback))
    fresh = tools._prompt(plan, snap, [], feedback, 1, research_state=state)
    cold_snap = deepcopy(snap)
    for name, value in vars(cold_snap).items():
        setattr(cold_snap, name, reordered(value))
    cold = tools._prompt(reordered(plan), cold_snap, [], reordered(feedback), 1,
                         research_state=reordered(state))
    assert fresh == cold and len(fresh) <= budget
    assert (plan, vars(snap), state, feedback) == before
    assert not backend.calls and costs == []
    # Byte binding remains strict for an actual changed source or list order.
    ledger = CallLedger(tmp_path / 'ledger', Capacity(), 'local')
    ledger.prepare('research:0:2', 'llm', {'prompt': fresh})
    ledger.prepare('research:0:2', 'llm', {'prompt': cold})
    with pytest.raises(ValueError, match='differs from its durable intent'):
        ledger.prepare('research:0:2', 'llm', {'prompt': cold + '\nchanged evidence'})


def test_diagnostic_ties_and_node_limit_selection_are_mapping_order_invariant():
    # Equal priorities previously inherited insertion order; the node limit
    # makes that affect which paths are inspected, not merely JSON formatting.
    artifacts = {'error': {f'field_{i:05d}': 'same error' for i in range(17000)}}
    raw = diagnostics.capture(EvalResult(False, 0., artifacts=artifacts),
                              source_sha256='b' * 64, evaluation_id='evaluation:fixture')
    cold = reordered(raw)
    sequence = diagnostics.capture(EvalResult(False, 0., artifacts={
        'error': [{'stderr': 'same error'} for _ in range(12)]}),
        source_sha256='b' * 64, evaluation_id='evaluation:fixture')
    entries, _ = diagnostics._entries(sequence)
    assert [entry[0] for entry in entries] == [f'/artifacts/error/{i}/stderr' for i in range(12)]
    for budget in (350, 700, 2000):
        first = diagnostics.project(raw, budget)
        second = diagnostics.project(cold, budget)
        assert first == second
        assert first['uninspected_values'] > 0
        assert first['entries'][0]['path'].startswith('/artifacts/error/field_00000')
        assert diagnostics.feedback({'status': 'eval_failed', 'failure_diagnostics': raw},
            'c' * 64, budget) == diagnostics.feedback(
                {'failure_diagnostics': cold, 'status': 'eval_failed'}, 'c' * 64, budget)


@pytest.mark.parametrize('budget', [80, 350, 700, 1200, 2200])
def test_bounded_state_and_failure_excerpts_keep_identical_bytes(budget):
    state = new_state()
    state['pending_method_trial'] = {'tool_id': 'a' * 64, 'source_program_id': 'p7',
        'source_sha256': 'b' * 64, 'submission_receipt_id': 'c' * 64}
    evidence = [{'status': 'evaluated', 'artifact_id': 'd' * 64,
                 'cases': {'z': 3., 'a': 1.}, 'conditions': {'z': 2, 'a': 1}}]
    assert state_prompt(state, step=0, maximum=3, budget=budget,
        canonical_evaluations=evidence) == state_prompt(reordered(state), step=0,
        maximum=3, budget=budget, canonical_evaluations=reordered(evidence))
    failed = {'reason': 'failure', 'result': {'error': 'same cause'},
              'output': {'z': {'b': [3, 1, 2], 'a': 'quoted "雪"' * 200}, 'a': 1}}
    assert _failure_context(failed, budget) == _failure_context(reordered(failed), budget)
    # _clip_json clips source strings, not mapping syntax; exact escaped bounds
    # and source order must remain unchanged at the boundary.
    source = 'def run(payload):\n    return "雪"\n' * 300
    excerpt = _clip_json(source, budget)
    assert len(canonical_bytes(excerpt)) <= budget
    assert excerpt == source or source.startswith(excerpt.removesuffix(' [excerpt]'))


def test_manifest_projection_and_prior_card_selection_are_order_invariant():
    observations = {'z_checks': [{'z': i, 'a': {'z': i + 2, 'a': i + 1}}
                                for i in range(20)],
        'a_summary': {'z': [3, 1, 2], 'a': .125},
        'comparison_results': [{'arm_id': arm, 'seed': seed, 'cpu_seconds': 1,
            'best_result': value, 'conditions': {'z': 2, 'a': 1}, 'saved_state': 'x' * 6000}
            for seed in (9, 3) for arm, value in (('control', 1), ('changed', 2))],
        'saved_state': {'z': 'x' * 10000, 'a': 'y' * 10000}}
    wrapper = {'output': result(observations), 'result': {'status': 'success'},
               'replay_policy': 'exact_observations', 'reason': None, 'diagnostic': None}
    original = deepcopy(wrapper)
    saved = {'status': 'available', 'reference': {'sha256': 'a' * 64},
             'record': {'result': {'artifacts': observations}}}
    card = prior_card()
    card['observations']['output_excerpt'] = canonical_bytes(wrapper).decode('ascii')
    for budget in (350, 700, 1500, 3000, 10000, len(canonical_bytes(wrapper))):
        projected = success_context(wrapper, budget)
        assert projected == success_context(reordered(wrapper), budget)
        assert len(projected) <= budget
        verify_projection(json.loads(projected), wrapper)
        assert artifact_manifest(saved, '/artifacts', budget) == artifact_manifest(
            reordered(saved), '/artifacts', budget)
        assert _research_results_prompt([card], budget) == _research_results_prompt(
            reordered([card]), budget)
    assert format_context([card]) == format_context(reordered([card]))
    assert json.loads(success_context(wrapper, len(canonical_bytes(wrapper)))) == wrapper
    assert wrapper == original and observations['a_summary']['z'] == [3, 1, 2]
