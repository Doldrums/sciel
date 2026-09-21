"""Pair-policy lifecycle through the real Engine, using inert providers/evaluators.

No candidate source, numerical optimizer, model API, subprocess or Docker probe
is executed. Synthetic H15 geometry is delegated to the existing pure fixture.
"""
import copy
import json
from pathlib import Path
import pytest
from sciel.campaign.budget import CampaignStopped
from sciel.engine import checkpoint
from sciel.engine.llm.client import LLMResponse
from sciel.engine.loop import Engine
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.hypothesis import circle_proposal_pair as pair
from sciel.hypothesis import circle_restart_body as body
from sciel.hypothesis.rendering import decode_evidence
from sciel.hypothesis.parsing import parse_hypothesis_value
from test_circle_generation_engine import BODIES, COST, config as single_config, response
from test_circle_intervention_engine import digest, rows, task_copy
from circle_pair_fixtures import (
    POLICY, candidate, paired, config,
    PairModel, build, observations, actions,
    context, assert_one_selected, _rewrite_checkpoint, _novelty_config,
)


def test_invalid_api_first_valid_second_uses_one_charge_and_second_forecast(tmp_path):
    first = candidate('return np.random.choice(26)\n', confidence=.99)
    second = candidate(BODIES[1], confidence=.15, predicted_low=.1, predicted_high=.3)
    raw = paired(first, second)
    engine = build(tmp_path/'run', [raw])
    assert engine.run()['B_evals'] == 1
    record = assert_one_selected(engine, raw, BODIES[1], second, selected_index=1)
    assert record['actual_delta'] == pytest.approx(.2)
    assert record['forecast_verdict'] == 'CONFIRMED'
    invalid = record['proposal_group']['entries'][0]
    assert invalid['disposition'] == 'invalid' and invalid['error'] == 'invalid_body'
    assert invalid['hypothesis'] == parse_hypothesis_value(first['hypothesis']).to_dict()
    assert 'np.random.choice' in invalid['body_error']


def test_baseline_rewrite_first_does_not_prevent_new_second_initializer(tmp_path):
    first = candidate('x = ordinary_params.copy()\nreturn x\n')
    second = candidate(BODIES[1])
    raw = paired(first, second)
    engine = build(tmp_path/'run', [raw])
    assert engine.run()['B_evals'] == 1
    record = assert_one_selected(engine, raw, BODIES[1], second, selected_index=1)
    rejected = record['proposal_group']['entries'][0]
    assert rejected['disposition'] == 'gate_rejected'
    assert 'baseline_initializer' in json.dumps(rejected['gate'])
    assert digest(body.source_for(first['body'])) not in {
        p['child_source_sha256'] for p in engine.circle_generation.seen_proofs}


def test_two_eligible_entries_select_first_even_if_second_confidence_is_higher(tmp_path):
    first = candidate(confidence=.01, predicted_low=-.2, predicted_high=.2)
    second = candidate(BODIES[1], confidence=.99)
    raw = paired(first, second)
    engine = build(tmp_path/'run', [raw])
    assert engine.run()['B_evals'] == 1
    record = assert_one_selected(engine, raw, BODIES[0], first)
    unselected = record['proposal_group']['entries'][1]
    assert unselected['eligible'] is True and unselected['disposition'] == 'eligible_unselected'
    assert unselected['hypothesis']['confidence'] == .99
    assert engine.fixture_evaluated.count(body.source_for(BODIES[1])) == 0


def test_invalid_first_forecast_does_not_consume_identical_second_body(tmp_path):
    first = candidate(predicted_low=1., predicted_high=-1.)
    second = candidate(confidence=.2)
    raw = paired(first, second)
    engine = build(tmp_path/'run', [raw])
    assert engine.run()['B_evals'] == 1
    record = assert_one_selected(engine, raw, BODIES[0], second, selected_index=1)
    assert record['proposal_group']['entries'][0]['error'] == 'invalid_hypothesis'
    assert record['proposal_group']['entries'][0]['body'] == BODIES[0]


def test_no_eligible_pair_is_one_unmeasured_nochild_not_two_experiments(tmp_path):
    raw = paired(candidate('return np.random.choice(26)\n'), candidate('return ordinary_params\n'))
    engine = build(tmp_path/'run', [raw])
    result = engine.run()
    assert result['B_evals'] == 0 and result['iteration_no'] == 1
    assert result['reason'] == 'backstop'
    assert engine.fixture_evaluated == [body.source_for()]
    assert len(engine.llm.calls) == 1 and engine.cost_total == COST
    record, = observations(engine)
    assert record['status'] in ('no_child', 'hypothesis_invalid')
    assert record['score'] is record['actual_delta'] is record['evaluation'] is None
    assert record['source_sha256'] is record['proposed_sha256'] is None
    assert record['forecast_verdict'] == 'INCONCLUSIVE'
    assert engine.research.attempts == engine.research.outcomes == 1
    assert not engine.circle_generation.seen_proofs and not actions(engine, 'source_binding')
    assert not engine.research.queue and engine.research.aux_calls == 0
    group = record['proposal_group']
    assert group['selected_index'] is group['selected_source_sha256'] is group['selected_proof_sha256'] is None
    assert group['assessment_complete'] is True and record['hypothesis'] is None
    assert [e['disposition'] for e in group['entries']] == ['invalid', 'gate_rejected']


@pytest.mark.parametrize('raw', [
    response(),
    '{"candidates":[' + json.dumps(candidate()) + ']}',
    '{"candidates":[' + json.dumps(candidate()) + ',' + json.dumps(candidate(BODIES[1])) + '],"candidates":[{},{}]}',
])
def test_invalid_outer_envelope_is_still_one_paid_attempt_without_partial_salvage(tmp_path, raw):
    engine = build(tmp_path/'run', [raw])
    assert engine.run()['B_evals'] == 0
    assert engine.fixture_evaluated == [body.source_for()]
    assert len(engine.llm.calls) == 1 and engine.cost_total == COST
    record, = observations(engine)
    assert record['score'] is record['actual_delta'] is record['evaluation'] is None
    assert record['source_sha256'] is record['proposed_sha256'] is None
    assert not engine.circle_generation.seen_proofs
    assert engine.research.attempts == engine.research.outcomes == 1
    returned, = actions(engine, 'request_result')
    assert returned['response'] == raw and returned['cost'] == COST
    assert record['proposal_group']['envelope_error'] == 'invalid_json_envelope'
    assert record['proposal_group']['entries'] == []
    assert record['proposal_group']['selected_index'] is None


@pytest.mark.parametrize('resume', [False, True])
def test_unselected_valid_source_remains_available_in_a_later_paid_pair(tmp_path, resume):
    first = paired(candidate(), candidate(BODIES[1]))
    second = paired(candidate(), candidate(BODIES[1], confidence=.25))
    cfg = config(budget=2, max_iterations=2, pop_cap=1, grid_bins=1)
    engine = build(tmp_path/'run', [first, second], cfg=cfg)
    result = engine.run()
    assert result['B_evals'] == 2
    assert engine.fixture_evaluated == [body.source_for(), body.source_for(BODIES[0]), body.source_for(BODIES[1])]
    assert len(engine.llm.calls) == 2 and engine.cost_total == 2*COST
    assert [r['source_sha256'] for r in observations(engine)] == [digest(body.source_for(b)) for b in BODIES]
    assert observations(engine)[1]['parent_source_sha256'] == digest(body.source_for(BODIES[0]))
    assert [r['proposal_group']['selected_index'] for r in observations(engine)] == [0, 1]
    assert observations(engine)[0]['proposal_group']['entries'][1]['disposition'] == 'eligible_unselected'
    historical = context(engine.llm.calls[1]['prompt'])['proposal_groups']['cards'][0]
    assert historical['resubmission_eligibility'] == 'not_assessed'
    assert historical['entries'][1]['eligible_at_original_request'] is True
    assert historical['entries'][1]['disposition_at_original_request'] == 'eligible_unselected'
    assert [p['child_source_sha256'] for p in engine.circle_generation.seen_proofs] == [digest(body.source_for(b)) for b in BODIES]
    if resume:
        split = build(tmp_path/'split', [first], cfg=cfg)
        split._stop_at = 1
        assert split.run()['reason'] == 'paused'
        restored = build(tmp_path/'split', [second], cfg=cfg)
        checkpoint.load(restored, str(tmp_path/'split/checkpoint.json'))
        assert observations(restored) == observations(split)
        assert restored.run() == result
        assert restored.fixture_evaluated == [body.source_for(BODIES[1])]
        assert restored.llm.calls == engine.llm.calls[1:]
        assert restored.research.state() == engine.research.state()
        assert restored.circle_generation.state() == engine.circle_generation.state()


def test_historical_eligibility_and_specific_repeat_rejection_reach_next_pair_prompt(tmp_path):
    alias = BODIES[0].rstrip('\n') + '   \n'
    cfg = config(budget=3, max_iterations=3, pop_cap=1, grid_bins=1)
    replies = [paired(), paired(candidate(alias), candidate(BODIES[1])), paired()]
    engine = build(tmp_path/'run', replies, cfg=cfg)
    engine.run()
    assert len(engine.llm.calls) == 3
    assert engine.fixture_evaluated == [body.source_for(), body.source_for(BODIES[0]), body.source_for(BODIES[1])]
    first, second, third = observations(engine)
    assert first['proposal_group']['entries'][0]['eligible'] is True
    assert first['status'] == 'evaluated' and first['score'] == .6
    assert second['proposal_group']['selected_index'] == 1
    assert second['proposal_group']['entries'][0]['gate']['reason'] == 'known_normalized_source'
    assert second['proposal_group']['entries'][0]['eligible'] is False
    third_context = context(engine.llm.calls[2]['prompt'])
    cards = {c['request_id']: c for c in third_context['proposal_groups']['cards']}
    old = cards[first['id']]['entries'][0]
    rejected = cards[second['id']]['entries'][0]
    assert old['eligible_at_original_request'] is True
    assert rejected['eligible_at_original_request'] is False
    assert rejected['gate_at_original_request']['reason'] == 'known_normalized_source'
    assert rejected['candidate_source_sha256'] == digest(body.source_for(alias))
    assert 'quality unmeasured' in rejected['diagnostic']
    assert cards[first['id']]['resubmission_eligibility'] == 'not_assessed'
    assert cards[second['id']]['resubmission_eligibility'] == 'not_assessed'
    assert 'unmeasured in that original request' in cards[second['id']]['interpretation']
    assert third['score'] is third['actual_delta'] is None
    assert engine.circle_generation.evaluations == 2


@pytest.mark.parametrize('raises', [False, True])
def test_selected_evaluation_failure_never_falls_back_to_second_entry(tmp_path, raises):
    raw = paired()
    engine = build(tmp_path/'run', [raw])
    ordinary = engine._evaluate

    def evaluate(source):
        if source == body.source_for():
            return ordinary(source)
        engine.fixture_evaluated.append(source)
        if raises:
            raise RuntimeError('private fake evaluator failure')
        return CascadeOutcome(EvalResult(False, 0., {'validity': 0., 'sum_radii': 0.},
            artifacts={'error': 'inert failure'}, timing=.01), 0., 'smoke', False)

    engine._evaluate = evaluate
    if raises:
        with pytest.raises(CampaignStopped):
            engine.run()
    else:
        assert engine.run()['B_evals'] == 0
    assert engine.fixture_evaluated == [body.source_for(), body.source_for(BODIES[0])]
    assert len(engine.llm.calls) == 1 and engine.cost_total == COST
    record, = observations(engine)
    assert record['status'] == 'eval_failed'
    assert record['source_sha256'] == digest(body.source_for(BODIES[0]))
    assert record['score'] is record['actual_delta'] is None
    assert record['forecast_verdict'] == 'INCONCLUSIVE'
    assert len(engine.circle_generation.seen_proofs) == 1
    assert record['proposal_group']['selected_index'] == 0
    assert record['proposal_group']['entries'][1]['disposition'] == 'eligible_unselected'
    if raises:
        assert record['evaluation'] is None and 'private fake' not in record['failure']
    else:
        assert record['evaluation']['ok'] is False


@pytest.mark.parametrize('finish,refusal', [('length', None), (None, None), ('stop', 'refused')])
def test_provider_refusal_or_truncation_preserves_raw_pair_but_selects_nothing(tmp_path, finish, refusal):
    raw = paired()
    reply = LLMResponse(raw, 'fixture', COST, 17, 11, finish_reason=finish, refusal=refusal)
    engine = build(tmp_path/'run', [reply])
    assert engine.run()['B_evals'] == 0
    assert engine.fixture_evaluated == [body.source_for()]
    assert len(engine.llm.calls) == 1 and engine.cost_total == COST
    record, = observations(engine)
    assert record['score'] is record['actual_delta'] is record['evaluation'] is None
    assert record['source_sha256'] is record['proposed_sha256'] is None
    returned, = actions(engine, 'request_result')
    assert returned['response'] == raw and returned['finish_reason'] == finish
    assert returned['refusal'] == refusal
    assert not actions(engine, 'source_binding') and not engine.circle_generation.seen_proofs
    group = record['proposal_group']
    assert group['selected_index'] is None and group['provider_error'] is not None
    assert [e['disposition'] for e in group['entries']] == ['provider_rejected'] * 2
    assert all(e['hypothesis'] is not None and e['body'] is not None for e in group['entries'])
    assert record['hypothesis'] is None


@pytest.mark.parametrize('cost', [None, float('nan'), float('inf'), -1., True])
def test_unknown_charge_stops_pair_without_free_evaluation_or_second_call(tmp_path, cost):
    raw = paired()
    reply = LLMResponse(raw, 'fixture', cost, 17, 11, finish_reason='stop')
    engine = build(tmp_path/'run', [reply])
    with pytest.raises(CampaignStopped):
        engine.run()
    assert engine.fixture_evaluated == [body.source_for()]
    assert engine.B_evals == 0 and len(engine.llm.calls) == 1
    assert engine.cost_total == 0
    returned, = actions(engine, 'request_result')
    assert returned['response'] == raw and returned['cost'] is None
    stopped, = actions(engine, 'stopped')
    assert stopped['current_call_cost_unknown'] is True
    record, = observations(engine)
    assert record['generation_dispatch'] == 'returned' and record['generation_cost'] is None
    assert record['score'] is record['actual_delta'] is record['source_sha256'] is None


def test_h15_population_receives_only_selected_child_and_one_comparison(tmp_path, monkeypatch):
    from test_repeat_population_engine import Draws, settings
    draws = Draws(monkeypatch)
    targets = ['# fixture_step = 1\n'+BODIES[0], '# fixture_step = 2\n'+BODIES[1]]
    cfg = settings(budget=1, max_iterations=1, num_islands=1,
        op_fixed='circle_generate', circle_generation={'enabled': True, 'proposal_policy': POLICY})
    raw = paired(candidate(targets[0]), candidate(targets[1], confidence=.99))
    engine = build(tmp_path/'run', [raw], cfg=cfg, scores={targets[0]: .6, targets[1]: .7})
    fake = engine._evaluate

    def cascade(task, path, *, record_candidate, python):
        return fake(Path(path).read_text())

    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate', cascade)
    # Keep real result logging and population source/evidence provenance.
    engine._evaluate = Engine._evaluate.__get__(engine, Engine)
    assert engine.run()['B_evals'] == 1
    assert engine.fixture_evaluated == [body.source_for(), body.source_for(targets[0])]
    assert len(draws.calls) == 9  # Shared bootstrap3 + one comparator/challenger6.
    assert {call['source'] for call in draws.calls} == {body.source_for(), body.source_for(targets[0])}
    comparison, = engine.research.knowledge.comparisons
    record, = observations(engine)
    assert comparison['origin_experiment_id'] == record['id']
    assert comparison['challenger_source_sha256'] == record['source_sha256'] == digest(body.source_for(targets[0]))
    assert comparison['forecast'] == 'not_scored'
    assert len(engine.llm.calls) == 1 and engine.cost_total == COST
    assert engine.research.aux_calls == 0


@pytest.mark.parametrize('reject_first', [False, True])
def test_real_engine_novelty_callback_assesses_both_entries_and_replays_receipts_without_callback(
        tmp_path, monkeypatch, reject_first):
    directory = tmp_path/'run'
    engine = build(directory, [paired()], cfg=_novelty_config())
    inspected = []

    def inspect(source, references):
        refs = list(references)
        inspected.append((source, refs))
        return {'accept': not (reject_first and source == body.source_for(BODIES[0])),
                'similarity': 1. if reject_first and source == body.source_for(BODIES[0]) else .1,
                'matched_index': 0, 'threshold': .98}

    monkeypatch.setattr(engine.novelty_gate, 'inspect', inspect)
    assert engine.run()['B_evals'] == 1
    chosen = int(reject_first)
    assert_one_selected(engine, paired(), BODIES[chosen], candidate(BODIES[chosen]), selected_index=chosen)
    assert inspected == [(body.source_for(b), [body.source_for()]) for b in BODIES]
    group = observations(engine)[0]['proposal_group']
    assert [e['gate']['status'] for e in group['entries']] == (['rejected', 'passed'] if reject_first else ['passed', 'passed'])
    assert len([e for e in rows(engine) if e.get('policy') == 'code_novelty']) == 2
    assessment, = actions(engine, 'proposal_group')
    assert assessment['selection_snapshot']['novelty_enabled'] is True
    assert [e['validation']['novelty']['accept'] for e in assessment['gate_evidence']] == [not reject_first, True]
    restored = build(directory, [], cfg=_novelty_config())

    def forbidden(*_args, **_kwargs):
        raise AssertionError('restoring journal receipts must not dispatch a novelty check')

    monkeypatch.setattr(restored.novelty_gate, 'inspect', forbidden)
    try:
        checkpoint.load(restored, str(directory/'checkpoint.json'))
        assert observations(restored) == observations(engine)
        assert restored.circle_generation.state() == engine.circle_generation.state()
        assert not restored.llm.calls and not restored.fixture_evaluated
    finally:
        restored.log.close()


@pytest.mark.parametrize('fail_index', [0, 1])
def test_unexpected_novelty_exception_stops_whole_paid_pair_without_fallback(tmp_path, monkeypatch, fail_index):
    engine = build(tmp_path/'run', [paired()], cfg=_novelty_config())
    calls = []

    def inspect(source, references):
        list(references)
        calls.append(source)
        if len(calls)-1 == fail_index:
            raise RuntimeError('private novelty infrastructure failure')
        return {'accept': True, 'similarity': .1, 'matched_index': 0, 'threshold': .98}

    monkeypatch.setattr(engine.novelty_gate, 'inspect', inspect)
    with pytest.raises(CampaignStopped):
        engine.run()
    assert calls == [body.source_for(b) for b in BODIES[:fail_index+1]]
    assert engine.fixture_evaluated == [body.source_for()]
    assert len(engine.llm.calls) == 1 and engine.cost_total == COST
    assert engine.circle_generation.evaluations == 0 and not engine.circle_generation.seen_proofs
    assert not actions(engine, 'source_binding')
    record, = observations(engine)
    assert record['status'] == 'generation_stopped'
    assert record['score'] is record['actual_delta'] is record['source_sha256'] is None
    # Assessment failed before the terminal group was bound. Preserve raw and
    # parsed receipts, without inventing a completed or partially selected group.
    assert 'proposal_group' not in record and not actions(engine, 'proposal_group')
    returned, = actions(engine, 'request_result')
    assert returned['response'] == paired() and returned['cost'] == COST
    parsed, = [e for e in rows(engine) if e.get('event') == 'research'
               and e.get('phase') == 'structured_pair']
    assert [e['index'] for e in parsed['entries']] == [0, 1]
    assert all(e['hypothesis'] is not None and e['error'] is e['body_error'] is None
               for e in parsed['entries'])
    assert 'private novelty' not in record['failure']


def test_proposal_policy_config_omission_keeps_single_and_absent_shapes():
    from sciel.config import apply_defaults
    single = apply_defaults(single_config())
    assert single['circle_generation'] == {'enabled': True}
    disabled = apply_defaults(single_config(op_fixed='full_rewrite',
                                            circle_generation={'enabled': False}))
    assert disabled['circle_generation'] == {'enabled': False}
    absent = single_config(op_fixed='full_rewrite')
    del absent['circle_generation']
    assert 'circle_generation' not in apply_defaults(absent)


def test_proposal_policy_config_accepts_only_explicit_enabled_pair_setting():
    from sciel.config import apply_defaults
    configured = apply_defaults(config())
    assert configured['circle_generation'] == {'enabled': True, 'proposal_policy': POLICY}


@pytest.mark.parametrize('setting', [
    {'enabled': True, 'proposal_policy': 'first_eligible_of_2'},
    {'enabled': True, 'proposal_policy': True},
    {'enabled': True, 'proposal_policy': False},
    {'enabled': True, 'proposal_policy': None},
    {'enabled': False, 'proposal_policy': POLICY},
    {'proposal_policy': POLICY},
])
def test_proposal_policy_config_rejects_typo_nonstring_or_disabled_selection(setting):
    from sciel.config import apply_defaults
    with pytest.raises(ValueError, match='proposal_policy'):
        apply_defaults(single_config(circle_generation=setting))
