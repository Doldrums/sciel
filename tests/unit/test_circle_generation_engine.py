"""Real Engine lifecycle, inert structured replies and fake measurements only.

No model API, candidate import, evaluator, optimizer, runtime probe or subprocess.
"""
import copy
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from sciel.config import apply_defaults
from sciel.campaign.budget import CampaignStopped
from sciel.engine import checkpoint
from sciel.engine.llm.client import LLMResponse
from sciel.engine.loop import Engine
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.hypothesis import circle_restart_body as body
from test_circle_intervention_engine import config as old_config, task_copy, rows, digest
from test_repeat_engine_integration import FakeDraws, config as repeat_config
from circle_generation_fixtures import (
    COST, BODIES, response, Model,
    config, build, read_checkpoint,
)


def test_real_engine_one_native_response_to_exact_source_and_measurement(tmp_path):
    engine = build(tmp_path/'run', [response()])
    result = engine.run()
    child = body.source_for(BODIES[0])
    assert result['B_evals'] == 1 and result['reason'] == 'budget'
    assert engine.fixture_evaluated == [body.source_for(), child]
    assert len(engine.llm.calls) == 1 and engine.cost_total == COST
    observation, = engine.research.knowledge.experiments
    assert observation['operator'] == 'circle_generate'
    assert observation['generation_dispatch'] == 'returned'
    assert observation['parent_source_sha256'] == digest(body.source_for())
    assert observation['source_sha256'] == observation['proposed_sha256'] == digest(child)
    assert observation['source_repaired'] is False
    assert observation['controlled_change'] == body.apply(body.source_for(), BODIES[0])['proof']
    assert observation['score'] == .6 and observation['actual_delta'] == pytest.approx(.1)
    assert observation['forecast_verdict'] == 'CONFIRMED'
    assert observation['evaluation']['cases']['sum_radii'] == .6*2.635
    assert engine.research.valid_hypotheses == engine.research.outcomes == 1
    assert engine.research.aux_calls == 0
    prompt = engine.llm.calls[0]['prompt']
    assert 'CURRENT_RESTART_BODY_JSON:' in prompt and '"body":"return None\\n"' in prompt
    assert 'class Incumbent' not in prompt and 'def solve_packing' not in prompt
    assert 'JSON object matching the supplied response schema' in prompt
    assert 'Output the COMPLETE program' not in prompt
    events = rows(engine)
    raw, = [e for e in events if e.get('phase') == 'circle_generation' and e.get('action') == 'request_result']
    assert raw['response'] == response() and raw['response_truncated'] is False
    saved = read_checkpoint(tmp_path/'run')
    assert saved['circle_generation'] == engine.circle_generation.state()
    assert saved['identity']['circle_generation']['identity']['response_format'] == body.response_format()


@pytest.mark.parametrize('text,error,body_error,validhyp', [
    ('not json', 'invalid_json_envelope', False, False),
    (response('return incumbent._params\n', predicted_low='bad'), 'invalid_hypothesis', True, False),
    (response('return incumbent._params\n'), 'invalid_body', True, True),
    (response(predicted_low=2, predicted_high=1), 'invalid_hypothesis', False, False),
])
def test_invalid_forecast_and_body_checks_are_independent_no_fake_source(tmp_path, text, error, body_error, validhyp):
    engine = build(tmp_path/'run', [text])
    result = engine.run()
    assert result['B_evals'] == 0 and result['reason'] == 'backstop'
    assert engine.fixture_evaluated == [body.source_for()] and len(engine.llm.calls) == 1
    assert engine.cost_total == COST
    record, = engine.research.knowledge.experiments
    assert (record['hypothesis'] is not None) is validhyp
    assert record['score'] is record['actual_delta'] is record['evaluation'] is None
    assert record['source_sha256'] is record['proposed_sha256'] is None
    assert record['status'] == ('no_child' if validhyp else 'hypothesis_invalid')
    check, = [e for e in rows(engine) if e.get('action') == 'response_checks']
    assert check['checks']['error'] == error
    assert bool(check['checks']['body_error']) is body_error
    assert engine.research.valid_hypotheses == int(validhyp)


@pytest.mark.parametrize('finish,refusal', [('length',None), (None,None), ('stop','refused')])
def test_incomplete_or_refused_valid_json_is_paid_rejected_once(tmp_path, finish, refusal):
    reply = LLMResponse(response(), 'fixture', COST, 17, 11, finish_reason=finish, refusal=refusal)
    engine = build(tmp_path/'run', [reply])
    assert engine.run()['B_evals'] == 0
    assert engine.fixture_evaluated == [body.source_for()] and engine.cost_total == COST
    record, = engine.research.knowledge.experiments
    assert record['hypothesis'] is not None and record['source_sha256'] is None
    check, = [e for e in rows(engine) if e.get('action') == 'response_checks']
    assert check['checks']['provider_error'] is not None


@pytest.mark.parametrize('target', [body.DEFAULT_BODY, body.DEFAULT_BODY.rstrip('\n'), 'return None   \n'])
def test_noop_body_and_structural_lf_alias_paid_once_without_evaluation(tmp_path, target):
    from sciel.engine.db import normalized_hash
    engine = build(tmp_path/'run', [response(target)])
    assert engine.run()['B_evals'] == 0
    assert engine.circle_generation.repeated_proposals == 1
    record, = engine.research.knowledge.experiments
    assert record['source_sha256'] is record['proposed_sha256'] is record['score'] is None
    assert engine.cost_total == COST
    proposed = body.source_for(target)
    exact = proposed == body.source_for()
    rejection, = [e for e in rows(engine) if e.get('action') == 'plan_rejected']
    assert rejection['match_kind'] == ('current_parent' if exact else 'known_normalized_source')
    assert rejection['proposed_source_sha256'] == digest(proposed)
    assert rejection['matched_fingerprint'] == (digest(proposed) if exact else normalized_hash(proposed))
    assert rejection['quality_measured'] is False
    assert rejection['match_kind'] in record['failure']
    assert rejection['proposed_source_sha256'] in record['failure']
    assert rejection['matched_fingerprint'] in record['failure']
    assert len(record['failure']) <= 664


def test_population_normalized_duplicate_is_known_history_not_active_source(tmp_path, monkeypatch):
    from sciel.engine.db import normalized_hash
    from sciel.hypothesis.rendering import decode_evidence
    from test_repeat_population_engine import Draws, settings
    draws = Draws(monkeypatch)
    draws.values = lambda step, batch: .4  # Exact tie rejects admission.
    alias = BODIES[0].rstrip('\n') + '   \n'
    cfg = settings(budget=2, max_iterations=3, num_islands=1,
        op_fixed='circle_generate', circle_generation={'enabled': True})
    engine = build(tmp_path/'run', [response(), response(alias), response(BODIES[1])], cfg=cfg)
    fake_evaluate = engine._evaluate
    def cascade(task, path, *, record_candidate, python):
        return fake_evaluate(Path(path).read_text())
    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate', cascade)
    # Population restore validates the real result journal as well as scores.
    # Retain Engine logging, replacing only the numerical cascade with our fake.
    engine._evaluate = Engine._evaluate.__get__(engine, Engine)
    result = engine.run()
    assert result['B_evals'] == 2 and len(engine.llm.calls) == 3
    assert len(draws.calls) == 15  # No repeats or ordinary evaluation for alias.
    assert engine.fixture_evaluated == [body.source_for(), body.source_for(BODIES[0]), body.source_for(BODIES[1])]
    assert [p.code for p in engine.db.islands[0].programs.values()] == [body.source_for()]
    rejection, = [e for e in rows(engine) if e.get('action') == 'plan_rejected']
    assert rejection['match_kind'] == 'known_normalized_source'
    assert rejection['proposed_source_sha256'] == digest(body.source_for(alias))
    assert rejection['matched_fingerprint'] == normalized_hash(body.source_for(BODIES[0]))
    assert rejection['proposed_source_sha256'] != digest(body.source_for(BODIES[0]))
    repeated = engine.research.knowledge.experiments[1]
    assert repeated['source_sha256'] is repeated['proposed_sha256'] is None
    assert repeated['score'] is repeated['actual_delta'] is repeated['evaluation'] is None
    assert 'known_normalized_source' in repeated['failure']
    prompt = engine.llm.calls[2]['prompt']
    wire, _ = json.JSONDecoder().raw_decode(prompt.split('RESEARCH_CONTEXT_JSON:\n', 1)[1])
    assert any(repeated['failure'] in f for f in decode_evidence(wire)['failed_hypotheses'])


def test_evicted_source_is_not_revisited_and_resume_matches_continuous(tmp_path):
    cfg = config(budget=2, max_iterations=3, pop_cap=1, grid_bins=1)
    texts = [response(), response(body.DEFAULT_BODY), response(BODIES[1])]
    continuous = build(tmp_path/'whole', texts, cfg=cfg)
    expected = continuous.run()
    first = build(tmp_path/'split', texts[:1], cfg=cfg)
    first._stop_at = 1
    assert first.run()['reason'] == 'paused'
    assert [p.code for p in first.db.islands[0].programs.values()] == [body.source_for(BODIES[0])]
    restored = build(tmp_path/'split', texts[1:], cfg=cfg)
    checkpoint.load(restored, str(tmp_path/'split/checkpoint.json'))
    assert restored.circle_generation.state() == first.circle_generation.state()
    assert restored.run() == expected
    assert restored.fixture_evaluated == [body.source_for(BODIES[1])]
    assert restored.llm.calls == continuous.llm.calls[1:]
    assert restored.research.state() == continuous.research.state()
    assert restored.circle_generation.state() == continuous.circle_generation.state()
    assert restored.cost_total == pytest.approx(3*COST)


@pytest.mark.parametrize('mutation', ['cost','valid_proof_substitution','identity','missing_state','pending'])
def test_checkpoint_tamper_rejected_even_force_before_generation(tmp_path, mutation):
    directory = tmp_path/'run'
    engine = build(directory, [response()])
    engine.run()
    state = read_checkpoint(directory)
    if mutation == 'cost':
        state['circle_generation']['known_cost'] *= 2
    elif mutation == 'valid_proof_substitution':
        state['circle_generation']['seen_proofs'][0] = body.apply(body.source_for(), BODIES[1])['proof']
    elif mutation == 'identity':
        state['identity']['circle_generation']['identity']['context_scope'] = 'other'
    elif mutation == 'missing_state':
        del state['circle_generation']
    else:
        state['circle_generation']['pending'] = {'generation_dispatch':'unknown'}
    tampered = directory/'tampered.json'; tampered.write_text(json.dumps(state))
    restored = build(directory, [])
    try:
        with pytest.raises(checkpoint.IdentityMismatch):
            checkpoint.load(restored, str(tampered), allow_identity_mismatch=True)
        assert not restored.llm.calls and not restored.fixture_evaluated
    finally:
        restored.log.close()


def test_disabling_generation_on_resume_is_not_forceable(tmp_path):
    directory = tmp_path/'run'
    engine = build(directory, [response()]); engine.run()
    cfg = config(op_fixed='full_rewrite', circle_generation={'enabled':False})
    restored = build(directory, [], cfg=cfg)
    try:
        with pytest.raises(checkpoint.IdentityMismatch, match='circle generation'):
            checkpoint.load(restored, str(directory/'checkpoint.json'), allow_identity_mismatch=True)
    finally:
        restored.log.close()


@pytest.mark.parametrize('change', [
    {'research':{'enabled':False}}, {'op_fixed':'full_rewrite'}, {'inflight_cap':2},
    {'inflight_cap':True}, {'portfolio_seed':True}, {'dynamic_islands':True},
    {'idea_island':True}, {'confirm_records':True}, {'circle_intervention':{'enabled':True}},
    {'idea_two_stage':{'enabled':True}}, {'task':'oe/function_minimization'},
    {'circle_generation':{'enabled':1}}, {'circle_generation':{'enabled':True,'typo':True}},
])
def test_wrong_mode_rejected_before_engine_side_effects(change):
    with pytest.raises(ValueError):
        apply_defaults(config(**change))


def test_unrecognized_seed_rejected_before_provider_probe_or_evaluation(tmp_path):
    task = task_copy(tmp_path/'task', initial='def run_packing(): return None\n')
    with pytest.raises(ValueError, match='pinned'):
        build(tmp_path/'run', [], task=task)
    assert not (tmp_path/'run').exists()


def test_absent_and_disabled_component_preserve_default_checkpoint_shape(tmp_path):
    saved = []
    for name, setting in [('absent',None), ('disabled',{'enabled':False})]:
        cfg = config(op_fixed='full_rewrite')
        if setting is None:
            cfg.pop('circle_generation')
        else:
            cfg['circle_generation'] = setting
        engine = build(tmp_path/name, [], cfg=cfg)
        engine._stop_at = 0
        result = engine.run()
        state = read_checkpoint(tmp_path/name)
        assert engine.circle_generation is None and 'circle_generation' not in result
        assert 'circle_generation' not in state and 'circle_generation' not in state['identity']
        saved.append({k:state[k] for k in ('db','research','rng','iteration_no','B_evals','cost_total')})
    assert saved[0] == saved[1]


@pytest.mark.parametrize('after_return', [False, True])
def test_uncertain_or_processing_failure_stops_without_retry_or_charge_loss(tmp_path, monkeypatch, after_return):
    engine = build(tmp_path/'run', [response() if after_return else RuntimeError('private provider detail')])
    if after_return:
        def broken(*a, **k): raise RuntimeError('private persistence detail')
        monkeypatch.setattr(engine.research, 'record_response', broken)
    with pytest.raises(CampaignStopped, match='Circle generation stopped'):
        engine.run()
    assert engine.B_evals == 0
    assert len(engine.llm.calls) == 1 and engine.fixture_evaluated == [body.source_for()]
    assert engine.cost_total == (COST if after_return else 0)
    assert 'private' not in engine.research.stopped_reason
    record, = engine.research.knowledge.experiments
    assert record['generation_dispatch'] == ('returned' if after_return else 'unknown')
    assert record['score'] is record['actual_delta'] is record['source_sha256'] is None
    restored = build(tmp_path/'run', [])
    try:
        with pytest.raises(checkpoint.IdentityMismatch):
            checkpoint.load(restored, str(tmp_path/'run/checkpoint.json'))
    finally:
        restored.log.close()


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), -1., True, None])
def test_unknown_or_invalid_charge_never_becomes_a_successful_free_call(tmp_path, bad):
    reply = LLMResponse(response(), 'fixture', bad, 17, 11, finish_reason='stop')
    engine = build(tmp_path/'run', [reply])
    with pytest.raises(CampaignStopped):
        engine.run()
    assert len(engine.llm.calls) == 1 and engine.B_evals == 0 and engine.cost_total == 0
    receipt, = [e for e in rows(engine) if e.get('action') == 'request_result']
    assert receipt['cost'] is None and receipt['response'] == response()
    stopped, = [e for e in rows(engine) if e.get('phase') == 'circle_generation' and e.get('action') == 'stopped']
    assert stopped['current_call_cost_unknown'] is True
    record, = engine.research.knowledge.experiments
    assert record['generation_cost'] is None and record['generation_dispatch'] == 'returned'


def test_proven_predispatch_stop_preserves_not_dispatched_exposure_and_no_cost(tmp_path):
    from sciel.campaign.budget import RequestNotDispatched
    engine = build(tmp_path/'run', [RequestNotDispatched('inert fixture cap')])
    with pytest.raises(CampaignStopped):
        engine.run()
    record, = engine.research.knowledge.experiments
    assert record['generation_dispatch'] == 'not_dispatched'
    assert record['generation_cost'] is None and engine.cost_total == 0
    assert engine.circle_generation.calls_started == 1 and engine.circle_generation.calls_returned == 0


@pytest.mark.parametrize('raises', [False, True])
def test_evaluator_failure_is_not_scored_as_zero_or_retried(tmp_path, raises):
    engine = build(tmp_path/'run', [response()])
    original = engine._evaluate
    def evaluate(source):
        if source == body.source_for():
            return original(source)
        if raises:
            raise RuntimeError('private evaluator exception')
        return CascadeOutcome(EvalResult(False, 0., {'validity':0.,'sum_radii':0.},
            artifacts={'error':'fixture failure'}, timing=.01), 0., 'smoke', False)
    engine._evaluate = evaluate
    if raises:
        with pytest.raises(CampaignStopped): engine.run()
    else:
        result = engine.run()
        assert result['B_evals'] == 0 and result['reason'] == 'backstop'
    record, = engine.research.knowledge.experiments
    assert record['status'] == 'eval_failed' and record['source_sha256'] == digest(body.source_for(BODIES[0]))
    assert record['score'] is record['actual_delta'] is None
    assert record['forecast_verdict'] == 'INCONCLUSIVE'
    assert engine.cost_total == COST and len(engine.llm.calls) == 1
    if raises:
        assert record['evaluation'] is None and 'private' not in record['failure']
    else:
        assert record['evaluation']['ok'] is False


def test_raw_response_persistence_failure_still_charges_known_return_once(tmp_path, monkeypatch):
    engine = build(tmp_path/'run', [response()])
    original = engine.circle_generation._emit
    def emit(action, **fields):
        if action == 'request_result': raise OSError('fixture durable output failed')
        return original(action, **fields)
    monkeypatch.setattr(engine.circle_generation, '_emit', emit)
    with pytest.raises(CampaignStopped): engine.run()
    assert engine.cost_total == COST and engine.circle_generation.known_cost == COST
    assert engine.circle_generation.calls_returned == 1 and len(engine.llm.calls) == 1
    record, = engine.research.knowledge.experiments
    assert record['generation_dispatch'] == 'returned' and record['generation_cost'] == COST
    assert record['source_sha256'] is None


@pytest.mark.parametrize('mutation', ['task','evaluator','nondefault_seed'])
def test_canonical_semantics_and_exact_default_seed_are_required_before_side_effects(tmp_path, mutation):
    from sciel.eval.registry import load_task
    task = task_copy(tmp_path/'task', initial=body.source_for())
    if mutation == 'task':
        path = Path(task.root)/'task.yaml'
        path.write_text(path.read_text().replace('known_best: 2.635', 'known_best: 9.0'))
    elif mutation == 'evaluator':
        path = Path(task.evaluator_path)
        path.write_text(path.read_text()+'\n# changed evaluator identity\n')
    else:
        Path(task.initial_program_path).write_text(body.source_for(BODIES[0]))
    task = load_task(task.root)
    with pytest.raises(ValueError, match='pinned'):
        build(tmp_path/'run', [], task=task)
    assert not (tmp_path/'run').exists()


@pytest.mark.parametrize('changed', ['engine/research.py', 'hypothesis/mechanism_review.py',
    'hypothesis/restart_motion.py', 'hypothesis/restart_motion_observations.py'])
def test_enabled_identity_pins_parser_and_integration_source_and_refuses_drift(tmp_path, monkeypatch, changed):
    import sciel.engine.circle_generation as generation
    engine = build(tmp_path/'run', [response()])
    expected = engine.circle_generation.identity['integration_sha256']
    assert {'hypothesis/parsing.py','engine/research.py','engine/circle_generation.py',
            'hypothesis/controlled_changes.py','hypothesis/circle_baseline_equivalence.py','hypothesis/mechanism_review.py',
            'engine/llm/client.py','campaign/budget.py',
            'hypothesis/restart_motion.py','hypothesis/restart_motion_observations.py'} <= set(expected)
    monkeypatch.setattr(generation, '_integration_identity', lambda: {**expected, changed:'0'*64})
    with pytest.raises(ValueError, match='identity changed'):
        engine.run()
    assert not engine.llm.calls and engine.fixture_evaluated == [body.source_for()]
    engine.log.close()
