"""Behavioral checks for circle generation feedback."""
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


def test_repeated_body_with_new_forecast_keeps_diagnostic_through_resume_and_next_prompt(tmp_path):
    from sciel.hypothesis.rendering import decode_evidence
    cfg = config(budget=2, max_iterations=3, pop_cap=1, grid_bins=1)
    revised = response(BODIES[0].rstrip('\n'), predicted_low=-.4, predicted_high=.4)
    texts = [response(), revised, response(BODIES[1])]
    scores = {BODIES[0]: .4}  # First proposal is rejected by ordinary retention.
    continuous = build(tmp_path/'whole', texts, cfg=cfg, scores=scores)
    expected = continuous.run()
    first = build(tmp_path/'split', texts[:2], cfg=cfg, scores=scores)
    original_step = first._step
    def pause_after_duplicate(island):
        original_step(island)
        if first.circle_generation.finished == 2:
            # _stop_at counts evaluations; a duplicate is an attempt only.
            first._stop_at = first.B_evals
    first._step = pause_after_duplicate
    assert first.run()['reason'] == 'paused'
    original, repeated = first.research.knowledge.experiments
    assert repeated['status'] == 'no_child' and repeated['forecast_verdict'] == 'INCONCLUSIVE'
    assert repeated['hypothesis'] == body.decode_response(revised)['hypothesis'].to_dict()
    assert repeated['score'] is repeated['actual_delta'] is repeated['evaluation'] is None
    assert repeated['source_sha256'] is repeated['proposed_sha256'] is None
    assert 'previous_source' in repeated['failure']
    assert digest(body.source_for(BODIES[0])) in repeated['failure']
    assert first.fixture_evaluated == [body.source_for(), body.source_for(BODIES[0])]
    assert first.circle_generation.evaluations == 1 and first.cost_total == 2*COST
    restored = build(tmp_path/'split', texts[2:], cfg=cfg, scores=scores)
    checkpoint.load(restored, str(tmp_path/'split/checkpoint.json'))
    assert restored.research.knowledge.experiments == first.research.knowledge.experiments
    assert restored.run() == expected
    assert restored.fixture_evaluated == [body.source_for(BODIES[1])]
    assert restored.llm.calls == continuous.llm.calls[2:]
    prompt = restored.llm.calls[0]['prompt']
    wire, _ = json.JSONDecoder().raw_decode(prompt.split('RESEARCH_CONTEXT_JSON:\n', 1)[1])
    context = decode_evidence(wire)
    failure = next(f for f in context['failed_hypotheses'] if repeated['id'] in f)
    assert repeated['failure'] in failure and 'quality not measured' in failure
    assert 'Changing only its forecast' in prompt
    assert context['parent_score'] == .5
    assert restored.research.knowledge.experiments[0] == original
    assert restored.research.state() == continuous.research.state()
    assert restored.circle_generation.state() == continuous.circle_generation.state()


def test_repeat_results_reach_next_generation_without_rewriting_single_forecast(tmp_path, monkeypatch):
    fake = FakeDraws(monkeypatch)
    # Fake draw helper recognizes inert comments; body strings are parsed as
    # data only. Its arithmetic certificates are ordinary strict geometry.
    targets = ['# fixture_step = 1\n'+BODIES[0], '# fixture_step = 2\n'+BODIES[1]]
    scores = {targets[0]:.6, targets[1]:.7}
    cfg = config(budget=2, max_iterations=2, pop_cap=1, grid_bins=1,
                 repeat_promotion=repeat_config()['repeat_promotion'])
    directory = tmp_path/'repeat'
    first = build(directory, [response(targets[0])], cfg=cfg, scores=scores)
    first._stop_at = 1
    assert first.run()['reason'] == 'paused'
    assert len(fake.calls) == 9
    first_record, = copy.deepcopy(first.research.knowledge.experiments)
    first_group, = copy.deepcopy(first.research.knowledge.comparisons)
    assert first_group['derived']['outcome'] == 'improved'
    resumed = build(directory, [response(targets[1])], cfg=cfg, scores=scores)
    checkpoint.load(resumed, str(directory/'checkpoint.json'))
    assert resumed._sync_repeat_comparisons() == 0
    result = resumed.run()
    assert result['B_evals'] == 2 and len(fake.calls) == 15
    assert len(resumed.llm.calls) == 1 and first_group['id'] in resumed.llm.calls[0]['prompt']
    assert resumed.research.aux_calls == 0 and resumed.research.attempts == resumed.research.outcomes == 2
    assert resumed.research.knowledge.experiments[0] == first_record
    groups = resumed.research.knowledge.comparisons
    assert [g['derived']['outcome'] for g in groups] == ['improved','regressed']
    assert all(g['forecast'] == 'not_scored' for g in groups)
    latest = resumed.research.knowledge.experiments[1]
    assert latest['actual_delta'] == pytest.approx(.1) and latest['forecast_verdict'] == 'CONFIRMED'
    assert groups[1]['origin_experiment_id'] == latest['id']
    assert groups[1]['comparator_source_sha256'] == digest(body.source_for(targets[0]))


@pytest.mark.parametrize('invalid_body,expected_diagnostic', [
    ('return incumbent._params\n', 'array attribute'),
    ('    x = np.copy(best_params)\n    return x\n', 'unexpected indent'),
])
def test_invalid_body_diagnostic_reaches_next_distinct_generation_with_forecast_intact(tmp_path,invalid_body,expected_diagnostic):
    from sciel.hypothesis.rendering import decode_evidence
    bad = response(invalid_body)
    engine = build(tmp_path/'run', [bad,response()], cfg=config(max_iterations=2))
    result = engine.run()
    assert result['B_evals']==1 and result['iteration_no']==2
    assert len(engine.llm.calls)==2 and engine.cost_total==2*COST
    first, second = engine.research.knowledge.experiments
    checks = body.decode_response(bad)
    assert first['status']=='no_child' and first['hypothesis']==checks['hypothesis'].to_dict()
    assert first['score'] is first['actual_delta'] is first['source_sha256'] is first['proposed_sha256'] is None
    assert first['forecast_verdict']=='INCONCLUSIVE'
    assert 'invalid_body' in first['failure']
    assert expected_diagnostic in first['failure']
    if expected_diagnostic=='unexpected indent':
        assert 'body line 1, column 4' in first['failure']
        assert 'leading whitespace characters: 4' in first['failure']
        assert 'np.copy(best_params)' in first['failure']
    else:
        assert checks['body_error'] in first['failure']
    assert len(first['failure']) <= 664
    prompt = engine.llm.calls[1]['prompt']
    wire, _ = json.JSONDecoder().raw_decode(prompt.split('RESEARCH_CONTEXT_JSON:\n',1)[1])
    context = decode_evidence(wire)
    failure, = context['failed_hypotheses']
    assert first['id'] in failure and first['failure'] in failure
    assert 'quality not measured' in failure and 'source_sha256=None' in failure
    assert context['parent_score']==first['parent_score']==second['parent_score']==.5
    assert engine.fixture_evaluated == [body.source_for(),body.source_for(BODIES[0])]
    finishes=[e for e in rows(engine) if e.get('phase')=='circle_generation' and e.get('action')=='finish']
    assert [e['status'] for e in finishes]==['generation_invalid_response','evaluated']


@pytest.mark.parametrize('arm', ['off', 'on'])
def test_archived_baseline_restatement_is_paid_once_without_child_or_fake_quality(tmp_path, arm):
    # Authored representatives of the two equivalent initializer surfaces.
    targets = {'off': 'return ordinary_params.copy()\n',
               'on': 'x = ordinary_params.copy()\nreturn x\n'}
    text = response(targets[arm])
    target = body.decode_response(text)['proposal'].body
    assert body.source_for(target) != body.source_for()
    engine = build(tmp_path/'run', [text])
    assert engine.run()['B_evals'] == 0
    assert engine.fixture_evaluated == [body.source_for()]
    assert len(engine.llm.calls) == 1 and engine.cost_total == COST
    assert engine.circle_generation.repeated_proposals == 1
    assert engine.circle_generation.evaluations == len(engine.circle_generation.seen_proofs) == 0
    record, = engine.research.knowledge.experiments
    assert record['status'] == 'no_child' and record['forecast_verdict'] == 'INCONCLUSIVE'
    assert record['hypothesis'] == body.decode_response(text)['hypothesis'].to_dict()
    assert record['source_sha256'] is record['proposed_sha256'] is record['score'] is record['actual_delta'] is record['evaluation'] is None
    rejection, = [e for e in rows(engine) if e.get('action') == 'plan_rejected']
    assert rejection['reason'] == 'baseline_initializer_equivalence'
    assert rejection['match_kind'] == 'baseline_initializer'
    assert rejection['proposed_source_sha256'] == digest(body.source_for(target))
    assert rejection['matched_fingerprint'] == digest(body.source_for())
    proof = rejection['initializer_equivalence_proof']
    assert proof['sha256'] == rejection['proof_sha256']
    assert proof['parent_source_sha256'] == digest(body.source_for())
    assert all(s['elements_proven_equal'] == 78 for s in proof['states'])
    assert 'baseline_initializer' in record['failure'] and proof['sha256'] in record['failure']
    assert 'quality unmeasured' in record['failure'] and len(record['failure']) <= 664
    assert not [e for e in rows(engine) if e.get('action') == 'source_binding']
    assert not engine.research.queue and engine.research.aux_calls == 0
    restored = build(tmp_path/'run', [])
    checkpoint.load(restored, str(tmp_path/'run/checkpoint.json'))
    assert restored.circle_generation.state() == engine.circle_generation.state()
    assert restored.research.knowledge.experiments == [record]
    restored.log.close()


def test_baseline_restatement_diagnostic_survives_resume_and_reaches_next_prompt(tmp_path):
    from sciel.hypothesis.rendering import decode_evidence
    cfg = config(budget=1, max_iterations=2, pop_cap=1, grid_bins=1)
    texts = [response('return ordinary_params\n'), response()]
    whole = build(tmp_path/'whole', texts, cfg=cfg)
    expected = whole.run()
    split = build(tmp_path/'split', texts[:1], cfg=cfg)
    original_step = split._step
    def pause_after_rejection(island):
        original_step(island)
        if split.circle_generation.finished == 1:
            split._stop_at = split.B_evals
    split._step = pause_after_rejection
    assert split.run()['reason'] == 'paused'
    rejected, = split.research.knowledge.experiments
    restored = build(tmp_path/'split', texts[1:], cfg=cfg)
    checkpoint.load(restored, str(tmp_path/'split/checkpoint.json'))
    assert restored.run() == expected
    assert restored.fixture_evaluated == [body.source_for(BODIES[0])]
    prompt = restored.llm.calls[0]['prompt']
    wire, _ = json.JSONDecoder().raw_decode(prompt.split('RESEARCH_CONTEXT_JSON:\n', 1)[1])
    failure, = decode_evidence(wire)['failed_hypotheses']
    assert rejected['failure'] in failure and 'baseline_initializer' in failure
    assert 'quality not measured' in failure
    assert 'CURRENT parent, not a rejected historical child' in prompt
    assert restored.research.state() == whole.research.state()
    assert restored.circle_generation.state() == whole.circle_generation.state()


def test_baseline_vector_is_allowed_as_new_change_from_a_nondefault_parent(tmp_path):
    target = 'x = ordinary_params.copy()\nreturn x\n'
    engine = build(tmp_path/'run', [response(), response(target)],
        cfg=config(budget=2, max_iterations=2, pop_cap=1, grid_bins=1), scores={target: .7})
    assert engine.run()['B_evals'] == 2
    assert engine.fixture_evaluated == [body.source_for(), body.source_for(BODIES[0]), body.source_for(target)]
    first, second = engine.research.knowledge.experiments
    assert second['parent_source_sha256'] == first['source_sha256']
    assert not [e for e in rows(engine) if e.get('action') == 'plan_rejected']


def test_archived_forbidden_rng_feedback_survives_resume_and_reaches_actual_next_prompt(tmp_path):
    from sciel.hypothesis.rendering import decode_evidence

    # Small authored case: reject new RNG while preserving the exact paid reply.
    raw = response('return np.random.choice(ordinary_params)\n')
    decoded = body.decode_response(raw)
    diagnostic = decoded['body_error']
    assert 'np.random.choice' in diagnostic and 'body line 1, column 8' in diagnostic
    assert 'use the supplied uniform_noise' in diagnostic
    cfg = config(budget=1, max_iterations=2, pop_cap=1, grid_bins=1)
    texts = [raw, response()]
    whole = build(tmp_path/'whole', texts, cfg=cfg)
    expected = whole.run()
    split = build(tmp_path/'split', texts[:1], cfg=cfg)
    original_step = split._step

    def pause_after_invalid_body(island):
        original_step(island)
        if split.circle_generation.finished == 1:
            split._stop_at = split.B_evals

    split._step = pause_after_invalid_body
    assert split.run()['reason'] == 'paused'
    rejected, = split.research.knowledge.experiments
    assert rejected['hypothesis'] == decoded['hypothesis'].to_dict()
    assert rejected['forecast_verdict'] == 'INCONCLUSIVE'
    assert rejected['status'] == 'no_child'
    assert all(rejected[key] is None for key in ('score', 'actual_delta', 'evaluation', 'source_sha256', 'proposed_sha256'))
    assert diagnostic in rejected['failure'] and len(rejected['failure']) <= 664
    assert split.fixture_evaluated == [body.source_for()]
    assert split.cost_total == COST and split.circle_generation.evaluations == 0
    assert not split.circle_generation.seen_proofs
    receipt, = [e for e in rows(split) if e.get('action') == 'request_result']
    assert receipt['response'] == raw

    restored = build(tmp_path/'split', texts[1:], cfg=cfg)
    checkpoint.load(restored, str(tmp_path/'split/checkpoint.json'))
    assert restored.research.knowledge.experiments == [rejected]
    assert restored.run() == expected
    assert restored.fixture_evaluated == [body.source_for(BODIES[0])]
    assert restored.llm.calls == whole.llm.calls[1:]
    prompt = restored.llm.calls[0]['prompt']
    wire, _ = json.JSONDecoder().raw_decode(prompt.split('RESEARCH_CONTEXT_JSON:\n', 1)[1])
    failure, = decode_evidence(wire)['failed_hypotheses']
    assert diagnostic in failure and 'quality not measured' in failure
    assert 'Additional RNG is unavailable' in failure
    assert restored.cost_total == 2 * COST and restored.research.attempts == restored.research.outcomes == 2
    assert restored.research.valid_hypotheses == 2 and restored.research.aux_calls == 0
    assert restored.circle_generation.evaluations == len(restored.circle_generation.seen_proofs) == 1
    assert restored.research.state() == whole.research.state()
    assert restored.circle_generation.state() == whole.circle_generation.state()


def test_unknown_dataflow_is_not_rejected_as_equivalent(tmp_path):
    target = 'for i in range(1):\n    pass\nreturn ordinary_params\n'
    engine = build(tmp_path/'run', [response(target)], scores={target: .6})
    assert engine.run()['B_evals'] == 1
    assert engine.fixture_evaluated == [body.source_for(), body.source_for(target)]
    assert engine.circle_generation.repeated_proposals == 0


def test_unicode_access_keeps_terminal_attribute_and_location_in_next_kb_prompt(tmp_path):
    from sciel.hypothesis.rendering import decode_evidence
    name = 'α' * 110
    raw = response(f'{name} = ordinary_params\nreturn {name}.forbidden\n')
    engine = build(tmp_path/'run', [raw, response()], cfg=config(max_iterations=2))
    assert engine.run()['B_evals'] == 1
    rejected, _ = engine.research.knowledge.experiments
    diagnostic = body.decode_response(raw)['body_error']
    assert '.forbidden' in diagnostic and 'body line 2, column 8' in diagnostic
    assert len(rejected['failure']) <= 664
    assert '.forbidden' in rejected['failure'] and 'body line 2, column 8' in rejected['failure']
    wire, _ = json.JSONDecoder().raw_decode(engine.llm.calls[1]['prompt'].split('RESEARCH_CONTEXT_JSON:\n', 1)[1])
    failure, = decode_evidence(wire)['failed_hypotheses']
    assert rejected['failure'] in failure and 'quality not measured' in failure
    assert engine.cost_total == 2 * COST
    assert rejected['score'] is rejected['actual_delta'] is rejected['evaluation'] is None


def test_population_baseline_restatement_does_not_dispatch_a_child_comparison(tmp_path, monkeypatch):
    from test_repeat_population_engine import Draws, settings
    draws = Draws(monkeypatch)
    cfg = settings(budget=1, max_iterations=1, num_islands=1,
        op_fixed='circle_generate', circle_generation={'enabled': True})
    engine = build(tmp_path/'run', [response('return ordinary_params\n')], cfg=cfg)
    fake_evaluate = engine._evaluate
    def cascade(task, path, *, record_candidate, python):
        return fake_evaluate(Path(path).read_text())
    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate', cascade)
    # Keep real baseline-result journaling required by the population route.
    engine._evaluate = Engine._evaluate.__get__(engine, Engine)
    assert engine.run()['B_evals'] == 0
    assert len(draws.calls) == 3  # Initial control bootstrap only.
    assert engine.fixture_evaluated == [body.source_for()]
    assert not engine.research.knowledge.comparisons
    assert engine.circle_generation.repeated_proposals == 1
