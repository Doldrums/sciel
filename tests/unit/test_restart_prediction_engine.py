"""Opt-in forecasts traverse real Engine control flow with inert model/draws only."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from sciel.config import apply_defaults
from sciel.engine import checkpoint
from sciel.engine.llm.client import LLMResponse
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.hypothesis import circle_optimizer_traced_body as builder
from sciel.hypothesis import restart_predictions as predictions
from sciel.hypothesis.prompts import restart_block
from sciel.hypothesis.rendering import decode_evidence
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
from sciel.knowledge_base.base import _snapshot
from sciel.knowledge_base.restart_prediction_context import PREFIX, block as prediction_block
from sciel.knowledge_base.restart_context import PREFIX as RESTART_PREFIX
from sciel.knowledge_base.memory_references import SCHEMA as SHOWN_SCHEMA, RULES as SHOWN_RULES
from test_circle_generation_engine import response
from test_circle_intervention_engine import task_copy
from test_optimizer_traced_engine import OutcomeDraws
from test_repeat_engine_integration import journal
from test_repeat_reflection_lifecycle import ReflectingModel
import test_traced_circle_engine as flow

_CONFIG = flow.config


def config(route='repeat_promotion', pair_mode=False):
    value = _CONFIG(route, pair_mode)
    value['circle_generation']['execution_profile'] = builder.EXECUTION_PROFILE
    value['research']['restart_predictions'] = True
    return value


def spec():
    return {'schema': predictions.SCHEMA,
            'expected': dict(zip(predictions.NAMES, (True, False, True)))}


class Model(ReflectingModel):
    missing = 'none'
    alternative_eligible = False

    def __init__(self, draws, pair_mode, start=0):
        super().__init__(draws, start)
        self.pair_mode = pair_mode

    def generate(self, prompt, model, temperature=None, *, response_format=None):
        if response_format is None:
            return super().generate(prompt, model, temperature)
        schema = response_format['json_schema']['schema']
        if self.pair_mode:
            schema = schema['properties']['candidates']['items']
        hypothesis_schema = schema['properties']['hypothesis']
        assert set(hypothesis_schema['required']) == {
            'hypothesis', 'predicted_low', 'predicted_high', 'reasoning', 'confidence', 'restart_predictions'}
        assert hypothesis_schema['properties']['restart_predictions'] == predictions.json_schema()
        assert 'Declare expected values before evaluation' in prompt
        index = self.mutation_count
        assert index < 2, 'unexpected generation replay'
        self.mutation_count += 1
        selected = json.loads(response(flow.BODIES[index]))
        selected['hypothesis']['restart_predictions'] = spec()
        alternative = json.loads(response(flow.BODIES[1] if self.alternative_eligible else 'return None\n'))
        alternative['hypothesis']['restart_predictions'] = spec()
        if self.missing in ('first', 'both'):
            del selected['hypothesis']['restart_predictions']
        if self.missing == 'both':
            del alternative['hypothesis']['restart_predictions']
        text = json.dumps({'candidates': [selected, alternative]} if self.pair_mode else selected)
        self.calls.append(dict(kind='mutation', prompt=prompt, text=text,
                               response_format=response_format))
        return LLMResponse(text, model, .001, 10, 10, finish_reason='stop')


@pytest.fixture
def profile(monkeypatch):
    monkeypatch.setattr(flow, 'traced', builder)
    monkeypatch.setattr(flow, 'Draws', OutcomeDraws)
    monkeypatch.setattr(flow, 'Model', Model)
    monkeypatch.setattr(flow, 'config', config)


def construct(task, folder, draws, route, pair_mode, monkeypatch, *, start=0, failed=False):
    engine, model = flow.construct(task, folder, draws, route, pair_mode, monkeypatch, start=start)
    from sciel.engine import loop
    evaluate = loop.cascade_evaluate
    engine.prediction_commits_before_evaluation = []

    def checked(_task, path, **kwargs):
        source = Path(path).read_text()
        if builder.body_of(source) != builder.DEFAULT_BODY:
            pending, = engine.research.pending.values()
            binding = pending['restart_prediction_binding']
            assert binding['spec_sha256'] == canonical_sha(spec())
            assert binding['source_sha256'] == binding['proposed_sha256'] == hashlib.sha256(source.encode()).hexdigest()
            generated = [call for call in model.calls if call['kind'] == 'mutation'][-1]
            assert binding['response_sha256'] == hashlib.sha256(generated['text'].encode()).hexdigest()
            # This event is durable before the evaluator is entered.
            events = journal(folder)
            committed = next(e for e in reversed(events) if e.get('phase') == 'hypothesis')
            assert committed['evidence']['restart_prediction_binding'] == binding
            engine.prediction_commits_before_evaluation.append(copy.deepcopy(binding))
            if failed:
                engine.fixture_evaluated.append(source)
                result = EvalResult(False, 0., {}, timing=.01, artifacts={'error': 'inert evaluation failure'})
                return CascadeOutcome(result, 0., 'full', False)
        return evaluate(_task, path, **kwargs)

    monkeypatch.setattr('sciel.engine.loop.cascade_evaluate', checked)
    return engine, model


def shown(prompt):
    return json.JSONDecoder().raw_decode(prompt.split(PREFIX, 1)[1].lstrip())[0]


@pytest.mark.parametrize('route', ['repeat_promotion', 'repeat_population'])
@pytest.mark.parametrize('pair_mode', [False, True])
def test_forecast_binding_repeat_reflection_and_restored_generation(
        tmp_path, monkeypatch, profile, route, pair_mode):
    draws = OutcomeDraws(monkeypatch)
    task = task_copy(tmp_path/'task', initial=builder.source_for())
    folder = tmp_path/'run'
    first, model = construct(task, folder, draws, route, pair_mode, monkeypatch)
    first._stop_at = 1
    first.run()
    assert len(first.prediction_commits_before_evaluation) == 1
    assert [call['kind'] for call in model.calls] == ['mutation', 'conclusion', 'reflection']
    assert PREFIX not in model.calls[0]['prompt'] and PREFIX not in model.calls[1]['prompt']
    reflected = shown(model.calls[2]['prompt'])
    group, = first.research.knowledge.comparisons
    card, = reflected['cards']
    assert card['group_id'] == group['id'] and card['origin_id'] == 'r000001a1'
    assert card['expected'] == [True, False, True]
    assert [row[0] for row in card['rows']] == [pair['seed'] for pair in group['pairs']]
    assert all(row[3] == [None]*3 and row[4] == ['INCONCLUSIVE']*3 for row in card['rows'])
    assert model.calls[2]['draw_count'] == 9
    assert len(draws.calls) == 9 and first.research.aux_calls == 2
    record, = first.research.knowledge.experiments
    assert record['hypothesis']['restart_predictions'] == spec()
    assert record['restart_prediction_binding'] == first.prediction_commits_before_evaluation[0]
    if pair_mode:
        assert record['proposal_group']['entries'][1]['hypothesis']['restart_predictions'] == spec()
        assert record['proposal_group']['selected_index'] == 0
    prior_group = canonical_json(group)
    original_state = canonical_json(first.research.state())
    restored, following = construct(task, folder, draws, route, pair_mode, monkeypatch, start=1)
    checkpoint.load(restored, str(folder/'checkpoint.json'))
    assert canonical_json(restored.research.state()) == original_state
    assert not following.calls and len(draws.calls) == 9
    restored.run()
    generated = next(call for call in following.calls if call['kind'] == 'mutation')
    context = shown(generated['prompt'])
    assert context['cards'] == reflected['cards']
    motion = flow.parse(generated['prompt'])
    # The unchanged 12000-character motion budget admits five whole synthetic
    # receipt cards, with the sixth explicitly omitted even before forecasts.
    assert motion['presented_count'] == 5 and motion['omitted_count'] == 1
    assert motion['cards'] == flow.parse(model.calls[2]['prompt'])['cards']
    assert motion['selected_count'] == motion['presented_count'] + motion['omitted_count'] == 6
    assert motion['omissions'][0]['reason'] == 'memory_char_limit'
    assert {card['role'] for card in motion['cards']} == {'comparator', 'challenger'}
    raw = json.JSONDecoder().raw_decode(generated['prompt'].split('RESEARCH_CONTEXT_JSON:\n', 1)[1].lstrip())[0]
    visible = decode_evidence(raw)
    attempt = next(event for event in journal(folder)
                   if event.get('phase') == 'attempt' and event.get('experiment_id') == 'r000002a1')
    restart_wire = json.JSONDecoder().raw_decode(generated['prompt'].split(RESTART_PREFIX, 1)[1].lstrip())[0]
    memory_chars = attempt['lesson_presentation']['rendered_chars']
    memory_chars += sum(len(canonical_json(visible[key])) for key in
                        ('paired_comparisons', 'comparison_facts', 'proposal_groups', 'conclusion_feedback') if key in visible)
    if visible.get('paired_comparisons', {}).get('schema') == SHOWN_SCHEMA:
        memory_chars += len(SHOWN_RULES)
    memory_chars += len(restart_block(restart_wire)) + len(prediction_block(context))
    assert memory_chars <= 32000
    if pair_mode:
        history = visible['proposal_groups']
        assert history['cards'] == [] and history['omitted_ids'] == ['r000001a1']
        assert history['omitted_count'] == history['retained_count'] == 1
    assert canonical_json(restored.research.knowledge.comparisons[0]) == prior_group
    assert len(draws.calls) == 15 and len(restored.prediction_commits_before_evaluation) == 1
    assert restored.research.attempts == restored.research.outcomes == restored.circle_generation.evaluations == 2
    assert restored.research.aux_calls == 4
    lesson = next(lesson for lesson in restored.research.knowledge.lessons
                  if lesson['source_support_ids'] == [group['id']])
    assert lesson['exposure_ids'] == ['r000002a1']
    assert all(not ident.startswith('prediction:') for ident in restored.research.knowledge._evidence())


@pytest.mark.parametrize('change', ['profile1', 'untraced', 'disabled_circle', 'disabled_research',
                                  'no_repeat', 'integer', 'null'])
def test_enabled_forecasts_require_explicit_supported_profile_before_side_effects(change):
    value = config()
    if change == 'profile1':
        value['circle_generation']['execution_profile'] = 'paired_restart_trace/1'
    elif change == 'untraced':
        del value['circle_generation']['execution_profile']
    elif change == 'disabled_circle':
        value['circle_generation']['enabled'] = False
    elif change == 'disabled_research':
        value['research']['enabled'] = False
    elif change == 'no_repeat':
        del value['repeat_promotion']
    else:
        value['research']['restart_predictions'] = 1 if change == 'integer' else None
    with pytest.raises(ValueError):
        apply_defaults(value)


@pytest.mark.parametrize('pair_mode', [False, True])
def test_missing_required_predictions_never_reach_candidate_evaluation(
        tmp_path, monkeypatch, profile, pair_mode):
    monkeypatch.setattr(Model, 'missing', 'both')
    draws = OutcomeDraws(monkeypatch)
    task = task_copy(tmp_path/'task', initial=builder.source_for())
    engine, model = construct(task, tmp_path/'run', draws, 'repeat_promotion', pair_mode, monkeypatch)
    engine.cfg['max_iterations'] = 1
    engine._stop_at = 1
    engine.run()
    assert len(engine.fixture_evaluated) == 1 and not engine.prediction_commits_before_evaluation
    assert len(draws.calls) == 3 and engine.circle_generation.evaluations == 0
    record, = engine.research.knowledge.experiments
    assert record['score'] is record['actual_delta'] is None
    assert 'restart_prediction_binding' not in record
    if pair_mode:
        assert record['proposal_group']['selected_index'] is None
        assert record['proposal_group']['entries'][0]['gate']['reason'] == 'missing_restart_predictions'
    assert len([call for call in model.calls if call['kind'] == 'mutation']) == 1


def test_selected_evaluation_failure_never_falls_back_to_other_predicted_alternative(
        tmp_path, monkeypatch, profile):
    monkeypatch.setattr(Model, 'alternative_eligible', True)
    draws = OutcomeDraws(monkeypatch)
    task = task_copy(tmp_path/'task', initial=builder.source_for())
    engine, model = construct(task, tmp_path/'run', draws, 'repeat_promotion', True, monkeypatch, failed=True)
    engine.cfg['max_iterations'] = 1
    engine._stop_at = 1
    engine.run()
    assert len(engine.fixture_evaluated) == 2 and len(engine.prediction_commits_before_evaluation) == 1
    assert len(draws.calls) == 3
    record, = engine.research.knowledge.experiments
    assert record['status'] == 'eval_failed' and record['score'] is record['actual_delta'] is None
    assert record['proposal_group']['selected_index'] == 0
    assert record['proposal_group']['entries'][1]['disposition'] == 'eligible_unselected'
    assert len([call for call in model.calls if call['kind'] == 'mutation']) == 1


def test_missing_first_forecast_selects_only_the_valid_predicted_alternative(
        tmp_path, monkeypatch, profile):
    monkeypatch.setattr(Model, 'missing', 'first')
    monkeypatch.setattr(Model, 'alternative_eligible', True)
    draws = OutcomeDraws(monkeypatch)
    task = task_copy(tmp_path/'task', initial=builder.source_for())
    engine, model = construct(task, tmp_path/'run', draws, 'repeat_promotion', True, monkeypatch)
    engine.cfg['max_iterations'] = 1
    engine._stop_at = 1
    engine.run()
    record, = engine.research.knowledge.experiments
    group = record['proposal_group']
    assert group['selected_index'] == 1
    assert group['entries'][0]['gate']['reason'] == 'missing_restart_predictions'
    assert group['entries'][0]['hypothesis'].get('restart_predictions') is None
    assert group['entries'][1]['hypothesis']['restart_predictions'] == spec()
    assert engine.fixture_evaluated == [builder.source_for(), builder.source_for(flow.BODIES[1])]
    assert len(engine.prediction_commits_before_evaluation) == 1
    assert len([call for call in model.calls if call['kind'] == 'mutation']) == 1
    assert len(draws.calls) == 9 and engine.research.outcomes == 1


@pytest.mark.parametrize('damage', ['response', 'spec', 'source', 'missing'])
def test_prediction_checkpoint_tamper_rejected_before_aux_or_runtime(
        tmp_path, monkeypatch, profile, damage):
    draws = OutcomeDraws(monkeypatch)
    task = task_copy(tmp_path/'task', initial=builder.source_for())
    folder = tmp_path/'run'
    first, _ = construct(task, folder, draws, 'repeat_promotion', False, monkeypatch)
    first._stop_at = 1
    first.run()
    state = json.loads((folder/'checkpoint.json').read_text())
    knowledge = state['research']['knowledge']
    record = knowledge['experiments'][0]
    if damage == 'response':
        # Recompute all checkpoint-local copies/digests to exercise the original
        # raw-response journal join, not just KB checksum verification.
        def replace(value):
            if isinstance(value, dict):
                binding = value.get('restart_prediction_binding')
                if binding is not None:
                    binding['response_sha256'] = 'f'*64
                for child in value.values():
                    replace(child)
            elif isinstance(value, list):
                for child in value:
                    replace(child)
        replace(knowledge)
        knowledge['seen_experiments'][record['id']] = canonical_sha(_snapshot(record))
    elif damage == 'spec':
        record['hypothesis']['restart_predictions']['expected']['result_changed'] = True
    elif damage == 'source':
        record['restart_prediction_binding']['source_sha256'] = 'f'*64
    else:
        del record['restart_prediction_binding']
    path = folder/'tampered.json'
    path.write_text(canonical_json(state))
    resumed, model = construct(task, folder, draws, 'repeat_promotion', False, monkeypatch, start=1)
    before = canonical_json(resumed.research.state())
    try:
        with pytest.raises(checkpoint.IdentityMismatch) as error:
            checkpoint.load(resumed, str(path), allow_identity_mismatch=True)
        if damage == 'response':
            assert 'raw request journal' in str(error.value)
        assert not model.calls and len(draws.calls) == 9 and not resumed.fixture_evaluated
        assert canonical_json(resumed.research.state()) == before
    finally:
        resumed.repeat_promotion.close()
        resumed.log.close()
