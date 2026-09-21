"""Post-evaluation admission with inert full-output fixtures; no source execution."""
from copy import deepcopy
from types import SimpleNamespace
import json

import pytest

from sciel.campaign.budget import CampaignStopped
from sciel.engine.idea import TwoStageIdea
from sciel.engine.output_novelty import OutputNoveltyRegistry
from sciel.engine.types import NoChild, Result
from sciel.engine.validation import source_sha256
from sciel.engine.worker import run_worker
from sciel.eval.output_identity import adapt_output, sealed
from sciel.hypothesis.measurement import evaluation_observation, metric_context
from sciel.knowledge_base import KnowledgeBase
from sciel.knowledge_base.decision import decision_card
from sciel.knowledge_base.presentation import observation_card, reflection_payload, expand_reflection_payload
from test_novelty_worker_retries import callback, registry_with, previous
from test_output_identity import fixture
from test_research_retry_feedback import setup, text, PARENT, CHILD
from test_two_stage_idea import setup as idea_setup, response, code_response, CARD
from test_research_knowledge import experiment, lesson, proposal

NEXT = CHILD.replace('0.6', '0.7')
LAST = CHILD.replace('0.6', '0.9')


def evaluator(tmp_path, outputs, *, sources=None):
    registry = OutputNoveltyRegistry({'task_hash': 'a' * 64})
    task, _, old, binding = fixture(tmp_path, source=source_sha256(PARENT), output={'value': 0})
    registry.observe('evaluation-old', adapt_output(task, binding['source_sha256'], old, binding), binding)
    registry.bind_program('evaluation-old', 'p-archived')
    calls, outcomes = [], []
    values = iter(outputs)

    def evaluate(code):
        calls.append(code)
        task, _, out, bind = fixture(tmp_path, source=source_sha256(code), output=next(values))
        out.evaluation_id = f'evaluation-{len(calls)}'
        out.output_novelty = registry.observe(out.evaluation_id, adapt_output(task, bind['source_sha256'], out, bind), bind)
        if sources is not None:
            sources.register(code, out.evaluation_id, island=0, evaluation_ok=True)
        outcomes.append(out)
        return out
    return evaluate, calls, outcomes


@pytest.mark.parametrize('research_on', [False, True])
def test_repeat_then_new_retains_both_measurements_and_exact_feedback(tmp_path, research_on):
    model, research, snap, _, _, _ = setup([text(), text(NEXT)], research_on=research_on)
    evaluate, calls, outcomes = evaluator(tmp_path, [{'value': 0}, {'value': 1}])
    result = run_worker('diff', snap, model, evaluate, research=research if research_on else None,
                        novelty_check=callback(registry_with()))
    assert isinstance(result, Result) and result.child_code == NEXT
    assert result.cost == pytest.approx(.02) and calls == [CHILD, NEXT]
    assert result.meta['evaluation_count'] == 2
    assert result.meta['evaluation_ids'] == ['evaluation-1', 'evaluation-2']
    assert result.meta['output_repeat_rejections'] == len(result.meta['output_repeats']) == 1
    prompt = model.calls[1][0]
    assert previous(prompt) == text()
    assert 'POST-EVALUATION OUTPUT REPEAT' in prompt and 'p-archived' in prompt
    assert '"quality_measured": true' in prompt
    assert 'evaluation-old' in prompt and 'evaluation-1' in prompt
    if research_on:
        first, final = research.knowledge.experiments
        assert first['status'] == 'evaluated' and first['score'] == .8 and first['actual_delta'] > 0
        assert first['evaluation']['ok'] and first['code_fragment'] == ''
        assert first['output_novelty'] == outcomes[0].output_novelty
        assert [row['id'] for row in research.queue] == [final['id']]
        assert len(research.knowledge.fragments) == 1
        assert result.meta['research_outcome_ids'] == [first['id'], final['id']]


def test_source_and_output_rejections_share_three_request_limit(tmp_path):
    model, research, snap, _, _, _ = setup([text(PARENT), text(), text(NEXT)])
    evaluate, calls, _ = evaluator(tmp_path, [{'value': 0}, {'value': 2}])
    result = run_worker('diff', snap, model, evaluate, research=research,
                        novelty_check=callback(registry_with()), dup_cap=19)
    assert isinstance(result, Result) and len(model.calls) == 3 and len(calls) == 2
    assert result.cost == pytest.approx(.03) and result.meta['evaluation_count'] == 2
    assert previous(model.calls[1][0]) == text(PARENT)
    assert previous(model.calls[2][0]) == text()
    assert len(research.queue) == 1


@pytest.mark.parametrize('prefix', [[], [text(PARENT), text('# only bytes\n'+PARENT)]])
def test_exhaustion_never_makes_fourth_request_or_efficacy_conclusion(tmp_path, prefix):
    replies = prefix + [text(code) for code in [CHILD, NEXT, LAST]][:3-len(prefix)]
    model, research, snap, _, _, _ = setup(replies)
    evaluate, calls, _ = evaluator(tmp_path, [{'value': 0}]*3)
    result = run_worker('diff', snap, model, evaluate, research=research,
                        novelty_check=callback(registry_with()))
    assert isinstance(result, NoChild) and result.reason == 'output_repeat_cap'
    assert len(model.calls) == 3 and len(calls) == 3-len(prefix)
    assert result.meta['evaluation_count'] == result.meta['output_repeat_rejections'] == len(calls)
    assert result.cost == pytest.approx(.03)
    assert research.queue == [] and research.knowledge.lessons == [] and research.knowledge.fragments == []
    assert len(research.knowledge.experiments) == 3
    clone = type(research)(research.config, model, SimpleNamespace(research=lambda **kw: None), lambda _: None)
    clone.load_state(research.state())
    assert clone.state() == research.state()


def test_known_source_remeasurement_does_not_regenerate(tmp_path):
    model, research, snap, _, _, _ = setup([text()])
    evaluate, calls, _ = evaluator(tmp_path, [{'value': 0}])
    result = run_worker('diff', snap, model, evaluate, research=research,
                        novelty_check=callback(registry_with(CHILD, ok=False)))
    assert isinstance(result, Result) and result.meta['remeasurement']
    assert len(model.calls) == len(calls) == 1 and result.meta['evaluation_count'] == 1
    assert result.meta['output_repeat_rejections'] == 0 and len(result.meta['output_repeats']) == 1
    assert research.queue == [] and research.knowledge.experiments[0]['status'] == 'no_child'


@pytest.mark.parametrize('damage', ['source', 'evaluation_id', 'seal'])
def test_bad_verdict_fails_closed_without_losing_paid_response(tmp_path, damage):
    model, research, snap, _, _, _ = setup([text()])
    charged = []
    research.charge = charged.append
    evaluate, _, _ = evaluator(tmp_path, [{'value': 0}])
    def broken(code):
        outcome = evaluate(code)
        if damage == 'source':
            outcome.output_novelty['binding']['source_sha256'] = 'f'*64
        elif damage == 'evaluation_id': outcome.evaluation_id = 'different-evaluation'
        else: outcome.output_novelty['sha256'] = 'f'*64
        return outcome
    with pytest.raises(CampaignStopped):
        run_worker('diff', snap, model, broken, research=research, novelty_check=callback(registry_with()))
    assert charged == [.01] and research.queue == []


@pytest.mark.parametrize('accepted', [False, True])
def test_idea_card_once_code_and_output_share_budget_and_state(tmp_path, accepted):
    codes = [CHILD, NEXT, LAST]
    research, idea, snap, model, _, charged, _, run = idea_setup(
        [response(json.dumps(CARD))]+[response(code_response(code)) for code in codes])
    evaluate, calls, _ = evaluator(tmp_path, [{'value': 0}]*2+[{'value': 1 if accepted else 0}])
    result = run(evaluator=evaluate, novelty_check=callback(registry_with()))
    assert isinstance(result, Result if accepted else NoChild)
    assert len(model.calls) == 4 and calls == codes and result.cost == pytest.approx(.12)
    assert idea.attempts == 1 and idea.evaluations == 3 and idea.code_retries == 2
    assert result.meta['evaluation_count'] == 3 and result.meta['output_repeat_rejections'] == (2 if accepted else 3)
    assert previous(model.calls[2]) == code_response(CHILD)
    assert previous(model.calls[3]) == code_response(NEXT)
    assert len(research.queue) == int(accepted) and charged == []
    restored = TwoStageIdea(research)
    restored.load_state(idea.state())
    assert restored.state() == idea.state()


def repeated_record(tmp_path, number=2, *, score=.8):
    raw = experiment(number, delta=score-.5, code_fragment='')
    evaluate, _, outcomes = evaluator(tmp_path, [{'value': 0}])
    outcome = evaluate(f'value = {number}\n')
    raw.update(metric=metric_context(), parent_cases={}, evaluation=evaluation_observation(outcome),
               output_novelty=outcome.output_novelty)
    return raw


def test_positive_repeat_is_not_lesson_support_fragment_or_exposure_gain(tmp_path):
    kb = KnowledgeBase()
    kb.record(experiment(1))
    lid = lesson(kb, 'r000001a1')
    repeated = repeated_record(tmp_path)
    repeated['injected_ids'] = [lid]
    kb.record(repeated)
    saved = kb.experiments[-1]
    assert saved['status'] == 'evaluated' and saved['score'] == .8 and saved['actual_delta'] > 0
    assert len(kb.fragments) == 1 and kb.lessons[0]['support_ids'] == kb.lessons[0]['refute_ids'] == []
    assert kb.lessons[0]['exposure_ids'] == [saved['id']] and kb.lessons[0]['confidence'] == .5
    with pytest.raises(ValueError):
        kb.add_lessons([proposal(saved['id'], rule='Claim a benefit')], allowed_evidence_ids=[saved['id']])
    for card in (observation_card(kb.lessons[0]), decision_card(kb.lessons[0])):
        counts = card['advice_exposures']['retained_unique_counts']
        assert counts['recorded_output_repeat'] == counts['total'] == 1
        assert counts['positive'] == counts['negative'] == 0
        row = card['advice_exposures']['representatives'][0]['observation']
        assert row['efficacy_eligible'] is False and 'output_novelty' not in row
        assert row['output_admission']['verdict_sha256'] == saved['output_novelty']['sha256']
        assert row['output_admission']['match'] == saved['output_novelty']['match']
    records, lessons = kb.reflection_context(10)
    payload = reflection_payload(records, lessons)
    expanded = expand_reflection_payload(payload)
    assert expanded['lessons'][0]['advice_exposures']['retained_unique_counts']['recorded_output_repeat'] == 1
    assert 'not admitted and not efficacy evidence' in kb.failed_hypotheses('seed0')[0]
    restored = KnowledgeBase()
    restored.load_state(json.loads(json.dumps(kb.state())))
    assert restored.state() == kb.state()


@pytest.mark.parametrize('damage', ['source', 'stage', 'failed', 'fragment', 'task', 'island'])
def test_repeat_record_validation_is_atomic(tmp_path, damage):
    kb = KnowledgeBase()
    raw = repeated_record(tmp_path)
    if damage == 'source': raw['source_sha256'] = 'd'*64
    elif damage == 'stage': raw['evaluation']['reached'] = 'smoke'
    elif damage == 'failed': raw.update(status='eval_failed',score=None,actual_delta=None)
    elif damage == 'fragment': raw['code_fragment'] = 'value = 2\n'
    elif damage == 'task': raw['metric'] = metric_context(SimpleNamespace(task_hash='d'*64))
    else: raw['island'] = 4
    before = kb.state()
    with pytest.raises(ValueError): kb.record(raw)
    assert kb.state() == before


def test_compact_new_source_and_repeat_exposure_do_not_duplicate_full_proofs(tmp_path):
    from sciel.knowledge_base.eligibility import admission_view
    kb = KnowledgeBase()
    new = repeated_record(tmp_path, 1)
    task, _, outcome, binding = fixture(tmp_path, source=new['source_sha256'], output={'first': 1})
    new['output_novelty'] = OutputNoveltyRegistry({'task_hash': 'a'*64}).observe('new-evaluation',
        adapt_output(task, new['source_sha256'], outcome, binding), binding)
    kb.record(new)
    lid = lesson(kb, new['id'])
    repeated = repeated_record(tmp_path, 2)
    repeated['injected_ids'] = [lid]
    kb.record(repeated)
    kept = kb.lessons[0]
    decision = decision_card(kept)
    card = observation_card(kept)
    for view in (decision, card):
        assert 'output_novelty' not in json.dumps(view)
        source = view['source_observations'][0]
        assert source['output_admission']['status'] == 'new_output' and source['efficacy_eligible'] is True
        assert source['output_admission']['verdict_sha256'] == new['output_novelty']['sha256']
    assert len(json.dumps(decision)) < 8000
    assert len(json.dumps(admission_view(new))) < len(json.dumps(new['output_novelty'])) / 2
    records, lessons = kb.reflection_context(10)
    compact = reflection_payload(records, lessons)
    expanded = expand_reflection_payload(compact)
    assert 'output_novelty' not in json.dumps(expanded)
    assert expanded['observation_catalog'][new['id']]['output_admission'] == admission_view(new)
    assert expanded['lessons'][0]['advice_exposures']['retained_unique_counts']['recorded_output_repeat'] == 1
    # The prompt projection does not mutate or replace the authoritative proof.
    assert kb.experiments[0]['output_novelty'] == new['output_novelty']
    clone = KnowledgeBase()
    clone.load_state(kb.state())
    assert clone.state() == kb.state()


def test_negative_repeat_retains_exposure_without_refutation_credit(tmp_path):
    kb = KnowledgeBase()
    kb.record(experiment(1))
    lid = lesson(kb, 'r000001a1')
    repeated = repeated_record(tmp_path)
    repeated.update(parent_score=1.0, actual_delta=.8-1.0, injected_ids=[lid])
    kb.record(repeated)
    assert kb.lessons[0]['exposure_ids'] == [repeated['id']]
    assert kb.lessons[0]['support_ids'] == kb.lessons[0]['refute_ids'] == []
    with pytest.raises(ValueError):
        kb.add_lessons([proposal(repeated['id'], rule='Infer damage', polarity='negative')],
                       allowed_evidence_ids=[repeated['id']])


@pytest.mark.parametrize('damage', ['numeric', 'artifact', 'invocation', 'dtype', 'evaluator', 'confirmation'])
def test_recorded_native_geometry_cannot_accept_spliced_output_identity(damage):
    from pathlib import Path
    from sciel.eval.cascade import CascadeOutcome
    from sciel.eval.facade import EvalResult
    from sciel.eval.output_identity import canonical
    from sciel.eval.registry import load_task
    from test_native_geometry_contract import fixture as geometry, observation
    task = load_task(str(Path(__file__).resolve().parents[2] / 'tasks/oe/circle_packing'))
    confirmation = (dict(trials=max(2, task.confirm_n), selected_trial=0,
        selection='first_failure_else_lowest_score') if damage == 'confirmation' else None)
    witness = geometry(confirmation=confirmation)
    observed = observation(witness)
    outcome = CascadeOutcome(EvalResult(True, observed['score'], observed['cases'],
        artifacts={'native_geometry': canonical(witness)}, confirmed_sampled=confirmation is not None),
        raw_first_score=.2, reached='confirm' if confirmation else 'full', confirmed=confirmation is not None)
    binding = dict(task_hash=task.task_hash, source_sha256=witness['source_sha256'], island_id=0)
    identity = adapt_output(task, witness['source_sha256'], outcome, binding)
    verdict = OutputNoveltyRegistry({'task_hash': task.task_hash}).observe('native-evaluation', identity, binding)
    if damage == 'numeric': identity['output_sha256'] = 'f'*64
    elif damage == 'artifact': identity['evidence']['artifact_sha256'] = 'f'*64
    elif damage == 'invocation': identity['evidence']['invocation_id'] = 'f'*32
    elif damage == 'dtype': identity['scope']['representation']['radii_dtype'] = 'float16'
    elif damage == 'confirmation': identity['scope']['stage']['confirmation']['trials'] += 1
    else: identity['scope']['evaluator_sha256'] = 'f'*64
    verdict['identity'] = sealed({k:v for k,v in identity.items() if k != 'sha256'})
    verdict = sealed({k:v for k,v in verdict.items() if k != 'sha256'})
    raw = experiment(1, score=observed['score'], parent_score=0., actual_delta=observed['score'],
        source_sha256=witness['source_sha256'], proposed_sha256=witness['source_sha256'], code_fragment='',
        metric=metric_context(task), parent_cases={}, evaluation=evaluation_observation(outcome),
        output_novelty=verdict)
    kb = KnowledgeBase()
    with pytest.raises(ValueError, match='recorded native geometry'): kb.record(raw)
    assert not kb.experiments
