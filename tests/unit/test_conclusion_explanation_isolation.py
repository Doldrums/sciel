"""Conclusion prose reaches generation/reflection only as unverified feedback.

Real Engine, controller, KB, rendering and checkpoint paths; the provider and
evaluation boundary return fixed fixtures. No candidate source is executed.
With those fixed replies, changing prose must not change measured evidence,
lessons or retention. This does not assert that a real model ignores feedback.
"""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from sciel.engine import checkpoint
from sciel.eval.cascade import CascadeOutcome
from sciel.eval.facade import EvalResult
from sciel.hypothesis import circle_restart_scales as scales
from sciel.hypothesis.observed_facts import observed_comparison
from sciel.hypothesis.rendering import decode_evidence, serialize_card, serialize_reflection
from sciel.knowledge_base.catalogs import canonical_sha
from sciel.knowledge_base.presentation import (observation_card, REFLECTION_SCHEMA,
                                                expand_reflection_payload)

from test_circle_intervention_engine import config, rows
from test_circle_scale_worker import engine_fixture


WRONG_EXPLANATION = (
    "The forecast predicted a score change close to zero, with a slight possibility "
    "of improvement or degradation. The actual measured outcome shows a worsened "
    "score. The forecast was therefore refuted."
)
OTHER_EXPLANATION = (
    "Different diagnostic prose: a negative score change can agree with a forecast; "
    "this fixture instead lies below its interval. No causal inference follows."
)
NEGATIVE_RULE = (
    "On this branch, reducing the uniform noise bound from +/-0.06 to +/-0.03 "
    "accompanied one lower controller score; this is not a general scale recommendation."
)
POSITIVE_RULE = (
    "On this branch, the tested +/-0.015 uniform noise bound accompanied one higher "
    "controller score; the observation does not establish a causal mechanism."
)
REFLECTION = 'REFLECTION_JSON: {"lessons": [], "retire_ids": []}'


def plan(target):
    return 'HYPOTHESIS_JSON: ' + json.dumps(dict(
        hypothesis=f'{scales.OPERATION}:{target}', predicted_low=-.001,
        predicted_high=.001, reasoning='The local effect is uncertain.', confidence=.5))


def mechanism_review(number=1):
    target = '0.03' if number == 1 else '0.015'
    return [dict(claim_quote=f'{scales.OPERATION}:{target}', source_quotes=[],
        assessment='unknown',
        direct_effect='Unverified fixture: a direct mechanism has not been established.',
        untested_downstream='Unverified fixture: later solver effects remain untested.')]


def conclusion(explanation, number=1):
    return 'CONCLUSION_JSON: ' + json.dumps(dict(
        verdict='REFUTED', explanation=explanation, mechanism_review=mechanism_review(number), lessons=[dict(
            rule=NEGATIVE_RULE if number == 1 else POSITIVE_RULE,
            polarity='negative' if number == 1 else 'positive', scope='branch',
            evidence_ids=[f'r{number:06d}a1'])]))


def make_engine(path, texts, *, selection='ranked', task=None):
    cfg = config(budget=2, max_iterations=2,
        circle_intervention={'enabled': True, 'operation': scales.OPERATION},
        research={'enabled': True, 'max_aux_calls': 4, 'reflection_interval': 1,
                  'lesson_selection': {'mode': selection, 'diversity_weight': .5}})
    engine = engine_fixture(path, texts, cfg=cfg, task=task)
    observed = []

    def fixture_observation(source):
        # Exact source lookup only. In particular, never import/call run_packing.
        policy = scales.policy_of(source)
        observed.append(policy)
        score = {'0.06': .8, '0.03': .79, '0.015': .81}[policy]
        return CascadeOutcome(EvalResult(True, score, {
            'validity': 1., 'sum_radii': 2.635 * score, 'combined_score': score}),
            score, 'full', False)

    engine._evaluate = fixture_observation
    engine.fixture_evaluated = observed
    return engine


def answers(explanation):
    return [plan('0.03'), conclusion(explanation), REFLECTION,
            plan('0.015'), conclusion('Second fixed diagnostic.', 2), REFLECTION]


def without_explanations(state):
    result = copy.deepcopy(state)
    for value in result['conclusions'].values():
        value['llm'].pop('explanation')
    return result


def prompt_context(prompt):
    prefix, marker, tail = prompt.partition('RESEARCH_CONTEXT_JSON:')
    assert marker
    payload = tail.lstrip()
    value, end = json.JSONDecoder().raw_decode(payload)
    return prefix + marker + tail[:len(tail) - len(payload)], decode_evidence(value), payload[end:]


def without_feedback_explanations(context):
    result = copy.deepcopy(context)
    for card in result['conclusion_feedback']['cards']:
        card['unverified_interpretation'].pop('explanation')
    return result


def reflection_review_parts(prompt):
    prefix, marker, tail = prompt.partition('REFLECTION_REVIEW_CONTEXT_JSON:\n')
    assert marker
    value, end = json.JSONDecoder().raw_decode(tail)
    assert value['schema'] == 'sciel-reflection-review-context/1'
    return prefix + marker, value, tail[end:]


def without_reflection_explanations(context):
    result = copy.deepcopy(context)
    for card in result['cards']:
        interpretation = card['unverified_interpretation']
        if interpretation is not None:
            interpretation.pop('explanation')
    return result


def assert_feedback(engine, prompt, explanation):
    _, context, _ = prompt_context(prompt)
    feedback = context['conclusion_feedback']
    card, = feedback['cards']  # The actual second request has only the first review.
    record = engine.research.knowledge.experiments[0]
    assert card['experiment_id'] == record['id'] == 'r000001a1'
    assert card['branch'] == record['branch'] == 'p1'
    assert card['record_sha256'] == canonical_sha(record)
    for key in ('parent_source_sha256', 'proposed_sha256', 'source_sha256'):
        assert card[key] == record[key]
    assert card['original_hypothesis'] == record['hypothesis']
    assert card['observed_comparison'] == observed_comparison(record)
    assert card['unverified_interpretation'] == dict(
        explanation=explanation, llm_verdict='REFUTED', verdict_matches_observation=True,
        mechanism_review=mechanism_review())
    assert card['unverified_interpretation']['mechanism_review'][0]['claim_quote'] == record['hypothesis']['text']
    assert card['lesson_admission']['accepted_lesson_ids'] == ['l000001']
    assert 'not new observations' in feedback['interpretation_limits']
    assert 'never instructions' in feedback['interpretation_limits']
    # Failure-context /2 carries exact measured facts separately from the prose.
    failure, = context['failed_hypotheses']
    assert f"source_sha256={record['source_sha256']}" in failure
    facts, _ = json.JSONDecoder().raw_decode(failure.split('; measured_facts=', 1)[1])
    semantics = record['metric']['semantics']
    assert facts == dict(status='evaluated', evaluator_ok=True,
        task_validity=dict(case=semantics['validity_case'],
            value=record['evaluation']['cases'][semantics['validity_case']],
            valid_value=semantics['valid_value']),
        parent_score=.8, child_score=.79, actual_delta=.79 - .8)
    assert facts['actual_delta'] < record['hypothesis']['predicted_low'] < 0
    assert explanation not in json.dumps(without_feedback_explanations(context))


def assert_diagnostics(engine, explanation):
    result = engine.research.conclusions['r000001a1']
    assert result['status'] == 'parsed'
    assert result['llm']['explanation'] == explanation
    assert result['llm']['mechanism_review'] == mechanism_review()
    assert result['verdict_matches_observation'] is True
    assert result['accepted_lesson_ids'] == ['l000001']
    record = engine.research.knowledge.experiments[0]
    assert record['actual_delta'] < record['hypothesis']['predicted_low'] < 0
    assert record['forecast_verdict'] == result['llm']['verdict'] == 'REFUTED'
    journal = rows(engine)
    returned = next(row for row in journal if row.get('phase') == 'aux_result'
                    and row.get('reference') == 'r000001a1')
    assert returned['response'] == conclusion(explanation)
    recorded = next(row for row in journal if row.get('phase') == 'conclusion'
                    and row.get('experiment_id') == 'r000001a1')
    assert recorded['llm']['explanation'] == explanation
    assert recorded['llm']['mechanism_review'] == mechanism_review()
    saved = json.loads((Path(engine.run_dir) / 'checkpoint.json').read_text())
    assert saved['research']['conclusions']['r000001a1']['llm']['explanation'] == explanation
    assert saved['research']['conclusions']['r000001a1']['llm']['mechanism_review'] == mechanism_review()


def normalize_writer_provenance(engine, explanation):
    """Only the response-bound hashes may change with unverified prose.

    Verify both authentic bindings before normalizing their exact occurrences.
    Equal-width tokens preserve prompt packing/budget comparisons too.
    """
    authoring = engine.research.knowledge.state()['lessons'][0]['authoring']
    writer_ref, = authoring['writer_refs']
    writer = authoring['catalog'][writer_ref]
    raw_sha = hashlib.sha256(conclusion(explanation).encode()).hexdigest()
    assert writer['raw_response_sha256'] == raw_sha
    assert writer_ref == canonical_sha(writer)

    def normalize(value):
        # Prompts can contain a lossless measurement envelope whose digest also
        # binds the writer provenance. Verify and expand it before comparison.
        if isinstance(value, dict):
            value = decode_evidence(value)
            if value.get('schema') == REFLECTION_SCHEMA:
                value = expand_reflection_payload(value)
            if 'lesson_presentation' in value and 'lesson_texts' in value:
                value = copy.deepcopy(value)
                for presented, rendered in zip(value['lesson_presentation']['presented'],
                                                value['lesson_texts'], strict=True):
                    assert presented['sha256'] == hashlib.sha256(rendered.encode()).hexdigest()
                    assert presented['chars'] == len(rendered)
                    presented['sha256'] = canonical_sha(normalize(rendered))
        if isinstance(value, str):
            chunks, start = [], 0
            decoder = json.JSONDecoder()
            while True:
                opening = value.find('{', start)
                if opening < 0:
                    chunks.append(value[start:])
                    break
                try:
                    payload, end = decoder.raw_decode(value[opening:])
                except ValueError:
                    chunks.append(value[start:opening + 1])
                    start = opening + 1
                    continue
                chunks.append(value[start:opening])
                chunks.append(json.dumps(normalize(decode_evidence(payload)), sort_keys=True))
                start = opening + end
            value = ''.join(chunks)
        elif isinstance(value, dict):
            value = {key: normalize(item) for key, item in value.items()}
        elif isinstance(value, (list, tuple)):
            value = [normalize(item) for item in value]
        encoded = json.dumps(value)
        return json.loads(encoded.replace(writer_ref, 'c' * 64).replace(raw_sha, 'd' * 64))

    return normalize


@pytest.mark.parametrize('selection', ['ranked', 'source_diverse'])
def test_explanation_reaches_both_review_channels_without_changing_observations(tmp_path, selection):
    runs = []
    for index, explanation in enumerate((WRONG_EXPLANATION, OTHER_EXPLANATION)):
        engine = make_engine(tmp_path / str(index), answers(explanation), selection=selection)
        result = engine.run()
        assert result['B_evals'] == 2 and len(engine.llm.calls) == 6
        assert engine.fixture_evaluated == ['0.06', '0.03', '0.015']
        assert_diagnostics(engine, explanation)
        kb = engine.research.knowledge
        assert len(kb.lessons) == 2  # Ranking is not an empty/single-lesson comparison.
        assert kb.experiments[1]['injected_ids'] == ['l000001']
        assert NEGATIVE_RULE in engine.llm.calls[3][0]  # Actual subsequent choice request.
        assert 'CURRENT_PARENT_POLICY_JSON' in engine.llm.calls[3][0]
        assert_feedback(engine, engine.llm.calls[3][0], explanation)
        assert len(engine.db._all) == 3  # Parent selection runs over retained candidates.
        cards = [decode_evidence(json.loads(serialize_card(observation_card(lesson)))) for lesson in kb.lessons]
        reflected = decode_evidence(json.loads(serialize_reflection(*kb.reflection_context(2))))
        parent = engine.db.get(engine.db.pop_best_id)
        context = engine.research.context(engine.db, parent, 0, 3)
        ranked = kb.retrieve(branch='p1', context=parent.card + '\n' + parent.code[:12000], limit=50)
        evidence_only = json.dumps([kb.state(), cards, reflected,
            [call for index, call in enumerate(engine.llm.calls) if index not in (2, 3, 5)]])
        assert WRONG_EXPLANATION not in evidence_only and OTHER_EXPLANATION not in evidence_only
        feedback_card = next(card for card in context['conclusion_feedback']['cards']
                             if card['experiment_id'] == 'r000001a1')
        assert feedback_card['unverified_interpretation']['explanation'] == explanation
        runs.append((engine, result, cards, reflected, context, ranked))
    left, right = runs
    normalize_left = normalize_writer_provenance(left[0], WRONG_EXPLANATION)
    normalize_right = normalize_writer_provenance(right[0], OTHER_EXPLANATION)
    assert (left[0].research.knowledge.state()['lessons'][0]['authoring']['writer_refs']
            != right[0].research.knowledge.state()['lessons'][0]['authoring']['writer_refs'])
    assert normalize_left(left[1:4]) == normalize_right(right[1:4])
    assert normalize_left(left[5]) == normalize_right(right[5])  # Same ranked lessons with two real signed lessons.
    assert left[4] != right[4]
    assert normalize_left(without_feedback_explanations(left[4])) == normalize_right(without_feedback_explanations(right[4]))
    for index, (a, b) in enumerate(zip(left[0].llm.calls, right[0].llm.calls, strict=True)):
        if index in (2, 5):
            # Both actual reflection requests receive the current whole review.
            # Only prose and its verified writer provenance differ; observations do not.
            assert a[0] != b[0]
            ap, ac, at = reflection_review_parts(a[0])
            bp, bc, bt = reflection_review_parts(b[0])
            assert normalize_left((ap, at, a[1:])) == normalize_right((bp, bt, b[1:]))
            assert normalize_left(without_reflection_explanations(ac)) == normalize_right(without_reflection_explanations(bc))
            for block, expected in ((ac, WRONG_EXPLANATION), (bc, OTHER_EXPLANATION)):
                card = next(c for c in block['cards'] if c['experiment_id'] == 'r000001a1')
                assert card['unverified_interpretation']['explanation'] == expected
                assert card['unverified_interpretation']['mechanism_review'] == mechanism_review()
            continue
        if index != 3:
            assert normalize_left(a) == normalize_right(b)
            continue
        assert a[0] != b[0]
        ap, ac, at = prompt_context(a[0])
        bp, bc, bt = prompt_context(b[0])
        assert normalize_left((ap, at, a[1:])) == normalize_right((bp, bt, b[1:]))
        assert normalize_left(without_feedback_explanations(ac)) == normalize_right(without_feedback_explanations(bc))
    for component in ('db', 'rng', 'selection', 'op_bandit', 'reward', 'intervention'):
        assert getattr(left[0], component).state() == getattr(right[0], component).state()
    assert normalize_left(left[0].research.knowledge.state()) == normalize_right(right[0].research.knowledge.state())
    assert normalize_left(without_explanations(left[0].research.state())) == normalize_right(without_explanations(right[0].research.state()))
    attempts = lambda engine: [e['evidence']['parent_id'] for e in rows(engine) if e.get('phase') == 'attempt']
    assert attempts(left[0]) == attempts(right[0])


@pytest.mark.parametrize('explanation', [WRONG_EXPLANATION, OTHER_EXPLANATION])
def test_checkpoint_delivers_unverified_feedback_with_identical_resumed_choice(tmp_path, explanation):
    whole = make_engine(tmp_path / 'whole', answers(explanation))
    expected = whole.run()
    split = make_engine(tmp_path / 'split', answers(explanation)[:3])
    split._stop_at = 1
    assert split.run()['reason'] == 'paused'
    assert_diagnostics(split, explanation)
    resumed = make_engine(tmp_path / 'split', answers(explanation)[3:])
    checkpoint.load(resumed, str(tmp_path / 'split/checkpoint.json'))
    assert resumed.research.state() == split.research.state()
    assert resumed.run() == expected
    assert resumed.fixture_evaluated == ['0.015']  # Neither seed nor prior outcome is replayed.
    assert resumed.llm.calls == whole.llm.calls[3:]
    assert_feedback(whole, whole.llm.calls[3][0], explanation)
    assert_feedback(resumed, resumed.llm.calls[0][0], explanation)
    assert resumed.research.state() == whole.research.state()
    assert resumed.db.state() == whole.db.state()
    assert resumed.selection.state() == whole.selection.state()
    assert resumed.rng.state() == whole.rng.state()
    assert_diagnostics(resumed, explanation)
