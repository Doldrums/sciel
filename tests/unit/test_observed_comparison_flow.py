"""An authored failed interpretation stays separate from immutable measured facts."""
import copy
import hashlib
import json
from types import SimpleNamespace

import pytest

from sciel.engine.research import ResearchController
from sciel.hypothesis import conclusion_prompt, parse_conclusion
from sciel.hypothesis.observed_facts import observed_comparison
from sciel.hypothesis.measurement import metric_context
from sciel.hypothesis.rendering import decode_evidence, presentation_identity
from sciel.knowledge_base import KnowledgeBase
from sciel.knowledge_base.catalogs import canonical_json
from sciel.knowledge_base.presentation import (
    observation_card, reflection_payload, reflection_inline_payload,
    expand_reflection_payload,
)


@pytest.fixture
def actual():
    from research_memory_fixtures import outside_forecast_history
    record, knowledge = outside_forecast_history()
    response = dict(text='CONCLUSION_JSON: ' + json.dumps(dict(verdict='REFUTED',
        explanation='The measured change falls outside the proposed interval.',
        lessons=[dict(rule='Incorrect interpretation: the score became worse.',
                      polarity='negative', scope='branch', evidence_ids=[record['id']])])))
    return record, response, knowledge


def test_positive_outside_forecast_is_explicit_and_raw_facts_are_exact(actual):
    record, _, _ = actual
    before = copy.deepcopy(record)
    prompt = conclusion_prompt(record)
    facts = json.JSONDecoder().raw_decode(prompt.split('OBSERVED_COMPARISON_JSON:\n', 1)[1])[0]
    assert facts == observed_comparison(record)
    assert facts['measurement']['direction'] == 'improved'
    assert facts['measurement']['actual_delta'] == 0.001855718577133958
    assert facts['forecast']['position'] == 'above'
    assert facts['forecast']['interval_sign'] == 'spans_zero'
    assert facts['forecast']['supplied_verdict'] == 'REFUTED'
    assert decode_evidence(json.loads(prompt.split('EVIDENCE_JSON:\n', 1)[1])) == before
    assert record == before


def test_bad_negative_lesson_stays_rejected_while_card_shows_true_direction(actual):
    record, response, knowledge = actual
    kb = KnowledgeBase(**knowledge['limits'])
    kb.load_state(knowledge)
    bad = parse_conclusion(response['text'])
    assert bad.verdict == 'REFUTED' and bad.lessons[0]['polarity'] == 'negative'
    before = kb.state()
    with pytest.raises(ValueError, match='polarity'):
        kb.add_lessons(bad.lessons, allowed_evidence_ids=[record['id']])
    assert kb.state() == before
    kb.add_lessons([{'rule': 'The recorded controller score increased beyond the forecast interval.',
                    'polarity': 'positive', 'scope': 'branch', 'evidence_ids': [record['id']]}],
                   allowed_evidence_ids=[record['id']])
    lesson = kb.lessons[-1]
    card = observation_card(lesson)
    assert card['schema'] == 'sciel-observation-card/7'
    assert card['comparison_catalog'][record['id']] == observed_comparison(record)
    text = ResearchController.lesson_text(lesson)
    assert decode_evidence(json.loads(text.split('RESEARCH_LESSON_JSON:\n', 1)[1])) == card
    records, lessons = kb.reflection_context(1)
    wire = reflection_payload(records, lessons)
    assert wire['comparison_catalog'][record['id']]['forecast']['position'] == 'above'
    assert expand_reflection_payload(wire) == reflection_inline_payload(records, lessons)
    assert wire['observation_catalog'][record['id']] == next(r for r in records if r['id'] == record['id'])


@pytest.mark.parametrize('corruption', ['direction', 'numeric_bool', 'extra_id'])
def test_self_consistent_digest_cannot_forge_comparison(actual, corruption):
    record, _, _ = actual
    wire = reflection_payload([record], [])
    inline = reflection_inline_payload([record], [])
    for value in (wire, inline):
        if corruption == 'direction':
            value['comparison_catalog'][record['id']]['measurement']['direction'] = 'worsened'
        elif corruption == 'numeric_bool':
            value['comparison_catalog'][record['id']]['measurement']['eligible'] = 1
        else:
            value['comparison_catalog']['invented'] = copy.deepcopy(value['comparison_catalog'][record['id']])
    wire['expanded_sha256'] = hashlib.sha256(canonical_json(inline).encode()).hexdigest()
    with pytest.raises(ValueError, match='comparisons differ'):
        expand_reflection_payload(wire)


def controller(record, text):
    cfg = dict(model=None, max_lessons=20, max_experiments=100, max_fragments=5,
               retrieve_k=3, max_aux_calls=1, reflection_interval=0)
    llm = SimpleNamespace(models=['fixture'], max_retries=1,
        generate=lambda *a, **k: SimpleNamespace(text=text, model='fixture', cost=0.,
                                                prompt_tokens=1, completion_tokens=1))
    return ResearchController(cfg, llm, SimpleNamespace(research=lambda **k: None),
                              lambda c: None, metric=record['metric'])


def test_conclusion_retains_authoritative_comparison_and_resume_checks_it(actual):
    from _mechanism_fixtures import unknown_mechanism
    record, response, knowledge = actual
    # Add a current mechanism review to the authored legacy-shaped response.
    current_fake = parse_conclusion(response['text']).to_dict()
    current_fake['mechanism_review'] = unknown_mechanism(record)
    text = json.dumps(current_fake)
    c = controller(record, text)
    c.knowledge = KnowledgeBase(**knowledge['limits'])
    c.knowledge.load_state(knowledge)
    c.config.update(max_lessons=c.knowledge.max_lessons, max_experiments=c.knowledge.max_experiments,
                    max_fragments=c.knowledge.max_fragments)
    c.queue = [copy.deepcopy(record)]
    c.attempts = c.outcomes = c.valid_hypotheses = 2
    c.complete_step()
    value = c.conclusions[record['id']]
    assert value['llm']['verdict'] == 'REFUTED'
    assert value['lesson_error'] == 'ValueError'
    assert value['observed_comparison']['measurement']['direction'] == 'improved'
    state = c.state()
    restored = controller(record, text)
    restored.config = copy.deepcopy(c.config)
    restored.load_state(state)
    assert restored.state() == state
    for change in ('comparison', 'old_presentation', 'old_aux_transport'):
        bad = copy.deepcopy(state)
        if change == 'comparison':
            bad['conclusions'][record['id']]['observed_comparison']['measurement']['eligible'] = 1
        elif change == 'old_presentation':
            del bad['presentation']['observed_comparison']
        else:
            del bad['presentation']['auxiliary_transport']
        before = restored.state()
        with pytest.raises(ValueError, match='comparison|presentation'):
            restored.load_state(bad)
        assert restored.state() == before
    assert presentation_identity()['observed_comparison'] == 'sciel-observed-comparison/1'
    assert presentation_identity()['auxiliary_transport'] == 'sciel-aux-json-transport/3'


def test_integer_evaluator_score_comparison_survives_normalized_kb_checkpoint():
    from _mechanism_fixtures import unknown_mechanism
    record = dict(id='r000001a1', branch='p1', parent_id='p1', parent_score=1., score=2,
        actual_delta=1., status='evaluated', injected_ids=[], source_sha256='a' * 64,
        proposed_sha256='a' * 64, source_repaired=False, metric=metric_context(),
        parent_cases=None, evaluation=None,
        forecast_verdict='INCONCLUSIVE', hypothesis=None)
    text = 'CONCLUSION_JSON: ' + json.dumps(dict(verdict='INCONCLUSIVE',
            explanation='Score increased; no forecast was supplied.', lessons=[],
            mechanism_review=unknown_mechanism(record)))
    c = controller(record, text)
    c.knowledge.record(record)
    c.queue = [copy.deepcopy(record)]
    c.attempts = c.outcomes = 1
    c.complete_step()
    restored = controller(record, text)
    # The pre-fix queue-based projection stored int 2 beside KB float 2.0.
    old_projection = c.state()
    old_projection['conclusions'][record['id']]['observed_comparison'] = observed_comparison(record)
    with pytest.raises(ValueError, match='comparison mismatch'):
        restored.load_state(old_projection)
    restored.load_state(c.state())
    assert restored.state() == c.state()
    assert type(record['score']) is int
    assert restored.conclusions[record['id']]['observed_comparison']['measurement']['direction'] == 'improved'


def test_queued_observation_evicted_before_conclusion_is_journaled_then_pruned():
    from _mechanism_fixtures import unknown_mechanism
    first = dict(id='r1', branch='p1', parent_id='p1', parent_score=1., score=2,
        actual_delta=1., status='evaluated', injected_ids=[], source_sha256='a' * 64,
        proposed_sha256='a' * 64, source_repaired=False, metric=metric_context(),
        parent_cases=None, evaluation=None, forecast_verdict='INCONCLUSIVE', hypothesis=None)
    second = {**first, 'id': 'r2', 'score': 3, 'actual_delta': 2.}
    text = 'CONCLUSION_JSON: ' + json.dumps(dict(verdict='INCONCLUSIVE', explanation='No forecast.', lessons=[],
                                               mechanism_review=unknown_mechanism(first)))
    c = controller(first, text)
    c.config.update(max_experiments=1, max_aux_calls=2)
    c.knowledge = KnowledgeBase(max_lessons=20, max_experiments=1, max_fragments=5)
    events = []
    c.log = SimpleNamespace(research=lambda **event: events.append(event))
    for record in (first, second):
        c.knowledge.record(record)
        c.queue.append(copy.deepcopy(record))
    c.attempts = c.outcomes = 2
    c.complete_step()
    assert [e['experiment_id'] for e in events if e['phase'] == 'conclusion'] == ['r1', 'r2']
    assert list(c.conclusions) == ['r2']
    restored = controller(first, text)
    restored.config = copy.deepcopy(c.config)
    restored.load_state(c.state())
    assert restored.state() == c.state()
