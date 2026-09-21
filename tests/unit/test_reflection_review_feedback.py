"""Current reviews reach the real auxiliary request without becoming evidence.

Only fixed in-memory model responses and saved synthetic observations are used.
No provider, candidate, evaluator, optimizer or container is executed.
"""
import copy
import json
from types import SimpleNamespace

import pytest

from sciel.config import apply_defaults
from sciel.engine import research as research_module
from sciel.engine import reflection_context as reflection_context_module
from sciel.engine.llm.client import LLMResponse
from sciel.engine.research import ResearchController, digest
from sciel.hypothesis import prompts
from sciel.hypothesis.conclusion_feedback import (
    MAX_CARD_CHARS, REFLECTION_REVIEW_PREFIX, REFLECTION_REVIEW_RULES,
    REFLECTION_REVIEW_SCHEMA, reflection_review_context,
)
from sciel.hypothesis.observed_facts import observed_comparison
from sciel.hypothesis.rendering import decode_evidence, presentation_identity, serialize_reflection
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha


CLAIM = 'Translating both centers reduces their internal gap.'
EXPLANATION = 'The common translation preserves the internal relative vector; later effects are untested.'
SOURCE = 'left += delta\nright += delta\n'
EMPTY_REFLECTION = 'REFLECTION_JSON: {"lessons":[],"retire_ids":[]}'


def review_text(*, explanation=EXPLANATION):
    return dict(verdict='REFUTED', explanation=explanation, lessons=[], mechanism_review=[dict(
        claim_quote=CLAIM, source_quotes=[dict(field='change_excerpt', text=SOURCE)],
        assessment='contradicted', direct_effect='Equal displacement preserves the relative vector.',
        untested_downstream='This does not establish later solver behavior or a quality effect.')])


class Model:
    models = ['fixture']
    max_retries = 1

    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def generate(self, prompt, model, temperature=None):
        self.calls.append(dict(prompt=prompt, model=model, temperature=temperature))
        value = next(self.responses)
        return LLMResponse(json.dumps(value) if type(value) is dict else value,
                           model, .001, 19, 11)


def controller(responses=(), *, interval=1, aux=8, metric=None, inject=True):
    cfg = apply_defaults({'research': dict(enabled=True, max_aux_calls=aux,
        reflection_interval=interval, inject_lessons=inject)})['research']
    model, events, charges = Model(responses), [], []
    c = ResearchController(cfg, model,
        SimpleNamespace(research=lambda **kw: events.append(copy.deepcopy(kw))), charges.append, metric=metric)
    return c, model, events, charges


def observation(c, number=1, *, queue=True, branch='p1'):
    code = f'def run_search():\n    return {number}\n'
    c.knowledge.record(dict(id=f'r{number}', branch=branch, parent_id=branch,
        status='evaluated', parent_score=.5, score=.6, actual_delta=.6-.5,
        parent_source_sha256=digest('parent'), proposed_sha256=digest(code), source_sha256=digest(code),
        source_repaired=False, code_fragment=code, change_excerpt=SOURCE,
        hypothesis=dict(text=CLAIM, reasoning='This could improve packing.',
            predicted_low=0., predicted_high=.2, confidence=.5),
        metric=c.metric, parent_cases={}, evaluation=dict(ok=True, score=.6, cases={},
            reached='full', confirmed=False, raw_first_score=.6), injected_ids=[]))
    record = c.knowledge.experiments[-1]
    c.attempts += 1
    c.outcomes += 1
    c.valid_hypotheses += 1
    if queue:
        c.queue.append(copy.deepcopy(record))
    return record


def saved_review(record, *, status='parsed', legacy=False):
    llm = review_text()
    if legacy:
        llm.pop('mechanism_review')
    return dict(status=status, llm=llm if status == 'parsed' else None,
        observed_forecast_verdict=record['forecast_verdict'], observed_comparison=observed_comparison(record),
        verdict_matches_observation=False if status == 'parsed' else None,
        accepted_lesson_ids=[], lesson_error=None, lesson_error_detail=None)


def block(prompt):
    return json.JSONDecoder().raw_decode(prompt.split(REFLECTION_REVIEW_PREFIX, 1)[1])[0]


def facts(prompt):
    return prompt.split('EVIDENCE_JSON:\n', 1)[1].rstrip('\n')


def test_actual_reflection_request_receives_whole_typed_correction_without_a_lesson():
    c, model, events, charges = controller([review_text(), EMPTY_REFLECTION])
    record = observation(c)
    expected_facts = serialize_reflection(*c.knowledge.reflection_context(1))
    before = c.knowledge.state()
    c.complete_step()
    assert len(model.calls) == c.aux_calls == 2 and charges == [.001, .001]
    prompt = model.calls[-1]['prompt']
    card, = block(prompt)['cards']
    assert card['unverified_interpretation'] == review_text()
    assert card['unverified_interpretation']['lessons'] == []
    assert card['review_status'] == 'parsed' and card['verdict_matches_observation'] is False
    assert card['record_sha256'] == canonical_sha(record)
    assert card['record_representation'] == 'ordinary_record'
    for key in ('parent_source_sha256', 'proposed_sha256', 'source_sha256', 'branch'):
        assert card[key] == record[key]
    assert facts(prompt) == expected_facts
    assert EXPLANATION not in expected_facts
    assert c.knowledge.state() == before
    assert c.reflections == 1 and c.reflection_changes == 0
    context = next(e for e in events if e['phase'] == 'reflection_context')
    assert context['review_context_sha256'] == canonical_sha(block(prompt))
    assert context['context_sha256'] == digest(expected_facts)
    dispatched = [e for e in events if e['phase'] == 'aux_start' and e['kind'] == 'reflection']
    assert len(dispatched) == 1 and dispatched[0]['prompt_sha256'] == digest(prompt)
    assert 'Literal quote validation is not semantic proof' in prompt


@pytest.mark.parametrize('failure', ['parse', 'binding'])
def test_invalid_new_review_cannot_rescue_old_review_but_measured_lesson_remains_eligible(failure):
    rejected = review_text(explanation='REJECTED_INTERPRETATION_MUST_NOT_REACH_REFLECTION')
    if failure == 'parse':
        rejected['mechanism_review'][0]['source_quotes'][0]['text'] = 'X' * 513
    else:
        rejected['mechanism_review'][0]['claim_quote'] = 'NOT_THE_ORIGINAL_CLAIM'
    proposal = dict(rule='This source had one positive local measured delta; causality is unverified.',
                    polarity='positive', scope='branch', evidence_ids=['r1'])
    c, model, events, _ = controller([rejected, dict(lessons=[proposal], retire_ids=[])])
    record = observation(c)
    old = saved_review(record)
    old['llm']['explanation'] = 'OLD_REVIEW_MUST_NOT_RESCUE_INVALID_CURRENT'
    c.conclusions['r1'] = old
    c.complete_step()
    prompt = model.calls[-1]['prompt']
    card, = block(prompt)['cards']
    assert card['review_status'] == 'invalid' and card['unverified_interpretation'] is None
    assert 'REJECTED_INTERPRETATION' not in prompt and 'OLD_REVIEW' not in prompt
    assert 'NOT_THE_ORIGINAL_CLAIM' not in prompt and 'X'*513 not in prompt
    assert c.conclusions['r1']['llm'] is None
    assert c.knowledge.experiments == [record]
    assert len(c.knowledge.lessons) == 1
    assert c.knowledge.lessons[0]['rule'] == proposal['rule']
    assert c.knowledge.lessons[0]['evidence_ids'] == ['r1']
    rejected_event, = [e for e in events if e['phase'] == 'conclusion_rejected']
    diagnostic = rejected_event['diagnostic']
    assert diagnostic['code'] == ('text_too_long' if failure == 'parse' else 'claim_quote_mismatch')
    assert diagnostic['path'].startswith('mechanism_review[0]')
    assert len(canonical_json(diagnostic)) < 256 and 'X'*513 not in canonical_json(diagnostic)
    if failure == 'binding':
        assert rejected_event['detail'] == 'mechanism claim quote does not match original hypothesis'


@pytest.mark.parametrize('status', ['skipped', 'invalid', 'unavailable'])
def test_current_nonparsed_status_is_explicit_in_actual_reflection(status):
    c, model, _, _ = controller([EMPTY_REFLECTION])
    record = observation(c, queue=False)
    if status != 'unavailable':
        c.conclusions['r1'] = saved_review(record, status=status)
    c.complete_step()
    card, = block(model.calls[0]['prompt'])['cards']
    assert card['review_status'] == status and card['unverified_interpretation'] is None
    assert 'lesson_admission' not in card and 'verdict_matches_observation' not in card
    assert len(model.calls) == 1 and c.knowledge.lessons == []


def test_only_selected_ordinary_records_and_their_own_branch_source_bindings_travel():
    c, _, _, _ = controller()
    a, b = observation(c, 1, queue=False), observation(c, 2, queue=False, branch='p2')
    evidence = c.knowledge._evidence()
    reviews = {'r1': saved_review(a), 'r2': saved_review(b)}
    before = copy.deepcopy((reviews, evidence))
    visible = reflection_review_context(reviews, evidence, ['r2'])
    card, = visible['cards']
    assert card['experiment_id'] == 'r2' and card['branch'] == 'p2'
    assert card['source_sha256'] == b['source_sha256']
    assert card['record_sha256'] == canonical_sha(evidence['r2'])
    assert card['record_representation'] == 'evidence_snapshot'
    assert 'r1' not in canonical_json(visible)
    card['unverified_interpretation']['mechanism_review'][0]['source_quotes'].clear()
    assert (reviews, evidence) == before
    with pytest.raises(ValueError, match='ordinary'):
        reflection_review_context({}, {'group': {'id': 'group', 'schema': 'paired'}}, ['group'])
    with pytest.raises(ValueError, match='unique selected'):
        reflection_review_context(reviews, evidence, ['r1', 'r1'])


def test_full_native_record_hash_survives_metadata_projection_in_reflection():
    from test_native_geometry_knowledge import controller as native_controller, observe
    c, parent, db, events = native_controller()
    record, _ = observe(c, parent, db)  # Fixed synthetic returned artifact; no run_stage.
    c.config.update(reflection_interval=1, max_aux_calls=1)
    c.llm = model = Model([EMPTY_REFLECTION])
    c.charge = lambda _: None
    c.conclusions[record['id']] = saved_review(record, legacy=True)
    c.complete_step()
    prompt = model.calls[0]['prompt']
    card, = block(prompt)['cards']
    assert card['record_sha256'] == canonical_sha(record)
    projected = decode_evidence(json.loads(facts(prompt)))['observation_catalog'][record['id']]
    assert projected['evaluation']['native_geometry'] != record['evaluation']['native_geometry']
    assert 'numeric' not in projected['evaluation']['native_geometry']
    assert card['record_sha256'] != canonical_sha(projected)
    assert c.knowledge.experiments[0] == record


def test_evicted_lesson_source_uses_snapshot_identity_without_resurrecting_pruned_review():
    from sciel.knowledge_base import KnowledgeBase
    c, model, _, _ = controller([EMPTY_REFLECTION], interval=0)
    c.knowledge = KnowledgeBase(max_experiments=1)
    c.config['max_experiments'] = 1
    old = observation(c, queue=False)
    c.conclusions['r1'] = saved_review(old)
    c.knowledge.add_lessons([dict(rule='One local positive measurement, not a causal claim.',
        polarity='positive', scope='branch', evidence_ids=['r1'])], allowed_evidence_ids=['r1'])
    observation(c, 2, queue=False)
    c.complete_step()  # Normal pruning, without a reflection or model request.
    assert 'r1' not in c.conclusions and [r['id'] for r in c.knowledge.experiments] == ['r2']
    snapshot = copy.deepcopy(c.knowledge._evidence()['r1'])
    assert 'code_fragment' not in snapshot and snapshot['code_fragment_sha256'] == digest(old['code_fragment'])
    c.config['reflection_interval'] = 1
    before = c.knowledge.state()
    c.complete_step()
    prompt = model.calls[0]['prompt']
    card = next(card for card in block(prompt)['cards'] if card['experiment_id'] == 'r1')
    assert card['record_representation'] == 'evidence_snapshot'
    assert card['record_sha256'] == canonical_sha(snapshot)
    assert card['review_status'] == 'unavailable' and card['unverified_interpretation'] is None
    assert EXPLANATION not in prompt and CLAIM in facts(prompt)
    assert c.knowledge.state() == before and len(model.calls) == 1


def test_legacy_review_restores_without_mechanism_fabrication():
    c, _, _, _ = controller()
    record = observation(c, queue=False)
    c.conclusions['r1'] = saved_review(record, legacy=True)
    saved = c.state()
    restored, model, _, _ = controller([EMPTY_REFLECTION])
    restored.load_state(copy.deepcopy(saved))
    assert restored.state() == saved
    restored.complete_step()
    card, = block(model.calls[0]['prompt'])['cards']
    assert card['unverified_interpretation'] == saved['conclusions']['r1']['llm']
    assert 'mechanism_review' not in card['unverified_interpretation']


def test_checkpointed_current_reviews_produce_identical_actual_reflection_after_resume():
    continuous, model, _, _ = controller([review_text(), review_text(), EMPTY_REFLECTION], interval=2)
    observation(continuous)
    continuous.complete_step()
    saved = continuous.state()
    assert len(model.calls) == 1 and saved['queue'] == []
    resumed, resumed_model, _, _ = controller([review_text(), EMPTY_REFLECTION], interval=2)
    resumed.load_state(copy.deepcopy(saved))
    for c in (continuous, resumed):
        observation(c, 2)
        c.complete_step()
    assert model.calls[1:] == resumed_model.calls
    assert continuous.state() == resumed.state()
    assert len(model.calls) == 3 and len(resumed_model.calls) == 2
    assert len(block(resumed_model.calls[-1]['prompt'])['cards']) == 2


def test_old_presentation_identity_and_tampered_current_review_restore_fail_atomically():
    c, _, _, _ = controller()
    record = observation(c, queue=False)
    c.conclusions['r1'] = saved_review(record)
    for kind in ('identity', 'binding'):
        saved = c.state()
        if kind == 'identity':
            saved['presentation'].pop('reflection_review')
        else:
            saved['conclusions']['r1']['llm']['mechanism_review'][0]['source_quotes'][0]['text'] = 'unseen source'
        target, model, _, _ = controller()
        before = target.state()
        with pytest.raises(ValueError):
            target.load_state(saved)
        assert target.state() == before and model.calls == []
    assert presentation_identity()['reflection_review'] == REFLECTION_REVIEW_SCHEMA


@pytest.mark.parametrize('remaining', [0, 1200, 10000])
def test_combined_budget_preserves_evidence_and_discloses_whole_review_omissions(monkeypatch, remaining):
    c, model, events, _ = controller([EMPTY_REFLECTION])
    record = observation(c, queue=False)
    c.conclusions['r1'] = saved_review(record)
    evidence = serialize_reflection(*c.knowledge.reflection_context(1))
    overhead = len(REFLECTION_REVIEW_RULES) + len(REFLECTION_REVIEW_PREFIX) + 1
    total = len(evidence) + overhead + remaining
    monkeypatch.setattr(reflection_context_module, 'MAX_CONTEXT_CHARS', total)
    monkeypatch.setattr(prompts, 'MAX_CONTEXT_CHARS', total)
    c.complete_step()
    prompt = model.calls[0]['prompt']
    assert facts(prompt) == evidence
    event = next(e for e in events if e['phase'] == 'reflection_context')
    assert event['review_context_max_chars'] == remaining
    if remaining == 0:
        assert REFLECTION_REVIEW_PREFIX not in prompt
        assert event['review_context_omission'] == 'memory_char_limit'
        assert event['review_omitted_count'] == 1 and event['review_presented_count'] == 0
    else:
        shown = block(prompt)
        assert len(evidence) + overhead + len(canonical_json(shown)) <= total
        assert shown['presented_count'] + shown['omitted_count'] == 1
        assert shown['omitted_count'] == len(shown['omitted']) + shown['omission_details_omitted_count']
        if shown['cards']:
            assert shown['cards'][0]['unverified_interpretation'] == review_text()
        else:
            assert EXPLANATION not in prompt


def test_whole_escaped_review_and_diagnostics_respect_card_budget_without_slicing():
    c, _, _, _ = controller()
    record = observation(c, queue=False)
    review = saved_review(record)
    review['llm']['explanation'] = '界' * 4096
    shown = reflection_review_context({'r1': review}, c.knowledge._evidence(), ['r1'])
    assert shown['cards'] == [] and '界' not in canonical_json(shown)
    omission, = shown['omitted']
    assert omission['reason'] == 'card_char_limit' and omission['card_chars'] > MAX_CARD_CHARS
    # Supplied contexts may not bypass the combined prompt cap either.
    with pytest.raises(ValueError, match='context character limit'):
        prompts.reflection_prompt([], [], review_context=dict(schema=REFLECTION_REVIEW_SCHEMA, huge='x'*140000))


def test_auxiliary_cap_and_generation_ablation_do_not_add_review_calls_or_exposures():
    c, model, events, charges = controller([], aux=0, inject=False)
    record = observation(c)
    before = c.knowledge.state()
    c.complete_step()
    assert c.conclusions['r1']['status'] == 'skipped'
    assert c.knowledge.state() == before and record['injected_ids'] == []
    assert model.calls == charges == [] and c.aux_calls == 0
    assert [e['kind'] for e in events if e['phase'] == 'aux_skipped'] == ['conclusion', 'reflection']


def test_prompt_default_keeps_legacy_evidence_only_surface():
    c, _, _, _ = controller()
    observation(c, queue=False)
    records, lessons = c.knowledge.reflection_context(1)
    prompt = prompts.reflection_prompt(records, lessons)
    assert REFLECTION_REVIEW_PREFIX not in prompt
    assert facts(prompt) == serialize_reflection(records, lessons)
