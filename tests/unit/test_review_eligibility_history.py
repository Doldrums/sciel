"""Host-gated R boundaries with real parsing/KB and synthetic saved journals.

No provider, solver, candidate evaluation or historical geometry is executed.
The standalone Job test stubs only pure preparation; its real returned-response
admission must reject lesson effects before touching the supplied detached KB.
"""
import copy
import json
from types import SimpleNamespace as NS

import pytest

from sciel.config import apply_defaults
from sciel.engine import external_research_resume as resume
from sciel.engine import external_research_v2 as reducer
from sciel.engine.external_reflection import ExternalReflectionJob, ExternalReflectionRejected
from sciel.hypothesis.auxiliary_schema import identity, response_format
from sciel.hypothesis.rendering import serialize_reflection
from sciel.knowledge_base import KnowledgeBase
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
from test_research_knowledge import experiment


def policy():
    return {'schema': 'sciel-review-text-policy/1', 'decisions': []}


def configured(**changes):
    return dict(comparison_reviews=True, structured_auxiliary=True,
        comparison_review_version=1, comparison_review_text=True,
        batch_only_lessons=True, comparison_review_text_policy=policy(), **changes)


@pytest.mark.parametrize('text', [True, False])
def test_config_accepts_detached_explicit_host_policy(text):
    raw = configured()
    raw['comparison_review_text'] = text
    normalized = apply_defaults({'research': raw})['research']
    assert normalized['comparison_review_text_policy'] == policy()
    assert normalized['comparison_review_text'] is text
    raw['comparison_review_text_policy']['decisions'].append({'untrusted': 'later mutation'})
    assert normalized['comparison_review_text_policy'] == policy()


@pytest.mark.parametrize('damage', ['reviews', 'structured', 'text_missing', 'text_int', 'batch'])
def test_config_and_standalone_reducer_enforce_gated_dependencies(damage):
    raw = configured()
    if damage == 'reviews': raw['comparison_reviews'] = False
    elif damage == 'structured': raw['structured_auxiliary'] = False
    elif damage == 'text_missing': raw.pop('comparison_review_text')
    elif damage == 'text_int': raw['comparison_review_text'] = 1
    else: raw['batch_only_lessons'] = False
    with pytest.raises(ValueError):
        apply_defaults({'research': raw})
    # The standalone reducer must validate policy before touching a capsule,
    # even when the CLI config normalizer was not used.
    with pytest.raises(ValueError, match='review text policy requires'):
        reducer.validate({}, config=raw, metric={})


@pytest.mark.parametrize('bad', [None, True, {'schema': 'sciel-review-text-policy/1', 'decisions': [], 'accepted': True}])
def test_malformed_host_policy_is_not_an_implicit_legacy_mode(bad):
    raw = configured()
    raw['comparison_review_text_policy'] = bad
    with pytest.raises(ValueError):
        apply_defaults({'research': raw})
    with pytest.raises(ValueError):
        reducer.validate({}, config=raw, metric={})


def reflection_fixture(kind):
    """A valid ordinary observation and an actually admissible lesson delta."""
    kb = KnowledgeBase()
    kb.record(experiment(1))
    ident = kb.experiments[0]['id']
    prior = dict(rule='The recorded local score increased.', polarity='positive',
                 scope='branch', branch='seed0', evidence_ids=[ident])
    old_id, = kb.add_lessons([prior], allowed_evidence_ids=[ident], writer_context={
        'phase': 'conclusion', 'reference': ident, 'raw_response_sha256': 'b' * 64})
    replacement = {**prior, 'rule': 'This saved local observation has a positive score delta.'}
    reply = dict(lessons=[] if kind in ('empty', 'retirement_only') else [replacement],
                 retire_ids=[old_id] if kind in ('replacement', 'retirement_only') else [],
                 comparison_reviews=[])
    raw = json.dumps(reply)
    records, lessons = kb.reflection_context(1)
    rows = [
        dict(event='research', phase='reflection_context', event_seq=1, through=1,
            experiment_ids=[r['id'] for r in records], lesson_ids=[x['id'] for x in lessons],
            omitted_lesson_count=len(kb.lessons)-len(lessons),
            context_sha256=reducer._digest(serialize_reflection(records, lessons)),
            comparison_review_source_ids=[], comparison_review_omitted_ids=[]),
        dict(event='research', phase='aux_start', event_seq=2, kind='reflection', reference=1,
            call_index=1, response_format=response_format('reflection', comparison_reviews=True),
            auxiliary_schema=identity(comparison_reviews=True)),
        dict(event='research', phase='aux_result', event_seq=3, kind='reflection', reference=1,
            call_index=1, response=raw, response_sha256=reducer._digest(raw),
            response_truncated=False, structured_response_usable=True,
            finish_reason='stop', refusal=None, cost=0.),
    ]
    after = KnowledgeBase()
    after.load_state(kb.state())
    applied = None
    if kind != 'retirement_only':
        applied = after.reflect(reply['lessons'], reply['retire_ids'], allowed_evidence_ids=[ident],
            writer_context={'phase': 'reflection', 'reference': 1, 'raw_response_sha256': reducer._digest(raw)})
    event = dict(event='research', phase='reflection', event_seq=4, through=1,
        status='parsed', error=None, error_detail=None, applied=applied,
        changed=after.lessons != kb.lessons, comparison_review_ids=[], comparison_reviews_changed=False)
    rows.append(event)
    return NS(kb=kb, after=after, records=records, lessons=lessons, rows=rows,
              event=event, raw=raw, reply=reply, before=copy.deepcopy(kb.state()))


@pytest.mark.parametrize('kind', ['addition', 'replacement'])
def test_reducer_rejects_real_lesson_effects_only_in_host_gated_mode(kind):
    f = reflection_fixture(kind)
    reviews = {}
    with pytest.raises(ValueError, match='review-only'):
        reducer._reflection(f.kb, reviews, f.rows, f.event, interval=1, config=configured())
    assert f.kb.state() == f.before and reviews == {}
    legacy = configured()
    legacy.pop('comparison_review_text_policy')
    reducer._reflection(f.kb, reviews, f.rows, f.event, interval=1, config=legacy)
    assert f.kb.state() == f.after.state()


def test_empty_historical_R_replays_unchanged_under_gated_policy():
    f = reflection_fixture('empty')
    before_rows = canonical_json(f.rows)
    reducer._reflection(f.kb, {}, f.rows, f.event, interval=1, config=configured())
    assert f.kb.state() == f.before
    assert canonical_json(f.rows) == before_rows


@pytest.mark.parametrize('historically_gated', [False, True])
def test_prompt_reconstruction_uses_original_effects_marker(historically_gated):
    from sciel.engine.reflection_context import prepare_reflection_context
    f = reflection_fixture('empty')
    historical_config = configured()
    if not historically_gated:
        historical_config.pop('comparison_review_text_policy')
    prepared = prepare_reflection_context(f.kb, {}, config=historical_config, interval=1, through=1)
    f.rows[0].update(prepared['context_event'])
    f.rows[1]['prompt_sha256'] = reducer._digest(prepared['prompt'])
    saved = canonical_json(f.rows)
    reducer._reflection(f.kb, {}, f.rows, f.event, interval=1, config=configured(), conclusions={})
    assert f.kb.state() == f.before and canonical_json(f.rows) == saved
    assert ('reflection_effects_policy' in f.rows[0]) is historically_gated
    # Neither dropping the new marker nor adding it to old bytes can change the
    # original request mode while keeping the bound prompt digest.
    if historically_gated:
        f.rows[0].pop('reflection_effects_policy')
    else:
        f.rows[0]['reflection_effects_policy'] = 'review_only/1'
    with pytest.raises(ValueError, match='complete prompt digest differs'):
        reducer._reflection(f.kb, {}, f.rows, f.event, interval=1, config=configured(), conclusions={})
    assert f.kb.state() == f.before


def test_marked_replay_reserves_review_only_instruction_inside_context_budget(monkeypatch):
    from sciel.engine import reflection_context as preparation
    from sciel.hypothesis import conclusion_feedback, rendering
    from sciel.hypothesis.prompts import comparison_review_source_block
    f = reflection_fixture('empty')
    source = comparison_review_source_block([], version=1,
        records=f.kb._evidence(), seen=f.kb._seen)
    # Put the real ordinary-review projection near the context cap, where
    # neglecting the additional instruction changes its bound metadata.
    maximum = (len(serialize_reflection(f.records, f.lessons))
        + len(conclusion_feedback.REFLECTION_REVIEW_RULES)
        + len(conclusion_feedback.REFLECTION_REVIEW_PREFIX) + 1
        + len(source) + len(preparation.REVIEW_ONLY_INSTRUCTION) + 512)
    monkeypatch.setattr(preparation, 'MAX_CONTEXT_CHARS', maximum)
    monkeypatch.setattr(rendering, 'MAX_CONTEXT_CHARS', maximum)
    prepared = preparation.prepare_reflection_context(f.kb, {},
        config=configured(), interval=1, through=1)
    assert prepared['context_event']['review_context_max_chars'] == 512
    f.rows[0].update(prepared['context_event'])
    f.rows[1]['prompt_sha256'] = reducer._digest(prepared['prompt'])
    saved = canonical_json(f.rows)
    reducer._reflection(f.kb, {}, f.rows, f.event, interval=1,
        config=configured(), conclusions={})
    assert f.kb.state() == f.before and canonical_json(f.rows) == saved


@pytest.mark.parametrize('marker', [None, True, 'unknown/1'])
def test_unknown_historical_effects_marker_is_not_ignored(marker):
    f = reflection_fixture('empty')
    f.rows[0]['reflection_effects_policy'] = marker
    with pytest.raises(ValueError, match='unsupported historical reflection effects policy'):
        reducer._reflection(f.kb, {}, f.rows, f.event, interval=1, config=configured())


@pytest.mark.parametrize('status', ['parsed', 'invalid'])
def test_resume_cannot_downgrade_marked_history_even_with_no_current_card(status):
    config = configured()
    config.pop('comparison_review_text_policy')
    controller = NS(config=config)
    rows = [dict(event='research', phase='reflection_context', reflection_effects_policy='review_only/1'),
            dict(event='research', phase='reflection', status=status)]
    with pytest.raises(ValueError, match='requires receiving review text policy'):
        resume.assert_authoring_journal(controller, rows)


@pytest.mark.parametrize('kind', ['addition', 'replacement'])
def test_resume_rejects_applied_lesson_R_even_without_retained_review_cards(monkeypatch, kind):
    f = reflection_fixture(kind)
    monkeypatch.setattr(resume, 'validate', lambda *args, **kwargs: {
        'knowledge': f.kb, 'comparison_reviews': {}})
    controller = NS(_external_history={'capsule': {}}, config=configured(), metric={},
        knowledge=f.after, _comparison_reviews={}, _retained_comparison_reviews=lambda: {},
        _review_groups=lambda: [])
    f.event['external_history_transition'] = resume.transition_binding(controller)
    before = canonical_json(f.after.state())
    with pytest.raises(ValueError, match='review-only'):
        resume.assert_authoring_journal(controller, f.rows)
    assert canonical_json(f.after.state()) == before
    # The exact same authenticated local authoring remains supported by legacy.
    controller.config.pop('comparison_review_text_policy')
    resume.assert_authoring_journal(controller, f.rows)


@pytest.mark.parametrize('kind', ['addition', 'replacement', 'retirement_only'])
def test_external_return_keeps_rejection_receipt_before_any_lesson_or_card_mutation(monkeypatch, kind):
    f = reflection_fixture(kind)
    job = object.__new__(ExternalReflectionJob)
    job._config = configured()
    plan = dict(run_id='inert-review-only', segment_id='inert-r1', interval=1,
        historical_through=1, model='inert',
        journal_prefix=''.join(canonical_json(row)+'\n' for row in f.rows[:2]))
    monkeypatch.setattr(job, 'prepare', lambda **kwargs: copy.deepcopy(plan))
    source = {'knowledge': f.kb, 'conclusions': {}, 'comparison_reviews': {}}
    prepared = {'records': f.records, 'lessons': f.lessons, 'comparison_review_groups': []}
    monkeypatch.setattr(job, '_preparation', lambda **kwargs: (source, 1, prepared))
    receipt = dict(text=f.raw, model='inert', cost=.01, prompt_tokens=1, completion_tokens=1,
                   finish_reason='stop', refusal=None)
    with pytest.raises(ExternalReflectionRejected) as rejected:
        job.admit(plan, receipt, manifest_sha256='a'*64, journal_prefix=plan['journal_prefix'])
    rows = [json.loads(line) for line in rejected.value.journal.splitlines()]
    result, event = rows[-2:]
    assert result['phase'] == 'aux_result' and result['response'] == f.raw and result['cost'] == .01
    assert event['phase'] == 'reflection' and event['status'] == 'parsed'
    assert event['error'] == 'ValueError' and 'review-only' in event['error_detail']
    assert event['applied'] is None and event['changed'] is False
    assert event['comparison_review_ids'] == [] and event['comparison_reviews_changed'] is False
    assert f.kb.state() == f.before and source['comparison_reviews'] == {}
