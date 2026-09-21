"""Pure whole-card delivery; synthetic approvals do not accept any paid prose."""
import copy
import json

import pytest

from sciel.hypothesis import generated_paired_review as review
from sciel.knowledge_base import generated_paired_review_context as context
from sciel.knowledge_base.catalogs import canonical_json, canonical_sha
from test_generated_paired_history import capsule as synthetic_capsule


POLICY_SHA = 'f' * 64
WRITER = dict(phase='reflection', reference=1, raw_response_sha256='d' * 64)


@pytest.fixture
def component():
    return synthetic_capsule.__wrapped__()


def proposals(component):
    result = []
    for group in component['comparisons']:
        result.append(dict(comparison_id=group['id'], comparison_sha256=canonical_sha(group),
            claims=[dict(fact_id='quality:stored_outcome', value=group['derived']['outcome'])],
            conjecture='SYNTHETIC_CONJECTURE: an untested explanation.',
            limitations='SYNTHETIC_LIMITATIONS: this fixture is not an execution claim.',
            next_hypothesis=dict(schema='sciel-comparison-target/1',
                base_source_sha256=group['comparator_source_sha256'],
                comparator_source_sha256=group['comparator_source_sha256'],
                change='SYNTHETIC_TARGET: a bounded future edit, not implemented here.',
                quality_gate='paired_raw_exact_zero_loss_strict_v1', diagnostics=[]),
            next_hypothesis_reason='SYNTHETIC_REASON: this is only a future test proposal.'))
    return result


@pytest.fixture
def cards(component):
    return review.build_cards(proposals(component), component, WRITER)


def decision(card, status='accepted', *, policy_sha256=POLICY_SHA):
    return dict(schema='sciel-review-text-decision/1', scope='comparison_review_text',
        decision=status, card_sha256=canonical_sha(card),
        raw_response_sha256=card['writer']['raw_response_sha256'],
        comparison_id=card['comparison_id'], comparison_sha256=card['comparison_sha256'],
        comparator_source_sha256=card['comparator_source_sha256'],
        challenger_source_sha256=card['challenger_source_sha256'], policy_sha256=policy_sha256,
        audit_sha256='a' * 64, criteria_sha256='b' * 64, issuer='Synthetic fixture host')


def policy(cards, status='accepted'):
    return dict(schema='sciel-review-text-policy/1', decisions=[] if status == 'pending' else [
        decision(card, 'accepted' if status == 'stale' else status,
                 policy_sha256='e' * 64 if status == 'stale' else POLICY_SHA)
        for card in cards.values()])


def shown(component, cards, **kwargs):
    options = dict(branch=component['root_parent']['id'], text_policy=policy(cards),
                   policy_sha256=POLICY_SHA)
    options.update(kwargs)
    return context.build_context(component, cards, **options)


def common(view):
    view = copy.deepcopy(view)
    for key in ('text_enabled', 'rendered_chars', 'context_sha256'):
        view.pop(key)
    for card in view['cards']:
        card.pop('unverified_interpretation')
        card.pop('interpretation_omission')
    return view


def test_three_complete_cards_fit_and_bind_authoritative_facts_without_repeating_them(component, cards):
    original = canonical_json([component, cards])
    view = shown(component, cards)
    assert view['presented_count'] == 3 and view['omitted_count'] == 0
    assert len(context.block(view)) == view['rendered_chars'] <= view['reserved_chars'] <= 8000
    assert view['source_component_sha256'] == canonical_sha(component)
    for projected in view['cards']:
        stored = cards[projected['comparison_id']]
        assert projected['branch'] == stored['branch'] != view['branch']
        assert projected['stored_card_sha256'] == canonical_sha(stored)
        assert projected['catalog_sha256'] == canonical_sha(stored['host_facts'])
        assert projected['writer'] == stored['writer']
        assert projected['unverified_interpretation'] == stored['unverified_interpretation']
        assert 'host_facts' not in projected and 'authored_forecasts' not in projected
    assert canonical_json([component, cards]) == original
    view['cards'][0]['writer']['raw_response_sha256'] = '0' * 64
    assert canonical_json([component, cards]) == original


@pytest.mark.parametrize('status', ['accepted', 'pending', 'rejected', 'stale'])
def test_full_block_mask_and_full_on_reservation_in_every_eligibility_state(component, cards, status):
    registry = policy(cards, status)
    on, off = [shown(component, cards, text_policy=registry, text_enabled=mode) for mode in (True, False)]
    assert common(on) == common(off)
    assert on['reserved_chars'] == off['reserved_chars']
    for mode, view in zip((True, False), (on, off), strict=True):
        assert view['presented_count'] == 3
        for projected in view['cards']:
            assert projected['text_eligibility']['status'] == status
            assert (projected['unverified_interpretation'] is not None) is (status == 'accepted' and mode)
        wire = context.block(view)
        for marker in ('SYNTHETIC_CONJECTURE', 'SYNTHETIC_LIMITATIONS', 'SYNTHETIC_TARGET', 'SYNTHETIC_REASON'):
            assert (marker in wire) is (status == 'accepted' and mode)


def test_pending_status_does_not_release_space_for_additional_cards(component, cards):
    accepted = shown(component, cards)
    views = [shown(component, cards, max_chars=accepted['reserved_chars'] - 1,
                   text_policy=policy(cards, status), text_enabled=mode)
             for status in ('accepted', 'pending', 'rejected', 'stale') for mode in (True, False)]
    expected = [card['comparison_id'] for card in views[0]['cards']]
    assert 0 < len(expected) < 3
    for view in views:
        assert [card['comparison_id'] for card in view['cards']] == expected
        assert view['reserved_chars'] == views[0]['reserved_chars']
        assert view['omitted_count'] == 3 - len(expected)
        assert all(item['reason'] == 'memory_char_limit' for item in view['omitted'])
        assert len(context.block(view)) <= view['reserved_chars'] <= view['max_chars']


@pytest.mark.parametrize('damage', ['text', 'catalog', 'writer', 'source_component'])
@pytest.mark.parametrize('options', [dict(enabled=False), dict(max_chars=0)])
def test_invalid_stored_card_is_rejected_even_if_never_presented(component, cards, damage, options):
    bad = copy.deepcopy(cards)
    card = next(iter(bad.values()))
    if damage == 'text':
        card['unverified_interpretation']['claims'][0]['value'] = 'invented'
    elif damage == 'catalog':
        card['host_facts']['facts']['quality:stored_outcome'] = 'invented'
    elif damage == 'writer':
        card['writer']['reference'] = True
    else:
        card['source_component_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        shown(component, bad, **options)


@pytest.mark.parametrize('damage', ['text', 'reserve', 'enabled', 'branch', 'omissions'])
def test_self_rehashed_snapshot_cannot_bypass_authoritative_context(component, cards, damage):
    view = shown(component, cards, text_enabled=False)
    if damage == 'text':
        ident = view['cards'][0]['comparison_id']
        view['cards'][0]['unverified_interpretation'] = copy.deepcopy(cards[ident]['unverified_interpretation'])
    elif damage == 'reserve':
        view['reserved_chars'] = view['rendered_chars']
    elif damage == 'enabled':
        view['text_enabled'] = True
    elif damage == 'branch':
        view['branch'] = 'another-root'
    else:
        view['omitted_count'] += 1
    view['context_sha256'] = canonical_sha({k: v for k, v in view.items() if k != 'context_sha256'})
    with pytest.raises(ValueError, match='authoritative'):
        context.validate_context(view, component, cards, branch=component['root_parent']['id'],
            text_policy=policy(cards), policy_sha256=POLICY_SHA, text_enabled=False)


def test_replaced_reason_requires_fresh_exact_card_decision(component, cards):
    old = policy(cards)
    proposals_after = proposals(component)
    proposals_after[0]['next_hypothesis_reason'] = 'A different unverified synthetic rationale.'
    updated = review.build_cards(proposals_after, component, WRITER)
    view = shown(component, updated, text_policy=old)
    by_id = {card['comparison_id']: card for card in view['cards']}
    revised = by_id[proposals_after[0]['comparison_id']]
    assert revised['text_eligibility']['status'] == 'pending'
    assert revised['unverified_interpretation'] is None


def test_redundant_decision_binding_fails_even_when_text_disabled(component, cards):
    registry = policy(cards)
    registry['decisions'][0]['raw_response_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='binding differs'):
        shown(component, cards, text_policy=registry, text_enabled=False)


def test_disabled_and_zero_budget_keep_explicit_omissions_and_validate_none(component, cards):
    disabled = shown(component, cards, enabled=False)
    assert disabled['presented_count'] == 0 and disabled['omitted_count'] == 3
    assert len(disabled['omitted']) == 3 and disabled['omission_details_omitted_count'] == 0
    assert {item['reason'] for item in disabled['omitted']} == {'disabled'}
    assert shown(component, cards, max_chars=0) is None and context.block(None) == ''
    assert context.validate_context(None, component, cards, branch='p1', max_chars=0,
        text_policy=policy(cards), policy_sha256=POLICY_SHA) is None
    with pytest.raises(ValueError, match='authoritative'):
        context.validate_context(None, component, cards, branch='p1',
            text_policy=policy(cards), policy_sha256=POLICY_SHA)


@pytest.mark.parametrize('options', [dict(branch='other'), dict(branch=True), dict(max_chars=True),
    dict(max_chars=-1), dict(max_chars=8001), dict(enabled=1), dict(text_enabled=0),
    dict(text_policy=None), dict(policy_sha256=None)])
def test_delivery_parameters_are_explicit_and_strict(component, cards, options):
    with pytest.raises(ValueError):
        shown(component, cards, **options)


def test_actual_partial_study_keeps_all_three_synthetic_reviews_in_eight_thousand_chars():
    from test_generated_paired_integration import pilot_artifacts, component as archived_component
    actual = archived_component.__wrapped__(pilot_artifacts.__wrapped__())
    cards = review.build_cards(proposals(actual), actual, WRITER)
    on, off = [shown(actual, cards, text_enabled=mode) for mode in (True, False)]
    assert on['presented_count'] == off['presented_count'] == 3
    assert on['reserved_chars'] == off['reserved_chars'] <= 8000
    assert common(on) == common(off)
    assert all(cards[row['comparison_id']]['origin_experiment_id'] is None for row in on['cards'])
    assert {row['branch'] for row in on['cards']} == {'on_off', 'on_control', 'off_control'}
    assert context.validate_context(on, actual, cards, branch='p1', text_policy=policy(cards),
        policy_sha256=POLICY_SHA) == on


@pytest.fixture
def reviewed_memory(component):
    from types import SimpleNamespace
    from sciel.engine.generated_paired_reflection import GeneratedPairedReflectionJob
    from sciel.engine.external_research_v2 import policy_identity
    from sciel.knowledge_base import KnowledgeBase
    from test_research_knowledge import experiment
    config = dict(enabled=True, restart_predictions=True, structured_auxiliary=True,
        comparison_reviews=True, comparison_review_version=3, inject_lessons=True,
        batch_only_lessons=True, comparison_review_text=True, max_aux_calls=1,
        model='fixture/model', external_component_sha256=canonical_sha(component),
        comparison_review_text_policy=policy({}, 'pending'))
    job = GeneratedPairedReflectionJob(component, expected_sha256=canonical_sha(component), config=config)
    plan = job.prepare(run_id='synthetic-R-run', segment_id='synthetic-R-segment')
    response = dict(text=json.dumps(dict(lessons=[], retire_ids=[], comparison_reviews=proposals(component))),
        model=config['model'], cost=0., prompt_tokens=1, completion_tokens=1, finish_reason='stop', refusal=None)
    admitted = job.admit(plan, response, manifest_sha256='c' * 64, journal_prefix=plan['journal_prefix'])
    sidecar = admitted['extended_capsule']
    kb = KnowledgeBase()
    kb.import_generated_paired_history(component, expected_sha256=canonical_sha(component), metric=component['metric'])
    kb.import_generated_paired_review(sidecar, expected_sha256=canonical_sha(sidecar))
    # A separately explicit local ordinary fixture owns through=1. The imported
    # study/review never manufactures this record or increments its watermark.
    kb.record(experiment(1, branch='p1', metric=component['metric'],
        parent_cases={'combined_score': .5}, evaluation=dict(ok=True, score=.6,
            cases={'combined_score': .6}, reached='full', confirmed=False, raw_first_score=.6)))
    config['external_component_review_sha256'] = canonical_sha(sidecar)
    config['comparison_review_text_policy'] = dict(schema='sciel-review-text-policy/1', decisions=[
        decision(card, policy_sha256=canonical_sha(policy_identity()))
        for card in sidecar['review_cards'].values()])
    return SimpleNamespace(kb=kb, config=config, sidecar=sidecar, component=component)


@pytest.mark.parametrize('selection_epoch', [None, 0], ids=['archive', 'live'])
def test_ordinary_reflection_receives_masked_sidecar_with_identical_reserved_other_memory(
        reviewed_memory, monkeypatch, selection_epoch):
    from sciel.engine.reflection_context import prepare_reflection_context
    from sciel.knowledge_base.generated_paired_context import block as component_block
    f = reviewed_memory
    before = f.kb.state()
    budgets = []
    selector = 'legacy_reflection_context' if selection_epoch is None else 'reflection_context'
    original = getattr(f.kb, selector)
    def capture(*args, **kwargs):
        budgets.append(kwargs['max_chars'])
        return original(*args, **kwargs)
    monkeypatch.setattr(f.kb, selector, capture)
    preparations = [prepare_reflection_context(f.kb, {}, config={**f.config, 'comparison_review_text': mode},
                    interval=1, through=1, selection_epoch=selection_epoch) for mode in (True, False)]
    on, off = preparations
    views = [p['context_event']['generated_paired_reviews'] for p in preparations]
    assert views[0]['presented_count'] == views[1]['presented_count'] == 3
    assert common(views[0]) == common(views[1])
    assert len(budgets) == 2
    assert budgets[0] == budgets[1] == 90000 - len(component_block(
        on['context_event']['generated_paired_history'])) - views[0]['reserved_chars']
    assert on['prompt'].replace(context.block(views[0]), '', 1) == off['prompt'].replace(context.block(views[1]), '', 1)
    assert on['records'] == off['records'] and on['lessons'] == off['lessons'] == []
    assert on['comparison_review_groups'] == off['comparison_review_groups'] == []
    assert on['context_event']['experiment_ids'] == off['context_event']['experiment_ids'] == ['r000001a1']
    assert on['context_event']['review_context_max_chars'] == off['context_event']['review_context_max_chars']
    assert f.kb.state() == before and len(f.kb.experiments) == 1 and f.kb.comparisons == []


@pytest.mark.parametrize('damage', ['digest_missing', 'digest_wrong', 'policy_missing', 'raw_writer'])
def test_reflection_sidecar_binding_rejects_before_projection(reviewed_memory, damage):
    from sciel.engine.reflection_context import prepare_reflection_context
    f = reviewed_memory
    config = copy.deepcopy(f.config)
    if damage == 'digest_missing':
        config.pop('external_component_review_sha256')
    elif damage == 'digest_wrong':
        config['external_component_review_sha256'] = '0' * 64
    elif damage == 'policy_missing':
        config.pop('comparison_review_text_policy')
    else:
        f.kb._generated_paired_review['response']['text'] += ' '
    before = f.kb.state()
    with pytest.raises(ValueError):
        prepare_reflection_context(f.kb, {}, config=config, interval=1, through=1)
    assert f.kb.state() == before


def test_historical_reflection_replay_ignores_receiving_annex_configuration():
    from sciel.engine import external_research_v2 as reducer
    from sciel.engine.reflection_context import prepare_reflection_context
    from test_review_eligibility_history import configured, reflection_fixture
    f = reflection_fixture('empty')
    config = configured()
    prepared = prepare_reflection_context(f.kb, {}, config=config, interval=1, through=1)
    f.rows[0].update(prepared['context_event'])
    f.rows[1]['prompt_sha256'] = reducer._digest(prepared['prompt'])
    original = canonical_json(f.rows)
    receiver = {**config, 'external_component_sha256': 'a' * 64,
                'external_component_review_sha256': 'b' * 64}
    reducer._reflection(f.kb, {}, f.rows, f.event, interval=1, config=receiver, conclusions={})
    assert canonical_json(f.rows) == original and f.kb.state() == f.before


@pytest.mark.parametrize('field', ['generated_paired_history', 'generated_paired_reviews'])
def test_legacy_reducer_rejects_annex_aware_history_even_without_prompt_reconstruction(field):
    from sciel.engine import external_research_v2 as reducer
    from test_review_eligibility_history import configured, reflection_fixture
    f = reflection_fixture('empty')
    f.rows[0][field] = None  # Presence is a new envelope, not an old empty marker.
    with pytest.raises(ValueError, match='annex-aware'):
        reducer._reflection(f.kb, {}, f.rows, f.event, interval=1, config=configured())
    assert f.kb.state() == f.before
