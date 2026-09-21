"""Existing stagnation detectors and bounded cells never fabricate task evidence."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from sciel.engine.research_cells_context import compact_context, merge_context, MAX_CHARS
from sciel.engine.research_tool_signals import snapshot_signals
from sciel.stats import OpportunityCostMonitor, PlateauMonitor


def card(ident='a', text='Observed an auxiliary diagnostic.'):
    return {'schema': 'sciel-research-check/3', 'replay_comparison': None, 'cell_id': ident * 64, 'revision_id': 'b' * 64,
        'question': 'Why has the search stalled?',
        'source': {'tool_id': 'c' * 64, 'invocation_id': 'd' * 64, 'receipt_id': 'e' * 64, 'candidate_id': 'p1', 'target_kind':'program',
            'target_source_sha256':'1'*64,'ancestry_parent_id':'p1', 'research_targets_sha256':'2'*64,
            'population_best_id':None,'population_best_source_sha256':None},
        'observations': {'status': 'ok', 'output_excerpt': text, 'resources': {'cpu_seconds': .1}},
        'interpretation': 'Unverified explanation.', 'next_check': 'Evaluate a normal mutation.',
        'limitations': ['This is not a canonical task score.']}


def replay_card():
    value = card()
    value['observations']['status'] = 'success'
    value['replay_comparison'] = {
        'schema': 'sciel-replay-feedback/1', 'tool_id': 'c' * 64,
        'policy': 'exact_observations', 'status': 'not_reproduced',
        'matched': False, 'measurement_reproduced': False,
        'reason': 'observations_differ', 'reference_receipt_id': 'f' * 64,
        'replay_receipt_id': 'e' * 64,
        'reference_observations_sha256': '1' * 64,
        'replay_observations_sha256': '2' * 64, 'binding_mismatches': [],
        'changed_paths': [{'path': '/elapsed_time', 'reference': '1.2', 'replay': '1.3'}],
        'changed_paths_omitted': 0,
    }
    return value


def test_failed_reproduction_remains_separate_from_successful_tool_execution():
    value = replay_card()
    result = compact_context([value])
    assert result['cards'] == [value] and result['omitted_count'] == 0
    assert result['cards'][0]['observations']['status'] == 'success'
    assert result['cards'][0]['replay_comparison']['matched'] is False
    assert result['cards'][0]['replay_comparison']['changed_paths'][0]['path'] == '/elapsed_time'


@pytest.mark.parametrize('field', ['tool_id', 'replay_receipt_id'])
def test_comparison_for_another_tool_or_invocation_cannot_enter_context(field):
    value = replay_card()
    value['replay_comparison'][field] = '9' * 64
    result = compact_context([value])
    assert result['cards'] == []
    assert result['omission_reasons'] == {'invalid_card': 1}


def test_old_card_schema_is_not_reinterpreted_as_comparison_aware():
    value = card()
    value['schema'] = 'sciel-research-check/2'
    del value['replay_comparison']
    assert compact_context([value])['omission_reasons'] == {'invalid_card': 1}


def test_pure_signals_preserve_detector_edges_and_distinguish_stale_shadow_segments():
    oc = OpportunityCostMonitor(threshold=.5, horizon=40, min_n=10)
    shadow = PlateauMonitor()
    for index in range(100):
        oc.observe(index == 0, phase=0)
        shadow.observe(False, phase=0)
    before = [deepcopy(oc.state()), deepcopy(shadow.state())]
    engine = SimpleNamespace(iteration_no=120, B_evals=100, _last_pop_improve=12,
        _last_pop_improve_iteration=18,
        cfg={'refine_stall': 8}, controller=SimpleNamespace(enabled=True, plateau_until={0: 130}),
        oc_monitors={0: oc}, oc_shift=False, plateau_monitors={0: shadow})
    options = dict(island=0, iteration_no=120, phase=0, mode='explore', effective_mode='explore',
        refine=False, forced_operator=True, stall=88, stalled=True, controller_plateau=True)
    current = snapshot_signals(engine, **options)
    stale = snapshot_signals(engine, **{**options, 'phase': 1})
    assert current['opportunity_cost']['detected'] is True
    assert current['opportunity_cost']['mode'] == current['betting_shadow']['mode'] == 'shadow'
    assert stale['opportunity_cost']['current_phase'] is False
    assert stale['opportunity_cost']['detected'] is stale['betting_shadow']['detected'] is None
    assert current['refinement'] == {'applied': False, 'forced_operator': True}
    assert current['controller_plateau']['active'] and current['population_stall']['elapsed_evaluations'] == 88
    assert [oc.state(), shadow.state()] == before  # Includes declared edge state.
    json.dumps(current, allow_nan=False)


def test_unstarted_forecast_is_unknown_and_disabled_detectors_stay_disabled():
    engine = SimpleNamespace(iteration_no=0, B_evals=0, _last_pop_improve=0,
        _last_pop_improve_iteration=0, cfg={},
        controller=SimpleNamespace(enabled=False, plateau_until={}), oc_monitors={0: OpportunityCostMonitor()},
        oc_shift=False, plateau_monitors=None)
    result = snapshot_signals(engine, island=0, iteration_no=0, phase=0, mode='base', effective_mode='base',
        refine=False, forced_operator=False, stall=0, stalled=False, controller_plateau=False)
    assert result['opportunity_cost']['forecast'] is None
    assert result['opportunity_cost']['detected'] is None
    assert result['betting_shadow']['enabled'] is False
    json.dumps(result, allow_nan=False)


def test_oversize_and_malformed_cards_are_reported_without_rejecting_the_collection():
    valid = card()
    malformed = {**card('f'), 'source': {'receipt_id': '/not-an-evidence-contract'}}
    result = compact_context([card('c', 'x' * 20000), valid, malformed])
    assert result['cards'] == [valid] and result['omitted_count'] == 2
    assert result['omission_reasons'] == {'character_limit': 1, 'invalid_card': 1}
    assert len(json.dumps(result)) < MAX_CHARS
    assert compact_context(result) == result


def test_current_revision_replaces_prior_cell_card_without_merging_unverified_claims():
    prior = card('a', 'Old observation')
    current = {**card('a', 'New observation'), 'revision_id': 'f' * 64}
    other = card('b')
    result = merge_context([prior, other], [current])
    assert result['cards'] == [current, other]
    assert prior['observations']['output_excerpt'] == 'Old observation'
    assert merge_context([], [prior, current])['cards'] == [current]


def test_configured_merge_limit_prioritizes_the_new_cell_over_retained_observations():
    prior = card('a', 'Previous check. ' + 'x' * 600)
    fresh = card('f', 'Fresh check. ' + 'y' * 600)
    before = deepcopy([prior, fresh])
    result = merge_context([prior], [fresh], max_chars=2048)
    assert result['cards'] == [fresh]
    assert result['omission_reasons'] == {'character_limit': 1}
    assert len(json.dumps(result, ensure_ascii=True)) <= 2048
    assert [prior, fresh] == before


@pytest.mark.parametrize('research_enabled', [False, True])
def test_combined_cell_delivery_respects_small_configured_budget_with_newest_first(
        tmp_path, monkeypatch, research_enabled):
    from sciel.research_tools.controller import ResearchTools
    from sciel.hypothesis.rendering import decode_evidence
    from test_parallel_engine import ResearchProvider
    import test_research_tool_engine as fixtures

    actual_formatter = ResearchTools.format_context
    saved_cards = {}
    monkeypatch.setitem(fixtures.TOOL_CONFIG, 'max_context_chars', 4096)
    provider = ResearchProvider(concurrent=1) if research_enabled else None
    build, provider, evaluations, _ = fixtures.setup_tools(tmp_path, monkeypatch,
        workers=1, max_iterations=1, provider=provider,
        research={'enabled': research_enabled, 'reflection_interval': 1, 'max_aux_calls': None})
    original_saved = fixtures.FakeTools.saved_observation
    def prior_context(self, snapshot):
        if 'prior' not in saved_cards:
            episode = original_saved(self, snapshot, label='older-check',
                observations={'finding': 'RETAINED_OLDER_CHECK ' + 'x' * 2000}, commit=True)
            saved_cards['prior'] = deepcopy(episode['cards'][0])
        return [saved_cards['prior']]
    monkeypatch.setattr(fixtures.FakeTools, 'context', prior_context)
    monkeypatch.setattr(fixtures.FakeTools, 'format_context', staticmethod(actual_formatter))
    def with_new_cell(self, snapshot, *, label, observations, **kwargs):
        # The current card and all invocation/source identifiers come from
        # one real immutable fixture receipt, not a replacement forged card.
        episode = original_saved(self, snapshot, label=label,
            observations={'finding': 'FRESH_NEW_CELL_CHECK ' + 'y' * 2000}, **kwargs)
        saved_cards['fresh'] = deepcopy(episode['cards'][0])
        return episode
    monkeypatch.setattr(fixtures.FakeTools, 'saved_observation', with_new_cell)

    result = build().run()
    assert result['B_evals'] == 1 and len(evaluations) == 2
    fresh = {**saved_cards['fresh'], 'schema': 'sciel-research-check/5',
        'comparison_assessment': {'schema': 'sciel-research-comparison-assessment/1',
            'receipt_id': saved_cards['fresh']['source']['receipt_id'], 'proposal_id': None,
            'status': 'unavailable', 'assessment': None}}
    assert fresh['source']['receipt_id'] != saved_cards['prior']['source']['receipt_id']
    assert fresh['cell_id'] != saved_cards['prior']['cell_id']
    prompts = [prompt for prompt in provider.calls
               if not prompt.startswith('SCIEL_RESEARCH_TOOL_EPISODE')]
    if research_enabled:
        prompt = next(prompt for prompt in prompts if 'RESEARCH_CONTEXT_JSON:\n' in prompt)
        context = decode_evidence(json.JSONDecoder().raw_decode(
            prompt.split('RESEARCH_CONTEXT_JSON:\n', 1)[1])[0])
        delivery = context['research_cells_context']
        assert delivery['cards'] == [fresh]
        assert delivery['omission_reasons'] == {'character_limit': 1}
        assert len(json.dumps(delivery, ensure_ascii=True)) <= 4096
        assert context['research_cells_reservation']['reserved_chars'] == 4096
    else:
        prompt = prompts[0]
        delivered = actual_formatter([fresh])
        assert delivered in prompt and len(delivered) <= 4096  # Includes heading and scope wrapper.
        assert 'RESEARCH CELL OBSERVATIONS' not in prompt  # No unbudgeted duplicate heading.
    assert 'FRESH_NEW_CELL_CHECK' in prompt and 'RETAINED_OLDER_CHECK' not in prompt


def test_prior_cells_reach_generation_on_steps_without_a_new_tool_episode(tmp_path, monkeypatch):
    from test_research_tool_engine import FakeTools, setup_tools
    build, provider, evaluations, _ = setup_tools(tmp_path, monkeypatch, workers=1, max_iterations=1)
    monkeypatch.setattr(FakeTools, 'prepare', lambda self, snapshot, *, signals: None)
    saved = []
    def prior_context(self, snapshot):
        if not saved:
            saved.extend(self.saved_observation(snapshot, label='earlier-diagnostic',
                observations={'finding': 'Observed an auxiliary diagnostic.'}, commit=True)['cards'])
        return saved
    monkeypatch.setattr(FakeTools, 'context', prior_context)
    engine = build()
    result = engine.run()
    assert len(provider.calls) == 1 and len(evaluations) == 2 and result['B_evals'] == 1
    assert 'RESEARCH CELL OBSERVATIONS' in provider.calls[0]
    assert 'Observed an auxiliary diagnostic.' in provider.calls[0]
    assert saved[0]['source']['receipt_id'] in provider.calls[0]
    assert not FakeTools.invocations


def test_injected_oversize_memory_still_reports_emergency_cell_omission(tmp_path, monkeypatch):
    from test_parallel_engine import ResearchProvider
    from test_research_tool_engine import FakeTools, setup_tools
    from sciel.engine.research import ResearchController
    from sciel.hypothesis.rendering import decode_evidence
    provider = ResearchProvider(concurrent=1)
    build, _, evaluations, _ = setup_tools(tmp_path, monkeypatch, workers=1, max_iterations=1,
        provider=provider, research={'enabled': True, 'reflection_interval': 1,
                                    'max_aux_calls': None, 'comparison_reviews': True,
                                    'structured_auxiliary': True, 'conclusion_source_refs': True})
    monkeypatch.setattr(FakeTools, 'prepare', lambda self, snapshot, *, signals: None)
    saved = []
    def prior_context(self, snapshot):
        if not saved:
            saved.extend(self.saved_observation(snapshot, label='large-earlier-diagnostic',
                observations={'finding': 'x' * 8000}, commit=True)['cards'])
        return saved
    monkeypatch.setattr(FakeTools, 'context', prior_context)
    original = ResearchController.context
    def with_full_memory(self, *args, **kwargs):
        context = original(self, *args, **kwargs)
        context['lesson_texts'] = ['Existing selected memory. ' + 'x' * 28900]
        return context
    monkeypatch.setattr(ResearchController, 'context', with_full_memory)
    result = build().run()
    assert result['B_evals'] == 1 and len(evaluations) == 2
    prompt = next(prompt for prompt in provider.calls if 'RESEARCH_CONTEXT_JSON:\n' in prompt)
    context = decode_evidence(json.JSONDecoder().raw_decode(prompt.split('RESEARCH_CONTEXT_JSON:\n', 1)[1])[0])
    delivery = context['research_cells_context']
    assert delivery['cards'] == [] and delivery['omission_reasons']['shared_memory_limit'] == 1


def test_reservation_selects_less_ordinary_memory_and_preserves_fresh_check_in_hypothesis(monkeypatch):
    from test_compact_lesson_delivery import controller, delivery_fixture
    from sciel.engine.types import Snapshot
    from sciel.engine.prompt.build import build
    from sciel.hypothesis.rendering import decode_evidence
    instance, db, parent, events = controller(delivery_fixture(), compact=False)
    ranked = [{**deepcopy(instance.knowledge.lessons[0]), 'id': 'l' + str(index)} for index in range(3)]
    monkeypatch.setattr(instance.knowledge, 'retrieve', lambda **kwargs: ranked)
    monkeypatch.setattr(instance, 'lesson_text', lambda lesson: lesson['id'] + ' existing evidence ' + 'x' * 14400)
    unreserved = instance.context(db, parent, 0, 10)
    reserved = instance.context(db, parent, 0, 10, reserve_aux_chars=10000)
    assert len(unreserved['lessons']) == 2 and len(reserved['lessons']) == 1
    assert 'research_cells_reservation' not in unreserved
    assert reserved['lesson_presentation']['reserved_aux_chars'] == 10000
    fresh = card(text='Fresh execution observation. ' + 'x' * 8000)
    reserved['research_cells_context'] = compact_context([fresh])
    snapshot = Snapshot(0, 'diff', 'fixture', 9, parent.id, parent.code, parent.score,
                        research_context=reserved, kb_lessons=reserved['lesson_texts'])
    ident, prompt = instance.begin(snapshot, 0, 'diff', build('diff', snapshot))
    actual = decode_evidence(json.JSONDecoder().raw_decode(prompt.split('RESEARCH_CONTEXT_JSON:\n', 1)[1])[0])
    assert actual['research_cells_context']['cards'] == [fresh]
    assert actual['research_cells_context']['omitted_count'] == 0
    assert actual['research_cells_reservation']['reserved_chars'] == 10000
    assert actual['parent_score'] == parent.score == instance.pending[ident]['parent_score']
    preparation = next(event for event in events if event['phase'] == 'context_prepared')
    assert preparation['research_cells_reservation']['reserved_chars'] == 10000
    assert preparation['research_cells_context']['omitted_count'] == 0


def test_reservation_refuses_experimental_import_budgets_before_context_delivery():
    import pytest
    from test_compact_lesson_delivery import controller, delivery_fixture
    instance, db, parent, events = controller(delivery_fixture(), compact=False)
    instance._external_component_digest = 'a' * 64
    with pytest.raises(ValueError, match='incompatible with imported paired'):
        instance.context(db, parent, 0, 1, reserve_aux_chars=10000)
    assert not events
