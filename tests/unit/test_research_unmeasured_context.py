"""Unmeasured decisions remain auditable, not shared experimental memory.

Real controller/lifecycle/artifacts; fake provider and executor leaves only.
"""
from copy import deepcopy
import json

import pytest

from sciel.engine.research_cells_context import compact_context
from sciel.research_tools.controller import RESULTS_VIEW_MARKER
from sciel.research_tools.research_steps import new_state
from test_research_tool_controller import IO, LLM, prepare, snapshot
from test_research_tool_result_flow import result
from test_sequential_research_episode import build, decision, finish

UNMEASURED = 'UNMEASURED_PRIVATE_DECISION: planning ended without running an experiment.'


def close_without_measurement(tools):
    current = snapshot()
    episode = tools.execute(prepare(tools, current), current, LLM([finish(
        conclusion='inconclusive', reason=UNMEASURED)]), io=IO())
    tools.commit(episode)
    return episode


def test_unmeasured_closure_excluded_from_other_cells_and_ordinary_but_own_record_survives(tmp_path):
    tools, backend, _, _ = build(tmp_path, [], shared_observations=True)
    episode = close_without_measurement(tools)
    cell_id = episode['plan']['cell_id']
    state = deepcopy(tools.state())
    for resumed in (False, True):
        if resumed:
            tools.load_state(deepcopy(state))
        assert tools.context(snapshot(1, 2)) == []
        assert tools.context(snapshot(0, 2)) == []  # Same island is not the same research cell.
        own = tools.lifecycle.delivery_context([], recipient_cell_id=cell_id)
        assert own['cards'][0]['decision']['interpretation'] == UNMEASURED
        assert tools.lifecycle.shared_observations(tools.library, 1, tools.task_identity)['entries'] == []
        assert tools.state() == state
    assert backend.calls == []


@pytest.mark.parametrize('budget', [8000, 64000])
def test_actual_next_planner_filters_frozen_foreign_card_and_keeps_own_history(tmp_path, budget):
    tools, _, _, _ = build(tmp_path, [], shared_observations=True, max_prompt_chars=budget)
    episode = close_without_measurement(tools)
    card = deepcopy(next(iter(tools.lifecycle._state['cards'].values())))
    current = snapshot(1, 2)
    current.research_cells_context = compact_context([card])  # A previously frozen V44-style context.
    before = deepcopy(current.research_cells_context)
    plan = prepare(tools, current)
    prompt = tools._prompt(plan, current, [], [], 0, research_state=new_state())
    assert UNMEASURED not in prompt
    view = json.JSONDecoder().raw_decode(prompt.split(RESULTS_VIEW_MARKER, 1)[1])[0]
    assert view['cards'] == [] and view['omission_reasons']['unmeasured_interpretation'] == 1
    assert len(prompt) <= budget and current.research_cells_context == before
    own = tools.lifecycle.delivery_context(before, recipient_cell_id=episode['plan']['cell_id'])
    assert own['cards'] == [card]


@pytest.mark.parametrize('failed', [False, True])
def test_measured_negative_and_invoked_failure_are_still_shared(tmp_path, failed):
    output = result(status='tool_error', error='Observed malformed response') if failed else result({'gain':0, 'seed':9})
    tools, backend, _, _ = build(tmp_path, [output], shared_observations=True)
    current = snapshot()
    episode = tools.execute(prepare(tools, current), current, LLM([decision(), finish(
        conclusion='inconclusive' if failed else 'refuted', reason='Measured failure or no gain.')]), io=IO())
    tools.commit(episode)
    tools.load_state(deepcopy(tools.state()))
    card, = tools.context(snapshot(1, 2))
    assert card['source']['receipt_id'] == episode['receipt_ids'][0]
    assert card['observations']['status'] == ('tool_error' if failed else 'success')
    assert card['decision']['interpretation'] == 'Measured failure or no gain.'
    shared = tools.lifecycle.shared_observations(tools.library, 1, tools.task_identity)
    assert shared['total_checks'] == 1 and shared['entries'][0]['receipt_id'] == episode['receipt_ids'][0]
    assert len(backend.calls) == 1


def test_finish_citing_previous_actual_measurement_remains_shareable(tmp_path):
    tools, backend, _, _ = build(tmp_path, [result({'gain':0})], shared_observations=True,
                               yield_after_success=True)
    current = snapshot()
    first = tools.execute(prepare(tools, current), current, LLM([decision()]), io=IO())
    tools.commit(first)
    later = snapshot(iteration=1)
    second = tools.execute(prepare(tools, later), later, LLM([finish(
        reason='PRIOR_MEASURED_NEGATIVE: the previous measured check found no gain.')]), io=IO())
    tools.commit(second)
    assert second['cards'][0]['source']['receipt_id'] is None
    for _ in range(2):
        measurement, delivered = tools.context(snapshot(1, 2))
        assert measurement['source'] == first['cards'][0]['source']
        assert measurement['observations'] == first['cards'][0]['observations']
        assert delivered['source']['receipt_id'] is None
        assert delivered['decision']['receipt_ids'] == first['receipt_ids']
        assert 'PRIOR_MEASURED_NEGATIVE' in delivered['decision']['interpretation']
        tools.load_state(deepcopy(tools.state()))
    assert len(backend.calls) == 1


def test_fresh_uncommitted_no_measurement_card_is_filtered_without_losing_decision(tmp_path):
    tools, _, _, _ = build(tmp_path, [])
    current = snapshot()
    episode = tools.execute(prepare(tools, current), current, LLM([finish(
        conclusion='inconclusive', reason=UNMEASURED)]), io=IO())
    before = deepcopy(episode)
    delivered = tools.lifecycle.delivery_context(episode['cards'], episode=episode)
    assert delivered['cards'] == []
    assert delivered['omission_reasons'] == {'unmeasured_interpretation': 1}
    assert episode == before
    tools.commit(episode)
    assert next(iter(tools.lifecycle._state['cards'].values()))['decision']['interpretation'] == UNMEASURED


def test_foreign_card_cannot_forge_receipts_or_mutate_saved_decision(tmp_path):
    tools, _, _, _ = build(tmp_path, [], shared_observations=True)
    episode = close_without_measurement(tools)
    card = deepcopy(next(iter(tools.lifecycle._state['cards'].values())))
    card['decision']['receipt_ids'] = ['f' * 64]
    with pytest.raises(ValueError, match='immutable proposal'):
        tools.lifecycle.delivery_context([card])
    # Warm index must not make a later artifact mutation safe to ignore.
    tools.context(snapshot(1, 2))
    path = tools.path / (episode['terminal_decision_proposal_id'] + '.json')
    path.write_bytes(path.read_bytes() + b' ')
    with pytest.raises(ValueError, match='changed after authentication'):
        tools.context(snapshot(1, 2))


def test_canonical_only_citation_is_preserved_without_relabeling_as_tool_measurement(tmp_path):
    from test_research_canonical_citations import fixture, designed
    tools, backend, _, _, _, ids, episode, _ = fixture(tmp_path, status='eval_failed')
    tools.commit(episode)
    foreign = snapshot(1, 1)
    close = designed(finish(conclusion='inconclusive', reason='CANONICAL_FAILURE: repair the saved interface error.'),
                     kind='closure', budget_action='none', prior_receipt_ids=[ids['canonical']])
    done = tools.execute(prepare(tools, foreign), foreign, LLM([close]), io=IO())
    assert done['cards'][0]['source']['receipt_id'] is None
    # Even before commit, the referenced earlier canonical result is authenticated.
    current = tools.lifecycle.delivery_context(done['cards'], episode=done)
    assert next(c for c in current['cards'] if c.get('decision'))['decision']['interpretation'].startswith('CANONICAL_FAILURE')
    assert any(c['source']['receipt_id'] == episode['receipt_ids'][0] for c in current['cards'])
    tools.commit(done)
    tools.load_state(deepcopy(tools.state()), expected_run_id='fixture-run')
    cards = tools.context(snapshot(2, 2))
    terminal = next(c for c in cards if c['cell_id'] == done['plan']['cell_id'])
    assert terminal['observations']['status'] == 'not_invoked_this_episode'
    assert terminal['decision']['receipt_ids'] == []  # Canonical citation is a separate evidence type.
    assert len(backend.calls) == 1


def test_closure_citing_foreign_measured_receipt_remains_shareable(tmp_path):
    from test_research_canonical_citations import designed
    tools, backend, _, _ = build(tmp_path, [result({'gain':0})], shared_observations=True,
        experimental_protocol=True, yield_after_success=True, share_between_islands=False)
    first = snapshot()
    measured = tools.execute(prepare(tools, first), first, LLM([designed(decision())]), io=IO())
    tools.commit(measured)
    other = snapshot(1, 1)
    close = designed(finish(conclusion='inconclusive', reason='FOREIGN_MEASURED: the prior check found no gain.'),
        kind='closure', budget_action='none', prior_receipt_ids=measured['receipt_ids'])
    episode = tools.execute(prepare(tools, other), other, LLM([close]), io=IO())
    assert episode['cards'][0]['decision']['receipt_ids'] == []  # No own tool invocation.
    fresh = tools.lifecycle.delivery_context(episode['cards'], episode=episode)
    assert next(c for c in fresh['cards'] if c.get('decision'))['decision']['interpretation'].startswith('FOREIGN_MEASURED')
    assert fresh['cards'][0]['source'] == measured['cards'][0]['source']
    assert fresh['cards'][0]['observations'] == measured['cards'][0]['observations']
    tools.commit(episode)
    for _ in range(2):
        cards = tools.context(snapshot(2, 2))
        assert any(c['cell_id'] == episode['plan']['cell_id'] for c in cards)
        tools.load_state(deepcopy(tools.state()))
    assert len(backend.calls) == 1


@pytest.mark.parametrize('research_enabled', [False, True])
def test_real_engine_ordinary_delivery_excludes_fresh_unmeasured_finish_and_resume(
        tmp_path, monkeypatch, research_enabled):
    import sys
    from sciel.engine import checkpoint
    from sciel.engine.factory import build_engine
    from sciel.engine.llm.client import LLMResponse
    from test_parallel_engine import ResearchProvider, config, evaluator, task_fixture
    from test_research_tool_controller import IMAGE
    from test_research_tool_result_flow import SavedExecutor

    tasks = tmp_path / 'tasks'
    task = task_fixture(tasks)
    evaluator(monkeypatch)
    backend = SavedExecutor([])
    monkeypatch.setattr('sciel.research_tools.library.DockerExecutor', lambda image: backend)

    class Provider(ResearchProvider):
        planner_calls = 0

        def generate(self, prompt, model, temperature=None, **options):
            if not prompt.startswith('SCIEL_RESEARCH_TOOL_EPISODE'):
                assert UNMEASURED not in prompt
                return super().generate(prompt, model, temperature, **options)
            self.planner_calls += 1
            return LLMResponse(json.dumps(finish(conclusion='inconclusive',
                reason=UNMEASURED + ' Planning details without measurements.' * 500)),
                               model, .03, 20, 20, finish_reason='stop')

    cfg = config(task, tmp_path / 'capacity', workers=1, num_islands=1, max_iterations=1,
        research={'enabled':research_enabled, 'reflection_interval':0, 'max_aux_calls':None},
        research_tools={'enabled':True, 'image_id':IMAGE, 'interval':1,
            'proactive_every_attempts':1, 'max_scientific_steps':3, 'max_requests':2,
            'max_context_chars':4096, 'share_between_islands':False})
    provider = Provider(concurrent=1)
    directory = tmp_path / 'run'
    engine = build_engine(cfg, str(tasks), str(directory), python=sys.executable, llm=provider)
    # An authenticated earlier observation must survive the 4 KiB selection,
    # not merely a synthetic card claiming a receipt that never existed.
    if not research_enabled:
        backend.outputs.append(result({'finding':'PRIOR_MEASURED_CHECK: observed zero gain.',
                                           'saved_state':'unchanged fixture state ' * 200}))
        def prior_context(_):
            from research_target_fixtures import bind_snapshot
            current = bind_snapshot(snapshot(), task_hash=engine.task.task_hash)  # p0 differs from seeded p1.
            from sciel.methods.models import content_hash
            current.research_targets['run_id'] = engine.log.run_id
            current.research_targets['sha256'] = content_hash({k:v for k,v in current.research_targets.items() if k != 'sha256'})
            prior_plan = engine.research_tools.prepare(current, signals={
                'schema':'sciel-stagnation-signals/1', 'committed_iteration':0,
                'population_stall':{'threshold_reached':True, 'last_record_B_evals':0}})
            prior_episode = engine.research_tools.execute(prior_plan, current,
                LLM([decision(), finish(reason='Measured earlier zero gain.')]), io=IO())
            engine.research_tools.commit(prior_episode)
            prior = deepcopy(prior_episode['cards'][0])
            assert 'PRIOR_MEASURED_CHECK' in prior['observations']['output_excerpt']
            assert len(engine.research_tools.format_context([prior])) > 2000
            monkeypatch.setattr(engine.research_tools, 'context_view', lambda _: compact_context([prior]))
            # Request the separate unmeasured check through the normal stall
            # trigger; creating the earlier observation used the initial clock.
            original_prepare = engine.research_tools.prepare
            def triggered(current, *, signals=None):
                return original_prepare(current, signals={**(signals or {}),
                    'population_stall':{'threshold_reached':True, 'last_record_B_evals':0}})
            monkeypatch.setattr(engine.research_tools, 'prepare', triggered)
            return compact_context([prior])
        monkeypatch.setattr(engine.research_tools, 'context_view', prior_context)
    engine.run()
    saved = deepcopy(engine.research_tools.state())
    assert any(card['decision']['interpretation'].startswith(UNMEASURED)
               for card in saved['lifecycle']['cards'].values())
    assert len([r for r in provider.roles if r == 'G']) == 1
    first_g = next(prompt for prompt, role in zip(provider.calls, provider.roles) if role == 'G')
    if not research_enabled:
        assert 'PRIOR_MEASURED_CHECK' in first_g
    restored = build_engine({**cfg, 'max_iterations':2}, str(tasks), str(directory),
                            python=sys.executable, llm=provider)
    checkpoint.load(restored, str(directory / 'checkpoint.json'))
    assert restored.research_tools.state() == saved
    monkeypatch.setattr(restored.research_tools, 'prepare', lambda *args, **kwargs: None)
    restored.run()
    assert len([r for r in provider.roles if r == 'G']) == 2
    assert provider.planner_calls == 1 and len(backend.calls) == (0 if research_enabled else 1)
    assert all(UNMEASURED not in p for p in provider.calls)
