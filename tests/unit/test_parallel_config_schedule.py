"""Parallel execution profiles and deterministic dispatch fairness, offline."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from sciel.config import DEFAULTS, apply_defaults, load_config, validate
from sciel.engine.schedule import Schedule


def enabled(**overrides):
    return {'parallel_search': {'enabled': True, **overrides},
            'candidate_novelty': {'enabled': True}}


def test_parallel_defaults_are_disabled_and_do_not_mutate_shared_defaults():
    before = deepcopy(DEFAULTS)
    cfg = validate({})
    assert cfg['parallel_search']['enabled'] is False
    assert cfg['parallel_search']['workers'] == 4
    assert apply_defaults(cfg) == cfg
    one = validate(enabled(workers=1))
    assert one['parallel_search']['workers'] == 1
    assert one['parallel_search']['host_evaluator_slots'] == 2
    assert DEFAULTS == before


@pytest.mark.parametrize('field', ['workers', 'host_llm_slots', 'provider_slots',
    'host_evaluator_slots', 'max_inflight_per_island', 'max_inflight_per_parent'])
@pytest.mark.parametrize('value', [0, -1, True, 1.5, None])
def test_parallel_capacity_must_be_explicit_positive_integer(field, value):
    with pytest.raises(ValueError, match='parallel_search ' + field):
        validate(enabled(**{field: value}))


@pytest.mark.parametrize('parallel', [None, False, {'enabled': 1}, {'unknown_limit': 3}])
def test_parallel_shape_rejects_ambiguous_or_unknown_options(parallel):
    with pytest.raises(ValueError, match='parallel_search'):
        validate({'parallel_search': parallel})


@pytest.mark.parametrize('directory', ['', '  ', 1, False, 'a\x00b'])
def test_parallel_capacity_directory_must_be_a_usable_path_string(directory):
    with pytest.raises(ValueError, match='capacity_dir'):
        validate(enabled(capacity_dir=directory))


def test_parallel_requires_whole_history_novelty_without_enabling_it_silently():
    with pytest.raises(ValueError, match='requires candidate_novelty'):
        validate({'parallel_search': {'enabled': True}})


def test_parallel_pins_one_transport_attempt_even_without_research():
    cfg = validate({**enabled(), 'llm': {'mock': True}})
    assert cfg['llm']['max_retries'] == 1
    assert apply_defaults(cfg) == cfg
    for retries in (None, 0, 2, True, 1.0):
        with pytest.raises(ValueError, match=r'parallel_search requires llm.max_retries=1'):
            validate({**enabled(), 'llm': {'max_retries': retries}})
    assert 'max_retries' not in validate({})['llm']


@pytest.mark.parametrize('islands', [0, -1, True, 1.5, None])
def test_parallel_requires_a_nonempty_integer_island_roster(islands):
    with pytest.raises(ValueError, match='positive integer num_islands'):
        validate({**enabled(), 'num_islands': islands})


def test_parallel_centre_selection_requires_a_supported_parent_mask():
    with pytest.raises(ValueError, match='centre selection'):
        validate({**enabled(), 'container': {'selection': 'centres'}})


def test_parallel_research_retains_a_whole_wave_of_bounded_attempts():
    with pytest.raises(ValueError, match=r'workers \* 3'):
        validate({**enabled(), 'research': {'enabled': True, 'max_experiments': 11}})
    cfg = validate({**enabled(), 'research': {'enabled': True, 'max_experiments': 12}})
    assert cfg['research']['max_experiments'] == 12
    # Retention restrictions belong to active research, not historical or off modes.
    validate({**enabled(), 'research': {'enabled': False, 'max_experiments': 1,
                                     'reflection_interval': 0}})


@pytest.mark.parametrize('route', ['repeat_population', 'repeat_promotion',
    'circle_generation', 'circle_intervention', 'kb_free', 'c2_event_bets'])
def test_parallel_rejects_unsupported_routes_without_mutating_config(route):
    cfg = enabled()
    cfg[route] = True if route in {'kb_free', 'c2_event_bets'} else {'enabled': True}
    before = deepcopy(cfg)
    with pytest.raises(ValueError, match='parallel_search does not support ' + route):
        validate(cfg)
    assert cfg == before


@pytest.mark.parametrize('suffix', ['', '_rl_full', '_rl_presets_vq'])
def test_parallel_profiles_preserve_their_original_research_and_controller_stacks(suffix):
    config_dir = Path(__file__).resolve().parents[2] / 'configs'
    original = validate(load_config(str(config_dir / f'circle_live_gemini{suffix}.yaml')))
    parallel = validate(load_config(str(config_dir / f'circle_parallel_gemini{suffix}.yaml')))
    for key in ('research', 'policy', 'candidate_novelty', 'llm', 'container',
                'embed_grid', 'embed_niche', 'idea_two_stage', 'preset_selection'):
        assert parallel.get(key) == original.get(key)
    assert parallel['num_islands'] == parallel['max_islands'] == 24
    assert parallel['pop_cap'] == 256 and parallel['dynamic_islands'] is False
    assert parallel['budget'] is parallel['max_iterations'] is None
    assert parallel['schedule_horizon'] == 768
    assert parallel['parallel_search']['enabled'] is True
    assert parallel['sh_min_fair_evals'] == 16


def test_fair_24_island_waves_reach_16_dispatches_each_before_first_sh_cut():
    schedule = Schedule(24, 'sh', (.5, .75), 1, 16, fair_dispatch=True)
    assert schedule.milestones_for(768) == [384, 576]
    for _ in range(96):
        flight = {}
        for _ in range(4):
            island = schedule.pick_island(list(range(24)), flight)
            assert flight.get(island, 0) == 0
            flight[island] = 1
    assert set(schedule.dispatched.values()) == {16}
    events = []
    db = SimpleNamespace(island_best=lambda i: 24 - i,
                         migrate_unique_elites=lambda *a, **kw: 0)
    schedule.on_milestone(384, db, SimpleNamespace(sh_cut=lambda **kw: events.append(kw)))
    assert len(schedule.survivors) == 12
    assert events[0]['protected'] == [] and len(events[0]['cut']) == 12


def test_fair_dispatch_catches_up_after_backpressure_and_respects_parent_eligibility():
    schedule = Schedule(4, 'round_robin', (), 1, fair_dispatch=True)
    for _ in range(8):
        schedule.pick_island([0, 1, 2, 3], {0: 1})
    assert schedule.dispatched[0] == 0
    assert [schedule.pick_island([0, 1, 2, 3], {}) for _ in range(2)] == [0, 0]
    schedule.policy = 'sh'
    schedule.milestones_passed = 1
    schedule.survivors = [0, 2]
    assert schedule.pick_island([1, 2, 3], {}) == 2
    before = dict(schedule.dispatched)
    assert schedule.pick_island([0, 2], {0: 1, 2: 1}) == -1
    assert schedule.dispatched == before


def test_parallel_fairness_state_roundtrip_and_legacy_cursor_behavior():
    schedule = Schedule(4, fair_dispatch=True)
    schedule.pick_island([0, 1, 2, 3], {})
    restored = Schedule(4)
    restored.load_state(schedule.state())
    assert restored.fair_dispatch is True
    assert restored.pick_island([0, 1, 2, 3], {}) == 1
    legacy_state = schedule.state()
    legacy_state.pop('fair_dispatch')
    restored.load_state(legacy_state)
    assert restored.fair_dispatch is False
    legacy = Schedule(4, 'round_robin', (), 1)
    flight, order = {}, []
    for _ in range(4):
        island = legacy.pick_island([0, 1, 2, 3], flight)
        flight[island] = 1
        order.append(island)
    assert order == [0, 2, 1, 3]
