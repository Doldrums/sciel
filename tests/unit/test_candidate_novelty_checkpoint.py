"""Whole-run source identity survives population turnover; no candidate executes."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from sciel.config import DEFAULTS, apply_defaults, load_config, validate
from sciel.engine import checkpoint
from sciel.engine.db import Database
from sciel.engine.novelty import CandidateNoveltyRegistry


ROOT = Path(__file__).resolve().parents[2]


class MemoryComponent:
    def __init__(self):
        self.value = {'fixture': 'unchanged'}

    def state(self):
        return deepcopy(self.value)

    def load_state(self, value):
        self.value = deepcopy(value)


def engine(path, *, enabled=True, neighbors=3, max_resamples=2):
    path.mkdir(exist_ok=True)
    journal = path / 'sciel_events.jsonl'
    prefix = b'{"event":"run_start","run_id":"fixture-run","segment_id":"fixture","event_seq":1}\n'
    journal.write_bytes(prefix)
    cfg = validate({'num_islands': 2, 'candidate_novelty': {
        'enabled': enabled, 'neighbors': neighbors, 'max_resamples': max_resamples}})
    value = SimpleNamespace(cfg=cfg, run_dir=str(path),
        task=SimpleNamespace(qualified_name='test/fixture', task_hash='task-hash'),
        descriptor_info={'kind': 'fixture'}, milestones_fired=set(), noop_parents=set(),
        lineage_fail={}, eval_times={}, kb_on=False, c2_on=False, plateau_shadow=False,
        plateau_monitors=None, oc_monitors=None,
        log=SimpleNamespace(run_id='fixture-run', path=str(journal), anchor=lambda: {
            'byte_offset':len(prefix), 'prefix_sha256':hashlib.sha256(prefix).hexdigest(),
            'segment_id':'fixture', 'event_seq':1}), close=Mock(),
        candidate_novelty=CandidateNoveltyRegistry(neighbors=neighbors) if enabled else None)
    for name in checkpoint._SCALARS:
        setattr(value, name, 0)
    for name in checkpoint._COMPONENTS:
        setattr(value, name, MemoryComponent())
    value.db = Database(2, pop_cap=1, archive_size=1, grid_bins=1)
    return value


def saved(value):
    path = Path(checkpoint.save(value, 'fixture'))
    return path, json.loads(path.read_text())


def test_novelty_configuration_is_opt_in_and_partial_overrides_are_detached():
    before = deepcopy(DEFAULTS)
    assert validate({})['candidate_novelty'] == {
        'enabled': False, 'max_resamples': 2, 'neighbors': 3, 'output_repeats': False, 'output_resolution': None}
    resolved = validate({'candidate_novelty': {'enabled': True, 'max_resamples': 0}})
    assert resolved['candidate_novelty'] == {'enabled': True, 'max_resamples': 0, 'neighbors': 3, 'output_repeats': False, 'output_resolution': None}
    assert apply_defaults(resolved) == resolved
    resolved['candidate_novelty']['neighbors'] = 1
    assert DEFAULTS == before


@pytest.mark.parametrize('config', [None, True, [], {'other': True}, {'enabled': 1},
    {'enabled': None}, {'max_resamples': True}, {'max_resamples': -1},
    {'max_resamples': 3}, {'max_resamples': 1.5}, {'neighbors': False},
    {'neighbors': 0}, {'neighbors': -2}, {'neighbors': 2.5}, {'neighbors': 11}])
def test_novelty_config_rejects_ambiguous_or_unbounded_retries(config):
    with pytest.raises(ValueError, match='candidate_novelty'):
        validate({'candidate_novelty': config})


@pytest.mark.parametrize('name', ['circle_live_gemini.yaml', 'circle_live_gemini_rl_full.yaml',
                                  'circle_live_gemini_rl_presets_vq.yaml'])
def test_all_live_profiles_resolve_to_enabled_bounded_novelty(name):
    cfg = validate(load_config(str(ROOT / 'configs' / name)))
    assert cfg['candidate_novelty'] == {'enabled': True, 'max_resamples': 2, 'neighbors': 3, 'output_repeats': False, 'output_resolution': None}
    assert cfg['budget'] is None and cfg['max_iterations'] is None


@pytest.mark.parametrize('route', [{'repeat_population': {'enabled': True}},
    {'circle_generation': {'enabled': True}}, {'circle_intervention': {'enabled': True}},
    {'circle_generation': {'enabled': True, 'proposal_policy': 'first_eligible_of_two'}}])
def test_novelty_rejects_specialized_routes_without_complete_registry_integration(route):
    with pytest.raises(ValueError, match='candidate_novelty does not support'):
        validate({'candidate_novelty': {'enabled': True}, **route})


def test_checkpoint_retains_evicted_and_never_admitted_sources(tmp_path):
    original = engine(tmp_path / 'original')
    source = 'def run_search():\n    return 1\n'
    old = original.db.add(source, .1, {}, 0, 'seed')
    original.candidate_novelty.register(source, old.id, island=0, evaluation_ok=True)
    new_source = 'def run_search():\n    return 8\n'
    original.db.add(new_source, .2, {}, 0, 'gen')
    assert old.id not in original.db.islands[0].programs
    assert old.id not in original.db.archive
    failed = 'def run_search():\n    raise ValueError("fixture")\n'
    original.candidate_novelty.register(failed, 'failed-before-admission', island=1, evaluation_ok=False)
    # A removed island's search history cannot be rebuilt from current members.
    original.db.islands[0].programs.clear()
    original.db.islands[0].grid.clear()
    original.db.islands[0].best_id = None
    original.n_remeasurements = 3
    original.n_remeasurement_successes = 2
    path, state = saved(original)
    restored = engine(tmp_path / 'restored')
    registry = restored.candidate_novelty
    checkpoint.load(restored, str(path))
    assert restored.candidate_novelty is registry
    assert registry.state() == state['candidate_novelty']
    assert not restored.db.islands[0].programs
    result = registry.inspect(source)
    assert not result['accept'] and result['matched_program_id'] == old.id
    retry = registry.inspect(failed)
    assert retry['accept'] and retry['remeasurement']
    assert restored.n_remeasurements == 3
    assert state['provenance']['counters']['n_remeasurements'] == 3
    assert restored.n_remeasurement_successes == 2
    assert state['provenance']['counters']['n_remeasurement_successes'] == 2


@pytest.mark.parametrize('damage', ['missing_state', 'missing_identity', 'state_null',
                                   'state_digest', 'neighbors', 'resamples', 'task'])
@pytest.mark.parametrize('force', [False, True])
def test_novelty_history_cannot_be_dropped_or_forced_across_identity(tmp_path, damage, force):
    original = engine(tmp_path / 'original')
    original.candidate_novelty.register('def main(): return 1\n', 'old', evaluation_ok=True)
    path, state = saved(original)
    restored = engine(tmp_path / 'restored', neighbors=2 if damage == 'neighbors' else 3,
                      max_resamples=1 if damage == 'resamples' else 2)
    if damage == 'missing_state':
        del state['candidate_novelty']
    elif damage == 'missing_identity':
        del state['identity']['candidate_novelty']
    elif damage == 'state_null':
        state['candidate_novelty'] = None
    elif damage == 'state_digest':
        state['candidate_novelty']['sha256'] = '0' * 64
    elif damage == 'task':
        restored.task.task_hash = 'different-task'
    path.write_text(json.dumps(state))
    before = restored.candidate_novelty.state()
    with pytest.raises(checkpoint.IdentityMismatch, match='candidate novelty'):
        checkpoint.load(restored, str(path), allow_identity_mismatch=force)
    assert restored.candidate_novelty.state() == before
    restored.close.assert_called_once_with()


@pytest.mark.parametrize('enable_after', [False, True])
def test_registry_cannot_be_enabled_or_disabled_mid_resume(tmp_path, enable_after):
    original = engine(tmp_path / 'original', enabled=not enable_after)
    path, state = saved(original)
    assert ('candidate_novelty' in state) is (not enable_after)
    restored = engine(tmp_path / 'restored', enabled=enable_after)
    with pytest.raises(checkpoint.IdentityMismatch, match='candidate novelty'):
        checkpoint.load(restored, str(path), allow_identity_mismatch=True)


def test_disabled_legacy_checkpoint_has_no_novelty_and_missing_counter_restores_zero(tmp_path):
    original = engine(tmp_path / 'original', enabled=False)
    path, state = saved(original)
    assert 'candidate_novelty' not in state and 'candidate_novelty' not in state['identity']
    del state['config']['candidate_novelty']
    del state['n_remeasurements']
    del state['provenance']['counters']['n_remeasurements']
    del state['n_remeasurement_successes']
    del state['provenance']['counters']['n_remeasurement_successes']
    path.write_text(json.dumps(state))
    restored = engine(tmp_path / 'restored', enabled=False)
    restored.n_remeasurements = 8
    restored.n_remeasurement_successes = 5
    checkpoint.load(restored, str(path))
    assert restored.n_remeasurements == 0 and restored.candidate_novelty is None
    assert restored.n_remeasurement_successes == 0


@pytest.mark.parametrize('counter', [-1, True, 2.5, None, '1'])
def test_invalid_remeasurement_counter_is_rejected(tmp_path, counter):
    original = engine(tmp_path / 'original', enabled=False)
    path, state = saved(original)
    state['n_remeasurements'] = counter
    state['provenance']['counters']['n_remeasurements'] = counter
    path.write_text(json.dumps(state))
    with pytest.raises(checkpoint.IdentityMismatch, match='n_remeasurements'):
        checkpoint.load(engine(tmp_path / 'restored', enabled=False), str(path))


@pytest.mark.parametrize('counter', [-1, True, 2.5, None, '1', 4])
def test_invalid_remeasurement_success_counter_is_rejected(tmp_path, counter):
    original = engine(tmp_path / 'original', enabled=False)
    original.n_remeasurements = 3
    path, state = saved(original)
    state['n_remeasurement_successes'] = counter
    state['provenance']['counters']['n_remeasurement_successes'] = counter
    path.write_text(json.dumps(state))
    with pytest.raises(checkpoint.IdentityMismatch, match='n_remeasurement_successes'):
        checkpoint.load(engine(tmp_path / 'restored', enabled=False), str(path))


@pytest.mark.parametrize('event', [
    {'event':'decision', 'policy':'candidate_measurement', 'action':'start'},
    {'event':'decision', 'policy':'evaluation', 'action':'result'},
    {'event':'decision', 'policy':'candidate_novelty', 'action':'reject'},
    {'event':'decision', 'policy':'portfolio', 'action':'generate'},
    {'event':'iteration', 'child_id':'uncheckpointed'},
    {'event':'research', 'phase':'attempt'},
])
def test_nonresearch_resume_rejects_post_checkpoint_candidate_activity(tmp_path, event):
    original = engine(tmp_path / 'original')
    path, _ = saved(original)
    restored = engine(tmp_path / 'restored')
    with Path(restored.log.path).open('a') as journal:
        journal.write(json.dumps(event)+'\n')
    with pytest.raises(checkpoint.IdentityMismatch, match='uncheckpointed candidate activity'):
        checkpoint.load(restored, str(path), allow_identity_mismatch=True)


def test_novelty_resume_allows_only_complete_control_boundaries_after_anchor(tmp_path):
    original = engine(tmp_path / 'original')
    path, _ = saved(original)
    restored = engine(tmp_path / 'restored')
    with Path(restored.log.path).open('a') as journal:
        for event in ('checkpoint', 'run_end', 'run_start', 'checkpoint_restore'):
            journal.write(json.dumps({'event':event})+'\n')
    checkpoint.load(restored, str(path))
    with Path(restored.log.path).open('a') as journal:
        journal.write('{"event":"checkpoint"}')
    with pytest.raises(checkpoint.IdentityMismatch, match='incomplete tail'):
        checkpoint.load(restored, str(path))
