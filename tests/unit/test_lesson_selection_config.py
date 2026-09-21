"""Source-diverse retrieval configuration and checkpoint identity; no model/evaluation."""
import copy
from types import SimpleNamespace

import pytest

from sciel.config import apply_defaults
from sciel.engine import checkpoint
from sciel.eval.registry import load_task


def settings(selection):
    return apply_defaults({'research': {'enabled': True, 'lesson_selection': selection}})


def identity(cfg):
    engine = SimpleNamespace(cfg=cfg, task=load_task('tasks/test/toy_quadratic'), descriptor_info={})
    return checkpoint._identity(engine)


def test_omitted_setting_preserves_existing_research_config_and_identity():
    before = apply_defaults({'research': {'enabled': True}})
    assert 'lesson_selection' not in before['research']
    assert 'lesson_selection' not in identity(before)['research']
    assert apply_defaults(before) == before
    assert 'research' not in apply_defaults({})


@pytest.mark.parametrize('selection', [
    None, [], 'source_diverse', True,
    {'mode': None}, {'mode': False}, {'mode': 'semantic'}, {'mode': []},
    {'extra': True}, {'mode': 'source_diverse', 'enabled': True},
    {'diversity_weight': True}, {'diversity_weight': False},
    {'diversity_weight': None}, {'diversity_weight': '0.5'},
    {'diversity_weight': []}, {'diversity_weight': -0.01},
    {'diversity_weight': 1.01}, {'diversity_weight': float('nan')},
    {'diversity_weight': float('inf')}, {'diversity_weight': -float('inf')},
    {'diversity_weight': 10**1000},
])
def test_invalid_settings_cannot_enter_a_frozen_experiment(selection):
    with pytest.raises(ValueError):
        settings(selection)


@pytest.mark.parametrize('mode', ['ranked', 'source_diverse'])
@pytest.mark.parametrize('weight', [0, 0.5, 1])
def test_normalized_explicit_settings_are_stable_and_do_not_mutate_input(mode, weight):
    raw = {'mode': mode, 'diversity_weight': weight}
    before = copy.deepcopy(raw)
    cfg = settings(raw)
    assert raw == before
    assert cfg['research']['lesson_selection'] == {'mode': mode, 'diversity_weight': float(weight)}
    assert apply_defaults(cfg) == cfg


def test_explicit_ranked_default_has_no_algorithm_identity_but_retains_config():
    cfg = settings({})
    assert cfg['research']['lesson_selection'] == {'mode': 'ranked', 'diversity_weight': 0.5}
    assert 'lesson_selection' not in identity(cfg)['research']
    assert identity(cfg)['research']['config'] == cfg['research']


def test_version_weight_and_mode_are_bound_even_with_ablation_or_zero_weight(monkeypatch):
    from sciel.knowledge_base import retrieval
    cfg = settings({'mode': 'source_diverse', 'diversity_weight': 0})
    cfg['research']['inject_lessons'] = False
    original = identity(cfg)
    assert original['research']['lesson_selection'] == {
        'schema': retrieval.CONTRACT, 'mode': 'source_diverse', 'diversity_weight': 0.0}
    changed_weight = copy.deepcopy(cfg)
    changed_weight['research']['lesson_selection']['diversity_weight'] = 0.5
    assert identity(changed_weight) != original
    changed_mode = copy.deepcopy(cfg)
    changed_mode['research']['lesson_selection']['mode'] = 'ranked'
    assert identity(changed_mode) != original
    monkeypatch.setattr(retrieval, 'CONTRACT', retrieval.CONTRACT + '-changed')
    assert identity(cfg)['research']['lesson_selection'] != original['research']['lesson_selection']
