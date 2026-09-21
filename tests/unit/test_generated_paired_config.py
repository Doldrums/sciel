"""Host opt-in validation, independent of saved artifact fixtures."""
import copy
from types import SimpleNamespace as NS

import pytest

from sciel.config import apply_defaults
from sciel.engine.research import ResearchController
from test_restart_prediction_engine import config


def configured():
    value = config()
    value['research']['external_component_sha256'] = 'a' * 64
    return value


def controller(cfg):
    return ResearchController(cfg, NS(models=['fixture'], max_retries=1),
        NS(research=lambda **_: pytest.fail('unexpected journal write')),
        lambda _: pytest.fail('unexpected charge'))


@pytest.mark.parametrize('value', [None, True, 12, '', 'A' * 64, 'a' * 63, 'g' * 64, {}, []])
def test_component_digest_requires_explicit_lowercase_sha256(value):
    cfg = configured()
    cfg['research']['external_component_sha256'] = value
    with pytest.raises(ValueError, match='external_component_sha256'):
        apply_defaults(cfg)
    direct = apply_defaults(config())['research']
    direct['external_component_sha256'] = value
    with pytest.raises(ValueError, match='digest'):
        controller(direct)


@pytest.mark.parametrize('change', ['pair_mode', 'without_predictions', 'without_generation'])
def test_component_requires_supported_execution_profile(change):
    cfg = configured()
    if change == 'pair_mode':
        cfg['circle_generation']['proposal_policy'] = 'first_eligible_of_two'
    elif change == 'without_predictions':
        cfg['research']['restart_predictions'] = False
    else:
        cfg.pop('circle_generation')
        cfg['op_fixed'] = 'mutate'
    with pytest.raises(ValueError):
        apply_defaults(cfg)


def test_legacy_state_does_not_acquire_component_identity():
    c = controller(apply_defaults(config())['research'])
    saved = c.state()
    assert 'external_component_identity' not in saved
    assert 'generated_paired_history' not in saved['knowledge']
    forged = copy.deepcopy(saved)
    forged['external_component_identity'] = {}
    with pytest.raises(ValueError, match='checkpoint policy differs'):
        c.load_state(forged)
    assert c.state() == saved


def test_configured_digest_cannot_change_during_run():
    c = controller(apply_defaults(configured())['research'])
    saved = c.state()
    c.load_state(saved)
    c.config['external_component_sha256'] = 'b' * 64
    with pytest.raises(ValueError, match='configured digest changed'):
        c.state()


@pytest.mark.parametrize('kind', ['conclusion', 'reflection'])
def test_auxiliary_cannot_skip_declared_import_even_before_budget_check(kind):
    c = controller(apply_defaults(configured())['research'])
    c.config['max_aux_calls'] = 0
    before = c.state()
    with pytest.raises(ValueError, match='has not been imported'):
        c._aux(kind, 'inert prompt; must never be dispatched', 1)
    assert c.state() == before
