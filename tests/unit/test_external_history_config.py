"""Import opt-in and pristine-state guards; no provider or numerical execution."""
import copy
from types import SimpleNamespace as NS

import pytest

from sciel.config import apply_defaults
from sciel.engine.research import ResearchController
from test_restart_prediction_engine import config


def configured():
    value = config()
    value['research']['external_history_sha256'] = 'a' * 64
    return value


def controller(cfg):
    return ResearchController(cfg, NS(models=['fixture'], max_retries=1),
        NS(research=lambda **_: pytest.fail('unexpected journal write')), lambda _: pytest.fail('charge'))


@pytest.mark.parametrize('value', [None, True, 12, '', 'A'*64, 'a'*63, 'g'*64, {}, []])
def test_invalid_import_digest_is_refused_before_construction(value):
    cfg = configured(); cfg['research']['external_history_sha256'] = value
    with pytest.raises(ValueError, match='external_history_sha256'):
        apply_defaults(cfg)
    direct = apply_defaults(config())['research']; direct['external_history_sha256'] = value
    with pytest.raises(ValueError, match='external_history_sha256'):
        controller(direct)


@pytest.mark.parametrize('change', ['pair_mode', 'without_predictions', 'without_generation'])
def test_import_requires_the_supported_source_and_journal_contract(change):
    cfg = configured()
    if change == 'pair_mode': cfg['circle_generation']['proposal_policy'] = 'first_eligible_of_two'
    elif change == 'without_predictions': cfg['research']['restart_predictions'] = False
    else: cfg.pop('circle_generation'); cfg['op_fixed'] = 'mutate'
    with pytest.raises(ValueError): apply_defaults(cfg)


def test_pristine_declared_state_roundtrips_but_cannot_drop_registry_after_work():
    cfg = apply_defaults(configured())['research']; c = controller(cfg)
    saved = c.state()
    assert 'external_history' not in saved and 'external_history_identity' in saved
    c.load_state(saved); assert c.state() == saved
    broken = copy.deepcopy(saved); broken.update(attempts=1, outcomes=1)
    with pytest.raises(ValueError, match='missing its declared external history'):
        c.load_state(broken)
    assert c.state() == saved


def test_external_identity_cannot_be_injected_into_legacy_state():
    c = controller(apply_defaults(config())['research']); saved = c.state()
    assert 'external_history' not in saved and 'external_history_identity' not in saved
    bad = copy.deepcopy(saved); bad['external_history_identity'] = {}
    with pytest.raises(ValueError, match='external history identity'):
        c.load_state(bad)
    assert c.state() == saved
