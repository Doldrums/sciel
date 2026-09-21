"""Explicit component review version opt-in and pre-import configuration pin."""
import copy

import pytest

from sciel.config import apply_defaults
from sciel.engine.generated_paired_review_import import configured_version
from test_generated_paired_config import configured, controller


def scoped():
    cfg = configured()
    cfg['research'].update(structured_auxiliary=True, comparison_reviews=True,
        comparison_review_version=3, generated_paired_review_version=2,
        comparison_review_text=True, batch_only_lessons=True,
        comparison_review_text_policy={'schema': 'sciel-review-text-policy/1', 'decisions': []},
        external_component_review_sha256='b' * 64)
    return cfg


@pytest.mark.parametrize('version', [True, False, None, 0, 3, 2., '2', {}, []])
def test_invalid_version_rejected_by_factory_and_direct_controller(version):
    cfg = scoped()
    direct = apply_defaults(cfg)['research']
    cfg['research']['generated_paired_review_version'] = version
    direct['generated_paired_review_version'] = version
    with pytest.raises(ValueError, match='generated_paired_review_version'):
        apply_defaults(cfg)
    with pytest.raises(ValueError, match='generated_paired_review_version'):
        controller(direct)


@pytest.mark.parametrize('missing', ['external_component_sha256', 'comparison_review_text_policy',
                                   'comparison_review_version', 'structured_auxiliary'])
def test_explicit_version_requires_declared_component_and_reasoned_text_gate(missing):
    cfg = scoped()['research']
    del cfg[missing]
    with pytest.raises(ValueError):
        configured_version(cfg)


def test_omission_preserves_legacy_state_and_wire_selection():
    cfg = scoped()
    del cfg['research']['generated_paired_review_version']
    research = apply_defaults(cfg)['research']
    assert 'generated_paired_review_version' not in research
    assert configured_version(research) == 1
    state = controller(research).state()
    assert state['external_component_review_identity']['schema'] == 'sciel-generated-paired-review-import/1'
    assert 'generated_paired_review_version' not in state['external_component_review_identity']


def test_explicit_version_is_pinned_before_any_component_or_review_is_imported():
    c = controller(apply_defaults(scoped())['research'])
    saved = c.state()
    assert saved['external_component_review_identity']['generated_paired_review_version'] == 2
    c.load_state(saved)
    c.config['generated_paired_review_version'] = 1
    with pytest.raises(ValueError, match='receiving policy changed'):
        c.state()
    with pytest.raises(ValueError):
        c.load_state(saved)


def test_changing_both_checkpoint_config_and_receiver_cannot_erase_old_version_identity():
    saved = controller(apply_defaults(scoped())['research']).state()
    downgraded = copy.deepcopy(saved)
    downgraded['config']['generated_paired_review_version'] = 1
    receiver = controller(downgraded['config'])
    with pytest.raises(ValueError, match='checkpoint policy differs'):
        receiver.load_state(downgraded)
