"""Cooperative research yield is optional and cannot silently alter resource policy."""
from copy import deepcopy

import pytest

from sciel.config import load_config, validate
from sciel.research_tools.config import DEFAULTS, validate_config


def test_default_and_explicit_false_preserve_existing_policy():
    defaults = deepcopy(DEFAULTS)
    resolved = validate_config()
    assert resolved['yield_after_success'] is False
    assert validate_config({'yield_after_success': False}) == resolved
    assert DEFAULTS == defaults


def test_true_changes_only_yield_without_enabling_tools_or_jobs():
    defaults = deepcopy(DEFAULTS)
    enabled_yield = validate_config({'yield_after_success': True})
    assert enabled_yield['yield_after_success'] is True
    assert enabled_yield['enabled'] is False
    assert enabled_yield['jobs']['enabled'] is False
    assert {**enabled_yield, 'yield_after_success': False} == validate_config()
    assert DEFAULTS == defaults


@pytest.mark.parametrize('value', [None, 0, 1, -1, 0.0, 1.0, float('nan'),
                                 float('inf'), 'true', 'false', '', [], {}])
def test_yield_requires_a_real_boolean(value):
    with pytest.raises(ValueError, match=r'research_tools\.yield_after_success must be boolean'):
        validate_config({'yield_after_success': value})


def test_yaml_inherits_true_and_false_override_preserves_limits(tmp_path):
    (tmp_path / 'base.yaml').write_text(
        'task: fixture/search\nresearch_tools:\n  yield_after_success: true\n'
        '  max_requests: 3\n  max_scientific_steps: 5\n'
        '  jobs:\n    max_cpu_seconds: 40\n    max_wall_seconds: 60\n')
    (tmp_path / 'inherited.yaml').write_text('extends: base.yaml\n')
    (tmp_path / 'disabled.yaml').write_text(
        'extends: base.yaml\nresearch_tools:\n  yield_after_success: false\n')
    inherited = validate(load_config(str(tmp_path / 'inherited.yaml')))
    disabled = validate(load_config(str(tmp_path / 'disabled.yaml')))
    assert inherited['research_tools']['yield_after_success'] is True
    assert disabled['research_tools']['yield_after_success'] is False
    expected = deepcopy(inherited)
    expected['research_tools']['yield_after_success'] = False
    assert disabled == expected
    assert inherited['research_tools']['enabled'] is False


def test_top_level_validation_rejects_string_boolean():
    with pytest.raises(ValueError, match=r'research_tools\.yield_after_success'):
        validate({'task': 'fixture/search', 'research_tools': {'yield_after_success': 'true'}})


def test_absent_research_block_remains_absent():
    assert 'research_tools' not in validate({'task': 'fixture/search'})


def test_yield_requires_the_sequential_research_route():
    with pytest.raises(ValueError, match=r'yield_after_success requires max_scientific_steps greater than 1'):
        validate_config({'yield_after_success': True, 'max_scientific_steps': 1})
    assert validate_config({'yield_after_success': True, 'max_scientific_steps': 2})['yield_after_success'] is True
    assert validate_config({'yield_after_success': False, 'max_scientific_steps': 1})['max_scientific_steps'] == 1
