"""Optional proactive cadence is explicit and independent of resource budgets."""
from copy import deepcopy

import pytest

from sciel.config import load_config, validate
from sciel.research_tools.config import DEFAULTS, validate_config


def test_default_and_null_keep_proactive_research_disabled():
    defaults = deepcopy(DEFAULTS)
    assert validate_config()['proactive_every_attempts'] is None
    assert validate_config({'proactive_every_attempts': None}) == validate_config()
    configured = validate_config({'proactive_every_attempts': 8})
    assert configured['enabled'] is False
    assert {**configured, 'proactive_every_attempts': None} == validate_config()
    assert DEFAULTS == defaults


@pytest.mark.parametrize('value', [1, 8, 10**9])
def test_positive_integer_cadence_is_preserved(value):
    assert validate_config({'proactive_every_attempts': value})['proactive_every_attempts'] == value


@pytest.mark.parametrize('value', [True, False, 0, -1, 1.0, 8.5,
                                 float('inf'), float('nan'), '8', [], {}])
def test_invalid_cadence_is_not_coerced(value):
    with pytest.raises(ValueError, match=r'research_tools\.proactive_every_attempts'):
        validate_config({'proactive_every_attempts': value})


def test_yaml_inheritance_retains_cadence_and_explicit_null_disables_it(tmp_path):
    (tmp_path / 'base.yaml').write_text(
        'task: fixture/search\nresearch_tools:\n  proactive_every_attempts: 8\n'
        '  jobs:\n    max_cpu_seconds: 40\n    max_wall_seconds: 60\n')
    (tmp_path / 'inherited.yaml').write_text('extends: base.yaml\n')
    (tmp_path / 'disabled.yaml').write_text(
        'extends: base.yaml\nresearch_tools:\n  proactive_every_attempts: null\n')
    inherited = validate(load_config(str(tmp_path / 'inherited.yaml')))
    disabled = validate(load_config(str(tmp_path / 'disabled.yaml')))
    assert inherited['research_tools']['proactive_every_attempts'] == 8
    assert disabled['research_tools']['proactive_every_attempts'] is None
    assert inherited['research_tools']['enabled'] is disabled['research_tools']['enabled'] is False
    assert inherited['research_tools']['jobs'] == disabled['research_tools']['jobs']
    assert inherited['research_tools']['limits'] == disabled['research_tools']['limits']
    assert inherited['research_tools']['max_requests'] == disabled['research_tools']['max_requests'] == 2


def test_top_level_config_rejects_boolean_cadence():
    with pytest.raises(ValueError, match=r'research_tools\.proactive_every_attempts'):
        validate({'task': 'fixture/search', 'research_tools': {'proactive_every_attempts': True}})
