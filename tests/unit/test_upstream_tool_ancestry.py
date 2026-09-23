"""Current /34 renderer closure is trusted code, never artifact-supplied code."""
from copy import deepcopy
import pytest

from sciel.adaptation.ancestry import ResolutionUnknown
from sciel.adaptation.tool_ancestry import ToolAncestry
from test_research_tool_controller import prepare, setup, snapshot


def reader(tools):
    value = ToolAncestry.__new__(ToolAncestry)
    value.config = deepcopy(tools.config)
    value.identity = tools.identity
    value.state = tools.state()
    value.library = tools.library
    return value


def test_current_split_renderer_reconstructs_actual_complete_prompt(tmp_path):
    tools, _, _, _ = setup(tmp_path)
    source = snapshot()
    plan = prepare(tools, source)
    actual = tools._prompt(plan, source, plan['visible_tool_ids'], [], 0)
    restored = reader(tools)._controller(tools.lifecycle)
    assert restored._prompt(plan, source, plan['visible_tool_ids'], [], 0) == actual
    assert 'create.code' in actual


@pytest.mark.parametrize('field,name', [
    ('implementation', 'prompt_builder.py'),
    ('implementation', 'tool_protocol.py'),
    ('implementation', 'controller.py'),
    ('integration', 'research_episodes.py'),
    ('integration', 'parallel_calls.py'),
])
def test_unchanged_facade_cannot_authorize_changed_delegate_or_native_source(tmp_path, field, name):
    tools, _, _, _ = setup(tmp_path)
    value = reader(tools)
    value.identity[field][name] = '0' * 64
    with pytest.raises(ResolutionUnknown, match='package source closure differs'):
        value._controller(tools.lifecycle)


@pytest.mark.parametrize('mutation', ['missing', 'extra', 'schema', 'historical'])
def test_current_renderer_requires_exact_complete_profile(tmp_path, mutation):
    tools, _, _, _ = setup(tmp_path)
    value = reader(tools)
    if mutation == 'missing':
        del value.identity['implementation']['prompt_builder.py']
    elif mutation == 'extra':
        value.identity['implementation']['untrusted_renderer.py'] = '0' * 64
    elif mutation == 'schema':
        value.state['schema'] = 'sciel-island-research-tools/26'
    else:
        value.identity['contract'] = 'sciel-island-research-tools/26'
    with pytest.raises(ResolutionUnknown, match='source closure|tool protocol|historical'):
        value._controller(tools.lifecycle)
