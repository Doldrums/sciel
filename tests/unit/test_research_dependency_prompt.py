"""Dependency authoring matches the real proxy, without provider/container calls.

Planner/executor leaves use fixed receipts. The sole loader test imports a tiny
test-owned module, never model-authored or historical experiment source.
"""
import importlib
import sys

import pytest

from sciel.methods._loader import make_method_loader
from sciel.research_tools.controller import _compact_tool_instructions, response_format
from test_research_tool_controller import IO, LLM, prepare
from test_research_tool_result_flow import action, controller, result
from test_sequential_research_episode import decision
from test_tool_input_projection import target_fixture, rendered_view, assert_view_matches_inputs


def assert_dependency_api(text):
    assert 'm_<id> proxy: ONLY METHOD_ID + invoke(payload)' in text
    assert 'no raw functions/constants/helpers despite visible source' in text
    assert "__import__('m_'+exact_id).invoke(payload)" in text
    assert 'five-field sciel-tool-result/1' in text
    assert "check r['status']/r['error'], read nested r['observations']" in text


@pytest.mark.parametrize('sequential', [False, True])
@pytest.mark.parametrize('budget', [8000, 64000])
def test_actual_initial_and_repair_prompts_explain_dependency_boundary_without_losing_evidence(
        tmp_path, sequential, budget):
    tools, backend, _, costs = controller(tmp_path,
        [result(status='tool_error', error='dependency missing helper: FAMILIES'),
         result({'measured_check': 7})],
        background=True, max_prompt_chars=budget,
        max_scientific_steps=3 if sequential else 1,
        yield_after_success=sequential)
    current = target_fixture('zero', huge=True)
    current.kb_lessons = ['Old unrelated knowledge ' * 5000]
    current.task_prompt = 'Long task description ' * 4000
    make_action = decision if sequential else action
    first = make_action()

    def repair(prompt):
        ident = next(tools.library.objects.glob('*.json')).stem
        assert_dependency_api(prompt)
        assert 'dependency missing helper: FAMILIES' in prompt
        fixed = make_action()
        fixed.update(name='composed_check', parents=[ident], dependencies=[ident],
            code="def run(payload):\n    return __import__('m_" + ident + "').invoke(payload)\n")
        return fixed

    llm = LLM([first, repair])
    episode = tools.execute(prepare(tools, current), current, llm, io=IO())
    assert len(llm.prompts) == len(backend.calls) == 2
    assert costs == [.03, .03]
    assert episode['invocations'][-1]['result']['status'] == 'success'
    for prompt, call in zip(llm.prompts, backend.calls):
        assert len(prompt) <= budget
        assert_dependency_api(prompt)
        assert_view_matches_inputs(rendered_view(prompt), call['inputs'])
        assert '"schema":"sciel-tool-result/1"' in prompt
        assert 'tool_error/nonempty error' in prompt
    original, composed = episode['registered_ids']
    assert tools.library.get(composed)['manifest']['dependencies'] == [original]
    assert backend.calls[1]['kwargs']['dependency_bundles'][0]['id'] == original


@pytest.mark.parametrize('sequential', [False, True])
def test_minimum_instruction_variant_and_structured_schema_keep_same_dependency_api(sequential):
    assert_dependency_api(_compact_tool_instructions(sequential))
    properties = response_format(True, sequential=sequential,
        visible_tool_ids=['a' * 64])['json_schema']['schema']['properties']
    assert_dependency_api(properties['dependencies']['description'])
    assert properties['dependencies']['items']['enum'] == ['a' * 64]
    assert 'tool lineage' in properties['parents']['description']
    assert 'proxy' not in properties['parents']['description']


def test_actual_dependency_proxy_exposes_invoke_not_visible_source_helpers(tmp_path):
    """Loading the dependency does not turn its source globals into proxy APIs."""
    ident = 'e' * 64
    source = '''FAMILIES = ["test_family"]
def private_helper(value):
    return value + 1
def run(payload):
    return {"schema": "sciel-tool-result/1", "status": "success",
            "observations": {"measured": private_helper(payload["value"]),
                             "families": FAMILIES},
            "diagnostics": {}, "error": None}
'''
    (tmp_path / 'tool.py').write_text(source)
    before_finders = list(sys.meta_path)
    prefix = '_sciel_bundle_' + ident
    try:
        make_method_loader({ident: {'root': str(tmp_path), 'entrypoint': 'tool.py:run'}})
        proxy = importlib.import_module('m_' + ident)
        assert proxy.METHOD_ID == ident
        assert {name for name in vars(proxy) if not name.startswith('__')} == {'METHOD_ID', 'invoke'}
        with pytest.raises(AttributeError):
            _ = proxy.FAMILIES
        nested_result = proxy.invoke({'value': 6})
        assert set(nested_result) == {'schema', 'status', 'observations', 'diagnostics', 'error'}
        assert nested_result['status'] == 'success' and nested_result['error'] is None
        assert nested_result['observations'] == {'measured': 7, 'families': ['test_family']}
        assert not hasattr(proxy, 'private_helper') and not hasattr(proxy, 'FAMILIES')
    finally:
        sys.meta_path[:] = before_finders
        for name in list(sys.modules):
            if name == 'm_' + ident or name == prefix or name.startswith(prefix + '.'):
                sys.modules.pop(name)
