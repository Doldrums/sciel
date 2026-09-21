"""Complete final-source structural inventory; no candidate execution."""
from copy import deepcopy
import hashlib
import json

import pytest

from sciel.hypothesis.change_audit import (SCHEMA, build_change_audit,
    canonical_json, summarize_change_audit, unit_catalog)


def indexed(audit):
    return {unit['id']: unit for unit in audit['units']}


def test_complete_inventory_retains_module_classes_nested_async_and_duplicate_ids():
    source = '''import os
LIMIT = 1
class C:
    """Class documentation."""
    @decorate(3)
    async def method(self, value=4):
        def nested():
            return 5
        return nested()
def outer():
    if True:
        def helper():
            return 1
    def helper():
        return 2
    return helper()
'''
    catalog = unit_catalog(source)
    assert [row['id'] for row in catalog] == ['<module>', 'C', 'C.method', 'C.method.nested', 'outer', 'outer.helper', 'outer.helper#2']
    assert [row['kind'] for row in catalog][:4] == ['module', 'class', 'async_function', 'function']
    assert all(set(row) == {'id', 'kind', 'local_ast_sha256', 'subtree_ast_sha256'} for row in catalog)
    audit = build_change_audit(source, source)
    assert audit['source_equal'] and audit['ast_equal'] and not audit['compound_change']
    assert all(row['status'] == 'unchanged' for row in audit['units'])
    assert indexed(audit)['outer.helper']['parent']['duplicate_name'] is True
    assert indexed(audit)['outer.helper#2']['parent']['occurrence'] == 2


def test_nested_edit_changes_subtree_but_does_not_claim_outer_local_edit():
    parent = 'def outer():\n    def inner():\n        return 1\n    return inner()\n'
    audit = build_change_audit(parent, parent.replace('return 1', 'return 2'), declared_targets=['outer'])
    rows = indexed(audit)
    assert rows['outer.inner']['edit_scope'] == 'direct'
    assert rows['outer']['edit_scope'] == rows['<module>']['edit_scope'] == 'descendant_only'
    assert rows['outer']['parent']['local_ast_sha256'] == rows['outer']['candidate']['local_ast_sha256']
    assert audit['direct_edited_unit_ids'] == ['outer.inner'] and not audit['compound_change']
    assert audit['declaration']['descendant_only_targets'] == ['outer']
    assert audit['declaration']['unchanged_targets'] == []
    assert audit['declaration']['undeclared_edited_units'] == ['outer.inner']


@pytest.mark.parametrize('before,after,target', [
    ('VALUE = 1\n', 'VALUE = 2\n', '<module>'),
    ('import os\n', 'import sys\n', '<module>'),
    ('def f():\n    return 1\n', 'def f():\n    return 2\n', 'f'),
    ('def f(x=1):\n    return x\n', 'def f(x=2):\n    return x\n', 'f'),
    ('@deco(1)\ndef f():\n    return 1\n', '@deco(2)\ndef f():\n    return 1\n', 'f'),
    ('def f(x: int):\n    return x\n', 'def f(x: str):\n    return x\n', 'f'),
    ('def f():\n    """old"""\n    return 1\n', 'def f():\n    """new"""\n    return 1\n', 'f'),
    ('def f():\n    return 1\n', 'async def f():\n    return 1\n', 'f'),
    ('class C(A):\n    pass\n', 'class C(B):\n    pass\n', 'C'),
    ('def f():\n    global a\n    a=1\n', 'def f():\n    global b\n    b=1\n', 'f'),
])
def test_constants_docstrings_headers_and_executable_module_fields_are_preserved(before, after, target):
    audit = build_change_audit(before, after, declared_targets=[target])
    assert not audit['ast_equal']
    assert audit['direct_edited_unit_ids'] == [target]
    assert indexed(audit)[target]['status'] == 'modified'
    assert not audit['declaration']['undeclared_edited_units']


def test_comment_formatting_literal_spelling_change_is_text_only_not_new_ast():
    before = 'def f():\n    return "a"\n'
    after = '# explanation\n\ndef f( ):\n    # interior comment\n    return \'a\'\n'
    audit = build_change_audit(before, after, declared_targets=['f'])
    assert audit['ast_equal'] is True and audit['source_equal'] is False and audit['text_only_change'] is True
    assert audit['direct_edited_unit_ids'] == [] and audit['declaration']['unchanged_targets'] == ['f']
    assert indexed(audit)['f']['edit_scope'] == 'text_only'
    assert not audit['compound_change']


def test_line_moves_do_not_change_type_ignore_ast_but_type_comment_content_does():
    before = 'x = 1  # type: ignore[name-defined]\n'
    same = '\n\n' + before
    assert build_change_audit(before, same)['ast_equal']
    assert not build_change_audit(before, before.replace('name-defined', 'assignment'))['ast_equal']


def test_one_added_function_is_not_counted_as_two_independent_edits():
    before = 'LIMIT = 1\n'
    after = before + 'def helper():\n    return LIMIT\n'
    audit = build_change_audit(before, after, declared_targets=['helper'])
    assert indexed(audit)['<module>']['edit_scope'] == 'definition_structure_only'
    assert indexed(audit)['helper']['status'] == 'added'
    assert audit['direct_edited_unit_ids'] == ['helper'] and not audit['compound_change']
    assert audit['declaration']['undeclared_edited_units'] == []
    removed = build_change_audit(after, before, declared_targets=['helper'])
    assert indexed(removed)['helper']['status'] == 'removed'
    assert removed['declaration']['missing_targets'] == []


def test_compound_change_and_unchanged_declared_target_are_exact_structural_diagnostics():
    before = 'LIMIT = 1\ndef f():\n    return 1\ndef untouched():\n    return 2\n'
    after = before.replace('LIMIT = 1', 'LIMIT = 2').replace('return 1', 'return 3')
    audit = build_change_audit(before, after, declared_targets=['untouched', 'absent'])
    assert audit['compound_change'] is True
    assert audit['direct_edited_unit_ids'] == ['<module>', 'f']
    assert audit['declaration']['unchanged_targets'] == ['untouched']
    assert audit['declaration']['missing_targets'] == ['absent']
    assert audit['declaration']['undeclared_edited_units'] == ['<module>', 'f']
    unavailable = build_change_audit(before, after)
    assert unavailable['declaration']['available'] is False
    assert unavailable['declaration']['undeclared_edited_units'] == []


def test_final_grafted_function_is_inventoried_as_unchanged_without_treating_prompt_as_source():
    parent = 'def solve():\n    return 7\n'
    final = '"""Repaired leading prose."""\ndef helper():\n    return 2\n\n' + parent
    audit = build_change_audit(parent, final, declared_targets=['solve'])
    assert indexed(audit)['solve']['status'] == 'unchanged'
    assert audit['declaration']['unchanged_targets'] == ['solve']
    assert audit['declaration']['undeclared_edited_units'] == ['<module>', 'helper']
    assert audit['candidate']['source_sha256'] == hashlib.sha256(final.encode()).hexdigest()


def test_unicode_literal_and_decorator_source_spans_use_python_byte_columns_and_line_rules():
    source = 'note = "α\u2028β"\n@deco("é")\ndef f():\n    return "γ"\n'
    audit = build_change_audit(source, source)
    actual = indexed(audit)['f']['parent']
    assert actual['span'] == dict(start_line=2, start_utf8_column=0, end_line=4, end_utf8_column=len('    return "γ"'.encode()))
    exact = '@deco("é")\ndef f():\n    return "γ"'
    assert actual['source_sha256'] == hashlib.sha256(exact.encode()).hexdigest()


def test_parse_failure_is_unavailable_not_a_claim_that_all_parent_functions_were_removed():
    audit = build_change_audit('def f():\n    return 1\n', 'def broken(:\n', declared_targets=['f'])
    assert audit['comparison_available'] is False
    assert audit['ast_equal'] is audit['compound_change'] is None
    assert audit['candidate']['parse_error']['kind'] == 'SyntaxError'
    assert audit['candidate']['source_sha256']
    assert all(row['status'] == 'unavailable' for row in audit['units'])
    assert audit['declaration']['unchanged_targets'] == audit['declaration']['undeclared_edited_units'] == []
    assert unit_catalog('def broken(:\n') == []


def test_full_artifact_keeps_all_changes_but_summary_is_bounded_and_hash_verified():
    before = '\n'.join(f'def f{i}():\n    return 1\n' for i in range(100))
    after = before.replace('return 1', 'return 2')
    audit = build_change_audit(before, after, declared_targets=['f99'])
    original = deepcopy(audit)
    summary = summarize_change_audit(audit, max_units=3)
    assert len(audit['units']) == 101 and len(audit['direct_edited_unit_ids']) == 100
    assert len(summary['units']) == 3 and summary['omitted_edited_units'] == 98
    assert summary['direct_edited_units_total'] == 100
    assert summary['declaration']['undeclared_edited_units_total'] == 99
    assert len(summary['declaration']['undeclared_edited_units']) == 3
    assert summary['artifact_sha256'] == audit['artifact_sha256'] and audit == original
    body = {key: value for key, value in audit.items() if key != 'artifact_sha256'}
    assert audit['artifact_sha256'] == hashlib.sha256(canonical_json(body).encode()).hexdigest()
    audit['units'][1]['status'] = 'unchanged'
    with pytest.raises(ValueError, match='identity mismatch'):
        summarize_change_audit(audit)


def test_long_unit_ids_are_not_silently_reused_as_truncated_exact_ids():
    name = 'function_' + 'x' * 1000
    before = f'def {name}():\n    return 1\n'
    after = before.replace('return 1', 'return 2')
    summary = summarize_change_audit(build_change_audit(before, after), max_units=2)
    selected = summary['units'][1]
    assert selected['id'] is None and selected['truncated'] is True
    assert len(selected['label']) == 160 and selected['id_sha256'] == hashlib.sha256(name.encode()).hexdigest()
    assert len(json.dumps(summary)) < 5000


@pytest.mark.parametrize('targets', ['f', [None], [''], {'f': True}])
def test_invalid_declared_target_shape_does_not_become_an_empty_valid_declaration(targets):
    with pytest.raises(ValueError):
        build_change_audit('x=1', 'x=2', declared_targets=targets)
