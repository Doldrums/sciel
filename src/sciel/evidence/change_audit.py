"""Complete, pure AST change inventory for a parent's final repaired candidate.

Hashes retain constants, docstrings, annotations, defaults and decorators. They
omit source locations, not executable AST fields. Structural correspondence is
not semantic equivalence, causal evidence or proof that a model's prose is true.
"""
from __future__ import annotations

import ast
from collections import Counter
from copy import deepcopy
import hashlib
import json
import re
import sys

SCHEMA = 'sciel-change-audit/1'
AST_SCHEMA = 'sciel-unit-ast-fields/1'
_DEFINITIONS = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
_LOCATIONS = {'lineno', 'col_offset', 'end_lineno', 'end_col_offset'}
_MODULE = '<module>'


def canonical_json(value):
    """Canonical artifact encoding; its digest excludes artifact_sha256 itself."""
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _sha(text):
    return hashlib.sha256(text.encode('utf-8', errors='surrogatepass')).hexdigest()


def _kind(node):
    return ('module' if isinstance(node, ast.Module) else 'class' if isinstance(node, ast.ClassDef)
            else 'async_function' if isinstance(node, ast.AsyncFunctionDef) else 'function')


def _ast_hash(root, unit_ids, *, mode='subtree'):
    """Iterative tagged encoding; local hashes replace only nested definitions.

    The separate nondefinition hash removes nested definition statements. It
    identifies container changes forced solely by adding/removing definitions,
    without labeling them independent executable edits.
    """
    digest = hashlib.sha256()
    stack = [('node', root)]
    while stack:
        tag, value = stack.pop()
        if tag == 'literal':
            digest.update(value.encode('utf-8', errors='surrogatepass'))
            continue
        if isinstance(value, ast.AST):
            if value is not root and isinstance(value, _DEFINITIONS) and mode != 'subtree':
                token = ['definition', unit_ids[id(value)]] if mode == 'local' else ['definition_omitted']
                digest.update(canonical_json(token).encode())
                continue
            digest.update(('(' + type(value).__name__).encode())
            stack.append(('literal', ')'))
            fields = [(key, item) for key, item in ast.iter_fields(value) if key not in _LOCATIONS]
            for key, item in reversed(fields):
                stack.extend([('node', item), ('literal', canonical_json(key) + '=')])
        elif isinstance(value, list):
            digest.update(b'[')
            stack.append(('literal', ']'))
            for item in reversed(value):
                if mode == 'nondefinition' and isinstance(item, _DEFINITIONS):
                    continue
                stack.append(('node', item))
        else:
            # repr plus explicit Python type distinguishes int/bool, bytes/str,
            # signed constants and all literal contents including docstrings.
            digest.update(canonical_json([type(value).__name__, repr(value)]).encode())
    return digest.hexdigest()


def _inventory(source):
    if type(source) is not str:
        raise TypeError('source must be text')
    result = dict(source_sha256=_sha(source), chars=len(source), ast_sha256=None,
                  parse_error=None, units=[])
    try:
        tree = ast.parse(source, type_comments=True)
    except (SyntaxError, ValueError, RecursionError, UnicodeError) as exc:
        result['parse_error'] = dict(kind=type(exc).__name__,
            **({'line': exc.lineno, 'column': exc.offset} if isinstance(exc, SyntaxError) else {}))
        return result
    nodes, ids, occurrences = [], {}, Counter()
    stack = [(tree, None)]
    while stack:
        node, owner = stack.pop()
        if isinstance(node, (ast.Module, *_DEFINITIONS)):
            if isinstance(node, ast.Module):
                ident, occurrence = _MODULE, 1
            else:
                occurrences[(owner, node.name)] += 1
                occurrence = occurrences[(owner, node.name)]
                name = node.name + (f'#{occurrence}' if occurrence > 1 else '')
                ident = name if owner == _MODULE else owner + '.' + name
            ids[id(node)] = ident
            nodes.append((node, ident, owner, occurrence))
            owner = ident
        stack.extend((child, owner) for child in reversed(list(ast.iter_child_nodes(node))))
    # Python source line numbers follow CR/LF, not every Unicode splitlines()
    # separator that may occur literally inside a string or comment.
    lines, offset = [], 0
    for match in re.finditer(r'\r\n|\r|\n', source):
        lines.append(source[offset:match.end()])
        offset = match.end()
    if offset < len(source):
        lines.append(source[offset:])
    for node, ident, owner, occurrence in nodes:
        if isinstance(node, ast.Module):
            span, excerpt = None, source
        else:
            start = min([node.lineno] + [part.lineno for part in node.decorator_list])
            end = node.end_lineno
            # AST columns are UTF-8 byte offsets, not Unicode character indices.
            selected = [line.encode('utf-8', errors='surrogatepass') for line in lines[start - 1:end]]
            if len(selected) == 1:
                excerpt = selected[0][node.col_offset:node.end_col_offset]
            else:
                excerpt = selected[0][node.col_offset:] + b''.join(selected[1:-1]) + selected[-1][:node.end_col_offset]
            excerpt = excerpt.decode('utf-8', errors='surrogatepass')
            span = dict(start_line=start, start_utf8_column=node.col_offset,
                        end_line=end, end_utf8_column=node.end_col_offset)
        result['units'].append(dict(id=ident, kind=_kind(node), parent_id=owner,
            occurrence=occurrence,
            duplicate_name=(False if owner is None else occurrences[(owner, node.name)] > 1),
            source_sha256=_sha(excerpt), span=span,
            local_ast_sha256=_ast_hash(node, ids, mode='local'),
            nondefinition_ast_sha256=_ast_hash(node, ids, mode='nondefinition'),
            subtree_ast_sha256=_ast_hash(node, ids)))
    result['ast_sha256'] = result['units'][0]['subtree_ast_sha256']
    return result


def unit_catalog(source):
    """All lexical units for an exact source; an unparseable source has no units.

    IDs use qualified lexical names; repeat definitions in the same scope have
    #2, #3 suffixes in source order. These are syntactic IDs, not rename tracking.
    """
    return [{key: unit[key] for key in ('id', 'kind', 'local_ast_sha256', 'subtree_ast_sha256')}
            for unit in _inventory(source)['units']]


def build_change_audit(parent_source, candidate_source, *, declared_targets=None):
    """Inventory ALL units against the final post-repair/graft source supplied.

    None means no valid declaration is available. [] explicitly declares no
    edited targets. Names absent on both sides remain missing_targets. No source
    is executed, truncated, repaired, imported or guessed from a model response.
    """
    if declared_targets is not None and (type(declared_targets) is not list
            or any(type(ident) is not str or not ident for ident in declared_targets)):
        raise ValueError('declared_targets must be a list of nonempty unit IDs or None')
    declared = list(dict.fromkeys(declared_targets or []))
    parent, candidate = _inventory(parent_source), _inventory(candidate_source)
    comparable = parent['parse_error'] is candidate['parse_error'] is None
    previous = {unit['id']: unit for unit in parent['units']}
    following = {unit['id']: unit for unit in candidate['units']}
    units = []
    for ident in dict.fromkeys([*previous, *following]):
        before, after = previous.get(ident), following.get(ident)
        if not comparable:
            status, scope = 'unavailable', 'unavailable'
        elif before is None:
            status, scope = 'added', 'direct'
        elif after is None:
            status, scope = 'removed', 'direct'
        elif before['local_ast_sha256'] != after['local_ast_sha256']:
            status = 'modified'
            scope = ('definition_structure_only' if before['nondefinition_ast_sha256'] == after['nondefinition_ast_sha256']
                     else 'direct')
        elif before['subtree_ast_sha256'] != after['subtree_ast_sha256']:
            status, scope = 'modified', 'descendant_only'
        elif before['source_sha256'] != after['source_sha256']:
            status, scope = 'unchanged', 'text_only'
        else:
            status, scope = 'unchanged', 'unchanged'
        units.append(dict(id=ident, kind=(after or before)['kind'], status=status, edit_scope=scope,
                          parent=before, candidate=after))
    by_id = {unit['id']: unit for unit in units}
    direct = [unit['id'] for unit in units if unit['edit_scope'] == 'direct']
    same_ast = parent['ast_sha256'] == candidate['ast_sha256'] if comparable else None
    declaration = dict(available=declared_targets is not None, targets=declared,
        comparison_available=comparable, missing_targets=[], unchanged_targets=[],
        descendant_only_targets=[], undeclared_edited_units=[])
    if comparable and declared_targets is not None:
        declaration.update(
            missing_targets=[ident for ident in declared if ident not in by_id],
            unchanged_targets=[ident for ident in declared if ident in by_id and by_id[ident]['status'] == 'unchanged'],
            descendant_only_targets=[ident for ident in declared if ident in by_id and by_id[ident]['edit_scope'] == 'descendant_only'],
            undeclared_edited_units=[ident for ident in direct if ident not in declared])
    result = dict(schema=SCHEMA, ast_policy=dict(schema=AST_SCHEMA,
        python_ast_version=f'{sys.version_info.major}.{sys.version_info.minor}', type_comments=True),
        parent={key: value for key, value in parent.items() if key != 'units'},
        candidate={key: value for key, value in candidate.items() if key != 'units'},
        comparison_available=comparable,
        source_equal=parent['source_sha256'] == candidate['source_sha256'], ast_equal=same_ast,
        text_only_change=(same_ast and parent['source_sha256'] != candidate['source_sha256']) if comparable else None,
        units=units, direct_edited_unit_ids=direct,
        counts={key: sum(unit['status'] == key for unit in units)
                for key in ('added', 'removed', 'modified', 'unchanged', 'unavailable')},
        compound_change=len(direct) > 1 if comparable else None, declaration=declaration,
        limitations=[
            'AST structure is not behavioral equivalence or a causal mechanism proof.',
            'Text-only change can include comments, formatting or equivalent literal spellings.',
            'Nested definitions use lexical IDs; duplicate names have source-order suffixes. Renames and moves are remove/add, not inferred correspondences.',
            'Container definition-structure-only changes are listed but excluded from direct edit and compound counts.',
            'Declared targets are compared as exact IDs; descendant-only and undeclared units are structural diagnostics, not semantic mismatch verdicts.',
            'Unit source spans include decorators and AST extent; unattached comments remain visible in the module source hash.',
        ])
    result['artifact_sha256'] = _sha(canonical_json(result))
    return result


def summarize_change_audit(audit, *, max_units=12):
    """Bounded detached UI projection, bound to the complete canonical artifact."""
    if type(max_units) is not int or not 0 <= max_units <= 100:
        raise ValueError('max_units must be an integer from 0 to 100')
    body = {key: value for key, value in audit.items() if key != 'artifact_sha256'}
    if audit.get('schema') != SCHEMA or audit.get('artifact_sha256') != _sha(canonical_json(body)):
        raise ValueError('change audit artifact identity mismatch')
    def identity(ident):
        return dict(id=ident if len(ident) <= 160 else None, label=ident if len(ident) <= 160 else ident[:157] + '...',
                    id_sha256=_sha(ident), truncated=len(ident) > 160)
    edited = [unit for unit in audit['units'] if unit['edit_scope'] not in ('unchanged', 'text_only')]
    result = {key: deepcopy(audit[key]) for key in ('schema', 'artifact_sha256', 'ast_policy', 'parent', 'candidate',
        'comparison_available', 'source_equal', 'ast_equal', 'text_only_change', 'counts', 'compound_change')}
    result['units'] = [{**identity(unit['id']), 'kind': unit['kind'], 'status': unit['status'],
                        'edit_scope': unit['edit_scope']} for unit in edited[:max_units]]
    result['units_total'] = len(audit['units'])
    result['edited_units_total'] = len(edited)
    result['omitted_edited_units'] = len(edited) - len(result['units'])
    result['direct_edited_units_total'] = len(audit['direct_edited_unit_ids'])
    declaration = audit['declaration']
    result['declaration'] = {key: declaration[key] for key in ('available', 'comparison_available')}
    for key in ('targets', 'missing_targets', 'unchanged_targets', 'descendant_only_targets', 'undeclared_edited_units'):
        result['declaration'][key] = [identity(ident) for ident in declaration[key][:max_units]]
        result['declaration'][key + '_total'] = len(declaration[key])
    result['scope'] = 'Bounded structural summary. The full inventory is in the hash-bound artifact; no semantic or causal verification.'
    return result
