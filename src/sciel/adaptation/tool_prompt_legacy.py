"""Trusted historical tool prompt renderers, never loaded from evidence.

The historical renderer bodies and guidance constants preserve source601
(controller016ea...) semantics. The older named native004 profile changes only
its known Python/JSON hint on this frozen renderer. Other helper source bytes are
pinned at runtime; changing them requires a separately reviewed compatibility
revision. Exact reconstructed-prompt comparison remains in ToolAncestry.

The finite /24 and optional-output profiles also retain their exact field hints
on the explicitly pinned /25 or /26 renderer logic. The /25 profile preserves
its argument guidance; /26 changes only dispatch budget guards, not prompt text. No stored controller code is loaded.
"""
from copy import deepcopy
from pathlib import Path
from types import FunctionType, MethodType
import ast
import hashlib
import json
from typing import TYPE_CHECKING

from .tool_prompt_legacy_contract import (
    CONTROLLER_601,
    CONTROLLER_004,
    CONTROLLER_24,
    _OUTPUT_CAP_CONTROLLER,
    _OUTPUT_CAP_CONFIG,
    CONTROLLER_OUTPUT_CAP,
    _ARGUMENT_GUIDANCE_CONTROLLER,
    CONTROLLER_ARGUMENT_GUIDANCE,
    _TOKEN_BUDGET_CONTROLLER,
    _TOKEN_BUDGET_CONFIG,
    PROFILES,
    _V24_PYTHON_CODE_GUIDANCE,
    _IMPORTS_SHA256,
    _CLASS_HEADER_SHA256,
    _SHARED_NODES,
    _DEPENDENCY_FILES,
    _SHARED_GLOBALS,
    _PYTHON_CODE_GUIDANCE,
    _RESEARCH_PERMISSION
)
from .tool_prompt_legacy_renderer import (
    _compact_search_evidence_prompt, _compact_tool_instructions, _target_prompt, _prompt)

def _shared_namespace():
    from ..research_tools import controller as shared
    path = Path(shared.__file__)
    text = path.read_text()
    tree = ast.parse(text)
    imports = '\n'.join(ast.get_source_segment(text, node) for node in tree.body
                        if isinstance(node, (ast.Import, ast.ImportFrom)))
    if hashlib.sha256(imports.encode()).hexdigest() not in _IMPORTS_SHA256:
        raise ValueError('unsupported historical tool prompt import bindings')
    cls = next(node for node in tree.body
               if isinstance(node, ast.ClassDef) and node.name == 'ResearchTools')
    header = '\n'.join(text.splitlines()[cls.lineno-1:cls.body[0].lineno-1])
    if hashlib.sha256(header.encode()).hexdigest() != _CLASS_HEADER_SHA256:
        raise ValueError('unsupported historical tool prompt class bindings')
    nodes = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            nodes[node.name] = node
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    nodes[target.id] = node
    for node in nodes['ResearchTools'].body:
        if isinstance(node, ast.FunctionDef):
            nodes[node.name] = node
    for name, expected in _SHARED_NODES.items():
        node = nodes.get(name)
        raw = ast.get_source_segment(text, node).encode() if node is not None else b''
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('unsupported historical tool prompt helper bytes: ' + name)
    source_root = path.parents[1]
    for relative, expected in _DEPENDENCY_FILES.items():
        allowed = {expected, _OUTPUT_CAP_CONFIG, _TOKEN_BUDGET_CONFIG} if relative == 'research_tools/config.py' else {expected}
        if hashlib.sha256((source_root / relative).read_bytes()).hexdigest() not in allowed:
            raise ValueError('unsupported historical tool prompt dependency bytes: ' + relative)
    # Explicit installed/trusted symbols only; no globals or code from artifacts.
    namespace = {name: getattr(shared, name) for name in _SHARED_GLOBALS}
    namespace.update(json=json, deepcopy=deepcopy,
                     _PYTHON_CODE_GUIDANCE=_PYTHON_CODE_GUIDANCE,
                     _RESEARCH_PERMISSION=_RESEARCH_PERMISSION)
    return namespace


def bind(controller, recorded):
    if recorded not in PROFILES:
        raise ValueError('unsupported historical tool prompt renderer')
    if 'context_tokens' in getattr(controller, 'config', {}):
        raise ValueError('historical tool renderer cannot accept context_tokens')
    if 'max_tokens' in getattr(controller, 'config', {}) and recorded not in (CONTROLLER_OUTPUT_CAP, CONTROLLER_ARGUMENT_GUIDANCE):
        raise ValueError('historical tool renderer cannot accept explicit max_tokens')
    namespace = _shared_namespace()
    if recorded in (CONTROLLER_24, CONTROLLER_OUTPUT_CAP, CONTROLLER_ARGUMENT_GUIDANCE):
        from ..research_tools import controller as current
        if hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest() not in {
            _ARGUMENT_GUIDANCE_CONTROLLER, _TOKEN_BUDGET_CONTROLLER,
        }:
            raise ValueError('unsupported historical /24-/25 tool prompt bytes')
        # /26 leaves the /25 renderer unchanged and adds dispatch-only bounds.
        # /24 differs only in field hints. Pin installed source before borrowing code,
        # and enumerate trusted bindings instead of importing artifact globals.
        names = ('PROMPT_MARKER', '_ACTION_GUIDANCE', '_DEPENDENCY_GUIDANCE',
            '_QUEUE_GUIDANCE', '_RESEARCH_PERMISSION', '_YIELD_GUIDANCE', '_clip',
            '_compact_search_evidence_prompt', '_experiment_guidance', '_feedback_card',
            '_research_results_prompt', '_sha', '_subject_prompt', 'add_canonical_diagnostics',
            'canonical_bytes', 'deepcopy', 'json', 'shared_evidence', 'state_prompt', 'validate_targets')
        namespace = {name: getattr(current, name) for name in names}
        if recorded == CONTROLLER_ARGUMENT_GUIDANCE:
            namespace.update(_PYTHON_CODE_GUIDANCE=current._PYTHON_CODE_GUIDANCE,
                _ARGUMENTS_COMPACT_GUIDANCE=current._ARGUMENTS_COMPACT_GUIDANCE,
                _ARGUMENTS_FULL_GUIDANCE=current._ARGUMENTS_FULL_GUIDANCE)
        else:
            namespace.update(_PYTHON_CODE_GUIDANCE=_V24_PYTHON_CODE_GUIDANCE,
                             _ARGUMENTS_COMPACT_GUIDANCE='arguments_json=object.\n',
                             _ARGUMENTS_FULL_GUIDANCE='arguments_json: object. Repairs count; immutable versions.\n')
        for original in (current._compact_tool_instructions, current.ResearchTools._prompt):
            function = FunctionType(original.__code__, namespace, original.__name__, original.__defaults__)
            function.__kwdefaults__ = original.__kwdefaults__
            namespace[original.__name__] = function
        controller._prompt = MethodType(namespace['_prompt'], controller)
        return controller
    for original in (_compact_search_evidence_prompt, _compact_tool_instructions,
                     _target_prompt, _prompt):
        name = original.__name__
        function = FunctionType(original.__code__, namespace, name, original.__defaults__)
        function.__kwdefaults__ = original.__kwdefaults__
        namespace[name] = function
    if recorded == CONTROLLER_004:
        fields = '{"schema":"sciel-tool-result/1","status":"success","observations":{},"diagnostics":{},"error":null}'
        modern_hint = namespace['_PYTHON_CODE_GUIDANCE']
        compact_renderer = namespace['_compact_tool_instructions']
        def compact(sequential, **options):
            if sequential:
                raise ValueError('unsupported historical sequential tool prompt')
            # Temporarily render the reviewed source601 compact body with its
            # own static hint. Avoid shared mutable state between renderers.
            local = dict(namespace, _PYTHON_CODE_GUIDANCE=modern_hint)
            original = compact_renderer
            function = FunctionType(original.__code__, local, original.__name__, original.__defaults__)
            function.__kwdefaults__ = original.__kwdefaults__
            text = function(False, **options).replace(modern_hint, '')
            marker = 'Name source/hash; replay pins inputs.\n'
            return text.replace(marker, marker + 'run(payload) returns exactly five fields: ' + fields + '.\n')
        namespace['_PYTHON_CODE_GUIDANCE'] = 'run(payload) returns exactly five fields:\n' + fields + '\n'
        namespace['_compact_tool_instructions'] = compact
    controller._target_prompt = namespace['_target_prompt']  # Original method is static.
    controller._prompt = MethodType(namespace['_prompt'], controller)
    return controller
