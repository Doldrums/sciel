"""Future typed-body-only local-list policy; the legacy helper guard is unchanged.

This module parses/compiles syntax only. It neither rewrites submitted source nor
executes a helper. It is an interface restriction, not a security/resource proof.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

from . import circle_proposals as legacy

POLICY_VERSION = 'sciel-circle-body-local-lists/1'
PINNED_PROPOSALS_SHA256 = '5def9eb52c65962e62c9269f1f953a715b2ba3adfed791c37a6cf35380c4cbb9'
POLICY = {
    'version': POLICY_VERSION,
    'method': 'append',
    'receiver': 'A simple local name, not a parameter or an alias receiver.',
    'declaration': 'Exactly one Name Store in the complete helper AST, from a single-target '
        'top-level assignment to a list literal, synchronous list comprehension, or list() '
        'with zero or one positional argument and no keywords.',
    'dominance': 'The declaration precedes the top-level statement containing the append call.',
    'call': 'Standalone expression statement receiver.append(value), exactly one positional argument '
        'and no keywords. The bound append attribute cannot be read, stored, returned, or passed.',
    'unchanged': 'All other syntax, names, module values/calls and attributes use the pinned legacy guard. '
        'extend/insert and array/parameter/alias receivers remain unavailable.',
    'scope': 'Typed-body validation only. No source rewriting, execution, allocation/time proof, '
        'retroactive reclassification of old attempts, or change to the legacy full-helper contract.',
}


class LocalListPolicyError(ValueError):
    def __init__(self, message, node):
        super().__init__(message)
        self.node = node  # Trusted AST node from the original submitted helper.


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _check_legacy():
    if _sha(Path(legacy.__file__).read_bytes()) != PINNED_PROPOSALS_SHA256:
        raise ValueError('local-list policy: pinned legacy guard changed')


def identity():
    _check_legacy()
    return dict(version=POLICY_VERSION,source_sha256=_sha(Path(__file__).read_bytes()),
        legacy_guard_sha256=PINNED_PROPOSALS_SHA256,
        policy_sha256=_sha(json.dumps(POLICY,sort_keys=True,separators=(',',':')).encode()))


def _fresh_list(value):
    return (type(value) in (ast.List,ast.ListComp) or
        (type(value) is ast.Call and type(value.func) is ast.Name and value.func.id=='list'
         and len(value.args)<=1 and not value.keywords))


def _approved_appends(function,nodes):
    """Static sufficient proof; deliberately reject valid but unproved programs."""
    parents={id(child):parent for parent in nodes for child in ast.iter_child_nodes(parent)}
    stores={}
    for node in nodes:
        if type(node) is ast.Name and type(node.ctx) is ast.Store:
            stores.setdefault(node.id,[]).append(node)
    declarations={}
    owners={id(node):index for index,statement in enumerate(function.body) for node in ast.walk(statement)}
    for index,statement in enumerate(function.body):
        if (type(statement) is ast.Assign and len(statement.targets)==1
                and type(statement.targets[0]) is ast.Name and _fresh_list(statement.value)):
            name=statement.targets[0].id
            if name not in legacy.ARGUMENTS and len(stores[name])==1:
                declarations[name]=index
    approved=set()
    for node in nodes:
        if type(node) is not ast.Attribute or node.attr!='append':
            continue
        call=parents.get(id(node));statement=parents.get(id(call))
        valid=(type(node.ctx) is ast.Load and type(node.value) is ast.Name
            and node.value.id in declarations and type(call) is ast.Call and call.func is node
            and len(call.args)==1 and not call.keywords and type(statement) is ast.Expr
            and statement.value is call and declarations[node.value.id]<owners.get(id(node),-1))
        if not valid:
            raise LocalListPolicyError('local-list append requires a once-declared dominating local list and a standalone one-argument call',node)
        approved.add(id(node))
    return approved


def _validate_extended(text):
    # This branch repeats the pinned guard's structural checks while sharing its
    # exact constant tables and ordinary attribute validator. The only exception
    # is the independently proved append Attribute/Call pair. No synthetic AST or
    # replacement source is passed to the old guard.
    require=legacy._require
    require(type(text) is str,'helper must be source text')
    try:raw=text.encode('utf-8')
    except UnicodeEncodeError as exc:raise ValueError('helper must be strict UTF8') from exc
    require(0<len(raw)<=legacy.MAX_HELPER_BYTES,'helper exceeds source byte limit')
    normalized=text.strip('\r\n')+'\n'
    require(len(normalized.encode())<=legacy.MAX_HELPER_BYTES,'normalized helper exceeds source byte limit')
    require(normalized.startswith('def propose_restart('),'only the exact helper function is accepted')
    try:
        tree=ast.parse(normalized)
        compile(tree,'<circle-body-local-list-interface>','exec')
    except (SyntaxError,ValueError,RecursionError) as exc:raise ValueError('invalid helper syntax') from exc
    require(len(tree.body)==1 and type(tree.body[0]) is ast.FunctionDef,'helper must contain exactly one function')
    function=tree.body[0];args=function.args
    require(function.name=='propose_restart' and not function.decorator_list and function.returns is None
        and not function.type_comment and not getattr(function,'type_params',[]),'helper signature/decorators differ')
    require(tuple(a.arg for a in args.args)==legacy.ARGUMENTS and not args.posonlyargs and not args.kwonlyargs
        and args.vararg is None and args.kwarg is None and not args.defaults and not args.kw_defaults
        and all(a.annotation is None and not a.type_comment for a in args.args),
        'helper requires the five exact unannotated positional arguments')
    nodes=list(ast.walk(tree))
    require(len(nodes)<=legacy.MAX_AST_NODES,'helper AST exceeds node limit')
    require(sum(type(n) is ast.FunctionDef for n in nodes)==1,'nested definitions are forbidden')
    reserved=legacy._BUILTINS|{'np','math','propose_restart'};locals_=set(legacy.ARGUMENTS)
    for node in nodes:
        require(type(node) in legacy._NODES,'helper syntax outside surface: '+type(node).__name__)
        if isinstance(node,ast.Name) and isinstance(node.ctx,ast.Store):
            require(node.id not in reserved and '__' not in node.id,'reserved name cannot be assigned')
            locals_.add(node.id)
    require(any(isinstance(n,ast.Return) for n in nodes),'helper must contain a return')
    approved=_approved_appends(function,nodes)
    for node in nodes:
        if isinstance(node,ast.Name):
            require('__' not in node.id and node.id in locals_|legacy._BUILTINS|{'np','math'},'helper uses an unavailable name')
        elif isinstance(node,ast.Attribute):
            if id(node) not in approved:legacy._attribute(node)
        elif isinstance(node,ast.Call):
            require(all(k.arg is not None for k in node.keywords),'dynamic keyword expansion is forbidden')
            callee=node.func
            if isinstance(callee,ast.Name):
                require(callee.id in legacy._BUILTINS,'dynamic/recursive calls are forbidden')
            elif isinstance(callee,ast.Attribute):
                if id(callee) not in approved:
                    legacy._attribute(callee)
                    if isinstance(callee.value,ast.Name) and callee.value.id in ('np','math'):
                        choices=legacy._NUMPY_CALLS if callee.value.id=='np' else legacy._MATH_CALLS
                        require(callee.attr in choices,'module value is not a permitted callable')
                    else:require(callee.attr in legacy._ARRAY_METHODS,'only explicit array methods can be called')
            else:raise ValueError('dynamic calls are forbidden')
        elif isinstance(node,ast.Constant):
            require(type(node.value) in (int,float,str,bool,type(None)),'unsupported literal')
            if type(node.value) is float:require(node.value==node.value and abs(node.value)!=float('inf'),'nonfinite literal')
        elif isinstance(node,ast.keyword):require(node.arg is not None and '__' not in node.arg,'invalid keyword')
        elif isinstance(node,ast.comprehension):require(node.is_async==0,'async comprehensions are forbidden')
    return normalized


def validate_helper(text):
    """Return exact legacy-normalized text; no body/helper execution or rewriting."""
    _check_legacy()
    try:
        return legacy.validate_helper(text)
    except ValueError as original:
        # Never parse text already rejected before the legacy parser. In
        # particular, bound characters before a second UTF8 encoding attempt.
        if type(text) is not str or len(text)>legacy.MAX_HELPER_BYTES:
            raise original
        try:
            raw=text.encode('utf-8')
            normalized=text.strip('\r\n')+'\n'
            normalized_bytes=normalized.encode('utf-8')
        except UnicodeError:
            raise original
        if (not 0<len(raw)<=legacy.MAX_HELPER_BYTES
                or len(normalized_bytes)>legacy.MAX_HELPER_BYTES
                or not normalized.startswith('def propose_restart(')):
            raise original
        # The extension is activated only for actual append attribute syntax.
        # Non-append failures retain the original diagnostics and traceback.
        try:tree=ast.parse(text)
        except (SyntaxError,ValueError,TypeError,RecursionError,UnicodeError):raise original
        if not any(type(n) is ast.Attribute and n.attr=='append' for n in ast.walk(tree)):
            raise original
    return _validate_extended(text)
