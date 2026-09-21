"""Viability contract: one typed Propose->Validate->Repair cycle for candidates.

The engine cannot constrain a closed API model's decoding with a grammar, so the
only available form of "constrained decoding" is post-hoc: classify what is wrong
with a completion into a typed Violation, then apply a declarative repair policy.
This module supplies classification and mechanical repairs. validation.validate_candidate
composes them with graft/duplicate/novelty decisions; run_worker owns requests and evals.

Violation kinds (checked in this priority order per attempt):
  empty      -- thinking-only / no code extracted
  syntax     -- does not parse (incl. the mid-docstring completion family)
  resource   -- statically-fatal in the sandbox (multiprocessing + local workers)
  interface  -- dropped an entry-point the evaluator calls (repaired by graft)
  duplicate  -- normalized code already seen on the island

`empty`, `syntax`, `resource` are PRE-EVAL: caught before spending a sandbox run.
`interface` is repaired unconditionally by graft (no-op when nothing was dropped).
`duplicate` is checked after graft. The repair table below is declarative; the
RepairPolicy indirection is where a learned policy will later live (the journal
already records grafted / empty_responses / gate_rejections outcomes)."""

from __future__ import annotations

import ast
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Violation:
    kind: str            # empty | syntax | resource | interface | duplicate
    detail: str = ""


def _docstring_repair(child: str) -> str | None:
    """One observed cheap syntax family: the completion starts mid-docstring
    (prose preamble swallowed the opening quotes). If prepending quotes makes the
    module parse, return the repaired text; else None."""
    if child.lstrip().startswith(('"""', "'''", "#", "import ", "from ", "def ",
                                  "class ", "@")):
        return None
    fixed = '"""' + child
    try:
        ast.parse(fixed)
        return fixed
    except SyntaxError:
        return None


_POOL_TYPES = {"multiprocessing.Pool", "multiprocessing.pool.Pool",
               "concurrent.futures.ProcessPoolExecutor",
               "concurrent.futures.process.ProcessPoolExecutor"}
_PROCESS_TYPES = {"multiprocessing.Process", "multiprocessing.context.Process",
                  "multiprocessing.context.SpawnProcess", "multiprocessing.context.ForkServerProcess"}
_POOL_METHODS = {"map", "imap", "imap_unordered", "starmap", "apply", "apply_async",
                 "map_async", "starmap_async", "submit"}


@dataclass
class _Scope:
    parent: _Scope | None = None
    kind: str = "module"
    bindings: dict = field(default_factory=dict)
    redirects: dict = field(default_factory=dict)

    def bind(self, name, value):
        scope = self.redirects.get(name, self)
        scope.bindings.setdefault(name, []).append(value)


class _ResourceScan(ast.NodeVisitor):
    """Resolve only unambiguous imports/aliases, respecting lexical shadowing.

    Bindings are collected before calls are inspected, so an import below a
    function definition and aliases such as ``P = mp.Pool`` work. An ambiguous
    reassignment is deliberately left for the evaluator, not called fatal here.
    """
    def __init__(self):
        self.scope = _Scope()
        self.calls = []
        self.parents = {}
        self._visiting = None

    def visit(self, node):
        parent = self._visiting
        self.parents[id(node)] = parent
        self._visiting = node
        try:
            return super().visit(node)
        finally:
            self._visiting = parent

    def _target(self, target, value=None):
        if isinstance(target, ast.Name):
            self.scope.bind(target.id, value)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for item in target.elts:
                self._target(item)
        elif isinstance(target, ast.Starred):
            self._target(target.value)

    def visit_Name(self, node):
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self._target(node)

    def visit_Import(self, node):
        for alias in node.names:
            self.scope.bind(alias.asname or alias.name.split(".")[0],
                            ("path", alias.name if alias.asname else alias.name.split(".")[0]))

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.scope.bind(alias.asname or alias.name,
                            ("path", f"{node.module}.{alias.name}") if not node.level else None)

    def visit_Assign(self, node):
        self.visit(node.value)
        for target in node.targets:
            self._target(target, ("expr", node.value, self.scope))

    def visit_AnnAssign(self, node):
        if node.value is not None:
            self.visit(node.value)
        self._target(node.target, ("expr", node.value, self.scope))

    def visit_NamedExpr(self, node):
        self.visit(node.value)
        self._target(node.target, ("expr", node.value, self.scope))

    def visit_With(self, node):
        for item in node.items:
            self.visit(item.context_expr)
            self._target(item.optional_vars, ("expr", item.context_expr, self.scope))
        for statement in node.body:
            self.visit(statement)

    visit_AsyncWith = visit_With

    def visit_Global(self, node):
        root = self.scope
        while root.parent is not None:
            root = root.parent
        self.scope.redirects.update((name, root) for name in node.names)

    def visit_Nonlocal(self, node):
        for name in node.names:
            outer = self.scope.parent
            while outer is not None and outer.parent is not None:
                if name in outer.bindings:
                    self.scope.redirects[name] = outer
                    break
                outer = outer.parent

    def _nested(self, node, kind):
        enclosing = self.scope
        parent = enclosing
        while parent.kind == "class":
            parent = parent.parent
        self.scope = _Scope(parent, kind)
        if kind == "function":
            args = node.args
            for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs, args.vararg, args.kwarg]:
                if arg is not None:
                    self.scope.bind(arg.arg, None)
        body = node.body if isinstance(node.body, list) else [node.body]
        for statement in body:
            self.visit(statement)
        self.scope = enclosing

    def visit_FunctionDef(self, node):
        outer = self.scope
        while outer.kind == "class":
            outer = outer.parent
        self.scope.bind(node.name, ("local", "function") if outer.kind == "function" else None)
        for expression in [*node.decorator_list, *node.args.defaults, *node.args.kw_defaults]:
            if expression is not None:
                self.visit(expression)
        self._nested(node, "function")

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_Lambda(self, node):
        for expression in [*node.args.defaults, *node.args.kw_defaults]:
            if expression is not None:
                self.visit(expression)
        self._nested(node, "function")

    def visit_ClassDef(self, node):
        self.scope.bind(node.name, None)
        for expression in [*node.decorator_list, *node.bases, *(k.value for k in node.keywords)]:
            self.visit(expression)
        self._nested(node, "class")

    def _comprehension(self, node):
        self.visit(node.generators[0].iter)
        enclosing = self.scope
        parent = enclosing.parent if enclosing.kind == "class" else enclosing
        self.scope = _Scope(parent, "comprehension")
        for i, generator in enumerate(node.generators):
            if i:
                self.visit(generator.iter)
            self._target(generator.target)
            for condition in generator.ifs:
                self.visit(condition)
        for expression in ([node.key, node.value] if isinstance(node, ast.DictComp) else [node.elt]):
            self.visit(expression)
        self.scope = enclosing

    visit_ListComp = visit_SetComp = visit_DictComp = visit_GeneratorExp = _comprehension

    def visit_Call(self, node):
        self.calls.append((node, self.scope))
        self.generic_visit(node)


def _resolve_resource(expr, scope, seen=frozenset()):
    if isinstance(expr, ast.Lambda):
        return "local", "lambda"
    if isinstance(expr, ast.Name):
        while scope is not None:
            scope = scope.redirects.get(expr.id, scope)
            if expr.id in scope.bindings:
                key = (id(scope), expr.id)
                if key in seen:
                    return None
                values = set()
                for binding in scope.bindings[expr.id]:
                    value = (_resolve_resource(binding[1], binding[2], seen | {key})
                             if binding is not None and binding[0] == "expr" else binding)
                    values.add(value)
                return next(iter(values)) if len(values) == 1 else None
            scope = scope.parent
    elif isinstance(expr, ast.Attribute):
        base = _resolve_resource(expr.value, scope, seen)
        if base is not None and base[0] == "path":
            return "path", f"{base[1]}.{expr.attr}"
        if base == ("pool", "") and expr.attr in _POOL_METHODS:
            return "worker_call", expr.attr
        if base is not None and base[0] == "context":
            if expr.attr == "Pool":
                return "path", "multiprocessing.Pool"
            if expr.attr == "Process":
                return "process", base[1]
    elif isinstance(expr, ast.Call):
        callee = _resolve_resource(expr.func, scope, seen)
        if callee is not None and callee[0] == "path":
            if callee[1] in _POOL_TYPES:
                return "pool", ""
            if callee[1] == "multiprocessing.get_context":
                method = next((kw.value for kw in expr.keywords if kw.arg == "method"),
                              expr.args[0] if expr.args else None)
                # Explicit fork does not pickle a Process target; pool submissions
                # still pickle workers under every start method.
                if method is None or isinstance(method, ast.Constant) and method.value is None:
                    return "context", "default"
                return "context", method.value if isinstance(method, ast.Constant) else "unknown"
    return None


def _statement_block(node, parents):
    while node is not None and not isinstance(node, ast.stmt):
        node = parents.get(id(node))
    parent = parents.get(id(node))
    if parent is not None:
        for field_name, value in ast.iter_fields(parent):
            if isinstance(value, list) and any(item is node for item in value):
                return (id(parent), field_name), node
    return None, node


def _guaranteed_fork(node, scope, scan):
    """Only a preceding unconditional setter in this same block proves fork.

    Setters in another function or a conditional branch do not establish the
    runtime context here. A later competing/unknown setter removes the proof.
    Pool workers and explicit spawn/forkserver contexts still require pickling.
    """
    prior = [(call, call_scope) for call, call_scope in scan.calls
             if (call.lineno, call.col_offset) < (node.lineno, node.col_offset)
             and _resolve_resource(call.func, call_scope) == ("path", "multiprocessing.set_start_method")]
    if not prior:
        return False
    setter, setter_scope = max(prior, key=lambda item: (item[0].lineno, item[0].col_offset))
    if setter_scope is not scope:
        return False
    setter_block, statement = _statement_block(setter, scan.parents)
    block, _ = _statement_block(node, scan.parents)
    if (setter_block is None or setter_block != block
            or not isinstance(statement, ast.Expr) or statement.value is not setter):
        return False
    # Default Process chooses its context at start(), not at construction. A
    # later conflicting setter can therefore invalidate a previously built one.
    for call, call_scope in scan.calls:
        if call_scope is scope and (call.lineno, call.col_offset) > (setter.lineno, setter.col_offset) \
                and _resolve_resource(call.func, call_scope) == ("path", "multiprocessing.set_start_method"):
            later_method = next((kw.value for kw in call.keywords if kw.arg == "method"),
                                call.args[0] if call.args else None)
            if not isinstance(later_method, ast.Constant) or later_method.value != "fork":
                return False
    # An arbitrary intervening helper could change the process-wide method.
    # Permit ordinary multiprocessing setup, but do not guess through user calls.
    setup_calls = {f"multiprocessing.{name}" for name in (
        "Queue", "SimpleQueue", "Pipe", "Lock", "RLock", "Semaphore",
        "BoundedSemaphore", "Condition", "Event", "Value", "Array",
        "RawValue", "RawArray", "get_context")}
    for call, call_scope in scan.calls:
        if (call_scope is scope
                and (setter.lineno, setter.col_offset) < (call.lineno, call.col_offset)
                < (node.lineno, node.col_offset)):
            callee = _resolve_resource(call.func, call_scope)
            if callee is None or callee[0] != "path" or callee[1] not in setup_calls:
                return False
    method = next((kw.value for kw in setter.keywords if kw.arg == "method"),
                  setter.args[0] if setter.args else None)
    return isinstance(method, ast.Constant) and method.value == "fork"


def _resource_fatal(tree: ast.AST) -> str | None:
    """Reject a local worker only when it is handed to a known process API."""
    scan = _ResourceScan()
    scan.visit(tree)
    for node, scope in scan.calls:
        callee = _resolve_resource(node.func, scope)
        if callee is None:
            continue
        if callee[0] == "worker_call":
            index, keyword = 0, "fn" if callee[1] == "submit" else "func"
        elif callee[0] == "process":
            if callee[1] == "fork" or callee[1] == "default" and _guaranteed_fork(node, scope, scan):
                continue
            index, keyword = 1, "target"
        elif callee[0] == "path" and callee[1] in _PROCESS_TYPES:
            if callee[1] in {"multiprocessing.Process", "multiprocessing.context.Process"} \
                    and _guaranteed_fork(node, scope, scan):
                continue
            index, keyword = 1, "target"
        else:
            continue
        worker = next((kw.value for kw in node.keywords if kw.arg == keyword),
                      node.args[index] if len(node.args) > index else None)
        resolved = _resolve_resource(worker, scope)
        if resolved is not None and resolved[0] == "local":
            return (f"multiprocessing cannot pickle a {resolved[1]} worker defined locally; "
                    "define the worker at module level or drop multiprocessing")
    return None


def precheck(child: str) -> Violation | None:
    """Pre-eval classification: empty / syntax / resource, in priority order.
    Returns None when the candidate may proceed to graft + duplicate check."""
    if not child.strip():
        return Violation("empty")
    try:
        tree = ast.parse(child)
    except SyntaxError as e:
        return Violation("syntax", f"{e.msg} (line {e.lineno})")
    res = _resource_fatal(tree)
    if res is not None:
        return Violation("resource", res)
    return None


# Declarative repair table: kind -> mechanical repair fn (None => no mechanical
# fix, resample instead). `interface` is handled by graft in validation, not here;
# `duplicate`/`empty` have no mechanical repair. This dict is the policy surface a
# learned RepairPolicy will consume.
REPAIR_TABLE = {
    "syntax": _docstring_repair,
    "resource": None,
    "empty": None,
    "duplicate": None,
}


def mechanical_repair(v: Violation, child: str) -> str | None:
    """Apply the declared mechanical repair for a violation, or None."""
    fn = REPAIR_TABLE.get(v.kind)
    return fn(child) if fn is not None else None


# Backward-compatible facade: the old pre_eval_gate returned a reason string.
def pre_eval_gate(child: str) -> str | None:
    v = precheck(child)
    if v is None or v.kind == "empty":
        return None
    return f"{v.kind} error before eval: {v.detail}" if v.kind == "syntax" else v.detail
