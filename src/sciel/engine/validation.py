"""One worker candidate's existing validation/repair policy, with source-free traces.

Pure source inspection only: no model requests, evaluator calls or search RNG.
The bootstrap profile shares repairs/traces while retaining its distinct policy.
"""

from __future__ import annotations

import ast
import builtins
from dataclasses import dataclass
import hashlib
import io
import symtable
import tokenize

from .db import normalized_hash
from .viability import Violation, mechanical_repair, precheck

TRACE_SCHEMA = "sciel-validate-repair/1"


def candidate_registry_enabled(check) -> bool:
    """Only an explicit registry callback replaces legacy duplicate semantics."""
    return getattr(check, 'candidate_registry', False) is True


def candidate_attempt_limit(check, legacy_limit):
    if not candidate_registry_enabled(check):
        return legacy_limit
    retries = getattr(check, 'max_resamples', 2)
    if type(retries) is not int or not 0 <= retries <= 2:
        raise ValueError('candidate registry max_resamples must be an integer from 0 to 2')
    return 1 + retries


def novelty_failure(novelty):
    kind = novelty.get('match_kind')
    reference = novelty.get('matched_program_id') or novelty.get('matched_evaluation_id')
    detail = str(novelty.get('reason') or 'candidate rejected by configured code novelty gate')
    if reference:
        detail = f'{kind or "known-source"} match {reference}: {detail}'
    return Violation('duplicate' if kind in ('sha256', 'ast') else 'code_novelty', detail)


def source_sha256(code: str) -> str:
    """Exact text digest; no whitespace/AST normalization or randomized hashing."""
    return hashlib.sha256(code.encode("utf-8", errors="surrogatepass")).hexdigest()


_BUILTINS = frozenset(dir(builtins))

_TRY_NODES = (ast.Try, ast.TryStar) if hasattr(ast, "TryStar") else (ast.Try,)


def _import_guard_value(test) -> bool | None:
    """Known truth value when imported; None leaves both branches possible."""
    if isinstance(test, ast.Constant) and isinstance(test.value, bool):
        return test.value
    if isinstance(test, ast.Compare) and len(test.ops) == 1 \
            and isinstance(test.ops[0], (ast.Eq, ast.NotEq)):
        left, right = test.left, test.comparators[0]
        if any(isinstance(a, ast.Name) and a.id == "__name__"
               and isinstance(b, ast.Constant) and b.value == "__main__"
               for a, b in ((left, right), (right, left))):
            return isinstance(test.ops[0], ast.NotEq)
    if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
        value = _import_guard_value(test.operand)
        return None if value is None else not value
    if isinstance(test, ast.BoolOp):
        values = [_import_guard_value(v) for v in test.values]
        if isinstance(test.op, ast.And):
            return False if False in values else (True if all(v is True for v in values) else None)
        return True if True in values else (False if all(v is False for v in values) else None)
    return None


def _is_main_guard(node) -> bool:
    return isinstance(node, ast.If) and _import_guard_value(node.test) is False


def _module_stmts(tree) -> list:
    """Module-level statements, looking through top-level If/Try wrappers -- a guarded
    `try: import numpy as np / except ImportError: np = None` still binds at module
    scope, so its statements must be visible to the graft's symbol tables. The one If
    that must NOT be looked through is `if __name__ == "__main__"`: its body only runs
    as a script, and grafting it would execute it at import time in the child."""
    out, stack = [], list(tree.body)
    while stack:
        node = stack.pop(0)
        if isinstance(node, ast.If):
            value = _import_guard_value(node.test)
            if value is not None:
                stack.extend(node.body if value else node.orelse)
                continue
        if isinstance(node, (ast.If, *_TRY_NODES)):
            for part in (node.body, node.orelse, getattr(node, "finalbody", [])):
                stack.extend(part)
            for h in getattr(node, "handlers", []):
                stack.extend(h.body)
        else:
            out.append(node)
    return out


def _bind_targets(node) -> list:
    """Names an Assign/AnnAssign binds, including tuple-unpacking targets."""
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    names = []
    for t in targets:
        if isinstance(t, ast.Name):
            names.append(t.id)
        elif isinstance(t, (ast.Tuple, ast.List)):
            names += [e.id for e in t.elts if isinstance(e, ast.Name)]
    return names


_DEF_NODES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)


def _global_references(node) -> set[str]:
    """Use Python's scope analysis, including defaults and nested closures.

    Class bodies use namespace lookups that can fall back to globals even for a
    locally bound name (``N = N + 1``), so retain those possible dependencies too.
    Function locals/free variables must never hide a global used in another scope.
    """
    root = symtable.symtable(ast.unparse(node), "<graft>", "exec")
    refs, pending = set(), [root]
    while pending:
        table = pending.pop()
        refs.update(s.get_name() for s in table.get_symbols()
                    if s.is_referenced() and (s.is_global() or table.get_type() == "class"))
        pending.extend(table.get_children())
    return refs


def graft_entrypoints(child: str, parent: str, names: list) -> tuple:
    """If the child dropped a required entry-point, re-attach it verbatim from the parent
    (which had it) -- together with whatever the re-attached code needs and the child does
    not bind: parent imports, helper functions, and module constants, closed transitively.
    Returns (code, grafted_names); only names actually re-attached are reported.

    "Present" means bound at MODULE level (through If/Try wrappers): that is what the
    evaluator's hasattr contract checks, so a child that moved the entry-point into a
    nested scope still gets the parent's module-level version grafted. A class binding
    satisfies hasattr too, so top-level classes count.

    Guards: a child with no top-level defs of its own is returned untouched -- grafting
    onto an empty or prose-only response does not rescue a candidate, it stitches a broken
    copy of the parent (the franken-child failure mode: NameError on the parent's imports/
    helpers). If either side does not parse, the child is returned as-is: evaluation fails
    exactly as it would have. No-op when names is empty (no contract detected).

    The dependency closure resolves names the grafted code loads but neither the child
    nor the grafted code itself binds (locally-assigned names are local to the function
    by Python scoping and are excluded, unless declared global). Grafted statements are
    appended after the child's own bindings and never shadow them; `if __name__` bodies
    are never treated as module bindings nor grafted."""
    if not names:
        return child, []
    try:
        ctree = ast.parse(child)
    except SyntaxError:
        return child, []
    cstmts = _module_stmts(ctree)
    defs = {n.name for n in cstmts if isinstance(n, _DEF_NODES)}
    # Every module-level binding counts as "the child has this name", not only `def`
    # and `class`. A child that binds the entry point by assignment -- `solve =
    # functools.partial(...)`, `run_search = _impl` -- used to be judged as missing
    # it, and the parent's def was then appended AFTER the child's binding, where the
    # later definition wins. The evaluated program was the parent's, scored
    # identically to it, and entered the population as a clone with no error anywhere.
    have = set(defs)
    for node in cstmts:
        if isinstance(node, ast.Import):
            have |= {(a.asname or a.name).split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom):
            have |= {a.asname or a.name for a in node.names}
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            have |= set(_bind_targets(node))
    if not defs:
        # A child with no definitions of its own is a prose-only or empty response.
        # Stitching the parent's entry points onto it builds a franken-child that
        # NameErrors on the parent's helpers, so the guard stays on `defs`, not on the
        # wider binding set: the two questions are different ones.
        return child, []
    missing = [n for n in names if n not in have]
    if not missing:
        return child, []
    try:
        ptree = ast.parse(parent)
        compile(ptree, "<parent>", "exec")
    except SyntaxError:
        return child, []

    pfuncs, pconsts, pimports = {}, {}, {}
    for node in _module_stmts(ptree):
        if isinstance(node, _DEF_NODES):
            pfuncs[node.name] = node
        elif isinstance(node, ast.Import):
            for a in node.names:
                pimports[(a.asname or a.name).split(".")[0]] = node
        elif isinstance(node, ast.ImportFrom):
            for a in node.names:
                pimports[a.asname or a.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            for name in _bind_targets(node):
                pconsts[name] = node

    cbound = have          # same set: every module-level binding the child makes

    grafted_names = [n for n in missing if n in pfuncs]
    queue = [pfuncs[n] for n in grafted_names]
    if not queue:
        return child, []
    picked, picked_ids, resolved = [], set(), set(missing)
    while queue:
        node = queue.pop(0)
        if id(node) in picked_ids:
            continue
        picked_ids.add(id(node))
        picked.append(node)
        refs = _global_references(node)
        for r in sorted(refs - cbound - resolved - _BUILTINS):
            resolved.add(r)
            dep = pfuncs.get(r) or pimports.get(r) or pconsts.get(r)
            if dep is not None:
                queue.append(dep)
    # parent source order preserves the parent's own execution order (a grafted
    # constant may call a grafted helper at module load)
    picked.sort(key=lambda n: n.lineno)
    # ast.get_source_segment starts at node.lineno, which for a decorated definition
    # is the `def`/`class` line: the decorators are silently dropped. A grafted
    # @functools.lru_cache function keeps its import and loses its memoization, so the
    # child times out and is blamed for it. Same for @njit, @dataclass, @wraps.
    segs = [seg for n in picked if (seg := _source_with_decorators(parent, n))]
    if not segs:
        return child, []
    return child.rstrip() + "\n\n\n" + "\n\n".join(segs) + "\n", grafted_names


_KNOWN_IMPORTS = {
    "np": "import numpy as np", "numpy": "import numpy", "math": "import math",
    "random": "import random", "time": "import time", "itertools": "import itertools",
    "scipy": "import scipy", "minimize": "from scipy.optimize import minimize",
    "linprog": "from scipy.optimize import linprog",
    "basinhopping": "from scipy.optimize import basinhopping",
    "differential_evolution": "from scipy.optimize import differential_evolution",
    "cdist": "from scipy.spatial.distance import cdist",
    "pdist": "from scipy.spatial.distance import pdist",
}


def add_missing_imports(code: str) -> tuple[str, list[str]]:
    """Import well-known modules that the code uses at module or function scope but never binds."""
    try:
        tree = ast.parse(code)
        compile(tree, "<candidate>", "exec")
    except SyntaxError:
        return code, []
    bound = set()
    for node in _module_stmts(tree):
        if isinstance(node, _DEF_NODES):
            bound.add(node.name)
        elif isinstance(node, ast.Import):
            bound.update((a.asname or a.name).split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            bound.update(a.asname or a.name for a in node.names)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            bound.update(_bind_targets(node))
    missing = sorted(_global_references(tree) - bound - _BUILTINS)
    lines = [_KNOWN_IMPORTS[name] for name in missing if name in _KNOWN_IMPORTS]
    if not lines:
        return code, []
    body = tree.body
    skip = 0
    while skip < len(body) and (
            isinstance(body[skip], ast.ImportFrom) and body[skip].module == "__future__"
            or skip == 0 and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant)
            and isinstance(body[0].value.value, str)):
        skip += 1
    at = body[skip - 1].end_lineno if skip else 0
    src = code.split("\n")
    return "\n".join(src[:at] + lines + src[at:]), [name for name in missing if name in _KNOWN_IMPORTS]


def _source_with_decorators(src: str, node) -> str | None:
    """Move a statement to module scope without changing multiline literals."""
    seg = ast.get_source_segment(src, node)
    if seg is None:
        return None
    line = src.splitlines()[node.lineno - 1]
    indent = line[:len(line) - len(line.lstrip())]
    seg = indent + seg
    decorators = getattr(node, "decorator_list", None)
    if decorators:
        first = min(d.lineno for d in decorators)
        head = "\n".join(src.splitlines()[first - 1:node.lineno - 1])
        seg = f"{head}\n{seg}"
    if not indent:
        return seg
    # textwrap.dedent would also alter string contents, or fail to move the code
    # at all when a multiline literal contains an unindented continuation line.
    literal_lines, fstrings = set(), []
    for token in tokenize.generate_tokens(io.StringIO(seg).readline):
        if token.type == tokenize.STRING:
            literal_lines.update(range(token.start[0] + 1, token.end[0] + 1))
        elif token.type == getattr(tokenize, "FSTRING_START", -1):
            fstrings.append(token.start[0])
        elif token.type == getattr(tokenize, "FSTRING_END", -1):
            literal_lines.update(range(fstrings.pop() + 1, token.end[0] + 1))
    return "\n".join(line[len(indent):] if i not in literal_lines and line.startswith(indent)
                     else line for i, line in enumerate(seg.splitlines(), 1))



@dataclass
class ValidationResult:
    code: str
    violation: Violation | None
    grafted: list[str]
    novelty: dict | None
    decisions: list[dict]
    profile: str = "worker"

    @property
    def disposition(self) -> str:
        return ("resample" if self.profile == "worker" else "reject") if self.violation else "evaluate"


def trace_attempt(validation: ValidationResult, *, extracted_code: str, response,
                  attempt: int, operator: str, temperature: float, cost) -> dict:
    """Describe a returned generation without copying prompts or source payloads."""
    trace = {
        "attempt": attempt, "operator": operator, "temperature": temperature,
        "disposition": validation.disposition, "generation_returned": True,
        "response_sha256": source_sha256(response.text),
        "extracted_code_sha256": source_sha256(extracted_code),
        "candidate_code_sha256": source_sha256(validation.code),
        "cost": cost, "decisions": validation.decisions}
    # The legacy cost argument is a known-USD contribution, not a price for a
    # completed subscription response. Preserve that distinction in audit data.
    from .llm.client import response_accounting, response_cost_provenance, response_model_identity
    identity = response_model_identity(response)
    if identity is not None:
        trace['model_identity'] = identity
    try:
        accounting = response_accounting(response)
    except ValueError:
        accounting = None
    if accounting is not None and accounting['unpriced_calls']:
        trace.update(cost=None, known_usd=accounting['known_usd'], unpriced_calls=1,
                     cost_provenance=response_cost_provenance(response))
    for field in ("prompt_tokens", "completion_tokens"):
        value = getattr(response, field, None)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            trace[field] = value
    return trace


def _decision(stage, action, before, after, *, violation=None, repair=None, **details):
    return {"stage": stage, "action": action,
            "violation": ({"kind": violation.kind, "detail": violation.detail}
                          if violation is not None else None),
            "repair": repair, "input_code_sha256": source_sha256(before),
            "output_code_sha256": source_sha256(after), **details}


def validate_candidate(code: str, *, parent_code: str, required_entrypoints: list,
                       code_hashes: set, novelty_check=None,
                       profile: str = "worker") -> ValidationResult:
    """Apply an existing ordered policy exactly once, without evaluating.

    Mechanical repair and graft are deliberately NOT followed by another precheck:
    that was not part of the original policy. An unchanged graft is not proof that
    all interfaces are present; historically such candidates reach the evaluator.
    The optional novelty callback inspects the final source once. A marked whole-run
    registry replaces the legacy normalized-source duplicate gate; an unmarked
    callback retains its historical position after that gate.
    The one-shot paradigm profile instead checks empty -> graft -> novelty, with no
    syntax/resource precheck, duplicate check or resampling.
    """
    if profile not in ("worker", "paradigm"):
        raise ValueError(f"unknown validation profile: {profile}")
    reject_action = "resample" if profile == "worker" else "reject"
    decisions = []
    violation = (precheck(code) if profile == "worker" else
                 Violation("empty") if not code.strip() else None)
    if profile == "paradigm":
        decisions.append(_decision("empty", "reject" if violation else "pass", code, code,
                                   violation=violation))
        if violation is not None:
            return ValidationResult(code, violation, [], None, decisions, profile)
    elif violation is not None:
        repaired = mechanical_repair(violation, code)
        fixed, repair = repaired if repaired is not None else (None, None)
        decisions.append(_decision(
            "precheck", "repair" if fixed is not None else "resample", code,
            fixed if fixed is not None else code, violation=violation,
            repair=repair, repair_applied=fixed is not None))
        if fixed is None:
            return ValidationResult(code, violation, [], None, decisions, profile)
        code = fixed
    else:
        decisions.append(_decision("precheck", "pass", code, code))

    before = code
    code, grafted = graft_entrypoints(code, parent_code, required_entrypoints)
    decisions.append(_decision(
        "interface", "repair" if grafted else "unchanged", before, code,
        violation=Violation("interface", "grafted parent entrypoints: " + ", ".join(grafted))
                  if grafted else None,
        repair="graft_entrypoints" if grafted else None,
        required_entrypoints=list(required_entrypoints), grafted=list(grafted),
        parent_code_sha256=source_sha256(parent_code)))

    if profile == "worker":
        before = code
        code, added = add_missing_imports(code)
        decisions.append(_decision("imports", "repair" if added else "unchanged", before, code,
                                   repair="add_imports" if added else None, added=added))

    if profile == "worker" and not candidate_registry_enabled(novelty_check):
        duplicate_hash = normalized_hash(code)
        duplicate = duplicate_hash in code_hashes
        violation = Violation("duplicate", "normalized source already present on the island") if duplicate else None
        decisions.append(_decision(
            "duplicate", "resample" if duplicate else "pass", code, code,
            violation=violation, normalized_code_hash=duplicate_hash,
            normalized_hash_kind="nonblank-lines-rstrip-sha256"))
        if duplicate:
            return ValidationResult(code, violation, grafted, None, decisions, profile)

    novelty = None
    if novelty_check is not None:
        novelty = novelty_check(code)
        violation = (None if novelty["accept"] else novelty_failure(novelty)
                     if candidate_registry_enabled(novelty_check) else
                     Violation("code_novelty", "candidate rejected by configured code novelty gate"))
        decisions.append(_decision(
            "code_novelty", "pass" if violation is None else reject_action, code, code,
            violation=violation, novelty=dict(novelty)))
    return ValidationResult(code, violation, grafted, novelty, decisions, profile)
