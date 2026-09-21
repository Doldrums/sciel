"""A single model-written restart function inside a byte-fixed Circle scaffold.

Building/validating source never imports or executes the solver or proposed
function. The AST subset is an interface guard, not a security or semantic
proof. Actual execution requires the separately bounded external container.
"""
from __future__ import annotations

import ast
import hashlib
import json

from . import circle_restart_scales as scales

CONTRACT = "sciel-circle-proposals/1"
OPERATION = "circle_restart_function/1"
PROOF_SCHEMA = "sciel-circle-proposal-source-edit/1"
PINNED_BASELINE_SHA256 = "78915475a194be9b871f1fde96eafec12561c87c72bef1aa18370154f770e06f"
MAX_HELPER_BYTES = 16384
MAX_AST_NODES = 4096
RESTART_PERIOD = 4
STATS_MARKER = "SCIEL_GENERATED_RESTART_STATS:"
ARGUMENTS = ("incumbent", "initial", "noise", "default_guess", "restart_index")
DEFAULT_HELPER = (
    "def propose_restart(incumbent, initial, noise, default_guess, restart_index):\n"
    "    return None\n"
)

_BUILTINS = frozenset(("abs", "min", "max", "sum", "len", "range", "enumerate",
                       "zip", "float", "int", "bool", "round", "sorted", "list", "tuple"))
_NUMPY_CALLS = frozenset((
    "array", "asarray", "copy", "zeros", "ones", "full", "zeros_like", "ones_like",
    "full_like", "arange", "linspace", "eye", "diag", "diagonal", "stack",
    "column_stack", "vstack", "hstack", "concatenate", "reshape", "ravel",
    "transpose", "clip", "maximum", "minimum", "where", "abs", "fabs", "sqrt",
    "hypot", "square", "sum", "min", "max", "mean", "argmin", "argmax",
    "argsort", "sort", "all", "any", "isfinite", "isnan", "isinf", "sign",
    "sin", "cos", "arctan2", "floor", "ceil", "round", "dot", "einsum",
    "outer", "roll", "repeat", "tile", "take", "nonzero", "flatnonzero",
    "count_nonzero", "fill_diagonal", "triu", "tril", "triu_indices", "tril_indices",
))
_NUMPY_VALUES = frozenset(("pi", "inf", "nan", "newaxis", "float64", "int64", "bool_"))
_MATH_CALLS = frozenset(("sqrt", "hypot", "fabs", "floor", "ceil", "sin", "cos",
                        "atan2", "isfinite", "isnan", "isinf", "copysign", "fsum"))
_MATH_VALUES = frozenset(("pi", "e", "tau", "inf", "nan"))
_ARRAY_METHODS = frozenset(("copy", "reshape", "ravel", "flatten", "transpose",
                           "astype", "sum", "min", "max", "mean", "argmin", "argmax",
                           "argsort", "sort", "clip", "all", "any", "nonzero", "fill"))
_ARRAY_VALUES = frozenset(("shape", "size", "ndim", "T", "real", "imag"))
_NODES = frozenset((
    ast.Module, ast.FunctionDef, ast.arguments, ast.arg, ast.Return, ast.Assign,
    ast.AugAssign, ast.If, ast.For, ast.Break, ast.Continue, ast.Pass, ast.Expr,
    ast.Name, ast.Load, ast.Store, ast.Constant, ast.List, ast.Tuple, ast.Subscript,
    ast.Slice, ast.Attribute, ast.Call, ast.keyword, ast.BinOp, ast.UnaryOp,
    ast.BoolOp, ast.Compare, ast.IfExp, ast.Add, ast.Sub, ast.Mult, ast.Div,
    ast.FloorDiv, ast.Mod, ast.Pow, ast.MatMult, ast.UAdd, ast.USub, ast.Not,
    ast.Invert, ast.And, ast.Or, ast.BitAnd, ast.BitOr, ast.BitXor, ast.Eq,
    ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.Is, ast.IsNot, ast.In, ast.NotIn,
    ast.ListComp, ast.GeneratorExp, ast.comprehension,
))


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _require(value, message):
    if not value:
        raise ValueError(message)


def _attribute(node):
    _require(isinstance(node.ctx, ast.Load) and "__" not in node.attr,
             "attribute mutation/dunder access is not allowed")
    if isinstance(node.value, ast.Name) and node.value.id in ("np", "math"):
        calls, values = ((_NUMPY_CALLS, _NUMPY_VALUES) if node.value.id == "np"
                         else (_MATH_CALLS, _MATH_VALUES))
        _require(node.attr in calls | values, "module attribute is outside the explicit surface")
    else:
        _require(node.attr in _ARRAY_METHODS | _ARRAY_VALUES,
                 "array attribute is outside the explicit surface")


def validate_helper(text):
    """Return exact function text with only outer CR/LF removal and final LF.

    Interior bytes, comments, spacing and literals are never rewritten. Fences,
    surrounding prose, imports and additional definitions are not accepted.
    Local loops may still be expensive; this is not a runtime termination proof.
    """
    _require(type(text) is str, "helper must be source text")
    try:
        raw = text.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("helper must be strict UTF8") from exc
    _require(0 < len(raw) <= MAX_HELPER_BYTES, "helper exceeds source byte limit")
    normalized = text.strip("\r\n") + "\n"
    _require(len(normalized.encode("utf-8")) <= MAX_HELPER_BYTES,
             "normalized helper exceeds source byte limit")
    _require(normalized.startswith("def propose_restart("), "only the exact helper function is accepted")
    try:
        tree = ast.parse(normalized)
        # Compilation checks contextual syntax (e.g. break outside a loop);
        # it never executes definitions, expressions or the proposed function.
        compile(tree, "<circle-proposal-interface>", "exec")
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ValueError("invalid helper syntax") from exc
    _require(len(tree.body) == 1 and type(tree.body[0]) is ast.FunctionDef,
             "helper must contain exactly one function")
    function = tree.body[0]
    args = function.args
    _require(function.name == "propose_restart" and not function.decorator_list
             and function.returns is None and not function.type_comment
             and not getattr(function, "type_params", []), "helper signature/decorators differ")
    _require(tuple(a.arg for a in args.args) == ARGUMENTS and not args.posonlyargs
             and not args.kwonlyargs and args.vararg is None and args.kwarg is None
             and not args.defaults and not args.kw_defaults
             and all(a.annotation is None and not a.type_comment for a in args.args),
             "helper requires the five exact unannotated positional arguments")
    nodes = list(ast.walk(tree))
    _require(len(nodes) <= MAX_AST_NODES, "helper AST exceeds node limit")
    _require(sum(type(n) is ast.FunctionDef for n in nodes) == 1, "nested definitions are forbidden")
    reserved = _BUILTINS | {"np", "math", "propose_restart"}
    locals_ = set(ARGUMENTS)
    for node in nodes:
        _require(type(node) in _NODES, "helper syntax outside surface: " + type(node).__name__)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
            _require(node.id not in reserved and "__" not in node.id, "reserved name cannot be assigned")
            locals_.add(node.id)
    _require(any(isinstance(n, ast.Return) for n in nodes), "helper must contain a return")
    for node in nodes:
        if isinstance(node, ast.Name):
            _require("__" not in node.id and node.id in locals_ | _BUILTINS | {"np", "math"},
                     "helper uses an unavailable name")
        elif isinstance(node, ast.Attribute):
            _attribute(node)
        elif isinstance(node, ast.Call):
            _require(all(k.arg is not None for k in node.keywords), "dynamic keyword expansion is forbidden")
            callee = node.func
            if isinstance(callee, ast.Name):
                _require(callee.id in _BUILTINS, "dynamic/recursive calls are forbidden")
            elif isinstance(callee, ast.Attribute):
                _attribute(callee)
                if isinstance(callee.value, ast.Name) and callee.value.id in ("np", "math"):
                    choices = _NUMPY_CALLS if callee.value.id == "np" else _MATH_CALLS
                    _require(callee.attr in choices, "module value is not a permitted callable")
                else:
                    _require(callee.attr in _ARRAY_METHODS, "only explicit array methods can be called")
            else:
                raise ValueError("dynamic calls are forbidden")
        elif isinstance(node, ast.Constant):
            _require(type(node.value) in (int, float, str, bool, type(None)), "unsupported literal")
            if type(node.value) is float:
                _require(node.value == node.value and abs(node.value) != float("inf"), "nonfinite literal")
        elif isinstance(node, ast.keyword):
            _require(node.arg is not None and "__" not in node.arg, "invalid keyword")
        elif isinstance(node, ast.comprehension):
            _require(node.is_async == 0, "async comprehensions are forbidden")
    return normalized


_RUNTIME = '''
# BEGIN trusted generated-restart runtime
import json as _proposal_json
import sys as _proposal_sys


def _proposal_stats():
    return {"proposal_attempts": 0, "proposal_accepted": 0, "proposal_none": 0,
            "proposal_errors": 0, "proposal_time_seconds": 0.0,
            "proposal_scheduled_starts": 0, "proposal_ordinary_starts": 0,
            "proposal_scheduled_completed": 0, "proposal_ordinary_completed": 0,
            "proposal_scheduled_improvements": 0, "proposal_ordinary_improvements": 0,
            "proposal_scheduled_gain": 0.0, "proposal_ordinary_gain": 0.0}


def _proposal_invoke(incumbent, initial, noise, default_guess, restart_index, stats):
    started = time.monotonic()
    stats["proposal_attempts"] += 1
    try:
        value = propose_restart(incumbent.copy(), initial.copy(), noise.copy(),
                                default_guess.copy(), restart_index)
        if value is None:
            stats["proposal_none"] += 1
            return default_guess.copy()
        array = np.asarray(value)
        if array.dtype.kind not in "fiu" or array.shape != (78,):
            raise ValueError("restart proposal must be a real numeric vector of shape (78,)")
        array = np.asarray(array, dtype=float).copy()
        if not np.isfinite(array).all():
            raise ValueError("restart proposal must contain finite values")
        stats["proposal_accepted"] += 1
        return array
    except Exception:
        stats["proposal_errors"] += 1
        raise
    finally:
        stats["proposal_time_seconds"] += max(0.0, time.monotonic() - started)
# END trusted generated-restart runtime


'''

_STATS_INSERT = '''    stats.update(_proposal_stats())
    active_role = None
    active_improvements = 0
    active_sum = incumbent._sum

    def finish_proposal_role():
        nonlocal active_role
        if active_role is not None:
            stats["proposal_" + active_role + "_improvements"] += incumbent.improvements - active_improvements
            stats["proposal_" + active_role + "_gain"] += incumbent._sum - active_sum
            active_role = None
'''
_CALL_INSERT = '''            if _ % 4 == 0:
                check_deadline()
                guess = _proposal_invoke(incumbent._params, initial, noise, guess, _, stats)
                check_deadline(guess)
            stats["proposal_" + active_role + "_starts"] += 1
'''


def _scaffold():
    baseline = scales.source_for("0.06")
    _require(_sha(baseline.encode("utf-8")) == PINNED_BASELINE_SHA256,
             "proposal baseline differs from pinned historical source")
    specs = [
        ("runtime_and_control_helper", "from scipy.optimize import minimize\n", True,
         _RUNTIME + DEFAULT_HELPER + "\n"),
        ("common_stats", "    LAST_STATS.clear()\n", True, _STATS_INSERT),
        ("role_begin", "        for _ in range(max_restarts):\n            check_deadline()\n", True,
         '            active_role = "scheduled" if _ % 4 == 0 else "ordinary"\n'
         '            active_improvements = incumbent.improvements\n'
         '            active_sum = incumbent._sum\n'),
        ("same_noise_draw", "            guess = incumbent._params + np.random.uniform(-0.06, 0.06, initial.shape)\n", False,
         "            noise = np.random.uniform(-0.06, 0.06, initial.shape)\n"
         "            guess = incumbent._params + noise\n"),
        ("common_proposal_call", "            guess[:2 * N_CIRCLES] = np.clip(guess[:2 * N_CIRCLES], 0.001, 0.999)\n", True, _CALL_INSERT),
        ("completed_role", '            stats["completed_starts"] += 1\n', True,
         '            stats["proposal_" + active_role + "_completed"] += 1\n'),
        ("role_finish", "            incumbent.consider(result.x)\n            check_deadline()\n", True,
         "            finish_proposal_role()\n"),
        ("final_role_finish", "    finally:\n        stats.update(candidate_checks=incumbent.candidate_checks,\n", False,
         "    finally:\n        finish_proposal_role()\n        stats.update(candidate_checks=incumbent.candidate_checks,\n"),
        ("common_diagnostics", "    return solve_packing(budget_seconds=27.0, max_restarts=10000, maxiter=15000)\n", False,
         "    try:\n        return solve_packing(budget_seconds=27.0, max_restarts=10000, maxiter=15000)\n"
         "    finally:\n        print(" + repr(STATS_MARKER) +
         " + _proposal_json.dumps(LAST_STATS, sort_keys=True, allow_nan=False), file=_proposal_sys.stderr)\n"),
    ]
    raw = baseline.encode("utf-8")
    patches = []
    for name, anchor, insert, replacement in specs:
        _require(baseline.count(anchor) == 1, "proposal scaffold anchor is not unique: " + name)
        start = raw.index(anchor.encode())
        end = start + len(anchor.encode())
        if insert:
            start = end
        patches.append((start, end, name, replacement.encode()))
    patches.sort()
    pieces, receipts, previous, offset = [], [], 0, 0
    for start, end, name, replacement in patches:
        _require(start >= previous, "overlapping scaffold patches")
        before = raw[start:end]
        pieces.extend((raw[previous:start], replacement))
        receipts.append({"name": name, "parent_byte_span": [start, end],
                         "child_byte_span": [start+offset, start+offset+len(replacement)],
                         "before": before.decode(), "after": replacement.decode(),
                         "before_sha256": _sha(before), "after_sha256": _sha(replacement)})
        offset += len(replacement)-len(before)
        previous = end
    pieces.append(raw[previous:])
    return b"".join(pieces).decode(), receipts


def source_for(helper_text=None):
    """The newly measured common control, or its single-function replacement."""
    return _scaffold()[0] if helper_text is None else build(helper_text)["code"]


def build(helper_text):
    normalized = validate_helper(helper_text)
    control, _ = _scaffold()
    _require(normalized != DEFAULT_HELPER, "candidate helper exactly repeats control")
    before = DEFAULT_HELPER.encode()
    raw = control.encode()
    _require(raw.count(before) == 1, "control helper anchor is not unique")
    start = raw.index(before)
    prefix, suffix = raw[:start], raw[start+len(before):]
    after = normalized.encode("utf-8")
    child = prefix + after + suffix
    proof = {"schema": PROOF_SCHEMA, "contract": CONTRACT, "operation": OPERATION,
             "baseline_source_sha256": PINNED_BASELINE_SHA256,
             "parent_source_sha256": _sha(raw), "child_source_sha256": _sha(child),
             "submitted_helper_sha256": _sha(helper_text.encode("utf-8")),
             "normalized_helper_sha256": _sha(after),
             "normalization": "Strip outer CR/LF characters only, then append one LF; preserve all interior bytes.",
             "before": {"text": DEFAULT_HELPER, "sha256": _sha(before), "byte_span": [start, start+len(before)]},
             "after": {"text": normalized, "sha256": _sha(after), "byte_span": [start, start+len(after)]},
             "unchanged_prefix": {"bytes": len(prefix), "sha256": _sha(prefix)},
             "unchanged_suffix": {"bytes": len(suffix), "sha256": _sha(suffix)},
             "unchanged_outside_helper": True,
             "scope": "Exact text replacement and AST interface only; not semantic equivalence, security, causality or quality proof."}
    proof["plan_key"] = OPERATION + ":" + _sha(raw) + ":" + _sha(after)
    return {"code": child.decode("utf-8"), "proof": proof}


def interface_description():
    return {"contract": CONTRACT, "operation": OPERATION, "function": "propose_restart",
            "arguments": list(ARGUMENTS), "array_shape": [78], "centers_slice": [0, 52], "radii_slice": [52, 78],
            "arrays": "All four arrays are detached copies; modifying them cannot mutate the retained incumbent, original noise or ordinary fallback.",
            "noise": "The same existing 78-value uniform[-0.06,0.06] draw; no extra RNG.",
            "schedule": "After the ordinary guess clamps, on restart_index % 4 == 0, including 0.",
            "return": "None selects the exact ordinary default guess; otherwise accept only a copied real finite numeric shape-(78,) vector. No clipping or feasibility repair; overlaps and out-of-bounds starts are allowed for unchanged SLSQP.",
            "exceptions": "Errors propagate as candidate failures; never silently fall back.",
            "max_helper_bytes": MAX_HELPER_BYTES, "max_ast_nodes": MAX_AST_NODES,
            "normalization": "Only remove outer CR/LF characters and append one LF; no AST rewriting.",
            "builtins": sorted(_BUILTINS), "numpy_calls": sorted(_NUMPY_CALLS), "numpy_values": sorted(_NUMPY_VALUES),
            "math_calls": sorted(_MATH_CALLS), "math_values": sorted(_MATH_VALUES),
            "array_methods": sorted(_ARRAY_METHODS), "array_values": sorted(_ARRAY_VALUES),
            "forbidden": "Imports, globals/nonlocals, other or nested definitions, decorators/defaults/annotations, recursion/dynamic calls, module mutation, dunders, I/O, random, optimizers and persistent state. Only assignments, if/for, return/break/continue/pass, synchronous list comprehensions/generator expressions and numeric expressions from the listed AST surface.",
            "limits": "Shared cooperative checks before/after helper, unchanged SLSQP and outer process watchdog. The AST guard does not prove termination, bounded allocation, semantic compliance, safety, feasibility or quality.",
            "baseline": "This control scaffold has common overhead and must be freshly measured; it is not the historical source's timing or outcome."}


def response_instruction():
    return ("Generate the actual Python restart proposal function, not a policy name or a complete solver. "
            "Return one lowercase python fenced block containing exactly the function below's signature, "
            "without imports or surrounding explanation; any separately requested hypothesis envelope precedes it. "
            "The common scaffold is fixed. The function may construct a new numerical neighborhood within this explicit interface:\n"
            + json.dumps(interface_description(), sort_keys=True, separators=(",", ":"), allow_nan=False)
            + "\nRequired signature: def propose_restart(incumbent, initial, noise, default_guess, restart_index):")


def identity():
    control, patches = _scaffold()
    surface = interface_description()
    return {"contract": CONTRACT, "operation": OPERATION,
            "baseline_source_sha256": PINNED_BASELINE_SHA256,
            "control_source_sha256": _sha(control.encode()),
            "control_helper_sha256": _sha(DEFAULT_HELPER.encode()),
            "runtime_sha256": _sha(_RUNTIME.encode()),
            "interface_sha256": _sha(json.dumps(surface, sort_keys=True, separators=(",", ":")).encode()),
            "scaffold_patches": patches, "restart_period": RESTART_PERIOD, "first_restart_index": 0,
            "stats_marker": STATS_MARKER,
            "scope": "Fresh common-scaffold comparison; only candidate helper bytes differ. Role diagnostics combine initializer and subsequent solver work, not causal effects."}
