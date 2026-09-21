"""Strict typed-array body transport and exact Circle source assembly.

No generated code, optimizer or evaluator is imported/executed here. The body
subset limits an interface, not security, runtime, mechanism or quality. Full
source artifacts stay available to audit; only the typed surface is writer-facing.
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

from . import circle_proposals as proposals
from . import circle_body_list_policy as list_policy
from .models import Hypothesis
from .parsing import MAX_REASONING_CHARS, MAX_TEXT_CHARS, parse_hypothesis_value

CONTRACT = OPERATION = "circle_restart_body/1"
DEFAULT_BODY = "return None\n"
MAX_BODY_BYTES = 8192
MAX_BODY_LINES = 512
MAX_RESPONSE_BYTES = 131072
PINNED_PROPOSALS_SHA256 = "5def9eb52c65962e62c9269f1f953a715b2ba3adfed791c37a6cf35380c4cbb9"
PINNED_LIST_POLICY_SHA256 = "311da6a7077491419c0d910d5045b2dd2eb8d10ffaf968313c536f95efba3b1e"
PINNED_SCAFFOLD_SHA256 = "3ac4659923530a3949903d6416c6a19b059623e0095ce0e9f032f7e016498734"
PINNED_BASELINE_SHA256 = "78915475a194be9b871f1fde96eafec12561c87c72bef1aa18370154f770e06f"
STATS_MARKER = proposals.STATS_MARKER
_OLD_NAMES = frozenset(("incumbent", "initial", "noise", "default_guess"))
_ALIASES = ("best_params", "initial_params", "uniform_noise", "ordinary_params")
_HEADER = (
    "def propose_restart(incumbent, initial, noise, default_guess, restart_index):\n"
    "    best_params = incumbent\n"
    "    initial_params = initial\n"
    "    uniform_noise = noise\n"
    "    ordinary_params = default_guess\n"
)


@dataclass(frozen=True)
class BodyProposal:
    hypothesis: Hypothesis
    body: str


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _encode(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def list_policy_identity():
    """Bind the imported syntax policy through this source-pinned builder."""
    value = list_policy.identity()
    _require(value['source_sha256'] == PINNED_LIST_POLICY_SHA256,
             "typed local-list policy differs from pinned dependency")
    return value


def _pinned_scaffold():
    _require(_sha(Path(proposals.__file__).read_bytes()) == PINNED_PROPOSALS_SHA256,
             "proposal builder source differs from pinned dependency")
    source = proposals.source_for()
    _require(_sha(source.encode("utf-8")) == PINNED_SCAFFOLD_SHA256,
             "common scaffold differs from pinned dependency")
    old = proposals.DEFAULT_HELPER
    _require(source.count(old) == 1, "control helper source anchor is not unique")
    prefix, _, suffix = source.partition(old)
    return prefix, suffix


def _indented(body):
    # Split on LF only: Unicode separators inside quoted strings stay untouched.
    terminated = body if body.endswith("\n") else body + "\n"
    return "".join("    " + line + "\n" for line in terminated.split("\n")[:-1])


def _surface_error_node(error):
    """Locate the actual pinned-validator rejection without repeating its policy.

    Only known attribute/call errors expose a node from trusted validators.
    Never render frame locals or inspect model runtime frames.
    If this diagnostic metadata is unavailable, retain the original error.
    """
    if isinstance(error, list_policy.LocalListPolicyError):
        return error.node
    if str(error) not in {
        "attribute mutation/dunder access is not allowed",
        "module attribute is outside the explicit surface",
        "array attribute is outside the explicit surface",
        "module value is not a permitted callable",
        "only explicit array methods can be called",
    }:
        return None
    node = None
    trace = error.__traceback__
    while trace is not None:
        frame = trace.tb_frame
        if frame.f_code is proposals._attribute.__code__:
            rejected = frame.f_locals.get("node")
            if isinstance(rejected, ast.Attribute):
                return rejected
        elif frame.f_code in (proposals.validate_helper.__code__, list_policy._validate_extended.__code__):
            rejected = frame.f_locals.get("node")
            if isinstance(rejected, ast.Call) and isinstance(rejected.func, ast.Attribute):
                node = rejected.func
        trace = trace.tb_next
    return node


def _surface_error_detail(body, error):
    node = _surface_error_node(error)
    if node is None:
        return None
    line = node.lineno - _HEADER.count("\n")
    lines = body.split("\n")
    if not 1 <= line <= len(lines):
        return None
    # AST columns count UTF8 bytes; expose 1-based submitted character columns.
    before = lines[line - 1].encode("utf-8")[:max(0, node.col_offset - 4)]
    column = len(before.decode("utf-8", errors="ignore")) + 1
    parts, base = [node.attr], node.value
    while isinstance(base, ast.Attribute) and len(parts) < 16:
        parts.append(base.attr)
        base = base.value
    access = (".".join([base.id, *reversed(parts)]) if isinstance(base, ast.Name)
              else "attribute " + node.attr)
    # Knowledge serializes diagnostics with ensure_ascii=True before bounding
    # them. Bound that representation, retaining the terminal attribute, so a
    # long Unicode name cannot displace the location and actionable hint.
    def ascii_prefix(text, limit):
        width = 0
        for index, char in enumerate(text):
            width += len(json.dumps(char, ensure_ascii=True)) - 2
            if width > limit:
                return text[:index]
        return text

    label = access
    if len(json.dumps(label, ensure_ascii=True)) - 2 > 120:
        label = ascii_prefix(access, 57) + "..." + ascii_prefix(access[::-1], 60)[::-1]
    rng_hint = (" Additional RNG is unavailable; use the supplied uniform_noise."
                if access == "np.random" or access.startswith("np.random.") else "")
    excerpt = repr(lines[line - 1][:120])
    shortened = ascii_prefix(excerpt, 192)
    if shortened != excerpt:
        excerpt = shortened + "..."
    return (f"{error}; rejected access {label!r} (body line {line}, column {column})."
            f"{rng_hint} Submitted line excerpt (untrusted; display whitespace may collapse): {excerpt}")


def validate_body(body):
    """Validate without rewriting; structural final LF is added only at assembly.

    LF and CRLF are preserved; a lone CR is explicitly unsupported. Body source
    cannot access the old object-oriented argument names, even through a local
    assignment. The pinned local-list extension permits proven local append
    calls; all other restrictions retain the proposal interface. No repair.
    """
    list_policy_identity()
    _require(type(body) is str, "body must be a string")
    try:
        raw = body.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("body must be strict UTF8") from exc
    _require(0 < len(raw) <= MAX_BODY_BYTES, "body byte length outside limit")
    _require(len(raw) + (not body.endswith("\n")) <= MAX_BODY_BYTES,
             "body plus structural final LF exceeds byte limit")
    _require("\r" not in body.replace("\r\n", ""), "body has unsupported lone CR")
    _require(len((body if body.endswith("\n") else body+"\n").split("\n"))-1 <= MAX_BODY_LINES,
             "body exceeds physical line limit")
    compiled = _HEADER + _indented(body)
    # Existing strict guard includes compile-only contextual syntax validation.
    try:
        list_policy.validate_helper(compiled)
    except ValueError as exc:
        # The pinned guard preserves the parser exception as its cause. Report
        # positions in the submitted body, not the compiler-owned signature and
        # aliases. Never dedent, repair or execute rejected text.
        cause = exc.__cause__
        if isinstance(cause, SyntaxError):
            line = (cause.lineno or 0) - _HEADER.count("\n")
            lines = body.split("\n")
            if 1 <= line <= len(lines):
                offset = cause.offset or 1
                # Contextual errors from the guard's compile(AST) have no
                # source text and use UTF8-byte offsets. Parser errors carry
                # text and already use character offsets. Neither executes code.
                if cause.text is None:
                    prefix = ("    " + lines[line - 1]).encode("utf-8")[:max(0, offset - 1)]
                    offset = len(prefix.decode("utf-8", errors="ignore")) + 1
                column = max(1, offset - 4)
                excerpt = repr(lines[line - 1][:160])
                leading = len(lines[line - 1]) - len(lines[line - 1].lstrip(" \t"))
                message = (f"invalid body syntax: {cause.msg[:160]} "
                           f"(body line {line}, column {column}); "
                           f"leading whitespace characters: {leading}; "
                           f"submitted line excerpt (untrusted; display whitespace may collapse): {excerpt}")
                raise ValueError(message) from exc
        detail = _surface_error_detail(body, exc)
        if detail is not None:
            raise ValueError(detail) from exc
        raise
    function = ast.parse(compiled).body[0]
    for statement in function.body[4:]:
        for node in ast.walk(statement):
            if isinstance(node, ast.Name):
                _require(node.id not in _OLD_NAMES,
                         "body may use only the typed array aliases, not old argument names")
    return body


def source_for(body=DEFAULT_BODY):
    """Build the common control or one arbitrary valid body; execute nothing."""
    validate_body(body)
    prefix, suffix = _pinned_scaffold()
    return prefix + _HEADER + _indented(body) + suffix


def body_of(source):
    """Recognize only this exact source family, returning canonical final-LF body.

    A raw body with no final LF and the same body with final LF assemble to the
    same source. This inverse returns the final-LF form without changing CRLF
    elsewhere. source_for(body_of(source)) must reproduce every source byte.
    """
    _require(type(source) is str, "parent source must be text")
    try:
        source.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("parent source must be strict UTF8") from exc
    prefix, suffix = _pinned_scaffold()
    beginning = prefix + _HEADER
    _require(source.startswith(beginning) and source.endswith(suffix)
             and len(source) >= len(beginning)+len(suffix), "source is outside the exact body family")
    middle = source[len(beginning):len(source)-len(suffix)]
    _require(middle.endswith("\n"), "compiled body lacks structural LF")
    lines = middle.split("\n")[:-1]
    _require(bool(lines) and all(line.startswith("    ") for line in lines),
             "compiled body indentation differs")
    body = "".join(line[4:]+"\n" for line in lines)
    validate_body(body)
    _require(source_for(body) == source, "source differs outside exact body assembly")
    return body


def apply(parent_source, target_body):
    """Replace the whole compiler-owned helper, retaining exact raw target body.

    before/after are actual compiled helper spans. parent_body is the canonical
    source inverse; target_body is the exact submitted string. Source equality,
    not raw-body equality, rejects final-LF aliases of an unchanged program.
    """
    parent_body = body_of(parent_source)
    child = source_for(target_body)
    _require(child != parent_source, "body edit leaves source unchanged")
    prefix, suffix = _pinned_scaffold()
    before, after = _HEADER+_indented(parent_body), _HEADER+_indented(target_body)
    p, s = prefix.encode("utf-8"), suffix.encode("utf-8")
    a, b = before.encode("utf-8"), after.encode("utf-8")
    parent_raw, child_raw = parent_source.encode("utf-8"), child.encode("utf-8")
    _require(parent_raw == p+a+s and child_raw == p+b+s, "body source assembly differs")
    proof = {"schema":"source-bound-edit/1", "contract":CONTRACT, "operation":OPERATION,
             "baseline_source_sha256":PINNED_BASELINE_SHA256,
             "common_scaffold_sha256":PINNED_SCAFFOLD_SHA256,
             "proposal_builder_sha256":PINNED_PROPOSALS_SHA256,
             "list_policy":list_policy_identity(),
             "parent_source_sha256":_sha(parent_raw), "child_source_sha256":_sha(child_raw),
             "parent_body":parent_body, "target_body":target_body,
             "parent_body_sha256":_sha(parent_body.encode("utf-8")),
             "target_body_sha256":_sha(target_body.encode("utf-8")),
             "assembly":{"indent":"    ", "append_final_lf":not target_body.endswith("\n"),
                         "line_endings":"LF/CRLF preserved; lone CR rejected", "parent_body_form":"canonical final-LF source inverse"},
             "before":{"text":before, "sha256":_sha(a), "byte_span":[len(p),len(p)+len(a)]},
             "after":{"text":after, "sha256":_sha(b), "byte_span":[len(p),len(p)+len(b)]},
             "unchanged_prefix":{"bytes":len(p), "sha256":_sha(p)},
             "unchanged_suffix":{"bytes":len(s), "sha256":_sha(s)},
             "unchanged_outside_helper":True,
             "scope":"Exact body assembly/source edit and interface validation only; no semantic, causal, safety, runtime or quality proof."}
    proof["plan_key"] = OPERATION+":"+_sha(parent_raw)+":"+_sha(child_raw)
    return {"code":child,"proof":proof}


def _unique(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError("nonfinite JSON constant")


def _envelope(raw):
    if type(raw) is not str:
        return None
    try:
        if len(raw.encode("utf-8")) > MAX_RESPONSE_BYTES:
            return None
        value = json.loads(raw, object_pairs_hook=_unique, parse_constant=_reject_constant)
    except (ValueError, TypeError, RecursionError, UnicodeEncodeError):
        return None
    if type(value) is not dict or set(value) != {"hypothesis", "body"}:
        return None
    return value


def parse_hypothesis_response(raw):
    """Independently retain a valid five-field hypothesis from a strict envelope."""
    value = _envelope(raw)
    return None if value is None else parse_hypothesis_value(value["hypothesis"])


# Readable alias for callers that adopted the initial API spelling.
parse_response_hypothesis = parse_hypothesis_response


def decode_response(raw):
    """Expose hypothesis/body failures independently, without repairing either."""
    value = _envelope(raw)
    if value is None:
        return {"proposal":None, "hypothesis":None, "error":"invalid_json_envelope", "body_error":None}
    hypothesis = parse_hypothesis_value(value["hypothesis"])
    body_error = None
    try:
        validate_body(value["body"])
    except ValueError as exc:
        body_error = str(exc)
    error = "invalid_hypothesis" if hypothesis is None else "invalid_body" if body_error else None
    proposal = BodyProposal(hypothesis, value["body"]) if error is None else None
    return {"proposal":proposal, "hypothesis":hypothesis, "error":error, "body_error":body_error}


def parse_response(raw):
    return decode_response(raw)["proposal"]


def response_format():
    """Provider JSON Schema; cross-field/range/UTF8/AST checks also run locally."""
    hypothesis = {"type":"object", "additionalProperties":False,
        "required":["hypothesis","predicted_low","predicted_high","reasoning","confidence"],
        "properties":{
            "hypothesis":{"type":"string","minLength":1,"maxLength":MAX_TEXT_CHARS},
            "predicted_low":{"type":"number","description":"Lower candidate-minus-measured-parent normalized-score DELTA, not absolute score."},
            "predicted_high":{"type":"number","description":"Upper normalized-score DELTA; must be >= predicted_low."},
            "reasoning":{"type":"string","minLength":1,"maxLength":MAX_REASONING_CHARS},
            "confidence":{"type":"number","minimum":0,"maximum":1}}}
    return {"type":"json_schema", "json_schema":{"name":"circle_restart_body_v1", "strict":True,
        "schema":{"type":"object", "additionalProperties":False, "required":["hypothesis","body"],
                  "properties":{"hypothesis":hypothesis,
                      "body":{"type":"string","minLength":1,"maxLength":MAX_BODY_BYTES,
                              "description":"Only Python body statements using the typed arrays. Top-level statements start in column 1 with no leading indent; indent only nested suites. The compiler adds the function indent. No def/import/fence. UTF8 cap8192 including an appended structural LF if needed, and physical line cap512, are validated locally. Final newline optional."}}}}}


def interface_description():
    old = proposals.interface_description()
    code_contract = {key:old[key] for key in ("builtins","array_methods","array_values","forbidden")}
    code_contract.update(module_bindings={"np":"numpy", "math":"math"},
        unbound_module_names=["numpy"],
        builtin_call_style="Call listed builtins directly, e.g. abs(x), range(n), sum(values).",
        numpy_calls=["np."+name for name in old["numpy_calls"]],
        numpy_values=["np."+name for name in old["numpy_values"]],
        math_calls=["math."+name for name in old["math_calls"]],
        math_values=["math."+name for name in old["math_values"]])
    code_contract["construction"] = {
        "method_scope": "Ordinary receivers use the non-module attribute/method whitelist. "
            "A proven local list additionally permits standalone name.append(value) under local_list_policy. "
            "extend and insert remain unavailable; append cannot be called through an alias or parameter.",
        "local_list_policy": {**dict(list_policy.POLICY), "identity": list_policy_identity()},
        "local_list_example": "indices = []\nfor i in range(4):\n    indices.append(i)\n",
        "indexed_array_example": "values = np.zeros(4)\nfor i in range(4):\n    values[i] = float(i)\n",
        "comprehension_example": "values = np.array([float(i) for i in range(4)], dtype=np.float64)\n",
        "scope": "These are syntax examples on a four-element local array, not complete restart bodies, "
            "a packing recipe, or evidence of numerical benefit. Return None or the required finite 78-vector."
    }
    return {"contract":CONTRACT, "editable":"Python body only; the compiler owns signature and array handoff.",
        "arrays":{name:{"type":"numpy.ndarray", "dtype":"float64", "shape":[78], "nullable":False, "ownership":"detached mutable copy"}
                  for name in _ALIASES},
        "meanings":{"best_params":"Always an ndarray, never None: retained tolerance-feasible incumbent vector",
            "initial_params":"fixed initial grid vector",
            "uniform_noise":"the existing full78 uniform[-0.06,0.06] draw; no extra RNG",
            "ordinary_params":"ordinary incumbent+noise guess after its original radius/center clamps"},
        "default_transformation":{
            "control_source_sha256":_sha(source_for().encode("utf-8")),
            "common_scaffold_sha256":PINNED_SCAFFOLD_SHA256,
            "existing_noise":{"distribution":"uniform", "low":-0.06, "high":0.06,
                "shape":[78], "draws_per_restart":1, "drawn_before_body":True},
            "readonly_array_equivalent":(
                "ordinary_params = best_params + uniform_noise\n"
                "ordinary_params[52:] = np.maximum(ordinary_params[52:], 0.001)\n"
                "ordinary_params[:52] = np.clip(ordinary_params[:52], 0.001, 0.999)\n"),
            "index_zero":"If the body is invoked at restart_index=0, best_params and initial_params "
                "have equal values before any SLSQP solve. The existing noise and both clamps "
                "have already been applied to ordinary_params; it is not the initial grid itself.",
            "none_fallback":"Returning None selects a copy of this original ordinary_params, "
                "even if the body mutates its detached array arguments. A returned vector is "
                "used without these automatic clamps.",
            "scope":"Read-only description of the source-pinned kernel, not a suggested new body. "
                "Incumbent plus uniform noise is already a baseline component. Compare an "
                "intended change against this formula and the current body, not against an "
                "unperturbed incumbent. These facts do not establish novelty, feasibility, "
                "numerical benefit or causality."},
        "layout":{"centers":"params[:52].reshape(26,2)", "radii":"params[52:]"},
        "restart_index":{"type":"int","range":"nonnegative multiples of4",
            "invoked_only_when":"restart_index % 4 == 0",
            "call_schedule":"Only called when restart_index % 4 == 0, including0. That condition is always true inside this body. The other three restarts skip the helper entirely."},
        "return":"None selects the original ordinary_params value: incumbent plus the existing noise, already radius/center-clamped, not the unperturbed incumbent. Mutations inside the body cannot change this saved fallback. Otherwise return a copied real finite numeric78-vector. No automatic clipping or feasibility repair; initializer overlaps/out-of-bounds are allowed.",
        "solver":"Unchanged joint SLSQP over26(x,y,r), maximize sum(r); analytical objective/constraint Jacobians, ftol1e-7, maxiter15000. All work shares27s cooperative deadline; external watchdog is authoritative.",
        "metric":"Controller score is tolerance-valid raw_sum/2.635, not capped at1. Forecast DELTA uses the separately supplied measured parent score. Radius-decreased exact certification is a separate metric/object, never the forecast score.",
        "source_isolation":"No solver objects, methods or historical implementation globals are part of this interface. Full kernel source is an audit artifact, not writer context.",
        "code_contract":code_contract,
        "syntax":"Assignments, if/for, return/break/continue/pass, synchronous list comprehensions/generator expressions and numeric expressions. No def/import/global/nonlocal/try/raise/while/dunder/dynamic call; no old argument names or external helpers.",
        "body_indentation":"Top-level statements start in column 1 (zero leading spaces/tabs). Only nested if/for suites are indented relative to that level. Do not include a function-level indent; the compiler adds it. Indented input is rejected, never dedented.",
        "assembly":"Preserve exact body; prefix each LF-delimited line with4spaces and append one structural LF only if absent. The8192-byte cap includes that LF. CRLF preserved; lone CR unsupported.",
        "limits":{"body_utf8_bytes":MAX_BODY_BYTES,"body_physical_lines":MAX_BODY_LINES,
                  "compiled_helper_utf8_bytes":proposals.MAX_HELPER_BYTES,"ast_nodes":proposals.MAX_AST_NODES},
        "scope":"The AST interface guard does not prove safety, bounded allocation/termination, algorithmic intent or quality. Candidate errors remain failures; no automatic repair or promotion."}


def response_instruction():
    return ("Return one JSON object matching the supplied response schema: hypothesis and body, no Markdown or other fields. "
            "Generate a new numerical restart transformation as Python body statements, not a function definition, full program or policy label. "
            "Start top-level statements in column 1 with no leading indent; indent only nested suites. The compiler adds the function-level indent. "
            "Use only the following typed array interface. All four named values are arrays, never solver objects. "
            "NumPy is bound as np (for example np.maximum), not numpy; math is bound as math. "
            "Forecast a normalized score DELTA relative to the separately declared measured parent; numerical effect is uncertain.\n"
            + json.dumps(interface_description(), sort_keys=True, separators=(",", ":"), allow_nan=False))


def identity():
    control = source_for()
    return {"contract":CONTRACT,"operation":OPERATION,"source_sha256":_sha(Path(__file__).read_bytes()),
        "proposal_builder_sha256":PINNED_PROPOSALS_SHA256,"common_scaffold_sha256":PINNED_SCAFFOLD_SHA256,
        "list_policy":list_policy_identity(),
        "baseline_source_sha256":PINNED_BASELINE_SHA256,"control_source_sha256":_sha(control.encode()),
        "default_body_sha256":_sha(DEFAULT_BODY.encode()),"interface_sha256":_sha(_encode(interface_description())),
        "response_format_sha256":_sha(_encode(response_format())),"stats_marker":STATS_MARKER,
        "scope":"New typed-body control must be measured; no unchanged timing claim relative to an older scaffold."}
