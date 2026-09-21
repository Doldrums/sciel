"""Transactional source composition. Submitted Python is parsed, never run here.

Generated code is an ordinary candidate and must execute only in the external
executor. A composition receipt proves inclusion, not invocation or usefulness.
The embedded loader has the same dependency namespaces as the Docker executor.
"""
from __future__ import annotations

import ast
from copy import deepcopy
import hashlib
import inspect
import io
import tokenize

from .models import canonical_bytes, content_hash, require_hash, validate_manifest

_MAX_BYTES = 8 * 1024 * 1024
_MARKER = "# SCIEL_METHODS"


class _Refusal(ValueError):
    def __init__(self, reason, detail):
        super().__init__(detail)
        self.reason = reason


def _sha(source):
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _parse(source):
    if type(source) is not str or len(source.encode("utf-8")) > _MAX_BYTES:
        raise _Refusal("invalid_source", "source must be bounded UTF-8 text")
    try:
        tree = ast.parse(source)
        # AST-only parsing does not reject all misplaced future imports. Compile
        # checks those too, without evaluating imports, annotations or statements.
        compile(source, "<sciel-composition>", "exec", dont_inherit=True)
    except (SyntaxError, ValueError) as exc:
        raise _Refusal("invalid_source", "source is not valid standalone Python") from exc
    return tree


def _bundles(methods):
    if type(methods) is not list or len(methods) > 64:
        raise _Refusal("invalid_bundle", "methods must be a list of at most 64 immutable bundles")
    by_id, total = {}, 0
    for value in methods:
        if type(value) is not dict or set(value) != {"id", "manifest", "files"}:
            raise _Refusal("invalid_bundle", "bundle requires exactly id, manifest and files")
        ident = require_hash(value["id"], "method ID")
        manifest = validate_manifest(value["manifest"], value["files"])
        if content_hash(dict(schema="sciel-method/1", manifest=manifest, files=value["files"])) != ident:
            raise _Refusal("bundle_hash_mismatch", "bundle bytes do not match the registered version")
        if ident in by_id:
            raise _Refusal("duplicate_bundle", "each registered bundle must be supplied once")
        for path, source in value["files"].items():
            total += len(source.encode("utf-8"))
            if total > _MAX_BYTES:
                raise _Refusal("bundle_size_limit", "combined registered files exceed the composition cap")
            if path.endswith(".py"):
                _parse(source)
        by_id[ident] = deepcopy(value)
    images = {value["manifest"]["runtime_lock"]["image_id"] for value in by_id.values()}
    if len(images) > 1:
        raise _Refusal("runtime_mismatch", "composition cannot silently combine different runtime images")
    visiting, visited = set(), set()

    def visit(ident):
        if ident not in by_id:
            raise _Refusal("missing_dependency", "all pinned dependency bundles must be supplied")
        if ident in visiting:
            raise _Refusal("dependency_cycle", "dependency graph must be acyclic")
        if ident in visited:
            return
        visiting.add(ident)
        for dependency in by_id[ident]["manifest"]["dependencies"]:
            visit(dependency)
        visiting.remove(ident)
        visited.add(ident)

    for ident in by_id:
        visit(ident)
    return by_id


def _patch(parent, adapters, by_id):
    if type(adapters) not in (list, tuple) or len(adapters) > 64:
        raise _Refusal("invalid_adapter", "adapters must be a bounded list")
    source, proofs, seen = parent, [], set()
    for adapter in adapters:
        if type(adapter) is not dict or set(adapter) != {"version_id", "expected_parent_sha256", "edits"}:
            raise _Refusal("invalid_adapter", "adapter requires version_id, expected_parent_sha256 and edits")
        ident = require_hash(adapter["version_id"], "adapter version")
        if ident not in by_id or by_id[ident]["manifest"]["kind"] != "adapter":
            raise _Refusal("unregistered_adapter", "adapter version must be a supplied registered adapter bundle")
        if ident in seen:
            raise _Refusal("duplicate_adapter", "an adapter version can be applied only once")
        seen.add(ident)
        protocol = by_id[ident]["manifest"]["protocol"]
        patch = {key: adapter[key] for key in ("expected_parent_sha256", "edits")}
        if type(protocol) is not dict or protocol.get("patch") != patch:
            raise _Refusal("adapter_protocol_mismatch", "patch must equal the registered adapter protocol.patch")
        before = _sha(source)
        if require_hash(adapter["expected_parent_sha256"], "adapter parent") != before:
            raise _Refusal("adapter_parent_mismatch", "adapter expected_parent_sha256 must bind the current pre-adapter source")
        if type(adapter["edits"]) is not list or not 1 <= len(adapter["edits"]) <= 256:
            raise _Refusal("invalid_adapter", "adapter edits must be a nonempty bounded list")
        changes = []
        for edit in adapter["edits"]:
            if (type(edit) is not dict or set(edit) != {"old", "new", "expected_count"}
                    or type(edit["old"]) is not str or not edit["old"] or type(edit["new"]) is not str
                    or type(edit["expected_count"]) is not int or not 1 <= edit["expected_count"] <= 100000):
                raise _Refusal("invalid_adapter", "each edit needs nonempty old, text new and positive integral expected_count")
            count = source.count(edit["old"])
            if count != edit["expected_count"]:
                raise _Refusal("adapter_match_mismatch", "exact sequential edit match count differed from its contract")
            source = source.replace(edit["old"], edit["new"])
            if len(source.encode("utf-8")) > _MAX_BYTES:
                raise _Refusal("source_size_limit", "adapter output exceeds the composition cap")
            changes.append(dict(old_sha256=_sha(edit["old"]), new_sha256=_sha(edit["new"]),
                                expected_count=count, applied_count=count))
        _parse(source)
        proofs.append(dict(version_id=ident, input_source_sha256=before,
                           output_source_sha256=_sha(source), edits=changes))
    return source, proofs


def _insertion(source, tree):
    prefix = []
    body = list(tree.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        prefix.append(body.pop(0))
    while body and isinstance(body[0], ast.ImportFrom) and body[0].module == "__future__":
        prefix.append(body.pop(0))
    prefix_line = prefix[-1].end_lineno if prefix else 0
    if body and body[0].lineno <= prefix_line:
        raise _Refusal("ambiguous_insertion", "docstring/future prefix must end on its own source line")
    markers = [token for token in tokenize.generate_tokens(io.StringIO(source).readline)
               if token.type == tokenize.COMMENT and token.string.strip() == _MARKER]
    if len(markers) > 1:
        raise _Refusal("ambiguous_marker", "at most one SCIEL_METHODS marker is allowed")
    lines = source.splitlines(keepends=True)
    if markers:
        token = markers[0]
        line = token.start[0]
        if (token.start[1] != 0 or line <= prefix_line
                or any(node.lineno <= line <= node.end_lineno for node in tree.body)):
            raise _Refusal("invalid_marker", "SCIEL_METHODS must occupy a standalone top-level line after future imports")
        return sum(map(len, lines[:line - 1])), len(lines[line - 1]), line
    line = prefix_line + 1 if prefix else (tree.body[0].lineno if tree.body else len(lines) + 1)
    return sum(map(len, lines[:line - 1])), 0, line


def _loader_source(by_id):
    # This imports trusted repository code only. No registered source is imported
    # or executed during assembly. Its exact text becomes part of the candidate.
    from ._loader import make_method_loader
    loader = inspect.getsource(make_method_loader)
    payload = {ident: {"files": value["files"], "entrypoint": value["manifest"]["entrypoint"]}
               for ident, value in sorted(by_id.items())}
    # Lazy temp files live in the external candidate process. Keeping the owner in
    # the closure ensures imports/resource reads remain valid throughout its run.
    bootstrap = '''
def _sciel_bootstrap():
    import json
    import tempfile
    from pathlib import Path
    bundles = PAYLOAD
    temporary = None
    invoke = None
    def call(method_id, value):
        nonlocal temporary, invoke
        if method_id not in bundles:
            raise ValueError("method is outside the composed bundle set")
        json.dumps(value, allow_nan=False)
        if invoke is None:
            temporary = tempfile.TemporaryDirectory(prefix="sciel-composed-")
            methods = {}
            for ident, bundle in bundles.items():
                root = Path(temporary.name) / ("m_" + ident)
                for name, code in bundle["files"].items():
                    path = root / name
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(code.encode("utf-8"))
                methods[ident] = {"root": str(root), "entrypoint": bundle["entrypoint"]}
            invoke = _sciel_make_method_loader(methods)
        result = invoke(method_id, value)
        json.dumps(result, allow_nan=False)
        return result
    return call
sciel_method = _sciel_bootstrap()
del _sciel_bootstrap
'''.replace("PAYLOAD", repr(payload))
    # Rename only the trusted factory definition, keeping exact original bundled
    # method source strings entirely untouched.
    loader = loader.replace("def make_method_loader(", "def _sciel_make_method_loader(", 1)
    return "\n# BEGIN SCIEL REGISTERED METHODS\n" + loader + bootstrap + "# END SCIEL REGISTERED METHODS\n", _sha(loader)


def materialize(parent: str, methods: list[dict], adapters=()):
    """Assemble all edits or return a typed refusal without any partial code.

    Each adapter binds the input hash *at its position in the adapter sequence*;
    its exact patch also lives in that adapter's registered manifest.protocol.
    All dependencies must be included in methods; provenance-only parents need
    not be executable dependencies. The returned runtime image must match the
    external executor. Inner calls do not receive independent resource limits.
    """
    try:
        _parse(parent)
        by_id = _bundles(methods)
        source, patches = _patch(parent, adapters, by_id)
        tree = _parse(source)
        for node in ast.walk(tree) if by_id else ():
            name = (node.id if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)) else
                    node.name if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) else
                    (node.asname or node.name.split(".")[0]) if isinstance(node, ast.alias) else None)
            if name and (name == "sciel_method" or name.startswith("_sciel_")):
                raise _Refusal("reserved_name_collision", "parent binds a reserved composition loader name")
        insertions, loader_sha = [], None
        if by_id:
            offset, replace_length, line = _insertion(source, tree)
            generated, loader_sha = _loader_source(by_id)
            if offset and source[offset - 1] != "\n":
                generated = "\n" + generated
            combined = source[:offset] + generated + source[offset + replace_length:]
            insertions.append(dict(at_line=line, replaced_marker=bool(replace_length),
                                   source_sha256=_sha(generated), method_ids=sorted(by_id)))
        else:
            combined = source
        _parse(combined)
        components = [dict(id=ident, kind=value["manifest"]["kind"],
                           manifest_sha256=content_hash(value["manifest"]),
                           files=[dict(path=path, sha256=_sha(code)) for path, code in sorted(value["files"].items())])
                      for ident, value in sorted(by_id.items())]
        receipt = dict(schema="sciel-composition/1", parent_source_sha256=_sha(parent),
                       adapted_parent_source_sha256=_sha(source), source_sha256=_sha(combined),
                       components=components, adapters=patches, insertions=insertions,
                       loader_sha256=loader_sha, runtime_image_id=next(iter(by_id.values()))["manifest"]["runtime_lock"]["image_id"] if by_id else None,
                       invocations_attested=False, resources_enforced="outer_candidate_only")
        receipt["id"] = content_hash(receipt)
        return dict(status="ok", code=combined, receipt=receipt)
    except _Refusal as exc:
        return dict(status="refused", reason=exc.reason, detail=str(exc))
    except (ValueError, TypeError, KeyError, UnicodeError, tokenize.TokenError, RecursionError):
        return dict(status="refused", reason="invalid_contract", detail="source or immutable bundle contract could not be validated")
