"""Pure, deliberately small contracts for immutable method packages.

This module validates data, never imports or executes a submitted source file.
Unsupported JSON Schema keywords are errors rather than ignored constraints.
"""
from __future__ import annotations

import ast
import hashlib
import json
import keyword
import math
import re
from typing import Any

HASH_RE = re.compile(r"[0-9a-f]{64}\Z")
KINDS = frozenset({"tool", "experiment", "search", "adapter"})
RESOURCE_KEYS = frozenset({"wall_seconds", "cpu_seconds", "memory_bytes", "pids",
                           "max_output_bytes", "max_stderr_bytes"})


def _json(value: Any, path: str = "$") -> None:
    if value is None or type(value) in (bool, str, int):
        return
    if type(value) is float and math.isfinite(value):
        return
    if type(value) is list:
        for i, item in enumerate(value):
            _json(item, f"{path}[{i}]")
        return
    if type(value) is dict and all(type(k) is str for k in value):
        for key, item in value.items():
            _json(item, f"{path}.{key}")
        return
    raise ValueError(f"{path}: expected finite JSON value")


def canonical_bytes(value: Any) -> bytes:
    _json(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode("utf-8")


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def detached(value: Any) -> Any:
    return json.loads(canonical_bytes(value))


def require_hash(value: Any, label: str = "hash") -> str:
    if not isinstance(value, str) or not HASH_RE.fullmatch(value):
        raise ValueError(f"{label}: expected lowercase SHA256")
    return value


def hash_list(value: Any, label: str) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{label}: expected list")
    result = [require_hash(item, label) for item in value]
    if len(set(result)) != len(result):
        raise ValueError(f"{label}: duplicate reference")
    return result


def safe_path(value: Any) -> str:
    if (not isinstance(value, str) or not value or len(value) > 512
            or "\\" in value or ":" in value or "\x00" in value
            or any(part in ("", ".", "..") for part in value.split("/"))
            or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", part) for part in value.split("/"))):
        raise ValueError("bundle path must be a safe relative POSIX path")
    return value


def validate_schema(schema: Any) -> None:
    _json(schema)
    if not isinstance(schema, dict):
        raise ValueError("schema must be an object")
    kind = schema.get("type")
    permitted = {
        "object": {"properties", "required", "additionalProperties", "minProperties", "maxProperties"},
        "array": {"items", "minItems", "maxItems"},
        "string": {"minLength", "maxLength"},
        "number": {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"},
        "integer": {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"},
        "boolean": set(), "null": set(),
    }
    if not isinstance(kind, str) or kind not in permitted:
        raise ValueError("schema requires one supported type")
    if set(schema) - permitted[kind] - {"type", "enum", "description", "title"}:
        raise ValueError("unsupported JSON Schema keyword")
    for key in ("description", "title"):
        if key in schema and not isinstance(schema[key], str):
            raise ValueError(f"schema {key} must be text")
    for key in ("minItems", "maxItems", "minLength", "maxLength", "minProperties", "maxProperties"):
        if key in schema and (type(schema[key]) is not int or schema[key] < 0):
            raise ValueError(f"schema {key} must be a nonnegative integer")
    for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum"):
        if key in schema and type(schema[key]) not in (float, int):
            raise ValueError(f"schema {key} must be a number")
    for low, high in (("minItems", "maxItems"), ("minLength", "maxLength"),
                      ("minProperties", "maxProperties"), ("minimum", "maximum")):
        if low in schema and high in schema and schema[low] > schema[high]:
            raise ValueError("schema lower bound exceeds upper bound")
    if kind == "object":
        props = schema.get("properties", {})
        required = schema.get("required", [])
        if not isinstance(props, dict):
            raise ValueError("schema properties must be an object")
        for child in props.values():
            validate_schema(child)
        if (not isinstance(required, list) or not all(isinstance(x, str) for x in required)
                or len(set(required)) != len(required) or set(required) - set(props)):
            raise ValueError("schema required must name unique declared properties")
        additional = schema.get("additionalProperties", True)
        if type(additional) is not bool:
            validate_schema(additional)
    if kind == "array":
        if "items" not in schema:
            raise ValueError("array schema requires items")
        validate_schema(schema["items"])
    if "enum" in schema:
        choices = schema["enum"]
        if not isinstance(choices, list) or not choices:
            raise ValueError("schema enum must be nonempty")
        if len({canonical_bytes(x) for x in choices}) != len(choices):
            raise ValueError("schema enum contains duplicates")
        base = {key: value for key, value in schema.items() if key != "enum"}
        for choice in choices:
            _validate_value(choice, base, "enum")


def _validate_value(value: Any, schema: dict, path: str) -> None:
    kind = schema["type"]
    valid = {"object": type(value) is dict, "array": type(value) is list,
             "string": type(value) is str, "number": type(value) in (int, float),
             "integer": type(value) is int, "boolean": type(value) is bool,
             "null": value is None}[kind]
    if not valid:
        raise ValueError(f"{path}: expected {kind}")
    if "enum" in schema and canonical_bytes(value) not in {canonical_bytes(x) for x in schema["enum"]}:
        raise ValueError(f"{path}: value outside enum")
    if kind == "object":
        if set(schema.get("required", [])) - set(value):
            raise ValueError(f"{path}: missing required property")
        props = schema.get("properties", {})
        for key, item in value.items():
            child = props.get(key, schema.get("additionalProperties", True))
            if child is False:
                raise ValueError(f"{path}: additional property {key}")
            if isinstance(child, dict):
                _validate_value(item, child, f"{path}.{key}")
    if kind == "array":
        for i, item in enumerate(value):
            _validate_value(item, schema["items"], f"{path}[{i}]")
    for low, high, applicable in (("minItems", "maxItems", "array"),
                                  ("minLength", "maxLength", "string"),
                                  ("minProperties", "maxProperties", "object")):
        if kind == applicable and (len(value) < schema.get(low, 0)
                                  or len(value) > schema.get(high, math.inf)):
            raise ValueError(f"{path}: size outside schema bounds")
    if kind in ("number", "integer"):
        for key, passed in (("minimum", lambda x: value >= x), ("maximum", lambda x: value <= x),
                            ("exclusiveMinimum", lambda x: value > x),
                            ("exclusiveMaximum", lambda x: value < x)):
            if key in schema and not passed(schema[key]):
                raise ValueError(f"{path}: number outside {key}")


def validate_value(value: Any, schema: dict) -> None:
    validate_schema(schema)
    _json(value)
    _validate_value(value, schema, "$")


def validate_runtime_lock(value: Any) -> None:
    _json(value)
    if not isinstance(value, dict) or not isinstance(value.get("image_id"), str):
        raise ValueError("runtime_lock requires immutable image_id")
    image = value["image_id"]
    if not image.startswith("sha256:"):
        raise ValueError("runtime image must use exact sha256 ID")
    require_hash(image[7:], "runtime image")


def validate_manifest(manifest: Any, bundle: Any) -> dict:
    _json(manifest)
    required = {"name", "kind", "author_lab", "entrypoint", "input_schema", "output_schema",
                "protocol", "runtime_lock", "dependencies", "parents", "resources", "scope", "examples"}
    if not isinstance(manifest, dict) or not required <= set(manifest) or set(manifest) - required - {"preconditions"}:
        raise ValueError("method manifest fields missing or unsupported")
    for key in ("name", "author_lab"):
        if not isinstance(manifest[key], str) or not manifest[key].strip():
            raise ValueError(f"manifest {key} must be nonempty text")
    if not isinstance(manifest["kind"], str) or manifest["kind"] not in KINDS:
        raise ValueError("unsupported method kind")
    if not isinstance(bundle, dict) or not bundle:
        raise ValueError("bundle must contain source files")
    for path, code in bundle.items():
        safe_path(path)
        if not isinstance(code, str):
            raise ValueError("bundle files must contain text")
        try:
            code.encode("utf-8")
        except UnicodeError as exc:
            raise ValueError("bundle files must contain valid UTF-8 text") from exc
        if any("/".join(path.split("/")[:i]) in bundle for i in range(1, len(path.split("/")))):
            raise ValueError("bundle file conflicts with package directory")
    entrypoint = manifest["entrypoint"]
    if not isinstance(entrypoint, str) or entrypoint.count(":") != 1:
        raise ValueError("entrypoint must be file.py:function")
    path, function = entrypoint.split(":")
    safe_path(path)
    if path not in bundle or not path.endswith(".py") or not function.isidentifier() or keyword.iskeyword(function):
        raise ValueError("entrypoint must name a bundled Python function")
    # Syntax checking is not execution or a claim about runtime correctness.
    try:
        tree = ast.parse(bundle[path])
    except (SyntaxError, ValueError, UnicodeError) as exc:
        raise ValueError("entrypoint source is not valid Python") from exc
    if not any(isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function for node in tree.body):
        raise ValueError("entrypoint function is not defined at module scope")
    for key in ("input_schema", "output_schema"):
        validate_schema(manifest[key])
    for key in ("dependencies", "parents"):
        hash_list(manifest[key], key)
    validate_runtime_lock(manifest["runtime_lock"])
    resources = manifest["resources"]
    if not isinstance(resources, dict) or set(resources) != RESOURCE_KEYS:
        raise ValueError("resources must specify all supported execution limits")
    for key, limit in resources.items():
        if type(limit) not in (int, float) or limit <= 0:
            raise ValueError("resource limits must be finite and positive")
        if key not in ("wall_seconds", "cpu_seconds") and type(limit) is not int:
            raise ValueError(f"resource {key} must be an integer")
    for key in ("protocol", "scope"):
        if not isinstance(manifest[key], (str, dict, list)) or not manifest[key]:
            raise ValueError(f"manifest {key} must be nonempty text or JSON structure")
    if not isinstance(manifest["examples"], list):
        raise ValueError("manifest examples must be a list")
    return detached(manifest)
